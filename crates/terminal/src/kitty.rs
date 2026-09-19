use base64::Engine;
use std::io::Write;
use std::time::Duration;
// -- Kitty graphics protocol ---------------------------------------------

/// Kitty graphics protocol payload chunk size, in base64-encoded bytes.
const KITTY_CHUNK_LIMIT: usize = 4096;

/// Builds one no-payload Kitty APC command, e.g. the delete-all command.
pub(crate) fn kitty_escape(control: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(control.len() + 5);
    out.extend_from_slice(b"\x1b_G");
    out.extend_from_slice(control.as_bytes());
    out.extend_from_slice(b"\x1b\\");
    out
}

/// Builds a Kitty APC command transmitting `payload`, base64-encoding it
/// and chunking at `KITTY_CHUNK_LIMIT` base64 bytes per the protocol's
/// `m=1` (more chunks follow) / `m=0` (final chunk) convention. Payloads at
/// or under the limit are sent as one unchunked command with no `m` key at
/// all.
pub(crate) fn kitty_transmit(control: &str, payload: &[u8]) -> Vec<u8> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(payload);
    let encoded = encoded.as_bytes();
    if encoded.len() <= KITTY_CHUNK_LIMIT {
        let mut out = Vec::with_capacity(encoded.len() + control.len() + 6);
        out.extend_from_slice(b"\x1b_G");
        out.extend_from_slice(control.as_bytes());
        out.push(b';');
        out.extend_from_slice(encoded);
        out.extend_from_slice(b"\x1b\\");
        return out;
    }
    let chunks: Vec<&[u8]> = encoded.chunks(KITTY_CHUNK_LIMIT).collect();
    let last_index = chunks.len() - 1;
    let mut out = Vec::new();
    for (index, chunk) in chunks.into_iter().enumerate() {
        out.extend_from_slice(b"\x1b_G");
        if index == 0 {
            out.extend_from_slice(control.as_bytes());
            out.extend_from_slice(b",m=1");
        } else if index == last_index {
            out.extend_from_slice(b"m=0");
        } else {
            out.extend_from_slice(b"m=1");
        }
        out.push(b';');
        out.extend_from_slice(chunk);
        out.extend_from_slice(b"\x1b\\");
    }
    out
}

/// Maps `--loops` onto the Kitty animation control protocol's `v=` key, per
/// https://sw.kovidgoyal.net/kitty/graphics-protocol/#animation: `v=1`
/// loops infinitely and any other positive number loops `number - 1`
/// times. `None` or `Some(0)` both mean "loop forever", matching normal
/// GIF playback (a GIF's Netscape loop extension also uses 0 to mean
/// infinite).
fn kitty_loop_key(loops: Option<u32>) -> u32 {
    match loops {
        None | Some(0) => 1,
        Some(n) => n.saturating_add(1),
    }
}

fn should_stop_looping(loops: Option<u32>, played: u32) -> bool {
    match loops {
        None | Some(0) => false,
        Some(limit) => played >= limit,
    }
}

/// Terminal-driven Kitty animation (WezTerm, plain Kitty): transmit the
/// root frame plus every additional frame with its gap, then hand playback
/// off to the terminal itself with one `a=a` control command. The process
/// can exit immediately afterward.
pub(crate) fn show_kitty_native_animation(
    sink: &mut dyn Write,
    frames: &[(Vec<u8>, u32)],
    loops: Option<u32>,
) -> std::io::Result<()> {
    let (first_png, first_delay) = &frames[0];
    sink.write_all(&kitty_transmit(
        &format!("a=T,f=100,i=1,q=2,z={first_delay}"),
        first_png,
    ))?;
    for (png, delay) in &frames[1..] {
        sink.write_all(&kitty_transmit(&format!("a=f,i=1,q=2,z={delay}"), png))?;
    }
    let loop_value = kitty_loop_key(loops);
    sink.write_all(&kitty_escape(&format!("a=a,i=1,q=2,s=3,v={loop_value}")))?;
    sink.flush()
}

/// Simulated animation for Kitty-protocol terminals without the animation
/// extension (Ghostty, cmux): loop transmitting each frame as a plain
/// static image reusing image id 1, sleeping for its delay in between,
/// until `loops` is exhausted or the process is killed.
///
/// `a=T` (transmit and display) creates a new on-screen placement and, by
/// default, advances the cursor past it -- so naively repeating it every
/// frame stacks a new placement below the last one each time, producing a
/// cascade of images scrolling down the screen instead of one frame
/// updating in place. `C=1` tells the terminal not to move the cursor
/// after displaying, and deleting image id 1's placement before each
/// redraw (`a=d,d=i,i=1`) removes the previous frame first, so every frame
/// lands at the same fixed position.
pub(crate) fn show_kitty_simulated_animation(
    sink: &mut dyn Write,
    frames: &[(Vec<u8>, u32)],
    loops: Option<u32>,
) -> std::io::Result<()> {
    let mut played = 0u32;
    let mut first_frame = true;
    loop {
        for (png, delay) in frames {
            if !first_frame {
                sink.write_all(&kitty_escape("a=d,d=i,i=1"))?;
            }
            first_frame = false;
            sink.write_all(&kitty_transmit("a=T,f=100,i=1,q=2,C=1", png))?;
            sink.flush()?;
            std::thread::sleep(Duration::from_millis(u64::from(*delay)));
        }
        played += 1;
        if should_stop_looping(loops, played) {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    use crate::kitty::kitty_escape;
    use crate::kitty::kitty_loop_key;
    use crate::kitty::kitty_transmit;
    use crate::kitty::should_stop_looping;

    use base64::Engine;

    #[test]
    fn kitty_transmit_does_not_chunk_small_payloads() {
        let payload = vec![9u8; 16];
        let escape = kitty_transmit("a=T,f=100,i=1,q=2", &payload);
        let text = String::from_utf8(escape).unwrap();
        assert_eq!(text.matches("\x1b_G").count(), 1);
        assert!(!text.contains("m=1"));
        assert!(!text.contains("m=0"));
        assert!(text.starts_with("\x1b_Ga=T,f=100,i=1,q=2;"));
        assert!(text.ends_with("\x1b\\"));
    }

    #[test]
    fn kitty_transmit_chunks_large_payloads_with_correct_boundary_flags() {
        let payload = vec![7u8; 6000];
        let escape = kitty_transmit("a=T,f=100,i=1,q=2", &payload);
        let text = String::from_utf8(escape).unwrap();
        let commands: Vec<&str> = text
            .split("\x1b_G")
            .filter(|segment| !segment.is_empty())
            .collect();
        assert!(commands.len() > 1, "expected the payload to be chunked");

        let mut recovered_b64 = String::new();
        for (index, command) in commands.iter().enumerate() {
            let body = command.strip_suffix("\x1b\\").unwrap();
            let (keys, chunk) = body.split_once(';').unwrap();
            if index == 0 {
                assert!(keys.contains("a=T,f=100,i=1,q=2"));
                assert!(keys.ends_with("m=1"));
            } else if index == commands.len() - 1 {
                assert_eq!(keys, "m=0");
            } else {
                assert_eq!(keys, "m=1");
            }
            recovered_b64.push_str(chunk);
        }

        let recovered = base64::engine::general_purpose::STANDARD
            .decode(recovered_b64)
            .unwrap();
        assert_eq!(recovered, payload);
    }

    #[test]
    fn kitty_transmit_boundary_exactly_at_chunk_limit_is_not_split() {
        // 3072 raw bytes base64-encode to exactly 4096 characters, with no
        // padding.
        let payload = vec![1u8; 3072];
        let escape = kitty_transmit("a=T,f=100,i=1,q=2", &payload);
        let text = String::from_utf8(escape).unwrap();
        assert_eq!(text.matches("\x1b_G").count(), 1);
    }

    #[test]
    fn kitty_transmit_boundary_one_byte_over_the_chunk_limit_splits_into_two() {
        // 3075 raw bytes base64-encode to 4100 characters: one full chunk
        // plus four more.
        let payload = vec![2u8; 3075];
        let escape = kitty_transmit("a=T,f=100,i=1,q=2", &payload);
        let text = String::from_utf8(escape).unwrap();
        assert_eq!(text.matches("\x1b_G").count(), 2);
        assert!(text.contains("m=1;"));
        assert!(text.contains("m=0;"));
    }

    #[test]
    fn kitty_escape_builds_the_delete_all_command() {
        assert_eq!(kitty_escape("a=d,d=A"), b"\x1b_Ga=d,d=A\x1b\\".to_vec());
    }

    #[test]
    fn kitty_loop_key_maps_loop_counts_to_the_protocols_off_by_one_encoding() {
        assert_eq!(kitty_loop_key(None), 1);
        assert_eq!(kitty_loop_key(Some(0)), 1);
        assert_eq!(kitty_loop_key(Some(1)), 2);
        assert_eq!(kitty_loop_key(Some(5)), 6);
    }

    #[test]
    fn should_stop_looping_treats_none_and_zero_as_infinite() {
        assert!(!should_stop_looping(None, 1000));
        assert!(!should_stop_looping(Some(0), 1000));
        assert!(!should_stop_looping(Some(3), 2));
        assert!(should_stop_looping(Some(3), 3));
        assert!(should_stop_looping(Some(3), 4));
    }
}
