use base64::Engine;
// -- iTerm2 protocol ------------------------------------------------------

/// OSC 1337 inline image (iTerm2's own protocol). Used unchanged for both
/// static images and animated GIFs -- iTerm2 decodes and loops a GIF's
/// animation itself, so the raw file bytes are enough either way.
pub(crate) fn iterm2_transmit(bytes: &[u8]) -> Vec<u8> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    let mut out = Vec::with_capacity(encoded.len() + 64);
    out.extend_from_slice(b"\x1b]1337;File=inline=1;size=");
    out.extend_from_slice(bytes.len().to_string().as_bytes());
    out.extend_from_slice(b";width=auto;height=auto;preserveAspectRatio=1:");
    out.extend_from_slice(encoded.as_bytes());
    out.push(0x07);
    out
}

#[cfg(test)]
mod tests {

    use crate::iterm2::iterm2_transmit;

    use base64::Engine;

    #[test]
    fn iterm2_transmit_matches_the_documented_osc_1337_format() {
        let bytes = b"hello-png-bytes";
        let escape = iterm2_transmit(bytes);
        let expected = format!(
            "\x1b]1337;File=inline=1;size={};width=auto;height=auto;preserveAspectRatio=1:{}\x07",
            bytes.len(),
            base64::engine::general_purpose::STANDARD.encode(bytes)
        );
        assert_eq!(escape, expected.into_bytes());
    }
}
