use crate::cmux_preview;
use crate::error::TerminalError;
use crate::frames::{decode_gif_frames, encode_frames_as_png};
use crate::iterm2::iterm2_transmit;
use crate::kitty::{
    kitty_escape, kitty_transmit, show_kitty_native_animation, show_kitty_simulated_animation,
};
use crate::protocol::{Protocol, resolve_protocol};
use crate::target::{
    TerminalResolution, open_sink, open_with_system_viewer, resolve_terminal_target,
};
use image::ImageFormat;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Message reported (as [`ShowOutcome::message`], status `"no_terminal"`)
/// when `run` is asked to clear and no terminal target can be found -- see
/// [`no_terminal_message`] for the equivalent used when displaying an image.
const NO_TERMINAL_CLEAR_MESSAGE: &str = "no interactive terminal detected; nothing to clear";

/// Message reported (as [`ShowOutcome::message`], status `"no_terminal"`)
/// when neither stdout nor any ancestor process has a real controlling
/// terminal to write escape sequences to -- not a failure, since the image
/// was still saved to disk at `path`.
fn no_terminal_message(path: &Path) -> String {
    format!(
        "no interactive terminal detected; image saved at {}, open it manually",
        path.display()
    )
}

/// Message reported (as [`ShowOutcome::message`], status `"displayed"`,
/// protocol `"system_viewer"`) when `run` opens the image with the OS's own
/// default file opener instead of writing terminal escape sequences -- see
/// [`open_with_system_viewer`] for why.
const SYSTEM_VIEWER_MESSAGE: &str = "no controlling terminal was safely writable for this \
     process (see docs); opened the image in the system's default viewer instead";

/// Message reported (as [`ShowOutcome::message`], status `"no_op"`) for a
/// clear request that has nothing to do, because the corresponding `show`
/// call (if any) went through [`open_with_system_viewer`] rather than
/// writing any inline terminal state that could be cleared.
const SYSTEM_VIEWER_CLEAR_MESSAGE: &str = "the image (if any) was opened in a separate system \
     viewer window rather than displayed inline; close that window manually";

/// Request to [`run`].
#[derive(Debug, Clone)]
pub struct ShowRequest {
    /// Path to a local PNG or GIF. Only allowed to be `None` when `clear`
    /// is `true`.
    pub path: Option<PathBuf>,
    /// Write escape sequences to this device file instead of discovering a
    /// terminal automatically (e.g. `/dev/ttys008`). Mainly useful for
    /// targeting a specific terminal session other than the caller's own.
    pub tty: Option<PathBuf>,
    /// Which terminal graphics protocol to use: `"auto"` (the default when
    /// `None`), `"kitty"`, or `"iterm2"`. Irrelevant whenever cmux or the
    /// system-viewer fallback ends up being used instead -- see [`run`].
    pub protocol: Option<String>,
    /// Send only a delete/clear command (a Kitty delete-all-images command,
    /// or cmux's surface-close call) and stop, instead of displaying an
    /// image.
    pub clear: bool,
    /// Number of times to loop a simulated/native GIF animation (`None` or
    /// `Some(0)` both mean loop forever, matching normal GIF playback).
    pub loops: Option<u32>,
}

/// Outcome of a successful [`run`] call.
///
/// `status` is one of `"displayed"`, `"cleared"`, `"no_terminal"`, or
/// `"no_op"`. The latter two are still successes (not a [`TerminalError`]):
/// `"no_terminal"` means no terminal target could be found at all (the
/// image, if any, was still saved to disk -- see `message`), and `"no_op"`
/// means a `clear` request had nothing to clear (no cmux preview surface
/// was recorded). Callers that want a human-readable summary rather than
/// matching on `status` can just print `message` when it is `Some`.
#[derive(Debug, Clone)]
pub struct ShowOutcome {
    pub status: &'static str,
    /// The protocol actually used: `"kitty"`, `"kitty-animation"`,
    /// `"kitty-simulated"`, `"iterm2"`, `"cmux"`, or `"system_viewer"`.
    /// `None` for the `"no_terminal"` status, and for a terminal-protocol
    /// `"cleared"` (only a cmux clear reports a protocol).
    pub protocol: Option<String>,
    /// The image path this outcome concerns, when relevant (`"displayed"`
    /// and the `"no_terminal"` case reached while displaying an image).
    pub path: Option<PathBuf>,
    /// A human-readable message, populated for the `"no_terminal"` and
    /// `"no_op"` statuses.
    pub message: Option<String>,
}

/// Entry point: display an image (or clear a previous one) per `request`.
pub fn run(request: ShowRequest) -> Result<ShowOutcome, TerminalError> {
    if request.clear {
        return run_clear(request.tty.as_deref());
    }
    let path = request.path.ok_or(TerminalError::MissingPath)?;

    // Prefer cmux's native file-preview panel over every terminal graphics
    // protocol below, when it's available -- see `cmux_preview` for why.
    // `Ok(None)` means cmux isn't available and we fall through unchanged
    // to the existing detection/behavior.
    if let Some(protocol_name) = cmux_preview::try_show(&path)? {
        return Ok(ShowOutcome {
            status: "displayed",
            protocol: Some(protocol_name.to_string()),
            path: Some(path),
            message: None,
        });
    }

    // Determine graphics-protocol capability purely from environment
    // signals, independent of whether there's actually somewhere safe to
    // write escape sequences to (that's resolved next). `None` means
    // neither Kitty's nor iTerm2's protocol was detected -- Apple's
    // Terminal.app included, since it implements neither -- in which case
    // there's no text-based fallback to reach for any more; go straight to
    // the system viewer.
    let Some(protocol_choice) = resolve_protocol(request.protocol.as_deref().unwrap_or("auto"))?
    else {
        return Ok(system_viewer_or_no_terminal_outcome(path));
    };

    let target = match resolve_terminal_target(request.tty.as_deref()) {
        TerminalResolution::NoTerminal => {
            return Ok(ShowOutcome {
                status: "no_terminal",
                protocol: None,
                message: Some(no_terminal_message(&path)),
                path: Some(path),
            });
        }
        TerminalResolution::DiscoveredDevice => {
            return Ok(system_viewer_or_no_terminal_outcome(path));
        }
        TerminalResolution::Target(target) => target,
    };

    let bytes = fs::read(&path).map_err(|source| TerminalError::Read {
        path: path.clone(),
        source,
    })?;
    let format = image::guess_format(&bytes).ok();
    let mut sink = open_sink(&target)?;

    let protocol_name = match format {
        Some(ImageFormat::Png) => show_png(&mut *sink, &bytes, protocol_choice)?,
        Some(ImageFormat::Gif) => {
            show_gif(&mut *sink, &path, &bytes, protocol_choice, request.loops)?
        }
        _ => return Err(TerminalError::UnsupportedImage { path }),
    };

    Ok(ShowOutcome {
        status: "displayed",
        protocol: Some(protocol_name.to_string()),
        path: Some(path),
        message: None,
    })
}

/// Shared by both places `run` gives up on writing escape sequences
/// anywhere (no known graphics-protocol terminal detected, or no tty this
/// process can safely write into): try the OS's own default file opener
/// instead, falling back to reporting `no_terminal` only if that itself
/// isn't available (see [`open_with_system_viewer`]).
fn system_viewer_or_no_terminal_outcome(path: PathBuf) -> ShowOutcome {
    if open_with_system_viewer(&path) {
        ShowOutcome {
            status: "displayed",
            protocol: Some("system_viewer".to_string()),
            path: Some(path),
            message: Some(SYSTEM_VIEWER_MESSAGE.to_string()),
        }
    } else {
        ShowOutcome {
            status: "no_terminal",
            protocol: None,
            message: Some(no_terminal_message(&path)),
            path: Some(path),
        }
    }
}

/// `clear=true`: send only a Kitty-protocol delete-all-images command (or,
/// under cmux, close the last-opened preview surface) and stop.
fn run_clear(tty: Option<&Path>) -> Result<ShowOutcome, TerminalError> {
    // Same preference as `run`: when cmux is available, handle `clear`
    // entirely through it (closing the last-opened preview surface) rather
    // than falling back to a Kitty delete-all command that would just
    // collide with the agent's TUI the same way this feature exists to
    // avoid.
    match cmux_preview::try_clear()? {
        cmux_preview::CmuxClearOutcome::Cleared => {
            return Ok(ShowOutcome {
                status: "cleared",
                protocol: Some("cmux".to_string()),
                path: None,
                message: None,
            });
        }
        cmux_preview::CmuxClearOutcome::NoSurface => {
            return Ok(ShowOutcome {
                status: "no_op",
                protocol: Some("cmux".to_string()),
                path: None,
                message: Some("no cmux preview surface recorded; nothing to clear".to_string()),
            });
        }
        cmux_preview::CmuxClearOutcome::Unavailable => {}
    }
    match resolve_terminal_target(tty) {
        TerminalResolution::NoTerminal => Ok(ShowOutcome {
            status: "no_terminal",
            protocol: None,
            path: None,
            message: Some(NO_TERMINAL_CLEAR_MESSAGE.to_string()),
        }),
        TerminalResolution::DiscoveredDevice => Ok(ShowOutcome {
            status: "no_op",
            protocol: Some("system_viewer".to_string()),
            path: None,
            message: Some(SYSTEM_VIEWER_CLEAR_MESSAGE.to_string()),
        }),
        TerminalResolution::Target(target) => {
            let mut sink = open_sink(&target)?;
            sink.write_all(&kitty_escape("a=d,d=A"))?;
            sink.flush()?;
            Ok(ShowOutcome {
                status: "cleared",
                protocol: None,
                path: None,
                message: None,
            })
        }
    }
}

fn show_png(
    sink: &mut dyn Write,
    bytes: &[u8],
    protocol: Protocol,
) -> Result<&'static str, TerminalError> {
    match protocol {
        Protocol::Kitty { .. } => {
            sink.write_all(&kitty_transmit("a=T,f=100,i=1,q=2", bytes))?;
            sink.flush()?;
            Ok("kitty")
        }
        Protocol::Iterm2 => {
            sink.write_all(&iterm2_transmit(bytes))?;
            sink.flush()?;
            Ok("iterm2")
        }
    }
}

fn show_gif(
    sink: &mut dyn Write,
    path: &Path,
    bytes: &[u8],
    protocol: Protocol,
    loops: Option<u32>,
) -> Result<&'static str, TerminalError> {
    match protocol {
        Protocol::Iterm2 => {
            // iTerm2 decodes and loops the animation itself; the raw GIF
            // bytes go through unchanged, same mechanism as a static image.
            sink.write_all(&iterm2_transmit(bytes))?;
            sink.flush()?;
            Ok("iterm2")
        }
        Protocol::Kitty { animation_capable } => {
            let frames = decode_gif_frames(path)?;
            if frames.is_empty() {
                return Err(TerminalError::UnsupportedImage {
                    path: path.to_path_buf(),
                });
            }
            let png_frames = encode_frames_as_png(path, &frames)?;
            if animation_capable {
                show_kitty_native_animation(sink, &png_frames, loops)?;
                Ok("kitty-animation")
            } else {
                show_kitty_simulated_animation(sink, &png_frames, loops)?;
                Ok("kitty-simulated")
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::show::ShowRequest;
    use crate::show::no_terminal_message;
    use crate::show::run;

    use std::path::Path;

    #[test]
    fn no_terminal_message_reports_exit_0_guidance_with_the_image_path() {
        let message = no_terminal_message(Path::new("/tmp/example.png"));
        assert!(message.contains("no interactive terminal detected"));
        assert!(message.contains("/tmp/example.png"));
        assert!(message.contains("open it manually"));
    }

    // `run`'s own top-level orchestration (path-required-unless-clear,
    // format dispatch, protocol resolution) is covered end to end by
    // `crates/cli/tests/cli.rs`'s `show_*` tests, which exercise it through
    // the real `renderer` binary with `CMUX_SOCKET_PATH` explicitly
    // controlled per test (removed, or pointed at a fake server). That
    // control isn't available to an in-process unit test here: `run` always
    // consults the real `CMUX_SOCKET_PATH` from this test process's actual
    // environment (which, on a machine developing this crate under cmux
    // itself, is live and connectable), so calling `run` directly with a
    // real path from a unit test would race a real cmux RPC rather than
    // exercising the format/protocol logic under test. `MissingPath` is the
    // one exception: it's returned before `run` ever consults cmux, so it's
    // safe to exercise directly here.
    #[test]
    fn run_reports_missing_path_when_not_clearing() {
        let request = ShowRequest {
            path: None,
            tty: None,
            protocol: None,
            clear: false,
            loops: None,
        };
        let error = run(request).unwrap_err();
        assert_eq!(error.code(), "invalid_argument");
    }
}
