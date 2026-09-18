//! Inline terminal image/GIF preview, shared by the `renderer` CLI's `show`
//! subcommand and the `renderer-mcp` server's `show_image` tool.
//!
//! This crate is the single implementation behind both surfaces, per this
//! project's founding "CLI and MCP expose the same capabilities" principle:
//! terminal detection, a Kitty graphics protocol encoder with animation
//! support, an iTerm2 OSC 1337 encoder, an ANSI 24-bit half-block fallback
//! encoder with its own animation redraw loop, a process-ancestry
//! tty-discovery mechanism for when the calling process has no controlling
//! terminal of its own, and a `cmux_preview` submodule that talks to cmux's
//! native file-preview feature over a JSON-RPC Unix socket when
//! `$CMUX_SOCKET_PATH` is available.
//!
//! The public entry point is [`run`]: build a [`ShowRequest`], call `run`,
//! and render the returned [`ShowOutcome`] (or [`TerminalError`]) however
//! the caller's own protocol requires -- the CLI turns it into its
//! structured JSON-on-stdout / `{"code","message"}`-on-stderr convention,
//! and the MCP server turns it into a text content block.

use base64::Engine;
use image::codecs::gif::GifDecoder;
use image::imageops::FilterType;
use image::{AnimationDecoder, ImageFormat, Rgba, RgbaImage};
use is_terminal::IsTerminal;
use std::{
    fs,
    io::{Cursor, Write},
    path::{Path, PathBuf},
    process::Command as ProcessCommand,
    time::Duration,
};

/// Cap on how many ancestor processes to walk before giving up on finding a
/// real controlling terminal -- avoids any pathological loop in a broken
/// process tree.
const MAX_ANCESTOR_DEPTH: u32 = 32;
/// Kitty graphics protocol payload chunk size, in base64-encoded bytes.
const KITTY_CHUNK_LIMIT: usize = 4096;
/// Fixed background the ANSI half-block fallback composites partially
/// transparent pixels against, since terminal cells have no alpha channel of
/// their own. Chosen to match typical dark terminal backgrounds -- there is
/// no universally "correct" choice here, so a transparent pixel simply
/// renders as this color rather than as the viewer's real terminal
/// background.
const ANSI_BACKGROUND: (u8, u8, u8) = (0, 0, 0);
/// Fallback terminal size used when writing to a discovered ancestor tty
/// device, where querying the real size isn't practical.
const DEFAULT_TERMINAL_COLUMNS: u16 = 80;
const DEFAULT_TERMINAL_ROWS: u16 = 24;
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
    /// `None`), `"kitty"`, `"iterm2"`, or `"ansi"`.
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
    /// `"kitty-simulated"`, `"iterm2"`, `"ansi"`, `"ansi-simulated"`, or
    /// `"cmux"`. `None` for the `"no_terminal"` status, and for a
    /// terminal-protocol `"cleared"` (only a cmux clear reports a
    /// protocol).
    pub protocol: Option<String>,
    /// The image path this outcome concerns, when relevant (`"displayed"`
    /// and the `"no_terminal"` case reached while displaying an image).
    pub path: Option<PathBuf>,
    /// A human-readable message, populated for the `"no_terminal"` and
    /// `"no_op"` statuses.
    pub message: Option<String>,
}

/// Every failure kind `run` can report. Mirrors the style of `RenderError`
/// in `crates/renderer/src/lib.rs` and `DaemonError` in
/// `crates/daemon/src/lib.rs`: one variant per distinguishable failure,
/// with a stable [`TerminalError::code`] a caller can surface directly
/// (the CLI's `show` command downcasts to this type for exactly that).
#[derive(Debug, thiserror::Error)]
pub enum TerminalError {
    /// `path` was `None` with `clear` false. The CLI's clap parser rejects
    /// this before it ever reaches this crate (`path` is
    /// `required_unless_present = "clear"`), but a direct caller like the
    /// MCP server's `show_image` tool has no such parser in front of it.
    #[error("path is required unless clear is true")]
    MissingPath,
    #[error("could not read {}: {source}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not inspect {}: {source}", path.display())]
    Image {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error(
        "{} is not a supported image for `show` (only PNG and GIF are accepted)",
        path.display()
    )]
    UnsupportedImage { path: PathBuf },
    #[error("unknown --protocol '{value}' (expected auto, kitty, iterm2, or ansi)")]
    InvalidProtocol { value: String },
    #[error("could not open terminal device {}: {source}", path.display())]
    TtyOpen {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Any failure talking to cmux's JSON-RPC control socket once a
    /// connection has actually been established -- a transport error, a
    /// malformed response, or an `ok: false` application-level error. A
    /// failure to connect at all is deliberately *not* represented here:
    /// that's treated as "cmux isn't available" and falls back to the
    /// terminal-protocol path instead of erroring.
    #[error("cmux error: {message}")]
    Cmux { message: String },
    /// A write/flush failure while sending an already-resolved protocol's
    /// escape sequences to an already-open terminal target. Distinct from
    /// [`TerminalError::TtyOpen`] (which covers failing to open the
    /// device in the first place).
    #[error("terminal write failed: {0}")]
    Io(#[from] std::io::Error),
}

impl TerminalError {
    /// A stable, machine-readable error code, matching the codes this
    /// crate's logic has always produced back when it lived in
    /// `crates/cli/src/main.rs`'s `CliError`.
    pub fn code(&self) -> &'static str {
        match self {
            TerminalError::MissingPath => "invalid_argument",
            TerminalError::Read { .. }
            | TerminalError::Image { .. }
            | TerminalError::TtyOpen { .. }
            | TerminalError::Io(_) => "io_error",
            TerminalError::UnsupportedImage { .. } => "unsupported_image",
            TerminalError::InvalidProtocol { .. } => "invalid_protocol",
            TerminalError::Cmux { .. } => "cmux_error",
        }
    }
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

    let target = match resolve_terminal_target(request.tty.as_deref()) {
        TerminalResolution::NoTerminal => {
            return Ok(ShowOutcome {
                status: "no_terminal",
                protocol: None,
                message: Some(no_terminal_message(&path)),
                path: Some(path),
            });
        }
        TerminalResolution::Target(target) => target,
    };

    let bytes = fs::read(&path).map_err(|source| TerminalError::Read {
        path: path.clone(),
        source,
    })?;
    let format = image::guess_format(&bytes).ok();
    let protocol_choice = resolve_protocol(request.protocol.as_deref().unwrap_or("auto"))?;
    let mut sink = open_sink(&target)?;

    let protocol_name = match format {
        Some(ImageFormat::Png) => show_png(&mut *sink, &path, &bytes, protocol_choice, &target)?,
        Some(ImageFormat::Gif) => show_gif(
            &mut *sink,
            &path,
            &bytes,
            protocol_choice,
            &target,
            request.loops,
        )?,
        _ => return Err(TerminalError::UnsupportedImage { path }),
    };

    Ok(ShowOutcome {
        status: "displayed",
        protocol: Some(protocol_name.to_string()),
        path: Some(path),
        message: None,
    })
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
    path: &Path,
    bytes: &[u8],
    protocol: Protocol,
    target: &TerminalTarget,
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
        Protocol::Ansi => {
            let decoded = image::load_from_memory(bytes)
                .map_err(|source| TerminalError::Image {
                    path: path.to_path_buf(),
                    source,
                })?
                .to_rgba8();
            let (columns, rows) = ansi_target_size(target);
            let (width, height) = fit_dimensions(decoded.width(), decoded.height(), columns, rows);
            let resized = image::imageops::resize(&decoded, width, height, FilterType::Triangle);
            let (rendered, _rows) = render_ansi_frame(&resized);
            sink.write_all(&rendered)?;
            sink.flush()?;
            Ok("ansi")
        }
    }
}

fn show_gif(
    sink: &mut dyn Write,
    path: &Path,
    bytes: &[u8],
    protocol: Protocol,
    target: &TerminalTarget,
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
        Protocol::Ansi => {
            let frames = decode_gif_frames(path)?;
            if frames.is_empty() {
                return Err(TerminalError::UnsupportedImage {
                    path: path.to_path_buf(),
                });
            }
            let (columns, rows) = ansi_target_size(target);
            let mut rendered_frames = Vec::with_capacity(frames.len());
            let mut row_count = 0usize;
            for (image, delay) in &frames {
                let (width, height) = fit_dimensions(image.width(), image.height(), columns, rows);
                let resized = image::imageops::resize(image, width, height, FilterType::Triangle);
                let (rendered, rows_used) = render_ansi_frame(&resized);
                row_count = rows_used;
                rendered_frames.push((rendered, *delay));
            }
            show_ansi_animation(sink, &rendered_frames, row_count, loops)?;
            Ok("ansi-simulated")
        }
    }
}

// -- Terminal capability detection -------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Protocol {
    Kitty { animation_capable: bool },
    Iterm2,
    Ansi,
}

fn resolve_protocol(value: &str) -> Result<Protocol, TerminalError> {
    match value {
        "auto" => Ok(detect_protocol_auto(&env_lookup)),
        "kitty" => Ok(Protocol::Kitty {
            animation_capable: kitty_animation_capable(&env_lookup),
        }),
        "iterm2" => Ok(Protocol::Iterm2),
        "ansi" => Ok(Protocol::Ansi),
        other => Err(TerminalError::InvalidProtocol {
            value: other.to_string(),
        }),
    }
}

fn env_lookup(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Kitty-graphics-protocol terminals, in the documented priority order:
/// Kitty itself, Ghostty, cmux (Ghostty-based), and WezTerm.
fn kitty_protocol_terminal(get: &dyn Fn(&str) -> Option<String>) -> bool {
    let set = |name: &str| get(name).map(|value| !value.is_empty()).unwrap_or(false);
    let eq = |name: &str, value: &str| get(name).as_deref() == Some(value);
    set("KITTY_WINDOW_ID")
        || eq("TERM", "xterm-kitty")
        || eq("TERM_PROGRAM", "ghostty")
        || set("GHOSTTY_RESOURCES_DIR")
        || set("CMUX_WORKSPACE_ID")
        || set("CMUX_SURFACE_ID")
        || eq("TERM_PROGRAM", "WezTerm")
}

/// Whether a detected Kitty-protocol terminal also supports the Kitty
/// animation extension. WezTerm and plain Kitty do; Ghostty and cmux
/// (Ghostty-based) do not, as of this research -- so a GIF headed for
/// either of those falls back to simulated animation instead.
fn kitty_animation_capable(get: &dyn Fn(&str) -> Option<String>) -> bool {
    let set = |name: &str| get(name).map(|value| !value.is_empty()).unwrap_or(false);
    let eq = |name: &str, value: &str| get(name).as_deref() == Some(value);
    eq("TERM_PROGRAM", "WezTerm") || eq("TERM", "xterm-kitty") || set("KITTY_WINDOW_ID")
}

fn detect_protocol_auto(get: &dyn Fn(&str) -> Option<String>) -> Protocol {
    if kitty_protocol_terminal(get) {
        Protocol::Kitty {
            animation_capable: kitty_animation_capable(get),
        }
    } else if get("TERM_PROGRAM").as_deref() == Some("iTerm.app") {
        Protocol::Iterm2
    } else {
        Protocol::Ansi
    }
}

// -- Terminal output target resolution ----------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum TerminalTarget {
    Stdout,
    Device(PathBuf),
}

#[derive(Debug)]
enum TerminalResolution {
    Target(TerminalTarget),
    NoTerminal,
}

fn resolve_terminal_target(explicit_tty: Option<&Path>) -> TerminalResolution {
    if let Some(path) = explicit_tty {
        return TerminalResolution::Target(TerminalTarget::Device(path.to_path_buf()));
    }
    if std::io::stdout().is_terminal() {
        return TerminalResolution::Target(TerminalTarget::Stdout);
    }
    #[cfg(unix)]
    {
        if let Some(device) = discover_ancestor_tty_device() {
            return TerminalResolution::Target(TerminalTarget::Device(device));
        }
    }
    TerminalResolution::NoTerminal
}

fn open_sink(target: &TerminalTarget) -> Result<Box<dyn Write>, TerminalError> {
    match target {
        TerminalTarget::Stdout => Ok(Box::new(std::io::stdout())),
        TerminalTarget::Device(path) => {
            let file = fs::OpenOptions::new()
                .write(true)
                .open(path)
                .map_err(|source| TerminalError::TtyOpen {
                    path: path.clone(),
                    source,
                })?;
            Ok(Box::new(file))
        }
    }
}

/// True when a `ps` `tty` column value names a real terminal device rather
/// than "no controlling terminal" -- BSD `ps` (macOS) prints `??`, GNU `ps`
/// (Linux) prints `?`, and both may print nothing at all.
fn is_real_tty_name(tty: &str) -> bool {
    let trimmed = tty.trim();
    !(trimmed.is_empty() || trimmed == "?" || trimmed == "??" || trimmed == "-")
}

/// Parses one line of `ps -o ppid=,tty= -p <pid>` output. Defensive about
/// whitespace since BSD and GNU `ps` pad columns differently.
fn parse_ps_ppid_tty(output: &str) -> Option<(u32, String)> {
    let line = output.lines().find(|line| !line.trim().is_empty())?;
    let mut fields = line.split_whitespace();
    let ppid = fields.next()?.parse::<u32>().ok()?;
    let tty = fields.next()?.to_string();
    Some((ppid, tty))
}

/// Walks the process ancestry starting at `start_pid`, calling `lookup`
/// (expected to behave like `ps -o ppid=,tty= -p <pid>`) for each ancestor
/// until one has a real controlling terminal, `ppid` reaches 0 or itself,
/// `lookup`/parsing fails, or `max_depth` is exceeded. `lookup` is injected
/// so this is testable against canned `ps` output instead of a live process
/// tree.
fn walk_ancestors_for_tty<F>(start_pid: u32, max_depth: u32, mut lookup: F) -> Option<String>
where
    F: FnMut(u32) -> Option<String>,
{
    let mut pid = start_pid;
    let mut visited = std::collections::HashSet::new();
    for _ in 0..max_depth {
        if !visited.insert(pid) {
            return None;
        }
        let raw = lookup(pid)?;
        let (ppid, tty) = parse_ps_ppid_tty(&raw)?;
        if is_real_tty_name(&tty) {
            return Some(tty);
        }
        if ppid == 0 || ppid == pid {
            return None;
        }
        pid = ppid;
    }
    None
}

#[cfg(unix)]
fn discover_ancestor_tty_device() -> Option<PathBuf> {
    let tty_name = walk_ancestors_for_tty(std::process::id(), MAX_ANCESTOR_DEPTH, |pid| {
        let output = ProcessCommand::new("ps")
            .args(["-o", "ppid=,tty=", "-p", &pid.to_string()])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&output.stdout).into_owned())
    })?;
    Some(PathBuf::from(format!("/dev/{tty_name}")))
}

// -- Kitty graphics protocol ---------------------------------------------

/// Builds one no-payload Kitty APC command, e.g. the delete-all command.
fn kitty_escape(control: &str) -> Vec<u8> {
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
fn kitty_transmit(control: &str, payload: &[u8]) -> Vec<u8> {
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
fn show_kitty_native_animation(
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
fn show_kitty_simulated_animation(
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

// -- iTerm2 protocol ------------------------------------------------------

/// OSC 1337 inline image (iTerm2's own protocol). Used unchanged for both
/// static images and animated GIFs -- iTerm2 decodes and loops a GIF's
/// animation itself, so the raw file bytes are enough either way.
fn iterm2_transmit(bytes: &[u8]) -> Vec<u8> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    let mut out = Vec::with_capacity(encoded.len() + 64);
    out.extend_from_slice(b"\x1b]1337;File=inline=1;size=");
    out.extend_from_slice(bytes.len().to_string().as_bytes());
    out.extend_from_slice(b";width=auto;height=auto;preserveAspectRatio=1:");
    out.extend_from_slice(encoded.as_bytes());
    out.push(0x07);
    out
}

// -- ANSI half-block fallback ---------------------------------------------

/// Composites a possibly-transparent pixel over `ANSI_BACKGROUND` --
/// terminal cells have no alpha channel, so partial transparency is
/// approximated by alpha-blending against a fixed background color rather
/// than the viewer's real (unknowable) terminal background.
fn composite_over_background(pixel: Rgba<u8>) -> (u8, u8, u8) {
    let [r, g, b, a] = pixel.0;
    let alpha = f32::from(a) / 255.0;
    let blend = |channel: u8, background: u8| -> u8 {
        (f32::from(channel) * alpha + f32::from(background) * (1.0 - alpha)).round() as u8
    };
    (
        blend(r, ANSI_BACKGROUND.0),
        blend(g, ANSI_BACKGROUND.1),
        blend(b, ANSI_BACKGROUND.2),
    )
}

/// Renders one frame as upper-half-block rows: each terminal row covers two
/// source pixel rows (foreground = top pixel, background = bottom pixel),
/// reset at the end of each row. An odd final source row reuses the top
/// pixel as its own bottom half. Returns the encoded bytes and the number
/// of terminal rows they occupy (for cursor-repositioning during
/// animation).
fn render_ansi_frame(image: &RgbaImage) -> (Vec<u8>, usize) {
    let width = image.width();
    let height = image.height();
    let rows = height.div_ceil(2) as usize;
    let mut out = Vec::new();
    for row in 0..rows {
        let top_y = (row * 2) as u32;
        let bottom_y = top_y + 1;
        for x in 0..width {
            let (tr, tg, tb) = composite_over_background(*image.get_pixel(x, top_y));
            let (br, bg, bb) = if bottom_y < height {
                composite_over_background(*image.get_pixel(x, bottom_y))
            } else {
                (tr, tg, tb)
            };
            out.extend_from_slice(
                format!("\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m\u{2580}").as_bytes(),
            );
        }
        out.extend_from_slice(b"\x1b[0m\n");
    }
    (out, rows)
}

/// Fits `source_width`x`source_height` into the given terminal grid,
/// leaving a small margin, at one column per pixel horizontally and one
/// row per two pixels vertically, preserving aspect ratio. Height is
/// always rounded up to an even number of source rows.
fn fit_dimensions(source_width: u32, source_height: u32, columns: u16, rows: u16) -> (u32, u32) {
    let margin_columns = 2u32;
    let margin_rows = 1u32;
    let max_columns = u32::from(columns).saturating_sub(margin_columns).max(1);
    let max_pixel_height = u32::from(rows).saturating_sub(margin_rows).max(1) * 2;
    let scale_w = f64::from(max_columns) / f64::from(source_width.max(1));
    let scale_h = f64::from(max_pixel_height) / f64::from(source_height.max(1));
    let scale = scale_w.min(scale_h);
    let width = ((f64::from(source_width) * scale).round() as u32).max(1);
    let mut height = ((f64::from(source_height) * scale).round() as u32).max(1);
    if !height.is_multiple_of(2) {
        height += 1;
    }
    (width, height.max(2))
}

fn ansi_target_size(target: &TerminalTarget) -> (u16, u16) {
    match target {
        TerminalTarget::Stdout => terminal_size::terminal_size()
            .map(|(width, height)| (width.0, height.0))
            .unwrap_or((DEFAULT_TERMINAL_COLUMNS, DEFAULT_TERMINAL_ROWS)),
        TerminalTarget::Device(_) => (DEFAULT_TERMINAL_COLUMNS, DEFAULT_TERMINAL_ROWS),
    }
}

/// Simulated animation for the ANSI half-block fallback: redraw each frame
/// in place by moving the cursor back up over the previous frame's rows
/// before drawing the next one, sleeping for its delay in between, until
/// `loops` is exhausted or the process is killed.
fn show_ansi_animation(
    sink: &mut dyn Write,
    frames: &[(Vec<u8>, u32)],
    row_count: usize,
    loops: Option<u32>,
) -> std::io::Result<()> {
    let mut played = 0u32;
    let mut drawn_before = false;
    loop {
        for (rendered, delay) in frames {
            if drawn_before {
                sink.write_all(format!("\x1b[{row_count}A").as_bytes())?;
            }
            sink.write_all(rendered)?;
            sink.flush()?;
            drawn_before = true;
            std::thread::sleep(Duration::from_millis(u64::from(*delay)));
        }
        played += 1;
        if should_stop_looping(loops, played) {
            break;
        }
    }
    Ok(())
}

// -- GIF decoding -----------------------------------------------------------

fn decode_gif_frames(path: &Path) -> Result<Vec<(RgbaImage, u32)>, TerminalError> {
    let file = fs::File::open(path).map_err(|source| TerminalError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let decoder =
        GifDecoder::new(std::io::BufReader::new(file)).map_err(|source| TerminalError::Image {
            path: path.to_path_buf(),
            source,
        })?;
    let frames = decoder
        .into_frames()
        .collect_frames()
        .map_err(|source| TerminalError::Image {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(frames
        .into_iter()
        .map(|frame| {
            let (numerator, denominator) = frame.delay().numer_denom_ms();
            let delay_ms = numerator.checked_div(denominator).unwrap_or(numerator);
            (frame.into_buffer(), delay_ms.max(1))
        })
        .collect())
}

fn encode_frame_png(path: &Path, image: &RgbaImage) -> Result<Vec<u8>, TerminalError> {
    let mut bytes = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);
    image::DynamicImage::ImageRgba8(image.clone())
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|source| TerminalError::Image {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(bytes)
}

fn encode_frames_as_png(
    path: &Path,
    frames: &[(RgbaImage, u32)],
) -> Result<Vec<(Vec<u8>, u32)>, TerminalError> {
    frames
        .iter()
        .map(|(image, delay)| Ok((encode_frame_png(path, image)?, *delay)))
        .collect()
}

// -- cmux native file preview --------------------------------------------
//
// cmux (a native macOS terminal for running coding agents, built on
// Ghostty) exposes a JSON-RPC-over-Unix-socket control API at the path
// named by `CMUX_SOCKET_PATH`. Its `file.open` method opens a local file in
// cmux's own native file-preview panel -- a UI surface entirely separate
// from the terminal grid, so unlike every protocol above it can never
// collide with an agent's own actively-redrawn TUI (the motivating
// problem: a detached `show` subprocess's raw pty write has no way to know
// where a chat transcript ends and an input box begins). This is tried
// first, ahead of all Kitty/iTerm2/ANSI detection, and used instead of it
// entirely whenever cmux is reachable. `UnixStream` is POSIX-only, so this
// whole integration is unix-only; on other platforms (this workspace's CI
// includes a Windows target) it compiles to a stub that always reports
// cmux as unavailable.
//
// The env-var/working-directory-reading entry points (`try_show`,
// `try_clear`) are thin wrappers around `_at`-suffixed functions that take
// the socket path and state-file path as explicit arguments -- mirroring
// `detect_protocol_auto`'s injected-`get` pattern above, and for the same
// reason: `std::env::set_var`/`set_current_dir` are process-global and
// would race across this crate's parallel test threads, so the tests
// exercise the `_at` functions directly instead.
#[cfg(unix)]
mod cmux_preview {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::io::{BufRead, BufReader};
    use std::os::unix::net::UnixStream;

    /// Env var cmux sets to the path of its control socket.
    const SOCKET_ENV: &str = "CMUX_SOCKET_PATH";
    /// Read/write timeout applied to the control-socket connection, so a
    /// hung or misbehaving socket can't make `show` hang forever.
    const SOCKET_TIMEOUT: Duration = Duration::from_secs(3);
    /// Where the most recently opened preview surface's id is persisted,
    /// so a later clear can close it. Relative to the working directory,
    /// following the same `.renderer/`-prefixed local generated-state
    /// convention as the daemon's `.renderer/metrics` directory (see
    /// `DEFAULT_METRICS_DIR` in `crates/daemon/src/lib.rs`).
    const STATE_PATH: &str = ".renderer/cmux-preview-surface.json";

    #[derive(Debug, Serialize)]
    struct Request<'a> {
        id: &'a str,
        method: &'a str,
        params: serde_json::Value,
    }

    #[derive(Debug, Deserialize)]
    struct Response {
        ok: bool,
        #[serde(default)]
        result: Option<serde_json::Value>,
        #[serde(default)]
        error: Option<RpcError>,
    }

    #[derive(Debug, Deserialize)]
    struct RpcError {
        message: String,
        #[serde(default)]
        code: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct State {
        surface_id: String,
    }

    /// The result of attempting a cmux `clear`, distinguishing "cmux isn't
    /// available at all" (the caller should fall back to the
    /// terminal-protocol clear path) from the two outcomes once cmux *is*
    /// reachable.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum CmuxClearOutcome {
        Unavailable,
        Cleared,
        NoSurface,
    }

    fn socket_path() -> Option<PathBuf> {
        std::env::var(SOCKET_ENV)
            .ok()
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    }

    fn state_path() -> PathBuf {
        PathBuf::from(STATE_PATH)
    }

    /// Connects to cmux's control socket, applying `SOCKET_TIMEOUT` in both
    /// directions. Any failure at all (no such socket, connection refused,
    /// a timeout while configuring the connection, ...) is treated as
    /// "cmux isn't available" rather than an error -- per the spec, callers
    /// fall back to the terminal-protocol path instead of failing `show`
    /// outright.
    fn connect(path: &Path) -> Option<UnixStream> {
        let stream = UnixStream::connect(path).ok()?;
        stream.set_read_timeout(Some(SOCKET_TIMEOUT)).ok()?;
        stream.set_write_timeout(Some(SOCKET_TIMEOUT)).ok()?;
        Some(stream)
    }

    /// One request/response round-trip: writes a single newline-terminated
    /// JSON request and reads a single newline-terminated JSON response,
    /// per cmux's documented framing. Returns the response's `result` on
    /// success, or a `TerminalError::Cmux` describing whatever went wrong
    /// -- a transport failure, a malformed response, or an `ok: false`
    /// application-level error. Unlike `connect`, every failure here is a
    /// hard error: once we've established a connection we know cmux is
    /// present, so a failure from here on is a real problem rather than
    /// "try something else".
    fn call(
        stream: &mut UnixStream,
        id: &str,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, TerminalError> {
        let cmux_error = |message: String| TerminalError::Cmux { message };

        let mut payload = serde_json::to_vec(&Request { id, method, params })
            .map_err(|source| cmux_error(format!("could not encode request: {source}")))?;
        payload.push(b'\n');
        stream
            .write_all(&payload)
            .map_err(|source| cmux_error(format!("could not write to socket: {source}")))?;

        let mut line = String::new();
        BufReader::new(&mut *stream)
            .read_line(&mut line)
            .map_err(|source| cmux_error(format!("could not read from socket: {source}")))?;
        if line.trim().is_empty() {
            return Err(cmux_error(
                "connection closed without a response".to_string(),
            ));
        }

        let response: Response = serde_json::from_str(&line)
            .map_err(|source| cmux_error(format!("malformed response: {source}")))?;
        if !response.ok {
            let detail = response
                .error
                .map(|error| match error.code {
                    Some(code) => format!("{} ({code})", error.message),
                    None => error.message,
                })
                .unwrap_or_else(|| "unknown error".to_string());
            return Err(cmux_error(format!("rejected `{method}`: {detail}")));
        }
        response
            .result
            .ok_or_else(|| cmux_error(format!("no result for `{method}`")))
    }

    fn load_state(state_file: &Path) -> Option<State> {
        let contents = fs::read_to_string(state_file).ok()?;
        serde_json::from_str(&contents).ok()
    }

    /// Best-effort: an unwritable working directory shouldn't fail an
    /// otherwise-successful `show`, it just means a later clear won't have
    /// a recorded surface to close.
    fn save_state(state_file: &Path, surface_id: &str) {
        if let Some(parent) = state_file.parent()
            && fs::create_dir_all(parent).is_err()
        {
            return;
        }
        if let Ok(json) = serde_json::to_string(&State {
            surface_id: surface_id.to_string(),
        }) {
            let _ = fs::write(state_file, json);
        }
    }

    fn clear_state(state_file: &Path) {
        let _ = fs::remove_file(state_file);
    }

    /// `run`'s display path via cmux. `Ok(None)` means cmux isn't
    /// available (`CMUX_SOCKET_PATH` unset, or its socket didn't accept a
    /// connection) and the caller should fall back to terminal-protocol
    /// detection unchanged. Once a connection is established, a bad path
    /// or a rejected/malformed RPC is surfaced as a hard error instead.
    pub(super) fn try_show(path: &Path) -> Result<Option<&'static str>, TerminalError> {
        let Some(socket) = socket_path() else {
            return Ok(None);
        };
        try_show_at(&socket, &state_path(), path)
    }

    /// The testable core of `try_show`, with the socket and state-file
    /// paths taken as explicit arguments instead of read from the
    /// environment/working directory.
    fn try_show_at(
        socket: &Path,
        state_file: &Path,
        path: &Path,
    ) -> Result<Option<&'static str>, TerminalError> {
        let Some(mut stream) = connect(socket) else {
            return Ok(None);
        };
        let absolute = fs::canonicalize(path).map_err(|source| TerminalError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let result = call(
            &mut stream,
            "renderer-show",
            "file.open",
            serde_json::json!({ "path": absolute.to_string_lossy() }),
        )?;
        if let Some(surface_id) = result.get("surface_id").and_then(|value| value.as_str()) {
            save_state(state_file, surface_id);
        }
        Ok(Some("cmux"))
    }

    /// `run`'s clear path via cmux: closes the most recently opened
    /// preview surface, if one was recorded, via `surface.close`.
    pub(super) fn try_clear() -> Result<CmuxClearOutcome, TerminalError> {
        let Some(socket) = socket_path() else {
            return Ok(CmuxClearOutcome::Unavailable);
        };
        try_clear_at(&socket, &state_path())
    }

    /// The testable core of `try_clear`, with the socket and state-file
    /// paths taken as explicit arguments.
    fn try_clear_at(socket: &Path, state_file: &Path) -> Result<CmuxClearOutcome, TerminalError> {
        let Some(mut stream) = connect(socket) else {
            return Ok(CmuxClearOutcome::Unavailable);
        };
        match load_state(state_file) {
            Some(state) => {
                call(
                    &mut stream,
                    "renderer-clear",
                    "surface.close",
                    serde_json::json!({ "surface_id": state.surface_id }),
                )?;
                clear_state(state_file);
                Ok(CmuxClearOutcome::Cleared)
            }
            None => Ok(CmuxClearOutcome::NoSurface),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::os::unix::net::UnixListener;

        /// Spawns a fake cmux control-socket server bound to a tempdir
        /// path: accepts exactly one connection, reads one
        /// newline-terminated request, and writes back `response` (with a
        /// trailing newline added if it doesn't already end in one).
        /// Returns the socket path and a join handle yielding the raw
        /// request bytes it read.
        fn fake_server(response: &'static str) -> (PathBuf, std::thread::JoinHandle<Vec<u8>>) {
            let dir = tempfile::tempdir().unwrap();
            // Keep the tempdir alive for the server thread's lifetime by
            // leaking it -- fine for a short-lived test process, and
            // simpler than threading an extra lifetime through the join
            // handle.
            let socket_path = dir.keep().join("cmux.sock");
            let listener = UnixListener::bind(&socket_path).unwrap();
            let handle = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = Vec::new();
                let mut reader = BufReader::new(&mut stream);
                let _ = reader.read_until(b'\n', &mut request);
                let mut body = response.as_bytes().to_vec();
                if !body.ends_with(b"\n") {
                    body.push(b'\n');
                }
                let _ = stream.write_all(&body);
                request
            });
            (socket_path, handle)
        }

        #[test]
        fn try_show_at_round_trips_a_successful_file_open() {
            let (socket_path, handle) = fake_server(
                r#"{"id":"renderer-show","ok":true,"result":{"surface_id":"11111111-1111-1111-1111-111111111111","pane_id":"22222222-2222-2222-2222-222222222222"}}"#,
            );
            let state_dir = tempfile::tempdir().unwrap();
            let state_file = state_dir.path().join("cmux-preview-surface.json");

            let image_dir = tempfile::tempdir().unwrap();
            let image_path = image_dir.path().join("test.png");
            fs::write(&image_path, b"not-really-a-png-but-unused-here").unwrap();

            let result = try_show_at(&socket_path, &state_file, &image_path);
            assert_eq!(result.unwrap(), Some("cmux"));

            let request = String::from_utf8(handle.join().unwrap()).unwrap();
            let request: serde_json::Value = serde_json::from_str(request.trim()).unwrap();
            assert_eq!(request["method"], "file.open");
            let sent_path = request["params"]["path"].as_str().unwrap();
            assert!(Path::new(sent_path).is_absolute());
            assert_eq!(
                fs::canonicalize(sent_path).unwrap(),
                fs::canonicalize(&image_path).unwrap()
            );

            let saved_state = fs::read_to_string(&state_file).unwrap();
            assert!(saved_state.contains("11111111-1111-1111-1111-111111111111"));
        }

        #[test]
        fn try_show_at_falls_back_when_the_socket_is_unreachable() {
            let missing_socket = tempfile::tempdir().unwrap().path().join("no-such.sock");
            let state_file = tempfile::tempdir()
                .unwrap()
                .path()
                .join("cmux-preview-surface.json");
            let image_dir = tempfile::tempdir().unwrap();
            let image_path = image_dir.path().join("test.png");
            fs::write(&image_path, b"unused").unwrap();

            let result = try_show_at(&missing_socket, &state_file, &image_path);
            assert_eq!(result.unwrap(), None);
        }

        #[test]
        fn try_show_at_returns_a_clear_error_for_an_ok_false_response() {
            let (socket_path, _handle) = fake_server(
                r#"{"id":"renderer-show","ok":false,"error":{"message":"File not found: /nope.png","code":"not_found"}}"#,
            );
            let state_file = tempfile::tempdir()
                .unwrap()
                .path()
                .join("cmux-preview-surface.json");
            let image_dir = tempfile::tempdir().unwrap();
            let image_path = image_dir.path().join("test.png");
            fs::write(&image_path, b"unused").unwrap();

            let error = try_show_at(&socket_path, &state_file, &image_path).unwrap_err();
            assert_eq!(error.code(), "cmux_error");
            assert!(error.to_string().contains("File not found"));
            assert!(error.to_string().contains("not_found"));
        }

        #[test]
        fn try_show_at_returns_a_clear_error_for_a_malformed_response() {
            let (socket_path, _handle) = fake_server("not json at all");
            let state_file = tempfile::tempdir()
                .unwrap()
                .path()
                .join("cmux-preview-surface.json");
            let image_dir = tempfile::tempdir().unwrap();
            let image_path = image_dir.path().join("test.png");
            fs::write(&image_path, b"unused").unwrap();

            let error = try_show_at(&socket_path, &state_file, &image_path).unwrap_err();
            assert_eq!(error.code(), "cmux_error");
            assert!(error.to_string().contains("malformed response"));
        }

        #[test]
        fn try_show_at_reports_a_clear_error_when_the_path_cannot_be_canonicalized() {
            // The server is reachable but should never receive a request:
            // canonicalization is checked first and fails here since the
            // path doesn't exist.
            let dir = tempfile::tempdir().unwrap();
            let socket_path = dir.path().join("cmux.sock");
            let listener = UnixListener::bind(&socket_path).unwrap();
            let handle = std::thread::spawn(move || listener.accept());
            let state_file = dir.path().join("cmux-preview-surface.json");

            let error = try_show_at(
                &socket_path,
                &state_file,
                Path::new("/no/such/path/definitely-missing.png"),
            )
            .unwrap_err();

            assert_eq!(error.code(), "io_error");
            assert!(error.to_string().contains("could not read"));
            // No connection should have been made, so the accept thread
            // never returns on its own; drop it without joining.
            drop(handle);
        }

        #[test]
        fn try_clear_at_closes_the_recorded_surface_and_removes_the_state_file() {
            let (socket_path, handle) = fake_server(
                r#"{"id":"renderer-clear","ok":true,"result":{"surface_id":"33333333-3333-3333-3333-333333333333"}}"#,
            );
            let state_dir = tempfile::tempdir().unwrap();
            let state_file = state_dir.path().join("cmux-preview-surface.json");
            fs::write(
                &state_file,
                r#"{"surface_id":"33333333-3333-3333-3333-333333333333"}"#,
            )
            .unwrap();

            let outcome = try_clear_at(&socket_path, &state_file).unwrap();
            assert_eq!(outcome, CmuxClearOutcome::Cleared);
            assert!(!state_file.exists());

            let request = String::from_utf8(handle.join().unwrap()).unwrap();
            let request: serde_json::Value = serde_json::from_str(request.trim()).unwrap();
            assert_eq!(request["method"], "surface.close");
            assert_eq!(
                request["params"]["surface_id"],
                "33333333-3333-3333-3333-333333333333"
            );
        }

        #[test]
        fn try_clear_at_reports_no_surface_recorded_without_erroring() {
            let (socket_path, handle) = fake_server("irrelevant: no request is sent");
            let state_file = tempfile::tempdir()
                .unwrap()
                .path()
                .join("cmux-preview-surface.json");

            let outcome = try_clear_at(&socket_path, &state_file).unwrap();
            assert_eq!(outcome, CmuxClearOutcome::NoSurface);

            // A connection is opened (to check availability) but no
            // request is ever sent since there's no surface to close;
            // drop the server's own connection without requiring it to
            // have received anything.
            let _ = handle.join();
        }

        #[test]
        fn try_clear_at_falls_back_when_the_socket_is_unreachable() {
            let missing_socket = tempfile::tempdir().unwrap().path().join("no-such.sock");
            let state_file = tempfile::tempdir()
                .unwrap()
                .path()
                .join("cmux-preview-surface.json");

            let outcome = try_clear_at(&missing_socket, &state_file).unwrap();
            assert_eq!(outcome, CmuxClearOutcome::Unavailable);
        }
    }
}

#[cfg(not(unix))]
mod cmux_preview {
    use super::*;

    pub(super) fn try_show(_path: &Path) -> Result<Option<&'static str>, TerminalError> {
        Ok(None)
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum CmuxClearOutcome {
        Unavailable,
    }

    pub(super) fn try_clear() -> Result<CmuxClearOutcome, TerminalError> {
        Ok(CmuxClearOutcome::Unavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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

    #[test]
    fn composite_over_background_blends_partial_alpha_against_black() {
        assert_eq!(
            composite_over_background(Rgba([200, 100, 50, 255])),
            (200, 100, 50)
        );
        assert_eq!(
            composite_over_background(Rgba([200, 100, 50, 0])),
            (0, 0, 0)
        );
        // alpha = 128/255 ~= 0.502; background is black, so the result is
        // approximately channel * alpha.
        assert_eq!(
            composite_over_background(Rgba([200, 100, 50, 128])),
            (100, 50, 25)
        );
    }

    #[test]
    fn render_ansi_frame_emits_the_documented_half_block_escape_sequence() {
        let mut image = RgbaImage::new(1, 2);
        image.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        image.put_pixel(0, 1, Rgba([0, 255, 0, 255]));
        let (rendered, rows) = render_ansi_frame(&image);
        assert_eq!(rows, 1);
        assert_eq!(
            String::from_utf8(rendered).unwrap(),
            "\x1b[38;2;255;0;0m\x1b[48;2;0;255;0m\u{2580}\x1b[0m\n"
        );
    }

    #[test]
    fn render_ansi_frame_reuses_the_top_pixel_for_an_odd_final_row() {
        let mut image = RgbaImage::new(1, 1);
        image.put_pixel(0, 0, Rgba([10, 20, 30, 255]));
        let (rendered, rows) = render_ansi_frame(&image);
        assert_eq!(rows, 1);
        assert_eq!(
            String::from_utf8(rendered).unwrap(),
            "\x1b[38;2;10;20;30m\x1b[48;2;10;20;30m\u{2580}\x1b[0m\n"
        );
    }

    #[test]
    fn parse_ps_ppid_tty_handles_macos_and_linux_formatting() {
        assert_eq!(
            parse_ps_ppid_tty("  501 ttys008\n"),
            Some((501, "ttys008".to_string()))
        );
        assert_eq!(
            parse_ps_ppid_tty("1234 pts/3\n"),
            Some((1234, "pts/3".to_string()))
        );
        assert_eq!(parse_ps_ppid_tty("   42 ?\n"), Some((42, "?".to_string())));
        assert_eq!(parse_ps_ppid_tty(""), None);
        assert_eq!(parse_ps_ppid_tty("not-a-pid tty\n"), None);
    }

    #[test]
    fn is_real_tty_name_rejects_placeholder_values() {
        assert!(!is_real_tty_name("?"));
        assert!(!is_real_tty_name("??"));
        assert!(!is_real_tty_name(""));
        assert!(!is_real_tty_name("   "));
        assert!(is_real_tty_name("ttys008"));
        assert!(is_real_tty_name("pts/3"));
    }

    #[test]
    fn walk_ancestors_for_tty_finds_the_first_ancestor_with_a_real_tty() {
        // pid 300 (tty "?") -> pid 200 (tty "??") -> pid 100 (tty "ttys003")
        let responses: HashMap<u32, &str> =
            [(300, "200 ?\n"), (200, "100 ??\n"), (100, "1 ttys003\n")]
                .into_iter()
                .collect();
        let tty = walk_ancestors_for_tty(300, 32, |pid| responses.get(&pid).map(|s| s.to_string()));
        assert_eq!(tty, Some("ttys003".to_string()));
    }

    #[test]
    fn walk_ancestors_for_tty_gives_up_at_pid_1_with_no_real_tty() {
        let responses: HashMap<u32, &str> = [(50, "1 ?\n"), (1, "0 ?\n")].into_iter().collect();
        let tty = walk_ancestors_for_tty(50, 32, |pid| responses.get(&pid).map(|s| s.to_string()));
        assert_eq!(tty, None);
    }

    #[test]
    fn walk_ancestors_for_tty_stops_after_max_depth() {
        // ppid always climbs by one and never yields a real tty, pid 0, or
        // a self-loop, so only the max_depth cap can stop this.
        let tty = walk_ancestors_for_tty(1000, 5, |pid| Some(format!("{} ?\n", pid + 1)));
        assert_eq!(tty, None);
    }

    #[test]
    fn resolve_terminal_target_honors_an_explicit_tty_override() {
        let target = resolve_terminal_target(Some(Path::new("/tmp/fake-tty")));
        match target {
            TerminalResolution::Target(TerminalTarget::Device(path)) => {
                assert_eq!(path, PathBuf::from("/tmp/fake-tty"));
            }
            other => panic!("expected an explicit device target, got {other:?}"),
        }
    }

    #[test]
    fn detect_protocol_prefers_kitty_signals_in_priority_order() {
        let lookup = |pairs: &'static [(&'static str, &'static str)]| {
            let map: HashMap<&str, &str> = pairs.iter().copied().collect();
            move |name: &str| map.get(name).map(|value| value.to_string())
        };

        assert_eq!(
            detect_protocol_auto(&lookup(&[("KITTY_WINDOW_ID", "1")])),
            Protocol::Kitty {
                animation_capable: true
            }
        );
        assert_eq!(
            detect_protocol_auto(&lookup(&[("TERM_PROGRAM", "ghostty")])),
            Protocol::Kitty {
                animation_capable: false
            }
        );
        assert_eq!(
            detect_protocol_auto(&lookup(&[("CMUX_WORKSPACE_ID", "abc")])),
            Protocol::Kitty {
                animation_capable: false
            }
        );
        assert_eq!(
            detect_protocol_auto(&lookup(&[("TERM_PROGRAM", "WezTerm")])),
            Protocol::Kitty {
                animation_capable: true
            }
        );
        assert_eq!(
            detect_protocol_auto(&lookup(&[("TERM_PROGRAM", "iTerm.app")])),
            Protocol::Iterm2
        );
        assert_eq!(detect_protocol_auto(&lookup(&[])), Protocol::Ansi);
    }

    #[test]
    fn resolve_protocol_rejects_unknown_values() {
        let error = resolve_protocol("bogus").unwrap_err();
        assert_eq!(error.code(), "invalid_protocol");
    }

    #[test]
    fn resolve_protocol_honors_explicit_choices() {
        assert_eq!(resolve_protocol("iterm2").unwrap(), Protocol::Iterm2);
        assert_eq!(resolve_protocol("ansi").unwrap(), Protocol::Ansi);
        assert!(matches!(
            resolve_protocol("kitty").unwrap(),
            Protocol::Kitty { .. }
        ));
    }

    #[test]
    fn fit_dimensions_fits_within_the_available_grid_and_keeps_height_even() {
        let (width, height) = fit_dimensions(400, 200, 82, 25);
        assert!(width <= 80);
        assert!(height <= 48);
        assert_eq!(height % 2, 0);
    }

    #[test]
    fn fit_dimensions_never_returns_zero() {
        let (width, height) = fit_dimensions(1, 1, 3, 2);
        assert!(width >= 1);
        assert!(height >= 2);
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
