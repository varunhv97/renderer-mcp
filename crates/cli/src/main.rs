use anyhow::Result;
use base64::Engine;
use clap::{Parser, Subcommand};
use image::AnimationDecoder;
use is_terminal::IsTerminal;
use renderer_daemon::{
    DaemonClient, DaemonError, DaemonRequest, DaemonResult, RenderResult, serve,
};
use renderer_schema::{ScenePatchV1, SceneV1};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Command as ProcessCommand,
    time::Duration,
};

#[derive(Parser)]
#[command(
    name = "renderer",
    about = "Local GPU visual runtime for coding agents"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Render a SceneV1 JSON document directly to PNG or GIF.
    Render {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Inspect image dimensions and SHA-256 without opening a browser.
    Inspect {
        #[arg(short, long)]
        input: PathBuf,
    },
    /// Start a foreground, loopback-only daemon.
    Daemon {
        #[command(subcommand)]
        command: DaemonCommand,
    },
    /// Work with a named scene held by a running local daemon.
    Scene {
        #[arg(long)]
        endpoint: SocketAddr,
        #[command(subcommand)]
        command: SceneCommand,
    },
    /// Print the local runtime contract version.
    Status,
    /// Display a local PNG or GIF directly inline in the user's terminal.
    Show {
        /// Path to a local PNG or GIF. Optional only when `--clear` is given.
        #[arg(required_unless_present = "clear")]
        path: Option<PathBuf>,
        /// Write escape sequences to this device file instead of discovering
        /// a terminal automatically (e.g. `/dev/ttys008`). Mainly useful for
        /// testing against a specific terminal session.
        #[arg(long)]
        tty: Option<PathBuf>,
        /// Which terminal graphics protocol to use.
        #[arg(long, default_value = "auto")]
        protocol: String,
        /// Send only a Kitty-protocol delete-all-images command and exit.
        #[arg(long)]
        clear: bool,
        /// Number of times to loop a simulated/native GIF animation (0 or
        /// omitted means loop forever, matching normal GIF playback).
        #[arg(long)]
        loops: Option<u32>,
    },
}

#[derive(Subcommand)]
enum DaemonCommand {
    Serve {
        #[arg(long, default_value = "127.0.0.1:9472")]
        endpoint: SocketAddr,
    },
}

#[derive(Subcommand)]
enum SceneCommand {
    Create {
        scene_id: String,
        #[arg(short, long)]
        input: PathBuf,
    },
    Get {
        scene_id: String,
    },
    Replace {
        scene_id: String,
        #[arg(short, long)]
        input: PathBuf,
        #[arg(long)]
        expected_revision: Option<u64>,
    },
    Patch {
        scene_id: String,
        #[arg(short, long)]
        input: PathBuf,
    },
    Render {
        scene_id: String,
        #[arg(short, long)]
        output: PathBuf,
    },
    Destroy {
        scene_id: String,
    },
    Health,
}

#[derive(Serialize)]
struct InspectResult {
    path: PathBuf,
    width: u32,
    height: u32,
    sha256: String,
}

/// A stable, machine-readable error origin local to the CLI crate: file I/O
/// and JSON parsing for `read_json`, plus `inspect`'s image decoding and
/// hashing. Anything that can instead be represented as a `DaemonError`
/// should be, so this only needs to cover errors that never touch the
/// daemon layer.
#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error("could not read {}: {source}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{} is not valid JSON: {source}", path.display())]
    InvalidJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
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
}

impl CliError {
    fn code(&self) -> &'static str {
        match self {
            CliError::Read { .. } | CliError::Image { .. } | CliError::TtyOpen { .. } => "io_error",
            CliError::InvalidJson { .. } => "invalid_json",
            CliError::UnsupportedImage { .. } => "unsupported_image",
            CliError::InvalidProtocol { .. } => "invalid_protocol",
        }
    }
}

/// One JSON object written to stderr for any command failure, per the UI/UX
/// specification: a stable `code` plus a human-readable `message` carrying
/// whatever path/scene ID/operation-index context the error had.
#[derive(Serialize)]
struct ErrorEnvelope {
    code: String,
    message: String,
}

fn main() {
    if let Err(error) = run() {
        report_error(&error);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Render { input, output } => render_direct(input, output),
        Command::Inspect { input } => inspect(input),
        Command::Daemon {
            command: DaemonCommand::Serve { endpoint },
        } => serve(endpoint).map_err(Into::into),
        Command::Scene { endpoint, command } => run_scene_command(endpoint, command),
        Command::Status => print_json(
            serde_json::json!({ "api_version": "renderer.scene.v1", "transport": "local" }),
        ),
        Command::Show {
            path,
            tty,
            protocol,
            clear,
            loops,
        } => show::run(path, tty, protocol, clear, loops),
    }
}

/// Prints the stable JSON error envelope required by the UI/UX
/// specification: `{"code": ..., "message": ...}` on stderr, nothing else.
/// `code` is recovered by downcasting to whichever concrete error type
/// produced this failure -- `DaemonError` for anything that round-tripped
/// through the daemon client (or the in-process daemon), `CliError` for
/// failures local to this crate, and `internal_error` for anything else.
fn report_error(error: &anyhow::Error) {
    let envelope = ErrorEnvelope {
        code: error_code(error),
        message: error.to_string(),
    };
    match serde_json::to_string(&envelope) {
        Ok(json) => eprintln!("{json}"),
        Err(_) => {
            eprintln!(r#"{{"code":"internal_error","message":"failed to format error"}}"#);
        }
    }
}

/// Recovers a stable code from an arbitrary top-level error by downcasting
/// to whichever concrete error type actually produced it. `DaemonError` is
/// checked first since it covers most command failures (anything that
/// touched `RendererDaemon`/`DaemonClient`); `CliError` covers the handful
/// of failures local to this crate; anything else (e.g. a `serde_json`
/// serialization failure on the success path) falls back to a generic code.
fn error_code(error: &anyhow::Error) -> String {
    if let Some(daemon_error) = error.downcast_ref::<DaemonError>() {
        daemon_error.code()
    } else if let Some(cli_error) = error.downcast_ref::<CliError>() {
        cli_error.code().to_string()
    } else {
        "internal_error".to_string()
    }
}

fn render_direct(input: PathBuf, output: Option<PathBuf>) -> Result<()> {
    let scene = read_json::<SceneV1>(&input)?;
    let output = output.unwrap_or_else(|| PathBuf::from(".renderer/output/render.png"));
    let daemon = renderer_daemon::RendererDaemon::new()?;
    let asset_root = asset_root_for(&input);
    let rendered = if output.extension().and_then(|extension| extension.to_str()) == Some("gif") {
        daemon.render_gif_inline_with_asset_root(&scene, &output, &asset_root)?
    } else {
        daemon.render_inline_with_asset_root(&scene, &output, &asset_root)?
    };
    print_json(render_metadata(output, rendered.into()))
}

fn inspect(input: PathBuf) -> Result<()> {
    let (width, height) = image::image_dimensions(&input).map_err(|source| CliError::Image {
        path: input.clone(),
        source,
    })?;
    let bytes = fs::read(&input).map_err(|source| CliError::Read {
        path: input.clone(),
        source,
    })?;
    let sha256 = format!("{:x}", Sha256::digest(bytes));
    println!(
        "{}",
        serde_json::to_string_pretty(&InspectResult {
            path: input,
            width,
            height,
            sha256,
        })?
    );
    Ok(())
}

fn run_scene_command(endpoint: SocketAddr, command: SceneCommand) -> Result<()> {
    let client = DaemonClient::new(endpoint)?;
    let result = match command {
        SceneCommand::Create { scene_id, input } => client.call(DaemonRequest::CreateScene {
            scene_id,
            scene: read_json(&input)?,
            asset_root: Some(asset_root_for(&input)),
        })?,
        SceneCommand::Get { scene_id } => client.call(DaemonRequest::GetScene { scene_id })?,
        SceneCommand::Replace {
            scene_id,
            input,
            expected_revision,
        } => client.call(DaemonRequest::ReplaceScene {
            scene_id,
            scene: read_json(&input)?,
            expected_revision,
            asset_root: Some(asset_root_for(&input)),
        })?,
        SceneCommand::Patch { scene_id, input } => client.call(DaemonRequest::PatchScene {
            scene_id,
            patch: read_json::<ScenePatchV1>(&input)?,
        })?,
        SceneCommand::Render { scene_id, output } => {
            let result = client.call(DaemonRequest::RenderScene {
                scene_id,
                output: output.clone(),
            })?;
            return match result {
                DaemonResult::Rendered { image, .. } => print_json(render_metadata(output, image)),
                _ => unreachable!("render requests return rendered results"),
            };
        }
        SceneCommand::Destroy { scene_id } => {
            client.call(DaemonRequest::DestroyScene { scene_id })?
        }
        SceneCommand::Health => client.call(DaemonRequest::Health)?,
    };
    print_daemon_result(result)
}

fn asset_root_for(input: &Path) -> PathBuf {
    input
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn read_json<T: serde::de::DeserializeOwned>(input: &PathBuf) -> Result<T, CliError> {
    let contents = fs::read_to_string(input).map_err(|source| CliError::Read {
        path: input.clone(),
        source,
    })?;
    serde_json::from_str(&contents).map_err(|source| CliError::InvalidJson {
        path: input.clone(),
        source,
    })
}

fn print_daemon_result(result: DaemonResult) -> Result<()> {
    match result {
        DaemonResult::Rendered { .. } => unreachable!("render output is handled by its command"),
        result => print_json(result),
    }
}

fn render_metadata(path: PathBuf, rendered: RenderResult) -> serde_json::Value {
    serde_json::json!({
        "path": path,
        "mime_type": if rendered.frame_count > 1 { "image/gif" } else { "image/png" },
        "width": rendered.width, "height": rendered.height, "frame_count": rendered.frame_count,
        "sha256": rendered.sha256, "warnings": rendered.warnings,
    })
}

fn print_json(value: impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string(&value)?);
    Ok(())
}

/// Inline terminal image/GIF preview (`renderer show`).
///
/// Kept as one module colocated with the rest of the CLI binary rather than
/// a separate crate: this is purely a terminal-display concern for the
/// `renderer` binary and is unrelated to the scene/rendering pipeline the
/// other workspace crates implement.
mod show {
    use super::*;
    use image::codecs::gif::GifDecoder;
    use image::imageops::FilterType;
    use image::{ImageFormat, Rgba, RgbaImage};
    use std::io::Cursor;

    /// Cap on how many ancestor processes to walk before giving up on
    /// finding a real controlling terminal -- avoids any pathological loop
    /// in a broken process tree.
    const MAX_ANCESTOR_DEPTH: u32 = 32;
    /// Kitty graphics protocol payload chunk size, in base64-encoded bytes.
    const KITTY_CHUNK_LIMIT: usize = 4096;
    /// Fixed background the ANSI half-block fallback composites partially
    /// transparent pixels against, since terminal cells have no alpha
    /// channel of their own. Chosen to match typical dark terminal
    /// backgrounds -- there is no universally "correct" choice here, so a
    /// transparent pixel simply renders as this color rather than as the
    /// viewer's real terminal background.
    const ANSI_BACKGROUND: (u8, u8, u8) = (0, 0, 0);
    /// Fallback terminal size used when writing to a discovered ancestor tty
    /// device, where querying the real size isn't practical.
    const DEFAULT_TERMINAL_COLUMNS: u16 = 80;
    const DEFAULT_TERMINAL_ROWS: u16 = 24;
    /// Printed (to stdout, exit 0) by `show --clear` when no terminal
    /// target can be found -- see `no_terminal_message` for the equivalent
    /// used by a plain `show <path>`.
    const NO_TERMINAL_CLEAR_MESSAGE: &str = "no interactive terminal detected; nothing to clear";

    /// The message `show <path>` prints (to stdout, exit 0 -- not a
    /// failure) when neither stdout nor any ancestor process has a real
    /// controlling terminal to write escape sequences to.
    fn no_terminal_message(path: &Path) -> String {
        format!(
            "no interactive terminal detected; image saved at {}, open it manually",
            path.display()
        )
    }

    /// Entry point for `renderer show`.
    pub(super) fn run(
        path: Option<PathBuf>,
        tty: Option<PathBuf>,
        protocol: String,
        clear: bool,
        loops: Option<u32>,
    ) -> Result<()> {
        if clear {
            return run_clear(tty.as_deref());
        }
        // clap enforces `required_unless_present = "clear"` on `path`, so
        // this is always populated once we get here.
        let path = path.expect("clap requires `path` unless --clear is given");

        let target = match resolve_terminal_target(tty.as_deref()) {
            TerminalResolution::NoTerminal => {
                println!("{}", no_terminal_message(&path));
                return Ok(());
            }
            TerminalResolution::Target(target) => target,
        };

        let bytes = fs::read(&path).map_err(|source| CliError::Read {
            path: path.clone(),
            source,
        })?;
        let format = image::guess_format(&bytes).ok();
        let protocol_choice = resolve_protocol(&protocol)?;
        let mut sink = open_sink(&target)?;

        let protocol_name = match format {
            Some(ImageFormat::Png) => {
                show_png(&mut *sink, &path, &bytes, protocol_choice, &target)?
            }
            Some(ImageFormat::Gif) => {
                show_gif(&mut *sink, &path, &bytes, protocol_choice, &target, loops)?
            }
            _ => return Err(CliError::UnsupportedImage { path }.into()),
        };

        print_json(serde_json::json!({
            "status": "displayed",
            "protocol": protocol_name,
            "path": path,
        }))
    }

    /// `renderer show --clear`: send only a Kitty-protocol delete-all-images
    /// command. A no-op (but still exit-0) message when no terminal target
    /// can be found, matching `run`'s treatment of that case.
    fn run_clear(tty: Option<&Path>) -> Result<()> {
        match resolve_terminal_target(tty) {
            TerminalResolution::NoTerminal => {
                println!("{NO_TERMINAL_CLEAR_MESSAGE}");
                Ok(())
            }
            TerminalResolution::Target(target) => {
                let mut sink = open_sink(&target)?;
                sink.write_all(&kitty_escape("a=d,d=A"))?;
                sink.flush()?;
                print_json(serde_json::json!({ "status": "cleared" }))
            }
        }
    }

    fn show_png(
        sink: &mut dyn Write,
        path: &Path,
        bytes: &[u8],
        protocol: Protocol,
        target: &TerminalTarget,
    ) -> Result<&'static str> {
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
                    .map_err(|source| CliError::Image {
                        path: path.to_path_buf(),
                        source,
                    })?
                    .to_rgba8();
                let (columns, rows) = ansi_target_size(target);
                let (width, height) =
                    fit_dimensions(decoded.width(), decoded.height(), columns, rows);
                let resized =
                    image::imageops::resize(&decoded, width, height, FilterType::Triangle);
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
    ) -> Result<&'static str> {
        match protocol {
            Protocol::Iterm2 => {
                // iTerm2 decodes and loops the animation itself; the raw
                // GIF bytes go through unchanged, same mechanism as a
                // static image.
                sink.write_all(&iterm2_transmit(bytes))?;
                sink.flush()?;
                Ok("iterm2")
            }
            Protocol::Kitty { animation_capable } => {
                let frames = decode_gif_frames(path)?;
                if frames.is_empty() {
                    return Err(CliError::UnsupportedImage {
                        path: path.to_path_buf(),
                    }
                    .into());
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
                    return Err(CliError::UnsupportedImage {
                        path: path.to_path_buf(),
                    }
                    .into());
                }
                let (columns, rows) = ansi_target_size(target);
                let mut rendered_frames = Vec::with_capacity(frames.len());
                let mut row_count = 0usize;
                for (image, delay) in &frames {
                    let (width, height) =
                        fit_dimensions(image.width(), image.height(), columns, rows);
                    let resized =
                        image::imageops::resize(image, width, height, FilterType::Triangle);
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

    fn resolve_protocol(value: &str) -> Result<Protocol, CliError> {
        match value {
            "auto" => Ok(detect_protocol_auto(&env_lookup)),
            "kitty" => Ok(Protocol::Kitty {
                animation_capable: kitty_animation_capable(&env_lookup),
            }),
            "iterm2" => Ok(Protocol::Iterm2),
            "ansi" => Ok(Protocol::Ansi),
            other => Err(CliError::InvalidProtocol {
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

    fn open_sink(target: &TerminalTarget) -> Result<Box<dyn Write>, CliError> {
        match target {
            TerminalTarget::Stdout => Ok(Box::new(std::io::stdout())),
            TerminalTarget::Device(path) => {
                let file = fs::OpenOptions::new()
                    .write(true)
                    .open(path)
                    .map_err(|source| CliError::TtyOpen {
                        path: path.clone(),
                        source,
                    })?;
                Ok(Box::new(file))
            }
        }
    }

    /// True when a `ps` `tty` column value names a real terminal device
    /// rather than "no controlling terminal" -- BSD `ps` (macOS) prints
    /// `??`, GNU `ps` (Linux) prints `?`, and both may print nothing at all.
    fn is_real_tty_name(tty: &str) -> bool {
        let trimmed = tty.trim();
        !(trimmed.is_empty() || trimmed == "?" || trimmed == "??" || trimmed == "-")
    }

    /// Parses one line of `ps -o ppid=,tty= -p <pid>` output. Defensive
    /// about whitespace since BSD and GNU `ps` pad columns differently.
    fn parse_ps_ppid_tty(output: &str) -> Option<(u32, String)> {
        let line = output.lines().find(|line| !line.trim().is_empty())?;
        let mut fields = line.split_whitespace();
        let ppid = fields.next()?.parse::<u32>().ok()?;
        let tty = fields.next()?.to_string();
        Some((ppid, tty))
    }

    /// Walks the process ancestry starting at `start_pid`, calling `lookup`
    /// (expected to behave like `ps -o ppid=,tty= -p <pid>`) for each
    /// ancestor until one has a real controlling terminal, `ppid` reaches 0
    /// or itself, `lookup`/parsing fails, or `max_depth` is exceeded.
    /// `lookup` is injected so this is testable against canned `ps` output
    /// instead of a live process tree.
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
    /// `m=1` (more chunks follow) / `m=0` (final chunk) convention. Payloads
    /// at or under the limit are sent as one unchunked command with no `m`
    /// key at all.
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

    /// Maps `--loops` onto the Kitty animation control protocol's `v=` key,
    /// per https://sw.kovidgoyal.net/kitty/graphics-protocol/#animation:
    /// `v=1` loops infinitely and any other positive number loops
    /// `number - 1` times. `None` or `Some(0)` both mean "loop forever",
    /// matching normal GIF playback (a GIF's Netscape loop extension also
    /// uses 0 to mean infinite).
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
    /// root frame plus every additional frame with its gap, then hand
    /// playback off to the terminal itself with one `a=a` control command.
    /// The process can exit immediately afterward.
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

    /// Simulated animation for Kitty-protocol terminals without the
    /// animation extension (Ghostty, cmux): loop transmitting each frame as
    /// a plain static image reusing image id 1 (so each transmission
    /// replaces the last), sleeping for its delay in between, until
    /// `loops` is exhausted or the process is killed.
    fn show_kitty_simulated_animation(
        sink: &mut dyn Write,
        frames: &[(Vec<u8>, u32)],
        loops: Option<u32>,
    ) -> std::io::Result<()> {
        let mut played = 0u32;
        loop {
            for (png, delay) in frames {
                sink.write_all(&kitty_transmit("a=T,f=100,i=1,q=2", png))?;
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

    /// OSC 1337 inline image (iTerm2's own protocol). Used unchanged for
    /// both static images and animated GIFs -- iTerm2 decodes and loops a
    /// GIF's animation itself, so the raw file bytes are enough either way.
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
    /// approximated by alpha-blending against a fixed background color
    /// rather than the viewer's real (unknowable) terminal background.
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

    /// Renders one frame as upper-half-block rows: each terminal row covers
    /// two source pixel rows (foreground = top pixel, background = bottom
    /// pixel), reset at the end of each row. An odd final source row reuses
    /// the top pixel as its own bottom half. Returns the encoded bytes and
    /// the number of terminal rows they occupy (for cursor-repositioning
    /// during animation).
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
                    format!("\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m\u{2580}")
                        .as_bytes(),
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
    fn fit_dimensions(
        source_width: u32,
        source_height: u32,
        columns: u16,
        rows: u16,
    ) -> (u32, u32) {
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

    /// Simulated animation for the ANSI half-block fallback: redraw each
    /// frame in place by moving the cursor back up over the previous
    /// frame's rows before drawing the next one, sleeping for its delay in
    /// between, until `loops` is exhausted or the process is killed.
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

    fn decode_gif_frames(path: &Path) -> Result<Vec<(RgbaImage, u32)>, CliError> {
        let file = fs::File::open(path).map_err(|source| CliError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let decoder =
            GifDecoder::new(std::io::BufReader::new(file)).map_err(|source| CliError::Image {
                path: path.to_path_buf(),
                source,
            })?;
        let frames = decoder
            .into_frames()
            .collect_frames()
            .map_err(|source| CliError::Image {
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

    fn encode_frame_png(path: &Path, image: &RgbaImage) -> Result<Vec<u8>, CliError> {
        let mut bytes = Vec::new();
        let mut cursor = Cursor::new(&mut bytes);
        image::DynamicImage::ImageRgba8(image.clone())
            .write_to(&mut cursor, ImageFormat::Png)
            .map_err(|source| CliError::Image {
                path: path.to_path_buf(),
                source,
            })?;
        Ok(bytes)
    }

    fn encode_frames_as_png(
        path: &Path,
        frames: &[(RgbaImage, u32)],
    ) -> Result<Vec<(Vec<u8>, u32)>, CliError> {
        frames
            .iter()
            .map(|(image, delay)| Ok((encode_frame_png(path, image)?, *delay)))
            .collect()
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
            // 3072 raw bytes base64-encode to exactly 4096 characters, with
            // no padding.
            let payload = vec![1u8; 3072];
            let escape = kitty_transmit("a=T,f=100,i=1,q=2", &payload);
            let text = String::from_utf8(escape).unwrap();
            assert_eq!(text.matches("\x1b_G").count(), 1);
        }

        #[test]
        fn kitty_transmit_boundary_one_byte_over_the_chunk_limit_splits_into_two() {
            // 3075 raw bytes base64-encode to 4100 characters: one full
            // chunk plus four more.
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
            // alpha = 128/255 ~= 0.502; background is black, so the result
            // is approximately channel * alpha.
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
            let tty =
                walk_ancestors_for_tty(300, 32, |pid| responses.get(&pid).map(|s| s.to_string()));
            assert_eq!(tty, Some("ttys003".to_string()));
        }

        #[test]
        fn walk_ancestors_for_tty_gives_up_at_pid_1_with_no_real_tty() {
            let responses: HashMap<u32, &str> = [(50, "1 ?\n"), (1, "0 ?\n")].into_iter().collect();
            let tty =
                walk_ancestors_for_tty(50, 32, |pid| responses.get(&pid).map(|s| s.to_string()));
            assert_eq!(tty, None);
        }

        #[test]
        fn walk_ancestors_for_tty_stops_after_max_depth() {
            // ppid always climbs by one and never yields a real tty, pid 0,
            // or a self-loop, so only the max_depth cap can stop this.
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_current_directory_for_bare_scene_filenames() {
        assert_eq!(asset_root_for(Path::new("scene.json")), PathBuf::from("."));
        assert_eq!(
            asset_root_for(Path::new("fixtures/scene.json")),
            PathBuf::from("fixtures")
        );
    }

    #[test]
    fn render_metadata_classifies_mime_type_by_frame_count() {
        let single_frame = render_metadata(
            PathBuf::from("out.png"),
            RenderResult {
                width: 4,
                height: 4,
                sha256: "abc".into(),
                frame_count: 1,
                warnings: vec![],
            },
        );
        assert_eq!(single_frame["mime_type"], "image/png");
        assert_eq!(single_frame["path"], "out.png");

        let animated = render_metadata(
            PathBuf::from("out.gif"),
            RenderResult {
                width: 4,
                height: 4,
                sha256: "abc".into(),
                frame_count: 3,
                warnings: vec!["slow".into()],
            },
        );
        assert_eq!(animated["mime_type"], "image/gif");
        assert_eq!(animated["frame_count"], 3);
    }

    #[test]
    fn read_json_reports_missing_files_and_invalid_json() {
        let missing = read_json::<SceneV1>(&PathBuf::from("/no/such/scene.json"));
        let error = missing.expect_err("missing file must fail");
        assert_eq!(error.code(), "io_error");
        assert!(error.to_string().contains("could not read"));

        let directory = tempfile::tempdir().unwrap();
        let bad = directory.path().join("bad.json");
        fs::write(&bad, "not json").unwrap();
        let invalid = read_json::<SceneV1>(&bad);
        let error = invalid.expect_err("invalid json must fail");
        assert_eq!(error.code(), "invalid_json");
        assert!(error.to_string().contains("is not valid JSON"));
    }

    #[test]
    fn cli_error_image_variant_reports_io_error() {
        let error = CliError::Image {
            path: PathBuf::from("missing.png"),
            source: image::ImageError::IoError(std::io::Error::from(std::io::ErrorKind::NotFound)),
        };
        assert_eq!(error.code(), "io_error");
        assert!(error.to_string().contains("could not inspect"));
    }

    #[test]
    fn error_code_downcasts_daemon_and_cli_errors_with_internal_fallback() {
        let daemon_error: anyhow::Error = DaemonError::SceneNotFound("missing-scene".into()).into();
        assert_eq!(error_code(&daemon_error), "not_found");

        let cli_error: anyhow::Error = CliError::Read {
            path: PathBuf::from("scene.json"),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        }
        .into();
        assert_eq!(error_code(&cli_error), "io_error");

        let other = anyhow::anyhow!("some other failure");
        assert_eq!(error_code(&other), "internal_error");
    }

    #[test]
    fn report_error_prints_a_single_json_object() {
        // report_error itself only writes to stderr, so this exercises the
        // envelope construction and serialization path it depends on
        // directly rather than capturing process stderr.
        let daemon_error: anyhow::Error = DaemonError::SceneNotFound("missing-scene".into()).into();
        let envelope = ErrorEnvelope {
            code: error_code(&daemon_error),
            message: daemon_error.to_string(),
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["code"], "not_found");
        assert!(
            parsed["message"]
                .as_str()
                .unwrap()
                .contains("missing-scene")
        );
    }

    #[test]
    fn print_daemon_result_handles_every_non_render_variant() {
        assert!(print_daemon_result(DaemonResult::Health).is_ok());
        assert!(print_daemon_result(DaemonResult::Destroyed).is_ok());
        assert!(print_daemon_result(DaemonResult::Revision { revision: 7 }).is_ok());
    }

    #[test]
    fn print_json_serializes_and_prints() {
        assert!(print_json(serde_json::json!({ "ok": true })).is_ok());
    }
}
