use anyhow::Result;
use clap::{Parser, Subcommand};
use renderer_daemon::{
    DaemonClient, DaemonError, DaemonRequest, DaemonResult, RenderResult, serve,
};
use renderer_schema::{ScenePatchV1, SceneV1};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
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
        /// Use the experimental analytic (signed-distance-field) anti-
        /// aliasing pipeline instead of the default MSAA+supersampling one,
        /// for side-by-side quality/performance comparison. A separate,
        /// additive pipeline -- see `renderer_core::GpuRenderer`'s
        /// `_analytic_aa` methods -- not used by any other command.
        #[arg(long)]
        experimental_analytic_aa: bool,
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
/// daemon layer. `show`'s own failure modes live in `renderer_terminal::
/// TerminalError` now -- see `error_code`.
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
}

impl CliError {
    fn code(&self) -> &'static str {
        match self {
            CliError::Read { .. } | CliError::Image { .. } => "io_error",
            CliError::InvalidJson { .. } => "invalid_json",
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
        Command::Render {
            input,
            output,
            experimental_analytic_aa,
        } => render_direct(input, output, experimental_analytic_aa),
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
/// of failures local to this crate; `TerminalError` covers `show`'s
/// failures (terminal detection, protocol encoding, cmux); anything else
/// (e.g. a `serde_json` serialization failure on the success path) falls
/// back to a generic code.
fn error_code(error: &anyhow::Error) -> String {
    if let Some(daemon_error) = error.downcast_ref::<DaemonError>() {
        daemon_error.code()
    } else if let Some(cli_error) = error.downcast_ref::<CliError>() {
        cli_error.code().to_string()
    } else if let Some(terminal_error) = error.downcast_ref::<renderer_terminal::TerminalError>() {
        terminal_error.code().to_string()
    } else {
        "internal_error".to_string()
    }
}

fn render_direct(
    input: PathBuf,
    output: Option<PathBuf>,
    experimental_analytic_aa: bool,
) -> Result<()> {
    let scene = read_json::<SceneV1>(&input)?;
    let output = output.unwrap_or_else(|| PathBuf::from(".renderer/output/render.png"));
    let asset_root = asset_root_for(&input);
    let is_gif = output.extension().and_then(|extension| extension.to_str()) == Some("gif");
    let rendered = if experimental_analytic_aa {
        // Bypasses `RendererDaemon` entirely: the analytic-AA pipeline is
        // additive to `renderer_core::GpuRenderer` and has no daemon-level
        // wrapper (and doesn't need one -- this flag exists purely for
        // local, one-shot quality/performance comparison against the
        // default path, not as a supported named-scene/daemon capability).
        let renderer = renderer_core::GpuRenderer::new()?;
        if is_gif {
            renderer.render_gif_with_asset_root_analytic_aa(&scene, &output, &asset_root)?
        } else {
            renderer.render_png_with_asset_root_analytic_aa(&scene, &output, &asset_root)?
        }
    } else {
        let daemon = renderer_daemon::RendererDaemon::new()?;
        if is_gif {
            daemon.render_gif_inline_with_asset_root(&scene, &output, &asset_root)?
        } else {
            daemon.render_inline_with_asset_root(&scene, &output, &asset_root)?
        }
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

/// `renderer show`: a thin wrapper around the shared `renderer_terminal`
/// crate. All terminal detection, protocol encoding, and cmux integration
/// lives there now (shared with the MCP server's `show_image` tool); this
/// module's only job is translating between the CLI's parsed arguments /
/// `renderer_terminal::ShowRequest`, and between
/// `renderer_terminal::ShowOutcome` / `Result<_, TerminalError>` and this
/// binary's existing structured JSON-on-stdout / `{"code","message"}`-on-
/// stderr conventions -- observable CLI behavior is unchanged.
mod show {
    use super::*;
    use renderer_terminal::{ShowOutcome, ShowRequest};

    /// Entry point for `renderer show`.
    pub(super) fn run(
        path: Option<PathBuf>,
        tty: Option<PathBuf>,
        protocol: String,
        clear: bool,
        loops: Option<u32>,
    ) -> Result<()> {
        let request = ShowRequest {
            path,
            tty,
            protocol: Some(protocol),
            clear,
            loops,
        };
        let outcome = renderer_terminal::run(request)?;
        report_outcome(outcome)
    }

    /// Renders a `ShowOutcome` exactly the way the pre-extraction `show`
    /// module used to build its own output inline: `"no_terminal"`/
    /// `"no_op"` are plain exit-0 messages on stdout (not JSON), a
    /// `"displayed"` outcome is `{"status","protocol","path"}`, and a
    /// `"cleared"` outcome includes `"protocol"` only when cmux produced
    /// it (a terminal-protocol clear never had a `protocol` field).
    fn report_outcome(outcome: ShowOutcome) -> Result<()> {
        match outcome.status {
            "no_terminal" | "no_op" => {
                if let Some(message) = outcome.message {
                    println!("{message}");
                }
                Ok(())
            }
            "displayed" => print_json(serde_json::json!({
                "status": "displayed",
                "protocol": outcome.protocol,
                "path": outcome.path,
            })),
            "cleared" => match outcome.protocol {
                Some(protocol) => {
                    print_json(serde_json::json!({ "status": "cleared", "protocol": protocol }))
                }
                None => print_json(serde_json::json!({ "status": "cleared" })),
            },
            other => unreachable!("unexpected renderer_terminal::ShowOutcome status: {other}"),
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
