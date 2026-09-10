use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use renderer_daemon::{DaemonClient, DaemonRequest, DaemonResult, RenderResult, serve};
use renderer_schema::{ScenePatchV1, SceneV1};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{fs, net::SocketAddr, path::PathBuf};

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

fn main() -> Result<()> {
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
    }
}

fn render_direct(input: PathBuf, output: Option<PathBuf>) -> Result<()> {
    let scene = read_json::<SceneV1>(&input)?;
    let output = output.unwrap_or_else(|| PathBuf::from(".renderer/output/render.png"));
    let daemon = renderer_daemon::RendererDaemon::new()?;
    let rendered = if output.extension().and_then(|extension| extension.to_str()) == Some("gif") {
        daemon.render_gif_inline(&scene, &output)?
    } else {
        daemon.render_inline(&scene, &output)?
    };
    print_json(render_metadata(output, rendered.into()))
}

fn inspect(input: PathBuf) -> Result<()> {
    let (width, height) = image::image_dimensions(&input)
        .with_context(|| format!("could not inspect {}", input.display()))?;
    let sha256 = format!("{:x}", Sha256::digest(fs::read(&input)?));
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

fn read_json<T: serde::de::DeserializeOwned>(input: &PathBuf) -> Result<T> {
    serde_json::from_str(
        &fs::read_to_string(input)
            .with_context(|| format!("could not read {}", input.display()))?,
    )
    .context("scene is not valid JSON")
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
