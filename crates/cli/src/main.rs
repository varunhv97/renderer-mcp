use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use renderer_daemon::RendererDaemon;
use renderer_schema::SceneV1;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

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
    /// Render a SceneV1 JSON document to PNG.
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
    /// Print the local runtime contract version.
    Status,
}

#[derive(Serialize)]
struct InspectResult {
    path: PathBuf,
    width: u32,
    height: u32,
    sha256: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Render { input, output } => {
            let source = fs::read_to_string(&input)
                .with_context(|| format!("could not read {}", input.display()))?;
            let scene: SceneV1 =
                serde_json::from_str(&source).context("scene is not valid JSON")?;
            let output = output.unwrap_or_else(|| PathBuf::from(".renderer/output/render.png"));
            let daemon = RendererDaemon::new()?;
            let rendered =
                if output.extension().and_then(|extension| extension.to_str()) == Some("gif") {
                    daemon.render_gif_inline(&scene, &output)?
                } else {
                    daemon.render_inline(&scene, &output)?
                };
            println!(
                "{}",
                serde_json::json!({
                    "path": output,
                    "mime_type": if rendered.frame_count > 1 { "image/gif" } else { "image/png" },
                    "width": rendered.width,
                    "height": rendered.height,
                    "frame_count": rendered.frame_count,
                    "sha256": rendered.sha256,
                    "warnings": rendered.warnings,
                })
            );
        }
        Command::Inspect { input } => {
            let (width, height) = image::image_dimensions(&input)
                .with_context(|| format!("could not inspect {}", input.display()))?;
            let sha256 = format!("{:x}", Sha256::digest(fs::read(&input)?));
            println!(
                "{}",
                serde_json::to_string_pretty(&InspectResult {
                    path: input,
                    width,
                    height,
                    sha256
                })?
            );
        }
        Command::Status => println!(
            "{}",
            serde_json::json!({ "api_version": "renderer.scene.v1", "transport": "local" })
        ),
    }
    Ok(())
}
