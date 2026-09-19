use crate::inspect::{inspect_image, mime_type_for_path};
use crate::output::{output_path, validate_output_path};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use renderer_daemon::{
    DaemonClient, DaemonRequest, DaemonResult, RenderResult, RendererDaemon, spawn_embedded,
};
use renderer_schema::{ScenePatchV1, SceneV1};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Mutex;

pub(crate) fn call_tool(
    request: &serde_json::Value,
    daemon: &mut Option<RendererDaemon>,
) -> Result<serde_json::Value, String> {
    let params = request.get("params").ok_or("tools/call needs params")?;
    let name = params
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or("tools/call needs a name")?;
    let arguments = params.get("arguments").cloned().unwrap_or_default();
    if name == "show_image" {
        return call_show_image_tool(&arguments);
    }
    if name != "render_scene" {
        return call_named_scene_tool(name, &arguments);
    }
    let scene: SceneV1 =
        serde_json::from_value(arguments.get("scene").cloned().ok_or("scene is required")?)
            .map_err(|error| format!("invalid SceneV1: {error}"))?;
    let output = arguments
        .get("output_path")
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".renderer/output/mcp-render.png"));
    validate_output_path(&output, "png")?;
    if daemon.is_none() {
        *daemon = Some(RendererDaemon::new().map_err(|error| error.to_string())?);
    }
    let rendered = daemon
        .as_ref()
        .expect("initialized above")
        .render_inline(&scene, &output)
        .map_err(|error| error.to_string())?;
    inline_render_response(output, rendered.into())
}

/// The address of the daemon backing the named-scene tools, started on demand.
///
/// `RENDERER_DAEMON_ENDPOINT`, when set, names a daemon shared with the CLI and
/// other clients; it's used as-is if it answers. Otherwise -- variable unset,
/// or set but nothing is listening yet -- this process starts its own daemon
/// on a background thread: on the configured address if there is one (so a
/// fixed port keeps working), else on a free loopback port. That daemon lives
/// only as long as this process, so its scenes go away with the MCP session.
fn resolve_daemon_endpoint() -> Result<SocketAddr, String> {
    static EMBEDDED: Mutex<Option<SocketAddr>> = Mutex::new(None);

    let configured = match std::env::var("RENDERER_DAEMON_ENDPOINT") {
        Ok(value) => Some(
            value
                .parse::<SocketAddr>()
                .map_err(|_| "RENDERER_DAEMON_ENDPOINT must be a socket address")?,
        ),
        Err(_) => None,
    };
    if let Some(endpoint) = configured
        && DaemonClient::new(endpoint)
            .and_then(|client| client.call(DaemonRequest::Health))
            .is_ok()
    {
        return Ok(endpoint);
    }
    let mut embedded = EMBEDDED.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(endpoint) = *embedded {
        return Ok(endpoint);
    }
    let bind = configured.unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 0)));
    let endpoint = spawn_embedded(bind).map_err(|error| {
        format!("could not start a built-in renderer daemon on {bind}: {error}")
    })?;
    *embedded = Some(endpoint);
    Ok(endpoint)
}

fn call_named_scene_tool(
    name: &str,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    if name == "inspect_image" {
        return inspect_image(arguments);
    }
    let endpoint = resolve_daemon_endpoint()?;
    let client = DaemonClient::new(endpoint).map_err(|error| error.to_string())?;
    let scene_id = || {
        arguments
            .get("scene_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or("scene_id is required")
    };
    let result = match name {
        "create_scene" => client.call(DaemonRequest::CreateScene {
            scene_id: scene_id()?,
            scene: serde_json::from_value(
                arguments.get("scene").cloned().ok_or("scene is required")?,
            )
            .map_err(|error| format!("invalid SceneV1: {error}"))?,
            asset_root: arguments
                .get("asset_root")
                .and_then(serde_json::Value::as_str)
                .map(PathBuf::from),
        }),
        "get_scene" => client.call(DaemonRequest::GetScene {
            scene_id: scene_id()?,
        }),
        "replace_scene" => client.call(DaemonRequest::ReplaceScene {
            scene_id: scene_id()?,
            scene: serde_json::from_value(
                arguments.get("scene").cloned().ok_or("scene is required")?,
            )
            .map_err(|error| format!("invalid SceneV1: {error}"))?,
            expected_revision: expected_revision(arguments)?,
            asset_root: arguments
                .get("asset_root")
                .and_then(serde_json::Value::as_str)
                .map(PathBuf::from),
        }),
        "patch_scene" => client.call(DaemonRequest::PatchScene {
            scene_id: scene_id()?,
            patch: serde_json::from_value::<ScenePatchV1>(
                arguments.get("patch").cloned().ok_or("patch is required")?,
            )
            .map_err(|error| format!("invalid ScenePatchV1: {error}"))?,
        }),
        "render_named_scene" => {
            let output = output_path(arguments, "png")?;
            client.call(DaemonRequest::RenderScene {
                scene_id: scene_id()?,
                output,
            })
        }
        "export_named_gif" => {
            let output = output_path(arguments, "gif")?;
            client.call(DaemonRequest::RenderGifScene {
                scene_id: scene_id()?,
                output,
            })
        }
        "destroy_scene" => client.call(DaemonRequest::DestroyScene {
            scene_id: scene_id()?,
        }),
        _ => return Err("unknown tool".into()),
    }
    .map_err(|error| error.to_string())?;
    match result {
        DaemonResult::Rendered { output, image } => inline_render_response(output, image),
        result => Ok(
            serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string(&result).map_err(|error| error.to_string())? }] }),
        ),
    }
}

/// Handler for `show_image`, calling straight into the shared
/// `renderer_terminal` crate -- the same `renderer_terminal::run` the CLI's
/// `renderer show` subcommand calls, so the two surfaces stay in lockstep.
/// Unlike the CLI's structured `{"code","message"}` error convention, this
/// crate's `tools/call` handlers just return `Result<_, String>` (see
/// `call_named_scene_tool` above for the existing pattern this follows), so
/// a `TerminalError` is flattened to its `Display` text.
fn call_show_image_tool(arguments: &serde_json::Value) -> Result<serde_json::Value, String> {
    let clear = arguments
        .get("clear")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let path = arguments
        .get("path")
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from);
    if path.is_none() && !clear {
        return Err("path is required unless clear is true".into());
    }
    let protocol = arguments
        .get("protocol")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let tty = arguments
        .get("tty")
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from);
    let loops = match arguments.get("loops") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => Some(
            value
                .as_u64()
                .and_then(|count| u32::try_from(count).ok())
                .ok_or("loops must be a non-negative integer that fits in 32 bits")?,
        ),
    };

    let outcome = renderer_terminal::run(renderer_terminal::ShowRequest {
        path,
        tty,
        protocol,
        clear,
        loops,
    })
    .map_err(|error| error.to_string())?;

    let mut summary = serde_json::json!({ "status": outcome.status });
    if let Some(protocol) = outcome.protocol {
        summary["protocol"] = serde_json::Value::String(protocol);
    }
    if let Some(path) = outcome.path {
        summary["path"] = serde_json::Value::String(path.to_string_lossy().into_owned());
    }
    if let Some(message) = outcome.message {
        summary["message"] = serde_json::Value::String(message);
    }
    Ok(serde_json::json!({ "content": [{ "type": "text", "text": summary.to_string() }] }))
}

pub(crate) fn expected_revision(arguments: &serde_json::Value) -> Result<Option<u64>, String> {
    let Some(value) = arguments.get("expected_revision") else {
        return Ok(None);
    };
    match value.as_u64().filter(|revision| *revision > 0) {
        Some(revision) => Ok(Some(revision)),
        None => Err("expected_revision must be a positive unsigned 64-bit integer".into()),
    }
}

fn inline_render_response(
    output: PathBuf,
    rendered: RenderResult,
) -> Result<serde_json::Value, String> {
    let bytes = std::fs::read(&output)
        .map_err(|error| format!("could not read rendered output: {error}"))?;
    let mime_type = mime_type_for_path(&output)?;
    let metadata = serde_json::json!({
        "path": output, "mime_type": mime_type, "width": rendered.width, "height": rendered.height,
        "frame_count": rendered.frame_count, "sha256": rendered.sha256, "warnings": rendered.warnings,
    });
    Ok(serde_json::json!({ "content": [
        { "type": "image", "data": STANDARD.encode(bytes), "mimeType": mime_type },
        { "type": "text", "text": metadata.to_string() }
    ] }))
}

#[cfg(test)]
mod tests {

    use crate::handlers::call_tool;

    #[test]
    fn renders_inline_content() {
        let directory = tempfile::tempdir().unwrap();
        let mut daemon = None;
        let _ = call_tool(&serde_json::json!({
            "params": { "name": "render_scene", "arguments": {
                "output_path": directory.path().join("render.png"),
                "scene": { "version": "renderer.scene.v1", "canvas": { "width": 8, "height": 8, "background": [0.0, 0.0, 0.0, 1.0] }, "nodes": [] }
            }}
        }), &mut daemon).ok();
    }
}
