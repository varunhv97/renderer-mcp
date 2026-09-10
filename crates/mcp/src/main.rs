//! MCP stdio server backed by one persistent local GPU daemon.

use base64::{Engine, engine::general_purpose::STANDARD};
use renderer_daemon::{DaemonClient, DaemonRequest, RendererDaemon};
use renderer_schema::{ScenePatchV1, SceneV1};
use std::{
    io::{self, BufRead, Write},
    net::SocketAddr,
    path::PathBuf,
};

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut daemon: Option<RendererDaemon> = None;
    for line in stdin.lock().lines().map_while(Result::ok) {
        let request: serde_json::Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let Some(id) = request.get("id").cloned() else {
            continue;
        };
        let envelope = match respond(&request, &mut daemon) {
            Ok(result) => serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err(message) => {
                serde_json::json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32000, "message": message } })
            }
        };
        let _ = writeln!(stdout, "{envelope}");
        let _ = stdout.flush();
    }
}

fn respond(
    request: &serde_json::Value,
    daemon: &mut Option<RendererDaemon>,
) -> Result<serde_json::Value, String> {
    match request.get("method").and_then(serde_json::Value::as_str) {
        Some("initialize") => Ok(serde_json::json!({
            "protocolVersion": "2025-06-18",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "renderer-mcp", "version": env!("CARGO_PKG_VERSION") }
        })),
        Some("tools/list") => Ok(
            serde_json::json!({ "tools": [render_tool(), named_scene_tool("create_scene"), named_scene_tool("get_scene"), named_scene_tool("patch_scene"), named_scene_tool("render_named_scene"), named_scene_tool("destroy_scene")] }),
        ),
        Some("tools/call") => call_tool(request, daemon),
        _ => Err("method not found".into()),
    }
}

fn named_scene_tool(name: &str) -> serde_json::Value {
    let (description, required, properties) = match name {
        "create_scene" => (
            "Create a named scene in RENDERER_DAEMON_ENDPOINT.",
            serde_json::json!(["scene_id", "scene"]),
            serde_json::json!({ "scene_id": { "type": "string" }, "scene": { "type": "object" } }),
        ),
        "get_scene" => (
            "Get a named scene and its revision.",
            serde_json::json!(["scene_id"]),
            serde_json::json!({ "scene_id": { "type": "string" } }),
        ),
        "patch_scene" => (
            "Atomically apply typed operations to a named scene.",
            serde_json::json!(["scene_id", "patch"]),
            serde_json::json!({ "scene_id": { "type": "string" }, "patch": { "type": "object" } }),
        ),
        "render_named_scene" => (
            "Render a named scene to a local PNG path.",
            serde_json::json!(["scene_id", "output_path"]),
            serde_json::json!({ "scene_id": { "type": "string" }, "output_path": { "type": "string" } }),
        ),
        _ => (
            "Destroy a named scene.",
            serde_json::json!(["scene_id"]),
            serde_json::json!({ "scene_id": { "type": "string" } }),
        ),
    };
    serde_json::json!({ "name": name, "description": description, "inputSchema": { "type": "object", "required": required, "properties": properties } })
}

fn render_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "render_scene",
        "description": "Render a renderer.scene.v1 JSON scene locally and return a PNG image plus metadata.",
        "inputSchema": {
            "type": "object",
            "required": ["scene"],
            "properties": {
                "scene": { "type": "object", "description": "A renderer.scene.v1 document." },
                "output_path": { "type": "string", "description": "Optional local PNG destination." }
            }
        }
    })
}

fn call_tool(
    request: &serde_json::Value,
    daemon: &mut Option<RendererDaemon>,
) -> Result<serde_json::Value, String> {
    let params = request.get("params").ok_or("tools/call needs params")?;
    let name = params
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or("tools/call needs a name")?;
    let arguments = params.get("arguments").cloned().unwrap_or_default();
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
    if daemon.is_none() {
        *daemon = Some(RendererDaemon::new().map_err(|error| error.to_string())?);
    }
    let rendered = daemon
        .as_ref()
        .expect("initialized above")
        .render_inline(&scene, &output)
        .map_err(|error| error.to_string())?;
    let png = std::fs::read(&output)
        .map_err(|error| format!("could not read rendered output: {error}"))?;
    let metadata = serde_json::json!({
        "path": output,
        "mime_type": "image/png",
        "width": rendered.width,
        "height": rendered.height,
        "sha256": rendered.sha256,
        "warnings": rendered.warnings,
    });
    Ok(serde_json::json!({
        "content": [
            { "type": "image", "data": STANDARD.encode(png), "mimeType": "image/png" },
            { "type": "text", "text": metadata.to_string() }
        ]
    }))
}

fn call_named_scene_tool(
    name: &str,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let endpoint: SocketAddr = std::env::var("RENDERER_DAEMON_ENDPOINT")
        .map_err(|_| "RENDERER_DAEMON_ENDPOINT is required for named-scene tools")?
        .parse()
        .map_err(|_| "RENDERER_DAEMON_ENDPOINT must be a socket address")?;
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
        }),
        "get_scene" => client.call(DaemonRequest::GetScene {
            scene_id: scene_id()?,
        }),
        "patch_scene" => client.call(DaemonRequest::PatchScene {
            scene_id: scene_id()?,
            patch: serde_json::from_value::<ScenePatchV1>(
                arguments.get("patch").cloned().ok_or("patch is required")?,
            )
            .map_err(|error| format!("invalid ScenePatchV1: {error}"))?,
        }),
        "render_named_scene" => client.call(DaemonRequest::RenderScene {
            scene_id: scene_id()?,
            output: PathBuf::from(
                arguments
                    .get("output_path")
                    .and_then(serde_json::Value::as_str)
                    .ok_or("output_path is required")?,
            ),
        }),
        "destroy_scene" => client.call(DaemonRequest::DestroyScene {
            scene_id: scene_id()?,
        }),
        _ => return Err("unknown tool".into()),
    }
    .map_err(|error| error.to_string())?;
    Ok(
        serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string(&result).map_err(|error| error.to_string())? }] }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describes_the_mcp_server_and_rejects_unknown_requests() {
        let mut daemon = None;
        let initialize =
            respond(&serde_json::json!({ "method": "initialize" }), &mut daemon).unwrap();
        assert_eq!(initialize["serverInfo"]["name"], "renderer-mcp");
        let tools = respond(&serde_json::json!({ "method": "tools/list" }), &mut daemon).unwrap();
        assert_eq!(tools["tools"][0]["name"], "render_scene");
        assert_eq!(
            respond(&serde_json::json!({ "method": "missing" }), &mut daemon),
            Err("method not found".into())
        );
        assert!(call_tool(&serde_json::json!({}), &mut daemon).is_err());
        assert!(
            call_tool(
                &serde_json::json!({ "params": { "name": "unknown" } }),
                &mut daemon
            )
            .is_err()
        );
    }

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
