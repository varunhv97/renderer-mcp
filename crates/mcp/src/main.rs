//! MCP stdio server backed by one persistent local GPU daemon.
//!
//! Exposes RendererCli to LLM/agent clients over the Model Context
//! Protocol: `render_scene` renders directly (in-process, via
//! `renderer_daemon::RendererDaemon`, lazily created on first use), while
//! the named-scene tools (`create_scene`, `patch_scene`, ...) instead talk to
//! a daemon over TCP via `renderer_daemon::DaemonClient` -- the one named by
//! `RENDERER_DAEMON_ENDPOINT` if it's running, else one this process starts
//! itself (see `resolve_daemon_endpoint`). Tool JSON Schemas below
//! are kept field-for-field in sync with `renderer_schema`'s types by hand
//! (see `scene_schema` and friends), since MCP has no way to derive one
//! from the Rust types directly. `show_image` instead delegates entirely
//! to `renderer_terminal`, the same crate behind the CLI's `renderer show`.

use renderer_daemon::RendererDaemon;
use std::io::{self, BufRead, Write};

use crate::handlers::call_tool;
use crate::tools::{named_scene_tool, render_tool, show_image_tool};

mod handlers;
mod inspect;
mod output;
mod schema;
mod tools;

/// Sent in the `initialize` response so MCP clients load it into the model's
/// context.
const SERVER_INSTRUCTIONS: &str = "ALWAYS auto-open every rendered image or GIF. After any \
tool that writes an image or GIF to disk (render_scene, render_named_scene, \
export_named_gif) returns successfully, immediately call show_image with the returned \
output path so the user sees the result, every time and without waiting to be asked. Do \
not rely on the inline preview alone: it is a still frame, even for a GIF.";

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut daemon: Option<RendererDaemon> = None;
    for line in stdin.lock().lines().map_while(Result::ok) {
        let request: serde_json::Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => continue,
        };
        // JSON-RPC notifications (no "id") expect no reply -- e.g. the
        // "notifications/initialized" message MCP clients send right after
        // `initialize` -- so skip anything without one rather than sending
        // a response nobody is waiting for.
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
            "serverInfo": { "name": "renderer-mcp", "version": env!("CARGO_PKG_VERSION") },
            "instructions": SERVER_INSTRUCTIONS
        })),
        Some("tools/list") => Ok(
            serde_json::json!({ "tools": [render_tool(), named_scene_tool("create_scene"), named_scene_tool("get_scene"), named_scene_tool("replace_scene"), named_scene_tool("patch_scene"), named_scene_tool("render_named_scene"), named_scene_tool("export_named_gif"), named_scene_tool("inspect_image"), named_scene_tool("destroy_scene"), show_image_tool()] }),
        ),
        Some("tools/call") => call_tool(request, daemon),
        _ => Err("method not found".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::call_tool;

    #[test]
    fn describes_the_mcp_server_and_rejects_unknown_requests() {
        let mut daemon = None;
        let initialize =
            respond(&serde_json::json!({ "method": "initialize" }), &mut daemon).unwrap();
        assert_eq!(initialize["serverInfo"]["name"], "renderer-mcp");
        assert!(
            initialize["instructions"]
                .as_str()
                .is_some_and(|text| text.contains("show_image"))
        );
        let tools = respond(&serde_json::json!({ "method": "tools/list" }), &mut daemon).unwrap();
        assert_eq!(tools["tools"][0]["name"], "render_scene");
        assert!(
            tools["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "replace_scene")
        );
        assert!(
            tools["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "export_named_gif")
        );
        assert!(
            tools["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["name"] == "inspect_image")
        );
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
}
