//! MCP stdio server backed by one persistent local GPU daemon.

use base64::{Engine, engine::general_purpose::STANDARD};
use image::{ImageFormat, ImageReader};
use renderer_daemon::{DaemonClient, DaemonRequest, DaemonResult, RenderResult, RendererDaemon};
use renderer_schema::{ScenePatchV1, SceneV1};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, BufRead, Read, Write},
    net::SocketAddr,
    path::PathBuf,
};

const MAX_OUTPUT_PATH_BYTES: usize = 4 * 1024;
const MAX_INSPECT_FILE_BYTES: u64 = 64 * 1024 * 1024;

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
            serde_json::json!({ "tools": [render_tool(), named_scene_tool("create_scene"), named_scene_tool("get_scene"), named_scene_tool("replace_scene"), named_scene_tool("patch_scene"), named_scene_tool("render_named_scene"), named_scene_tool("export_named_gif"), named_scene_tool("inspect_image"), named_scene_tool("destroy_scene")] }),
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
            serde_json::json!({ "scene_id": { "type": "string" }, "scene": { "type": "object" }, "asset_root": { "type": "string" } }),
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
        "replace_scene" => (
            "Replace a named scene, optionally requiring its current revision.",
            serde_json::json!(["scene_id", "scene"]),
            serde_json::json!({ "scene_id": { "type": "string" }, "scene": { "type": "object" }, "expected_revision": { "type": "integer", "minimum": 1 }, "asset_root": { "type": "string" } }),
        ),
        "render_named_scene" => (
            "Render a named scene to a local PNG path and return inline image content.",
            serde_json::json!(["scene_id", "output_path"]),
            serde_json::json!({ "scene_id": { "type": "string" }, "output_path": { "type": "string" } }),
        ),
        "export_named_gif" => (
            "Export a named animated scene to a local GIF path and return inline image content.",
            serde_json::json!(["scene_id", "output_path"]),
            serde_json::json!({ "scene_id": { "type": "string" }, "output_path": { "type": "string" } }),
        ),
        "inspect_image" => (
            "Inspect a local image's dimensions, MIME type, and SHA-256.",
            serde_json::json!(["path"]),
            serde_json::json!({ "path": { "type": "string" } }),
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

fn call_named_scene_tool(
    name: &str,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    if name == "inspect_image" {
        return inspect_image(arguments);
    }
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

fn expected_revision(arguments: &serde_json::Value) -> Result<Option<u64>, String> {
    let Some(value) = arguments.get("expected_revision") else {
        return Ok(None);
    };
    match value.as_u64().filter(|revision| *revision > 0) {
        Some(revision) => Ok(Some(revision)),
        None => Err("expected_revision must be a positive unsigned 64-bit integer".into()),
    }
}

fn output_path(arguments: &serde_json::Value, extension: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(
        arguments
            .get("output_path")
            .and_then(serde_json::Value::as_str)
            .ok_or("output_path is required")?,
    );
    validate_output_path(&path, extension)?;
    Ok(path)
}

fn validate_output_path(path: &std::path::Path, extension: &str) -> Result<(), String> {
    if path.as_os_str().len() > MAX_OUTPUT_PATH_BYTES {
        return Err("output path exceeds 4 KiB".into());
    }
    if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case(extension))
    {
        Ok(())
    } else {
        Err(format!("output_path must use a .{extension} extension"))
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

fn inspect_image(arguments: &serde_json::Value) -> Result<serde_json::Value, String> {
    let path = PathBuf::from(
        arguments
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or("path is required")?,
    );
    if path.as_os_str().len() > MAX_OUTPUT_PATH_BYTES {
        return Err("path exceeds 4 KiB".into());
    }
    let reader = ImageReader::open(&path)
        .map_err(|error| format!("could not open image: {error}"))?
        .with_guessed_format()
        .map_err(|error| format!("could not detect image format: {error}"))?;
    let format = reader.format().ok_or("could not detect image format")?;
    let (width, height) = reader
        .into_dimensions()
        .map_err(|error| format!("could not inspect image: {error}"))?;
    let sha256 = hash_image_file(&path)?;
    let metadata = serde_json::json!({ "path": path, "mime_type": mime_type_for_format(format)?, "width": width, "height": height, "sha256": sha256 });
    Ok(serde_json::json!({ "content": [{ "type": "text", "text": metadata.to_string() }] }))
}

fn hash_image_file(path: &std::path::Path) -> Result<String, String> {
    let file = File::open(path).map_err(|error| format!("could not read image: {error}"))?;
    if file
        .metadata()
        .map_err(|error| format!("could not inspect image size: {error}"))?
        .len()
        > MAX_INSPECT_FILE_BYTES
    {
        return Err("image exceeds 64 MiB inspection limit".into());
    }
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("could not read image: {error}"))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn mime_type_for_path(path: &std::path::Path) -> Result<&'static str, String> {
    let reader = ImageReader::open(path)
        .map_err(|error| format!("could not open rendered output: {error}"))?
        .with_guessed_format()
        .map_err(|error| format!("could not detect rendered output format: {error}"))?;
    mime_type_for_format(
        reader
            .format()
            .ok_or("could not detect rendered output format")?,
    )
}

fn mime_type_for_format(format: ImageFormat) -> Result<&'static str, String> {
    match format {
        ImageFormat::Png => Ok("image/png"),
        ImageFormat::Gif => Ok("image/gif"),
        ImageFormat::Jpeg => Ok("image/jpeg"),
        ImageFormat::WebP => Ok("image/webp"),
        _ => Err("unsupported image format".into()),
    }
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

    #[test]
    fn validates_output_paths_and_inspects_exact_file_bytes() {
        assert!(validate_output_path(std::path::Path::new("result.png"), "png").is_ok());
        assert!(validate_output_path(std::path::Path::new("result.gif"), "png").is_err());
        assert_eq!(mime_type_for_format(ImageFormat::Gif), Ok("image/gif"));
        assert!(mime_type_for_format(ImageFormat::Bmp).is_err());

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("inspect.png");
        image::RgbaImage::from_pixel(3, 2, image::Rgba([1, 2, 3, 4]))
            .save_with_format(&path, ImageFormat::Png)
            .unwrap();
        let response = inspect_image(&serde_json::json!({ "path": path })).unwrap();
        let metadata: serde_json::Value =
            serde_json::from_str(response["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(metadata["mime_type"], "image/png");
        assert_eq!(metadata["width"], 3);
        assert_eq!(metadata["height"], 2);
        assert_eq!(metadata["sha256"].as_str().unwrap().len(), 64);

        assert_eq!(
            expected_revision(&serde_json::json!({ "expected_revision": 4 })),
            Ok(Some(4))
        );
        assert_eq!(expected_revision(&serde_json::json!({})), Ok(None));
        for value in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("4"),
        ] {
            assert!(expected_revision(&serde_json::json!({ "expected_revision": value })).is_err());
        }

        let large_path = directory.path().join("large.png");
        std::fs::File::create(&large_path)
            .unwrap()
            .set_len(MAX_INSPECT_FILE_BYTES + 1)
            .unwrap();
        assert_eq!(
            hash_image_file(&large_path),
            Err("image exceeds 64 MiB inspection limit".into())
        );
    }
}
