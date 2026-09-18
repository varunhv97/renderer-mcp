//! MCP stdio server backed by one persistent local GPU daemon.

use base64::{Engine, engine::general_purpose::STANDARD};
use image::{ImageFormat, ImageReader};
use renderer_daemon::{DaemonClient, DaemonRequest, DaemonResult, RenderResult, RendererDaemon};
use renderer_schema::{
    MAX_CANVAS_DIMENSION, MAX_EFFECT_SHADER_BYTES, MAX_NODES, MAX_PATCH_OPERATIONS,
    MAX_PATH_POINTS, SCENE_VERSION_V1, ScenePatchV1, SceneV1,
};
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
            serde_json::json!({ "tools": [render_tool(), named_scene_tool("create_scene"), named_scene_tool("get_scene"), named_scene_tool("replace_scene"), named_scene_tool("patch_scene"), named_scene_tool("render_named_scene"), named_scene_tool("export_named_gif"), named_scene_tool("inspect_image"), named_scene_tool("destroy_scene"), show_image_tool()] }),
        ),
        Some("tools/call") => call_tool(request, daemon),
        _ => Err("method not found".into()),
    }
}

/// Description for the `scene_id` parameter shared by every named-scene tool.
///
/// Called out explicitly because this was a specific point of user confusion:
/// `scene_id` is a caller-chosen label used only to look the scene back up
/// through this MCP server (and the daemon it talks to) — it is never read
/// from, written into, or validated against the scene document itself.
fn scene_id_param_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "string",
        "description": "A caller-chosen name for this scene, used only to refer back to it in later tool calls (get_scene, replace_scene, patch_scene, render_named_scene, export_named_gif, destroy_scene). It is arbitrary: it is not read from, written into, or validated against the scene document's own fields, and need not be unique across different tools or sessions beyond your own use of it."
    })
}

fn named_scene_tool(name: &str) -> serde_json::Value {
    let (description, required, properties) = match name {
        "create_scene" => (
            "Create a named scene in RENDERER_DAEMON_ENDPOINT so it can be referenced by scene_id in later calls (render_named_scene, patch_scene, replace_scene, export_named_gif, get_scene, destroy_scene) instead of resending the full scene document each time. scene_id is a caller-chosen label unrelated to anything inside the scene document; the daemon does not inspect it beyond using it as a lookup key.",
            serde_json::json!(["scene_id", "scene"]),
            serde_json::json!({
                "scene_id": scene_id_param_schema(),
                "scene": scene_schema(),
                "asset_root": asset_root_param_schema(),
            }),
        ),
        "get_scene" => (
            "Get a previously created named scene's current document and revision number, looked up by the caller-chosen scene_id passed to create_scene (not by any field inside the scene document itself).",
            serde_json::json!(["scene_id"]),
            serde_json::json!({ "scene_id": scene_id_param_schema() }),
        ),
        "patch_scene" => (
            "Atomically apply one or more typed operations to a named scene. All operations in `patch.operations` are applied together (all-or-nothing) and, if expected_revision is set, only if the scene has not changed since that revision.",
            serde_json::json!(["scene_id", "patch"]),
            serde_json::json!({ "scene_id": scene_id_param_schema(), "patch": scene_patch_schema() }),
        ),
        "replace_scene" => (
            "Replace a named scene's entire document, optionally requiring its current revision (optimistic concurrency control) so the replace fails instead of clobbering a concurrent change.",
            serde_json::json!(["scene_id", "scene"]),
            serde_json::json!({
                "scene_id": scene_id_param_schema(),
                "scene": scene_schema(),
                "expected_revision": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional optimistic-concurrency check. If provided, the replace is rejected unless the scene's current revision (as returned by get_scene or create_scene) equals this value."
                },
                "asset_root": asset_root_param_schema(),
            }),
        ),
        "render_named_scene" => (
            "Render a named scene to a local PNG path and return inline image content.",
            serde_json::json!(["scene_id", "output_path"]),
            serde_json::json!({
                "scene_id": scene_id_param_schema(),
                "output_path": { "type": "string", "description": "Local destination path for the rendered PNG; must use a .png extension." },
            }),
        ),
        "export_named_gif" => (
            "Export a named animated scene (one with a timeline) to a local GIF path and return inline image content.",
            serde_json::json!(["scene_id", "output_path"]),
            serde_json::json!({
                "scene_id": scene_id_param_schema(),
                "output_path": { "type": "string", "description": "Local destination path for the exported GIF; must use a .gif extension." },
            }),
        ),
        "inspect_image" => (
            "Inspect a local image file's dimensions, MIME type, and SHA-256 hash without rendering anything.",
            serde_json::json!(["path"]),
            serde_json::json!({ "path": { "type": "string", "description": "Local filesystem path to an image file to inspect." } }),
        ),
        _ => (
            "Destroy a named scene, freeing its stored state in the daemon. This does not delete any rendered output files already written to disk.",
            serde_json::json!(["scene_id"]),
            serde_json::json!({ "scene_id": scene_id_param_schema() }),
        ),
    };
    serde_json::json!({ "name": name, "description": description, "inputSchema": { "type": "object", "required": required, "properties": properties } })
}

fn asset_root_param_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "string",
        "description": "Local directory that `image` node `source` paths are resolved against. Required if, and only if, the scene contains one or more `image` nodes."
    })
}

fn render_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "render_scene",
        "description": "Render a renderer.scene.v1 JSON scene locally and return a PNG image plus metadata. Unlike the named-scene tools, this renders directly without storing the scene for later reference.",
        "inputSchema": {
            "type": "object",
            "required": ["scene"],
            "properties": {
                "scene": scene_schema(),
                "output_path": { "type": "string", "description": "Optional local destination path for the rendered PNG; must use a .png extension. Defaults to a path under .renderer/output/ when omitted." }
            }
        }
    })
}

/// Schema for `show_image`, the MCP counterpart to the CLI's `renderer show`
/// subcommand -- both are thin wrappers around the shared
/// `renderer_terminal` crate, per this project's "CLI and MCP expose the
/// same capabilities" principle. Parameters mirror `renderer show`'s flags
/// one-for-one.
fn show_image_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "show_image",
        "description": "Display a local PNG or GIF inline in the caller's own terminal -- the same mechanism as the CLI's `renderer show <path>`. When running inside cmux (detected via $CMUX_SOCKET_PATH), the image opens in cmux's native file-preview panel instead of being drawn into the terminal grid; otherwise this detects a Kitty-graphics-protocol terminal (Kitty, Ghostty, cmux, WezTerm), iTerm2, or falls back to an ANSI true-color half-block rendering. This tool returns only a text summary of what happened (status and protocol used) -- unlike render_scene/render_named_scene, no image content is returned, because the whole point of this tool is that the image is already visible to the user through the terminal/cmux mechanism, not base64-encoded back to the caller.",
        "inputSchema": {
            "type": "object",
            "required": [],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Local filesystem path to a PNG or GIF to display. Required unless `clear` is true, in which case it is ignored."
                },
                "protocol": {
                    "type": "string",
                    "enum": ["auto", "kitty", "iterm2", "ansi"],
                    "default": "auto",
                    "description": "Which terminal graphics protocol to use. \"auto\" (the default) detects the caller's terminal from environment variables and picks, in order: the Kitty graphics protocol (Kitty, Ghostty, cmux, or WezTerm), iTerm2's OSC 1337 inline-image protocol, or an ANSI true-color half-block fallback that works in any terminal. \"kitty\" forces the Kitty graphics protocol (with native terminal-driven animation for Kitty/WezTerm, or a simulated frame-by-frame redraw for Ghostty/cmux, which lack that extension). \"iterm2\" forces iTerm2's protocol, which decodes and loops an animated GIF itself. \"ansi\" forces the half-block fallback, simulating GIF animation with its own redraw loop. Ignored whenever cmux's native file-preview panel is used instead (see the top-level description) -- cmux always uses its own panel regardless of this value."
                },
                "tty": {
                    "type": "string",
                    "description": "Write escape sequences to this device file (e.g. \"/dev/ttys008\") instead of discovering a terminal automatically. Mainly useful for targeting a specific terminal session other than the one this MCP server's own process is attached to."
                },
                "clear": {
                    "type": "boolean",
                    "default": false,
                    "description": "Send only a delete/clear command and stop, instead of displaying an image: a Kitty delete-all-images command against the resolved terminal target, or, under cmux, a close of the most recently opened preview surface. When true, `path` is not required."
                },
                "loops": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of times to loop an animated GIF's simulated or native playback. 0 or omitted means loop forever, matching normal GIF playback. Has no effect on a static image or when `clear` is true."
                }
            }
        }
    })
}

/// Full JSON Schema for a `renderer.scene.v1` `SceneV1` document, matching
/// `crates/schema/src/lib.rs` field-for-field (including its
/// `deny_unknown_fields` attributes, reflected here as `additionalProperties:
/// false` at every object level) so an MCP client can construct a valid
/// scene from this description alone.
fn scene_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["version", "canvas"],
        "description": "A renderer.scene.v1 scene document. `version` and `canvas` are required; `nodes` defaults to an empty list, and `timeline`/`effect` are optional.",
        "properties": {
            "version": {
                "type": "string",
                "const": SCENE_VERSION_V1,
                "description": format!("Schema version identifier. Must be exactly \"{SCENE_VERSION_V1}\" — no other value is accepted.")
            },
            "canvas": canvas_schema(),
            "nodes": {
                "type": "array",
                "items": node_schema(),
                "maxItems": MAX_NODES,
                "default": [],
                "description": format!("Flat list of the scene's visual nodes (no grouping/hierarchy). Optional; defaults to an empty list. Each node's `id` must be unique within the scene. At most {MAX_NODES} nodes.")
            },
            "timeline": timeline_schema(),
            "effect": effect_schema(),
        }
    })
}

/// Schema for `SceneV1::canvas` (`CanvasV1`).
fn canvas_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["width", "height"],
        "description": "The scene's output canvas. `background`, when set, belongs nested here (i.e. at `scene.canvas.background`) — there is no top-level `background` field on the scene document itself.",
        "properties": {
            "width": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_CANVAS_DIMENSION,
                "description": format!("Canvas width in pixels. Must be from 1 through {MAX_CANVAS_DIMENSION}.")
            },
            "height": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_CANVAS_DIMENSION,
                "description": format!("Canvas height in pixels. Must be from 1 through {MAX_CANVAS_DIMENSION}.")
            },
            "background": {
                "type": "array",
                "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                "minItems": 4,
                "maxItems": 4,
                "default": [0.0, 0.0, 0.0, 0.0],
                "description": "Optional background color as an RGBA array [r, g, b, a], each channel a float from 0.0 through 1.0. Defaults to fully transparent ([0.0, 0.0, 0.0, 0.0]) when omitted. This field is nested under `canvas` (`scene.canvas.background`) — it is NOT a top-level field of the scene document; a top-level `scene.background` is rejected as an unknown field."
            }
        }
    })
}

/// Shared RGBA color schema for node/keyframe colors.
///
/// `description_prefix` is prepended to the generic color-format
/// explanation so each call site can say what the color is *for* (fill,
/// stroke, keyframe target, ...).
fn color_schema(description_prefix: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "minItems": 4,
        "maxItems": 4,
        "description": format!("{description_prefix} RGBA color as an array [r, g, b, a], each channel a float from 0.0 through 1.0.")
    })
}

/// Schema for a node's `id` field, shared across all six node kinds.
fn node_id_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "string",
        "minLength": 1,
        "description": "Unique identifier for this node within the scene (must be non-empty, and unique across all of `nodes`). Referenced by timeline keyframes (`target`) and by the `remove_node` patch operation's `id`."
    })
}

/// Schema for one `{x, y}` point in a `path` node's `points` array
/// (`PointV1`).
fn point_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["x", "y"],
        "properties": {
            "x": { "type": "number", "description": "X coordinate in pixels, measured from the canvas's top-left corner (0,0); increases rightward." },
            "y": { "type": "number", "description": "Y coordinate in pixels, measured from the canvas's top-left corner (0,0); increases downward." }
        }
    })
}

/// Schema for `NodeV1`/`NodeKindV1`: a tagged union discriminated by the
/// `kind` field, matching `#[serde(tag = "kind", rename_all = "snake_case",
/// deny_unknown_fields)]` on `NodeKindV1` plus the flattened `id` from
/// `NodeV1`. Expressed as `oneOf` with a `const` on `kind` per branch
/// (rather than `if`/`then`) because that is the standard JSON Schema
/// idiom for a tagged union and is the form MCP/LLM clients most reliably
/// parse into "pick one of these N shapes based on this field".
fn node_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "description": "A single scene node. `kind` is a discriminator selecting one of six node types — \"rect\", \"ellipse\", \"line\", \"path\", \"text\", \"image\" — each requiring its own set of additional fields, as listed in the matching oneOf branch. All coordinates are in pixels measured from the canvas's top-left corner (0,0), with x increasing rightward and y increasing downward.",
        "oneOf": [
            rect_node_schema(),
            ellipse_node_schema(),
            line_node_schema(),
            path_node_schema(),
            text_node_schema(),
            image_node_schema(),
        ]
    })
}

fn rect_node_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "kind", "x", "y", "width", "height", "color"],
        "description": "A filled, axis-aligned rectangle.",
        "properties": {
            "id": node_id_schema(),
            "kind": { "const": "rect", "description": "Node type discriminator." },
            "x": { "type": "number", "description": "Left edge X coordinate in pixels, from the canvas's top-left corner." },
            "y": { "type": "number", "description": "Top edge Y coordinate in pixels, from the canvas's top-left corner." },
            "width": { "type": "number", "exclusiveMinimum": 0, "description": "Width in pixels; must be finite and greater than 0." },
            "height": { "type": "number", "exclusiveMinimum": 0, "description": "Height in pixels; must be finite and greater than 0." },
            "color": color_schema("Fill color."),
        }
    })
}

fn ellipse_node_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "kind", "cx", "cy", "rx", "ry", "color"],
        "description": "A filled ellipse (or circle, when rx equals ry).",
        "properties": {
            "id": node_id_schema(),
            "kind": { "const": "ellipse", "description": "Node type discriminator." },
            "cx": { "type": "number", "description": "Center X coordinate in pixels, from the canvas's top-left corner." },
            "cy": { "type": "number", "description": "Center Y coordinate in pixels, from the canvas's top-left corner." },
            "rx": { "type": "number", "exclusiveMinimum": 0, "description": "Horizontal radius in pixels; must be finite and greater than 0." },
            "ry": { "type": "number", "exclusiveMinimum": 0, "description": "Vertical radius in pixels; must be finite and greater than 0." },
            "color": color_schema("Fill color."),
        }
    })
}

fn line_node_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "kind", "x1", "y1", "x2", "y2", "thickness", "color"],
        "description": "A straight line segment between two points.",
        "properties": {
            "id": node_id_schema(),
            "kind": { "const": "line", "description": "Node type discriminator." },
            "x1": { "type": "number", "description": "Start point X coordinate in pixels, from the canvas's top-left corner." },
            "y1": { "type": "number", "description": "Start point Y coordinate in pixels, from the canvas's top-left corner." },
            "x2": { "type": "number", "description": "End point X coordinate in pixels, from the canvas's top-left corner." },
            "y2": { "type": "number", "description": "End point Y coordinate in pixels, from the canvas's top-left corner." },
            "thickness": { "type": "number", "exclusiveMinimum": 0, "description": "Line thickness in pixels; must be finite and greater than 0." },
            "color": color_schema("Line color."),
        }
    })
}

fn path_node_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "kind", "points", "color"],
        "description": "A filled polygon rendered as a triangle fan from its first point, in the order given.",
        "properties": {
            "id": node_id_schema(),
            "kind": { "const": "path", "description": "Node type discriminator." },
            "points": {
                "type": "array",
                "items": point_schema(),
                "minItems": 3,
                "maxItems": MAX_PATH_POINTS,
                "description": format!("Ordered vertices of the polygon. At least 3 points are required (fewer cannot enclose an area); at most {MAX_PATH_POINTS}.")
            },
            "color": color_schema("Fill color."),
        }
    })
}

fn text_node_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "kind", "x", "y", "text", "size", "color"],
        "description": "A run of text rendered with the renderer's built-in font.",
        "properties": {
            "id": node_id_schema(),
            "kind": { "const": "text", "description": "Node type discriminator." },
            "x": { "type": "number", "description": "Left edge X coordinate in pixels, from the canvas's top-left corner, where the first glyph starts." },
            "y": { "type": "number", "description": "Y coordinate in pixels, from the canvas's top-left corner, positioning the top of the text's em-square (not its baseline)." },
            "text": { "type": "string", "minLength": 1, "description": "The text to render; must be non-empty." },
            "size": { "type": "number", "exclusiveMinimum": 0, "description": "Font size in pixels; must be finite and greater than 0." },
            "color": color_schema("Text color."),
        }
    })
}

fn image_node_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "kind", "x", "y", "width", "height", "source"],
        "description": "A raster image drawn scaled to fit an axis-aligned box.",
        "properties": {
            "id": node_id_schema(),
            "kind": { "const": "image", "description": "Node type discriminator." },
            "x": { "type": "number", "description": "Left edge X coordinate in pixels, from the canvas's top-left corner." },
            "y": { "type": "number", "description": "Top edge Y coordinate in pixels, from the canvas's top-left corner." },
            "width": { "type": "number", "exclusiveMinimum": 0, "description": "Drawn width in pixels; must be finite and greater than 0. The source image is scaled to fit." },
            "height": { "type": "number", "exclusiveMinimum": 0, "description": "Drawn height in pixels; must be finite and greater than 0. The source image is scaled to fit." },
            "source": { "type": "string", "description": "A path to a local image file, relative to the configured asset root (the `asset_root` argument passed to create_scene/replace_scene). A scene with any `image` node requires `asset_root` to be set. Remote URLs are not fetched." },
        }
    })
}

/// Schema for `SceneV1::timeline` / `PatchOperationV1::SetTimeline`
/// (`TimelineV1`).
fn timeline_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["fps", "duration_ms"],
        "description": "Optional animation timeline for the scene. When present, render output can vary over time (see export_named_gif). `fps` and `duration_ms` are also jointly bounded so total rendered frames stay <= 300 and aggregate raster work (canvas pixels times frame count) stays <= 67108864 (64 * 1024 * 1024) pixels.",
        "properties": {
            "fps": { "type": "integer", "minimum": 1, "maximum": 60, "description": "Frames per second, from 1 through 60." },
            "duration_ms": { "type": "integer", "minimum": 1, "maximum": 10_000, "description": "Total timeline duration in milliseconds, from 1 through 10000 (10 seconds)." },
            "keyframes": {
                "type": "array",
                "default": [],
                "items": keyframe_schema(),
                "description": "Keyframes describing how node properties change over time. Optional; defaults to an empty list."
            }
        }
    })
}

/// Schema for one entry of `TimelineV1::keyframes` (`KeyframeV1`).
fn keyframe_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["at_ms", "target", "property"],
        "properties": {
            "at_ms": { "type": "integer", "minimum": 0, "description": "Time offset in milliseconds from the start of the timeline when this keyframe applies. Must not exceed the timeline's duration_ms." },
            "target": { "type": "string", "description": "The `id` of the scene node this keyframe animates. Must match an existing node in `nodes`." },
            "property": animated_property_schema(),
        }
    })
}

/// Schema for `KeyframeV1::property` (`AnimatedPropertyV1`), matching its
/// `#[serde(tag = "kind", content = "value", rename_all = "snake_case")]`
/// representation: `{"kind": "opacity", "value": <number>}` or
/// `{"kind": "color", "value": [r, g, b, a]}`.
fn animated_property_schema() -> serde_json::Value {
    serde_json::json!({
        "description": "The animated property. `kind` selects which property is animated and determines the shape of `value`: \"opacity\" (a single 0.0-1.0 number) or \"color\" (an [r, g, b, a] array).",
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["kind", "value"],
                "description": "Animates the target node's opacity.",
                "properties": {
                    "kind": { "const": "opacity" },
                    "value": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Opacity at this keyframe, from 0.0 (fully transparent) through 1.0 (fully opaque)." }
                }
            },
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["kind", "value"],
                "description": "Animates the target node's color.",
                "properties": {
                    "kind": { "const": "color" },
                    "value": color_schema("Target color at this keyframe.")
                }
            }
        ]
    })
}

/// Schema for `SceneV1::effect` (`EffectV1`).
fn effect_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["shader"],
        "description": "Optional scene-level, full-canvas WGSL post-process effect applied after the scene is rendered.",
        "properties": {
            "shader": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_EFFECT_SHADER_BYTES,
                "description": format!("WGSL source that must define exactly one function with the exact signature `fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32>`, and nothing else about the pipeline: `uv` is the normalized [0,1] screen-space coordinate and `color` is the already-rendered pixel color at that coordinate; the function returns the transformed color. The renderer wraps this pure per-pixel color transform in an internal, fixed template (vertex stage, texture bindings) — authors never write or control bindings or vertex data directly. Must be non-empty and at most {MAX_EFFECT_SHADER_BYTES} bytes (64 KiB) of UTF-8 source.")
            }
        }
    })
}

/// Full JSON Schema for a `ScenePatchV1` `patch` parameter, matching
/// `crates/schema/src/lib.rs` field-for-field.
fn scene_patch_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["operations"],
        "description": "A bounded, typed set of scene mutations. All operations in `operations` are applied atomically (all-or-nothing), in order.",
        "properties": {
            "expected_revision": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional optimistic-concurrency check. If provided, the patch is rejected unless the scene's current revision (as returned by get_scene or create_scene) equals this value. Omit to apply unconditionally."
            },
            "operations": {
                "type": "array",
                "items": patch_operation_schema(),
                "minItems": 1,
                "maxItems": MAX_PATCH_OPERATIONS,
                "description": format!("Ordered list of typed operations to apply. Must contain 1 through {MAX_PATCH_OPERATIONS} operations.")
            }
        }
    })
}

/// Schema for one entry of `ScenePatchV1::operations` (`PatchOperationV1`):
/// a tagged union discriminated by `op`, matching
/// `#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]`.
/// Uses `oneOf` with a `const` on `op` per branch for the same reason as
/// `node_schema` above.
fn patch_operation_schema() -> serde_json::Value {
    serde_json::json!({
        "description": "A single typed scene mutation. `op` selects which operation this is and determines which other field is required, as listed in the matching oneOf branch.",
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["op", "canvas"],
                "description": "Replaces the scene's canvas.",
                "properties": {
                    "op": { "const": "set_canvas" },
                    "canvas": canvas_schema(),
                }
            },
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["op", "node"],
                "description": "Inserts a new node, or replaces the existing node with the same `id`.",
                "properties": {
                    "op": { "const": "upsert_node" },
                    "node": node_schema(),
                }
            },
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["op", "id"],
                "description": "Removes the node with the given `id`, if present.",
                "properties": {
                    "op": { "const": "remove_node" },
                    "id": { "type": "string", "description": "The `id` of the node to remove." },
                }
            },
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["op", "timeline"],
                "description": "Replaces the scene's timeline (or sets one if the scene had none).",
                "properties": {
                    "op": { "const": "set_timeline" },
                    "timeline": timeline_schema(),
                }
            },
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["op"],
                "description": "Removes the scene's timeline, if it has one. Takes no other fields.",
                "properties": {
                    "op": { "const": "clear_timeline" },
                }
            },
        ]
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

    #[test]
    fn scene_and_patch_schemas_document_every_field_and_variant() {
        let create = named_scene_tool("create_scene");
        let scene_schema = &create["inputSchema"]["properties"]["scene"];
        assert_eq!(
            scene_schema["properties"]["version"]["const"],
            SCENE_VERSION_V1
        );
        assert_eq!(
            scene_schema["required"].as_array().unwrap().as_slice(),
            &[
                serde_json::Value::String("version".into()),
                serde_json::Value::String("canvas".into())
            ]
        );
        let background_description =
            scene_schema["properties"]["canvas"]["properties"]["background"]["description"]
                .as_str()
                .unwrap();
        assert!(background_description.contains("canvas"));
        assert!(background_description.contains("NOT a top-level field"));

        let node_branches = scene_schema["properties"]["nodes"]["items"]["oneOf"]
            .as_array()
            .unwrap();
        let kinds: Vec<_> = node_branches
            .iter()
            .map(|branch| branch["properties"]["kind"]["const"].as_str().unwrap())
            .collect();
        assert_eq!(
            kinds,
            vec!["rect", "ellipse", "line", "path", "text", "image"]
        );

        let patch_tool = named_scene_tool("patch_scene");
        let patch_schema = &patch_tool["inputSchema"]["properties"]["patch"];
        let op_branches = patch_schema["properties"]["operations"]["items"]["oneOf"]
            .as_array()
            .unwrap();
        let ops: Vec<_> = op_branches
            .iter()
            .map(|branch| branch["properties"]["op"]["const"].as_str().unwrap())
            .collect();
        assert_eq!(
            ops,
            vec![
                "set_canvas",
                "upsert_node",
                "remove_node",
                "set_timeline",
                "clear_timeline"
            ]
        );

        let scene_id_description = create["inputSchema"]["properties"]["scene_id"]["description"]
            .as_str()
            .unwrap();
        assert!(scene_id_description.contains("caller-chosen"));
        let get_scene_description = named_scene_tool("get_scene")["description"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(get_scene_description.contains("caller-chosen"));
    }

    #[test]
    fn a_full_example_scene_matches_the_documented_shape_and_validates() {
        let json = serde_json::json!({
            "version": SCENE_VERSION_V1,
            "canvas": { "width": 64, "height": 64, "background": [0.0, 0.0, 0.0, 1.0] },
            "nodes": [
                { "id": "r", "kind": "rect", "x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0, "color": [1.0, 0.0, 0.0, 1.0] },
                { "id": "e", "kind": "ellipse", "cx": 5.0, "cy": 5.0, "rx": 2.0, "ry": 2.0, "color": [0.0, 1.0, 0.0, 1.0] },
                { "id": "l", "kind": "line", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "thickness": 1.0, "color": [0.0, 0.0, 1.0, 1.0] },
                { "id": "p", "kind": "path", "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 0.0, "y": 1.0}], "color": [1.0, 1.0, 0.0, 1.0] },
                { "id": "t", "kind": "text", "x": 0.0, "y": 0.0, "text": "hi", "size": 12.0, "color": [1.0, 1.0, 1.0, 1.0] },
                { "id": "i", "kind": "image", "x": 0.0, "y": 0.0, "width": 8.0, "height": 8.0, "source": "asset.png" }
            ],
            "timeline": { "fps": 30, "duration_ms": 1000, "keyframes": [
                { "at_ms": 0, "target": "r", "property": { "kind": "opacity", "value": 1.0 } },
                { "at_ms": 500, "target": "r", "property": { "kind": "color", "value": [1.0, 1.0, 1.0, 1.0] } }
            ]},
            "effect": { "shader": "fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32> { return color; }" }
        });
        let scene: SceneV1 = serde_json::from_value(json).unwrap();
        assert_eq!(scene.validate(), Ok(()));
    }
}
