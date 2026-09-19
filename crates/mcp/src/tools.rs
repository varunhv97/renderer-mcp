use crate::schema::{
    asset_root_param_schema, scene_id_param_schema, scene_patch_schema, scene_schema,
};

pub(crate) fn named_scene_tool(name: &str) -> serde_json::Value {
    let (description, required, properties) = match name {
        "create_scene" => (
            "Create a named scene in the renderer daemon (RENDERER_DAEMON_ENDPOINT if set and running, otherwise a built-in daemon started on demand) so it can be referenced by scene_id in later calls (render_named_scene, patch_scene, replace_scene, export_named_gif, get_scene, destroy_scene) instead of resending the full scene document each time. scene_id is a caller-chosen label unrelated to anything inside the scene document; the daemon does not inspect it beyond using it as a lookup key.",
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

pub(crate) fn render_tool() -> serde_json::Value {
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
pub(crate) fn show_image_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "show_image",
        "description": "Display a local PNG or GIF inline in the caller's own terminal -- the same mechanism as the CLI's `renderer show <path>`. When running inside cmux (detected via $CMUX_SOCKET_PATH), the image opens in cmux's native file-preview panel instead of being drawn into the terminal grid; otherwise this detects a Kitty-graphics-protocol terminal (Kitty, Ghostty, cmux, WezTerm) or iTerm2 and writes real graphics-protocol escape sequences, or -- when neither is detected, or there's nowhere safe to write escape sequences to at all (e.g. this process has no controlling terminal of its own) -- opens the image in the OS's own default viewer (e.g. Preview.app via `open` on macOS) as a separate window instead. There is no text-based terminal fallback: it only ever mattered for terminals supporting neither real graphics protocol (in practice, just Apple's Terminal.app), and a full-quality external viewer is strictly better than a blocky, palette-limited approximation. This tool returns only a text summary of what happened (status and protocol used) -- unlike render_scene/render_named_scene, no image content is returned, because the whole point of this tool is that the image is already visible to the user through the terminal/cmux/viewer mechanism, not base64-encoded back to the caller.",
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
                    "enum": ["auto", "kitty", "iterm2"],
                    "default": "auto",
                    "description": "Which terminal graphics protocol to use. \"auto\" (the default) detects the caller's terminal from environment variables and picks, in order: the Kitty graphics protocol (Kitty, Ghostty, cmux, or WezTerm), or iTerm2's OSC 1337 inline-image protocol; if neither is detected, the image opens in the OS's own default viewer instead (see the top-level description) regardless of this setting. \"kitty\" forces the Kitty graphics protocol (with native terminal-driven animation for Kitty/WezTerm, or a simulated frame-by-frame redraw for Ghostty/cmux, which lack that extension). \"iterm2\" forces iTerm2's protocol, which decodes and loops an animated GIF itself. Ignored whenever cmux's native file-preview panel is used instead (see the top-level description) -- cmux always uses its own panel regardless of this value."
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
