use renderer_schema::{
    MAX_CANVAS_DIMENSION, MAX_EFFECT_SHADER_BYTES, MAX_NODES, MAX_PATCH_OPERATIONS,
    MAX_PATH_POINTS, SCENE_VERSION_V1,
};

/// Description for the `scene_id` parameter shared by every named-scene tool.
///
/// Called out explicitly because this was a specific point of user confusion:
/// `scene_id` is a caller-chosen label used only to look the scene back up
/// through this MCP server (and the daemon it talks to) — it is never read
/// from, written into, or validated against the scene document itself.
pub(crate) fn scene_id_param_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "string",
        "description": "A caller-chosen name for this scene, used only to refer back to it in later tool calls (get_scene, replace_scene, patch_scene, render_named_scene, export_named_gif, destroy_scene). It is arbitrary: it is not read from, written into, or validated against the scene document's own fields, and need not be unique across different tools or sessions beyond your own use of it."
    })
}

pub(crate) fn asset_root_param_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "string",
        "description": "Local directory that `image` node `source` paths are resolved against. Required if, and only if, the scene contains one or more `image` nodes."
    })
}

/// Full JSON Schema for a `renderer.scene.v1` `SceneV1` document, matching
/// `crates/schema/src/lib.rs` field-for-field (including its
/// `deny_unknown_fields` attributes, reflected here as `additionalProperties:
/// false` at every object level) so an MCP client can construct a valid
/// scene from this description alone.
pub(crate) fn scene_schema() -> serde_json::Value {
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
/// stroke, keyframe target, gradient stop, ...).
fn color_schema(description_prefix: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "minItems": 4,
        "maxItems": 4,
        "description": format!("{description_prefix} RGBA color as an array [r, g, b, a], each channel a float from 0.0 through 1.0.")
    })
}

/// Shared fill schema for a node's `color` field (`renderer_schema::
/// FillV1`): accepts either a plain 4-element RGBA array (a solid fill,
/// identical to -- and 100% backward compatible with -- every scene
/// authored before gradients existed) or a gradient object. The JSON
/// *property key* stays `color` on the wire even though the underlying
/// Rust field is named `fill` (see `FillV1`'s doc comment in
/// `renderer_schema` for why); this schema is wired up under that same
/// `"color"` property name at every call site below.
///
/// `supports_gradient` distinguishes node kinds that render a true,
/// continuous per-pixel gradient (`Rect`/`Ellipse`) from those that only
/// resolve a gradient to one flat, representative color (`Line`/`Path`/
/// `Text` -- the 50/50 midpoint blend of the gradient's two stops); the
/// generated description states this plainly either way, so callers don't
/// assume gradient support that a given node kind doesn't actually have.
fn fill_schema(description_prefix: &str, supports_gradient: bool) -> serde_json::Value {
    let gradient_note = if supports_gradient {
        "This node kind renders a gradient object as a real, continuous per-pixel gradient."
    } else {
        "This node kind does not render a true per-pixel gradient: a gradient object here is \
         resolved to one flat color, the 50/50 midpoint blend of its two stops."
    };
    serde_json::json!({
        "description": format!(
            "{description_prefix} Either a plain solid RGBA array [r, g, b, a] (each channel a \
             float from 0.0 through 1.0), or a gradient object: {{\"kind\": \"linear_gradient\", \
             \"from\": [r,g,b,a], \"to\": [r,g,b,a], \"angle_degrees\": 0.0}} for a linear \
             gradient, or {{\"kind\": \"radial_gradient\", \"center\": [r,g,b,a], \"edge\": \
             [r,g,b,a]}} for a radial gradient (center-to-edge). {gradient_note}"
        ),
        "oneOf": [
            {
                "type": "array",
                "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                "minItems": 4,
                "maxItems": 4,
                "description": "A solid RGBA color as an array [r, g, b, a], each channel a float from 0.0 through 1.0."
            },
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["kind", "from", "to", "angle_degrees"],
                "description": "A linear gradient between two colors along a direction.",
                "properties": {
                    "kind": { "const": "linear_gradient", "description": "Gradient type discriminator." },
                    "from": color_schema("Gradient start color, at the gradient direction's start."),
                    "to": color_schema("Gradient end color, at the gradient direction's end."),
                    "angle_degrees": {
                        "type": "number",
                        "description": "Gradient direction in degrees. 0 points along +x (left-to-right); increasing values rotate clockwise."
                    }
                }
            },
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["kind", "center", "edge"],
                "description": "A radial gradient from a center color to an edge color.",
                "properties": {
                    "kind": { "const": "radial_gradient", "description": "Gradient type discriminator." },
                    "center": color_schema("Gradient color at the shape's center."),
                    "edge": color_schema("Gradient color at the shape's boundary.")
                }
            }
        ]
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

/// Schema for a node's optional `translate` field (`NodeV1::translate`),
/// shared across all six node kinds. Unlike `x`/`y`/`cx`/`cy`/etc. (which
/// differ per node kind), `translate` is uniform: an additive [dx, dy]
/// offset applied on top of the node's own coordinates at render time, e.g.
/// for a rect, effectively `(x + translate[0], y + translate[1])`. This is
/// also the only property `kind: "translate"` keyframes animate (see
/// `animated_property_schema`).
fn translate_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "items": { "type": "number" },
        "minItems": 2,
        "maxItems": 2,
        "default": [0.0, 0.0],
        "description": "Optional additive [dx, dy] pixel offset applied on top of this node's own coordinates at render time. Defaults to [0.0, 0.0] (no offset) when omitted. Values are unconstrained (any finite number); this is the same field `kind: \"translate\"` timeline keyframes animate."
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
        "description": "A filled, axis-aligned rectangle, optionally with rounded corners.",
        "properties": {
            "id": node_id_schema(),
            "translate": translate_schema(),
            "kind": { "const": "rect", "description": "Node type discriminator." },
            "x": { "type": "number", "description": "Left edge X coordinate in pixels, from the canvas's top-left corner." },
            "y": { "type": "number", "description": "Top edge Y coordinate in pixels, from the canvas's top-left corner." },
            "width": { "type": "number", "exclusiveMinimum": 0, "description": "Width in pixels; must be finite and greater than 0." },
            "height": { "type": "number", "exclusiveMinimum": 0, "description": "Height in pixels; must be finite and greater than 0." },
            "corner_radius": {
                "type": "number",
                "minimum": 0.0,
                "default": 0.0,
                "description": "Optional corner rounding radius in pixels. Defaults to 0.0 (sharp, right-angle corners) when omitted. Must be finite, non-negative, and no larger than half of the smaller of `width`/`height` -- larger values are rejected, not clamped."
            },
            "color": fill_schema("Fill.", true),
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
            "translate": translate_schema(),
            "kind": { "const": "ellipse", "description": "Node type discriminator." },
            "cx": { "type": "number", "description": "Center X coordinate in pixels, from the canvas's top-left corner." },
            "cy": { "type": "number", "description": "Center Y coordinate in pixels, from the canvas's top-left corner." },
            "rx": { "type": "number", "exclusiveMinimum": 0, "description": "Horizontal radius in pixels; must be finite and greater than 0." },
            "ry": { "type": "number", "exclusiveMinimum": 0, "description": "Vertical radius in pixels; must be finite and greater than 0." },
            "color": fill_schema("Fill.", true),
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
            "translate": translate_schema(),
            "kind": { "const": "line", "description": "Node type discriminator." },
            "x1": { "type": "number", "description": "Start point X coordinate in pixels, from the canvas's top-left corner." },
            "y1": { "type": "number", "description": "Start point Y coordinate in pixels, from the canvas's top-left corner." },
            "x2": { "type": "number", "description": "End point X coordinate in pixels, from the canvas's top-left corner." },
            "y2": { "type": "number", "description": "End point Y coordinate in pixels, from the canvas's top-left corner." },
            "thickness": { "type": "number", "exclusiveMinimum": 0, "description": "Line thickness in pixels; must be finite and greater than 0." },
            "color": fill_schema("Line color.", false),
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
            "translate": translate_schema(),
            "kind": { "const": "path", "description": "Node type discriminator." },
            "points": {
                "type": "array",
                "items": point_schema(),
                "minItems": 3,
                "maxItems": MAX_PATH_POINTS,
                "description": format!("Ordered vertices of the polygon. At least 3 points are required (fewer cannot enclose an area); at most {MAX_PATH_POINTS}.")
            },
            "color": fill_schema("Fill.", false),
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
            "translate": translate_schema(),
            "kind": { "const": "text", "description": "Node type discriminator." },
            "x": { "type": "number", "description": "Left edge X coordinate in pixels, from the canvas's top-left corner, where the first glyph starts." },
            "y": { "type": "number", "description": "Y coordinate in pixels, from the canvas's top-left corner, positioning the top of the text's em-square (not its baseline)." },
            "text": { "type": "string", "minLength": 1, "description": "The text to render; must be non-empty." },
            "size": { "type": "number", "exclusiveMinimum": 0, "description": "Font size in pixels; must be finite and greater than 0." },
            "color": fill_schema("Text color.", false),
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
            "translate": translate_schema(),
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
        "description": "The animated property. `kind` selects which property is animated and determines the shape of `value`: \"opacity\" (a single 0.0-1.0 number), \"color\" (an [r, g, b, a] array), or \"translate\" (a [dx, dy] array).",
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
            },
            {
                "type": "object",
                "additionalProperties": false,
                "required": ["kind", "value"],
                "description": "Animates the target node's position by setting its `translate` offset (see the node schema's `translate` field). This sets the absolute [dx, dy] offset at this keyframe -- it is not an additive delta on top of other keyframes.",
                "properties": {
                    "kind": { "const": "translate" },
                    "value": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Target [dx, dy] pixel offset at this keyframe. Values are unconstrained (any finite number)."
                    }
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
pub(crate) fn scene_patch_schema() -> serde_json::Value {
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

#[cfg(test)]
mod tests {

    use crate::tools::named_scene_tool;

    use renderer_schema::{SCENE_VERSION_V1, SceneV1};

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
        for branch in node_branches {
            let translate_description = branch["properties"]["translate"]["description"]
                .as_str()
                .unwrap();
            assert!(translate_description.contains("offset"));
        }

        // Every node kind's `color` property documents both the solid-array
        // shape and the gradient-object shape (`FillV1`), and states
        // whether that node kind renders a gradient for real or only a flat
        // representative color.
        for branch in node_branches {
            let kind = branch["properties"]["kind"]["const"].as_str().unwrap();
            if kind == "image" {
                continue; // Image nodes have no color/fill field at all.
            }
            let fill_schema = &branch["properties"]["color"];
            let fill_variants = fill_schema["oneOf"].as_array().unwrap();
            assert_eq!(
                fill_variants.len(),
                3,
                "expected solid + linear + radial for {kind}"
            );
            let gradient_kinds: Vec<_> = fill_variants[1..]
                .iter()
                .map(|variant| variant["properties"]["kind"]["const"].as_str().unwrap())
                .collect();
            assert_eq!(gradient_kinds, vec!["linear_gradient", "radial_gradient"]);
            let description = fill_schema["description"].as_str().unwrap();
            let supports_gradient = matches!(kind, "rect" | "ellipse");
            assert_eq!(
                description.contains("real, continuous per-pixel gradient"),
                supports_gradient,
                "{kind}'s color schema should state whether it renders a real gradient"
            );
        }

        // `rect`'s corner_radius is documented, defaults to 0.0, and is
        // optional (not in `required`).
        let rect_branch = node_branches
            .iter()
            .find(|branch| branch["properties"]["kind"]["const"] == "rect")
            .unwrap();
        assert_eq!(rect_branch["properties"]["corner_radius"]["default"], 0.0);
        let corner_radius_description = rect_branch["properties"]["corner_radius"]["description"]
            .as_str()
            .unwrap();
        assert!(corner_radius_description.contains("rounding"));
        let rect_required: Vec<_> = rect_branch["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert!(!rect_required.contains(&"corner_radius"));

        let property_branches = scene_schema["properties"]["timeline"]["properties"]["keyframes"]
            ["items"]["properties"]["property"]["oneOf"]
            .as_array()
            .unwrap();
        let property_kinds: Vec<_> = property_branches
            .iter()
            .map(|branch| branch["properties"]["kind"]["const"].as_str().unwrap())
            .collect();
        assert_eq!(property_kinds, vec!["opacity", "color", "translate"]);

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
                { "id": "r", "kind": "rect", "translate": [2.0, -3.0], "x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0, "color": [1.0, 0.0, 0.0, 1.0] },
                { "id": "e", "kind": "ellipse", "cx": 5.0, "cy": 5.0, "rx": 2.0, "ry": 2.0, "color": [0.0, 1.0, 0.0, 1.0] },
                { "id": "l", "kind": "line", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "thickness": 1.0, "color": [0.0, 0.0, 1.0, 1.0] },
                { "id": "p", "kind": "path", "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 0.0, "y": 1.0}], "color": [1.0, 1.0, 0.0, 1.0] },
                { "id": "t", "kind": "text", "x": 0.0, "y": 0.0, "text": "hi", "size": 12.0, "color": [1.0, 1.0, 1.0, 1.0] },
                { "id": "i", "kind": "image", "x": 0.0, "y": 0.0, "width": 8.0, "height": 8.0, "source": "asset.png" }
            ],
            "timeline": { "fps": 30, "duration_ms": 1000, "keyframes": [
                { "at_ms": 0, "target": "r", "property": { "kind": "opacity", "value": 1.0 } },
                { "at_ms": 500, "target": "r", "property": { "kind": "color", "value": [1.0, 1.0, 1.0, 1.0] } },
                { "at_ms": 500, "target": "r", "property": { "kind": "translate", "value": [12.0, 8.0] } }
            ]},
            "effect": { "shader": "fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32> { return color; }" }
        });
        let scene: SceneV1 = serde_json::from_value(json).unwrap();
        assert_eq!(scene.validate(), Ok(()));
    }
}
