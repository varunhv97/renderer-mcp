//! Versioned, transport-neutral scene contracts for RendererCli.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

pub const SCENE_VERSION_V1: &str = "renderer.scene.v1";
pub const MAX_CANVAS_DIMENSION: u32 = 4_096;
pub const MAX_NODES: usize = 10_000;
pub const MAX_PATH_POINTS: usize = 4_096;
pub const MAX_PATCH_OPERATIONS: usize = 1_000;
/// Maximum number of encoded animation frames. This bounds per-frame setup work.
pub const MAX_ANIMATION_FRAMES: u64 = 300;
/// Maximum aggregate raster work for one animation, measured in output pixels.
pub const MAX_ANIMATION_PIXELS: u64 = 64 * 1024 * 1024;
/// Maximum byte length of a user-supplied post-process effect shader.
pub const MAX_EFFECT_SHADER_BYTES: usize = 64 * 1024;

pub type Color = [f32; 4];

/// A node's fill: either a solid color or a gradient.
///
/// The wire/JSON shape is either a plain 4-element `[r, g, b, a]` array
/// (a solid fill -- identical to how every `color` field worked before
/// this type existed) or a tagged object describing a gradient, e.g.
/// `{"kind": "linear_gradient", "from": [...], "to": [...],
/// "angle_degrees": 0.0}` or `{"kind": "radial_gradient", "center": [...],
/// "edge": [...]}`. `#[serde(untagged)]` tries `Solid` (a bare 4-element
/// array) first and falls back to `Gradient` (a tagged object) -- so every
/// existing scene document's `"color": [r, g, b, a]` continues to
/// deserialize exactly as before, into `FillV1::Solid`.
///
/// Every `NodeKindV1` variant that used to carry a plain `color: Color`
/// field now carries `fill: FillV1` instead, but keeps the *JSON key* named
/// `color` via `#[serde(rename = "color")]` (see e.g. `NodeKindV1::Rect`).
/// This is a deliberate choice: it gives the Rust API the more accurate
/// `fill` name (a fill is not always "one color") while keeping every
/// existing scene document -- including this crate's own fixtures and the
/// renderer's checked-in golden `*.scene.json` files -- byte-for-byte
/// unchanged and valid, since the wire shape and key are identical to
/// before for the solid case. Renaming the wire key to `fill` as well would
/// have bought nothing (both names are equally clear on the wire) while
/// breaking every existing scene document and fixture for no benefit.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum FillV1 {
    Solid(Color),
    Gradient(GradientV1),
}

impl FillV1 {
    fn validate(&self) -> Result<(), SceneValidationError> {
        match self {
            Self::Solid(color) => validate_color(*color),
            Self::Gradient(GradientV1::LinearGradient {
                from,
                to,
                angle_degrees,
            }) => {
                validate_color(*from)?;
                validate_color(*to)?;
                if !angle_degrees.is_finite() {
                    return Err(SceneValidationError::InvalidGradientAngle);
                }
                Ok(())
            }
            Self::Gradient(GradientV1::RadialGradient { center, edge }) => {
                validate_color(*center)?;
                validate_color(*edge)
            }
        }
    }

    /// Resolves this fill to a single representative flat color, for
    /// contexts that don't implement true gradient rendering (this crate's
    /// renderer scopes real per-pixel gradient rendering to `Rect`/
    /// `Ellipse` only; see its doc comments). A solid fill resolves to
    /// itself; a gradient resolves to the 50/50 midpoint blend of its two
    /// stops -- a reasonable flat approximation that reflects both ends
    /// rather than silently picking just one.
    pub fn resolve_solid(&self) -> Color {
        match self {
            Self::Solid(color) => *color,
            Self::Gradient(GradientV1::LinearGradient { from, to, .. }) => midpoint(*from, *to),
            Self::Gradient(GradientV1::RadialGradient { center, edge }) => midpoint(*center, *edge),
        }
    }

    /// Multiplies the alpha channel of every color this fill carries by
    /// `opacity`: for `Solid`, exactly the existing `color[3] *= opacity`
    /// behavior; for a gradient, both stops are scaled so an
    /// animated-opacity gradient fades as a whole.
    pub fn multiply_alpha(&mut self, opacity: f32) {
        match self {
            Self::Solid(color) => color[3] *= opacity,
            Self::Gradient(GradientV1::LinearGradient { from, to, .. }) => {
                from[3] *= opacity;
                to[3] *= opacity;
            }
            Self::Gradient(GradientV1::RadialGradient { center, edge }) => {
                center[3] *= opacity;
                edge[3] *= opacity;
            }
        }
    }

    /// Replaces this fill with a solid color in place -- used by legacy,
    /// flat-`Color`-only keyframe animation (`AnimatedPropertyV1::Color`).
    pub fn set_solid(&mut self, color: Color) {
        *self = Self::Solid(color);
    }

    /// `Some(color)` iff this fill is currently solid.
    pub fn as_solid(&self) -> Option<Color> {
        match self {
            Self::Solid(color) => Some(*color),
            Self::Gradient(_) => None,
        }
    }
}

fn midpoint(a: Color, b: Color) -> Color {
    [
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
        (a[3] + b[3]) / 2.0,
    ]
}

/// The object form of `FillV1`. `deny_unknown_fields` here means a gradient
/// object with an unrecognized field (e.g. a typo) is rejected rather than
/// silently ignored, consistent with the rest of this schema.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GradientV1 {
    LinearGradient {
        from: Color,
        to: Color,
        /// Gradient direction, in degrees, measured the same way angles are
        /// conventionally authored for CSS-style linear gradients: `0.0`
        /// points along `+x` (left-to-right); increasing values rotate
        /// clockwise in this schema's y-down scene-pixel space.
        angle_degrees: f32,
    },
    RadialGradient {
        center: Color,
        edge: Color,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SceneV1 {
    pub version: String,
    pub canvas: CanvasV1,
    #[serde(default)]
    pub nodes: Vec<NodeV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeline: Option<TimelineV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<EffectV1>,
}

impl SceneV1 {
    pub fn validate(&self) -> Result<(), SceneValidationError> {
        if self.version != SCENE_VERSION_V1 {
            return Err(SceneValidationError::UnsupportedVersion(
                self.version.clone(),
            ));
        }
        self.canvas.validate()?;
        if self.nodes.len() > MAX_NODES {
            return Err(SceneValidationError::TooManyNodes {
                actual: self.nodes.len(),
                maximum: MAX_NODES,
            });
        }

        let mut ids = HashSet::with_capacity(self.nodes.len());
        for node in &self.nodes {
            if node.id.trim().is_empty() {
                return Err(SceneValidationError::EmptyNodeId);
            }
            if !ids.insert(node.id.as_str()) {
                return Err(SceneValidationError::DuplicateNodeId(node.id.clone()));
            }
            node.validate()?;
        }
        if let Some(timeline) = &self.timeline {
            timeline.validate(&ids, self.canvas.width, self.canvas.height)?;
        }
        if let Some(effect) = &self.effect {
            effect.validate()?;
        }
        Ok(())
    }
}

/// A scene-level, full-canvas WGSL post-process effect.
///
/// `shader` is WGSL source that must define exactly one function with the
/// signature `fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32>`. The
/// renderer wraps this pure per-pixel color transform in an internal,
/// fixed template (vertex stage, texture bindings) so authors never touch
/// bindings or vertex data directly; that template is the entire security
/// boundary between untrusted scene input and the GPU pipeline.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EffectV1 {
    pub shader: String,
}

impl EffectV1 {
    fn validate(&self) -> Result<(), SceneValidationError> {
        if self.shader.is_empty() {
            return Err(SceneValidationError::EmptyEffectShader);
        }
        if self.shader.len() > MAX_EFFECT_SHADER_BYTES {
            return Err(SceneValidationError::EffectShaderTooLarge {
                actual: self.shader.len(),
                maximum: MAX_EFFECT_SHADER_BYTES,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanvasV1 {
    pub width: u32,
    pub height: u32,
    #[serde(default = "transparent")]
    pub background: Color,
}

impl CanvasV1 {
    fn validate(&self) -> Result<(), SceneValidationError> {
        if self.width == 0 || self.height == 0 {
            return Err(SceneValidationError::InvalidCanvasDimensions);
        }
        if self.width > MAX_CANVAS_DIMENSION || self.height > MAX_CANVAS_DIMENSION {
            return Err(SceneValidationError::CanvasTooLarge {
                width: self.width,
                height: self.height,
                maximum: MAX_CANVAS_DIMENSION,
            });
        }
        validate_color(self.background)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NodeV1 {
    pub id: String,
    /// Additive x/y offset applied on top of this node's own coordinates at
    /// render time (e.g. for `Rect`, effectively `(x + translate[0], y +
    /// translate[1])`). Uniform across every node kind -- unlike `x`/`y`/
    /// `cx`/`cy`/etc., which differ per shape -- so it lives here on `NodeV1`
    /// rather than inside `NodeKindV1`. Defaults to `[0.0, 0.0]` so every
    /// existing scene document with no `translate` field is unaffected.
    #[serde(default)]
    pub translate: [f32; 2],
    #[serde(flatten)]
    pub kind: NodeKindV1,
}

impl NodeV1 {
    fn validate(&self) -> Result<(), SceneValidationError> {
        self.kind.validate()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NodeKindV1 {
    Rect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        /// Corner rounding radius, in scene-pixel units. Defaults to `0.0`
        /// (a plain right-angle rect, byte-identical to every scene
        /// authored before this field existed) when omitted from JSON.
        /// Validated to be non-negative and no larger than half of the
        /// smaller of `width`/`height` -- rejected rather than silently
        /// clamped, consistent with this schema's existing "reject invalid
        /// geometry" convention for `width`/`height`/`thickness` (see
        /// `NodeKindV1::validate`).
        #[serde(default)]
        corner_radius: f32,
        #[serde(rename = "color")]
        fill: FillV1,
    },
    Ellipse {
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        #[serde(rename = "color")]
        fill: FillV1,
    },
    Line {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        thickness: f32,
        #[serde(rename = "color")]
        fill: FillV1,
    },
    Path {
        points: Vec<PointV1>,
        #[serde(rename = "color")]
        fill: FillV1,
    },
    Text {
        x: f32,
        y: f32,
        text: String,
        size: f32,
        #[serde(rename = "color")]
        fill: FillV1,
    },
    Image {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        source: String,
    },
}

impl NodeKindV1 {
    fn validate(&self) -> Result<(), SceneValidationError> {
        match self {
            Self::Rect {
                width,
                height,
                corner_radius,
                fill,
                ..
            } => {
                validate_positive(*width, "width")?;
                validate_positive(*height, "height")?;
                if !corner_radius.is_finite() || *corner_radius < 0.0 {
                    return Err(SceneValidationError::InvalidCornerRadius);
                }
                if *corner_radius > width.min(*height) / 2.0 {
                    return Err(SceneValidationError::InvalidCornerRadius);
                }
                fill.validate()?;
            }
            Self::Image { width, height, .. } => {
                validate_positive(*width, "width")?;
                validate_positive(*height, "height")?;
            }
            Self::Ellipse { rx, ry, fill, .. } => {
                validate_positive(*rx, "rx")?;
                validate_positive(*ry, "ry")?;
                fill.validate()?;
            }
            Self::Line {
                thickness, fill, ..
            } => {
                validate_positive(*thickness, "thickness")?;
                fill.validate()?;
            }
            Self::Path { points, fill } => {
                if points.len() < 3 {
                    return Err(SceneValidationError::InvalidPath);
                }
                if points.len() > MAX_PATH_POINTS {
                    return Err(SceneValidationError::TooManyPathPoints {
                        actual: points.len(),
                        maximum: MAX_PATH_POINTS,
                    });
                }
                fill.validate()?;
            }
            Self::Text {
                text, size, fill, ..
            } => {
                if text.is_empty() {
                    return Err(SceneValidationError::EmptyText);
                }
                validate_positive(*size, "size")?;
                fill.validate()?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PointV1 {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TimelineV1 {
    pub fps: u16,
    pub duration_ms: u32,
    #[serde(default)]
    pub keyframes: Vec<KeyframeV1>,
}

impl TimelineV1 {
    fn validate(
        &self,
        node_ids: &HashSet<&str>,
        canvas_width: u32,
        canvas_height: u32,
    ) -> Result<(), SceneValidationError> {
        if self.fps == 0 || self.fps > 60 || self.duration_ms == 0 || self.duration_ms > 10_000 {
            return Err(SceneValidationError::InvalidTimeline);
        }
        let frame_count = (u64::from(self.duration_ms) * u64::from(self.fps)).div_ceil(1_000);
        if frame_count > MAX_ANIMATION_FRAMES {
            return Err(SceneValidationError::TooManyAnimationFrames {
                actual: frame_count,
                maximum: MAX_ANIMATION_FRAMES,
            });
        }
        let pixel_count = u64::from(canvas_width) * u64::from(canvas_height) * frame_count;
        if pixel_count > MAX_ANIMATION_PIXELS {
            return Err(SceneValidationError::TooManyAnimationPixels {
                actual: pixel_count,
                maximum: MAX_ANIMATION_PIXELS,
            });
        }
        for frame in &self.keyframes {
            if frame.at_ms > self.duration_ms {
                return Err(SceneValidationError::InvalidTimeline);
            }
            if frame.target.trim().is_empty() || !node_ids.contains(frame.target.as_str()) {
                return Err(SceneValidationError::UnknownKeyframeTarget(
                    frame.target.clone(),
                ));
            }
            match frame.property {
                AnimatedPropertyV1::Opacity(value) => validate_unit_interval(value)?,
                AnimatedPropertyV1::Color(color) => validate_color(color)?,
                AnimatedPropertyV1::Translate(value) => validate_finite_pair(value)?,
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KeyframeV1 {
    pub at_ms: u32,
    pub target: String,
    pub property: AnimatedPropertyV1,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum AnimatedPropertyV1 {
    Opacity(f32),
    Color(Color),
    Translate([f32; 2]),
}

/// A bounded, typed scene mutation. Applying all operations is atomic.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScenePatchV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_revision: Option<u64>,
    pub operations: Vec<PatchOperationV1>,
}

impl ScenePatchV1 {
    pub fn validate(&self) -> Result<(), SceneValidationError> {
        if self.operations.is_empty() || self.operations.len() > MAX_PATCH_OPERATIONS {
            return Err(SceneValidationError::InvalidPatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum PatchOperationV1 {
    SetCanvas { canvas: CanvasV1 },
    UpsertNode { node: NodeV1 },
    RemoveNode { id: String },
    SetTimeline { timeline: TimelineV1 },
    ClearTimeline,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum SceneValidationError {
    #[error("unsupported scene version: {0}")]
    UnsupportedVersion(String),
    #[error("canvas dimensions must be greater than zero")]
    InvalidCanvasDimensions,
    #[error("canvas {width}x{height} exceeds {maximum}px per dimension")]
    CanvasTooLarge {
        width: u32,
        height: u32,
        maximum: u32,
    },
    #[error("scene has {actual} nodes; maximum is {maximum}")]
    TooManyNodes { actual: usize, maximum: usize },
    #[error("node IDs must not be empty")]
    EmptyNodeId,
    #[error("duplicate node ID: {0}")]
    DuplicateNodeId(String),
    #[error("{0} must be finite and greater than zero")]
    InvalidPositiveValue(&'static str),
    #[error("colors must contain finite values from 0.0 through 1.0")]
    InvalidColor,
    #[error("gradient angle_degrees must be finite")]
    InvalidGradientAngle,
    #[error(
        "corner_radius must be finite, non-negative, and no larger than half of the smaller of width/height"
    )]
    InvalidCornerRadius,
    #[error("translate values must be finite")]
    InvalidTranslate,
    #[error("paths require at least three points")]
    InvalidPath,
    #[error("path has {actual} points; maximum is {maximum}")]
    TooManyPathPoints { actual: usize, maximum: usize },
    #[error("text nodes must not be empty")]
    EmptyText,
    #[error("patch must contain 1 through {MAX_PATCH_OPERATIONS} operations")]
    InvalidPatch,
    #[error("keyframe target does not identify a scene node: {0}")]
    UnknownKeyframeTarget(String),
    #[error("animation has {actual} frames; maximum is {maximum}")]
    TooManyAnimationFrames { actual: u64, maximum: u64 },
    #[error("animation raster work is {actual} pixels; maximum is {maximum}")]
    TooManyAnimationPixels { actual: u64, maximum: u64 },
    #[error("timeline must use 1-60 FPS, last no more than 10 seconds, and contain valid frames")]
    InvalidTimeline,
    #[error("effect shader must not be empty")]
    EmptyEffectShader,
    #[error("effect shader is {actual} bytes; maximum is {maximum}")]
    EffectShaderTooLarge { actual: usize, maximum: usize },
}

fn transparent() -> Color {
    [0.0, 0.0, 0.0, 0.0]
}

fn validate_positive(value: f32, name: &'static str) -> Result<(), SceneValidationError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(SceneValidationError::InvalidPositiveValue(name));
    }
    Ok(())
}

fn validate_color(color: Color) -> Result<(), SceneValidationError> {
    if color
        .iter()
        .all(|value| validate_unit_interval(*value).is_ok())
    {
        Ok(())
    } else {
        Err(SceneValidationError::InvalidColor)
    }
}

fn validate_unit_interval(value: f32) -> Result<(), SceneValidationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(SceneValidationError::InvalidColor)
    }
}

/// Validates an unconstrained `[f32; 2]` offset (e.g. `translate`): both
/// components must be finite, but -- unlike opacity or color -- there is no
/// range restriction, consistent with how a `Rect`'s `x`/`y` aren't
/// range-restricted today either.
fn validate_finite_pair(value: [f32; 2]) -> Result<(), SceneValidationError> {
    if value.iter().all(|component| component.is_finite()) {
        Ok(())
    } else {
        Err(SceneValidationError::InvalidTranslate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> SceneV1 {
        SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: transparent(),
            },
            nodes: vec![NodeV1 {
                id: "box".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0; 4]),
                },
            }],
            timeline: None,
            effect: None,
        }
    }

    #[test]
    fn validates_a_minimal_scene() {
        assert_eq!(scene().validate(), Ok(()));
    }

    #[test]
    fn rejects_duplicate_ids() {
        let mut value = scene();
        value.nodes.push(value.nodes[0].clone());
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::DuplicateNodeId(_))
        ));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut value = scene();
        value.version = "v0".into();
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::UnsupportedVersion(_))
        ));
    }

    #[test]
    fn rejects_a_misplaced_top_level_background_field_instead_of_ignoring_it() {
        // `background` belongs under `canvas`, not at the scene's top level.
        // Before `deny_unknown_fields`, a misplaced field like this was
        // silently dropped by serde: the scene deserialized and validated
        // successfully, but rendered with the default (transparent)
        // background instead of the caller's intended color, with no
        // diagnostic anywhere pointing at the mistake.
        let json = r#"{
            "version": "renderer.scene.v1",
            "canvas": {"width": 64, "height": 64},
            "background": {"kind": "solid", "color": [0.039, 0.067, 0.157, 1.0]},
            "nodes": []
        }"#;
        let error = serde_json::from_str::<SceneV1>(json)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("background"),
            "expected the unknown `background` field to be named in the error, got: {error}"
        );
    }

    #[test]
    fn rejects_unknown_fields_nested_under_canvas() {
        let json = r#"{
            "version": "renderer.scene.v1",
            "canvas": {"width": 64, "height": 64, "colour": [1.0, 1.0, 1.0, 1.0]},
            "nodes": []
        }"#;
        assert!(serde_json::from_str::<SceneV1>(json).is_err());
    }

    #[test]
    fn rejects_unknown_fields_on_a_node() {
        let json = r#"{
            "version": "renderer.scene.v1",
            "canvas": {"width": 64, "height": 64},
            "nodes": [{
                "id": "box", "kind": "rect",
                "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0,
                "color": [1.0, 1.0, 1.0, 1.0],
                "strokeWidth": 2.0
            }]
        }"#;
        assert!(serde_json::from_str::<SceneV1>(json).is_err());
    }

    #[test]
    fn validates_every_supported_node_kind() {
        let mut value = scene();
        value.nodes = vec![
            NodeV1 {
                id: "ellipse".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Ellipse {
                    cx: 4.0,
                    cy: 4.0,
                    rx: 2.0,
                    ry: 2.0,
                    fill: FillV1::Solid([0.0; 4]),
                },
            },
            NodeV1 {
                id: "line".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Line {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 3.0,
                    y2: 3.0,
                    thickness: 1.0,
                    fill: FillV1::Solid([0.0; 4]),
                },
            },
            NodeV1 {
                id: "path".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Path {
                    points: vec![
                        PointV1 { x: 0.0, y: 0.0 },
                        PointV1 { x: 1.0, y: 0.0 },
                        PointV1 { x: 0.0, y: 1.0 },
                    ],
                    fill: FillV1::Solid([0.0; 4]),
                },
            },
            NodeV1 {
                id: "text".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Text {
                    x: 0.0,
                    y: 0.0,
                    text: "ok".into(),
                    size: 1.0,
                    fill: FillV1::Solid([0.0; 4]),
                },
            },
            NodeV1 {
                id: "image".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Image {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                    source: "asset.png".into(),
                },
            },
        ];
        assert_eq!(value.validate(), Ok(()));
    }

    #[test]
    fn rejects_invalid_canvas_node_and_timeline_values() {
        let mut value = scene();
        value.canvas.width = 0;
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::InvalidCanvasDimensions)
        );

        let mut value = scene();
        value.canvas.width = MAX_CANVAS_DIMENSION + 1;
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::CanvasTooLarge { .. })
        ));

        let mut value = scene();
        value.canvas.background = [2.0, 0.0, 0.0, 0.0];
        assert_eq!(value.validate(), Err(SceneValidationError::InvalidColor));

        let mut value = scene();
        value.nodes = vec![value.nodes[0].clone(); MAX_NODES + 1];
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::TooManyNodes { .. })
        ));

        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: "box".into(),
                property: AnimatedPropertyV1::Color([2.0, 0.0, 0.0, 1.0]),
            }],
        });
        assert_eq!(value.validate(), Err(SceneValidationError::InvalidColor));

        let mut value = scene();
        value.nodes[0].id.clear();
        assert_eq!(value.validate(), Err(SceneValidationError::EmptyNodeId));

        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Rect {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 1.0,
            corner_radius: 0.0,
            fill: FillV1::Solid([0.0; 4]),
        };
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::InvalidPositiveValue("width"))
        );

        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Path {
            points: vec![],
            fill: FillV1::Solid([0.0; 4]),
        };
        assert_eq!(value.validate(), Err(SceneValidationError::InvalidPath));

        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Path {
            points: vec![PointV1 { x: 0.0, y: 0.0 }; MAX_PATH_POINTS + 1],
            fill: FillV1::Solid([0.0; 4]),
        };
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::TooManyPathPoints { .. })
        ));

        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Text {
            x: 0.0,
            y: 0.0,
            text: String::new(),
            size: 1.0,
            fill: FillV1::Solid([0.0; 4]),
        };
        assert_eq!(value.validate(), Err(SceneValidationError::EmptyText));

        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 0,
            duration_ms: 1,
            keyframes: vec![],
        });
        assert_eq!(value.validate(), Err(SceneValidationError::InvalidTimeline));

        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: "box".into(),
                property: AnimatedPropertyV1::Opacity(2.0),
            }],
        });
        assert_eq!(value.validate(), Err(SceneValidationError::InvalidColor));

        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 2,
                target: "box".into(),
                property: AnimatedPropertyV1::Opacity(1.0),
            }],
        });
        assert_eq!(value.validate(), Err(SceneValidationError::InvalidTimeline));
    }

    #[test]
    fn rejects_unknown_keyframe_targets_and_excessive_animation_work() {
        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: String::new(),
                property: AnimatedPropertyV1::Opacity(1.0),
            }],
        });
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::UnknownKeyframeTarget(_))
        ));

        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: "missing".into(),
                property: AnimatedPropertyV1::Opacity(1.0),
            }],
        });
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::UnknownKeyframeTarget(_))
        ));

        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 60,
            duration_ms: 10_000,
            keyframes: vec![],
        });
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::TooManyAnimationFrames { .. })
        ));

        let mut value = scene();
        value.canvas.width = MAX_CANVAS_DIMENSION;
        value.canvas.height = MAX_CANVAS_DIMENSION;
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 5_000,
            keyframes: vec![],
        });
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::TooManyAnimationPixels { .. })
        ));
    }

    #[test]
    fn validates_bounded_nonempty_patches() {
        assert_eq!(
            ScenePatchV1 {
                expected_revision: None,
                operations: vec![],
            }
            .validate(),
            Err(SceneValidationError::InvalidPatch)
        );
        assert_eq!(
            ScenePatchV1 {
                expected_revision: Some(1),
                operations: vec![PatchOperationV1::ClearTimeline],
            }
            .validate(),
            Ok(())
        );
    }

    #[test]
    fn rejects_an_empty_effect_shader() {
        let mut value = scene();
        value.effect = Some(EffectV1 {
            shader: String::new(),
        });
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::EmptyEffectShader)
        );
    }

    #[test]
    fn rejects_an_oversized_effect_shader() {
        let mut value = scene();
        value.effect = Some(EffectV1 {
            shader: "a".repeat(MAX_EFFECT_SHADER_BYTES + 1),
        });
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::EffectShaderTooLarge { .. })
        ));
    }

    #[test]
    fn accepts_a_valid_effect_shader() {
        let mut value = scene();
        value.effect = Some(EffectV1 {
            shader: "fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32> { return color; }"
                .into(),
        });
        assert_eq!(value.validate(), Ok(()));
    }

    #[test]
    fn accepts_a_valid_translate_keyframe() {
        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: "box".into(),
                property: AnimatedPropertyV1::Translate([12.5, -7.0]),
            }],
        });
        assert_eq!(value.validate(), Ok(()));
    }

    #[test]
    fn rejects_a_non_finite_translate_keyframe_value() {
        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: "box".into(),
                property: AnimatedPropertyV1::Translate([f32::NAN, 0.0]),
            }],
        });
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::InvalidTranslate)
        );

        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: "box".into(),
                property: AnimatedPropertyV1::Translate([0.0, f32::INFINITY]),
            }],
        });
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::InvalidTranslate)
        );
    }

    #[test]
    fn node_translate_defaults_to_zero_when_absent_from_json() {
        let json = r#"{
            "version": "renderer.scene.v1",
            "canvas": {"width": 64, "height": 64},
            "nodes": [{
                "id": "box", "kind": "rect",
                "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0,
                "color": [1.0, 1.0, 1.0, 1.0]
            }]
        }"#;
        let parsed: SceneV1 = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.nodes[0].translate, [0.0, 0.0]);
    }

    #[test]
    fn rect_corner_radius_defaults_to_zero_when_absent_from_json() {
        let json = r#"{
            "version": "renderer.scene.v1",
            "canvas": {"width": 64, "height": 64},
            "nodes": [{
                "id": "box", "kind": "rect",
                "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0,
                "color": [1.0, 1.0, 1.0, 1.0]
            }]
        }"#;
        let parsed: SceneV1 = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.nodes[0].kind,
            NodeKindV1::Rect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                corner_radius: 0.0,
                fill: FillV1::Solid([1.0, 1.0, 1.0, 1.0]),
            }
        );
        assert_eq!(parsed.validate(), Ok(()));
    }

    #[test]
    fn accepts_a_valid_corner_radius() {
        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 6.0,
            corner_radius: 3.0,
            fill: FillV1::Solid([1.0; 4]),
        };
        assert_eq!(value.validate(), Ok(()));
    }

    #[test]
    fn rejects_a_negative_corner_radius() {
        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
            corner_radius: -1.0,
            fill: FillV1::Solid([1.0; 4]),
        };
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::InvalidCornerRadius)
        );
    }

    #[test]
    fn rejects_a_non_finite_corner_radius() {
        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
            corner_radius: f32::NAN,
            fill: FillV1::Solid([1.0; 4]),
        };
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::InvalidCornerRadius)
        );
    }

    #[test]
    fn rejects_a_corner_radius_larger_than_half_the_smaller_dimension() {
        let mut value = scene();
        // width=10 height=6 -> max valid radius is 3.0 (half of the
        // smaller dimension, height).
        value.nodes[0].kind = NodeKindV1::Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 6.0,
            corner_radius: 3.0001,
            fill: FillV1::Solid([1.0; 4]),
        };
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::InvalidCornerRadius)
        );
    }

    #[test]
    fn fill_v1_round_trips_a_plain_solid_array() {
        let json = r#"[1.0, 0.5, 0.25, 1.0]"#;
        let fill: FillV1 = serde_json::from_str(json).unwrap();
        assert_eq!(fill, FillV1::Solid([1.0, 0.5, 0.25, 1.0]));
        let re_encoded = serde_json::to_string(&fill).unwrap();
        assert_eq!(re_encoded, "[1.0,0.5,0.25,1.0]");
    }

    #[test]
    fn fill_v1_round_trips_a_linear_gradient_object() {
        let json = r#"{"kind":"linear_gradient","from":[1.0,0.0,0.0,1.0],"to":[0.0,0.0,1.0,1.0],"angle_degrees":45.0}"#;
        let fill: FillV1 = serde_json::from_str(json).unwrap();
        assert_eq!(
            fill,
            FillV1::Gradient(GradientV1::LinearGradient {
                from: [1.0, 0.0, 0.0, 1.0],
                to: [0.0, 0.0, 1.0, 1.0],
                angle_degrees: 45.0,
            })
        );
        let re_encoded = serde_json::to_string(&fill).unwrap();
        let re_decoded: FillV1 = serde_json::from_str(&re_encoded).unwrap();
        assert_eq!(re_decoded, fill);
    }

    #[test]
    fn fill_v1_round_trips_a_radial_gradient_object() {
        let json =
            r#"{"kind":"radial_gradient","center":[1.0,1.0,1.0,1.0],"edge":[0.0,0.0,0.0,1.0]}"#;
        let fill: FillV1 = serde_json::from_str(json).unwrap();
        assert_eq!(
            fill,
            FillV1::Gradient(GradientV1::RadialGradient {
                center: [1.0, 1.0, 1.0, 1.0],
                edge: [0.0, 0.0, 0.0, 1.0],
            })
        );
        let re_encoded = serde_json::to_string(&fill).unwrap();
        let re_decoded: FillV1 = serde_json::from_str(&re_encoded).unwrap();
        assert_eq!(re_decoded, fill);
    }

    #[test]
    fn fill_v1_rejects_unknown_fields_in_a_gradient_object() {
        let json = r#"{"kind":"linear_gradient","from":[1.0,0.0,0.0,1.0],"to":[0.0,0.0,1.0,1.0],"angle_degrees":0.0,"bogus":1.0}"#;
        assert!(serde_json::from_str::<FillV1>(json).is_err());
    }

    #[test]
    fn fill_v1_rejects_malformed_input_that_is_neither_array_nor_gradient_object() {
        assert!(serde_json::from_str::<FillV1>(r#"{"kind":"not_a_real_kind"}"#).is_err());
        assert!(serde_json::from_str::<FillV1>(r#"[1.0, 2.0]"#).is_err());
        assert!(serde_json::from_str::<FillV1>(r#""red""#).is_err());
    }

    #[test]
    fn a_rect_with_a_gradient_fill_round_trips_through_a_full_scene_and_validates() {
        let json = r#"{
            "version": "renderer.scene.v1",
            "canvas": {"width": 64, "height": 64},
            "nodes": [{
                "id": "box", "kind": "rect",
                "x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0,
                "corner_radius": 2.0,
                "color": {"kind": "linear_gradient", "from": [1.0, 0.0, 0.0, 1.0], "to": [0.0, 0.0, 1.0, 1.0], "angle_degrees": 0.0}
            }]
        }"#;
        let parsed: SceneV1 = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.validate(), Ok(()));
        assert_eq!(
            parsed.nodes[0].kind,
            NodeKindV1::Rect {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
                corner_radius: 2.0,
                fill: FillV1::Gradient(GradientV1::LinearGradient {
                    from: [1.0, 0.0, 0.0, 1.0],
                    to: [0.0, 0.0, 1.0, 1.0],
                    angle_degrees: 0.0,
                }),
            }
        );
    }

    #[test]
    fn rejects_a_gradient_with_an_out_of_range_color_component() {
        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
            corner_radius: 0.0,
            fill: FillV1::Gradient(GradientV1::LinearGradient {
                from: [2.0, 0.0, 0.0, 1.0],
                to: [0.0, 0.0, 1.0, 1.0],
                angle_degrees: 0.0,
            }),
        };
        assert_eq!(value.validate(), Err(SceneValidationError::InvalidColor));
    }

    #[test]
    fn rejects_a_gradient_with_a_non_finite_angle() {
        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
            corner_radius: 0.0,
            fill: FillV1::Gradient(GradientV1::LinearGradient {
                from: [1.0, 0.0, 0.0, 1.0],
                to: [0.0, 0.0, 1.0, 1.0],
                angle_degrees: f32::NAN,
            }),
        };
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::InvalidGradientAngle)
        );
    }

    #[test]
    fn rejects_a_radial_gradient_with_an_out_of_range_color_component() {
        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
            corner_radius: 0.0,
            fill: FillV1::Gradient(GradientV1::RadialGradient {
                center: [1.0, 1.0, 1.0, 1.0],
                edge: [0.0, 0.0, 0.0, -1.0],
            }),
        };
        assert_eq!(value.validate(), Err(SceneValidationError::InvalidColor));
    }

    #[test]
    fn fill_v1_resolve_solid_returns_the_color_for_a_solid_fill() {
        assert_eq!(
            FillV1::Solid([1.0, 0.5, 0.25, 1.0]).resolve_solid(),
            [1.0, 0.5, 0.25, 1.0]
        );
    }

    #[test]
    fn fill_v1_resolve_solid_returns_the_midpoint_of_a_gradient() {
        let fill = FillV1::Gradient(GradientV1::LinearGradient {
            from: [0.0, 0.0, 0.0, 0.0],
            to: [1.0, 1.0, 1.0, 1.0],
            angle_degrees: 0.0,
        });
        assert_eq!(fill.resolve_solid(), [0.5, 0.5, 0.5, 0.5]);

        let fill = FillV1::Gradient(GradientV1::RadialGradient {
            center: [1.0, 0.0, 0.0, 1.0],
            edge: [0.0, 1.0, 0.0, 0.0],
        });
        assert_eq!(fill.resolve_solid(), [0.5, 0.5, 0.0, 0.5]);
    }

    #[test]
    fn fill_v1_multiply_alpha_scales_every_stop_a_gradient_carries() {
        let mut fill = FillV1::Gradient(GradientV1::LinearGradient {
            from: [1.0, 0.0, 0.0, 1.0],
            to: [0.0, 0.0, 1.0, 0.5],
            angle_degrees: 0.0,
        });
        fill.multiply_alpha(0.5);
        assert_eq!(
            fill,
            FillV1::Gradient(GradientV1::LinearGradient {
                from: [1.0, 0.0, 0.0, 0.5],
                to: [0.0, 0.0, 1.0, 0.25],
                angle_degrees: 0.0,
            })
        );
    }

    #[test]
    fn fill_v1_as_solid_and_set_solid_round_trip() {
        let mut fill = FillV1::Solid([1.0, 0.0, 0.0, 1.0]);
        assert_eq!(fill.as_solid(), Some([1.0, 0.0, 0.0, 1.0]));
        fill.set_solid([0.0, 1.0, 0.0, 1.0]);
        assert_eq!(fill, FillV1::Solid([0.0, 1.0, 0.0, 1.0]));

        let gradient = FillV1::Gradient(GradientV1::RadialGradient {
            center: [1.0; 4],
            edge: [0.0; 4],
        });
        assert_eq!(gradient.as_solid(), None);
    }
}
