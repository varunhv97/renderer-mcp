use crate::*;

/// A complete, renderable scene: a canvas, a flat list of nodes (no
/// grouping/hierarchy), and an optional animation timeline and post-process
/// effect. This is the unit of storage in the daemon's named-scene store and
/// the document CLI/MCP callers author or generate.
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
    /// Checks every structural and resource-limit invariant this scene must
    /// satisfy before it can be handed to the renderer: version, canvas
    /// bounds, node count/uniqueness, and (if present) timeline/effect
    /// validity. The renderer itself does not re-validate, so every caller
    /// of untrusted scene input must run it through here first.
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
    pub(crate) fn validate(&self) -> Result<(), SceneValidationError> {
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

/// The scene's output surface: pixel dimensions and background color.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanvasV1 {
    pub width: u32,
    pub height: u32,
    #[serde(default = "transparent")]
    pub background: Color,
}

impl CanvasV1 {
    pub(crate) fn validate(&self) -> Result<(), SceneValidationError> {
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

pub(crate) fn transparent() -> Color {
    [0.0, 0.0, 0.0, 0.0]
}
