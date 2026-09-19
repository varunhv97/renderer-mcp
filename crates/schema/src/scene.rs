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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;

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
}
