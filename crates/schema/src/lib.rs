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

pub type Color = [f32; 4];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SceneV1 {
    pub version: String,
    pub canvas: CanvasV1,
    #[serde(default)]
    pub nodes: Vec<NodeV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeline: Option<TimelineV1>,
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
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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
    #[serde(flatten)]
    pub kind: NodeKindV1,
}

impl NodeV1 {
    fn validate(&self) -> Result<(), SceneValidationError> {
        self.kind.validate()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NodeKindV1 {
    Rect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: Color,
    },
    Ellipse {
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        color: Color,
    },
    Line {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        thickness: f32,
        color: Color,
    },
    Path {
        points: Vec<PointV1>,
        color: Color,
    },
    Text {
        x: f32,
        y: f32,
        text: String,
        size: f32,
        color: Color,
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
                color,
                ..
            } => {
                validate_positive(*width, "width")?;
                validate_positive(*height, "height")?;
                validate_color(*color)?;
            }
            Self::Image { width, height, .. } => {
                validate_positive(*width, "width")?;
                validate_positive(*height, "height")?;
            }
            Self::Ellipse { rx, ry, color, .. } => {
                validate_positive(*rx, "rx")?;
                validate_positive(*ry, "ry")?;
                validate_color(*color)?;
            }
            Self::Line {
                thickness, color, ..
            } => {
                validate_positive(*thickness, "thickness")?;
                validate_color(*color)?;
            }
            Self::Path { points, color } => {
                if points.len() < 3 {
                    return Err(SceneValidationError::InvalidPath);
                }
                if points.len() > MAX_PATH_POINTS {
                    return Err(SceneValidationError::TooManyPathPoints {
                        actual: points.len(),
                        maximum: MAX_PATH_POINTS,
                    });
                }
                validate_color(*color)?;
            }
            Self::Text {
                text, size, color, ..
            } => {
                if text.is_empty() {
                    return Err(SceneValidationError::EmptyText);
                }
                validate_positive(*size, "size")?;
                validate_color(*color)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct PointV1 {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct KeyframeV1 {
    pub at_ms: u32,
    pub target: String,
    pub property: AnimatedPropertyV1,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum AnimatedPropertyV1 {
    Opacity(f32),
    Color(Color),
}

/// A bounded, typed scene mutation. Applying all operations is atomic.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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
#[serde(tag = "op", rename_all = "snake_case")]
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
                kind: NodeKindV1::Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                    color: [1.0; 4],
                },
            }],
            timeline: None,
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
    fn validates_every_supported_node_kind() {
        let mut value = scene();
        value.nodes = vec![
            NodeV1 {
                id: "ellipse".into(),
                kind: NodeKindV1::Ellipse {
                    cx: 4.0,
                    cy: 4.0,
                    rx: 2.0,
                    ry: 2.0,
                    color: [0.0; 4],
                },
            },
            NodeV1 {
                id: "line".into(),
                kind: NodeKindV1::Line {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 3.0,
                    y2: 3.0,
                    thickness: 1.0,
                    color: [0.0; 4],
                },
            },
            NodeV1 {
                id: "path".into(),
                kind: NodeKindV1::Path {
                    points: vec![
                        PointV1 { x: 0.0, y: 0.0 },
                        PointV1 { x: 1.0, y: 0.0 },
                        PointV1 { x: 0.0, y: 1.0 },
                    ],
                    color: [0.0; 4],
                },
            },
            NodeV1 {
                id: "text".into(),
                kind: NodeKindV1::Text {
                    x: 0.0,
                    y: 0.0,
                    text: "ok".into(),
                    size: 1.0,
                    color: [0.0; 4],
                },
            },
            NodeV1 {
                id: "image".into(),
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
            color: [0.0; 4],
        };
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::InvalidPositiveValue("width"))
        );

        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Path {
            points: vec![],
            color: [0.0; 4],
        };
        assert_eq!(value.validate(), Err(SceneValidationError::InvalidPath));

        let mut value = scene();
        value.nodes[0].kind = NodeKindV1::Path {
            points: vec![PointV1 { x: 0.0, y: 0.0 }; MAX_PATH_POINTS + 1],
            color: [0.0; 4],
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
            color: [0.0; 4],
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
}
