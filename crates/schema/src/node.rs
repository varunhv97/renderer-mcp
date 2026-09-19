use crate::error::SceneValidationError;
use crate::fill::FillV1;
use crate::limits::MAX_PATH_POINTS;
use crate::validate::validate_positive;
use serde::{Deserialize, Serialize};

/// One visual element placed in a scene, identified by a scene-unique `id`
/// and dispatching on `kind` (flattened from [`NodeKindV1`]) to one of six
/// shape variants.
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
    pub(crate) fn validate(&self) -> Result<(), SceneValidationError> {
        self.kind.validate()
    }
}

/// The six supported node shapes, discriminated by a `kind` JSON tag. Each
/// variant carries exactly the fields that shape needs (no shared struct),
/// so adding a shape means extending this match everywhere it's exhaustively
/// matched -- `validate` below, and the renderer's draw dispatch -- rather
/// than risking a silently-unhandled default case.
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
    pub(crate) fn validate(&self) -> Result<(), SceneValidationError> {
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

/// One vertex of a `Path` node's polygon.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PointV1 {
    pub x: f32,
    pub y: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SceneValidationError;
    use crate::fill::FillV1;
    use crate::scene::SceneV1;
    use crate::test_support::*;

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
}
