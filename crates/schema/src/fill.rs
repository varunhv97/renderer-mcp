use crate::error::SceneValidationError;
use crate::validate::validate_color;
use serde::{Deserialize, Serialize};

/// RGBA, each channel a finite float from 0.0 through 1.0 (see
/// `validate_color`); values outside that range fail scene validation.
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
    pub(crate) fn validate(&self) -> Result<(), SceneValidationError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SceneValidationError;
    use crate::node::NodeKindV1;
    use crate::scene::SceneV1;
    use crate::test_support::*;

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
