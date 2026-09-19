use crate::*;

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

pub(crate) fn midpoint(a: Color, b: Color) -> Color {
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
