use renderer_schema::{Color, FillV1, GradientV1};

/// Resolves the actual RGBA color one vertex at scene-pixel position
/// `(px, py)` should carry for a shape's `fill`, given that shape's bounding
/// center and half-extent (half-width/half-height for a `Rect`, `rx`/`ry`
/// for an `Ellipse`).
///
/// For a solid fill every vertex gets the same color -- exactly today's
/// existing flat-fill behavior, preserved unchanged. For a gradient, each
/// vertex gets its *own* color computed from its own position, and the GPU
/// rasterizer linearly interpolates between a triangle's vertex colors
/// across its interior automatically -- this project's existing flat-color
/// rendering already relies on this same hardware behavior (every vertex of
/// one shape simply happens to receive the same color today). That means a
/// gradient needs no fragment-shader changes in either the default (MSAA,
/// `Vertex`/`PRIMITIVE_SHADER`) or analytic-AA (`AnalyticVertex`/
/// `ANALYTIC_SHADER`) pipeline: both call this same function per emitted
/// vertex and let interpolation do the rest. (Confirmed empirically, not
/// just assumed: a rect built with deliberately different literal per-corner
/// colors was rendered and its raw RGBA output showed a smooth blend across
/// the shape rather than a flat or hard-cut result.)
///
/// Linear gradients project `(px, py) - center` onto the unit direction
/// vector derived from `angle_degrees` (`0` = `+x`; increasing values rotate
/// clockwise in this schema's y-down scene-pixel space), normalized by the
/// shape's bounding half-extent projected onto that same direction, then
/// clamp to `[0, 1]` and mix `from`/`to` by that fraction.
///
/// Radial gradients normalize `(px, py)`'s offset from `center` by an
/// elliptical metric using `half_extent` as the two radii (so the gradient
/// reaches `edge` exactly at the ellipse inscribed in the shape's bounding
/// box -- for a `Rect` this means straight edge midpoints reach `edge`
/// exactly while corners clamp to `edge` slightly before the true corner),
/// then mix `center`/`edge` by that fraction.
pub(crate) fn fill_vertex_color(
    fill: &FillV1,
    px: f32,
    py: f32,
    center: [f32; 2],
    half_extent: [f32; 2],
) -> Color {
    match fill {
        FillV1::Solid(color) => *color,
        FillV1::Gradient(GradientV1::LinearGradient {
            from,
            to,
            angle_degrees,
        }) => {
            let angle = angle_degrees.to_radians();
            let direction = [angle.cos(), angle.sin()];
            let relative = [px - center[0], py - center[1]];
            let projected = relative[0] * direction[0] + relative[1] * direction[1];
            let extent = (half_extent[0] * direction[0].abs()
                + half_extent[1] * direction[1].abs())
            .max(1e-6);
            let fraction = ((projected / extent) + 1.0) / 2.0;
            lerp_color(*from, *to, fraction.clamp(0.0, 1.0))
        }
        FillV1::Gradient(GradientV1::RadialGradient {
            center: stop_center,
            edge,
        }) => {
            let rx = half_extent[0].max(1e-6);
            let ry = half_extent[1].max(1e-6);
            let dx = (px - center[0]) / rx;
            let dy = (py - center[1]) / ry;
            let fraction = (dx * dx + dy * dy).sqrt();
            lerp_color(*stop_center, *edge, fraction.clamp(0.0, 1.0))
        }
    }
}

fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}
