use crate::{fill::*, tessellation::*, vertex::AnalyticVertex};
use renderer_schema::{Color, FillV1, SceneV1};

const ANALYTIC_SHAPE_RECT: u32 = 0;

const ANALYTIC_SHAPE_ELLIPSE: u32 = 1;

const ANALYTIC_SHAPE_LINE: u32 = 2;

/// Geometry margin, in scene-pixel units, that every analytic-AA shape
/// builder expands its emitted triangle(s) by beyond the shape's true
/// mathematical boundary (while the SDF math itself still measures distance
/// to the *true*, unexpanded boundary -- see each builder function). The
/// rasterizer only ever runs the fragment shader on pixels actually covered
/// by the submitted geometry; without this margin, pixels just outside the
/// exact edge -- exactly the pixels `fwidth`-based coverage needs to fade
/// smoothly through on its way to 0 -- would never be shaded at all,
/// producing a hard clip at the true edge instead of a smooth fade beside
/// it. 2 scene pixels is comfortably wider than the ~1-pixel coverage band
/// `fs_main` computes for any shape/canvas size this renderer supports.
const ANALYTIC_AA_MARGIN: f32 = 2.0;

/// Same clip-space mapping as `vertex()`/`add_textured_rect`'s `point()`
/// helper above, factored out for the three analytic-AA shape builders.
fn analytic_clip_position(x: f32, y: f32, scene: &SceneV1) -> [f32; 2] {
    [
        x / scene.canvas.width as f32 * 2.0 - 1.0,
        1.0 - y / scene.canvas.height as f32 * 2.0,
    ]
}

/// Emits an analytic-AA rect: a single quad expanded by `ANALYTIC_AA_MARGIN`
/// on every side, carrying the *true* (unexpanded) half-extent in `param0`
/// and the *true* corner radius in `param1.x` so `rect_sdf` in
/// `ANALYTIC_SHADER` measures distance to the actual declared (and possibly
/// rounded) rect boundary. Each vertex's color is resolved individually via
/// `fill_vertex_color`, so a gradient `fill` renders correctly here exactly
/// as it does in the default (MSAA) pipeline's `add_rect` -- see that
/// function's sibling doc comment on `fill_vertex_color` above.
// Every parameter here is a genuinely distinct, independently-meaningful
// piece of geometry/styling/rendering context (not accidental duplication);
// bundling them into a params struct for this one internal helper would
// only add indirection, not clarity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_rect_analytic(
    vertices: &mut Vec<AnalyticVertex>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    corner_radius: f32,
    fill: &FillV1,
    scene: &SceneV1,
) {
    let half_size = [width / 2.0, height / 2.0];
    let center = [x + half_size[0], y + half_size[1]];
    let margin = ANALYTIC_AA_MARGIN;
    let make = |px: f32, py: f32| AnalyticVertex {
        clip_position: analytic_clip_position(px, py, scene),
        color: fill_vertex_color(fill, px, py, center, half_size),
        shape_kind: ANALYTIC_SHAPE_RECT,
        local: [px - center[0], py - center[1]],
        param0: half_size,
        param1: [corner_radius, 0.0],
        param2: [0.0; 2],
    };
    let a = make(x - margin, y - margin);
    let b = make(x + width + margin, y - margin);
    let c = make(x + width + margin, y + height + margin);
    let d = make(x - margin, y + height + margin);
    vertices.extend([a, b, c, a, c, d]);
}

/// Emits an analytic-AA ellipse as a triangle fan (mirroring `add_ellipse`'s
/// shape), but with the fan's *geometry* radii expanded by
/// `ANALYTIC_AA_MARGIN` (so the rasterizer shades a ring of pixels just
/// outside the true boundary) while `local` -- and therefore `ellipse_sdf`
/// in `ANALYTIC_SHADER` -- is always computed against the *true*,
/// unexpanded `rx`/`ry`. Each vertex's color is resolved individually via
/// `fill_vertex_color` (using the *true*, unexpanded `rx`/`ry` as its
/// half-extent, matching `add_ellipse`), so a gradient `fill` renders
/// correctly here exactly as it does in the default (MSAA) pipeline.
pub(crate) fn add_ellipse_analytic(
    vertices: &mut Vec<AnalyticVertex>,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    fill: &FillV1,
    scene: &SceneV1,
) {
    let geo_rx = rx + ANALYTIC_AA_MARGIN;
    let geo_ry = ry + ANALYTIC_AA_MARGIN;
    let center = [cx, cy];
    let half_extent = [rx, ry];
    let make = |px: f32, py: f32| AnalyticVertex {
        clip_position: analytic_clip_position(px, py, scene),
        color: fill_vertex_color(fill, px, py, center, half_extent),
        shape_kind: ANALYTIC_SHAPE_ELLIPSE,
        local: [(px - cx) / rx, (py - cy) / ry],
        param0: [0.0; 2],
        param1: [0.0; 2],
        param2: [0.0; 2],
    };
    let center = make(cx, cy);
    for index in 0..ELLIPSE_SEGMENTS {
        let start = std::f32::consts::TAU * index as f32 / ELLIPSE_SEGMENTS as f32;
        let end = std::f32::consts::TAU * (index + 1) as f32 / ELLIPSE_SEGMENTS as f32;
        vertices.extend([
            center,
            make(cx + geo_rx * start.cos(), cy + geo_ry * start.sin()),
            make(cx + geo_rx * end.cos(), cy + geo_ry * end.sin()),
        ]);
    }
}

/// Emits an analytic-AA line as a single quad wide/long enough to cover the
/// entire capsule (both rounded ends included) plus `ANALYTIC_AA_MARGIN`,
/// carrying the *true* (unexpanded) endpoints and half-thickness so
/// `capsule_sdf` in `ANALYTIC_SHADER` measures distance to the actual
/// declared stroke.
pub(crate) fn add_line_analytic(
    vertices: &mut Vec<AnalyticVertex>,
    start: [f32; 2],
    end: [f32; 2],
    thickness: f32,
    color: Color,
    scene: &SceneV1,
) {
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let length = (dx * dx + dy * dy).sqrt();
    if length == 0.0 {
        return;
    }
    let dir = [dx / length, dy / length];
    let perp = [-dir[1], dir[0]];
    let half_thickness = thickness / 2.0;
    let extent = half_thickness + ANALYTIC_AA_MARGIN;
    let offset = |along: f32, across: f32| {
        [
            dir[0] * along + perp[0] * across,
            dir[1] * along + perp[1] * across,
        ]
    };
    let corner = |base: [f32; 2], along: f32, across: f32| {
        let delta = offset(along, across);
        [base[0] + delta[0], base[1] + delta[1]]
    };
    let p1 = corner(start, -extent, -extent);
    let p2 = corner(end, extent, -extent);
    let p3 = corner(end, extent, extent);
    let p4 = corner(start, -extent, extent);
    let make = |p: [f32; 2]| AnalyticVertex {
        clip_position: analytic_clip_position(p[0], p[1], scene),
        color,
        shape_kind: ANALYTIC_SHAPE_LINE,
        local: p,
        param0: start,
        param1: end,
        param2: [half_thickness, 0.0],
    };
    let a = make(p1);
    let b = make(p2);
    let c = make(p3);
    let d = make(p4);
    vertices.extend([a, b, c, a, c, d]);
}
