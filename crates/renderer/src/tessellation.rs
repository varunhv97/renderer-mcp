use crate::{fill::*, vertex::Vertex};
use renderer_schema::{Color, FillV1, NodeKindV1, SceneV1};

pub(crate) const ELLIPSE_SEGMENTS: usize = 32;

/// Arc tessellation resolution for one rounded rect corner (a 90-degree
/// sweep), reusing `ELLIPSE_SEGMENTS`' angular resolution: `ELLIPSE_SEGMENTS`
/// segments cover a full 360-degree ellipse, so `ELLIPSE_SEGMENTS / 4`
/// segments cover one 90-degree corner at the same degrees-per-segment
/// density (`add_rect`'s tessellation technique otherwise mirrors
/// `add_ellipse`'s triangle-fan-from-center approach directly).
pub(crate) const RECT_CORNER_SEGMENTS: usize = ELLIPSE_SEGMENTS / 4;

#[cfg(test)]
pub(crate) fn vertices_for_scene(scene: &SceneV1) -> (Vec<Vertex>, Vec<String>) {
    let mut vertices = Vec::new();
    let warnings = Vec::new();
    for node in &scene.nodes {
        add_node_vertices(&mut vertices, node, scene);
    }
    (vertices, warnings)
}

pub(crate) fn add_node_vertices(
    vertices: &mut Vec<Vertex>,
    node: &renderer_schema::NodeV1,
    scene: &SceneV1,
) {
    let [dx, dy] = node.translate;
    match &node.kind {
        NodeKindV1::Rect {
            x,
            y,
            width,
            height,
            corner_radius,
            fill,
        } => {
            let x = *x + dx;
            let y = *y + dy;
            add_rect(vertices, x, y, *width, *height, *corner_radius, fill, scene);
        }
        NodeKindV1::Ellipse {
            cx,
            cy,
            rx,
            ry,
            fill,
        } => {
            let cx = *cx + dx;
            let cy = *cy + dy;
            add_ellipse(vertices, cx, cy, *rx, *ry, fill, scene);
        }
        NodeKindV1::Line {
            x1,
            y1,
            x2,
            y2,
            thickness,
            fill,
        } => {
            let start = [*x1 + dx, *y1 + dy];
            let end = [*x2 + dx, *y2 + dy];
            add_line(
                vertices,
                start,
                end,
                *thickness,
                fill.resolve_solid(),
                scene,
            );
        }
        NodeKindV1::Path { points, fill } => {
            let translated: Vec<_> = points
                .iter()
                .map(|point| renderer_schema::PointV1 {
                    x: point.x + dx,
                    y: point.y + dy,
                })
                .collect();
            add_path(vertices, &translated, fill.resolve_solid(), scene);
        }
        NodeKindV1::Text { .. } | NodeKindV1::Image { .. } => {}
    }
}

/// Emits a (optionally rounded, optionally gradient-filled) rect for the
/// default MSAA pipeline. `corner_radius <= 0.0` takes the exact same
/// 2-triangle quad path this function always used before rounded corners
/// existed -- byte-identical output to before this feature existed, since
/// `fill_vertex_color` also returns the plain per-vertex `color` unchanged
/// for a `FillV1::Solid` fill (see that function's doc comment). This is
/// verified directly by
/// `rounded_rect_with_zero_radius_matches_the_original_plain_rect_tessellation`
/// below.
///
/// `corner_radius > 0.0` tessellates properly rather than approximating:
/// straight edges plus a small triangle-fan arc at each of the 4 corners,
/// all fanned from the rect's center -- mirroring `add_ellipse`'s
/// triangle-fan-from-center technique and reusing its angular resolution
/// via `RECT_CORNER_SEGMENTS` (`ELLIPSE_SEGMENTS / 4`, i.e. the same
/// degrees-per-segment density for one 90-degree corner as `add_ellipse`
/// uses for a full 360-degree ellipse). A fan from the center to each
/// consecutive pair of perimeter points is correct regardless of whether
/// that pair spans a curved arc segment or a straight edge -- no special
/// casing is needed for the 4 straight edges, since a single triangle
/// between two straight-edge endpoints and the center is already exact.
// See `add_rect_analytic`'s identical justification for this attribute.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_rect(
    vertices: &mut Vec<Vertex>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    corner_radius: f32,
    fill: &FillV1,
    scene: &SceneV1,
) {
    let center = [x + width / 2.0, y + height / 2.0];
    let half_extent = [width / 2.0, height / 2.0];
    let make = |px: f32, py: f32| {
        vertex(
            px,
            py,
            fill_vertex_color(fill, px, py, center, half_extent),
            scene,
        )
    };
    if corner_radius <= 0.0 {
        let a = make(x, y);
        let b = make(x + width, y);
        let c = make(x + width, y + height);
        let d = make(x, y + height);
        vertices.extend([a, b, c, a, c, d]);
        return;
    }
    let r = corner_radius;
    let center_vertex = make(center[0], center[1]);
    // One (arc_center_x, arc_center_y, start_angle, end_angle) tuple per
    // corner, in clockwise perimeter order starting at the top-right
    // corner (this schema's scene-pixel space is y-down, so angle 0 is
    // `+x` and increasing angle sweeps clockwise on screen).
    let quarter = std::f32::consts::FRAC_PI_2;
    let corners = [
        (x + width - r, y + r, -quarter, 0.0),
        (x + width - r, y + height - r, 0.0, quarter),
        (x + r, y + height - r, quarter, std::f32::consts::PI),
        (
            x + r,
            y + r,
            std::f32::consts::PI,
            std::f32::consts::PI + quarter,
        ),
    ];
    let mut perimeter = Vec::with_capacity(4 * (RECT_CORNER_SEGMENTS + 1));
    for (arc_cx, arc_cy, start, end) in corners {
        for segment in 0..=RECT_CORNER_SEGMENTS {
            let angle = start + (end - start) * segment as f32 / RECT_CORNER_SEGMENTS as f32;
            perimeter.push((arc_cx + r * angle.cos(), arc_cy + r * angle.sin()));
        }
    }
    for pair in perimeter.windows(2) {
        vertices.extend([
            center_vertex,
            make(pair[0].0, pair[0].1),
            make(pair[1].0, pair[1].1),
        ]);
    }
    let first = perimeter[0];
    let last = *perimeter.last().expect("perimeter is never empty");
    vertices.extend([center_vertex, make(last.0, last.1), make(first.0, first.1)]);
}

pub(crate) fn add_ellipse(
    vertices: &mut Vec<Vertex>,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    fill: &FillV1,
    scene: &SceneV1,
) {
    let center = [cx, cy];
    let half_extent = [rx, ry];
    let make = |px: f32, py: f32| {
        vertex(
            px,
            py,
            fill_vertex_color(fill, px, py, center, half_extent),
            scene,
        )
    };
    let center_vertex = make(cx, cy);
    for index in 0..ELLIPSE_SEGMENTS {
        let start = std::f32::consts::TAU * index as f32 / ELLIPSE_SEGMENTS as f32;
        let end = std::f32::consts::TAU * (index + 1) as f32 / ELLIPSE_SEGMENTS as f32;
        vertices.extend([
            center_vertex,
            make(cx + rx * start.cos(), cy + ry * start.sin()),
            make(cx + rx * end.cos(), cy + ry * end.sin()),
        ]);
    }
}

pub(crate) fn add_line(
    vertices: &mut Vec<Vertex>,
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
    let offset_x = -dy / length * thickness / 2.0;
    let offset_y = dx / length * thickness / 2.0;
    let a = vertex(start[0] + offset_x, start[1] + offset_y, color, scene);
    let b = vertex(end[0] + offset_x, end[1] + offset_y, color, scene);
    let c = vertex(end[0] - offset_x, end[1] - offset_y, color, scene);
    let d = vertex(start[0] - offset_x, start[1] - offset_y, color, scene);
    vertices.extend([a, b, c, a, c, d]);
}

pub(crate) fn add_path(
    vertices: &mut Vec<Vertex>,
    points: &[renderer_schema::PointV1],
    color: Color,
    scene: &SceneV1,
) {
    let origin = vertex(points[0].x, points[0].y, color, scene);
    for pair in points[1..].windows(2) {
        vertices.extend([
            origin,
            vertex(pair[0].x, pair[0].y, color, scene),
            vertex(pair[1].x, pair[1].y, color, scene),
        ]);
    }
}

pub(crate) fn vertex(x: f32, y: f32, color: Color, scene: &SceneV1) -> Vertex {
    Vertex {
        position: [
            x / scene.canvas.width as f32 * 2.0 - 1.0,
            1.0 - y / scene.canvas.height as f32 * 2.0,
        ],
        color,
    }
}
