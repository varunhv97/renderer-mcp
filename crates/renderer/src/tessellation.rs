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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::*;
    use crate::test_support::*;
    use crate::util::*;
    use renderer_schema::CanvasV1;
    use renderer_schema::FillV1;
    use renderer_schema::NodeKindV1;
    use renderer_schema::NodeV1;
    use renderer_schema::SCENE_VERSION_V1;
    use renderer_schema::SceneV1;

    #[test]
    fn compiles_rectangle_vertices() {
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 100,
                height: 100,
                background: [0.0; 4],
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
        };
        let (vertices, warnings) = vertices_for_scene(&scene);
        assert_eq!(vertices.len(), 6);
        assert!(warnings.is_empty());
    }

    #[test]
    fn compiles_all_geometry_and_reports_unrasterized_nodes() {
        let mut scene = test_scene();
        scene.nodes.extend([
            NodeV1 {
                id: "ellipse".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Ellipse {
                    cx: 20.0,
                    cy: 20.0,
                    rx: 5.0,
                    ry: 5.0,
                    fill: FillV1::Solid([0.0, 1.0, 0.0, 1.0]),
                },
            },
            NodeV1 {
                id: "line".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Line {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 20.0,
                    y2: 20.0,
                    thickness: 2.0,
                    fill: FillV1::Solid([0.0, 0.0, 1.0, 1.0]),
                },
            },
            NodeV1 {
                id: "path".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Path {
                    points: vec![
                        renderer_schema::PointV1 { x: 0.0, y: 0.0 },
                        renderer_schema::PointV1 { x: 10.0, y: 0.0 },
                        renderer_schema::PointV1 { x: 0.0, y: 10.0 },
                    ],
                    fill: FillV1::Solid([1.0; 4]),
                },
            },
            NodeV1 {
                id: "text".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Text {
                    x: 0.0,
                    y: 0.0,
                    text: "t".into(),
                    size: 8.0,
                    fill: FillV1::Solid([1.0; 4]),
                },
            },
            NodeV1 {
                id: "vector".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 16.0,
                    y: 1.0,
                    width: 2.0,
                    height: 2.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0; 4]),
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
                    source: "x".into(),
                },
            },
        ]);
        let (vertices, warnings) = vertices_for_scene(&scene);
        assert!(vertices.len() > 100);
        assert!(warnings.is_empty());
    }

    #[test]
    fn covers_geometry_and_animation_edge_cases() {
        let mut scene = test_scene();
        scene.nodes.push(NodeV1 {
            id: "zero".into(),
            translate: [0.0, 0.0],
            kind: NodeKindV1::Line {
                x1: 1.0,
                y1: 1.0,
                x2: 1.0,
                y2: 1.0,
                thickness: 1.0,
                fill: FillV1::Solid([1.0; 4]),
            },
        });
        assert_eq!(vertices_for_scene(&scene).0.len(), 6);
        assert_eq!(scene_at(&scene, 1), scene);
        let variants = [
            NodeKindV1::Ellipse {
                cx: 0.0,
                cy: 0.0,
                rx: 1.0,
                ry: 1.0,
                fill: FillV1::Solid([1.0; 4]),
            },
            NodeKindV1::Line {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
                thickness: 1.0,
                fill: FillV1::Solid([1.0; 4]),
            },
            NodeKindV1::Path {
                points: vec![renderer_schema::PointV1 { x: 0.0, y: 0.0 }; 3],
                fill: FillV1::Solid([1.0; 4]),
            },
            NodeKindV1::Text {
                x: 0.0,
                y: 0.0,
                text: "x".into(),
                size: 1.0,
                fill: FillV1::Solid([1.0; 4]),
            },
            NodeKindV1::Image {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                source: "x".into(),
            },
        ];
        for mut kind in variants {
            let _ = fill_of(&kind);
            let _ = fill_mut(&mut kind);
        }
        assert_eq!(
            interpolate(&[(10, 1.0_f32), (20, 2.0)], 0, |a, b, t| a + (b - a) * t),
            Some(1.0)
        );
        assert_eq!(
            interpolate(&[(10, 1.0_f32), (20, 2.0)], 30, |a, b, t| a + (b - a) * t),
            Some(2.0)
        );
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("output"), b"bytes").unwrap();
        assert!(hash_file(&directory.path().join("output")).is_ok());
        assert!(hash_file(&directory.path().join("missing")).is_err());
    }

    /// Portable (no-GPU) proof that a zero (or omitted) `corner_radius`
    /// leaves `add_rect`'s output byte-identical to the plain 2-triangle
    /// quad it always emitted before rounded corners existed. Reconstructs
    /// that original tessellation by hand (the same `vertex()` calls in the
    /// same `[a, b, c, a, c, d]` order `add_rect`'s original implementation
    /// used) and compares every emitted `Vertex`'s position and color
    /// field-by-field against `test_scene()`'s zero-radius rect.
    #[test]
    fn rounded_rect_with_zero_radius_matches_the_original_plain_rect_tessellation() {
        let scene = test_scene();
        let NodeKindV1::Rect {
            x,
            y,
            width,
            height,
            corner_radius,
            fill,
        } = &scene.nodes[0].kind
        else {
            panic!("test_scene()'s only node must be a Rect");
        };
        assert_eq!(*corner_radius, 0.0);
        let color = fill.resolve_solid();

        let (actual, _) = vertices_for_scene(&scene);
        let a = vertex(*x, *y, color, &scene);
        let b = vertex(*x + *width, *y, color, &scene);
        let c = vertex(*x + *width, *y + *height, color, &scene);
        let d = vertex(*x, *y + *height, color, &scene);
        let expected = [a, b, c, a, c, d];

        assert_eq!(actual.len(), expected.len());
        for (index, (found, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                found.position, want.position,
                "vertex {index} position mismatch"
            );
            assert_eq!(found.color, want.color, "vertex {index} color mismatch");
        }
    }

    /// Portable (no-GPU) proof that a positive `corner_radius` actually
    /// changes the emitted geometry (more triangles than the flat 2-triangle
    /// quad, since corners are now tessellated as arcs) rather than being
    /// silently ignored by the vertex builder.
    #[test]
    fn rounded_rect_emits_more_triangles_than_a_plain_rect() {
        let mut scene = test_scene();
        let (plain, _) = vertices_for_scene(&scene);

        let NodeKindV1::Rect { corner_radius, .. } = &mut scene.nodes[0].kind else {
            panic!("test_scene()'s only node must be a Rect");
        };
        *corner_radius = 3.0;
        scene.validate().unwrap();
        let (rounded, _) = vertices_for_scene(&scene);

        assert_eq!(
            plain.len(),
            6,
            "a plain (zero-radius) rect is always 2 triangles"
        );
        assert!(
            rounded.len() > plain.len(),
            "expected a rounded rect to tessellate into more triangles ({}) than a plain \
             rect ({})",
            rounded.len(),
            plain.len()
        );
    }
}
