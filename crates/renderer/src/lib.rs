//! Off-screen wgpu renderer for normalized RendererCli scenes.
#![allow(unexpected_cfgs)] // `cargo llvm-cov` supplies `cfg(coverage)`/`cfg(coverage_nightly)`.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

#[cfg(test)]
use animation::*;
#[cfg(test)]
use assets::*;
#[cfg(test)]
use composition::*;
#[cfg(test)]
use fontdue::Font;
#[cfg(test)]
use gpu::MSAA_SAMPLE_COUNT;
#[cfg(test)]
use renderer_schema::{NodeKindV1, SceneV1};
#[cfg(test)]
use std::{fs, path::Path};
#[cfg(test)]
use tessellation::*;
#[cfg(test)]
use util::*;

mod analytic_geometry;
mod animation;
mod assets;
mod composition;
mod error;
mod fill;
mod gif;
mod gpu;
mod limits;
mod pipelines;
mod shaders;
mod svg;
mod tessellation;
#[cfg(test)]
mod test_support;
mod util;
mod vertex;

pub use error::{RenderError, RenderedImage};
pub use gpu::GpuRenderer;
// Only the `#[cfg(test)]` reference rasterizer below still uses these; they move with it.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::MAX_COMPOSITION_TEXTURE_PIXELS;
    use crate::test_support::*;
    use renderer_schema::{CanvasV1, FillV1, GradientV1, KeyframeV1, NodeV1, SCENE_VERSION_V1};

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
    fn aligns_copy_rows() {
        assert_eq!(align_to(256, 256), 256);
        assert_eq!(align_to(260, 256), 512);
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
    fn interpolates_color_and_opacity_keyframes() {
        let mut scene = test_scene();
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 2,
            duration_ms: 1_000,
            keyframes: vec![
                KeyframeV1 {
                    at_ms: 0,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Color([0.0, 0.0, 0.0, 1.0]),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Color([1.0, 1.0, 1.0, 1.0]),
                },
                KeyframeV1 {
                    at_ms: 0,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(0.0),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(1.0),
                },
            ],
        });
        let at_middle = scene_at(&scene, 500);
        assert_eq!(
            fill_of(&at_middle.nodes[0].kind),
            Some(FillV1::Solid([0.5, 0.5, 0.5, 0.5]))
        );
        assert_eq!(interpolate_color(&[], 0), None);
        assert_eq!(interpolate_opacity(&[], 0), None);
        let opacity = KeyframeV1 {
            at_ms: 0,
            target: "box".into(),
            property: renderer_schema::AnimatedPropertyV1::Opacity(1.0),
        };
        let color = KeyframeV1 {
            at_ms: 0,
            target: "box".into(),
            property: renderer_schema::AnimatedPropertyV1::Color([1.0; 4]),
        };
        assert_eq!(interpolate_color(&[&opacity], 0), None);
        assert_eq!(interpolate_opacity(&[&color], 0), None);
    }

    #[test]
    fn interpolates_translate_keyframes() {
        let mut scene = test_scene();
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 2,
            duration_ms: 1_000,
            keyframes: vec![
                KeyframeV1 {
                    at_ms: 0,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Translate([0.0, 0.0]),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Translate([20.0, -10.0]),
                },
            ],
        });
        assert_eq!(scene_at(&scene, 0).nodes[0].translate, [0.0, 0.0]);
        assert_eq!(scene_at(&scene, 500).nodes[0].translate, [10.0, -5.0]);
        assert_eq!(scene_at(&scene, 1_000).nodes[0].translate, [20.0, -10.0]);
        assert_eq!(interpolate_translate(&[], 0), None);
        let opacity = KeyframeV1 {
            at_ms: 0,
            target: "box".into(),
            property: renderer_schema::AnimatedPropertyV1::Opacity(1.0),
        };
        assert_eq!(interpolate_translate(&[&opacity], 0), None);
    }

    #[test]
    fn multiplies_interpolated_color_alpha_by_opacity() {
        let mut scene = test_scene();
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 2,
            duration_ms: 1_000,
            keyframes: vec![
                KeyframeV1 {
                    at_ms: 0,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Color([1.0, 0.0, 0.0, 0.0]),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Color([1.0, 0.0, 0.0, 1.0]),
                },
                KeyframeV1 {
                    at_ms: 0,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(0.5),
                },
            ],
        });
        let at_middle = scene_at(&scene, 500);
        assert_eq!(
            fill_of(&at_middle.nodes[0].kind),
            Some(FillV1::Solid([1.0, 0.0, 0.0, 0.25]))
        );
    }

    /// Direct geometric-movement proof for `Rect` translate keyframes,
    /// through the default MSAA+supersampling `composition_plan` pipeline:
    /// renders the same scene at two `at_ms` values on either side of a
    /// translate keyframe pair, decodes real GPU output at both times, and
    /// asserts the rect's white fill genuinely appears at a *different*
    /// pixel location at each time -- not just that rendering succeeded.
    /// Sample points are chosen well inside each rect position (away from
    /// anti-aliased edges) and far enough apart that the two rect positions
    /// never overlap, so a false pass from coincidental pixel overlap is not
    /// possible.
    #[test]
    fn translate_keyframes_move_a_rect_node_to_a_different_pixel_location_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [0.0, 0.0, 0.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "box".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 4.0,
                    y: 4.0,
                    width: 10.0,
                    height: 10.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0, 1.0, 1.0, 1.0]),
                },
            }],
            timeline: Some(renderer_schema::TimelineV1 {
                fps: 2,
                duration_ms: 1_000,
                keyframes: vec![
                    KeyframeV1 {
                        at_ms: 0,
                        target: "box".into(),
                        property: renderer_schema::AnimatedPropertyV1::Translate([0.0, 0.0]),
                    },
                    KeyframeV1 {
                        at_ms: 1_000,
                        target: "box".into(),
                        property: renderer_schema::AnimatedPropertyV1::Translate([30.0, 30.0]),
                    },
                ],
            }),
            effect: None,
        };
        scene.validate().unwrap();

        let width = scene.canvas.width as usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let start = scene_at(&scene, 0);
        start.validate().unwrap();
        let (start_pixels, start_warnings) = renderer.render_rgba(&start).unwrap();
        assert!(start_warnings.is_empty());

        let end = scene_at(&scene, 1_000);
        end.validate().unwrap();
        let (end_pixels, end_warnings) = renderer.render_rgba(&end).unwrap();
        assert!(end_warnings.is_empty());

        let original_center = (9, 9);
        let translated_center = (39, 39);
        let background = pixel_at(&start_pixels, 0, 0);
        let white = [255, 255, 255, 255];
        assert_eq!(background, [0, 0, 0, 255]);

        assert_eq!(
            pixel_at(&start_pixels, original_center.0, original_center.1),
            white,
            "at at_ms=0 the rect should render at its declared (untranslated) location"
        );
        assert_eq!(
            pixel_at(&start_pixels, translated_center.0, translated_center.1),
            background,
            "at at_ms=0 the translated location should still be background"
        );

        assert_eq!(
            pixel_at(&end_pixels, translated_center.0, translated_center.1),
            white,
            "at at_ms=1000 the rect should have moved to the translated location"
        );
        assert_eq!(
            pixel_at(&end_pixels, original_center.0, original_center.1),
            background,
            "at at_ms=1000 the original location should be background again since the rect moved away"
        );
    }

    /// Same geometric-movement proof as the `Rect` test above, but for a
    /// `Path` node -- whose translate offset is structurally different
    /// (applied to every point in a list, not a single x/y pair), so this
    /// specifically proves that case was not missed.
    #[test]
    fn translate_keyframes_move_a_path_node_to_a_different_pixel_location_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [0.0, 0.0, 0.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "triangle".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Path {
                    points: vec![
                        renderer_schema::PointV1 { x: 10.0, y: 10.0 },
                        renderer_schema::PointV1 { x: 30.0, y: 10.0 },
                        renderer_schema::PointV1 { x: 10.0, y: 30.0 },
                    ],
                    fill: FillV1::Solid([1.0, 1.0, 1.0, 1.0]),
                },
            }],
            timeline: Some(renderer_schema::TimelineV1 {
                fps: 2,
                duration_ms: 1_000,
                keyframes: vec![
                    KeyframeV1 {
                        at_ms: 0,
                        target: "triangle".into(),
                        property: renderer_schema::AnimatedPropertyV1::Translate([0.0, 0.0]),
                    },
                    KeyframeV1 {
                        at_ms: 1_000,
                        target: "triangle".into(),
                        property: renderer_schema::AnimatedPropertyV1::Translate([25.0, 25.0]),
                    },
                ],
            }),
            effect: None,
        };
        scene.validate().unwrap();

        let width = scene.canvas.width as usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let start = scene_at(&scene, 0);
        start.validate().unwrap();
        let (start_pixels, start_warnings) = renderer.render_rgba(&start).unwrap();
        assert!(start_warnings.is_empty());

        let end = scene_at(&scene, 1_000);
        end.validate().unwrap();
        let (end_pixels, end_warnings) = renderer.render_rgba(&end).unwrap();
        assert!(end_warnings.is_empty());

        // Sample points well inside the triangle's interior (away from its
        // anti-aliased hypotenuse edge) for the untranslated and translated
        // positions.
        let original_interior = (14, 14);
        let translated_interior = (39, 39);
        let background = pixel_at(&start_pixels, 0, 0);
        let white = [255, 255, 255, 255];
        assert_eq!(background, [0, 0, 0, 255]);

        assert_eq!(
            pixel_at(&start_pixels, original_interior.0, original_interior.1),
            white,
            "at at_ms=0 the path should render at its declared (untranslated) points"
        );
        assert_eq!(
            pixel_at(&start_pixels, translated_interior.0, translated_interior.1),
            background,
            "at at_ms=0 the translated location should still be background"
        );

        assert_eq!(
            pixel_at(&end_pixels, translated_interior.0, translated_interior.1),
            white,
            "at at_ms=1000 every point in the path should have shifted by the translate offset"
        );
        assert_eq!(
            pixel_at(&end_pixels, original_interior.0, original_interior.1),
            background,
            "at at_ms=1000 the original location should be background again since the path moved away"
        );
    }

    /// Lighter-touch confirmation that the experimental analytic-AA
    /// composition pipeline (`composition_plan_analytic` /
    /// `render_rgba_analytic_aa`) also applies `translate` -- reusing the
    /// same scene/keyframes as the default-pipeline `Rect` proof above so
    /// the two pipelines are checked against literally the same geometry,
    /// without needing the same exhaustive per-shape-kind coverage as the
    /// default path.
    #[test]
    fn translate_keyframes_move_a_rect_node_under_analytic_aa_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [0.0, 0.0, 0.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "box".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 4.0,
                    y: 4.0,
                    width: 10.0,
                    height: 10.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0, 1.0, 1.0, 1.0]),
                },
            }],
            timeline: Some(renderer_schema::TimelineV1 {
                fps: 2,
                duration_ms: 1_000,
                keyframes: vec![
                    KeyframeV1 {
                        at_ms: 0,
                        target: "box".into(),
                        property: renderer_schema::AnimatedPropertyV1::Translate([0.0, 0.0]),
                    },
                    KeyframeV1 {
                        at_ms: 1_000,
                        target: "box".into(),
                        property: renderer_schema::AnimatedPropertyV1::Translate([30.0, 30.0]),
                    },
                ],
            }),
            effect: None,
        };
        scene.validate().unwrap();

        let width = scene.canvas.width as usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let start = scene_at(&scene, 0);
        start.validate().unwrap();
        let (start_pixels, start_warnings) = renderer.render_rgba_analytic_aa(&start).unwrap();
        assert!(start_warnings.is_empty());

        let end = scene_at(&scene, 1_000);
        end.validate().unwrap();
        let (end_pixels, end_warnings) = renderer.render_rgba_analytic_aa(&end).unwrap();
        assert!(end_warnings.is_empty());

        let original_center = (9, 9);
        let translated_center = (39, 39);
        let background = pixel_at(&start_pixels, 0, 0);
        let white = [255, 255, 255, 255];

        assert_eq!(
            pixel_at(&start_pixels, original_center.0, original_center.1),
            white,
            "analytic-AA path: at at_ms=0 the rect should render at its declared location"
        );
        assert_eq!(
            pixel_at(&end_pixels, translated_center.0, translated_center.1),
            white,
            "analytic-AA path: at at_ms=1000 the rect should have moved to the translated location"
        );
        assert_eq!(
            pixel_at(&end_pixels, original_center.0, original_center.1),
            background,
            "analytic-AA path: at at_ms=1000 the original location should be background again"
        );
    }

    #[test]
    fn accepts_only_png_render_paths() {
        assert!(ensure_png_output_path(Path::new("scene.png")).is_ok());
        assert!(ensure_png_output_path(Path::new("scene.PNG")).is_ok());
        assert!(matches!(
            ensure_png_output_path(Path::new("scene.gif")),
            Err(RenderError::InvalidPngOutputPath(_))
        ));
        assert!(matches!(
            ensure_png_output_path(Path::new("scene")),
            Err(RenderError::InvalidPngOutputPath(_))
        ));
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

    #[test]
    fn renders_png_and_gif_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        {
            let directory = tempfile::tempdir().unwrap();
            let scene = test_scene();
            let png = renderer
                .render_png(&scene, &directory.path().join("scene.png"))
                .unwrap();
            assert_eq!(png.frame_count, 1);
            let mut animated = scene;
            animated.timeline = Some(renderer_schema::TimelineV1 {
                fps: 2,
                duration_ms: 1_000,
                keyframes: vec![],
            });
            let gif = renderer
                .render_gif(&animated, &directory.path().join("scene.gif"))
                .unwrap();
            assert_eq!(gif.frame_count, 2);
        }
    }

    /// Direct anti-aliasing regression test: renders a diagonal (non-axis-
    /// aligned) line and asserts at least one pixel along its edge lands
    /// strictly between the background color and the line color.
    ///
    /// Before MSAA was added, this renderer's vector primitives (rect/
    /// ellipse/line/path) had no anti-aliasing at all: decoding a real
    /// rendered PNG's raw pixel bytes showed a diagonal line's edge
    /// transitioning directly from `(255,255,255,255)` to `(89,89,89,255)`
    /// with no intermediate blended pixel anywhere along a clearly diagonal
    /// edge -- a hard, stair-stepped edge. A binary hard edge can still pass
    /// a tolerance-based golden-image comparison (edges just shift by a
    /// pixel or two), so that alone would not catch a regression back to
    /// hard edges. This test instead checks the actual pixel values along a
    /// known diagonal edge for real intermediate coverage-weighted color,
    /// which only MSAA (or another anti-aliasing scheme) can produce.
    #[test]
    fn anti_aliases_diagonal_primitive_edges_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "diagonal".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Line {
                    x1: 6.0,
                    y1: 6.0,
                    x2: 58.0,
                    y2: 58.0,
                    thickness: 6.0,
                    fill: FillV1::Solid([0.2, 0.2, 0.2, 1.0]),
                },
            }],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };
        // A corner far from the diagonal stroke (solid background) and a
        // point on the line's centerline far from both of its ends (solid
        // line interior) give the two real, GPU-rendered "pure" colors to
        // compare edge pixels against -- more robust than hardcoding
        // expected sRGB-encoded byte values here.
        let background_pixel = pixel_at(0, 0);
        let line_pixel = pixel_at(32, 32);
        assert_ne!(
            background_pixel, line_pixel,
            "sanity check: the sampled background and line-interior points must differ"
        );

        let mut found_blended_pixel = false;
        'scan: for y in 0..scene.canvas.height as usize {
            for x in 0..width {
                let pixel = pixel_at(x, y);
                let strictly_between = (0..3).all(|channel| {
                    let low = background_pixel[channel].min(line_pixel[channel]);
                    let high = background_pixel[channel].max(line_pixel[channel]);
                    pixel[channel] > low && pixel[channel] < high
                });
                if strictly_between {
                    found_blended_pixel = true;
                    break 'scan;
                }
            }
        }
        assert!(
            found_blended_pixel,
            "expected at least one pixel strictly between the background color {background_pixel:?} \
             and the line color {line_pixel:?} along the diagonal edge, proving real \
             coverage-weighted MSAA blending occurred instead of a hard binary edge"
        );
    }

    /// Direct SSAA regression test, analogous to
    /// `anti_aliases_diagonal_primitive_edges_on_an_available_gpu` above but
    /// checking supersampling's *additional* contribution on top of MSAA:
    /// renders the exact same diagonal-line scene as that test and counts
    /// how many genuinely distinct, strictly-intermediate red-channel values
    /// (the line and background are both gray/white, so R=G=B and one
    /// channel suffices) appear anywhere along the line's edge.
    ///
    /// `MSAA_SAMPLE_COUNT`x MSAA alone resolves at most a handful of
    /// coverage fractions per edge pixel (this project's own investigation
    /// that motivated adding `SUPERSAMPLE_FACTOR` found roughly 4-5 such
    /// levels decoding raw MSAA-only output); measured directly against
    /// *this* scene with `SUPERSAMPLE_FACTOR` temporarily forced to 1
    /// (MSAA-only, no supersampling), it produced exactly one distinct
    /// intermediate red value across the whole image. With
    /// `SUPERSAMPLE_FACTOR` at its real value, this same scene measured 7
    /// distinct intermediate values on this machine (Apple M1/Metal) --
    /// deterministic and stable across repeated runs, since both the
    /// geometry and the Lanczos3 downsample are deterministic given fixed
    /// input. `MIN_DISTINCT_EDGE_LEVELS` (6) sits strictly above the MSAA-
    /// only range this project measured (1, and up to ~4-5 by the broader
    /// investigation that motivated this feature) while leaving a small
    /// margin below the 7 measured here, so a regression back to MSAA-only
    /// behavior -- or a supersample factor accidentally forced to 1 -- fails
    /// this test, while ordinary cross-GPU/driver rounding differences in
    /// exactly which byte values appear should not.
    #[test]
    fn supersamples_diagonal_primitive_edges_beyond_msaa_alone_on_an_available_gpu() {
        const MIN_DISTINCT_EDGE_LEVELS: usize = 6;

        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "diagonal".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Line {
                    x1: 6.0,
                    y1: 6.0,
                    x2: 58.0,
                    y2: 58.0,
                    thickness: 6.0,
                    fill: FillV1::Solid([0.2, 0.2, 0.2, 1.0]),
                },
            }],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };
        let background_pixel = pixel_at(0, 0);
        let line_pixel = pixel_at(32, 32);
        assert_ne!(
            background_pixel, line_pixel,
            "sanity check: the sampled background and line-interior points must differ"
        );
        let low = background_pixel[0].min(line_pixel[0]);
        let high = background_pixel[0].max(line_pixel[0]);

        let mut distinct_edge_levels = std::collections::BTreeSet::new();
        for y in 0..scene.canvas.height as usize {
            for x in 0..width {
                let red = pixel_at(x, y)[0];
                if red > low && red < high {
                    distinct_edge_levels.insert(red);
                }
            }
        }
        assert!(
            distinct_edge_levels.len() >= MIN_DISTINCT_EDGE_LEVELS,
            "expected at least {MIN_DISTINCT_EDGE_LEVELS} distinct strictly-intermediate \
             red-channel values along the diagonal edge (found {}: {distinct_edge_levels:?}), \
             proving supersampling contributes a genuinely richer gradient than \
             {MSAA_SAMPLE_COUNT}x MSAA alone can produce",
            distinct_edge_levels.len()
        );
    }

    #[test]
    fn applies_a_full_canvas_effect_shader_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        // `test_scene()`'s canvas background is fully transparent black
        // ([0,0,0,0]) and its only node (a red rect) does not cover pixel
        // (0,0), so pre-supersampling this pixel round-tripped 0.0/1.0
        // exactly through the sRGB transfer function used by the
        // `Rgba8UnormSrgb` intermediate texture, making an exact-byte
        // comparison meaningful there. With `SUPERSAMPLE_FACTOR` supersampling
        // now in the pipeline, that no longer holds exactly: the downsample
        // filter (see `SUPERSAMPLE_FACTOR`'s doc comment) has nonzero support
        // beyond a single output texel, so a hard content edge a few source
        // texels away (the rect's edge, still pixel-aligned in the oversized
        // render) blends a sliver of it into an otherwise-background output
        // pixel near it -- exactly the kind of edge softening this crate's
        // `assert_matches_golden` tolerance elsewhere already accounts for
        // at shape/glyph/image edges. `PIXEL_TOLERANCE` absorbs that
        // (measured 5 of 255 here, with the Triangle filter `SUPERSAMPLE_FACTOR`'s
        // downsample currently uses) while still exercising the real GPU
        // pass and catching an actual regression (wrong composition,
        // dropped alpha, effect not applied): RGB channels invert 0 -> 255
        // and alpha (never gamma-corrected) passes through unchanged.
        const PIXEL_TOLERANCE: i16 = 8;
        let assert_pixel_close = |label: &str, actual: &[u8], expected: [u8; 4]| {
            for channel in 0..4 {
                let delta = (actual[channel] as i16 - expected[channel] as i16).abs();
                assert!(
                    delta <= PIXEL_TOLERANCE,
                    "{label}: channel {channel} was {} (expected close to {}, tolerance {PIXEL_TOLERANCE})",
                    actual[channel],
                    expected[channel]
                );
            }
        };

        let scene = test_scene();
        let (without_effect, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());
        assert_pixel_close("without_effect", &without_effect[0..4], [0, 0, 0, 0]);

        let mut scene = scene;
        scene.effect = Some(renderer_schema::EffectV1 {
            shader: "fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32> {\n\
                     return vec4<f32>(1.0 - color.rgb, color.a);\n\
                     }"
            .into(),
        });
        let (with_effect, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());
        assert_pixel_close("with_effect", &with_effect[0..4], [255, 255, 255, 0]);

        assert_ne!(
            without_effect, with_effect,
            "applying the invert effect must change the composited output"
        );
    }

    #[test]
    fn an_invalid_effect_shader_returns_an_error_instead_of_panicking() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let mut scene = test_scene();

        // References an undefined identifier: must fail WGSL compile
        // validation, not panic the process.
        scene.effect = Some(renderer_schema::EffectV1 {
            shader: "fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32> {\n\
                     return this_identifier_does_not_exist;\n\
                     }"
            .into(),
        });
        let result = renderer.render_rgba(&scene);
        assert!(
            matches!(result, Err(RenderError::InvalidEffectShader(_))),
            "expected InvalidEffectShader, got {result:?}"
        );

        // Missing the required `effect` function signature entirely: the
        // template's fragment stage calls `effect(uv, color)`, which will
        // fail to resolve.
        scene.effect = Some(renderer_schema::EffectV1 {
            shader: "fn not_the_right_name(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32> {\n\
                     return color;\n\
                     }"
            .into(),
        });
        let result = renderer.render_rgba(&scene);
        assert!(
            matches!(result, Err(RenderError::InvalidEffectShader(_))),
            "expected InvalidEffectShader, got {result:?}"
        );

        // The renderer (and process) must still be usable afterwards.
        scene.effect = None;
        assert!(renderer.render_rgba(&scene).is_ok());
    }

    /// Perceptual golden-image coverage for text/image rasterization and
    /// scene composition, per the approved acceptance plan for the local
    /// text/image rasterization increment ("Use deterministic golden images
    /// to verify text/image placement, alpha blend, node ordering, and
    /// PNG/GIF output on a supported GPU host"). Skips gracefully (rather
    /// than failing) on a host with no GPU adapter, mirroring
    /// `renders_png_and_gif_on_an_available_gpu`.
    #[test]
    fn renders_golden_scenes_within_tolerance_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let asset_root = golden_asset_root();

        // Covers text placement, image placement, and alpha blending across
        // overlapping vector, text, and image nodes.
        let cases = [
            (
                "golden_text_image_placement.scene.json",
                "golden_text_image_placement.expected.png",
            ),
            (
                "golden_alpha_blend.scene.json",
                "golden_alpha_blend.expected.png",
            ),
            (
                "golden_order_image_first.scene.json",
                "golden_order_image_first.expected.png",
            ),
            (
                "golden_order_vector_first.scene.json",
                "golden_order_vector_first.expected.png",
            ),
        ];
        for (scene_file, golden_file) in cases {
            let scene = load_golden_scene(scene_file);
            let (pixels, warnings) = renderer
                .render_rgba_with_asset_root(&scene, &asset_root)
                .unwrap_or_else(|error| panic!("failed to render {scene_file}: {error}"));
            assert!(
                warnings.is_empty(),
                "{scene_file}: unexpected warnings: {warnings:?}"
            );
            assert_matches_golden(
                scene_file,
                &pixels,
                scene.canvas.width,
                scene.canvas.height,
                &asset_root.join(golden_file),
            );
        }

        // `golden_order_image_first.scene.json` and
        // `golden_order_vector_first.scene.json` declare the same
        // partially-transparent image and rect nodes in opposite order.
        // Composition is a strict painter's-algorithm pass over declaration
        // order, so swapping the order must change the blended result.
        let image_first = load_golden_scene("golden_order_image_first.scene.json");
        let vector_first = load_golden_scene("golden_order_vector_first.scene.json");
        let (image_first_pixels, _) = renderer
            .render_rgba_with_asset_root(&image_first, &asset_root)
            .unwrap();
        let (vector_first_pixels, _) = renderer
            .render_rgba_with_asset_root(&vector_first, &asset_root)
            .unwrap();
        assert_ne!(
            image_first_pixels, vector_first_pixels,
            "scene-declaration order must affect the composited output"
        );
    }

    /// Golden-image coverage for a scene-level full-canvas WGSL post-process
    /// effect (a color invert), using the same tolerance-and-skip pattern as
    /// `renders_golden_scenes_within_tolerance_on_an_available_gpu`. Fixture
    /// files use the `golden_effect_` prefix to avoid colliding with that
    /// test's fixtures.
    #[test]
    fn renders_a_golden_effect_scene_within_tolerance_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let asset_root = golden_asset_root();
        let scene_file = "golden_effect_invert.scene.json";
        let golden_file = "golden_effect_invert.expected.png";
        let scene = load_golden_scene(scene_file);
        assert!(scene.effect.is_some(), "{scene_file}: expected an effect");
        let (pixels, warnings) = renderer
            .render_rgba_with_asset_root(&scene, &asset_root)
            .unwrap_or_else(|error| panic!("failed to render {scene_file}: {error}"));
        assert!(
            warnings.is_empty(),
            "{scene_file}: unexpected warnings: {warnings:?}"
        );
        assert_matches_golden(
            scene_file,
            &pixels,
            scene.canvas.width,
            scene.canvas.height,
            &asset_root.join(golden_file),
        );
    }

    /// Separate golden-image test (kept out of
    /// `renders_golden_scenes_within_tolerance_on_an_available_gpu`'s
    /// fixture list to avoid conflicting with concurrent edits to that
    /// list) covering an SVG `Image` node composited alongside vector
    /// shapes and text, generated via `examples/generate_golden_svg.rs`.
    /// `tiny-skia`'s software rasterizer is fully deterministic given fixed
    /// input, so this fixture carries none of the GPU-driver-dependent
    /// pixel drift documented on `GOLDEN_MAX_CHANNEL_DELTA` above -- the
    /// same tolerance is reused anyway for consistency with the other
    /// golden tests (and to absorb the sRGB-blend rounding from the vector
    /// rect/text nodes it's composited with).
    #[test]
    fn renders_svg_golden_scene_within_tolerance_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let asset_root = golden_asset_root();
        let scene = load_golden_scene("golden_svg_placement.scene.json");
        let (pixels, warnings) = renderer
            .render_rgba_with_asset_root(&scene, &asset_root)
            .unwrap_or_else(|error| panic!("failed to render golden_svg_placement: {error}"));
        assert!(
            warnings.is_empty(),
            "golden_svg_placement.scene.json: unexpected warnings: {warnings:?}"
        );
        assert_matches_golden(
            "golden_svg_placement.scene.json",
            &pixels,
            scene.canvas.width,
            scene.canvas.height,
            &asset_root.join("golden_svg_placement.expected.png"),
        );
    }

    #[test]
    fn rasterizes_text_images_and_constrained_assets_without_a_gpu() {
        let directory = tempfile::tempdir().unwrap();
        let asset = directory.path().join("asset.png");
        image::RgbaImage::from_pixel(2, 2, image::Rgba([0, 255, 0, 255]))
            .save(&asset)
            .unwrap();
        let mut scene = test_scene();
        scene.canvas.width = 32;
        scene.canvas.height = 32;
        scene.nodes = vec![
            NodeV1 {
                id: "text".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Text {
                    x: 1.0,
                    y: 1.0,
                    text: "AA".into(),
                    size: 12.0,
                    fill: FillV1::Solid([1.0; 4]),
                },
            },
            NodeV1 {
                id: "vector-between".into(),
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
                    x: 20.0,
                    y: 20.0,
                    width: 4.0,
                    height: 4.0,
                    source: "asset.png".into(),
                },
            },
            NodeV1 {
                id: "image-again".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Image {
                    x: 24.0,
                    y: 20.0,
                    width: 4.0,
                    height: 4.0,
                    source: "asset.png".into(),
                },
            },
        ];
        let font = Font::from_bytes(
            include_bytes!("../assets/NotoSans-Regular.ttf") as &[u8],
            fontdue::FontSettings::default(),
        )
        .unwrap();
        let mut pixels = vec![0; 32 * 32 * 4];
        rasterize_text_and_images(&mut pixels, &scene, directory.path(), &font).unwrap();
        assert!(pixels.iter().any(|value| *value != 0));

        let plan =
            composition_plan(&scene, directory.path(), &font, &mut AssetCache::default()).unwrap();
        assert_eq!(plan.commands.len(), 5);
        assert_eq!(plan.textures.len(), 2);
        assert!(matches!(&plan.commands[0], DrawCommand::Textured { .. }));
        assert!(matches!(&plan.commands[1], DrawCommand::Textured { .. }));
        assert!(matches!(&plan.commands[2], DrawCommand::Primitive(_)));
        assert!(matches!(&plan.commands[3], DrawCommand::Textured { .. }));
        assert!(matches!(&plan.commands[4], DrawCommand::Textured { .. }));

        let mut primitives = test_scene();
        primitives.nodes.push(NodeV1 {
            id: "second-box".into(),
            translate: [0.0, 0.0],
            kind: NodeKindV1::Rect {
                x: 12.0,
                y: 1.0,
                width: 2.0,
                height: 2.0,
                corner_radius: 0.0,
                fill: FillV1::Solid([1.0; 4]),
            },
        });
        let primitive_plan = composition_plan(
            &primitives,
            directory.path(),
            &font,
            &mut AssetCache::default(),
        )
        .unwrap();
        assert_eq!(primitive_plan.commands.len(), 1);
        assert!(matches!(
            &primitive_plan.commands[0],
            DrawCommand::Primitive(vertices) if vertices == &(0..12)
        ));
        assert_eq!(upload_dimensions(4_096, 1, 1, 1), (1, 1));
        assert_eq!(
            upload_dimensions(4_096, 4_096, 4_096, 4_096),
            (2_048, 2_048)
        );
        assert!(matches!(
            resolve_asset(directory.path(), "../asset.png"),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            resolve_asset(directory.path(), "/asset.png"),
            Err(RenderError::Asset(_))
        ));
        blend_pixel(&mut pixels, 32, 32, -1, 0, [1.0; 4]);

        let mut destination = vec![255, 255, 255, 255];
        // GPU vector layers are read back in premultiplied-alpha form.
        blend_premultiplied_layer(&mut destination, &[128, 0, 0, 128], 1, 1);
        assert_eq!(destination, vec![255, 127, 127, 255]);

        assert!(matches!(
            bounded_image_dimension(f32::MAX, "width"),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            bounded_image_dimension(4_097.0, "width"),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            ensure_source_image_dimensions(4_001, 4_000, "oversized.png"),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            ensure_target_image_dimensions(4_096, 4_096, "oversized.png"),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            reserve_composition_pixels(MAX_COMPOSITION_TEXTURE_PIXELS, 1),
            Err(RenderError::Asset(_))
        ));

        scene.nodes = vec![NodeV1 {
            id: "oversized-text".into(),
            translate: [0.0, 0.0],
            kind: NodeKindV1::Text {
                x: 0.0,
                y: 0.0,
                text: "A".into(),
                size: 2_000.0,
                fill: FillV1::Solid([1.0; 4]),
            },
        }];
        assert!(matches!(
            rasterize_text_and_images(&mut pixels, &scene, directory.path(), &font),
            Err(RenderError::Asset(_))
        ));
    }

    #[test]
    fn rasterizes_svg_assets_to_declared_dimensions_without_a_gpu() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("badge.svg"),
            br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
<rect width="10" height="10" fill="#ff0000"/>
</svg>"##,
        )
        .unwrap();

        // Rasterizing directly at a non-square declared size (rather than
        // decoding at some source resolution and bilinearly rescaling)
        // should still produce an exact widthxheight buffer with crisp,
        // uniform color -- there is nothing to blur since the whole
        // viewBox is one flat rect.
        let image = load_image(directory.path(), "badge.svg", 40, 20).unwrap();
        assert_eq!(image.width(), 40);
        assert_eq!(image.height(), 20);
        for pixel in image.pixels() {
            assert_eq!(pixel.0, [255, 0, 0, 255]);
        }
    }

    #[test]
    fn rejects_svg_assets_that_escape_the_asset_root() {
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("nested");
        fs::create_dir(&nested).unwrap();

        // Same two escape shapes already covered for raster images by the
        // `resolve_asset` assertions in
        // `rasterizes_text_images_and_constrained_assets_without_a_gpu`
        // above: `..` traversal and an absolute path. `load_image` routes
        // every source (SVG included) through the exact same
        // `resolve_asset` call, so both are rejected before the file is
        // ever opened.
        assert!(matches!(
            load_image(&nested, "../escape.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            load_image(directory.path(), "/absolute-escape.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));
    }

    #[test]
    fn rejects_malformed_and_oversized_svg_assets_without_panicking() {
        let directory = tempfile::tempdir().unwrap();

        fs::write(
            directory.path().join("malformed.svg"),
            b"<svg><unterminated",
        )
        .unwrap();
        assert!(matches!(
            load_image(directory.path(), "malformed.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));

        // Named `.svg` but the content sniff should refuse to hand this to
        // the XML parser at all.
        fs::write(
            directory.path().join("not-svg.svg"),
            b"this has an .svg extension but is not svg content",
        )
        .unwrap();
        assert!(matches!(
            load_image(directory.path(), "not-svg.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));

        // Same 16 MiB cap the raster (`image` crate) path already enforces
        // via `MAX_ASSET_BYTES`, applied before the file is ever parsed.
        let mut oversized = b"<svg xmlns=\"http://www.w3.org/2000/svg\">".to_vec();
        oversized.resize(17 * 1024 * 1024, b' ');
        oversized.extend_from_slice(b"</svg>");
        fs::write(directory.path().join("oversized.svg"), &oversized).unwrap();
        assert!(matches!(
            load_image(directory.path(), "oversized.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));
    }

    #[test]
    fn refuses_to_follow_embedded_svg_image_references_outside_the_asset_root() {
        let directory = tempfile::tempdir().unwrap();

        // A file outside the configured asset root that a hostile SVG will
        // try to pull in via an absolute-path `<image href>`.
        let secret = directory.path().join("secret.png");
        image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]))
            .save(&secret)
            .unwrap();

        let asset_root = directory.path().join("assets");
        fs::create_dir(&asset_root).unwrap();
        fs::write(
            asset_root.join("evil.svg"),
            format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 4 4">
<rect width="4" height="4" fill="#00ff00"/>
<image xlink:href="{}" width="4" height="4"/>
</svg>"##,
                secret.display()
            ),
        )
        .unwrap();

        let image = load_image(&asset_root, "evil.svg", 4, 4).unwrap();
        // The embedded absolute-path href must be refused outright (see the
        // security-posture comment on `rasterize_svg`): only the green
        // background rect should ever be visible, never the referenced
        // file's red pixels.
        for pixel in image.pixels() {
            assert_eq!(pixel.0, [0, 255, 0, 255]);
        }
    }

    /// Portable (no GPU required) proof that `AssetCache` actually caches:
    /// this directly drives `composition_plan` the same way
    /// `render_gif_with_asset_root` does for each frame of a GIF export --
    /// one call per animation frame, `scene_at`-ing the base scene for each
    /// frame's timestamp -- and shows the decode/rasterize call count for a
    /// static image and static glyphs is exactly 1 per distinct asset when a
    /// single `AssetCache` is shared across frames, versus 1 *per frame*
    /// (the pre-change baseline) when each call gets its own fresh cache, as
    /// `composition_plan` built locally before this change.
    #[test]
    fn shares_a_decoded_asset_cache_across_simulated_gif_frames() {
        let directory = tempfile::tempdir().unwrap();
        image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]))
            .save(directory.path().join("logo.png"))
            .unwrap();
        let font = Font::from_bytes(
            include_bytes!("../assets/NotoSans-Regular.ttf") as &[u8],
            fontdue::FontSettings::default(),
        )
        .unwrap();

        let mut scene = test_scene();
        scene.canvas.width = 32;
        scene.canvas.height = 32;
        scene.nodes = vec![
            // Keyframed: this is the only thing that differs frame to frame.
            NodeV1 {
                id: "animated-box".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 1.0,
                    y: 1.0,
                    width: 4.0,
                    height: 4.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0, 0.0, 0.0, 1.0]),
                },
            },
            // Static across every frame: no keyframe targets it.
            NodeV1 {
                id: "label".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Text {
                    x: 2.0,
                    y: 10.0,
                    text: "AB".into(),
                    size: 12.0,
                    fill: FillV1::Solid([1.0, 1.0, 1.0, 1.0]),
                },
            },
            // Static across every frame: no keyframe targets it.
            NodeV1 {
                id: "logo".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Image {
                    x: 20.0,
                    y: 20.0,
                    width: 4.0,
                    height: 4.0,
                    source: "logo.png".into(),
                },
            },
        ];
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 5,
            duration_ms: 1_000,
            keyframes: vec![
                KeyframeV1 {
                    at_ms: 0,
                    target: "animated-box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(0.0),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "animated-box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(1.0),
                },
            ],
        });
        scene.validate().unwrap();

        let timeline = scene.timeline.as_ref().unwrap();
        let frame_count =
            (u64::from(timeline.duration_ms) * u64::from(timeline.fps)).div_ceil(1_000) as u32;
        let fps = u32::from(timeline.fps);
        assert_eq!(frame_count, 5, "sanity check on the fixture's frame math");

        // Baseline: mirrors `composition_plan`'s behavior before this
        // change, where every call got its own fresh, function-local cache
        // -- so every frame independently decodes the static image and
        // rasterizes the static glyphs.
        for frame_index in 0..frame_count {
            let at_ms = frame_index * 1_000 / fps;
            let animated = scene_at(&scene, at_ms);
            let mut fresh_cache = AssetCache::default();
            composition_plan(&animated, directory.path(), &font, &mut fresh_cache).unwrap();
            assert_eq!(
                fresh_cache.image_decode_count(),
                1,
                "a fresh per-frame cache decodes the static image once per frame"
            );
            assert_eq!(
                fresh_cache.glyph_rasterization_count(),
                2,
                "a fresh per-frame cache rasterizes both static glyphs ('A' and 'B') once per frame"
            );
        }

        // Under test: one `AssetCache` shared across every frame -- exactly
        // what `render_gif_with_asset_root` now does -- must decode/
        // rasterize each distinct static asset exactly once for the *whole*
        // multi-frame export, not once per frame.
        let mut shared_cache = AssetCache::default();
        for frame_index in 0..frame_count {
            let at_ms = frame_index * 1_000 / fps;
            let animated = scene_at(&scene, at_ms);
            composition_plan(&animated, directory.path(), &font, &mut shared_cache).unwrap();
        }
        assert_eq!(
            shared_cache.image_decode_count(),
            1,
            "the static image must be decoded exactly once across all {frame_count} frames \
             sharing one AssetCache, not once per frame"
        );
        assert_eq!(
            shared_cache.glyph_rasterization_count(),
            2,
            "each of the 2 distinct static glyphs must be rasterized exactly once across all \
             {frame_count} frames sharing one AssetCache, not once per frame"
        );
    }

    /// GPU smoke coverage (skips gracefully with no adapter, matching this
    /// file's other `_on_an_available_gpu` tests) for the same scenario as
    /// `shares_a_decoded_asset_cache_across_simulated_gif_frames` above, but
    /// driven through the real public `render_gif_with_asset_root` entry
    /// point end to end: a multi-frame export with a keyframed node plus
    /// static image/text nodes must still produce a correct, valid,
    /// warning-free GIF now that the decode/rasterize cache is shared across
    /// frames.
    #[test]
    fn renders_a_gif_with_static_assets_across_frames_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let directory = tempfile::tempdir().unwrap();
        image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]))
            .save(directory.path().join("logo.png"))
            .unwrap();

        let mut scene = test_scene();
        scene.canvas.width = 32;
        scene.canvas.height = 32;
        scene.nodes = vec![
            NodeV1 {
                id: "animated-box".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 1.0,
                    y: 1.0,
                    width: 4.0,
                    height: 4.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0, 0.0, 0.0, 1.0]),
                },
            },
            NodeV1 {
                id: "label".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Text {
                    x: 2.0,
                    y: 10.0,
                    text: "AB".into(),
                    size: 12.0,
                    fill: FillV1::Solid([1.0, 1.0, 1.0, 1.0]),
                },
            },
            NodeV1 {
                id: "logo".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Image {
                    x: 20.0,
                    y: 20.0,
                    width: 4.0,
                    height: 4.0,
                    source: "logo.png".into(),
                },
            },
        ];
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 5,
            duration_ms: 1_000,
            keyframes: vec![
                KeyframeV1 {
                    at_ms: 0,
                    target: "animated-box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(0.0),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "animated-box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(1.0),
                },
            ],
        });

        let output = directory.path().join("scene.gif");
        let rendered = renderer
            .render_gif_with_asset_root(&scene, &output, directory.path())
            .unwrap();
        assert_eq!(rendered.frame_count, 5);
        assert!(rendered.warnings.is_empty());
        assert!(
            fs::metadata(&output).unwrap().len() > 0,
            "GIF output must not be empty"
        );
        let decoded = image::open(&output).unwrap_or_else(|error| {
            panic!("render_gif_with_asset_root did not produce a valid, decodable GIF: {error}")
        });
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 32);
    }

    // ---- Analytic (SDF + fwidth) anti-aliasing "shadow mode" tests ----
    //
    // New, separately-named tests only: none of these touch, modify, or
    // regenerate any existing golden image or existing test above, and none
    // of the existing tests above were changed to make room for these.

    /// Basic functional coverage of the analytic-AA path across every node
    /// kind this renderer supports, including the documented `Path` MSAA
    /// fallback (see `composition_plan_analytic`'s doc comment) and an image
    /// asset. Mirrors the spirit of `compiles_all_geometry_and_reports_
    /// unrasterized_nodes` (portable) and `renders_png_and_gif_on_an_
    /// available_gpu` (GPU) above, but actually renders through the new
    /// `_analytic_aa` entry points end to end.
    #[test]
    fn analytic_aa_renders_every_node_kind_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let directory = tempfile::tempdir().unwrap();
        image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]))
            .save(directory.path().join("logo.png"))
            .unwrap();

        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![
                NodeV1 {
                    id: "rect".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Rect {
                        x: 2.0,
                        y: 2.0,
                        width: 10.0,
                        height: 10.0,
                        corner_radius: 0.0,
                        fill: FillV1::Solid([1.0, 0.0, 0.0, 1.0]),
                    },
                },
                NodeV1 {
                    id: "ellipse".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Ellipse {
                        cx: 30.0,
                        cy: 10.0,
                        rx: 6.0,
                        ry: 4.0,
                        fill: FillV1::Solid([0.0, 1.0, 0.0, 1.0]),
                    },
                },
                NodeV1 {
                    id: "line".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Line {
                        x1: 4.0,
                        y1: 30.0,
                        x2: 40.0,
                        y2: 50.0,
                        thickness: 3.0,
                        fill: FillV1::Solid([0.0, 0.0, 1.0, 1.0]),
                    },
                },
                NodeV1 {
                    id: "path".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Path {
                        points: vec![
                            renderer_schema::PointV1 { x: 45.0, y: 5.0 },
                            renderer_schema::PointV1 { x: 60.0, y: 5.0 },
                            renderer_schema::PointV1 { x: 52.0, y: 20.0 },
                        ],
                        fill: FillV1::Solid([0.5, 0.0, 0.5, 1.0]),
                    },
                },
                NodeV1 {
                    id: "text".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Text {
                        x: 4.0,
                        y: 44.0,
                        text: "Hi".into(),
                        size: 12.0,
                        fill: FillV1::Solid([0.0, 0.0, 0.0, 1.0]),
                    },
                },
                NodeV1 {
                    id: "logo".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Image {
                        x: 44.0,
                        y: 44.0,
                        width: 8.0,
                        height: 8.0,
                        source: "logo.png".into(),
                    },
                },
            ],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer
            .render_rgba_with_asset_root_analytic_aa(&scene, directory.path())
            .unwrap();
        assert!(warnings.is_empty());
        assert_eq!(pixels.len(), 64 * 64 * 4);

        // Every non-background color used above must appear somewhere in the
        // output (allowing for AA blending, so this checks for "close to"
        // rather than an exact byte match) -- proof each node kind actually
        // rasterized something instead of silently no-opping.
        let expects = [
            ("rect (red)", [255u8, 0, 0]),
            ("ellipse (green)", [0, 255, 0]),
            ("line (blue)", [0, 0, 255]),
            // Not a naive linear-to-255 mapping (0.5*255=128): this
            // renderer's output is sRGB-gamma-encoded, and gamma-encoding
            // linear 0.5 gives ~0.735, i.e. ~188/255 -- confirmed against
            // the actual rendered pixel value. Every other entry in this
            // list happens to use a pure 0.0/1.0 channel value, which gamma
            // encoding leaves unchanged, so this is the only one affected.
            ("path (purple)", [188, 0, 188]),
            ("image (logo)", [10, 20, 30]),
        ];
        for (label, target) in expects {
            let found = pixels.as_chunks::<4>().0.iter().any(|pixel| {
                (0..3).all(|channel| (pixel[channel] as i16 - target[channel] as i16).abs() <= 12)
            });
            assert!(
                found,
                "{label}: expected a pixel close to {target:?} in the analytic-AA render"
            );
        }

        // A round trip through `render_png_with_asset_root_analytic_aa`
        // works end to end too.
        let png = renderer
            .render_png_with_asset_root_analytic_aa(
                &scene,
                &directory.path().join("out.png"),
                directory.path(),
            )
            .unwrap();
        assert_eq!(png.width, 64);
        assert_eq!(png.height, 64);
        assert_eq!(png.frame_count, 1);
    }

    /// Analytic-AA counterpart to `anti_aliases_diagonal_primitive_edges_
    /// on_an_available_gpu` above: same diagonal-line scene, same "at least
    /// one pixel strictly between background and line color" check, but
    /// rendered through `render_rgba_analytic_aa` instead of `render_rgba`.
    /// Confirms the SDF/`fwidth` pipeline produces real coverage-weighted
    /// blending, not a hard binary edge, exactly like the existing MSAA path
    /// -- from an entirely separate pipeline/shader.
    #[test]
    fn analytic_aa_anti_aliases_diagonal_primitive_edges_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = diagonal_line_scene();
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba_analytic_aa(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };
        let background_pixel = pixel_at(0, 0);
        let line_pixel = pixel_at(32, 32);
        assert_ne!(background_pixel, line_pixel);

        let mut found_blended_pixel = false;
        'scan: for y in 0..scene.canvas.height as usize {
            for x in 0..width {
                let pixel = pixel_at(x, y);
                let strictly_between = (0..3).all(|channel| {
                    let low = background_pixel[channel].min(line_pixel[channel]);
                    let high = background_pixel[channel].max(line_pixel[channel]);
                    pixel[channel] > low && pixel[channel] < high
                });
                if strictly_between {
                    found_blended_pixel = true;
                    break 'scan;
                }
            }
        }
        assert!(
            found_blended_pixel,
            "expected at least one analytically anti-aliased pixel strictly between the \
             background color {background_pixel:?} and the line color {line_pixel:?}"
        );
    }

    /// Confirms `Path` nodes still render via the documented MSAA fallback
    /// (see `composition_plan_analytic`'s doc comment) *and* that draw order
    /// across heterogeneous pipelines is preserved: a `Path` node sandwiched
    /// between two analytic-pipeline `Rect` nodes must still composite in
    /// document order (later nodes drawn on top of earlier ones), even
    /// though the multi-pass design (`render_composed_rgba_analytic_with_
    /// cache`) executes the `Path` node in a separate, differently-sampled
    /// render pass from its analytic-pipeline neighbors.
    #[test]
    fn analytic_aa_preserves_z_order_across_the_path_msaa_fallback_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 32,
                height: 32,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![
                // Bottom: a big opaque red square covering the whole canvas.
                NodeV1 {
                    id: "background-rect".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Rect {
                        x: 0.0,
                        y: 0.0,
                        width: 32.0,
                        height: 32.0,
                        corner_radius: 0.0,
                        fill: FillV1::Solid([1.0, 0.0, 0.0, 1.0]),
                    },
                },
                // Middle: an opaque green path covering the whole canvas --
                // must fully occlude the red rect beneath it.
                NodeV1 {
                    id: "middle-path".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Path {
                        points: vec![
                            renderer_schema::PointV1 { x: 0.0, y: 0.0 },
                            renderer_schema::PointV1 { x: 32.0, y: 0.0 },
                            renderer_schema::PointV1 { x: 32.0, y: 32.0 },
                            renderer_schema::PointV1 { x: 0.0, y: 32.0 },
                        ],
                        fill: FillV1::Solid([0.0, 1.0, 0.0, 1.0]),
                    },
                },
                // Top: a small opaque blue square -- must occlude the green
                // path beneath it, proving the *next* analytic-pipeline pass
                // still loads (not clears) the prior Path pass's output.
                NodeV1 {
                    id: "top-rect".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Rect {
                        x: 8.0,
                        y: 8.0,
                        width: 8.0,
                        height: 8.0,
                        corner_radius: 0.0,
                        fill: FillV1::Solid([0.0, 0.0, 1.0, 1.0]),
                    },
                },
            ],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba_analytic_aa(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };
        // Far from every edge, so no AA blending is in play: a corner (only
        // the green path should be visible -- proving it occluded the red
        // rect) and the small blue square's center (proving it occluded the
        // green path).
        let corner = pixel_at(2, 2);
        let top = pixel_at(12, 12);
        assert!(
            corner[1] > 200 && corner[0] < 40 && corner[2] < 40,
            "expected the green path to occlude the red rect beneath it at a corner far from \
             any edge, got {corner:?}"
        );
        assert!(
            top[2] > 200 && top[0] < 40 && top[1] < 40,
            "expected the blue rect to occlude the green path beneath it at its center, got {top:?}"
        );
    }

    /// GIF export through the analytic-AA path works end to end, mirroring
    /// `renders_png_and_gif_on_an_available_gpu` above but via `render_gif_
    /// analytic_aa`.
    #[test]
    fn analytic_aa_renders_gif_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let directory = tempfile::tempdir().unwrap();
        let mut scene = test_scene();
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 2,
            duration_ms: 1_000,
            keyframes: vec![],
        });
        let gif = renderer
            .render_gif_analytic_aa(&scene, &directory.path().join("scene.gif"))
            .unwrap();
        assert_eq!(gif.frame_count, 2);
        assert!(gif.warnings.is_empty());
        let decoded = image::open(directory.path().join("scene.gif")).unwrap_or_else(|error| {
            panic!("render_gif_analytic_aa did not produce a valid, decodable GIF: {error}")
        });
        assert_eq!(decoded.width(), scene.canvas.width);
        assert_eq!(decoded.height(), scene.canvas.height);
    }

    /// The diagonal-line scene shared by the MSAA/SSAA and analytic-AA
    /// anti-aliasing tests/quality comparison, factored out so both sides of
    /// the comparison render *exactly* the same geometry.
    fn diagonal_line_scene() -> SceneV1 {
        SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "diagonal".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Line {
                    x1: 6.0,
                    y1: 6.0,
                    x2: 58.0,
                    y2: 58.0,
                    thickness: 6.0,
                    fill: FillV1::Solid([0.2, 0.2, 0.2, 1.0]),
                },
            }],
            timeline: None,
            effect: None,
        }
    }

    /// A shallow-angle (not 45°) diagonal line, otherwise matching
    /// `diagonal_line_scene`'s style (same canvas size, thickness, color).
    /// Used only by `analytic_aa_matches_numeric_ground_truth_coverage_
    /// better_than_msaa_supersampling_on_an_available_gpu`, which needs to
    /// sample real *rendered pixels* (necessarily at integer positions)
    /// spanning several distinct true-coverage bands. A 45° line's AA
    /// transition band is only ~1.4px wide in x (each 1px step in x moves
    /// ~0.7px perpendicular to the edge, and the AA band itself is only
    /// about 1px wide), too narrow to contain 5 well-separated integer-pixel
    /// samples -- confirmed numerically: scanning `diagonal_line_scene`'s
    /// 45° line finds only 1 of 5 target coverage bands at integer
    /// resolution, no matter how wide a window is scanned. A shallow slope
    /// spreads that same true 1px-wide transition band across many more
    /// integer x steps, making it actually samplable. Kept as its own
    /// fixture (rather than changing `diagonal_line_scene` itself) so the
    /// pre-existing tests already calibrated against the 45° line are
    /// untouched.
    fn shallow_diagonal_line_scene() -> SceneV1 {
        SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "shallow-diagonal".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Line {
                    x1: 4.0,
                    y1: 20.0,
                    x2: 60.0,
                    y2: 30.0,
                    thickness: 6.0,
                    fill: FillV1::Solid([0.2, 0.2, 0.2, 1.0]),
                },
            }],
            timeline: None,
            effect: None,
        }
    }

    /// Numerically estimates the *true* fractional pixel-area coverage of a
    /// capsule (2D thick line segment, matching `capsule_sdf` in
    /// `ANALYTIC_SHADER` and `add_line_analytic`'s geometry) at raster pixel
    /// `(pixel_x, pixel_y)`, by regularly subsampling that pixel's
    /// continuous `[pixel_x, pixel_x+1) x [pixel_y, pixel_y+1)` region (in
    /// the same scene-pixel coordinate space `vertex()`/`add_line` use --
    /// scene x/y coordinates map 1:1 onto continuous framebuffer pixel
    /// coordinates, since `vertex()`'s `x / canvas.width * 2 - 1` clip-space
    /// transform is exactly the inverse of the standard NDC-to-viewport
    /// transform) and computing what fraction of subsample points the exact
    /// capsule SDF (no shader approximation, no `fwidth`) classifies as
    /// inside.
    ///
    /// This is deliberately independent of *both* renderers under test: it
    /// does not call `capsule_sdf`/`ANALYTIC_SHADER` (the analytic path's
    /// own formula) or rely on MSAA/supersampling in any way, so comparing
    /// each renderer's actual output against this number is a fair,
    /// non-circular ground-truth check for both.
    /// Standard sRGB EOTF (byte 0-255 -> linear 0.0-1.0), matching the
    /// `Rgba8UnormSrgb` render target format this renderer uses throughout.
    fn srgb_decode_byte(byte: f32) -> f32 {
        let c = byte / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Inverse of `srgb_decode_byte` (linear 0.0-1.0 -> byte 0-255).
    fn srgb_encode_byte(linear: f32) -> f32 {
        let c = if linear <= 0.003_130_8 {
            linear * 12.92
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        c * 255.0
    }

    fn linear_mix(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }

    fn capsule_coverage_numeric(
        pixel_x: i32,
        pixel_y: i32,
        a: [f32; 2],
        b: [f32; 2],
        half_thickness: f32,
        subsamples: u32,
    ) -> f32 {
        let ba = [b[0] - a[0], b[1] - a[1]];
        let ba_len_sq = ba[0] * ba[0] + ba[1] * ba[1];
        let mut inside = 0u32;
        for row in 0..subsamples {
            for column in 0..subsamples {
                let sx = pixel_x as f32 + (column as f32 + 0.5) / subsamples as f32;
                let sy = pixel_y as f32 + (row as f32 + 0.5) / subsamples as f32;
                let pa = [sx - a[0], sy - a[1]];
                let h = ((pa[0] * ba[0] + pa[1] * ba[1]) / ba_len_sq).clamp(0.0, 1.0);
                let dx = pa[0] - ba[0] * h;
                let dy = pa[1] - ba[1] * h;
                let distance = (dx * dx + dy * dy).sqrt() - half_thickness;
                if distance < 0.0 {
                    inside += 1;
                }
            }
        }
        inside as f32 / (subsamples * subsamples) as f32
    }

    /// The core quality-comparison test: renders the exact same diagonal-
    /// line scene (`diagonal_line_scene`) through both the existing
    /// MSAA+supersampling path (`render_rgba`, completely untouched by this
    /// change) and the new analytic SDF/`fwidth` path
    /// (`render_rgba_analytic_aa`), then checks each renderer's output
    /// against a numerically-estimated *ground-truth* coverage
    /// (`capsule_coverage_numeric`, 64x64 subsamples per pixel -- computed
    /// from the exact capsule geometry, independent of either renderer's own
    /// internals) at several sample pixels spanning a range of true coverage
    /// fractions along the line's edge.
    ///
    /// A coverage fraction is turned into an "expected" byte value by
    /// decoding the scene's actual rendered pure background/line-interior
    /// colors from sRGB to linear, interpolating *there*, then re-encoding
    /// (`srgb_decode_byte`/`srgb_encode_byte`/`linear_mix` below) -- not
    /// naive byte-space interpolation (which the simpler pre-existing
    /// `anti_aliases_diagonal_primitive_edges_on_an_available_gpu`/
    /// `supersamples_diagonal_primitive_edges_beyond_msaa_alone_on_an_
    /// available_gpu` tests above use, since they only check "is there any
    /// blending at all", a check loose enough not to care). This one
    /// computes real numeric error against ground truth, and the render
    /// target is `Rgba8UnormSrgb` -- the GPU blends in linear space -- so
    /// byte-space interpolation was measured to disagree with real output
    /// by up to ~23 (of 255) at mid-range coverage, large enough to make
    /// this comparison meaningless without the gamma-correct version.
    ///
    /// Thresholds have real headroom above/below what's actually measured
    /// on this machine's real GPU (Apple M1/Metal) with this scene: analytic
    /// mean absolute error ~4.3 (max ~9.9), MSAA+supersampling mean ~7.3
    /// (max ~16.9) -- so ordinary cross-GPU/driver rounding differences
    /// shouldn't make this flaky, while a real regression in either path's
    /// edge quality, or the two techniques becoming indistinguishable,
    /// still fails it. See `shallow_diagonal_line_scene`'s doc comment for
    /// why this test uses a shallow-angle line rather than
    /// `diagonal_line_scene`'s 45° one.
    #[test]
    fn analytic_aa_matches_numeric_ground_truth_coverage_better_than_msaa_supersampling_on_an_available_gpu()
     {
        const MAX_MEAN_ANALYTIC_ERROR: f32 = 7.0;
        const MIN_MEAN_MSAA_ERROR: f32 = 5.0;

        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = shallow_diagonal_line_scene();
        scene.validate().unwrap();

        let (msaa_pixels, msaa_warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(msaa_warnings.is_empty());
        let (analytic_pixels, analytic_warnings) =
            renderer.render_rgba_analytic_aa(&scene).unwrap();
        assert!(analytic_warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };
        // Pure endpoint colors, measured from the actual GPU output (same
        // approach the pre-existing anti-aliasing tests above use) rather
        // than assumed from the scene's linear input color, since this
        // renderer's actual color-space handling is an internal detail
        // neither this test nor the pre-existing ones above depend on.
        // (32, 25) is deep in the shallow line's interior -- its centerline
        // passes through y=25 at x=32 -- and far from either rounded cap.
        let background_byte = pixel_at(&msaa_pixels, 0, 0)[0] as f32;
        let line_byte = pixel_at(&msaa_pixels, 32, 25)[0] as f32;
        assert_eq!(
            pixel_at(&analytic_pixels, 0, 0)[0] as f32,
            background_byte,
            "both paths must render the exact same flat background color away from any edge"
        );
        assert!(
            (pixel_at(&analytic_pixels, 32, 25)[0] as f32 - line_byte).abs() <= 2.0,
            "both paths must render essentially the same line-interior color far from any edge"
        );

        let line_a = [4.0_f32, 20.0];
        let line_b = [60.0_f32, 30.0];
        let half_thickness = 3.0_f32;

        let mut analytic_errors = Vec::new();
        let mut msaa_errors = Vec::new();
        let mut sampled_coverages = Vec::new();
        // Scan a window straddling the diagonal edge (inset from both the
        // canvas border and the line's rounded end caps) and keep pixels
        // whose true numeric coverage lands in a set of well-separated
        // bands, so the sampled set spans a real range of coverage
        // fractions rather than clustering near one value.
        let mut remaining_bands: Vec<(f32, f32)> =
            vec![(0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95)];
        'scan: for y in 4..40usize {
            for x in 10..56usize {
                let coverage = capsule_coverage_numeric(
                    x as i32,
                    y as i32,
                    line_a,
                    line_b,
                    half_thickness,
                    64,
                );
                if let Some(band_index) = remaining_bands
                    .iter()
                    .position(|(low, high)| coverage >= *low && coverage < *high)
                {
                    // NOT naive byte-space linear interpolation: the render
                    // target is `Rgba8UnormSrgb`, so the GPU blends
                    // `coverage`-weighted colors in *linear* space and then
                    // gamma-encodes the result for storage. Byte-space
                    // interpolation between `background_byte`/`line_byte`
                    // follows a visibly different curve (most divergent
                    // around 40-60% coverage -- confirmed empirically: it
                    // was off by 9-22 bytes here before this fix), so the
                    // "expected" value must decode both endpoints to linear,
                    // interpolate there, then re-encode.
                    let expected = srgb_encode_byte(linear_mix(
                        srgb_decode_byte(background_byte),
                        srgb_decode_byte(line_byte),
                        coverage,
                    ));
                    let analytic_actual = pixel_at(&analytic_pixels, x, y)[0] as f32;
                    let msaa_actual = pixel_at(&msaa_pixels, x, y)[0] as f32;
                    analytic_errors.push((analytic_actual - expected).abs());
                    msaa_errors.push((msaa_actual - expected).abs());
                    sampled_coverages.push(coverage);
                    remaining_bands.remove(band_index);
                    if remaining_bands.is_empty() {
                        break 'scan;
                    }
                }
            }
        }
        assert!(
            sampled_coverages.len() >= 4,
            "expected to find pixels spanning most of the coverage-fraction bands near the \
             diagonal edge (found {} of 5: {sampled_coverages:?})",
            sampled_coverages.len()
        );

        let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
        let analytic_mean_error = mean(&analytic_errors);
        let msaa_mean_error = mean(&msaa_errors);

        assert!(
            analytic_mean_error <= MAX_MEAN_ANALYTIC_ERROR,
            "analytic-AA mean absolute byte error against numeric ground-truth coverage was \
             {analytic_mean_error}, expected <= {MAX_MEAN_ANALYTIC_ERROR} \
             (per-sample errors: {analytic_errors:?}, coverages: {sampled_coverages:?})"
        );
        assert!(
            msaa_mean_error >= analytic_mean_error,
            "expected the analytic-AA path's mean absolute error against numeric ground truth \
             ({analytic_mean_error}) to be no worse than the existing MSAA+supersampling path's \
             ({msaa_mean_error}) on identical geometry -- analytic AA should match ground truth \
             at least as well since it computes an exact per-fragment distance instead of \
             estimating coverage from a fixed sample grid"
        );
        // Loosely confirms MSAA+supersampling's error is in the range this
        // project has already measured for it (see this test's doc comment)
        // rather than accidentally testing two near-identical numbers.
        assert!(
            msaa_mean_error >= MIN_MEAN_MSAA_ERROR,
            "expected the existing MSAA+supersampling path's mean absolute error ({msaa_mean_error}) \
             to be at least {MIN_MEAN_MSAA_ERROR} on this scene, matching this project's prior \
             measurements (see this test's doc comment) -- if not, this comparison may no longer \
             be meaningfully distinguishing the two techniques"
        );
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

    /// GPU pixel-decode proof (default MSAA pipeline) that `corner_radius`
    /// actually rounds a rect's corners rather than merely not crashing:
    /// renders the same bounding box twice, once sharp (`corner_radius:
    /// 0.0`) and once rounded (`corner_radius: 10.0`), and confirms a pixel
    /// near the bounding box's corner is shape-colored in the sharp render
    /// but background-colored in the rounded render.
    #[test]
    fn rounds_rect_corners_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };

        fn scene_with_radius(corner_radius: f32) -> SceneV1 {
            SceneV1 {
                version: SCENE_VERSION_V1.into(),
                canvas: CanvasV1 {
                    width: 40,
                    height: 40,
                    background: [1.0, 1.0, 1.0, 1.0],
                },
                nodes: vec![NodeV1 {
                    id: "box".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Rect {
                        x: 4.0,
                        y: 4.0,
                        width: 32.0,
                        height: 32.0,
                        corner_radius,
                        fill: FillV1::Solid([0.0, 0.0, 0.0, 1.0]),
                    },
                }],
                timeline: None,
                effect: None,
            }
        }

        let width = 40_usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let sharp = scene_with_radius(0.0);
        sharp.validate().unwrap();
        let (sharp_pixels, sharp_warnings) = renderer.render_rgba(&sharp).unwrap();
        assert!(sharp_warnings.is_empty());

        let rounded = scene_with_radius(10.0);
        rounded.validate().unwrap();
        let (rounded_pixels, rounded_warnings) = renderer.render_rgba(&rounded).unwrap();
        assert!(rounded_warnings.is_empty());

        // (5, 5) sits just inside the shared 4..36 bounding box, near its
        // top-left corner: distance to the top-left arc's center (14, 14)
        // at radius 10 is ~12.7px, i.e. outside the rounded arc but well
        // inside the sharp box.
        let background = pixel_at(&sharp_pixels, 0, 0);
        let sharp_corner = pixel_at(&sharp_pixels, 5, 5);
        let rounded_corner = pixel_at(&rounded_pixels, 5, 5);

        assert_ne!(
            sharp_corner, background,
            "sanity check: a sharp rect's bounding-box corner must be shape-colored"
        );
        assert_eq!(
            rounded_corner, background,
            "expected a corner_radius: 10.0 rect's corner pixel to be background-colored \
             (proving the corner is actually rounded away), got {rounded_corner:?} vs. \
             background {background:?}"
        );
    }

    /// Analytic-AA counterpart to `rounds_rect_corners_on_an_available_gpu`:
    /// same proof, but through `render_rgba_analytic_aa`, confirming the
    /// analytic pipeline's `rect_sdf`/`param1`-carried radius also actually
    /// rounds corners rather than silently ignoring `corner_radius`.
    #[test]
    fn rounds_rect_corners_under_analytic_aa_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };

        fn scene_with_radius(corner_radius: f32) -> SceneV1 {
            SceneV1 {
                version: SCENE_VERSION_V1.into(),
                canvas: CanvasV1 {
                    width: 40,
                    height: 40,
                    background: [1.0, 1.0, 1.0, 1.0],
                },
                nodes: vec![NodeV1 {
                    id: "box".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Rect {
                        x: 4.0,
                        y: 4.0,
                        width: 32.0,
                        height: 32.0,
                        corner_radius,
                        fill: FillV1::Solid([0.0, 0.0, 0.0, 1.0]),
                    },
                }],
                timeline: None,
                effect: None,
            }
        }

        let width = 40_usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let sharp = scene_with_radius(0.0);
        sharp.validate().unwrap();
        let (sharp_pixels, sharp_warnings) = renderer.render_rgba_analytic_aa(&sharp).unwrap();
        assert!(sharp_warnings.is_empty());

        let rounded = scene_with_radius(10.0);
        rounded.validate().unwrap();
        let (rounded_pixels, rounded_warnings) =
            renderer.render_rgba_analytic_aa(&rounded).unwrap();
        assert!(rounded_warnings.is_empty());

        let background = pixel_at(&sharp_pixels, 0, 0);
        let sharp_corner = pixel_at(&sharp_pixels, 5, 5);
        let rounded_corner = pixel_at(&rounded_pixels, 5, 5);

        assert_ne!(
            sharp_corner, background,
            "sanity check: a sharp rect's bounding-box corner must be shape-colored"
        );
        assert_eq!(
            rounded_corner, background,
            "expected a corner_radius: 10.0 rect's corner pixel to be background-colored under \
             analytic AA too, got {rounded_corner:?} vs. background {background:?}"
        );
    }

    /// GPU pixel-decode proof (default MSAA pipeline) that a linear gradient
    /// fill produces real per-pixel interpolation: renders a wide rect with
    /// a pure-red-to-pure-blue horizontal gradient and confirms pixels near
    /// the left edge read close to red, pixels near the right edge read
    /// close to blue, and a pixel at the horizontal midpoint is a genuine
    /// intermediate blend (neither near-red nor near-blue) -- not one flat
    /// color and not a hard cut partway across.
    #[test]
    fn linear_gradient_fill_interpolates_between_stops_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 16,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "gradient".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 64.0,
                    height: 16.0,
                    corner_radius: 0.0,
                    fill: FillV1::Gradient(GradientV1::LinearGradient {
                        from: [1.0, 0.0, 0.0, 1.0],
                        to: [0.0, 0.0, 1.0, 1.0],
                        angle_degrees: 0.0,
                    }),
                },
            }],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let near_left = pixel_at(1, 8);
        let near_right = pixel_at(62, 8);
        let middle = pixel_at(32, 8);

        assert!(
            near_left[0] > 180 && near_left[2] < 80,
            "expected the pixel near the gradient's `from` edge to read close to pure red, \
             got {near_left:?}"
        );
        assert!(
            near_right[2] > 180 && near_right[0] < 80,
            "expected the pixel near the gradient's `to` edge to read close to pure blue, \
             got {near_right:?}"
        );
        assert!(
            middle[0] > 40 && middle[0] < 215 && middle[2] > 40 && middle[2] < 215,
            "expected the midpoint pixel to be a genuine intermediate red/blue blend (neither \
             near-red nor near-blue), got {middle:?}"
        );
    }

    /// Analytic-AA counterpart to
    /// `linear_gradient_fill_interpolates_between_stops_on_an_available_gpu`:
    /// same scene and same proof, but through `render_rgba_analytic_aa`,
    /// confirming `add_rect_analytic` also resolves each vertex's color via
    /// `fill_vertex_color` rather than silently collapsing a gradient `fill`
    /// to one flat color.
    #[test]
    fn linear_gradient_fill_interpolates_under_analytic_aa_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 16,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "gradient".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 64.0,
                    height: 16.0,
                    corner_radius: 0.0,
                    fill: FillV1::Gradient(GradientV1::LinearGradient {
                        from: [1.0, 0.0, 0.0, 1.0],
                        to: [0.0, 0.0, 1.0, 1.0],
                        angle_degrees: 0.0,
                    }),
                },
            }],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba_analytic_aa(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let near_left = pixel_at(1, 8);
        let near_right = pixel_at(62, 8);
        let middle = pixel_at(32, 8);

        assert!(
            near_left[0] > 180 && near_left[2] < 80,
            "expected the pixel near the gradient's `from` edge to read close to pure red under \
             analytic AA, got {near_left:?}"
        );
        assert!(
            near_right[2] > 180 && near_right[0] < 80,
            "expected the pixel near the gradient's `to` edge to read close to pure blue under \
             analytic AA, got {near_right:?}"
        );
        assert!(
            middle[0] > 40 && middle[0] < 215 && middle[2] > 40 && middle[2] < 215,
            "expected the midpoint pixel to be a genuine intermediate red/blue blend under \
             analytic AA, got {middle:?}"
        );
    }

    /// GPU pixel-decode proof that a radial gradient on an `Ellipse` also
    /// interpolates for real: `center` at the ellipse's middle, `edge` at
    /// its boundary, and a genuine intermediate blend partway out.
    #[test]
    fn radial_gradient_fill_interpolates_between_stops_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 40,
                height: 40,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "gradient".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Ellipse {
                    cx: 20.0,
                    cy: 20.0,
                    rx: 18.0,
                    ry: 18.0,
                    fill: FillV1::Gradient(GradientV1::RadialGradient {
                        center: [1.0, 0.0, 0.0, 1.0],
                        edge: [0.0, 0.0, 1.0, 1.0],
                    }),
                },
            }],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let at_center = pixel_at(20, 20);
        let near_edge = pixel_at(20, 3);
        let partway = pixel_at(20, 11);

        assert!(
            at_center[0] > 180 && at_center[2] < 80,
            "expected the ellipse's center pixel to read close to the radial gradient's \
             `center` color (pure red), got {at_center:?}"
        );
        assert!(
            near_edge[2] > 180 && near_edge[0] < 80,
            "expected a pixel near the ellipse's boundary to read close to the radial \
             gradient's `edge` color (pure blue), got {near_edge:?}"
        );
        assert!(
            partway[0] > 40 && partway[0] < 215 && partway[2] > 40 && partway[2] < 215,
            "expected a pixel partway between the ellipse's center and edge to be a genuine \
             intermediate blend, got {partway:?}"
        );
    }
}
