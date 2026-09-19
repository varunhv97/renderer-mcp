use crate::GpuRenderer;
use crate::animation::*;
use crate::test_support::*;
use renderer_schema::CanvasV1;
use renderer_schema::FillV1;
use renderer_schema::GradientV1;
use renderer_schema::KeyframeV1;
use renderer_schema::NodeKindV1;
use renderer_schema::NodeV1;
use renderer_schema::SCENE_VERSION_V1;
use renderer_schema::SceneV1;

// ---- Analytic (SDF + fwidth) anti-aliasing "shadow mode" tests ----
//
// New, separately-named tests only: none of these touch, modify, or
// regenerate any existing golden image or existing test above, and none
// of the existing tests above were changed to make room for these.

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
    let (analytic_pixels, analytic_warnings) = renderer.render_rgba_analytic_aa(&scene).unwrap();
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
            let coverage =
                capsule_coverage_numeric(x as i32, y as i32, line_a, line_b, half_thickness, 64);
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
    let (rounded_pixels, rounded_warnings) = renderer.render_rgba_analytic_aa(&rounded).unwrap();
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
