use crate::GpuRenderer;
use crate::gpu::*;
use renderer_schema::CanvasV1;
use renderer_schema::FillV1;
use renderer_schema::NodeKindV1;
use renderer_schema::NodeV1;
use renderer_schema::SCENE_VERSION_V1;
use renderer_schema::SceneV1;

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
