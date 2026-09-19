use crate::GpuRenderer;
use crate::test_support::*;

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
