use crate::GpuRenderer;
use crate::error::*;
use crate::test_support::*;

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
