use super::*;
use crate::animation::*;
use crate::assets::*;
use crate::error::*;
use crate::limits::*;
use crate::test_support::*;
use fontdue::Font;
use renderer_schema::FillV1;
use renderer_schema::KeyframeV1;
use renderer_schema::NodeKindV1;
use renderer_schema::NodeV1;

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
        include_bytes!("../../assets/NotoSans-Regular.ttf") as &[u8],
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
        include_bytes!("../../assets/NotoSans-Regular.ttf") as &[u8],
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
