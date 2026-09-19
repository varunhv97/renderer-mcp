use crate::GpuRenderer;
use crate::test_support::*;
use renderer_schema::FillV1;
use renderer_schema::KeyframeV1;
use renderer_schema::NodeKindV1;
use renderer_schema::NodeV1;
use std::fs;

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
