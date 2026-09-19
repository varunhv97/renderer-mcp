use crate::GpuRenderer;
use crate::animation::*;
use renderer_schema::CanvasV1;
use renderer_schema::FillV1;
use renderer_schema::KeyframeV1;
use renderer_schema::NodeKindV1;
use renderer_schema::NodeV1;
use renderer_schema::SCENE_VERSION_V1;
use renderer_schema::SceneV1;

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
