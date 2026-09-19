use crate::GpuRenderer;
use renderer_schema::CanvasV1;
use renderer_schema::FillV1;
use renderer_schema::GradientV1;
use renderer_schema::NodeKindV1;
use renderer_schema::NodeV1;
use renderer_schema::SCENE_VERSION_V1;
use renderer_schema::SceneV1;

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
