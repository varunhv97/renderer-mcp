//! One-shot generator for the checked-in golden PNGs under `assets/golden/`.
//! Not part of the test suite; run manually with
//! `cargo run -p renderer-core --example generate_golden` after intentionally
//! changing rendering behavior and re-approving the new goldens.
use renderer_core::GpuRenderer;
use renderer_schema::SceneV1;
use std::path::Path;

fn main() {
    let renderer = GpuRenderer::new().expect("GPU adapter required to generate golden images");
    let asset_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/golden");
    for scene_file in [
        "golden_text_image_placement.scene.json",
        "golden_alpha_blend.scene.json",
        "golden_order_image_first.scene.json",
        "golden_order_vector_first.scene.json",
        "golden_effect_invert.scene.json",
    ] {
        let raw = std::fs::read_to_string(asset_root.join(scene_file)).unwrap();
        let scene: SceneV1 = serde_json::from_str(&raw).unwrap();
        let output_name = scene_file.replace(".scene.json", ".expected.png");
        let output = asset_root.join(&output_name);
        renderer
            .render_png_with_asset_root(&scene, &output, &asset_root)
            .unwrap();
        println!("wrote {}", output.display());
    }
}
