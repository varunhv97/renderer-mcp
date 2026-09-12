//! One-shot generator for the checked-in golden PNG backing
//! `golden_svg_placement.scene.json` under `assets/golden/`. Kept separate
//! from `generate_golden.rs` (rather than adding to its fixture list) to
//! avoid conflicting with other in-flight changes to that shared list. Not
//! part of the test suite; run manually with
//! `cargo run -p renderer-core --example generate_golden_svg` after
//! intentionally changing SVG rasterization behavior and re-approving the
//! new golden.
use renderer_core::GpuRenderer;
use renderer_schema::SceneV1;
use std::path::Path;

fn main() {
    let renderer = GpuRenderer::new().expect("GPU adapter required to generate golden images");
    let asset_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/golden");
    let scene_file = "golden_svg_placement.scene.json";
    let raw = std::fs::read_to_string(asset_root.join(scene_file)).unwrap();
    let scene: SceneV1 = serde_json::from_str(&raw).unwrap();
    let output_name = scene_file.replace(".scene.json", ".expected.png");
    let output = asset_root.join(&output_name);
    renderer
        .render_png_with_asset_root(&scene, &output, &asset_root)
        .unwrap();
    println!("wrote {}", output.display());
}
