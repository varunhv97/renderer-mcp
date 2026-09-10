use std::process::Command;

#[test]
fn reports_status_and_inspects_a_saved_image() {
    let binary = env!("CARGO_BIN_EXE_renderer");
    let status = Command::new(binary).arg("status").output().unwrap();
    assert!(status.status.success());
    assert!(
        String::from_utf8(status.stdout)
            .unwrap()
            .contains("renderer.scene.v1")
    );

    let directory = tempfile::tempdir().unwrap();
    let image_path = directory.path().join("image.png");
    image::RgbaImage::new(2, 3).save(&image_path).unwrap();
    let inspect = Command::new(binary)
        .args(["inspect", "--input", image_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(inspect.status.success());
    let output = String::from_utf8(inspect.stdout).unwrap();
    assert!(output.contains("\"width\": 2"));
    assert!(output.contains("\"height\": 3"));
}

#[test]
fn rejects_invalid_render_input_without_needing_a_gpu() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("invalid.json");
    std::fs::write(&input, "not json").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args(["render", "--input", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("scene is not valid JSON")
    );
}

#[test]
fn renders_png_and_gif_when_a_gpu_is_available() {
    let directory = tempfile::tempdir().unwrap();
    let png_scene = directory.path().join("scene.json");
    std::fs::write(&png_scene, r#"{"version":"renderer.scene.v1","canvas":{"width":8,"height":8,"background":[0.0,0.0,0.0,1.0]},"nodes":[{"id":"box","kind":"rect","x":0.0,"y":0.0,"width":8.0,"height":8.0,"color":[1.0,0.0,0.0,1.0]}]}"#).unwrap();
    let png = directory.path().join("scene.png");
    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args([
            "render",
            "--input",
            png_scene.to_str().unwrap(),
            "--output",
            png.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    if !output.status.success() {
        assert!(
            String::from_utf8(output.stderr)
                .unwrap()
                .contains("no compatible GPU adapter")
        );
        return;
    }
    assert!(png.exists());

    let gif_scene = directory.path().join("animation.json");
    std::fs::write(&gif_scene, r#"{"version":"renderer.scene.v1","canvas":{"width":8,"height":8,"background":[0.0,0.0,0.0,1.0]},"nodes":[{"id":"box","kind":"rect","x":0.0,"y":0.0,"width":8.0,"height":8.0,"color":[1.0,0.0,0.0,1.0]}],"timeline":{"fps":1,"duration_ms":1000,"keyframes":[]}}"#).unwrap();
    let gif = directory.path().join("animation.gif");
    assert!(
        Command::new(env!("CARGO_BIN_EXE_renderer"))
            .args([
                "render",
                "--input",
                gif_scene.to_str().unwrap(),
                "--output",
                gif.to_str().unwrap()
            ])
            .status()
            .unwrap()
            .success()
    );
    assert!(gif.exists());
}
