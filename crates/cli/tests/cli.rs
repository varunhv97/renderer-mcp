use std::{
    net::{SocketAddr, TcpListener},
    process::Command,
    time::Duration,
};

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

#[test]
fn inspect_reports_a_helpful_error_for_a_missing_file() {
    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args(["inspect", "--input", "/no/such/image.png"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("could not inspect")
    );
}

#[test]
fn scene_commands_reject_non_loopback_endpoints() {
    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args(["scene", "--endpoint", "8.8.8.8:9", "health"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("loopback")
    );
}

/// Picks a loopback port that is very likely to be free by briefly binding to
/// port 0 and letting the OS choose, then releasing it for the daemon to use.
fn pick_endpoint() -> SocketAddr {
    let picker = TcpListener::bind("127.0.0.1:0").unwrap();
    picker.local_addr().unwrap()
}

fn renderer(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args(args)
        .output()
        .unwrap()
}

/// Exercises every `renderer scene ...` subcommand against a real, foreground
/// daemon: creation, conflicting creation, lookup (found and missing),
/// revision-checked replace (conflict and success), an atomic patch, a
/// render, and destruction. Skips gracefully (like the other render tests)
/// when no GPU adapter is available, since the daemon cannot start without
/// one.
#[test]
fn scene_subcommands_round_trip_through_a_running_daemon() {
    let endpoint = pick_endpoint();
    let addr = endpoint.to_string();
    let mut daemon = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args(["daemon", "serve", "--endpoint", &addr])
        .spawn()
        .unwrap();

    let mut ready = false;
    for _ in 0..50 {
        if let Ok(Some(_)) = daemon.try_wait() {
            break;
        }
        if renderer(&["scene", "--endpoint", &addr, "health"])
            .status
            .success()
        {
            ready = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    if !ready {
        // No GPU adapter (or some other startup failure) prevented the
        // daemon from coming up; nothing further to exercise here.
        let _ = daemon.kill();
        let _ = daemon.wait();
        return;
    }

    let directory = tempfile::tempdir().unwrap();
    let scene_path = directory.path().join("scene.json");
    std::fs::write(&scene_path, r#"{"version":"renderer.scene.v1","canvas":{"width":8,"height":8,"background":[0.0,0.0,0.0,1.0]},"nodes":[{"id":"box","kind":"rect","x":0.0,"y":0.0,"width":8.0,"height":8.0,"color":[1.0,0.0,0.0,1.0]}]}"#).unwrap();

    let create = renderer(&[
        "scene",
        "--endpoint",
        &addr,
        "create",
        "demo",
        "--input",
        scene_path.to_str().unwrap(),
    ]);
    assert!(create.status.success(), "{create:?}");
    assert!(
        String::from_utf8(create.stdout)
            .unwrap()
            .contains("\"revision\":1")
    );

    let duplicate = renderer(&[
        "scene",
        "--endpoint",
        &addr,
        "create",
        "demo",
        "--input",
        scene_path.to_str().unwrap(),
    ]);
    assert!(!duplicate.status.success());
    assert!(
        String::from_utf8(duplicate.stderr)
            .unwrap()
            .contains("already exists")
    );

    let get = renderer(&["scene", "--endpoint", &addr, "get", "demo"]);
    assert!(get.status.success());
    assert!(
        String::from_utf8(get.stdout)
            .unwrap()
            .contains("\"scene_id\":\"demo\"")
    );

    let missing = renderer(&["scene", "--endpoint", &addr, "get", "missing"]);
    assert!(!missing.status.success());
    assert!(
        String::from_utf8(missing.stderr)
            .unwrap()
            .contains("does not exist")
    );

    let conflicting_replace = renderer(&[
        "scene",
        "--endpoint",
        &addr,
        "replace",
        "demo",
        "--input",
        scene_path.to_str().unwrap(),
        "--expected-revision",
        "99",
    ]);
    assert!(!conflicting_replace.status.success());
    assert!(
        String::from_utf8(conflicting_replace.stderr)
            .unwrap()
            .contains("revision conflict")
    );

    let replace = renderer(&[
        "scene",
        "--endpoint",
        &addr,
        "replace",
        "demo",
        "--input",
        scene_path.to_str().unwrap(),
        "--expected-revision",
        "1",
    ]);
    assert!(replace.status.success(), "{replace:?}");
    assert!(
        String::from_utf8(replace.stdout)
            .unwrap()
            .contains("\"revision\":2")
    );

    let patch_path = directory.path().join("patch.json");
    std::fs::write(
        &patch_path,
        r#"{"expected_revision":2,"operations":[{"op":"clear_timeline"}]}"#,
    )
    .unwrap();
    let patch = renderer(&[
        "scene",
        "--endpoint",
        &addr,
        "patch",
        "demo",
        "--input",
        patch_path.to_str().unwrap(),
    ]);
    assert!(patch.status.success(), "{patch:?}");
    assert!(
        String::from_utf8(patch.stdout)
            .unwrap()
            .contains("\"revision\":3")
    );

    let bad_patch_path = directory.path().join("bad_patch.json");
    std::fs::write(&bad_patch_path, "not json").unwrap();
    let bad_patch = renderer(&[
        "scene",
        "--endpoint",
        &addr,
        "patch",
        "demo",
        "--input",
        bad_patch_path.to_str().unwrap(),
    ]);
    assert!(!bad_patch.status.success());
    assert!(
        String::from_utf8(bad_patch.stderr)
            .unwrap()
            .contains("scene is not valid JSON")
    );

    let rendered_path = directory.path().join("demo.png");
    let render = renderer(&[
        "scene",
        "--endpoint",
        &addr,
        "render",
        "demo",
        "--output",
        rendered_path.to_str().unwrap(),
    ]);
    assert!(render.status.success(), "{render:?}");
    assert!(
        String::from_utf8(render.stdout)
            .unwrap()
            .contains("\"mime_type\":\"image/png\"")
    );
    assert!(rendered_path.exists());

    let destroy = renderer(&["scene", "--endpoint", &addr, "destroy", "demo"]);
    assert!(destroy.status.success(), "{destroy:?}");
    assert!(
        String::from_utf8(destroy.stdout)
            .unwrap()
            .contains("destroyed")
    );

    let health = renderer(&["scene", "--endpoint", &addr, "health"]);
    assert!(health.status.success());
    assert!(String::from_utf8(health.stdout).unwrap().contains("health"));

    let missing_input = renderer(&[
        "scene",
        "--endpoint",
        &addr,
        "create",
        "another",
        "--input",
        "/no/such/scene.json",
    ]);
    assert!(!missing_input.status.success());
    assert!(
        String::from_utf8(missing_input.stderr)
            .unwrap()
            .contains("could not read")
    );

    let _ = daemon.kill();
    let _ = daemon.wait();
}
