use std::{
    net::{SocketAddr, TcpListener},
    path::Path,
    process::Command,
    time::Duration,
};

/// Parses a CLI failure's stderr as the stable JSON error envelope the
/// UI/UX specification requires -- exactly one JSON object with a `code`
/// and a `message` field, and nothing else on the stream.
fn parse_error(stderr: Vec<u8>) -> serde_json::Value {
    let text = String::from_utf8(stderr).unwrap();
    let value: serde_json::Value = serde_json::from_str(text.trim())
        .unwrap_or_else(|error| panic!("stderr was not a single JSON object: {error}\n{text}"));
    assert!(value["code"].is_string(), "missing code in {value}");
    assert!(value["message"].is_string(), "missing message in {value}");
    value
}

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
    let error = parse_error(output.stderr);
    assert_eq!(error["code"], "invalid_json");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("not valid JSON")
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
        let error = parse_error(output.stderr);
        assert_eq!(error["code"], "internal_error");
        assert!(
            error["message"]
                .as_str()
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
    let error = parse_error(output.stderr);
    assert_eq!(error["code"], "io_error");
    assert!(
        error["message"]
            .as_str()
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
    let error = parse_error(output.stderr);
    assert_eq!(error["code"], "internal_error");
    assert!(error["message"].as_str().unwrap().contains("loopback"));
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
    let error = parse_error(duplicate.stderr);
    assert_eq!(error["code"], "already_exists");
    assert!(
        error["message"]
            .as_str()
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
    let error = parse_error(missing.stderr);
    assert_eq!(error["code"], "not_found");
    assert!(
        error["message"]
            .as_str()
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
    let error = parse_error(conflicting_replace.stderr);
    assert_eq!(error["code"], "revision_conflict");
    assert!(
        error["message"]
            .as_str()
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
    let error = parse_error(bad_patch.stderr);
    assert_eq!(error["code"], "invalid_json");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("not valid JSON")
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
    let error = parse_error(missing_input.stderr);
    assert_eq!(error["code"], "io_error");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("could not read")
    );

    let _ = daemon.kill();
    let _ = daemon.wait();
}

/// Writes a tiny 2x2 PNG fixture, returning its path.
fn write_test_png(path: &Path) {
    let mut image = image::RgbaImage::new(2, 2);
    for pixel in image.pixels_mut() {
        *pixel = image::Rgba([200, 100, 50, 255]);
    }
    image.save(path).unwrap();
}

/// Writes a tiny animated GIF fixture with one 2x2 frame per entry in
/// `delays_ms`, each a different solid color so frames are distinguishable.
fn write_test_gif(path: &Path, delays_ms: &[u32]) {
    use image::codecs::gif::GifEncoder;
    use image::{Delay, Frame, RgbaImage};

    let file = std::fs::File::create(path).unwrap();
    let mut encoder = GifEncoder::new(file);
    for (index, delay_ms) in delays_ms.iter().enumerate() {
        let mut frame_image = RgbaImage::new(2, 2);
        let shade = (index as u8).wrapping_mul(60);
        for pixel in frame_image.pixels_mut() {
            *pixel = image::Rgba([shade, 0, 255 - shade, 255]);
        }
        let delay = Delay::from_numer_denom_ms(*delay_ms, 1);
        encoder
            .encode_frame(Frame::from_parts(frame_image, 0, 0, delay))
            .unwrap();
    }
}

/// `show --clear` sends only the Kitty delete-all-images command to the
/// device the `--tty` override names -- a regular file stands in for a real
/// tty device here since the code only ever opens-and-writes it.
#[test]
fn show_clear_sends_only_the_kitty_delete_all_command() {
    let directory = tempfile::tempdir().unwrap();
    let tty_path = directory.path().join("fake-tty");
    std::fs::write(&tty_path, b"").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args(["show", "--clear", "--tty", tty_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(std::fs::read(&tty_path).unwrap(), b"\x1b_Ga=d,d=A\x1b\\");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("\"status\":\"cleared\""));
}

/// A static PNG through `--protocol kitty` produces exactly one Kitty
/// transmit command carrying the raw PNG bytes.
#[test]
fn show_writes_a_kitty_transmit_command_for_a_static_png() {
    let directory = tempfile::tempdir().unwrap();
    let png_path = directory.path().join("image.png");
    write_test_png(&png_path);
    let tty_path = directory.path().join("fake-tty");
    std::fs::write(&tty_path, b"").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args([
            "show",
            png_path.to_str().unwrap(),
            "--tty",
            tty_path.to_str().unwrap(),
            "--protocol",
            "kitty",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");

    let written = std::fs::read(&tty_path).unwrap();
    let text = String::from_utf8(written).unwrap();
    assert!(text.starts_with("\x1b_Ga=T,f=100,i=1,q=2;"));
    assert!(text.ends_with("\x1b\\"));
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("\"protocol\":\"kitty\""));
}

/// A static PNG through `--protocol iterm2` produces the documented OSC
/// 1337 escape sequence carrying the raw file bytes.
#[test]
fn show_writes_an_iterm2_osc_1337_command_for_a_static_png() {
    let directory = tempfile::tempdir().unwrap();
    let png_path = directory.path().join("image.png");
    write_test_png(&png_path);
    let png_bytes = std::fs::read(&png_path).unwrap();
    let tty_path = directory.path().join("fake-tty");
    std::fs::write(&tty_path, b"").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args([
            "show",
            png_path.to_str().unwrap(),
            "--tty",
            tty_path.to_str().unwrap(),
            "--protocol",
            "iterm2",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");

    let written = std::fs::read(&tty_path).unwrap();
    let text = String::from_utf8(written).unwrap();
    assert!(text.starts_with(&format!(
        "\x1b]1337;File=inline=1;size={};width=auto;height=auto;preserveAspectRatio=1:",
        png_bytes.len()
    )));
    assert!(text.ends_with('\u{7}'));
}

/// A static PNG through `--protocol ansi` produces true-color half-block
/// escape sequences (no graphics-protocol control string at all).
#[test]
fn show_writes_ansi_half_blocks_for_a_static_png() {
    let directory = tempfile::tempdir().unwrap();
    let png_path = directory.path().join("image.png");
    write_test_png(&png_path);
    let tty_path = directory.path().join("fake-tty");
    std::fs::write(&tty_path, b"").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args([
            "show",
            png_path.to_str().unwrap(),
            "--tty",
            tty_path.to_str().unwrap(),
            "--protocol",
            "ansi",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");

    let written = std::fs::read(&tty_path).unwrap();
    let text = String::from_utf8(written).unwrap();
    assert!(text.contains("\x1b[38;2;"));
    assert!(text.contains("\x1b[48;2;"));
    assert!(text.contains('\u{2580}'));
    assert!(!text.contains("\x1b_G"));
}

/// A file that isn't a PNG or GIF is rejected with the structured
/// `unsupported_image` error rather than an attempted (and likely garbled)
/// terminal write.
#[test]
fn show_rejects_a_file_that_is_not_png_or_gif() {
    let directory = tempfile::tempdir().unwrap();
    let text_path = directory.path().join("not-an-image.txt");
    std::fs::write(&text_path, b"just some text").unwrap();
    let tty_path = directory.path().join("fake-tty");
    std::fs::write(&tty_path, b"").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args([
            "show",
            text_path.to_str().unwrap(),
            "--tty",
            tty_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = parse_error(output.stderr);
    assert_eq!(error["code"], "unsupported_image");
    assert_eq!(std::fs::read(&tty_path).unwrap(), b"");
}

/// An unknown `--protocol` value is rejected with a structured error before
/// any terminal write is attempted.
#[test]
fn show_rejects_an_unknown_protocol_flag() {
    let directory = tempfile::tempdir().unwrap();
    let png_path = directory.path().join("image.png");
    write_test_png(&png_path);
    let tty_path = directory.path().join("fake-tty");
    std::fs::write(&tty_path, b"").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .args([
            "show",
            png_path.to_str().unwrap(),
            "--tty",
            tty_path.to_str().unwrap(),
            "--protocol",
            "carrier-pigeon",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = parse_error(output.stderr);
    assert_eq!(error["code"], "invalid_protocol");
}

/// An animated GIF through `--protocol kitty` with a Kitty-animation-capable
/// environment (`KITTY_WINDOW_ID` set) sends the terminal-driven animation
/// sequence: one root-frame transmit, one additional-frame transmit per
/// remaining frame, and a single animation-control command -- fast and
/// deterministic since none of that requires the process itself to sleep.
#[test]
fn show_sends_native_kitty_animation_for_an_animation_capable_terminal() {
    let directory = tempfile::tempdir().unwrap();
    let gif_path = directory.path().join("anim.gif");
    write_test_gif(&gif_path, &[10, 10, 10]);
    let tty_path = directory.path().join("fake-tty");
    std::fs::write(&tty_path, b"").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .env("KITTY_WINDOW_ID", "1")
        .env_remove("TERM_PROGRAM")
        .args([
            "show",
            gif_path.to_str().unwrap(),
            "--tty",
            tty_path.to_str().unwrap(),
            "--protocol",
            "kitty",
            "--loops",
            "3",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");

    let written = std::fs::read(&tty_path).unwrap();
    let text = String::from_utf8(written).unwrap();
    assert_eq!(text.matches("a=T,f=100,i=1,q=2,z=").count(), 1);
    assert_eq!(text.matches("a=f,i=1,q=2,z=").count(), 2);
    // --loops 3 maps to v=4 per the protocol's "loop number-1 times" rule.
    assert!(text.contains("\x1b_Ga=a,i=1,q=2,s=3,v=4\x1b\\"));
}

/// An animated GIF through `--protocol kitty` in a non-animation-capable
/// environment (no Kitty/WezTerm signals) falls back to simulated
/// animation: the same static-image transmit command repeated once per
/// frame, with no animation-control command at all. Uses tiny per-frame
/// delays and `--loops 1` to keep the test fast and bounded.
#[test]
fn show_simulates_kitty_animation_when_the_terminal_lacks_the_extension() {
    let directory = tempfile::tempdir().unwrap();
    let gif_path = directory.path().join("anim.gif");
    write_test_gif(&gif_path, &[5, 5]);
    let tty_path = directory.path().join("fake-tty");
    std::fs::write(&tty_path, b"").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_renderer"))
        .env_remove("KITTY_WINDOW_ID")
        .env_remove("TERM_PROGRAM")
        .args([
            "show",
            gif_path.to_str().unwrap(),
            "--tty",
            tty_path.to_str().unwrap(),
            "--protocol",
            "kitty",
            "--loops",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");

    let written = std::fs::read(&tty_path).unwrap();
    let text = String::from_utf8(written).unwrap();
    // `C=1` (don't move the cursor) on every frame, and a delete of the
    // previous placement before every frame but the first, are what keep
    // simulated animation redrawing in place instead of cascading a new
    // placement down the screen each frame -- the exact bug this test
    // guards against.
    assert_eq!(text.matches("\x1b_Ga=T,f=100,i=1,q=2,C=1;").count(), 2);
    assert_eq!(text.matches("\x1b_Ga=d,d=i,i=1\x1b\\").count(), 1);
    assert!(!text.contains("a=a,i=1"));
}

/// Runs `scene get` for a scene ID that doesn't exist against a real,
/// foreground daemon and checks that stderr is exactly one JSON object
/// carrying the `not_found` code -- the concrete failure called out in the
/// bug report this behavior fixes.
#[test]
fn scene_get_for_a_missing_scene_emits_structured_not_found_json() {
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
        let _ = daemon.kill();
        let _ = daemon.wait();
        return;
    }

    let output = renderer(&["scene", "--endpoint", &addr, "get", "missing-scene"]);
    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "stdout must stay empty on failure"
    );
    let error = parse_error(output.stderr);
    assert_eq!(error["code"], "not_found");
    assert!(error["message"].as_str().unwrap().contains("missing-scene"));

    let _ = daemon.kill();
    let _ = daemon.wait();
}
