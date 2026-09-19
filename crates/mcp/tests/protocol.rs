use std::{
    io::Write,
    process::{Command, Stdio},
};

/// Parses one line of `renderer-mcp` stdout as a JSON-RPC response and
/// returns its `result.content[0].text` field, itself re-parsed as JSON --
/// the shape every `tools/call` success (including `show_image`'s) uses.
fn parse_tool_text_result(stdout: Vec<u8>) -> serde_json::Value {
    let envelope: serde_json::Value =
        serde_json::from_str(String::from_utf8(stdout).unwrap().trim()).unwrap();
    let text = envelope["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("no text content in response: {envelope}"));
    serde_json::from_str(text).unwrap()
}

#[test]
fn ignores_notifications_and_replies_to_requests() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_renderer-mcp"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(
        input,
        "{{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}}"
    )
    .unwrap();
    writeln!(input, "not json").unwrap();
    writeln!(
        input,
        "{{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"unknown\"}}"
    )
    .unwrap();
    writeln!(
        input,
        "{{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}}"
    )
    .unwrap();
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let lines: Vec<_> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains("method not found"));
    assert!(lines[1].contains("render_scene"));
}

/// `tools/list`'s `show_image` entry: parameter shape and the enum of
/// accepted `protocol` values, following this file's existing style for
/// asserting on protocol responses via `describes_the_mcp_server`-flavored
/// checks (see `ignores_notifications_and_replies_to_requests` above).
#[test]
fn show_image_tool_schema_describes_its_parameters() {
    let request = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/list"
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_renderer-mcp"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.as_mut().unwrap(), "{request}").unwrap();
    let output = child.wait_with_output().unwrap();
    let response: serde_json::Value =
        serde_json::from_str(String::from_utf8(output.stdout).unwrap().trim()).unwrap();

    let tools = response["result"]["tools"].as_array().unwrap();
    let show_image = tools
        .iter()
        .find(|tool| tool["name"] == "show_image")
        .expect("show_image must be listed alongside the other tools");

    let schema = &show_image["inputSchema"];
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["required"].as_array().unwrap().len(), 0);

    let properties = &schema["properties"];
    assert_eq!(properties["path"]["type"], "string");
    assert_eq!(properties["tty"]["type"], "string");
    assert_eq!(properties["clear"]["type"], "boolean");
    assert_eq!(properties["clear"]["default"], false);
    assert_eq!(properties["loops"]["type"], "integer");

    assert_eq!(properties["protocol"]["type"], "string");
    assert_eq!(properties["protocol"]["default"], "auto");
    let protocol_values: Vec<&str> = properties["protocol"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    assert_eq!(protocol_values, ["auto", "kitty", "iterm2"]);
}

/// `path` is required unless `clear` is true -- checked by the shared
/// `renderer_terminal` crate itself (`TerminalError::MissingPath`), but
/// this confirms the MCP tool surfaces it as a `tools/call` error rather
/// than, say, panicking or silently no-op'ing.
#[test]
fn show_image_rejects_a_missing_path_when_not_clearing() {
    let request = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "show_image", "arguments": {}
        }
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_renderer-mcp"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.as_mut().unwrap(), "{request}").unwrap();
    let output = child.wait_with_output().unwrap();
    let response: serde_json::Value =
        serde_json::from_str(String::from_utf8(output.stdout).unwrap().trim()).unwrap();
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("path is required"),
        "{response}"
    );
}

/// A static PNG through `show_image` with an explicit `tty` override and
/// `protocol: "kitty"` produces the same Kitty transmit command the CLI's
/// `renderer show` writes -- confirming this tool actually calls into the
/// shared `renderer_terminal::run` rather than reimplementing anything.
/// `CMUX_SOCKET_PATH` is removed so this deterministically exercises the
/// terminal-protocol path even in an environment (like this one) where a
/// live cmux socket is also available -- see
/// `show_image_displays_a_real_file_through_a_live_cmux_socket_when_available`
/// below for cmux's own path.
#[test]
fn show_image_writes_a_kitty_transmit_command_for_a_static_png() {
    let directory = tempfile::tempdir().unwrap();
    let tty_path = directory.path().join("fake-tty");
    std::fs::write(&tty_path, b"").unwrap();
    let image_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../renderer/assets/golden/golden_fixture.png"
    );

    let request = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "show_image", "arguments": {
                "path": image_path,
                "tty": tty_path,
                "protocol": "kitty",
            }
        }
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_renderer-mcp"))
        .env_remove("CMUX_SOCKET_PATH")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.as_mut().unwrap(), "{request}").unwrap();
    let output = child.wait_with_output().unwrap();
    let summary = parse_tool_text_result(output.stdout);
    assert_eq!(summary["status"], "displayed");
    assert_eq!(summary["protocol"], "kitty");

    let written = std::fs::read(&tty_path).unwrap();
    let text = String::from_utf8(written).unwrap();
    assert!(text.starts_with("\x1b_Ga=T,f=100,i=1,q=2;"));
    assert!(text.ends_with("\x1b\\"));
}

/// `clear: true` against an explicit `tty` override sends only the Kitty
/// delete-all-images command, matching `renderer show --clear`.
#[test]
fn show_image_clear_sends_only_the_kitty_delete_all_command() {
    let directory = tempfile::tempdir().unwrap();
    let tty_path = directory.path().join("fake-tty");
    std::fs::write(&tty_path, b"").unwrap();

    let request = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "show_image", "arguments": { "clear": true, "tty": tty_path }
        }
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_renderer-mcp"))
        .env_remove("CMUX_SOCKET_PATH")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.as_mut().unwrap(), "{request}").unwrap();
    let output = child.wait_with_output().unwrap();
    let summary = parse_tool_text_result(output.stdout);
    assert_eq!(summary["status"], "cleared");
    assert_eq!(std::fs::read(&tty_path).unwrap(), b"\x1b_Ga=d,d=A\x1b\\");
}

/// End-to-end coverage of `show_image` against cmux's *real* control
/// socket, when this test environment has one (`$CMUX_SOCKET_PATH`,
/// connectable) -- exercising the actual `renderer-mcp` binary over real
/// stdio JSON-RPC, with no mocked server standing in for cmux. When no live
/// socket is connectable (e.g. in CI, which has no cmux), this is a no-op:
/// `crates/terminal/src/lib.rs`'s `cmux_preview::tests` module covers the
/// same request/response logic against a mocked Unix socket instead.
#[cfg(unix)]
#[test]
fn show_image_displays_a_real_file_through_a_live_cmux_socket_when_available() {
    let Ok(socket_path) = std::env::var("CMUX_SOCKET_PATH") else {
        eprintln!(
            "CMUX_SOCKET_PATH is not set in this environment; skipping the live cmux \
             end-to-end test for show_image (covered instead by \
             crates/terminal/src/lib.rs's mocked cmux_preview::tests)"
        );
        return;
    };
    if std::os::unix::net::UnixStream::connect(&socket_path).is_err() {
        eprintln!(
            "CMUX_SOCKET_PATH={socket_path} is set but not connectable in this environment; \
             skipping the live cmux end-to-end test for show_image (covered instead by \
             crates/terminal/src/lib.rs's mocked cmux_preview::tests)"
        );
        return;
    }

    let directory = tempfile::tempdir().unwrap();
    let image_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../renderer/assets/golden/golden_fixture.png"
    );

    let show_request = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "show_image", "arguments": { "path": image_path }
        }
    });
    let mut show_child = Command::new(env!("CARGO_BIN_EXE_renderer-mcp"))
        .current_dir(directory.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(show_child.stdin.as_mut().unwrap(), "{show_request}").unwrap();
    let show_output = show_child.wait_with_output().unwrap();
    let show_summary = parse_tool_text_result(show_output.stdout);
    assert_eq!(show_summary["status"], "displayed");
    assert_eq!(show_summary["protocol"], "cmux");

    // Clean up the preview surface we just opened in the user's real cmux
    // instance, the same way `renderer show --clear` would.
    let clear_request = serde_json::json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "show_image", "arguments": { "clear": true }
        }
    });
    let mut clear_child = Command::new(env!("CARGO_BIN_EXE_renderer-mcp"))
        .current_dir(directory.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(clear_child.stdin.as_mut().unwrap(), "{clear_request}").unwrap();
    let clear_output = clear_child.wait_with_output().unwrap();
    let clear_summary = parse_tool_text_result(clear_output.stdout);
    assert_eq!(clear_summary["status"], "cleared");
}

#[test]
fn renders_inline_image_through_the_mcp_executable() {
    let directory = tempfile::tempdir().unwrap();
    let output_path = directory.path().join("mcp.png");
    let request = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "render_scene", "arguments": {
                "output_path": output_path,
                "scene": { "version": "renderer.scene.v1", "canvas": { "width": 8, "height": 8, "background": [0.0, 0.0, 0.0, 1.0] }, "nodes": [{ "id": "box", "kind": "rect", "x": 0.0, "y": 0.0, "width": 8.0, "height": 8.0, "color": [1.0, 0.0, 0.0, 1.0] }] }
            }
        }
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_renderer-mcp"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.as_mut().unwrap(), "{request}").unwrap();
    let output = child.wait_with_output().unwrap();
    let response = String::from_utf8(output.stdout).unwrap();
    assert!(
        response.contains("\"type\":\"image\"") || response.contains("no compatible GPU adapter")
    );
}

/// Runs one `renderer-mcp` session that creates a named scene and reads it back
/// through the named-scene tools, with `RENDERER_DAEMON_ENDPOINT` set to
/// `endpoint` (or removed for `None`). Returns the two tool-call response
/// lines, or `None` when this machine has no GPU adapter for the built-in
/// daemon to use.
fn create_then_get_scene(endpoint: Option<&str>) -> Option<Vec<String>> {
    let scene = serde_json::json!({
        "version": "renderer.scene.v1",
        "canvas": { "width": 8, "height": 8, "background": [0.0, 0.0, 0.0, 1.0] },
        "nodes": []
    });
    let call = |id: u32, name: &str, arguments: serde_json::Value| {
        serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": { "name": name, "arguments": arguments }
        })
    };
    let mut command = Command::new(env!("CARGO_BIN_EXE_renderer-mcp"));
    match endpoint {
        Some(endpoint) => command.env("RENDERER_DAEMON_ENDPOINT", endpoint),
        None => command.env_remove("RENDERER_DAEMON_ENDPOINT"),
    };
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(
        input,
        "{}",
        call(
            1,
            "create_scene",
            serde_json::json!({ "scene_id": "embedded", "scene": scene })
        )
    )
    .unwrap();
    writeln!(
        input,
        "{}",
        call(
            2,
            "get_scene",
            serde_json::json!({ "scene_id": "embedded" })
        )
    )
    .unwrap();
    drop(input);
    let output = child.wait_with_output().unwrap();
    let lines: Vec<String> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(lines.len(), 2, "expected two responses, got {lines:?}");
    if lines[0].contains("could not start a built-in renderer daemon") {
        return None;
    }
    Some(lines)
}

#[test]
fn named_scene_tools_start_a_built_in_daemon_when_no_endpoint_is_set() {
    let Some(lines) = create_then_get_scene(None) else {
        return;
    };
    assert!(lines[0].contains("revision"), "create_scene: {}", lines[0]);
    assert!(
        lines[1].contains("embedded") && lines[1].contains("renderer.scene.v1"),
        "get_scene should see the scene create_scene stored: {}",
        lines[1]
    );
}

#[test]
fn named_scene_tools_start_a_built_in_daemon_on_a_configured_endpoint_nothing_serves() {
    let Ok(picker) = std::net::TcpListener::bind("127.0.0.1:0") else {
        return;
    };
    let endpoint = picker.local_addr().unwrap().to_string();
    drop(picker);
    let Some(lines) = create_then_get_scene(Some(&endpoint)) else {
        return;
    };
    assert!(lines[0].contains("revision"), "create_scene: {}", lines[0]);
    assert!(lines[1].contains("embedded"), "get_scene: {}", lines[1]);
}
