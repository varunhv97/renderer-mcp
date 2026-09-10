use std::{
    io::Write,
    process::{Command, Stdio},
};

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
