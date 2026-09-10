# RendererCli

A local GPU-accelerated visual runtime for coding agents.

The workspace is intentionally CLI and MCP first: agents submit a versioned JSON
scene or drawing-command document, receive a saved image plus structured
metadata, and can keep a named scene alive during a local daemon session.

## Development

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Use `cargo run -p renderer-cli -- render --input examples/basic.scene.json` to
make a PNG once the native graphics backend is available.

## Persistent named scenes

Start a foreground local daemon in one terminal:

```sh
cargo run -p renderer-cli -- daemon serve --endpoint 127.0.0.1:9472
```

In another terminal, create, patch, and render a scene through the same
loopback-only endpoint. Patch files contain typed `ScenePatchV1` operations and
may include an `expected_revision` to avoid overwriting a newer scene revision.

```sh
cargo run -p renderer-cli -- scene --endpoint 127.0.0.1:9472 create demo --input examples/basic.scene.json
cargo run -p renderer-cli -- scene --endpoint 127.0.0.1:9472 get demo
cargo run -p renderer-cli -- scene --endpoint 127.0.0.1:9472 render demo --output .renderer/output/demo.png
```

MCP named-scene tools use the same endpoint supplied in
`RENDERER_DAEMON_ENDPOINT`. The existing `render_scene` MCP tool remains
available for one-shot inline renders. Named-scene MCP tools include create,
get, replace, patch, inline PNG rendering, inline GIF export, local inspection,
and destroy. Named exports require an explicit `.png` or `.gif` output path.
