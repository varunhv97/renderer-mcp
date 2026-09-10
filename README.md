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
