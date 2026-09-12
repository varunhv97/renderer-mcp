# RendererCli

A local GPU-accelerated visual runtime for coding agents.

The workspace is intentionally CLI and MCP first: agents submit a versioned JSON
scene or drawing-command document, receive a saved image plus structured
metadata, and can keep a named scene alive during a local daemon session.
Scenes compose vector shapes, local raster images, and embedded-font text into
one ordered GPU pass; PNG and keyframed GIF export both work from the same
scene document.

## Development

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo llvm-cov --workspace --all-targets --all-features --fail-under-lines 65
```

`cargo llvm-cov` needs `llvm-tools`. If your toolchain came from `rustup`,
`rustup component add llvm-tools-preview` is enough. On a Homebrew-installed
stable toolchain (no `rustup`), point it at a matching LLVM instead:

```sh
brew install llvm
LLVM_COV="$(brew --prefix llvm)/bin/llvm-cov" \
LLVM_PROFDATA="$(brew --prefix llvm)/bin/llvm-profdata" \
cargo llvm-cov --workspace --all-targets --all-features --fail-under-lines 65
```

Building produces two binaries: `renderer` (CLI, package `renderer-cli`) and
`renderer-mcp` (MCP server, package `renderer-mcp`).

```sh
cargo build --release
```

## One-shot rendering

```sh
cargo run -p renderer-cli -- render --input examples/basic.scene.json --output out.png
cargo run -p renderer-cli -- render --input examples/pulse.scene.json --output out.gif
cargo run -p renderer-cli -- inspect --input out.png
```

`render` accepts a `SceneV1` JSON document and writes `.png` or `.gif` based on
the output extension; GIF export uses the scene's keyframe timeline. `inspect`
reports dimensions, MIME type, and the SHA-256 of the exact output bytes.
Diagnostics (missing assets, invalid scenes, resource-limit violations) are
returned as structured JSON, not silent failures.

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
cargo run -p renderer-cli -- scene --endpoint 127.0.0.1:9472 destroy demo
```

The daemon only ever binds a loopback address, caps requests at 1 MiB, and
rejects unknown fields/operations. Everything under `.renderer/` is generated
local output and is not committed.

## MCP server

`renderer-mcp` speaks newline-delimited JSON-RPC over stdio (`initialize`,
`tools/list`, `tools/call`) and never listens on a network socket. It exposes
one-shot `render_scene` plus named-scene tools (`create_scene`, `get_scene`,
`replace_scene`, `patch_scene`, `render_named_scene`, `export_named_gif`,
`inspect_image`, `destroy_scene`) backed by the same daemon protocol used by
the CLI. Named-scene tools require a running `daemon serve` and the endpoint
supplied via `RENDERER_DAEMON_ENDPOINT`; `render_scene` needs neither.

To register it with an MCP-capable agent host, point the host at the built
binary and set the endpoint as an environment variable, for example:

```json
{
  "mcpServers": {
    "renderer": {
      "command": "/absolute/path/to/target/release/renderer-mcp",
      "env": { "RENDERER_DAEMON_ENDPOINT": "127.0.0.1:9472" }
    }
  }
}
```

Start `renderer daemon serve --endpoint 127.0.0.1:9472` before using any
named-scene tool; `render_scene` works even without a daemon running. Named
PNG/GIF exports return the daemon's metadata followed by inline base64 image
content so compatible hosts can display the result immediately.

## Testing

Unit tests are colocated with each crate; cross-crate CLI and MCP protocol
tests live under `crates/cli/tests` and `crates/mcp/tests`. The renderer crate
also includes deterministic golden-image tests (fixtures under
`crates/renderer/assets/golden/`) that render representative scenes and diff
the output against checked-in reference PNGs within a documented pixel
tolerance; like the other GPU-backed tests, they skip rather than fail on a
host with no GPU adapter. CI enforces formatting, Clippy, the full test suite,
and a 65% workspace line-coverage floor via `cargo llvm-cov` on macOS, Linux,
and Windows.
