# RendererCli

A local GPU-accelerated visual runtime for coding agents.

The workspace is intentionally CLI and MCP first: agents submit a versioned JSON
scene or drawing-command document, receive a saved image plus structured
metadata, and can keep a named scene alive during a local daemon session.
Scenes compose vector shapes, local raster and SVG images, and embedded-font
text into one ordered GPU pass; PNG and keyframed GIF export both work from
the same scene document. Image nodes accept local PNG, JPEG, GIF, WebP, or
SVG assets; SVG assets are rasterized directly at each node's declared size
rather than decoded and rescaled.

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

A scene may also include an optional, scene-level full-canvas post-process
`effect`: a WGSL `shader` string defining exactly one function,
`fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32>`, that transforms
the already-composited color at each pixel (see `examples/effect.scene.json`,
a color-invert effect). The renderer wraps this function in a fixed internal
template (vertex stage, texture bindings) so authors only ever write a pure
color transform; they never supply bindings, vertex data, or a full pipeline.
A shader that fails WGSL validation is rejected with a structured error
rather than crashing the process.

```sh
cargo run -p renderer-cli -- render --input examples/effect.scene.json --output out-effect.png
```

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

### Per-session performance metrics

Each `daemon serve` run is one session. A background thread appends one JSON
line per request to `.renderer/metrics/<session-id>.jsonl` (method, scene ID,
duration, and outcome) so local performance testing doesn't need extra
instrumentation:

```sh
cat .renderer/metrics/*.jsonl | jq -c '{method, duration_ms, success}'
```

Recording only enqueues onto a channel from the request-handling path, so it
adds no latency there; if the metrics directory can't be created, the daemon
logs a warning and keeps serving without it.

## Inline terminal preview

`renderer show <path>` displays a local PNG or GIF directly inline in the
terminal -- useful for a coding-agent workflow where the agent renders a
visual and wants to show it to the human without them opening a separate
image viewer:

```sh
cargo run -p renderer-cli -- show out.png
cargo run -p renderer-cli -- show out.gif --loops 3
cargo run -p renderer-cli -- show --clear
```

### cmux native file preview

In [cmux](https://cmux.dev) (a native macOS terminal for running coding
agents, built on Ghostty), `show` skips terminal escape sequences entirely
and instead opens the file in cmux's own **native file-preview panel** -- a
UI surface completely separate from the terminal grid, so unlike a raw pty
write it can never land on top of an agent's own actively-redrawn TUI (e.g.
Claude Code's input box). This is detected automatically: if the
`CMUX_SOCKET_PATH` environment variable is set and its Unix domain socket
actually accepts a connection, `show` sends a `file.open` JSON-RPC request
over that socket and reports `"protocol":"cmux"` in its status JSON, instead
of running any of the Kitty/iTerm2/ANSI detection below. If the variable is
unset, or the socket can't be reached, `show` falls back to the
terminal-protocol behavior described in the rest of this section, unchanged.
Both static PNGs and animated GIFs open the same way -- cmux decodes and
loops the GIF itself, so there's no simulated-animation loop to run.
`show --clear` in cmux closes the most recently opened preview surface (its
id is persisted to `.renderer/cmux-preview-surface.json`, following the same
local-generated-state convention as `.renderer/metrics/`) via cmux's
`surface.close` RPC, or is a no-op if nothing was recorded yet -- rather than
sending a Kitty delete-all command that would have no effect there.

If stdout is itself a terminal, escape sequences are written there directly.
If not (for example when `renderer` is invoked as a detached subprocess by
an agent's tool-calling mechanism, so stdout is piped rather than a tty),
`show` walks the process ancestry (`ps -o ppid=,tty= -p <pid>`, capped at 32
ancestors) looking for the nearest ancestor process that does have a
controlling terminal, and writes there instead. This only works on macOS and
Linux; on Windows, or if no real terminal can be found either way, `show`
prints `no interactive terminal detected; image saved at <path>, open it
manually` and exits 0 rather than failing.

The terminal protocol is auto-detected from environment variables unless
overridden:

| Terminal signal | Protocol used |
| --- | --- |
| `KITTY_WINDOW_ID`, `TERM=xterm-kitty`, `TERM_PROGRAM=ghostty`, `GHOSTTY_RESOURCES_DIR`, `CMUX_WORKSPACE_ID`/`CMUX_SURFACE_ID` (cmux is Ghostty-based), or `TERM_PROGRAM=WezTerm` | Kitty graphics protocol |
| `TERM_PROGRAM=iTerm.app` | iTerm2 OSC 1337 inline images |
| anything else | ANSI 24-bit half-block fallback (e.g. Terminal.app) |

For an animated GIF, WezTerm and plain Kitty use the Kitty protocol's native
animation extension (the terminal itself handles frame timing and looping).
Ghostty and cmux support the Kitty graphics protocol but not its animation
extension, so a GIF headed there -- and any GIF on the ANSI fallback -- is
played back as *simulated* animation instead: `show` itself loops,
retransmitting each frame and sleeping for its delay, until `--loops` (0 or
omitted means loop forever, like a normal GIF) is exhausted or the process is
killed. iTerm2 always gets the raw GIF bytes as-is and handles decoding and
looping itself.

Flags: `--tty <path>` writes to a specific device file instead of
auto-detecting one (mainly for testing against a particular terminal
session); `--protocol <auto|kitty|iterm2|ansi>` overrides detection (this
flag has no effect on the cmux path, which is tried first regardless);
`--clear` sends only a Kitty delete-all-images command and exits (or, in
cmux, closes the last-opened preview surface as described above); `--loops
<n>` bounds a simulated/native animation's loop count.

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
