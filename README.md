# RendererCli

A local GPU-accelerated visual runtime for coding agents.

Agents submit a versioned JSON scene (vector shapes, local images and SVG,
text), and get back a saved PNG or animated GIF plus structured metadata. It
ships as a CLI (`renderer`) and an MCP server (`renderer-mcp`), so an agent can
draw something, look at it, and iterate.

## Install

You need a recent stable Rust toolchain (Rust 1.85 or newer) and a GPU that
[wgpu](https://wgpu.rs) can use: Metal on macOS, Vulkan on Linux, or DX12 on
Windows. There is no software fallback, so a headless machine without a GPU
cannot render.

```sh
git clone https://github.com/varunhv97/renderer-mcp.git
cd renderer-mcp
cargo build --release
```

This produces `target/release/renderer` and `target/release/renderer-mcp`. To
put them on your `PATH` instead:

```sh
cargo install --path crates/cli
cargo install --path crates/mcp
```

Check that it works:

```sh
renderer render --input examples/basic.scene.json --output out.png
renderer show out.png
```

## Use it from an agent

Register the MCP server. That's all: named scenes work out of the box, because
`renderer-mcp` starts its own renderer daemon the first time one is needed.

```sh
claude mcp add renderer -- /absolute/path/to/renderer-mcp
```

Restart the agent so it picks the server up.

The built-in daemon lives and dies with the agent session, so its scenes are
gone when the session ends. To keep scenes around, or to share one daemon
between the CLI and your agent, run it yourself and point the server at it. If
nothing is listening there yet, `renderer-mcp` starts one on that address:

```sh
renderer daemon serve --endpoint 127.0.0.1:9472

claude mcp add renderer \
  --env RENDERER_DAEMON_ENDPOINT=127.0.0.1:9472 \
  -- /absolute/path/to/renderer-mcp
```

After rebuilding, restart a daemon you started yourself: it keeps running the
old code.

Typical flow: `create_scene`, optionally `patch_scene`, then `export_named_gif`
(or `render_named_scene` for a still), then `show_image`.

| Tool | Uses a daemon | What it does |
| --- | --- | --- |
| `render_scene` | no | One-shot render of an inline scene to PNG |
| `create_scene`, `get_scene`, `replace_scene`, `patch_scene`, `destroy_scene` | yes | Manage a named scene that stays alive in the daemon |
| `render_named_scene` | yes | Render a named scene to PNG (a static frame, ignores the timeline) |
| `export_named_gif` | yes | Export a scene's timeline as an animated GIF |
| `inspect_image` | no | Dimensions, MIME type and SHA-256 of an image file |
| `show_image` | no | Display a PNG or GIF inline in your terminal |

## Supported terminals

`show_image` and `renderer show` display results inline when the terminal has a
real graphics protocol, and open the OS viewer otherwise.

| Environment | How the image is shown | Animated GIFs |
| --- | --- | --- |
| [cmux](https://cmux.dev) | Native file-preview panel | Animate natively |
| Kitty, WezTerm | Kitty graphics protocol | Animate natively |
| Ghostty | Kitty graphics protocol | Simulated by re-sending frames |
| iTerm2 | Inline images (OSC 1337) | iTerm2 loops the GIF itself |
| Apple Terminal.app, anything else | OS default viewer (`open` on macOS, `xdg-open` on Linux) | Handled by the viewer |
| Windows | No viewer is launched; the path is printed | none |

Limitations:

- There is no text or ANSI fallback.
- Use through tmux, screen, or SSH has not been tested.
- The daemon client has a fixed 5 second timeout, so a very large or long
  animated export can fail with a connection error.
- The daemon binds loopback addresses only, and scenes can reference local
  assets only. Remote asset fetching is intentionally not supported.

## Docs

- [Rendering and scenes](docs/rendering.md): the scene format, node types,
  animation, gradients, effects, and one-shot rendering
- [Persistent named scenes](docs/daemon.md): the daemon, patching, and metrics
- [MCP server](docs/mcp.md): the JSON-RPC server and its tools in detail
- [Inline terminal preview](docs/terminal-preview.md): protocol detection,
  cmux, and every terminal flag
- [Development](docs/development.md): checks, coverage, and testing

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
