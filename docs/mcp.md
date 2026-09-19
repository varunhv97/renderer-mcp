# MCP server

`renderer-mcp` speaks newline-delimited JSON-RPC over stdio (`initialize`,
`tools/list`, `tools/call`) and never listens on a network socket. It exposes
one-shot `render_scene` plus named-scene tools (`create_scene`, `get_scene`,
`replace_scene`, `patch_scene`, `render_named_scene`, `export_named_gif`,
`inspect_image`, `destroy_scene`) backed by the same daemon protocol used by
the CLI, plus `show_image` -- the MCP counterpart to `renderer show`
described in [Inline terminal preview](terminal-preview.md), sharing its implementation via the `renderer-terminal`
crate. Named-scene tools require a running `daemon serve` and the endpoint
supplied via `RENDERER_DAEMON_ENDPOINT`; `render_scene` and `show_image` need
neither. `show_image` takes `path` (required unless `clear` is true),
`protocol` (`auto`/`kitty`/`iterm2`, default `auto`), `tty`, `clear`,
and `loops` -- the same parameters as the CLI's flags -- and returns only a
text summary (`status` and `protocol` used), not image content: the image is
already visible to the user via the terminal/cmux mechanism, so there's
nothing to hand back as base64.

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
