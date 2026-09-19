# MCP server

`renderer-mcp` speaks newline-delimited JSON-RPC over stdio (`initialize`,
`tools/list`, `tools/call`) and never listens on a network socket. It exposes
one-shot `render_scene` plus named-scene tools (`create_scene`, `get_scene`,
`replace_scene`, `patch_scene`, `render_named_scene`, `export_named_gif`,
`inspect_image`, `destroy_scene`) backed by the same daemon protocol used by
the CLI, plus `show_image` -- the MCP counterpart to `renderer show`
described in [Inline terminal preview](terminal-preview.md), sharing its implementation via the `renderer-terminal`
crate. Named-scene tools talk to a renderer daemon, which `renderer-mcp` starts
itself on first use (see below); `render_scene` and `show_image` need none. `show_image` takes `path` (required unless `clear` is true),
`protocol` (`auto`/`kitty`/`iterm2`, default `auto`), `tty`, `clear`,
and `loops` -- the same parameters as the CLI's flags -- and returns only a
text summary (`status` and `protocol` used), not image content: the image is
already visible to the user via the terminal/cmux mechanism, so there's
nothing to hand back as base64.

To register it with an MCP-capable agent host, point the host at the built
binary, for example:

```json
{
  "mcpServers": {
    "renderer": {
      "command": "/absolute/path/to/target/release/renderer-mcp"
    }
  }
}
```

## Which daemon the named-scene tools use

The first named-scene call resolves a daemon like this:

1. If `RENDERER_DAEMON_ENDPOINT` is set and a daemon answers there, use it.
2. Otherwise start a daemon inside the `renderer-mcp` process, on a background
   thread: on the `RENDERER_DAEMON_ENDPOINT` address if one is set (so a fixed
   port keeps working), else on a free loopback port chosen by the OS.

The built-in daemon needs a GPU adapter like any other, and reports a clear
error if it cannot get one or if the configured address is already in use by
something that isn't a renderer daemon. It lives only as long as the
`renderer-mcp` process, so its scenes are gone when the agent session ends, and
it doesn't write `.renderer/metrics` files. To keep scenes across sessions, or
to share one daemon with the CLI, run `renderer daemon serve --endpoint
127.0.0.1:9472` yourself and set `RENDERER_DAEMON_ENDPOINT`:

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

Named PNG/GIF exports return the daemon's metadata followed by inline base64
image content so compatible hosts can display the result immediately.
