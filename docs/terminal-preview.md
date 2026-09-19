# Inline terminal preview

`renderer show <path>` displays a local PNG or GIF directly inline in the
terminal -- useful for a coding-agent workflow where the agent renders a
visual and wants to show it to the human without them opening a separate
image viewer:

```sh
cargo run -p renderer-cli -- show out.png
cargo run -p renderer-cli -- show out.gif --loops 3
cargo run -p renderer-cli -- show --clear
```

## cmux native file preview

In [cmux](https://cmux.dev) (a native macOS terminal for running coding
agents, built on Ghostty), `show` skips terminal escape sequences entirely
and instead opens the file in cmux's own **native file-preview panel** -- a
UI surface completely separate from the terminal grid, so unlike a raw pty
write it can never land on top of an agent's own actively-redrawn TUI (e.g.
Claude Code's input box). This is detected automatically: if the
`CMUX_SOCKET_PATH` environment variable is set and its Unix domain socket
actually accepts a connection, `show` sends a `file.open` JSON-RPC request
over that socket and reports `"protocol":"cmux"` in its status JSON, instead
of running any of the Kitty/iTerm2 detection below. If the variable is
unset, or the socket can't be reached, `show` falls back to the
terminal-protocol behavior described in the rest of this section, unchanged.
Both static PNGs and animated GIFs open the same way -- cmux decodes and
loops the GIF itself, so there's no simulated-animation loop to run.
`show --clear` in cmux closes the most recently opened preview surface (its
id is persisted to `.renderer/cmux-preview-surface.json`, following the same
local-generated-state convention as `.renderer/metrics/`) via cmux's
`surface.close` RPC, or is a no-op if nothing was recorded yet -- rather than
sending a Kitty delete-all command that would have no effect there.

When cmux isn't available, `show` only ever reaches for a **real graphics
protocol** -- there is no text-based fallback. The terminal protocol is
auto-detected from environment variables unless overridden:

| Terminal signal | Protocol used |
| --- | --- |
| `KITTY_WINDOW_ID`, `TERM=xterm-kitty`, `TERM_PROGRAM=ghostty`, `GHOSTTY_RESOURCES_DIR`, `CMUX_WORKSPACE_ID`/`CMUX_SURFACE_ID` (cmux is Ghostty-based), or `TERM_PROGRAM=WezTerm` | Kitty graphics protocol |
| `TERM_PROGRAM=iTerm.app` | iTerm2 OSC 1337 inline images |
| anything else (Apple's Terminal.app included -- it implements neither) | the OS's own default file opener |

Earlier versions of this tool had a third path: an ANSI 24-bit half-block
text approximation for terminals supporting neither real protocol, driven by
writing raw escape sequences directly into whatever tty could be found.
Removed entirely after a full round of live debugging in a real Terminal.app
window turned up a real quality ceiling as well as a safety issue: writing
into a discovered-but-unowned tty (the case when `renderer` runs as a
detached subprocess of an agent's tool-calling mechanism, so stdout is piped
rather than a real terminal) risks two writers' raw bytes interleaving
mid-escape-sequence, which once corrupted a real terminal window badly
enough that not even a terminal reset run the same way could recover it (it
needs tty ioctl access that subprocess doesn't have); and even after fixing
that and the resulting color-palette bugs, the text approximation's quality
ceiling turned out to be low regardless -- confirmed live, even proper
Floyd-Steinberg error-diffusion dithering read as visible noise rather than
a smoother gradient at real terminal-cell resolution. A full-quality
external viewer is strictly better than a blocky, palette-limited
approximation, so that's the fallback now for every case that isn't a real
graphics protocol: no known Kitty/iTerm2 signals, or nowhere safe to write
escape sequences to at all (the same discovered-tty case above, which now
never attempts a raw write regardless of which protocol would have been
used). `show` launches the OS's own default file opener as a separate
process -- `open` on macOS, `xdg-open` on Linux -- reports
`"protocol":"system_viewer"`, and lets that GUI application (e.g. Preview)
display the image in its own window, untangled from the terminal entirely.
`--tty <path>` still writes Kitty/iTerm2 escape sequences directly to an
explicitly named device when a real protocol is detected, for a caller that
knows that target is safe. On an OS with no such opener, or if no terminal
or opener is available at all, `show` prints `no interactive terminal
detected; image saved at <path>, open it manually` and exits 0 rather than
failing.

For an animated GIF, WezTerm and plain Kitty use the Kitty protocol's native
animation extension (the terminal itself handles frame timing and looping).
Ghostty and cmux support the Kitty graphics protocol but not its animation
extension, so a GIF headed there is played back as *simulated* animation
instead: `show` itself loops, retransmitting each frame and sleeping for its
delay, until `--loops` (0 or omitted means loop forever, like a normal GIF)
is exhausted or the process is killed. iTerm2 always gets the raw GIF bytes
as-is and handles decoding and looping itself. The system-viewer path
doesn't loop anything itself -- Preview/QuickLook handle GIF animation on
their own once the file is open.

Flags: `--tty <path>` writes to a specific device file instead of
auto-detecting one (mainly for testing against a particular terminal
session); `--protocol <auto|kitty|iterm2>` overrides detection (this flag has
no effect on the cmux path, which is tried first regardless, or on the
system-viewer fallback, which needs no protocol at all); `--clear` sends
only a Kitty delete-all-images command and exits (or, in cmux, closes the
last-opened preview surface as described above); `--loops <n>` bounds a
simulated/native animation's loop count.

All of the above -- terminal detection, the Kitty/iTerm2 encoders, the
system-viewer fallback, and the cmux integration -- lives in the shared
`renderer-terminal` crate (`crates/terminal`), not the CLI binary itself, so
it also backs the MCP server's `show_image` tool ([MCP server](mcp.md)) with identical
behavior (same protocol detection, same cmux preference, same fallback
messages), per this workspace's CLI/MCP parity principle.
