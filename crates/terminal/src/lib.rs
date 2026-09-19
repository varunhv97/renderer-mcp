//! Inline terminal image/GIF preview, shared by the `renderer` CLI's `show`
//! subcommand and the `renderer-mcp` server's `show_image` tool.
//!
//! This crate is the single implementation behind both surfaces, per this
//! project's founding "CLI and MCP expose the same capabilities" principle:
//! terminal detection, a Kitty graphics protocol encoder with animation
//! support, an iTerm2 OSC 1337 encoder, a process-ancestry tty-discovery
//! mechanism for when the calling process has no controlling terminal of
//! its own, a `cmux_preview` submodule that talks to cmux's native
//! file-preview feature over a JSON-RPC Unix socket when
//! `$CMUX_SOCKET_PATH` is available, and an `open_with_system_viewer`
//! fallback (the OS's own default file opener, e.g. Preview.app via `open`
//! on macOS) for every other case -- no Kitty/iTerm2 protocol support
//! detected, or nowhere safely writable to send escape sequences to. There
//! is deliberately no text-based (ANSI half-block) fallback: it only ever
//! mattered for terminals supporting neither real graphics protocol (in
//! practice, just Apple's Terminal.app), and building on real feedback
//! this session -- three real bugs found live, then a fundamental quality
//! ceiling (confirmed live: even Floyd-Steinberg dithering read as visible
//! noise, not smoother color, at real terminal-cell resolution) -- a
//! full-quality external viewer is a strictly better fallback than a
//! blocky, palette-limited text approximation.
//!
//! The public entry point is [`run`]: build a [`ShowRequest`], call `run`,
//! and render the returned [`ShowOutcome`] (or [`TerminalError`]) however
//! the caller's own protocol requires -- the CLI turns it into its
//! structured JSON-on-stdout / `{"code","message"}`-on-stderr convention,
//! and the MCP server turns it into a text content block.

#[cfg(unix)]
mod cmux_preview;
#[cfg(not(unix))]
#[path = "cmux_preview_stub.rs"]
mod cmux_preview;
mod error;
mod frames;
mod iterm2;
mod kitty;
mod protocol;
mod show;
mod target;

pub use error::*;
pub use show::*;
