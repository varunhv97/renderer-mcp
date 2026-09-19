use std::path::PathBuf;

/// Every failure kind `run` can report. Mirrors the style of `RenderError`
/// in `crates/renderer/src/lib.rs` and `DaemonError` in
/// `crates/daemon/src/lib.rs`: one variant per distinguishable failure,
/// with a stable [`TerminalError::code`] a caller can surface directly
/// (the CLI's `show` command downcasts to this type for exactly that).
#[derive(Debug, thiserror::Error)]
pub enum TerminalError {
    /// `path` was `None` with `clear` false. The CLI's clap parser rejects
    /// this before it ever reaches this crate (`path` is
    /// `required_unless_present = "clear"`), but a direct caller like the
    /// MCP server's `show_image` tool has no such parser in front of it.
    #[error("path is required unless clear is true")]
    MissingPath,
    #[error("could not read {}: {source}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not inspect {}: {source}", path.display())]
    Image {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error(
        "{} is not a supported image for `show` (only PNG and GIF are accepted)",
        path.display()
    )]
    UnsupportedImage { path: PathBuf },
    #[error("unknown --protocol '{value}' (expected auto, kitty, or iterm2)")]
    InvalidProtocol { value: String },
    #[error("could not open terminal device {}: {source}", path.display())]
    TtyOpen {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Any failure talking to cmux's JSON-RPC control socket once a
    /// connection has actually been established -- a transport error, a
    /// malformed response, or an `ok: false` application-level error. A
    /// failure to connect at all is deliberately *not* represented here:
    /// that's treated as "cmux isn't available" and falls back to the
    /// terminal-protocol path instead of erroring.
    #[error("cmux error: {message}")]
    Cmux { message: String },
    /// A write/flush failure while sending an already-resolved protocol's
    /// escape sequences to an already-open terminal target. Distinct from
    /// [`TerminalError::TtyOpen`] (which covers failing to open the
    /// device in the first place).
    #[error("terminal write failed: {0}")]
    Io(#[from] std::io::Error),
}

impl TerminalError {
    /// A stable, machine-readable error code, matching the codes this
    /// crate's logic has always produced back when it lived in
    /// `crates/cli/src/main.rs`'s `CliError`.
    pub fn code(&self) -> &'static str {
        match self {
            TerminalError::MissingPath => "invalid_argument",
            TerminalError::Read { .. }
            | TerminalError::Image { .. }
            | TerminalError::TtyOpen { .. }
            | TerminalError::Io(_) => "io_error",
            TerminalError::UnsupportedImage { .. } => "unsupported_image",
            TerminalError::InvalidProtocol { .. } => "invalid_protocol",
            TerminalError::Cmux { .. } => "cmux_error",
        }
    }
}
