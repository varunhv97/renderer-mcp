use crate::error::TerminalError;
use std::path::Path;

pub(crate) fn try_show(_path: &Path) -> Result<Option<&'static str>, TerminalError> {
    Ok(None)
}

// Mirrors the unix module's variants so `run_clear`'s exhaustive match
// compiles on every platform; only `Unavailable` is ever produced here,
// since cmux is a macOS app reached over a Unix domain socket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum CmuxClearOutcome {
    Cleared,
    NoSurface,
    Unavailable,
}

pub(crate) fn try_clear() -> Result<CmuxClearOutcome, TerminalError> {
    Ok(CmuxClearOutcome::Unavailable)
}
