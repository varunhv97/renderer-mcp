#[cfg(doc)]
use crate::server::error_response;
use renderer_core::RenderError;
use std::net::SocketAddr;
use thiserror::Error;

/// Every failure this crate can report, whether raised locally (in-process
/// or server-side over TCP) or reconstructed client-side from a remote
/// error response ([`DaemonError::Remote`]). One variant per distinguishable
/// failure, each mapped to a stable wire code by [`DaemonError::code`].
#[derive(Debug, Error)]
pub enum DaemonError {
    #[error("GPU initialization failed: {0}")]
    Renderer(#[from] RenderError),
    #[error("scene validation failed: {0}")]
    InvalidScene(#[from] renderer_schema::SceneValidationError),
    #[error("scene ID must not be empty")]
    EmptySceneId,
    #[error("named scene does not exist: {0}")]
    SceneNotFound(String),
    #[error("named scene already exists: {0}")]
    SceneAlreadyExists(String),
    #[error("scene is too large to return within the daemon response limit")]
    SceneResponseTooLarge,
    #[error("image scenes require an explicit asset root")]
    AssetRootRequired,
    #[error("scene revision conflict: expected {expected}, current {actual}")]
    RevisionConflict { expected: u64, actual: u64 },
    #[error("patch operation {index} cannot remove unknown node: {id}")]
    NodeNotFound { index: usize, id: String },
    #[error("daemon endpoint must use a loopback IP address: {0}")]
    NonLoopbackEndpoint(SocketAddr),
    #[error("could not bind daemon endpoint: {0}")]
    Bind(#[source] std::io::Error),
    #[error("daemon protocol error: {0}")]
    Protocol(String),
    #[error("daemon connection failed: {0}")]
    Connection(#[source] std::io::Error),
    #[error("{message}")]
    Remote { code: String, message: String },
}

impl DaemonError {
    /// Stable, machine-readable code identifying this error's kind.
    ///
    /// This is the single source of truth for the mapping from error kind to
    /// wire-level code: [`error_response`] calls this method rather than
    /// duplicating the mapping, so a server-side error and a client-side
    /// error of the same kind always report the same code. For
    /// [`DaemonError::Remote`], this returns the code exactly as reported by
    /// the daemon that produced it, rather than recomputing one locally.
    pub fn code(&self) -> String {
        match self {
            DaemonError::RevisionConflict { .. } => "revision_conflict".to_string(),
            DaemonError::SceneNotFound(_) | DaemonError::NodeNotFound { .. } => {
                "not_found".to_string()
            }
            DaemonError::SceneAlreadyExists(_) => "already_exists".to_string(),
            DaemonError::InvalidScene(_)
            | DaemonError::EmptySceneId
            | DaemonError::SceneResponseTooLarge => "invalid_scene".to_string(),
            DaemonError::Protocol(_) => "invalid_request".to_string(),
            DaemonError::Remote { code, .. } => code.clone(),
            DaemonError::Renderer(_)
            | DaemonError::AssetRootRequired
            | DaemonError::NonLoopbackEndpoint(_)
            | DaemonError::Bind(_)
            | DaemonError::Connection(_) => "internal_error".to_string(),
        }
    }
}
