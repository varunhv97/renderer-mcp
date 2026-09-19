#[cfg(doc)]
use crate::client::DaemonClient;
#[cfg(doc)]
use crate::daemon::RendererDaemon;
use crate::daemon::{RenderResult, SceneSnapshot};
#[cfg(doc)]
use crate::error::DaemonError;
#[cfg(doc)]
use crate::server::{dispatch, error_response, read_request};
#[cfg(doc)]
use crate::store::ensure_snapshot_fits_response;
use renderer_schema::{ScenePatchV1, SceneV1};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The only value [`DaemonEnvelope::version`] accepts, on both the request
/// and response side. Checked exactly (see [`read_request`] and
/// [`DaemonClient::call`]) rather than negotiated, so a version mismatch
/// fails fast with a clear protocol error instead of a confusing downstream
/// deserialization failure.
pub const DAEMON_PROTOCOL_VERSION: &str = "renderer.daemon.v1";

/// Hard cap on one request or response's serialized size, enforced while
/// reading ([`read_request`], [`DaemonClient::call`]) and, for responses,
/// pre-checked before a mutation commits (see
/// [`ensure_snapshot_fits_response`]) so a caller never gets back a scene
/// it can't parse.
pub const MAX_REQUEST_BYTES: usize = 1024 * 1024;

/// One daemon RPC, tagged by `method` in JSON. Carried inside a
/// [`DaemonEnvelope`] (which adds the protocol version) on the wire; sent
/// directly when calling [`RendererDaemon`]'s methods in-process.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(
    tag = "method",
    content = "params",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum DaemonRequest {
    Health,
    CreateScene {
        scene_id: String,
        scene: SceneV1,
        asset_root: Option<PathBuf>,
    },
    GetScene {
        scene_id: String,
    },
    ReplaceScene {
        scene_id: String,
        scene: SceneV1,
        expected_revision: Option<u64>,
        asset_root: Option<PathBuf>,
    },
    PatchScene {
        scene_id: String,
        patch: ScenePatchV1,
    },
    RenderScene {
        scene_id: String,
        output: PathBuf,
    },
    RenderGifScene {
        scene_id: String,
        output: PathBuf,
    },
    DestroyScene {
        scene_id: String,
    },
}

/// The wire envelope wrapping every request: `version` must equal
/// [`DAEMON_PROTOCOL_VERSION`] (checked in [`read_request`]) so a client and
/// server built from different schema revisions fail fast with a clear
/// protocol error instead of silently misinterpreting fields.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonEnvelope {
    pub version: String,
    #[serde(flatten)]
    pub request: DaemonRequest,
}

/// The success payload of a [`DaemonResponse`], tagged by `kind` in JSON.
/// `Rendered` is reused for both `render_scene` and `render_gif_scene`,
/// distinguished by the caller already knowing which request it sent (and,
/// for MCP, by [`RenderResult`]'s `frame_count`).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DaemonResult {
    Health,
    Revision {
        revision: u64,
    },
    Scene {
        snapshot: SceneSnapshot,
    },
    Rendered {
        output: PathBuf,
        image: RenderResult,
    },
    Destroyed,
}

/// The wire envelope wrapping every response: exactly one of `result`/
/// `error` is populated (never both, never neither) -- see [`dispatch`] and
/// [`error_response`], the only two places that construct one.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DaemonResponse {
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<DaemonResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<DaemonProtocolError>,
}

/// A [`DaemonResponse`]'s error case: a stable `code` (see
/// [`DaemonError::code`]) plus a human-readable `message`. `code` is what a
/// client should match on; `message` is for display only and isn't
/// guaranteed stable across versions.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DaemonProtocolError {
    pub code: String,
    pub message: String,
}
