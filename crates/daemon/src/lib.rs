//! Persistent local scene state and its loopback-only daemon protocol.
//!
//! This crate is the stateful named-scene server the `cli` and `mcp` crates
//! both talk to over a line-delimited JSON protocol on a loopback TCP
//! socket ([`serve`] / [`DaemonClient`]): it holds an in-memory,
//! revisioned [`SceneV1`] per `scene_id` (see [`SceneStore`]) so a caller
//! can create a scene once and then `get`/`replace`/`patch`/`render` it by
//! name across many separate requests, instead of resending the whole
//! document every time. It also owns the one [`GpuRenderer`] used for both
//! named-scene and inline (one-shot, unstored) renders, and can be driven
//! in-process (`RendererDaemon`, used by `cli`'s `render`/`show` commands
//! and by `mcp`'s `render_scene` tool) without going over TCP at all.
//!
//! [`SceneV1`]: renderer_schema::SceneV1
//! [`GpuRenderer`]: renderer_core::GpuRenderer

use renderer_core::{GpuRenderer, RenderError, RenderedImage};
use renderer_schema::{PatchOperationV1, ScenePatchV1, SceneV1};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

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
const CLIENT_IO_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_METRICS_DIR: &str = ".renderer/metrics";

/// One GPU renderer plus the in-memory named-scene store, either driven
/// in-process or wrapped by [`serve`] for TCP access. A single
/// `GpuRenderer` is shared across every scene (named or inline) rather than
/// one per scene, since GPU device/pipeline setup is the expensive part.
#[derive(Debug)]
pub struct RendererDaemon {
    renderer: GpuRenderer,
    scenes: SceneStore,
}

/// Named scenes held by one running daemon, keyed by caller-chosen
/// `scene_id`. A `BTreeMap` (not a `HashMap`) mainly for deterministic
/// iteration/debugging; lookups aren't hot enough for the difference to
/// matter.
#[derive(Clone, Debug, Default)]
struct SceneStore {
    scenes: BTreeMap<String, StoredScene>,
}

/// A stored scene's current state: the document itself, a revision counter
/// incremented on every successful `replace`/`patch` (for optimistic
/// concurrency -- see `check_revision`), and the asset root image nodes in
/// it resolve `source` paths against.
#[derive(Clone, Debug)]
struct StoredScene {
    revision: u64,
    scene: SceneV1,
    asset_root: Option<PathBuf>,
}

/// A named scene's document plus the `scene_id`/`revision` it was fetched
/// under -- the payload of `get_scene` and every mutating call's response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SceneSnapshot {
    pub scene_id: String,
    pub revision: u64,
    pub scene: SceneV1,
}

/// Metadata about one completed render, returned instead of the raw image
/// bytes: callers read the file at the output path they gave (or, for MCP,
/// this daemon crate isn't the one that base64-encodes it -- see
/// `renderer_mcp::inline_render_response`).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RenderResult {
    pub width: u32,
    pub height: u32,
    pub sha256: String,
    pub frame_count: u32,
    pub warnings: Vec<String>,
}

impl From<RenderedImage> for RenderResult {
    fn from(image: RenderedImage) -> Self {
        Self {
            width: image.width,
            height: image.height,
            sha256: image.sha256,
            frame_count: image.frame_count,
            warnings: image.warnings,
        }
    }
}

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

impl RendererDaemon {
    pub fn new() -> Result<Self, DaemonError> {
        Ok(Self {
            renderer: GpuRenderer::new()?,
            scenes: SceneStore::default(),
        })
    }

    pub fn create_scene(
        &mut self,
        scene_id: String,
        scene: SceneV1,
        asset_root: Option<PathBuf>,
    ) -> Result<u64, DaemonError> {
        self.scenes.create(scene_id, scene, asset_root)
    }

    pub fn get_scene(&self, scene_id: &str) -> Result<SceneSnapshot, DaemonError> {
        self.scenes.get(scene_id)
    }

    pub fn replace_scene(
        &mut self,
        scene_id: String,
        scene: SceneV1,
        expected_revision: Option<u64>,
        asset_root: Option<PathBuf>,
    ) -> Result<u64, DaemonError> {
        self.scenes
            .replace(scene_id, scene, expected_revision, asset_root)
    }

    pub fn patch_scene(&mut self, scene_id: &str, patch: ScenePatchV1) -> Result<u64, DaemonError> {
        self.scenes.patch(scene_id, patch)
    }

    pub fn destroy_scene(&mut self, scene_id: &str) -> Result<(), DaemonError> {
        self.scenes.destroy(scene_id)
    }

    pub fn render_scene(
        &self,
        scene_id: &str,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        let stored = self
            .scenes
            .scenes
            .get(scene_id)
            .ok_or_else(|| DaemonError::SceneNotFound(scene_id.into()))?;
        require_asset_root(&stored.scene, &stored.asset_root)?;
        let root = stored.asset_root.as_deref().unwrap_or(Path::new("."));
        Ok(self
            .renderer
            .render_png_with_asset_root(&stored.scene, output, root)?)
    }

    pub fn render_gif_scene(
        &self,
        scene_id: &str,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        let stored = self
            .scenes
            .scenes
            .get(scene_id)
            .ok_or_else(|| DaemonError::SceneNotFound(scene_id.into()))?;
        require_asset_root(&stored.scene, &stored.asset_root)?;
        let root = stored.asset_root.as_deref().unwrap_or(Path::new("."));
        Ok(self
            .renderer
            .render_gif_with_asset_root(&stored.scene, output, root)?)
    }

    pub fn render_inline(
        &self,
        scene: &SceneV1,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self.renderer.render_png(scene, output)?)
    }

    pub fn render_inline_with_asset_root(
        &self,
        scene: &SceneV1,
        output: &Path,
        asset_root: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self
            .renderer
            .render_png_with_asset_root(scene, output, asset_root)?)
    }

    pub fn render_gif_inline(
        &self,
        scene: &SceneV1,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self.renderer.render_gif(scene, output)?)
    }

    pub fn render_gif_inline_with_asset_root(
        &self,
        scene: &SceneV1,
        output: &Path,
        asset_root: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self
            .renderer
            .render_gif_with_asset_root(scene, output, asset_root)?)
    }

    pub fn scene_count(&self) -> usize {
        self.scenes.scenes.len()
    }
}

impl SceneStore {
    fn create(
        &mut self,
        scene_id: String,
        scene: SceneV1,
        asset_root: Option<PathBuf>,
    ) -> Result<u64, DaemonError> {
        ensure_scene_id(&scene_id)?;
        scene.validate()?;
        if self.scenes.contains_key(&scene_id) {
            return Err(DaemonError::SceneAlreadyExists(scene_id));
        }
        ensure_snapshot_fits_response(&scene_id, 1, &scene)?;
        require_asset_root(&scene, &asset_root)?;
        self.scenes.insert(
            scene_id,
            StoredScene {
                revision: 1,
                scene,
                asset_root,
            },
        );
        Ok(1)
    }

    fn get(&self, scene_id: &str) -> Result<SceneSnapshot, DaemonError> {
        let stored = self
            .scenes
            .get(scene_id)
            .ok_or_else(|| DaemonError::SceneNotFound(scene_id.into()))?;
        Ok(SceneSnapshot {
            scene_id: scene_id.into(),
            revision: stored.revision,
            scene: stored.scene.clone(),
        })
    }

    fn replace(
        &mut self,
        scene_id: String,
        scene: SceneV1,
        expected_revision: Option<u64>,
        asset_root: Option<PathBuf>,
    ) -> Result<u64, DaemonError> {
        ensure_scene_id(&scene_id)?;
        scene.validate()?;
        let stored = self
            .scenes
            .get_mut(&scene_id)
            .ok_or_else(|| DaemonError::SceneNotFound(scene_id.clone()))?;
        check_revision(expected_revision, stored.revision)?;
        require_asset_root(&scene, &asset_root)?;
        ensure_snapshot_fits_response(&scene_id, stored.revision + 1, &scene)?;
        stored.revision += 1;
        stored.scene = scene;
        stored.asset_root = asset_root;
        Ok(stored.revision)
    }

    fn patch(&mut self, scene_id: &str, patch: ScenePatchV1) -> Result<u64, DaemonError> {
        patch.validate()?;
        let stored = self
            .scenes
            .get_mut(scene_id)
            .ok_or_else(|| DaemonError::SceneNotFound(scene_id.into()))?;
        check_revision(patch.expected_revision, stored.revision)?;
        let mut candidate = stored.scene.clone();
        for (index, operation) in patch.operations.into_iter().enumerate() {
            apply_operation(&mut candidate, operation, index)?;
        }
        candidate.validate()?;
        require_asset_root(&candidate, &stored.asset_root)?;
        ensure_snapshot_fits_response(scene_id, stored.revision + 1, &candidate)?;
        stored.revision += 1;
        stored.scene = candidate;
        Ok(stored.revision)
    }

    fn destroy(&mut self, scene_id: &str) -> Result<(), DaemonError> {
        self.scenes
            .remove(scene_id)
            .ok_or_else(|| DaemonError::SceneNotFound(scene_id.into()))?;
        Ok(())
    }
}

/// Rejects a create/replace/patch before committing it if the resulting
/// scene couldn't later be returned by `get_scene`: this serializes the
/// exact [`DaemonResponse`] shape a `get_scene` reply would use (not just
/// the raw scene) so the size check matches [`MAX_REQUEST_BYTES`] exactly,
/// including envelope overhead, rather than approximating it.
fn ensure_snapshot_fits_response(
    scene_id: &str,
    revision: u64,
    scene: &SceneV1,
) -> Result<(), DaemonError> {
    let response = DaemonResponse {
        version: DAEMON_PROTOCOL_VERSION.into(),
        result: Some(DaemonResult::Scene {
            snapshot: SceneSnapshot {
                scene_id: scene_id.into(),
                revision,
                scene: scene.clone(),
            },
        }),
        error: None,
    };
    let size = serde_json::to_vec(&response)
        .map_err(|error| DaemonError::Protocol(error.to_string()))?
        .len()
        + 1;
    if size > MAX_REQUEST_BYTES {
        Err(DaemonError::SceneResponseTooLarge)
    } else {
        Ok(())
    }
}

fn ensure_scene_id(scene_id: &str) -> Result<(), DaemonError> {
    if scene_id.trim().is_empty() {
        Err(DaemonError::EmptySceneId)
    } else {
        Ok(())
    }
}

fn require_asset_root(scene: &SceneV1, asset_root: &Option<PathBuf>) -> Result<(), DaemonError> {
    if scene
        .nodes
        .iter()
        .any(|node| matches!(node.kind, renderer_schema::NodeKindV1::Image { .. }))
        && asset_root.is_none()
    {
        Err(DaemonError::AssetRootRequired)
    } else {
        Ok(())
    }
}

fn check_revision(expected: Option<u64>, actual: u64) -> Result<(), DaemonError> {
    if let Some(expected) = expected.filter(|expected| *expected != actual) {
        Err(DaemonError::RevisionConflict { expected, actual })
    } else {
        Ok(())
    }
}

fn apply_operation(
    scene: &mut SceneV1,
    operation: PatchOperationV1,
    index: usize,
) -> Result<(), DaemonError> {
    match operation {
        PatchOperationV1::SetCanvas { canvas } => scene.canvas = canvas,
        PatchOperationV1::UpsertNode { node } => {
            if let Some(existing) = scene
                .nodes
                .iter_mut()
                .find(|existing| existing.id == node.id)
            {
                *existing = node;
            } else {
                scene.nodes.push(node);
            }
        }
        PatchOperationV1::RemoveNode { id } => {
            let position = scene
                .nodes
                .iter()
                .position(|node| node.id == id)
                .ok_or_else(|| DaemonError::NodeNotFound {
                    index,
                    id: id.clone(),
                })?;
            scene.nodes.remove(position);
        }
        PatchOperationV1::SetTimeline { timeline } => scene.timeline = Some(timeline),
        PatchOperationV1::ClearTimeline => scene.timeline = None,
    }
    Ok(())
}

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

/// One JSON-lines event describing a single daemon request, appended to a
/// per-session metrics file by a dedicated background thread so recording
/// never adds latency to request handling.
#[derive(Clone, Debug, Serialize)]
struct MetricEvent {
    session_id: String,
    timestamp_ms: u128,
    method: String,
    scene_id: Option<String>,
    duration_ms: f64,
    success: bool,
    error_code: Option<String>,
}

/// Records request timings for one daemon session (one `serve` invocation)
/// without blocking the connection-handling loop: `record` only pushes onto
/// an unbounded channel, and a background thread owns the actual file I/O.
struct MetricsRecorder {
    session_id: String,
    sender: mpsc::Sender<MetricEvent>,
}

impl MetricsRecorder {
    /// Starts a session, creating `metrics_dir` and a `<session_id>.jsonl`
    /// file inside it. Returns `Err` only if the directory/file cannot be
    /// created; callers should treat that as non-fatal and serve without
    /// metrics rather than refuse to start the daemon.
    fn start(metrics_dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(metrics_dir)?;
        let session_id = format!(
            "session-{}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or_default(),
            std::process::id()
        );
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(metrics_dir.join(format!("{session_id}.jsonl")))?;
        let (sender, receiver) = mpsc::channel::<MetricEvent>();
        thread::spawn(move || {
            for event in receiver {
                if let Ok(line) = serde_json::to_string(&event) {
                    let _ = writeln!(file, "{line}");
                    let _ = file.flush();
                }
            }
        });
        Ok(Self { session_id, sender })
    }

    fn record(
        &self,
        method: &str,
        scene_id: Option<&str>,
        duration: Duration,
        success: bool,
        error_code: Option<&str>,
    ) {
        let event = MetricEvent {
            session_id: self.session_id.clone(),
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or_default(),
            method: method.into(),
            scene_id: scene_id.map(str::to_owned),
            duration_ms: duration.as_secs_f64() * 1000.0,
            success,
            error_code: error_code.map(str::to_owned),
        };
        // An unbounded send only fails if the receiver thread is gone, which
        // only happens if it panicked; dropping the event is preferable to
        // letting a metrics failure affect scene rendering.
        let _ = self.sender.send(event);
    }
}

/// A short label plus the scene ID (if any) a request applies to, used only
/// for metrics: it must not require cloning the (potentially large) scene
/// payload out of the request.
fn request_label(request: &DaemonRequest) -> (&'static str, Option<&str>) {
    match request {
        DaemonRequest::Health => ("health", None),
        DaemonRequest::CreateScene { scene_id, .. } => ("create_scene", Some(scene_id)),
        DaemonRequest::GetScene { scene_id } => ("get_scene", Some(scene_id)),
        DaemonRequest::ReplaceScene { scene_id, .. } => ("replace_scene", Some(scene_id)),
        DaemonRequest::PatchScene { scene_id, .. } => ("patch_scene", Some(scene_id)),
        DaemonRequest::RenderScene { scene_id, .. } => ("render_scene", Some(scene_id)),
        DaemonRequest::RenderGifScene { scene_id, .. } => ("render_gif_scene", Some(scene_id)),
        DaemonRequest::DestroyScene { scene_id } => ("destroy_scene", Some(scene_id)),
    }
}

pub fn serve(endpoint: SocketAddr) -> Result<(), DaemonError> {
    serve_with_metrics_dir(endpoint, Some(Path::new(DEFAULT_METRICS_DIR)))
}

/// Same as [`serve`], but lets callers redirect (or disable, via `None`)
/// per-session metrics output. Tests use this to avoid writing into the
/// repository's `.renderer/` directory.
pub fn serve_with_metrics_dir(
    endpoint: SocketAddr,
    metrics_dir: Option<&Path>,
) -> Result<(), DaemonError> {
    ensure_loopback(endpoint)?;
    let listener = TcpListener::bind(endpoint).map_err(DaemonError::Bind)?;
    let mut daemon = RendererDaemon::new()?;
    let metrics = metrics_dir.and_then(|dir| match MetricsRecorder::start(dir) {
        Ok(recorder) => Some(recorder),
        Err(error) => {
            eprintln!("warning: session metrics disabled, could not start ({error})");
            None
        }
    });
    for mut stream in listener.incoming().flatten() {
        let _ = handle_connection(&mut daemon, &mut stream, metrics.as_ref());
    }
    Ok(())
}

fn handle_connection(
    daemon: &mut RendererDaemon,
    stream: &mut TcpStream,
    metrics: Option<&MetricsRecorder>,
) -> Result<(), DaemonError> {
    let start = Instant::now();
    let request = read_request(stream);
    // Own the label's strings up front: `request_label` borrows from
    // `request`, and that borrow can't outlive `request` being moved into
    // `dispatch` below.
    let label = request.as_ref().ok().map(|request| {
        let (method, scene_id) = request_label(request);
        (method, scene_id.map(str::to_owned))
    });
    let response = match request {
        Ok(request) => dispatch(daemon, request),
        Err(error) => error_response(error),
    };
    if let Some(recorder) = metrics {
        let (method, scene_id) = label.unwrap_or(("invalid_request", None));
        recorder.record(
            method,
            scene_id.as_deref(),
            start.elapsed(),
            response.error.is_none(),
            response.error.as_ref().map(|error| error.code.as_str()),
        );
    }
    serde_json::to_writer(&mut *stream, &response)
        .map_err(|error| DaemonError::Protocol(error.to_string()))?;
    stream.write_all(b"\n").map_err(DaemonError::Connection)?;
    stream.flush().map_err(DaemonError::Connection)
}

fn read_request(stream: &mut TcpStream) -> Result<DaemonRequest, DaemonError> {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(DaemonError::Connection)?;
    let mut bytes = Vec::with_capacity(1024);
    let read = BufReader::new(stream)
        .take((MAX_REQUEST_BYTES + 1) as u64)
        .read_until(b'\n', &mut bytes)
        .map_err(DaemonError::Connection)?;
    if read > MAX_REQUEST_BYTES {
        return Err(DaemonError::Protocol("request exceeds 1 MiB limit".into()));
    }
    let envelope: DaemonEnvelope =
        serde_json::from_slice(&bytes).map_err(|error| DaemonError::Protocol(error.to_string()))?;
    if envelope.version != DAEMON_PROTOCOL_VERSION {
        return Err(DaemonError::Protocol(
            "unsupported daemon request version".into(),
        ));
    }
    Ok(envelope.request)
}

fn dispatch(daemon: &mut RendererDaemon, request: DaemonRequest) -> DaemonResponse {
    let result = match request {
        DaemonRequest::Health => Ok(DaemonResult::Health),
        DaemonRequest::CreateScene {
            scene_id,
            scene,
            asset_root,
        } => daemon
            .create_scene(scene_id, scene, asset_root)
            .map(|revision| DaemonResult::Revision { revision }),
        DaemonRequest::GetScene { scene_id } => daemon
            .get_scene(&scene_id)
            .map(|snapshot| DaemonResult::Scene { snapshot }),
        DaemonRequest::ReplaceScene {
            scene_id,
            scene,
            expected_revision,
            asset_root,
        } => daemon
            .replace_scene(scene_id, scene, expected_revision, asset_root)
            .map(|revision| DaemonResult::Revision { revision }),
        DaemonRequest::PatchScene { scene_id, patch } => daemon
            .patch_scene(&scene_id, patch)
            .map(|revision| DaemonResult::Revision { revision }),
        DaemonRequest::RenderScene { scene_id, output } => daemon
            .render_scene(&scene_id, &output)
            .map(|image| DaemonResult::Rendered {
                output,
                image: image.into(),
            }),
        DaemonRequest::RenderGifScene { scene_id, output } => daemon
            .render_gif_scene(&scene_id, &output)
            .map(|image| DaemonResult::Rendered {
                output,
                image: image.into(),
            }),
        DaemonRequest::DestroyScene { scene_id } => daemon
            .destroy_scene(&scene_id)
            .map(|()| DaemonResult::Destroyed),
    };
    match result {
        Ok(result) => DaemonResponse {
            version: DAEMON_PROTOCOL_VERSION.into(),
            result: Some(result),
            error: None,
        },
        Err(error) => error_response(error),
    }
}

fn error_response(error: DaemonError) -> DaemonResponse {
    let code = error.code();
    let message = error.to_string();
    DaemonResponse {
        version: DAEMON_PROTOCOL_VERSION.into(),
        result: None,
        error: Some(DaemonProtocolError { code, message }),
    }
}

/// A TCP client for a running [`serve`]d daemon: one request/response round
/// trip per [`DaemonClient::call`], each on its own freshly connected
/// stream (no persistent connection/session to manage).
#[derive(Clone, Debug)]
pub struct DaemonClient {
    endpoint: SocketAddr,
    timeout: Duration,
}

impl DaemonClient {
    pub fn new(endpoint: SocketAddr) -> Result<Self, DaemonError> {
        ensure_loopback(endpoint)?;
        Ok(Self {
            endpoint,
            timeout: CLIENT_IO_TIMEOUT,
        })
    }

    pub fn call(&self, request: DaemonRequest) -> Result<DaemonResult, DaemonError> {
        let mut stream = TcpStream::connect_timeout(&self.endpoint, self.timeout)
            .map_err(DaemonError::Connection)?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(DaemonError::Connection)?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(DaemonError::Connection)?;
        serde_json::to_writer(
            &mut stream,
            &DaemonEnvelope {
                version: DAEMON_PROTOCOL_VERSION.into(),
                request,
            },
        )
        .map_err(|error| DaemonError::Protocol(error.to_string()))?;
        stream.write_all(b"\n").map_err(DaemonError::Connection)?;
        stream.flush().map_err(DaemonError::Connection)?;
        let mut bytes = Vec::with_capacity(1024);
        BufReader::new(stream)
            .take((MAX_REQUEST_BYTES + 1) as u64)
            .read_until(b'\n', &mut bytes)
            .map_err(DaemonError::Connection)?;
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err(DaemonError::Protocol("response exceeds 1 MiB limit".into()));
        }
        let response: DaemonResponse = serde_json::from_slice(&bytes)
            .map_err(|error| DaemonError::Protocol(error.to_string()))?;
        if response.version != DAEMON_PROTOCOL_VERSION {
            return Err(DaemonError::Protocol(
                "unsupported daemon response version".into(),
            ));
        }
        response.result.ok_or_else(|| match response.error {
            Some(error) => DaemonError::Remote {
                code: error.code,
                message: error.message,
            },
            None => DaemonError::Protocol("daemon returned no result".into()),
        })
    }
}

/// The daemon holds unauthenticated, unencrypted scene state and accepts
/// requests with no access control beyond "can reach this socket" -- so
/// both [`serve_with_metrics_dir`] (binding) and [`DaemonClient::new`]
/// (connecting) refuse anything but a loopback address, rather than trust
/// callers to only ever pass one.
fn ensure_loopback(endpoint: SocketAddr) -> Result<(), DaemonError> {
    if endpoint.ip().is_loopback() {
        Ok(())
    } else {
        Err(DaemonError::NonLoopbackEndpoint(endpoint))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use renderer_schema::{
        CanvasV1, NodeKindV1, NodeV1, SCENE_VERSION_V1, SceneValidationError, TimelineV1,
    };
    use std::thread;

    /// Polls a metrics directory until it contains a file whose contents
    /// satisfy `ready`, or gives up after ~1s. Metrics are written by a
    /// background thread, so tests can't assume events have landed the
    /// instant a client call returns.
    fn poll_metrics_file(dir: &Path, ready: impl Fn(&str) -> bool) -> String {
        for _ in 0..50 {
            if let Ok(mut entries) = std::fs::read_dir(dir)
                && let Some(Ok(entry)) = entries.next()
                && let Ok(contents) = std::fs::read_to_string(entry.path())
                && ready(&contents)
            {
                return contents;
            }
            thread::sleep(Duration::from_millis(20));
        }
        String::new()
    }

    fn scene() -> SceneV1 {
        SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 16,
                height: 16,
                background: [0.0; 4],
            },
            nodes: vec![NodeV1 {
                id: "box".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 8.0,
                    height: 8.0,
                    color: [1.0; 4],
                },
            }],
            timeline: None,
            effect: None,
        }
    }

    #[test]
    fn patches_are_atomic_and_revisioned_without_a_gpu() {
        let mut store = SceneStore::default();
        assert_eq!(store.create("scene".into(), scene(), None).unwrap(), 1);
        assert!(matches!(
            store.create("scene".into(), scene(), None),
            Err(DaemonError::SceneAlreadyExists(_))
        ));
        assert_eq!(
            store
                .patch(
                    "scene",
                    ScenePatchV1 {
                        expected_revision: Some(1),
                        operations: vec![PatchOperationV1::ClearTimeline]
                    }
                )
                .unwrap(),
            2
        );
        assert!(matches!(
            store.patch(
                "scene",
                ScenePatchV1 {
                    expected_revision: Some(1),
                    operations: vec![PatchOperationV1::ClearTimeline]
                }
            ),
            Err(DaemonError::RevisionConflict { .. })
        ));
        let before = store.get("scene").unwrap();
        assert!(matches!(
            store.patch(
                "scene",
                ScenePatchV1 {
                    expected_revision: Some(2),
                    operations: vec![
                        PatchOperationV1::RemoveNode { id: "box".into() },
                        PatchOperationV1::RemoveNode {
                            id: "missing".into()
                        }
                    ]
                }
            ),
            Err(DaemonError::NodeNotFound { .. })
        ));
        assert_eq!(store.get("scene").unwrap(), before);

        assert!(matches!(
            store.patch(
                "scene",
                ScenePatchV1 {
                    expected_revision: Some(2),
                    operations: vec![PatchOperationV1::UpsertNode {
                        node: NodeV1 {
                            id: "image".into(),
                            translate: [0.0, 0.0],
                            kind: NodeKindV1::Image {
                                x: 0.0,
                                y: 0.0,
                                width: 1.0,
                                height: 1.0,
                                source: "asset.png".into(),
                            },
                        },
                    }],
                }
            ),
            Err(DaemonError::AssetRootRequired)
        ));
        assert_eq!(store.get("scene").unwrap(), before);
    }

    #[test]
    fn validates_protocol_and_loopback_endpoints() {
        assert!(DaemonClient::new("127.0.0.1:9999".parse().unwrap()).is_ok());
        assert!(matches!(
            DaemonClient::new("192.168.1.1:9999".parse().unwrap()),
            Err(DaemonError::NonLoopbackEndpoint(_))
        ));
        let envelope: DaemonEnvelope =
            serde_json::from_str(r#"{"version":"renderer.daemon.v1","method":"health"}"#).unwrap();
        assert!(matches!(envelope.request, DaemonRequest::Health));
        assert!(
            serde_json::from_str::<DaemonEnvelope>(
                r#"{"version":"renderer.daemon.v1","method":"health","extra":true}"#
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_scenes_that_cannot_fit_in_a_get_response() {
        let mut store = SceneStore::default();
        let mut oversized = scene();
        oversized.nodes.push(NodeV1 {
            id: "large".into(),
            translate: [0.0, 0.0],
            kind: NodeKindV1::Text {
                x: 0.0,
                y: 0.0,
                text: "x".repeat(MAX_REQUEST_BYTES),
                size: 1.0,
                color: [1.0; 4],
            },
        });
        assert!(matches!(
            store.create("scene".into(), oversized, None),
            Err(DaemonError::SceneResponseTooLarge)
        ));
        assert!(matches!(
            store.get("scene"),
            Err(DaemonError::SceneNotFound(_))
        ));
    }

    #[test]
    fn client_times_out_when_a_peer_never_responds() {
        let Ok(listener) = TcpListener::bind("127.0.0.1:0") else {
            return;
        };
        let endpoint = listener.local_addr().unwrap();
        let peer = thread::spawn(move || {
            let _stream = listener.accept().unwrap().0;
            thread::sleep(Duration::from_millis(50));
        });
        let client = DaemonClient {
            endpoint,
            timeout: Duration::from_millis(10),
        };
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Err(DaemonError::Connection(error)) if matches!(error.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock)
        ));
        peer.join().unwrap();
    }

    #[test]
    fn ensure_scene_id_rejects_blank_ids() {
        assert!(matches!(
            ensure_scene_id(""),
            Err(DaemonError::EmptySceneId)
        ));
        assert!(matches!(
            ensure_scene_id("   "),
            Err(DaemonError::EmptySceneId)
        ));
        assert!(ensure_scene_id("ok").is_ok());
    }

    #[test]
    fn error_response_maps_every_error_kind_to_a_stable_code() {
        let cases: Vec<(DaemonError, &str)> = vec![
            (
                DaemonError::RevisionConflict {
                    expected: 1,
                    actual: 2,
                },
                "revision_conflict",
            ),
            (DaemonError::SceneNotFound("s".into()), "not_found"),
            (
                DaemonError::NodeNotFound {
                    index: 0,
                    id: "n".into(),
                },
                "not_found",
            ),
            (
                DaemonError::SceneAlreadyExists("s".into()),
                "already_exists",
            ),
            (
                DaemonError::InvalidScene(SceneValidationError::EmptyNodeId),
                "invalid_scene",
            ),
            (DaemonError::EmptySceneId, "invalid_scene"),
            (DaemonError::SceneResponseTooLarge, "invalid_scene"),
            (DaemonError::Protocol("bad".into()), "invalid_request"),
            (DaemonError::AssetRootRequired, "internal_error"),
        ];
        for (error, expected_code) in cases {
            let message = error.to_string();
            let response = error_response(error);
            assert!(response.result.is_none());
            let protocol_error = response.error.unwrap();
            assert_eq!(protocol_error.code, expected_code);
            assert_eq!(protocol_error.message, message);
        }
    }

    #[test]
    fn scene_count_and_gif_inline_rendering_without_a_named_scene() {
        let Ok(daemon) = RendererDaemon::new() else {
            return;
        };
        assert_eq!(daemon.scene_count(), 0);
        let mut animated = scene();
        animated.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1000,
            keyframes: vec![],
        });
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("inline.gif");
        assert!(daemon.render_gif_inline(&animated, &output).is_ok());
    }

    #[test]
    fn dispatch_handles_every_request_kind_end_to_end() {
        let Ok(mut daemon) = RendererDaemon::new() else {
            return;
        };
        let directory = tempfile::tempdir().unwrap();

        let health = dispatch(&mut daemon, DaemonRequest::Health);
        assert!(matches!(health.result, Some(DaemonResult::Health)));

        let create = dispatch(
            &mut daemon,
            DaemonRequest::CreateScene {
                scene_id: "s".into(),
                scene: scene(),
                asset_root: None,
            },
        );
        assert!(matches!(
            create.result,
            Some(DaemonResult::Revision { revision: 1 })
        ));

        let get = dispatch(
            &mut daemon,
            DaemonRequest::GetScene {
                scene_id: "s".into(),
            },
        );
        assert!(matches!(get.result, Some(DaemonResult::Scene { .. })));

        let missing = dispatch(
            &mut daemon,
            DaemonRequest::GetScene {
                scene_id: "nope".into(),
            },
        );
        assert_eq!(missing.error.unwrap().code, "not_found");

        let conflict = dispatch(
            &mut daemon,
            DaemonRequest::ReplaceScene {
                scene_id: "s".into(),
                scene: scene(),
                expected_revision: Some(99),
                asset_root: None,
            },
        );
        assert_eq!(conflict.error.unwrap().code, "revision_conflict");

        let replace = dispatch(
            &mut daemon,
            DaemonRequest::ReplaceScene {
                scene_id: "s".into(),
                scene: scene(),
                expected_revision: Some(1),
                asset_root: None,
            },
        );
        assert!(matches!(
            replace.result,
            Some(DaemonResult::Revision { revision: 2 })
        ));

        let patch = dispatch(
            &mut daemon,
            DaemonRequest::PatchScene {
                scene_id: "s".into(),
                patch: ScenePatchV1 {
                    expected_revision: Some(2),
                    operations: vec![
                        PatchOperationV1::SetCanvas {
                            canvas: CanvasV1 {
                                width: 16,
                                height: 16,
                                background: [0.0; 4],
                            },
                        },
                        // "box" already exists: this replaces it in place
                        // rather than pushing a new node.
                        PatchOperationV1::UpsertNode {
                            node: NodeV1 {
                                id: "box".into(),
                                translate: [0.0, 0.0],
                                kind: NodeKindV1::Rect {
                                    x: 0.0,
                                    y: 0.0,
                                    width: 4.0,
                                    height: 4.0,
                                    color: [1.0; 4],
                                },
                            },
                        },
                        PatchOperationV1::SetTimeline {
                            timeline: TimelineV1 {
                                fps: 1,
                                duration_ms: 1000,
                                keyframes: vec![],
                            },
                        },
                    ],
                },
            },
        );
        assert!(matches!(
            patch.result,
            Some(DaemonResult::Revision { revision: 3 })
        ));

        let render = dispatch(
            &mut daemon,
            DaemonRequest::RenderScene {
                scene_id: "s".into(),
                output: directory.path().join("s.png"),
            },
        );
        assert!(matches!(render.result, Some(DaemonResult::Rendered { .. })));

        let render_gif = dispatch(
            &mut daemon,
            DaemonRequest::RenderGifScene {
                scene_id: "s".into(),
                output: directory.path().join("s.gif"),
            },
        );
        assert!(matches!(
            render_gif.result,
            Some(DaemonResult::Rendered { .. })
        ));

        let destroy = dispatch(
            &mut daemon,
            DaemonRequest::DestroyScene {
                scene_id: "s".into(),
            },
        );
        assert!(matches!(destroy.result, Some(DaemonResult::Destroyed)));

        let destroy_missing = dispatch(
            &mut daemon,
            DaemonRequest::DestroyScene {
                scene_id: "s".into(),
            },
        );
        assert_eq!(destroy_missing.error.unwrap().code, "not_found");
    }

    #[test]
    fn serve_dispatches_over_tcp_and_enforces_protocol_limits() {
        let Ok(picker) = TcpListener::bind("127.0.0.1:0") else {
            return;
        };
        let endpoint = picker.local_addr().unwrap();
        drop(picker);
        let metrics_dir = tempfile::tempdir().unwrap();
        let metrics_path = metrics_dir.path().to_path_buf();
        thread::spawn(move || {
            let _ = serve_with_metrics_dir(endpoint, Some(&metrics_path));
        });

        let client = DaemonClient::new(endpoint).unwrap();
        let mut ready = false;
        for _ in 0..50 {
            if client.call(DaemonRequest::Health).is_ok() {
                ready = true;
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        if !ready {
            // No GPU adapter available in this environment: `serve` cannot
            // construct a `RendererDaemon`, so there is nothing listening.
            return;
        }

        let mut malformed = TcpStream::connect(endpoint).unwrap();
        malformed.write_all(b"not json\n").unwrap();
        let mut response = String::new();
        malformed.read_to_string(&mut response).unwrap();
        assert!(response.contains("invalid_request"));

        let mut wrong_version = TcpStream::connect(endpoint).unwrap();
        wrong_version
            .write_all(br#"{"version":"nope","method":"health"}"#)
            .unwrap();
        wrong_version.write_all(b"\n").unwrap();
        let mut response = String::new();
        wrong_version.read_to_string(&mut response).unwrap();
        assert!(response.contains("unsupported daemon request version"));

        let mut oversized = TcpStream::connect(endpoint).unwrap();
        oversized
            .write_all(&vec![b'a'; MAX_REQUEST_BYTES + 2])
            .unwrap();
        let mut response = String::new();
        oversized.read_to_string(&mut response).unwrap();
        assert!(response.contains("exceeds 1 MiB limit"));

        assert!(matches!(
            client.call(DaemonRequest::Health),
            Ok(DaemonResult::Health)
        ));

        // Metrics are written by a background thread, so poll briefly rather
        // than assuming the events have landed the instant the client calls
        // above return.
        let logged = poll_metrics_file(metrics_dir.path(), |contents| {
            contents.matches("\"method\":\"health\"").count() >= 2
        });
        assert!(
            logged.contains("\"success\":true"),
            "expected at least one successful health event in the session metrics file"
        );
        assert!(
            logged.contains("\"method\":\"invalid_request\""),
            "expected the malformed-JSON request to be recorded too"
        );
    }

    #[test]
    fn metrics_recorder_appends_json_lines_without_blocking_callers() {
        let directory = tempfile::tempdir().unwrap();
        let recorder = MetricsRecorder::start(directory.path()).unwrap();
        recorder.record(
            "render_scene",
            Some("demo"),
            Duration::from_millis(12),
            true,
            None,
        );
        recorder.record(
            "get_scene",
            Some("missing"),
            Duration::from_micros(500),
            false,
            Some("not_found"),
        );
        drop(recorder);

        let contents = poll_metrics_file(directory.path(), |text| text.lines().count() >= 2);
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["method"], "render_scene");
        assert_eq!(first["scene_id"], "demo");
        assert_eq!(first["success"], true);
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["method"], "get_scene");
        assert_eq!(second["success"], false);
        assert_eq!(second["error_code"], "not_found");
    }

    #[test]
    fn serve_with_metrics_dir_none_disables_metrics_without_failing() {
        let Ok(picker) = TcpListener::bind("127.0.0.1:0") else {
            return;
        };
        let endpoint = picker.local_addr().unwrap();
        drop(picker);
        thread::spawn(move || {
            let _ = serve_with_metrics_dir(endpoint, None);
        });

        let client = DaemonClient::new(endpoint).unwrap();
        let mut ready = false;
        for _ in 0..50 {
            if client.call(DaemonRequest::Health).is_ok() {
                ready = true;
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        if !ready {
            return;
        }
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Ok(DaemonResult::Health)
        ));
    }

    #[test]
    fn request_label_identifies_every_request_kind() {
        assert_eq!(request_label(&DaemonRequest::Health), ("health", None));
        assert_eq!(
            request_label(&DaemonRequest::CreateScene {
                scene_id: "s".into(),
                scene: scene(),
                asset_root: None,
            }),
            ("create_scene", Some("s"))
        );
        assert_eq!(
            request_label(&DaemonRequest::DestroyScene {
                scene_id: "s".into()
            }),
            ("destroy_scene", Some("s"))
        );
    }

    #[test]
    fn client_call_reports_oversized_or_malformed_responses() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let _ = BufReader::new(&stream)
                .take(MAX_REQUEST_BYTES as u64)
                .read_until(b'\n', &mut request);
            let _ = stream.write_all(&vec![b'a'; MAX_REQUEST_BYTES + 2]);
        });
        let client = DaemonClient::new(endpoint).unwrap();
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Err(DaemonError::Protocol(message)) if message.contains("exceeds 1 MiB limit")
        ));
        server.join().unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let _ = BufReader::new(&stream)
                .take(MAX_REQUEST_BYTES as u64)
                .read_until(b'\n', &mut request);
            let _ =
                stream.write_all(br#"{"version":"nope","result":{"kind":"health"},"error":null}"#);
            let _ = stream.write_all(b"\n");
        });
        let client = DaemonClient::new(endpoint).unwrap();
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Err(DaemonError::Protocol(message)) if message.contains("unsupported daemon response version")
        ));
        server.join().unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let _ = BufReader::new(&stream)
                .take(MAX_REQUEST_BYTES as u64)
                .read_until(b'\n', &mut request);
            let _ =
                stream.write_all(br#"{"version":"renderer.daemon.v1","result":null,"error":null}"#);
            let _ = stream.write_all(b"\n");
        });
        let client = DaemonClient::new(endpoint).unwrap();
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Err(DaemonError::Protocol(message)) if message == "daemon returned no result"
        ));
        server.join().unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let _ = BufReader::new(&stream)
                .take(MAX_REQUEST_BYTES as u64)
                .read_until(b'\n', &mut request);
            let _ = stream.write_all(
                br#"{"version":"renderer.daemon.v1","result":null,"error":{"code":"not_found","message":"missing"}}"#,
            );
            let _ = stream.write_all(b"\n");
        });
        let client = DaemonClient::new(endpoint).unwrap();
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Err(DaemonError::Remote { code, message })
                if code == "not_found" && message == "missing"
        ));
        server.join().unwrap();
    }

    #[test]
    fn code_matches_error_response_for_every_local_variant_and_remote_passes_through() {
        let cases: Vec<(DaemonError, &str)> = vec![
            (
                DaemonError::RevisionConflict {
                    expected: 1,
                    actual: 2,
                },
                "revision_conflict",
            ),
            (DaemonError::SceneNotFound("s".into()), "not_found"),
            (
                DaemonError::NodeNotFound {
                    index: 0,
                    id: "n".into(),
                },
                "not_found",
            ),
            (
                DaemonError::SceneAlreadyExists("s".into()),
                "already_exists",
            ),
            (
                DaemonError::InvalidScene(SceneValidationError::EmptyNodeId),
                "invalid_scene",
            ),
            (DaemonError::EmptySceneId, "invalid_scene"),
            (DaemonError::SceneResponseTooLarge, "invalid_scene"),
            (DaemonError::Protocol("bad".into()), "invalid_request"),
            (DaemonError::AssetRootRequired, "internal_error"),
        ];
        for (error, expected_code) in cases {
            assert_eq!(error.code(), expected_code);
        }
        assert_eq!(
            DaemonError::Remote {
                code: "custom_code".into(),
                message: "m".into(),
            }
            .code(),
            "custom_code"
        );
    }

    #[test]
    fn client_call_surfaces_remote_errors_with_their_code_and_message() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let _ = BufReader::new(&stream)
                .take(MAX_REQUEST_BYTES as u64)
                .read_until(b'\n', &mut request);
            let _ = stream.write_all(
                br#"{"version":"renderer.daemon.v1","result":null,"error":{"code":"revision_conflict","message":"scene revision conflict: expected 1, current 2"}}"#,
            );
            let _ = stream.write_all(b"\n");
        });
        let client = DaemonClient::new(endpoint).unwrap();
        let error = client.call(DaemonRequest::Health).unwrap_err();
        assert_eq!(error.code(), "revision_conflict");
        assert!(matches!(
            error,
            DaemonError::Remote { code, message }
                if code == "revision_conflict"
                    && message == "scene revision conflict: expected 1, current 2"
        ));
        server.join().unwrap();
    }
}
