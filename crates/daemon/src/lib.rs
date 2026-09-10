//! Persistent local scene state and its loopback-only daemon protocol.

use renderer_core::{GpuRenderer, RenderError, RenderedImage};
use renderer_schema::{PatchOperationV1, ScenePatchV1, SceneV1};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    time::Duration,
};
use thiserror::Error;

pub const DAEMON_PROTOCOL_VERSION: &str = "renderer.daemon.v1";
pub const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const CLIENT_IO_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug)]
pub struct RendererDaemon {
    renderer: GpuRenderer,
    scenes: SceneStore,
}

#[derive(Clone, Debug, Default)]
struct SceneStore {
    scenes: BTreeMap<String, StoredScene>,
}

#[derive(Clone, Debug)]
struct StoredScene {
    revision: u64,
    scene: SceneV1,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SceneSnapshot {
    pub scene_id: String,
    pub revision: u64,
    pub scene: SceneV1,
}

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
}

impl RendererDaemon {
    pub fn new() -> Result<Self, DaemonError> {
        Ok(Self {
            renderer: GpuRenderer::new()?,
            scenes: SceneStore::default(),
        })
    }

    pub fn create_scene(&mut self, scene_id: String, scene: SceneV1) -> Result<u64, DaemonError> {
        self.scenes.create(scene_id, scene)
    }

    pub fn get_scene(&self, scene_id: &str) -> Result<SceneSnapshot, DaemonError> {
        self.scenes.get(scene_id)
    }

    pub fn replace_scene(
        &mut self,
        scene_id: String,
        scene: SceneV1,
        expected_revision: Option<u64>,
    ) -> Result<u64, DaemonError> {
        self.scenes.replace(scene_id, scene, expected_revision)
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
        Ok(self
            .renderer
            .render_png(&self.scenes.get(scene_id)?.scene, output)?)
    }

    pub fn render_gif_scene(
        &self,
        scene_id: &str,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self
            .renderer
            .render_gif(&self.scenes.get(scene_id)?.scene, output)?)
    }

    pub fn render_inline(
        &self,
        scene: &SceneV1,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self.renderer.render_png(scene, output)?)
    }

    pub fn render_gif_inline(
        &self,
        scene: &SceneV1,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self.renderer.render_gif(scene, output)?)
    }

    pub fn scene_count(&self) -> usize {
        self.scenes.scenes.len()
    }
}

impl SceneStore {
    fn create(&mut self, scene_id: String, scene: SceneV1) -> Result<u64, DaemonError> {
        ensure_scene_id(&scene_id)?;
        scene.validate()?;
        if self.scenes.contains_key(&scene_id) {
            return Err(DaemonError::SceneAlreadyExists(scene_id));
        }
        ensure_snapshot_fits_response(&scene_id, 1, &scene)?;
        self.scenes
            .insert(scene_id, StoredScene { revision: 1, scene });
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
    ) -> Result<u64, DaemonError> {
        ensure_scene_id(&scene_id)?;
        scene.validate()?;
        let stored = self
            .scenes
            .get_mut(&scene_id)
            .ok_or_else(|| DaemonError::SceneNotFound(scene_id.clone()))?;
        check_revision(expected_revision, stored.revision)?;
        ensure_snapshot_fits_response(&scene_id, stored.revision + 1, &scene)?;
        stored.revision += 1;
        stored.scene = scene;
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
    },
    GetScene {
        scene_id: String,
    },
    ReplaceScene {
        scene_id: String,
        scene: SceneV1,
        expected_revision: Option<u64>,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonEnvelope {
    pub version: String,
    #[serde(flatten)]
    pub request: DaemonRequest,
}

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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DaemonResponse {
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<DaemonResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<DaemonProtocolError>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DaemonProtocolError {
    pub code: String,
    pub message: String,
}

pub fn serve(endpoint: SocketAddr) -> Result<(), DaemonError> {
    ensure_loopback(endpoint)?;
    let listener = TcpListener::bind(endpoint).map_err(DaemonError::Bind)?;
    let mut daemon = RendererDaemon::new()?;
    for mut stream in listener.incoming().flatten() {
        let _ = handle_connection(&mut daemon, &mut stream);
    }
    Ok(())
}

fn handle_connection(
    daemon: &mut RendererDaemon,
    stream: &mut TcpStream,
) -> Result<(), DaemonError> {
    let response = match read_request(stream) {
        Ok(request) => dispatch(daemon, request),
        Err(error) => error_response(error),
    };
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
        DaemonRequest::CreateScene { scene_id, scene } => daemon
            .create_scene(scene_id, scene)
            .map(|revision| DaemonResult::Revision { revision }),
        DaemonRequest::GetScene { scene_id } => daemon
            .get_scene(&scene_id)
            .map(|snapshot| DaemonResult::Scene { snapshot }),
        DaemonRequest::ReplaceScene {
            scene_id,
            scene,
            expected_revision,
        } => daemon
            .replace_scene(scene_id, scene, expected_revision)
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
    let code = match error {
        DaemonError::RevisionConflict { .. } => "revision_conflict",
        DaemonError::SceneNotFound(_) | DaemonError::NodeNotFound { .. } => "not_found",
        DaemonError::SceneAlreadyExists(_) => "already_exists",
        DaemonError::InvalidScene(_)
        | DaemonError::EmptySceneId
        | DaemonError::SceneResponseTooLarge => "invalid_scene",
        DaemonError::Protocol(_) => "invalid_request",
        _ => "internal_error",
    };
    DaemonResponse {
        version: DAEMON_PROTOCOL_VERSION.into(),
        result: None,
        error: Some(DaemonProtocolError {
            code: code.into(),
            message: error.to_string(),
        }),
    }
}

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
        response.result.ok_or_else(|| {
            DaemonError::Protocol(response.error.map_or_else(
                || "daemon returned no result".into(),
                |error| format!("{}: {}", error.code, error.message),
            ))
        })
    }
}

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
    use renderer_schema::{CanvasV1, NodeKindV1, NodeV1, SCENE_VERSION_V1};
    use std::thread;

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
                kind: NodeKindV1::Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 8.0,
                    height: 8.0,
                    color: [1.0; 4],
                },
            }],
            timeline: None,
        }
    }

    #[test]
    fn patches_are_atomic_and_revisioned_without_a_gpu() {
        let mut store = SceneStore::default();
        assert_eq!(store.create("scene".into(), scene()).unwrap(), 1);
        assert!(matches!(
            store.create("scene".into(), scene()),
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
            kind: NodeKindV1::Text {
                x: 0.0,
                y: 0.0,
                text: "x".repeat(MAX_REQUEST_BYTES),
                size: 1.0,
                color: [1.0; 4],
            },
        });
        assert!(matches!(
            store.create("scene".into(), oversized),
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
}
