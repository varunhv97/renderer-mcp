use crate::daemon::RendererDaemon;
use crate::endpoint::ensure_loopback;
use crate::error::DaemonError;
use crate::metrics::{MetricsRecorder, request_label};
use crate::protocol::{
    DAEMON_PROTOCOL_VERSION, DaemonEnvelope, DaemonProtocolError, DaemonRequest, DaemonResponse,
    DaemonResult, MAX_REQUEST_BYTES,
};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

const DEFAULT_METRICS_DIR: &str = ".renderer/metrics";

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
    let daemon = RendererDaemon::new()?;
    let metrics = metrics_dir.and_then(|dir| match MetricsRecorder::start(dir) {
        Ok(recorder) => Some(recorder),
        Err(error) => {
            eprintln!("warning: session metrics disabled, could not start ({error})");
            None
        }
    });
    accept_loop(&listener, &daemon, metrics.as_ref());
    Ok(())
}

/// Starts a daemon on a background thread of the calling process and returns
/// the address it is listening on, so a host program (the MCP server) can
/// offer named scenes without the user running `daemon serve` themselves.
///
/// `endpoint` may use port 0 to let the OS pick a free port; the address
/// actually bound is what's returned. The listener is bound and the GPU
/// renderer created *before* this returns, so a missing GPU adapter or an
/// address already in use is reported here, synchronously, rather than
/// surfacing later as a confusing connection failure. Session metrics are
/// left off: an embedded daemon runs in whatever directory the host was
/// launched from, and shouldn't scatter `.renderer/metrics` files there.
pub fn spawn_embedded(endpoint: SocketAddr) -> Result<SocketAddr, DaemonError> {
    ensure_loopback(endpoint)?;
    let listener = TcpListener::bind(endpoint).map_err(DaemonError::Bind)?;
    let bound = listener.local_addr().map_err(DaemonError::Bind)?;
    let daemon = RendererDaemon::new()?;
    thread::spawn(move || accept_loop(&listener, &daemon, None));
    Ok(bound)
}

/// One thread per connection: `RendererDaemon` is cheap to clone (two
/// `Arc`s), so a slow request (an animated GIF export can take seconds) only
/// blocks the connection that made it, not every other in-flight MCP/CLI
/// call against this daemon.
fn accept_loop(listener: &TcpListener, daemon: &RendererDaemon, metrics: Option<&MetricsRecorder>) {
    for mut stream in listener.incoming().flatten() {
        let daemon = daemon.clone();
        let metrics = metrics.cloned();
        thread::spawn(move || {
            let _ = handle_connection(&daemon, &mut stream, metrics.as_ref());
        });
    }
}

fn handle_connection(
    daemon: &RendererDaemon,
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

pub(crate) fn read_request(stream: &mut TcpStream) -> Result<DaemonRequest, DaemonError> {
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

pub(crate) fn dispatch(daemon: &RendererDaemon, request: DaemonRequest) -> DaemonResponse {
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

pub(crate) fn error_response(error: DaemonError) -> DaemonResponse {
    let code = error.code();
    let message = error.to_string();
    DaemonResponse {
        version: DAEMON_PROTOCOL_VERSION.into(),
        result: None,
        error: Some(DaemonProtocolError { code, message }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::DaemonClient;
    use crate::test_support::{poll_metrics_file, scene};
    use renderer_schema::CanvasV1;
    use renderer_schema::FillV1;
    use renderer_schema::NodeKindV1;
    use renderer_schema::NodeV1;
    use renderer_schema::PatchOperationV1;
    use renderer_schema::ScenePatchV1;
    use renderer_schema::SceneValidationError;
    use renderer_schema::TimelineV1;
    use std::io::Read;
    use std::io::Write;
    use std::net::TcpListener;
    use std::net::TcpStream;
    use std::thread;
    use std::time::Duration;

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
    fn dispatch_handles_every_request_kind_end_to_end() {
        let Ok(daemon) = RendererDaemon::new() else {
            return;
        };
        let directory = tempfile::tempdir().unwrap();

        let health = dispatch(&daemon, DaemonRequest::Health);
        assert!(matches!(health.result, Some(DaemonResult::Health)));

        let create = dispatch(
            &daemon,
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
            &daemon,
            DaemonRequest::GetScene {
                scene_id: "s".into(),
            },
        );
        assert!(matches!(get.result, Some(DaemonResult::Scene { .. })));

        let missing = dispatch(
            &daemon,
            DaemonRequest::GetScene {
                scene_id: "nope".into(),
            },
        );
        assert_eq!(missing.error.unwrap().code, "not_found");

        let conflict = dispatch(
            &daemon,
            DaemonRequest::ReplaceScene {
                scene_id: "s".into(),
                scene: scene(),
                expected_revision: Some(99),
                asset_root: None,
            },
        );
        assert_eq!(conflict.error.unwrap().code, "revision_conflict");

        let replace = dispatch(
            &daemon,
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
            &daemon,
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
                                    corner_radius: 0.0,
                                    fill: FillV1::Solid([1.0; 4]),
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
            &daemon,
            DaemonRequest::RenderScene {
                scene_id: "s".into(),
                output: directory.path().join("s.png"),
            },
        );
        assert!(matches!(render.result, Some(DaemonResult::Rendered { .. })));

        let render_gif = dispatch(
            &daemon,
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
            &daemon,
            DaemonRequest::DestroyScene {
                scene_id: "s".into(),
            },
        );
        assert!(matches!(destroy.result, Some(DaemonResult::Destroyed)));

        let destroy_missing = dispatch(
            &daemon,
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
    fn spawn_embedded_serves_on_an_os_chosen_loopback_port() {
        assert!(matches!(
            spawn_embedded("192.168.1.1:0".parse().unwrap()),
            Err(DaemonError::NonLoopbackEndpoint(_))
        ));
        // No GPU adapter here means `spawn_embedded` fails before serving.
        let Ok(endpoint) = spawn_embedded("127.0.0.1:0".parse().unwrap()) else {
            return;
        };
        assert_ne!(endpoint.port(), 0);
        let client = DaemonClient::new(endpoint).unwrap();
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Ok(DaemonResult::Health)
        ));
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
}
