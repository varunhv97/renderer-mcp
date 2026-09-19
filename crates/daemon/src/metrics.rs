use crate::protocol::DaemonRequest;
use serde::Serialize;
use std::io::Write;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
/// `Clone` (a `String` and an `mpsc::Sender`, both cheap) so every
/// per-connection thread spawned by [`serve_with_metrics_dir`] can hold its
/// own handle to the same session's recorder.
#[derive(Clone)]
pub(crate) struct MetricsRecorder {
    session_id: String,
    sender: mpsc::Sender<MetricEvent>,
}

impl MetricsRecorder {
    /// Starts a session, creating `metrics_dir` and a `<session_id>.jsonl`
    /// file inside it. Returns `Err` only if the directory/file cannot be
    /// created; callers should treat that as non-fatal and serve without
    /// metrics rather than refuse to start the daemon.
    pub(crate) fn start(metrics_dir: &Path) -> std::io::Result<Self> {
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

    pub(crate) fn record(
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
pub(crate) fn request_label(request: &DaemonRequest) -> (&'static str, Option<&str>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{poll_metrics_file, scene};
    use std::time::Duration;

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
}
