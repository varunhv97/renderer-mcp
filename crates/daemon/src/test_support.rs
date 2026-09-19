use renderer_schema::SceneV1;
use renderer_schema::{CanvasV1, FillV1, NodeKindV1, NodeV1, SCENE_VERSION_V1};
use std::path::Path;
use std::thread;
use std::time::Duration;

/// Polls a metrics directory until it contains a file whose contents
/// satisfy `ready`, or gives up after ~1s. Metrics are written by a
/// background thread, so tests can't assume events have landed the
/// instant a client call returns.
pub(crate) fn poll_metrics_file(dir: &Path, ready: impl Fn(&str) -> bool) -> String {
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

pub(crate) fn scene() -> SceneV1 {
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
                corner_radius: 0.0,
                fill: FillV1::Solid([1.0; 4]),
            },
        }],
        timeline: None,
        effect: None,
    }
}
