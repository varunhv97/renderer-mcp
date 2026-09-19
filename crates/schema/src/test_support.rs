use crate::fill::FillV1;
use crate::limits::SCENE_VERSION_V1;
use crate::node::{NodeKindV1, NodeV1};
use crate::scene::{CanvasV1, SceneV1, transparent};

pub(crate) fn scene() -> SceneV1 {
    SceneV1 {
        version: SCENE_VERSION_V1.into(),
        canvas: CanvasV1 {
            width: 64,
            height: 64,
            background: transparent(),
        },
        nodes: vec![NodeV1 {
            id: "box".into(),
            translate: [0.0, 0.0],
            kind: NodeKindV1::Rect {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
                corner_radius: 0.0,
                fill: FillV1::Solid([1.0; 4]),
            },
        }],
        timeline: None,
        effect: None,
    }
}
