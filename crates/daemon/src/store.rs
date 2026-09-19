use crate::daemon::SceneSnapshot;
use crate::error::DaemonError;
use crate::protocol::{DAEMON_PROTOCOL_VERSION, DaemonResponse, DaemonResult, MAX_REQUEST_BYTES};
use renderer_schema::{PatchOperationV1, ScenePatchV1, SceneV1};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Named scenes held by one running daemon, keyed by caller-chosen
/// `scene_id`. A `BTreeMap` (not a `HashMap`) mainly for deterministic
/// iteration/debugging; lookups aren't hot enough for the difference to
/// matter.
#[derive(Clone, Debug, Default)]
pub(crate) struct SceneStore {
    pub(crate) scenes: BTreeMap<String, StoredScene>,
}

/// A stored scene's current state: the document itself, a revision counter
/// incremented on every successful `replace`/`patch` (for optimistic
/// concurrency -- see `check_revision`), and the asset root image nodes in
/// it resolve `source` paths against.
#[derive(Clone, Debug)]
pub(crate) struct StoredScene {
    revision: u64,
    pub(crate) scene: SceneV1,
    pub(crate) asset_root: Option<PathBuf>,
}

impl SceneStore {
    pub(crate) fn create(
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

    pub(crate) fn get(&self, scene_id: &str) -> Result<SceneSnapshot, DaemonError> {
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

    pub(crate) fn replace(
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

    pub(crate) fn patch(
        &mut self,
        scene_id: &str,
        patch: ScenePatchV1,
    ) -> Result<u64, DaemonError> {
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

    pub(crate) fn destroy(&mut self, scene_id: &str) -> Result<(), DaemonError> {
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
pub(crate) fn ensure_snapshot_fits_response(
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

pub(crate) fn require_asset_root(
    scene: &SceneV1,
    asset_root: &Option<PathBuf>,
) -> Result<(), DaemonError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scene;
    use renderer_schema::FillV1;
    use renderer_schema::NodeKindV1;
    use renderer_schema::NodeV1;
    use renderer_schema::PatchOperationV1;
    use renderer_schema::ScenePatchV1;

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
                fill: FillV1::Solid([1.0; 4]),
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
}
