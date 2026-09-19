use crate::*;

/// A bounded, typed scene mutation. Applying all operations is atomic.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScenePatchV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_revision: Option<u64>,
    pub operations: Vec<PatchOperationV1>,
}

impl ScenePatchV1 {
    pub fn validate(&self) -> Result<(), SceneValidationError> {
        if self.operations.is_empty() || self.operations.len() > MAX_PATCH_OPERATIONS {
            return Err(SceneValidationError::InvalidPatch);
        }
        Ok(())
    }
}

/// One operation within a [`ScenePatchV1`], tagged by `op` in JSON. The
/// daemon applies these against a clone of the stored scene and only
/// commits the result if every operation succeeds and the whole scene
/// re-validates -- see `renderer_daemon::apply_operation`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum PatchOperationV1 {
    SetCanvas { canvas: CanvasV1 },
    UpsertNode { node: NodeV1 },
    RemoveNode { id: String },
    SetTimeline { timeline: TimelineV1 },
    ClearTimeline,
}
