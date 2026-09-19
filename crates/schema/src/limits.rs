/// The only value `SceneV1::version` accepts today. An exact string match,
/// not a semver comparison -- a future incompatible schema change gets its
/// own `renderer.scene.v2` constant and `SceneV2` type rather than trying
/// to version this one in place.
pub const SCENE_VERSION_V1: &str = "renderer.scene.v1";

// The constants below bound resource use so an untrusted scene or patch
// document can't force the renderer/daemon to do unbounded work.
pub const MAX_CANVAS_DIMENSION: u32 = 4_096;
pub const MAX_NODES: usize = 10_000;
pub const MAX_PATH_POINTS: usize = 4_096;
pub const MAX_PATCH_OPERATIONS: usize = 1_000;

/// Maximum number of encoded animation frames. This bounds per-frame setup work.
pub const MAX_ANIMATION_FRAMES: u64 = 300;

/// Maximum aggregate raster work for one animation, measured in output pixels.
pub const MAX_ANIMATION_PIXELS: u64 = 64 * 1024 * 1024;

/// Maximum byte length of a user-supplied post-process effect shader.
pub const MAX_EFFECT_SHADER_BYTES: usize = 64 * 1024;
