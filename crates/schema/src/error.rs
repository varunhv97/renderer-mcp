use crate::limits::*;
use thiserror::Error;

/// Every way a [`SceneV1`] or [`ScenePatchV1`] can fail [`SceneV1::validate`]
/// / [`ScenePatchV1::validate`]. One variant per distinguishable failure
/// (rather than an opaque string) so callers -- notably `renderer_daemon`,
/// which wraps this in `DaemonError::InvalidScene` -- can report a stable
/// error without string-matching a message.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SceneValidationError {
    #[error("unsupported scene version: {0}")]
    UnsupportedVersion(String),
    #[error("canvas dimensions must be greater than zero")]
    InvalidCanvasDimensions,
    #[error("canvas {width}x{height} exceeds {maximum}px per dimension")]
    CanvasTooLarge {
        width: u32,
        height: u32,
        maximum: u32,
    },
    #[error("scene has {actual} nodes; maximum is {maximum}")]
    TooManyNodes { actual: usize, maximum: usize },
    #[error("node IDs must not be empty")]
    EmptyNodeId,
    #[error("duplicate node ID: {0}")]
    DuplicateNodeId(String),
    #[error("{0} must be finite and greater than zero")]
    InvalidPositiveValue(&'static str),
    #[error("colors must contain finite values from 0.0 through 1.0")]
    InvalidColor,
    #[error("gradient angle_degrees must be finite")]
    InvalidGradientAngle,
    #[error(
        "corner_radius must be finite, non-negative, and no larger than half of the smaller of width/height"
    )]
    InvalidCornerRadius,
    #[error("translate values must be finite")]
    InvalidTranslate,
    #[error("paths require at least three points")]
    InvalidPath,
    #[error("path has {actual} points; maximum is {maximum}")]
    TooManyPathPoints { actual: usize, maximum: usize },
    #[error("text nodes must not be empty")]
    EmptyText,
    #[error("patch must contain 1 through {MAX_PATCH_OPERATIONS} operations")]
    InvalidPatch,
    #[error("keyframe target does not identify a scene node: {0}")]
    UnknownKeyframeTarget(String),
    #[error("animation has {actual} frames; maximum is {maximum}")]
    TooManyAnimationFrames { actual: u64, maximum: u64 },
    #[error("animation raster work is {actual} pixels; maximum is {maximum}")]
    TooManyAnimationPixels { actual: u64, maximum: u64 },
    #[error("timeline must use 1-60 FPS, last no more than 10 seconds, and contain valid frames")]
    InvalidTimeline,
    #[error("effect shader must not be empty")]
    EmptyEffectShader,
    #[error("effect shader is {actual} bytes; maximum is {maximum}")]
    EffectShaderTooLarge { actual: usize, maximum: usize },
}
