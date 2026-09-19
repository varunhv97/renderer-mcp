use crate::error::SceneValidationError;
use crate::fill::Color;
use crate::limits::{MAX_ANIMATION_FRAMES, MAX_ANIMATION_PIXELS};
use crate::validate::{validate_color, validate_finite_pair, validate_unit_interval};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Optional per-scene animation: a fixed frame rate and duration plus a list
/// of keyframes. `fps`/`duration_ms` are validated together against
/// [`MAX_ANIMATION_FRAMES`] and [`MAX_ANIMATION_PIXELS`] so a scene can't
/// request more raster work than the renderer is willing to do in one
/// export; the renderer linearly interpolates each animated property
/// between the keyframes bracketing a given frame's timestamp.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TimelineV1 {
    pub fps: u16,
    pub duration_ms: u32,
    #[serde(default)]
    pub keyframes: Vec<KeyframeV1>,
}

impl TimelineV1 {
    pub(crate) fn validate(
        &self,
        node_ids: &HashSet<&str>,
        canvas_width: u32,
        canvas_height: u32,
    ) -> Result<(), SceneValidationError> {
        if self.fps == 0 || self.fps > 60 || self.duration_ms == 0 || self.duration_ms > 10_000 {
            return Err(SceneValidationError::InvalidTimeline);
        }
        let frame_count = (u64::from(self.duration_ms) * u64::from(self.fps)).div_ceil(1_000);
        if frame_count > MAX_ANIMATION_FRAMES {
            return Err(SceneValidationError::TooManyAnimationFrames {
                actual: frame_count,
                maximum: MAX_ANIMATION_FRAMES,
            });
        }
        let pixel_count = u64::from(canvas_width) * u64::from(canvas_height) * frame_count;
        if pixel_count > MAX_ANIMATION_PIXELS {
            return Err(SceneValidationError::TooManyAnimationPixels {
                actual: pixel_count,
                maximum: MAX_ANIMATION_PIXELS,
            });
        }
        for frame in &self.keyframes {
            if frame.at_ms > self.duration_ms {
                return Err(SceneValidationError::InvalidTimeline);
            }
            if frame.target.trim().is_empty() || !node_ids.contains(frame.target.as_str()) {
                return Err(SceneValidationError::UnknownKeyframeTarget(
                    frame.target.clone(),
                ));
            }
            match frame.property {
                AnimatedPropertyV1::Opacity(value) => validate_unit_interval(value)?,
                AnimatedPropertyV1::Color(color) => validate_color(color)?,
                AnimatedPropertyV1::Translate(value) => validate_finite_pair(value)?,
            }
        }
        Ok(())
    }
}

/// One point in time where a `target` node's animated property takes a
/// specific value; `target` must name a node ID present in the same scene.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KeyframeV1 {
    pub at_ms: u32,
    pub target: String,
    pub property: AnimatedPropertyV1,
}

/// The node property a [`KeyframeV1`] animates, tagged by `kind` in JSON
/// (`{"kind": "opacity", "value": ...}`). `Translate` sets the node's
/// absolute `translate` offset at that keyframe -- it is not an additive
/// delta on top of other keyframes.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum AnimatedPropertyV1 {
    Opacity(f32),
    Color(Color),
    Translate([f32; 2]),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SceneValidationError;
    use crate::limits::MAX_CANVAS_DIMENSION;
    use crate::test_support::*;

    #[test]
    fn rejects_unknown_keyframe_targets_and_excessive_animation_work() {
        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: String::new(),
                property: AnimatedPropertyV1::Opacity(1.0),
            }],
        });
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::UnknownKeyframeTarget(_))
        ));

        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: "missing".into(),
                property: AnimatedPropertyV1::Opacity(1.0),
            }],
        });
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::UnknownKeyframeTarget(_))
        ));

        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 60,
            duration_ms: 10_000,
            keyframes: vec![],
        });
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::TooManyAnimationFrames { .. })
        ));

        let mut value = scene();
        value.canvas.width = MAX_CANVAS_DIMENSION;
        value.canvas.height = MAX_CANVAS_DIMENSION;
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 5_000,
            keyframes: vec![],
        });
        assert!(matches!(
            value.validate(),
            Err(SceneValidationError::TooManyAnimationPixels { .. })
        ));
    }

    #[test]
    fn accepts_a_valid_translate_keyframe() {
        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: "box".into(),
                property: AnimatedPropertyV1::Translate([12.5, -7.0]),
            }],
        });
        assert_eq!(value.validate(), Ok(()));
    }

    #[test]
    fn rejects_a_non_finite_translate_keyframe_value() {
        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: "box".into(),
                property: AnimatedPropertyV1::Translate([f32::NAN, 0.0]),
            }],
        });
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::InvalidTranslate)
        );

        let mut value = scene();
        value.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1,
            keyframes: vec![KeyframeV1 {
                at_ms: 0,
                target: "box".into(),
                property: AnimatedPropertyV1::Translate([0.0, f32::INFINITY]),
            }],
        });
        assert_eq!(
            value.validate(),
            Err(SceneValidationError::InvalidTranslate)
        );
    }
}
