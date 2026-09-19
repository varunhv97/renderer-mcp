use renderer_schema::{Color, FillV1, KeyframeV1, NodeKindV1, SceneV1};

pub(crate) fn scene_at(scene: &SceneV1, at_ms: u32) -> SceneV1 {
    let mut output = scene.clone();
    let Some(timeline) = &scene.timeline else {
        return output;
    };
    for node in &mut output.nodes {
        let color_keyframes: Vec<_> = timeline
            .keyframes
            .iter()
            .filter(|frame| {
                frame.target == node.id
                    && matches!(
                        frame.property,
                        renderer_schema::AnimatedPropertyV1::Color(_)
                    )
            })
            .collect();
        if let (Some(fill), Some(value)) = (
            fill_mut(&mut node.kind),
            interpolate_color(&color_keyframes, at_ms),
        ) {
            fill.set_solid(value);
        }
        let opacity_keyframes: Vec<_> = timeline
            .keyframes
            .iter()
            .filter(|frame| {
                frame.target == node.id
                    && matches!(
                        frame.property,
                        renderer_schema::AnimatedPropertyV1::Opacity(_)
                    )
            })
            .collect();
        if let (Some(fill), Some(opacity)) = (
            fill_mut(&mut node.kind),
            interpolate_opacity(&opacity_keyframes, at_ms),
        ) {
            fill.multiply_alpha(opacity);
        }
        let translate_keyframes: Vec<_> = timeline
            .keyframes
            .iter()
            .filter(|frame| {
                frame.target == node.id
                    && matches!(
                        frame.property,
                        renderer_schema::AnimatedPropertyV1::Translate(_)
                    )
            })
            .collect();
        if let Some(value) = interpolate_translate(&translate_keyframes, at_ms) {
            node.translate = value;
        }
    }
    output
}

pub(crate) fn fill_mut(kind: &mut NodeKindV1) -> Option<&mut FillV1> {
    match kind {
        NodeKindV1::Rect { fill, .. }
        | NodeKindV1::Ellipse { fill, .. }
        | NodeKindV1::Line { fill, .. }
        | NodeKindV1::Path { fill, .. }
        | NodeKindV1::Text { fill, .. } => Some(fill),
        NodeKindV1::Image { .. } => None,
    }
}

#[cfg(test)]
pub(crate) fn fill_of(kind: &NodeKindV1) -> Option<FillV1> {
    match kind {
        NodeKindV1::Rect { fill, .. }
        | NodeKindV1::Ellipse { fill, .. }
        | NodeKindV1::Line { fill, .. }
        | NodeKindV1::Path { fill, .. }
        | NodeKindV1::Text { fill, .. } => Some(fill.clone()),
        NodeKindV1::Image { .. } => None,
    }
}

fn interpolate_color(frames: &[&KeyframeV1], at_ms: u32) -> Option<Color> {
    let values: Vec<_> = frames
        .iter()
        .filter_map(|frame| match frame.property {
            renderer_schema::AnimatedPropertyV1::Color(color) => Some((frame.at_ms, color)),
            _ => None,
        })
        .collect();
    interpolate(&values, at_ms, |left, right, progress| {
        std::array::from_fn(|index| left[index] + (right[index] - left[index]) * progress)
    })
}

fn interpolate_opacity(frames: &[&KeyframeV1], at_ms: u32) -> Option<f32> {
    let values: Vec<_> = frames
        .iter()
        .filter_map(|frame| match frame.property {
            renderer_schema::AnimatedPropertyV1::Opacity(value) => Some((frame.at_ms, value)),
            _ => None,
        })
        .collect();
    interpolate(&values, at_ms, |left, right, progress| {
        left + (right - left) * progress
    })
}

fn interpolate_translate(frames: &[&KeyframeV1], at_ms: u32) -> Option<[f32; 2]> {
    let values: Vec<_> = frames
        .iter()
        .filter_map(|frame| match frame.property {
            renderer_schema::AnimatedPropertyV1::Translate(value) => Some((frame.at_ms, value)),
            _ => None,
        })
        .collect();
    interpolate(&values, at_ms, |left, right, progress| {
        std::array::from_fn(|index| left[index] + (right[index] - left[index]) * progress)
    })
}

pub(crate) fn interpolate<T: Copy>(
    values: &[(u32, T)],
    at_ms: u32,
    between: impl Fn(T, T, f32) -> T,
) -> Option<T> {
    let first = *values.first()?;
    let mut sorted = values.to_vec();
    sorted.sort_by_key(|(time, _)| *time);
    let (first_time, first_value) = sorted[0];
    if at_ms <= first_time {
        return Some(first_value);
    }
    for pair in sorted.windows(2) {
        let (left_time, left) = pair[0];
        let (right_time, right) = pair[1];
        if at_ms <= right_time {
            let progress = (at_ms - left_time) as f32 / (right_time - left_time).max(1) as f32;
            return Some(between(left, right, progress));
        }
    }
    Some(sorted.last().unwrap_or(&first).1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use renderer_schema::FillV1;
    use renderer_schema::KeyframeV1;

    #[test]
    fn interpolates_color_and_opacity_keyframes() {
        let mut scene = test_scene();
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 2,
            duration_ms: 1_000,
            keyframes: vec![
                KeyframeV1 {
                    at_ms: 0,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Color([0.0, 0.0, 0.0, 1.0]),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Color([1.0, 1.0, 1.0, 1.0]),
                },
                KeyframeV1 {
                    at_ms: 0,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(0.0),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(1.0),
                },
            ],
        });
        let at_middle = scene_at(&scene, 500);
        assert_eq!(
            fill_of(&at_middle.nodes[0].kind),
            Some(FillV1::Solid([0.5, 0.5, 0.5, 0.5]))
        );
        assert_eq!(interpolate_color(&[], 0), None);
        assert_eq!(interpolate_opacity(&[], 0), None);
        let opacity = KeyframeV1 {
            at_ms: 0,
            target: "box".into(),
            property: renderer_schema::AnimatedPropertyV1::Opacity(1.0),
        };
        let color = KeyframeV1 {
            at_ms: 0,
            target: "box".into(),
            property: renderer_schema::AnimatedPropertyV1::Color([1.0; 4]),
        };
        assert_eq!(interpolate_color(&[&opacity], 0), None);
        assert_eq!(interpolate_opacity(&[&color], 0), None);
    }

    #[test]
    fn interpolates_translate_keyframes() {
        let mut scene = test_scene();
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 2,
            duration_ms: 1_000,
            keyframes: vec![
                KeyframeV1 {
                    at_ms: 0,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Translate([0.0, 0.0]),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Translate([20.0, -10.0]),
                },
            ],
        });
        assert_eq!(scene_at(&scene, 0).nodes[0].translate, [0.0, 0.0]);
        assert_eq!(scene_at(&scene, 500).nodes[0].translate, [10.0, -5.0]);
        assert_eq!(scene_at(&scene, 1_000).nodes[0].translate, [20.0, -10.0]);
        assert_eq!(interpolate_translate(&[], 0), None);
        let opacity = KeyframeV1 {
            at_ms: 0,
            target: "box".into(),
            property: renderer_schema::AnimatedPropertyV1::Opacity(1.0),
        };
        assert_eq!(interpolate_translate(&[&opacity], 0), None);
    }

    #[test]
    fn multiplies_interpolated_color_alpha_by_opacity() {
        let mut scene = test_scene();
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 2,
            duration_ms: 1_000,
            keyframes: vec![
                KeyframeV1 {
                    at_ms: 0,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Color([1.0, 0.0, 0.0, 0.0]),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Color([1.0, 0.0, 0.0, 1.0]),
                },
                KeyframeV1 {
                    at_ms: 0,
                    target: "box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(0.5),
                },
            ],
        });
        let at_middle = scene_at(&scene, 500);
        assert_eq!(
            fill_of(&at_middle.nodes[0].kind),
            Some(FillV1::Solid([1.0, 0.0, 0.0, 0.25]))
        );
    }
}
