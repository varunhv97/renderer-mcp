//! In-memory session state around one persistent GPU renderer.

use renderer_core::{GpuRenderer, RenderError, RenderedImage};
use renderer_schema::SceneV1;
use std::{collections::BTreeMap, path::Path};
use thiserror::Error;

#[derive(Debug)]
pub struct RendererDaemon {
    renderer: GpuRenderer,
    scenes: BTreeMap<String, SceneV1>,
}

#[derive(Debug, Error)]
pub enum DaemonError {
    #[error("GPU initialization failed: {0}")]
    Renderer(#[from] RenderError),
    #[error("scene validation failed: {0}")]
    InvalidScene(#[from] renderer_schema::SceneValidationError),
    #[error("scene ID must not be empty")]
    EmptySceneId,
    #[error("named scene does not exist: {0}")]
    SceneNotFound(String),
}

impl RendererDaemon {
    pub fn new() -> Result<Self, DaemonError> {
        Ok(Self {
            renderer: GpuRenderer::new()?,
            scenes: BTreeMap::new(),
        })
    }

    pub fn create_scene(&mut self, scene_id: String, scene: SceneV1) -> Result<(), DaemonError> {
        self.ensure_id(&scene_id)?;
        scene.validate()?;
        self.scenes.insert(scene_id, scene);
        Ok(())
    }

    pub fn replace_scene(&mut self, scene_id: String, scene: SceneV1) -> Result<(), DaemonError> {
        self.ensure_id(&scene_id)?;
        if !self.scenes.contains_key(&scene_id) {
            return Err(DaemonError::SceneNotFound(scene_id));
        }
        scene.validate()?;
        self.scenes.insert(scene_id, scene);
        Ok(())
    }

    pub fn destroy_scene(&mut self, scene_id: &str) -> Result<(), DaemonError> {
        self.scenes
            .remove(scene_id)
            .ok_or_else(|| DaemonError::SceneNotFound(scene_id.into()))?;
        Ok(())
    }

    pub fn render_scene(
        &self,
        scene_id: &str,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        let scene = self
            .scenes
            .get(scene_id)
            .ok_or_else(|| DaemonError::SceneNotFound(scene_id.into()))?;
        Ok(self.renderer.render_png(scene, output)?)
    }

    pub fn render_inline(
        &self,
        scene: &SceneV1,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self.renderer.render_png(scene, output)?)
    }

    pub fn render_gif_inline(
        &self,
        scene: &SceneV1,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self.renderer.render_gif(scene, output)?)
    }

    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    fn ensure_id(&self, scene_id: &str) -> Result<(), DaemonError> {
        if scene_id.trim().is_empty() {
            Err(DaemonError::EmptySceneId)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use renderer_schema::{CanvasV1, NodeKindV1, NodeV1, SCENE_VERSION_V1};

    fn scene() -> SceneV1 {
        SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 16,
                height: 16,
                background: [0.0; 4],
            },
            nodes: vec![NodeV1 {
                id: "box".into(),
                kind: NodeKindV1::Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 8.0,
                    height: 8.0,
                    color: [1.0; 4],
                },
            }],
            timeline: None,
        }
    }

    #[test]
    fn manages_named_scenes_and_renders_them() {
        let _ = RendererDaemon::new().map(|mut daemon| {
            assert!(matches!(
                daemon.create_scene(String::new(), scene()),
                Err(DaemonError::EmptySceneId)
            ));
            assert!(matches!(
                daemon.replace_scene("missing".into(), scene()),
                Err(DaemonError::SceneNotFound(_))
            ));
            daemon.create_scene("scene".into(), scene()).unwrap();
            assert_eq!(daemon.scene_count(), 1);
            daemon.replace_scene("scene".into(), scene()).unwrap();
            let directory = tempfile::tempdir().unwrap();
            assert_eq!(
                daemon
                    .render_scene("scene", &directory.path().join("scene.png"))
                    .unwrap()
                    .frame_count,
                1
            );
            let mut animated = scene();
            animated.timeline = Some(renderer_schema::TimelineV1 {
                fps: 1,
                duration_ms: 1_000,
                keyframes: vec![],
            });
            assert_eq!(
                daemon
                    .render_gif_inline(&animated, &directory.path().join("scene.gif"))
                    .unwrap()
                    .frame_count,
                1
            );
            assert!(matches!(
                daemon.render_scene("missing", directory.path()),
                Err(DaemonError::SceneNotFound(_))
            ));
            daemon.destroy_scene("scene").unwrap();
            assert!(matches!(
                daemon.destroy_scene("scene"),
                Err(DaemonError::SceneNotFound(_))
            ));
        });
    }
}
