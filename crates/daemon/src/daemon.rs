use crate::error::DaemonError;
#[cfg(doc)]
use crate::server::{serve, serve_with_metrics_dir};
use crate::store::{SceneStore, require_asset_root};
use renderer_core::{GpuRenderer, RenderedImage};
use renderer_schema::{ScenePatchV1, SceneV1};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

/// One GPU renderer plus the in-memory named-scene store, either driven
/// in-process or wrapped by [`serve`] for TCP access. A single
/// `GpuRenderer` is shared across every scene (named or inline) rather than
/// one per scene, since GPU device/pipeline setup is the expensive part.
///
/// Cheaply `Clone`able (both fields are `Arc`s) so [`serve_with_metrics_dir`]
/// can hand one clone to a thread per connection: `GpuRenderer`'s render
/// methods take `&self` and touch no shared mutable state (every per-frame
/// cache is a local built inside that call), so renders on different
/// connections genuinely run concurrently; `SceneStore` mutations go through
/// the `Mutex` and are held only long enough to read or update the map, never
/// for the duration of a render, so a slow `export_named_gif` on one
/// connection doesn't block a `get_scene`/`patch_scene` on another.
#[derive(Debug, Clone)]
pub struct RendererDaemon {
    renderer: Arc<GpuRenderer>,
    scenes: Arc<Mutex<SceneStore>>,
}

/// A named scene's document plus the `scene_id`/`revision` it was fetched
/// under -- the payload of `get_scene` and every mutating call's response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SceneSnapshot {
    pub scene_id: String,
    pub revision: u64,
    pub scene: SceneV1,
}

/// Metadata about one completed render, returned instead of the raw image
/// bytes: callers read the file at the output path they gave (or, for MCP,
/// this daemon crate isn't the one that base64-encodes it -- see
/// `renderer_mcp::inline_render_response`).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RenderResult {
    pub width: u32,
    pub height: u32,
    pub sha256: String,
    pub frame_count: u32,
    pub warnings: Vec<String>,
}

impl From<RenderedImage> for RenderResult {
    fn from(image: RenderedImage) -> Self {
        Self {
            width: image.width,
            height: image.height,
            sha256: image.sha256,
            frame_count: image.frame_count,
            warnings: image.warnings,
        }
    }
}

impl RendererDaemon {
    pub fn new() -> Result<Self, DaemonError> {
        Ok(Self {
            renderer: Arc::new(GpuRenderer::new()?),
            scenes: Arc::new(Mutex::new(SceneStore::default())),
        })
    }

    /// Locks the scene store, recovering rather than panicking if a prior
    /// panic (e.g. inside a patch operation on some other connection's
    /// thread) poisoned it -- one connection's bug shouldn't wedge every
    /// other connection's access to the store.
    fn lock_scenes(&self) -> MutexGuard<'_, SceneStore> {
        self.scenes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn create_scene(
        &self,
        scene_id: String,
        scene: SceneV1,
        asset_root: Option<PathBuf>,
    ) -> Result<u64, DaemonError> {
        self.lock_scenes().create(scene_id, scene, asset_root)
    }

    pub fn get_scene(&self, scene_id: &str) -> Result<SceneSnapshot, DaemonError> {
        self.lock_scenes().get(scene_id)
    }

    pub fn replace_scene(
        &self,
        scene_id: String,
        scene: SceneV1,
        expected_revision: Option<u64>,
        asset_root: Option<PathBuf>,
    ) -> Result<u64, DaemonError> {
        self.lock_scenes()
            .replace(scene_id, scene, expected_revision, asset_root)
    }

    pub fn patch_scene(&self, scene_id: &str, patch: ScenePatchV1) -> Result<u64, DaemonError> {
        self.lock_scenes().patch(scene_id, patch)
    }

    pub fn destroy_scene(&self, scene_id: &str) -> Result<(), DaemonError> {
        self.lock_scenes().destroy(scene_id)
    }

    /// Looks up the named scene and clones just the (scene, asset_root) pair
    /// needed to render it, releasing the scene-store lock before handing
    /// off to the GPU: a render can take seconds, and nothing else touching
    /// the store should have to wait behind it.
    fn scene_for_render(&self, scene_id: &str) -> Result<(SceneV1, Option<PathBuf>), DaemonError> {
        let stored = self.lock_scenes();
        let stored = stored
            .scenes
            .get(scene_id)
            .ok_or_else(|| DaemonError::SceneNotFound(scene_id.into()))?;
        Ok((stored.scene.clone(), stored.asset_root.clone()))
    }

    pub fn render_scene(
        &self,
        scene_id: &str,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        let (scene, asset_root) = self.scene_for_render(scene_id)?;
        require_asset_root(&scene, &asset_root)?;
        let root = asset_root.as_deref().unwrap_or(Path::new("."));
        Ok(self
            .renderer
            .render_png_with_asset_root(&scene, output, root)?)
    }

    pub fn render_gif_scene(
        &self,
        scene_id: &str,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        let (scene, asset_root) = self.scene_for_render(scene_id)?;
        require_asset_root(&scene, &asset_root)?;
        let root = asset_root.as_deref().unwrap_or(Path::new("."));
        Ok(self
            .renderer
            .render_gif_with_asset_root(&scene, output, root)?)
    }

    pub fn render_inline(
        &self,
        scene: &SceneV1,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self.renderer.render_png(scene, output)?)
    }

    pub fn render_inline_with_asset_root(
        &self,
        scene: &SceneV1,
        output: &Path,
        asset_root: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self
            .renderer
            .render_png_with_asset_root(scene, output, asset_root)?)
    }

    pub fn render_gif_inline(
        &self,
        scene: &SceneV1,
        output: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self.renderer.render_gif(scene, output)?)
    }

    pub fn render_gif_inline_with_asset_root(
        &self,
        scene: &SceneV1,
        output: &Path,
        asset_root: &Path,
    ) -> Result<RenderedImage, DaemonError> {
        Ok(self
            .renderer
            .render_gif_with_asset_root(scene, output, asset_root)?)
    }

    pub fn scene_count(&self) -> usize {
        self.lock_scenes().scenes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scene;
    use renderer_schema::TimelineV1;

    #[test]
    fn scene_count_and_gif_inline_rendering_without_a_named_scene() {
        let Ok(daemon) = RendererDaemon::new() else {
            return;
        };
        assert_eq!(daemon.scene_count(), 0);
        let mut animated = scene();
        animated.timeline = Some(TimelineV1 {
            fps: 1,
            duration_ms: 1000,
            keyframes: vec![],
        });
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("inline.gif");
        assert!(daemon.render_gif_inline(&animated, &output).is_ok());
    }
}
