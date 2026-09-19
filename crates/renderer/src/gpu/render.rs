use super::PendingFrame;
use crate::{
    GpuRenderer,
    animation::scene_at,
    assets::AssetCache,
    error::{RenderError, RenderedImage},
    gif::*,
    util::*,
};
use image::RgbaImage;
use renderer_schema::SceneV1;
use std::{fs, path::Path};

impl GpuRenderer {
    pub fn render_png(&self, scene: &SceneV1, output: &Path) -> Result<RenderedImage, RenderError> {
        let root =
            std::env::current_dir().map_err(|error| RenderError::Asset(error.to_string()))?;
        self.render_png_with_asset_root(scene, output, &root)
    }

    pub fn render_png_with_asset_root(
        &self,
        scene: &SceneV1,
        output: &Path,
        asset_root: &Path,
    ) -> Result<RenderedImage, RenderError> {
        scene.validate()?;
        ensure_png_output_path(output)?;
        let (pixels, warnings) = self.render_rgba_with_asset_root(scene, asset_root)?;
        fs::create_dir_all(output.parent().unwrap_or(Path::new(".")))
            .map_err(RenderError::OutputDirectory)?;
        RgbaImage::from_raw(scene.canvas.width, scene.canvas.height, pixels.clone())
            .expect("validated dimensions match readback length")
            .save_with_format(output, image::ImageFormat::Png)?;
        let sha256 = hash_file(output)?;
        Ok(RenderedImage {
            width: scene.canvas.width,
            height: scene.canvas.height,
            sha256,
            frame_count: 1,
            warnings,
        })
    }

    pub fn render_gif(&self, scene: &SceneV1, output: &Path) -> Result<RenderedImage, RenderError> {
        let root =
            std::env::current_dir().map_err(|error| RenderError::Asset(error.to_string()))?;
        self.render_gif_with_asset_root(scene, output, &root)
    }

    pub fn render_gif_with_asset_root(
        &self,
        scene: &SceneV1,
        output: &Path,
        asset_root: &Path,
    ) -> Result<RenderedImage, RenderError> {
        scene.validate()?;
        let timeline = scene.timeline.as_ref().ok_or(RenderError::InvalidScene(
            renderer_schema::SceneValidationError::InvalidTimeline,
        ))?;
        fs::create_dir_all(output.parent().unwrap_or(Path::new(".")))
            .map_err(RenderError::OutputDirectory)?;
        let frame_count =
            (u64::from(timeline.duration_ms) * u64::from(timeline.fps)).div_ceil(1_000) as u32;
        let fps = u32::from(timeline.fps);
        let mut warnings = Vec::new();
        // Created once, outside the frame loop, and shared (by `&mut`) across
        // every frame's composition below: this is what lets a static
        // asset -- one referenced identically by several/all frames, e.g. an
        // unanimated background or logo image, or any glyph in an
        // unanimated text node -- get decoded/rasterized once for the whole
        // GIF export instead of once per frame. See `AssetCache`'s doc
        // comment for the full rationale.
        let mut cache = AssetCache::default();
        // Likewise created once: `scene.canvas`'s dimensions and
        // `scene.effect`'s presence are both scene-level, not animatable
        // (only node properties can carry timeline keyframes -- see
        // `AnimatedPropertyV1`), so every frame renders into the exact same
        // GPU textures/buffer instead of allocating and tearing down a fresh
        // multi-megabyte render target `frame_count` times. See
        // `FrameTargets`'s doc comment.
        let targets = self.create_frame_targets(
            scene.canvas.width,
            scene.canvas.height,
            scene.effect.as_ref(),
        )?;
        // Render every frame first into one flat, contiguous buffer, instead
        // of encoding each one as it's produced: `encode_gif_with_shared_palette`
        // needs every frame's actual pixels in hand before it can build one
        // color palette to share across all of them (see that function's
        // doc comment for why this is the whole point). Flat rather than
        // `Vec<Vec<u8>>` so the palette-training pass below can hand
        // `color_quant`/the exact-palette path one contiguous slice without
        // a second, doubled-memory copy.
        let frame_len = (scene.canvas.width as usize) * (scene.canvas.height as usize) * 4;
        let mut pixels = Vec::with_capacity(frame_len * frame_count as usize);
        // Pipelined: record and submit each frame's GPU work into whichever
        // of `targets.output_buffers` the *previous* frame isn't currently
        // being read back from, then only wait on the previous frame's
        // readback -- which has had this frame's `composition_plan` +
        // vertex/texture buffer construction + command recording time to
        // finish on the GPU in the background. See `record_frame`/
        // `finish_frame`'s doc comments. The very first frame has no
        // previous frame to overlap with; the very last frame's readback is
        // drained after the loop, once there's no next frame left to record
        // while waiting.
        let mut pending: Option<(usize, PendingFrame)> = None;
        for frame_index in 0..frame_count {
            let at_ms = frame_index * 1_000 / fps;
            let animated = scene_at(scene, at_ms);
            animated.validate()?;
            let buffer_index = (frame_index % 2) as usize;
            let new_pending = self.record_frame(
                &animated,
                asset_root,
                &mut cache,
                &targets,
                &targets.output_buffers[buffer_index],
            )?;
            if let Some((prev_buffer_index, prev_pending)) = pending.take() {
                let (frame_pixels, frame_warnings) =
                    self.finish_frame(&targets.output_buffers[prev_buffer_index], prev_pending)?;
                warnings.extend(frame_warnings);
                pixels.extend_from_slice(&frame_pixels);
            }
            pending = Some((buffer_index, new_pending));
        }
        if let Some((buffer_index, last_pending)) = pending {
            let (frame_pixels, frame_warnings) =
                self.finish_frame(&targets.output_buffers[buffer_index], last_pending)?;
            warnings.extend(frame_warnings);
            pixels.extend_from_slice(&frame_pixels);
        }
        encode_gif_with_shared_palette(
            output,
            scene.canvas.width,
            scene.canvas.height,
            &mut pixels,
            frame_count,
            fps,
            GIF_QUANTIZATION_SPEED,
        )?;
        warnings.sort();
        warnings.dedup();
        Ok(RenderedImage {
            width: scene.canvas.width,
            height: scene.canvas.height,
            sha256: hash_file(output)?,
            frame_count,
            warnings,
        })
    }

    pub fn render_rgba(&self, scene: &SceneV1) -> Result<(Vec<u8>, Vec<String>), RenderError> {
        let root =
            std::env::current_dir().map_err(|error| RenderError::Asset(error.to_string()))?;
        self.render_rgba_with_asset_root(scene, &root)
    }

    pub fn render_rgba_with_asset_root(
        &self,
        scene: &SceneV1,
        asset_root: &Path,
    ) -> Result<(Vec<u8>, Vec<String>), RenderError> {
        scene.validate()?;
        self.render_composed_rgba(scene, asset_root)
    }
}
