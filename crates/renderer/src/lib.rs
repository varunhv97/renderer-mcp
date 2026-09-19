//! Off-screen wgpu renderer for normalized RendererCli scenes.
#![allow(unexpected_cfgs)] // `cargo llvm-cov` supplies `cfg(coverage)`/`cfg(coverage_nightly)`.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use fontdue::Font;
use image::{
    Delay, Frame, RgbaImage,
    codecs::gif::{GifEncoder, Repeat},
};
use renderer_schema::{
    Color, EffectV1, FillV1, GradientV1, MAX_CANVAS_DIMENSION, NodeKindV1, SceneV1,
};
use std::{
    collections::HashMap,
    fs,
    fs::File,
    path::{Path, PathBuf},
    sync::mpsc,
};
use wgpu::util::DeviceExt;

mod animation;
mod error;
mod gif;
mod limits;
mod shaders;
mod svg;
mod util;
mod vertex;

use animation::*;
pub use error::{RenderError, RenderedImage};
use gif::*;
use limits::{
    MAX_ASSET_BYTES, MAX_ASSET_PIXELS, MAX_COMPOSITION_TEXTURE_PIXELS, MAX_GLYPH_SIZE,
    MAX_GPU_TEXTURE_DIMENSION, MAX_IMAGE_RASTER_PIXELS, MAX_TEXT_BYTES, MAX_TEXT_GLYPHS,
    MAX_TEXT_RASTER_PIXELS,
};
use shaders::*;
use svg::*;
use util::*;
use vertex::*;

const ELLIPSE_SEGMENTS: usize = 32;
/// Arc tessellation resolution for one rounded rect corner (a 90-degree
/// sweep), reusing `ELLIPSE_SEGMENTS`' angular resolution: `ELLIPSE_SEGMENTS`
/// segments cover a full 360-degree ellipse, so `ELLIPSE_SEGMENTS / 4`
/// segments cover one 90-degree corner at the same degrees-per-segment
/// density (`add_rect`'s tessellation technique otherwise mirrors
/// `add_ellipse`'s triangle-fan-from-center approach directly).
const RECT_CORNER_SEGMENTS: usize = ELLIPSE_SEGMENTS / 4;
/// MSAA sample count used to anti-alias vector primitives (rect/ellipse/
/// line/path). 4x is the standard, broadly-supported choice for
/// `Rgba8UnormSrgb` render targets on desktop GPUs (Metal/Vulkan/DX12) and is
/// what this renderer's node-composition pass uses; see
/// `GpuRenderer::new_async` for an adapter-capability check confirming this
/// value is supported before it is relied on. The effect pass intentionally
/// stays single-sampled (see `build_effect_pipeline`): it introduces no new
/// geometric edges, so multisampling it would add cost with no benefit.
const MSAA_SAMPLE_COUNT: u32 = 4;
/// Supersampling (SSAA) factor applied on top of `MSAA_SAMPLE_COUNT`x MSAA:
/// the node-composition pass renders into a target `SUPERSAMPLE_FACTOR`x the
/// declared canvas width/height (i.e. `SUPERSAMPLE_FACTOR^2` the pixel
/// count), and `render_composed_rgba_with_cache` downsamples the GPU
/// readback back to the declared size with a Lanczos3 filter before any
/// caller (PNG encode, GIF frame assembly) sees it.
///
/// This exists because MSAA alone still falls well short of this renderer's
/// SVG path (`resvg`/`tiny-skia`, used for SVG `Image` nodes): decoding raw
/// pixel bytes along a rendered curve, SVG output shows dozens of distinct
/// intermediate alpha/color levels while 4x MSAA -- the most this renderer's
/// target hardware supports for `Rgba8UnormSrgb` (confirmed on Apple M1/
/// Metal: `sample_count_supported(8)` is `false`) -- only produces a coarse
/// 4-5 levels per edge, since MSAA only ever resolves 4 sample points per
/// pixel regardless of how the edge cuts through them.
///
/// 2 was picked over 3 after measuring both against MSAA-only (factor 1) on
/// an 8-shape, 2-text-node 320x220 scene (several rects/ellipses, two
/// crossing diagonal lines, a path, on this machine's Apple M1/Metal
/// adapter): counting distinct blended RGBA colors sampled along the scene's
/// curved/diagonal edges, MSAA-only produced 399 distinct edge colors,
/// factor 2 produced 896 (2.2x), and factor 3 produced 887 -- i.e. factor 3
/// bought no further measurable quality beyond factor 2 (within run-to-run
/// noise of the same ~890), because 4x MSAA is still the per-pixel coverage
/// resolution *within* each supersampled texel; supersampling adds more
/// texels to resolve MSAA edges into, not more coverage precision per texel,
/// and factor 2 already saturates the visible benefit of that for this
/// renderer's shapes. Single-PNG-render wall-clock cost measured ~2.8x-4x
/// MSAA-only for factor 2 (some run-to-run system noise on this machine)
/// versus ~4.6x for factor 3 (both cheap in absolute
/// terms here -- single-digit vs. low-double-digit milliseconds); a 12-frame
/// GIF export (speed=10 quantization, see `render_gif_with_asset_root`)
/// measured ~1.57x MSAA-only for factor 2 (73.7ms -> ~115ms) versus ~2.1x
/// for factor 3 (-> ~154ms). Factor 2 is the clear pick: a real, measured
/// quality improvement at a reasonable cost for a local, on-demand tool;
/// factor 3 is strictly worse on cost for no measured quality gain here.
/// Memory cost at factor 2 is real and worth stating plainly: the
/// multisampled texture, resolve texture, and CPU readback buffer are all
/// 4x the pixel count (factor^2) of a declared-size render.
const SUPERSAMPLE_FACTOR: u32 = 2;

#[derive(Debug)]
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    font: Font,
    primitive_pipeline: wgpu::RenderPipeline,
    textured_pipeline: wgpu::RenderPipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Pipeline for the analytic (signed-distance-field + `fwidth`)
    /// anti-aliasing "shadow mode" path -- see the `..._analytic_aa` methods
    /// below and `ANALYTIC_SHADER`'s doc comment. Wholly separate from
    /// `primitive_pipeline`; never used by any pre-existing render entry
    /// point.
    analytic_pipeline: wgpu::RenderPipeline,
    /// Single-sampled twin of `textured_pipeline` (same shader, same vertex
    /// layout), used only by the analytic-AA path's text/image draws -- see
    /// `create_analytic_pipelines`'s doc comment for why a separate pipeline
    /// object is required here even though the shader is identical.
    textured_pipeline_single: wgpu::RenderPipeline,
}

impl GpuRenderer {
    pub fn new() -> Result<Self, RenderError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, RenderError> {
        let instance = wgpu::Instance::default();
        let adapter = if let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            adapter
        } else {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                })
                .await
                .ok_or(RenderError::NoAdapter)?
        };
        // `MSAA_SAMPLE_COUNT`x MSAA on `Rgba8UnormSrgb` (the format used for
        // every render target in `render_composed_rgba_with_cache`) is what
        // anti-aliases vector primitives; confirm the chosen adapter actually
        // supports it rather than silently relying on it. Every adapter this
        // renderer has been exercised against (Metal/Vulkan/DX12 desktop
        // GPUs) supports 4x here, so this is a debug-only sanity check, not a
        // hard runtime requirement -- a future portable/software adapter
        // that lacks it would otherwise fail obscurely deep inside
        // `create_texture`/`create_render_pipeline` instead of here.
        debug_assert!(
            adapter
                .get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb)
                .flags
                .sample_count_supported(MSAA_SAMPLE_COUNT),
            "adapter {:?} does not support {}x MSAA for Rgba8UnormSrgb",
            adapter.get_info().name,
            MSAA_SAMPLE_COUNT,
        );
        // `wgpu::Limits::downlevel_defaults()` caps `max_texture_dimension_2d`
        // and `max_buffer_size` at levels tuned for lowest-common-denominator
        // (WebGL2-class) hardware -- 2048px textures -- which is already
        // below `MAX_CANVAS_DIMENSION` (the largest canvas this renderer's
        // schema allows) even before supersampling, and `SUPERSAMPLE_FACTOR`x
        // supersampling raises the *internal* render target size further
        // still, so a declared canvas well under `MAX_CANVAS_DIMENSION` could
        // already exceed these downlevel limits once supersampled. Request
        // enough headroom for the largest canvas this renderer will ever
        // attempt to supersample-render, capped by what this adapter actually
        // reports supporting -- so a genuinely limited adapter fails here,
        // with a clear `request_device` error, instead of panicking deep
        // inside `create_texture`/`create_buffer` in
        // `render_composed_rgba_with_cache`.
        let adapter_limits = adapter.limits();
        let max_supersampled_canvas_dimension =
            MAX_CANVAS_DIMENSION.saturating_mul(SUPERSAMPLE_FACTOR);
        let max_supersampled_canvas_bytes = u64::from(max_supersampled_canvas_dimension)
            * u64::from(max_supersampled_canvas_dimension)
            * 4;
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_texture_dimension_1d = max_supersampled_canvas_dimension
            .min(adapter_limits.max_texture_dimension_1d)
            .max(required_limits.max_texture_dimension_1d);
        required_limits.max_texture_dimension_2d = max_supersampled_canvas_dimension
            .min(adapter_limits.max_texture_dimension_2d)
            .max(required_limits.max_texture_dimension_2d);
        required_limits.max_buffer_size = max_supersampled_canvas_bytes
            .min(adapter_limits.max_buffer_size)
            .max(required_limits.max_buffer_size);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("renderer-cli device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                },
                None,
            )
            .await?;
        let font = Font::from_bytes(
            include_bytes!("../assets/NotoSans-Regular.ttf") as &[u8],
            fontdue::FontSettings::default(),
        )
        .map_err(|_| RenderError::Font)?;
        let (primitive_pipeline, textured_pipeline, texture_bind_group_layout, sampler) =
            create_pipelines(&device);
        // Additive: builds the two extra pipelines the analytic-AA "shadow
        // mode" path uses (see `create_analytic_pipelines`'s doc comment).
        // Every pipeline/shader/field above this line is untouched by this
        // call -- it only reads `texture_bind_group_layout`.
        let (analytic_pipeline, textured_pipeline_single) =
            create_analytic_pipelines(&device, &texture_bind_group_layout);
        Ok(Self {
            device,
            queue,
            font,
            primitive_pipeline,
            textured_pipeline,
            texture_bind_group_layout,
            sampler,
            analytic_pipeline,
            textured_pipeline_single,
        })
    }

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

    #[cfg_attr(coverage_nightly, coverage(off))]
    fn render_composed_rgba(
        &self,
        scene: &SceneV1,
        asset_root: &Path,
    ) -> Result<(Vec<u8>, Vec<String>), RenderError> {
        // A fresh, single-use `AssetCache` per call: every key this call
        // touches is necessarily a first (and only) use, so behavior here is
        // byte-identical to before `AssetCache` existed -- see `AssetCache`'s
        // doc comment.
        let mut cache = AssetCache::default();
        self.render_composed_rgba_with_cache(scene, asset_root, &mut cache)
    }

    /// Same as [`Self::render_composed_rgba`], but takes the decode/
    /// rasterize cache by `&mut` reference instead of creating one, so a
    /// caller (namely `render_gif_with_asset_root`) can share one
    /// `AssetCache` across many calls -- e.g. once per GIF frame -- so a
    /// given asset is decoded/rasterized at most once across all of them.
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn render_composed_rgba_with_cache(
        &self,
        scene: &SceneV1,
        asset_root: &Path,
        cache: &mut AssetCache,
    ) -> Result<(Vec<u8>, Vec<String>), RenderError> {
        let targets = self.create_frame_targets(
            scene.canvas.width,
            scene.canvas.height,
            scene.effect.as_ref(),
        )?;
        // A one-shot render has no next frame to overlap with, so there's no
        // pipelining benefit here -- just record and immediately finish,
        // always against `output_buffers[0]`.
        let pending = self.record_frame(
            scene,
            asset_root,
            cache,
            &targets,
            &targets.output_buffers[0],
        )?;
        self.finish_frame(&targets.output_buffers[0], pending)
    }

    /// Allocates the GPU render targets one frame -- or a whole GIF export's
    /// worth of frames, see [`FrameTargets`] -- composites into: the
    /// `SUPERSAMPLE_FACTOR`x-scaled composite texture and its MSAA
    /// intermediate, the readback buffer, and, only when the scene declares
    /// a post-process `effect`, that effect's shader pipeline, target
    /// texture, and bind group. Everything here is sized from
    /// `declared_width`/`declared_height`/`effect` alone, never from a
    /// specific frame's node content, so the result is valid to reuse for
    /// every frame of one scene's timeline: canvas dimensions and effect
    /// presence are scene-level, not animatable (only node properties carry
    /// timeline keyframes -- see `AnimatedPropertyV1`).
    fn create_frame_targets(
        &self,
        declared_width: u32,
        declared_height: u32,
        effect: Option<&EffectV1>,
    ) -> Result<FrameTargets, RenderError> {
        let width = declared_width.saturating_mul(SUPERSAMPLE_FACTOR);
        let height = declared_height.saturating_mul(SUPERSAMPLE_FACTOR);
        // When a scene-level effect is present, nodes are composited into
        // this texture as an *intermediate* (sampled, not read back) and a
        // second full-screen pass below writes the final, effect-applied
        // pixels elsewhere. With no effect, this texture is the one and only
        // render target and is read back directly, exactly as before this
        // feature existed: no extra texture or pass is allocated.
        let has_effect = effect.is_some();
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderer-cli target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: if has_effect {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            } else {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
            },
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Multisampled intermediate the node-composition pass draws into; it
        // is resolved into `view` (the single-sampled `texture` above) at
        // the end of that pass, which is what performs the anti-aliasing
        // (see `MSAA_SAMPLE_COUNT`'s doc comment). A multisampled texture
        // can only ever be a resolve source, so its usage is restricted to
        // `RENDER_ATTACHMENT` -- it can't be `COPY_SRC` or
        // `TEXTURE_BINDING` -- and it's never read back or sampled directly.
        let msaa_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderer-cli msaa target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let unpadded_bytes_per_row = width * 4;
        let padded_bytes_per_row =
            align_to(unpadded_bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        // Two, not one: see `FrameTargets::output_buffers`' doc comment --
        // this is what lets a GIF export's frame loop record and submit
        // frame N+1's GPU work while frame N's readback is still in flight,
        // instead of fully blocking on each frame before starting the next.
        let make_output_buffer = || {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("renderer-cli readback"),
                size: u64::from(padded_bytes_per_row) * u64::from(height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let output_buffers = [make_output_buffer(), make_output_buffer()];
        let effect = match effect {
            Some(effect) => {
                let pipeline = self.build_effect_pipeline(&effect.shader)?;
                let effect_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("renderer-cli effect target"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let effect_view =
                    effect_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("renderer-cli effect bind group"),
                    layout: &self.texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });
                Some(EffectTargets {
                    pipeline,
                    texture: effect_texture,
                    view: effect_view,
                    bind_group,
                })
            }
            None => None,
        };
        Ok(FrameTargets {
            texture,
            view,
            msaa_texture,
            msaa_view,
            output_buffers,
            effect,
            width,
            height,
        })
    }

    /// Composites one frame of `scene` into the already-allocated `targets`
    /// (see [`Self::create_frame_targets`]), reusing its GPU textures instead
    /// of allocating fresh ones. The only per-call GPU allocations left here
    /// are the vertex buffers and any image/glyph textures `plan.textures`
    /// calls for, both of which genuinely depend on this frame's node
    /// content.
    ///
    /// Submits this frame's GPU work and kicks off an async readback into
    /// `output_buffer` (one of `targets.output_buffers` -- the caller picks
    /// which, so a pipelined loop can alternate) via `map_async`, then
    /// returns immediately without waiting for either to finish -- see
    /// [`Self::finish_frame`], which does that waiting, and
    /// [`Self::render_gif_with_asset_root`]'s frame loop for why splitting
    /// "submit" from "wait" this way is worth doing: it lets the CPU spend
    /// the time an already-submitted frame's GPU work is still running on
    /// building the *next* frame (`composition_plan`, vertex/texture
    /// buffers, command recording) instead of sitting idle.
    fn record_frame(
        &self,
        scene: &SceneV1,
        asset_root: &Path,
        cache: &mut AssetCache,
        targets: &FrameTargets,
        output_buffer: &wgpu::Buffer,
    ) -> Result<PendingFrame, RenderError> {
        let plan = composition_plan(scene, asset_root, &self.font, cache)?;
        let declared_width = scene.canvas.width;
        let declared_height = scene.canvas.height;
        // `targets` was sized `SUPERSAMPLE_FACTOR`x the declared canvas size
        // by `create_frame_targets` (see that constant's doc comment);
        // `vertex()` and `add_textured_rect()` already normalize every
        // position as a fraction of `scene.canvas.width`/`height` rather
        // than any absolute pixel count, so drawing into a larger
        // same-aspect-ratio target needs no change to vertex generation.
        // `pixels`, built from the GPU readback below, is downsampled back
        // to `declared_width`x`declared_height` with a Lanczos3 filter
        // before this function returns, so every caller (PNG encode, GIF
        // frame assembly) keeps receiving pixels already at the scene's
        // declared size, unaware supersampling happened.
        let width = targets.width;
        let height = targets.height;
        let primitive_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("renderer-cli primitive vertices"),
                contents: bytemuck::cast_slice(&plan.primitive_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let textured_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("renderer-cli textured vertices"),
                contents: bytemuck::cast_slice(&plan.textured_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let mut texture_resources = Vec::with_capacity(plan.textures.len());
        for data in &plan.textures {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&data.label),
                size: wgpu::Extent3d {
                    width: data.width,
                    height: data.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data.pixels,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(data.width * 4),
                    rows_per_image: Some(data.height),
                },
                wgpu::Extent3d {
                    width: data.width,
                    height: data.height,
                    depth_or_array_layers: 1,
                },
            );
            let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&data.label),
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            texture_resources.push((texture, texture_view, bind_group));
        }
        let unpadded_bytes_per_row = width * 4;
        let padded_bytes_per_row =
            align_to(unpadded_bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("renderer-cli commands"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("renderer-cli pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.msaa_view,
                    resolve_target: Some(&targets.view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(to_wgpu_color(scene.canvas.background)),
                        // The multisampled contents themselves are never
                        // read -- only the resolve into `targets.view`
                        // matters -- so they don't need to be stored.
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            for command in &plan.commands {
                match command {
                    DrawCommand::Primitive(vertices) => {
                        pass.set_pipeline(&self.primitive_pipeline);
                        pass.set_vertex_buffer(0, primitive_buffer.slice(..));
                        pass.draw(vertices.clone(), 0..1);
                    }
                    DrawCommand::Textured {
                        texture_index,
                        vertices,
                    } => {
                        pass.set_pipeline(&self.textured_pipeline);
                        pass.set_vertex_buffer(0, textured_buffer.slice(..));
                        pass.set_bind_group(0, &texture_resources[*texture_index].2, &[]);
                        pass.draw(vertices.clone(), 0..1);
                    }
                }
            }
        }
        // Second, optional full-screen pass: run the scene's post-process
        // effect, sampling the just-composited scene texture and writing
        // the transformed pixels to a separate texture that gets read back.
        // This keeps the no-effect path's pass count unchanged (see
        // `targets.effect` above).
        let final_texture = if let Some(effect) = &targets.effect {
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("renderer-cli effect pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &effect.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&effect.pipeline);
                pass.set_bind_group(0, &effect.bind_group, &[]);
                // Full-screen triangle: the vertex shader derives clip-space
                // position and UV from `vertex_index` alone, so no vertex
                // buffer is bound here.
                pass.draw(0..3, 0..1);
            }
            &effect.texture
        } else {
            &targets.texture
        };
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: final_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = std::sync::mpsc::channel();
        output_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        Ok(PendingFrame {
            receiver,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
            width,
            height,
            declared_width,
            declared_height,
        })
    }

    /// Waits for a frame `record_frame` already submitted (against the same
    /// `output_buffer` passed to that call) to finish rendering and reading
    /// back, then downsamples it to the scene's declared size. Split from
    /// `record_frame` so a pipelined caller can submit the *next* frame
    /// before waiting on this one -- see that method's doc comment.
    fn finish_frame(
        &self,
        output_buffer: &wgpu::Buffer,
        pending: PendingFrame,
    ) -> Result<(Vec<u8>, Vec<String>), RenderError> {
        let PendingFrame {
            receiver,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
            width,
            height,
            declared_width,
            declared_height,
        } = pending;
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| RenderError::Readback)?
            .map_err(|_| RenderError::Readback)?;
        let slice = output_buffer.slice(..);
        let mapped = slice.get_mapped_range();
        let mut pixels = vec![0; (unpadded_bytes_per_row * height) as usize];
        for (row, target) in pixels
            .chunks_exact_mut(unpadded_bytes_per_row as usize)
            .enumerate()
        {
            let start = row * padded_bytes_per_row as usize;
            target.copy_from_slice(&mapped[start..start + unpadded_bytes_per_row as usize]);
        }
        drop(mapped);
        output_buffer.unmap();
        // `pixels` is still at the oversized `width`x`height` (the
        // `SUPERSAMPLE_FACTOR`x-scaled render target); downsample it to the
        // scene's declared size before any caller sees it, so the existing
        // readback/PNG-encode/GIF-frame-assembly logic above and in every
        // caller stays completely unaware supersampling happened -- it just
        // receives pixels already at the declared size, exactly as before
        // `SUPERSAMPLE_FACTOR` existed.
        if width == declared_width && height == declared_height {
            return Ok((pixels, Vec::new()));
        }
        let oversized = RgbaImage::from_raw(width, height, pixels)
            .expect("readback buffer matches the oversized render target dimensions");
        // Triangle (linear/tent, support radius 1), not Lanczos3 (windowed
        // sinc, support radius 3): `SUPERSAMPLE_FACTOR` always downsamples by
        // exactly 2x, and for an exact integer ratio like that, a triangle
        // filter is close to averaging each 2x2 source block -- the
        // textbook-correct downsample for supersampling antialiasing, not an
        // approximation of one. It's also far cheaper: roughly 9x fewer
        // sample taps per output pixel in 2D than Lanczos3's wider kernel,
        // which measured as the dominant remaining per-frame cost in GIF
        // export profiling (see `architecture.md`). Confirmed side-by-side
        // against Lanczos3 output before switching: the difference is only
        // visible at extreme zoom, mostly as marginally softer text edges,
        // and antialiasing quality itself is preserved (still measurably
        // better than MSAA alone -- see
        // `supersamples_diagonal_primitive_edges_beyond_msaa_alone_on_an_available_gpu`).
        // Golden-image references were regenerated for this change; see
        // `examples/generate_golden.rs`/`generate_golden_svg.rs`.
        let resized = image::imageops::resize(
            &oversized,
            declared_width,
            declared_height,
            image::imageops::FilterType::Triangle,
        );
        Ok((resized.into_raw(), Vec::new()))
    }

    /// Builds the render pipeline for a scene's full-canvas post-process
    /// effect by wrapping the author-supplied WGSL `effect` function in a
    /// fixed template (see `wrap_effect_shader`): a full-screen-triangle
    /// vertex stage and a fragment stage that samples the composited scene
    /// texture and calls the user's function. This is the entire security
    /// boundary between untrusted shader text and the GPU: authors never
    /// supply bindings, vertex data, or a full pipeline.
    ///
    /// `wgpu::Device::create_shader_module` and `create_render_pipeline` do
    /// not return `Result` — by default, invalid WGSL is reported to the
    /// device's uncaptured-error handler, which panics. Since shader text
    /// here comes from untrusted scene JSON that a client can submit to a
    /// long-running daemon, both calls are wrapped in an error scope so a
    /// malformed effect shader becomes a normal `Err` instead of a process
    /// crash.
    ///
    /// This validates that the shader *compiles*; it cannot and does not
    /// bound how expensive a well-formed shader is to *run* (e.g. an
    /// intentionally expensive loop in `effect`). Slow-but-valid shaders are
    /// an accepted residual risk for V1, consistent with how the rest of
    /// this renderer treats resource limits as safe defaults rather than
    /// exhaustive guarantees.
    fn build_effect_pipeline(
        &self,
        user_shader: &str,
    ) -> Result<wgpu::RenderPipeline, RenderError> {
        let wrapped = wrap_effect_shader(user_shader);

        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("renderer-cli effect shader"),
                source: wgpu::ShaderSource::Wgsl(wrapped.into()),
            });
        if let Some(error) = pollster::block_on(self.device.pop_error_scope()) {
            return Err(RenderError::InvalidEffectShader(error.to_string()));
        }

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("renderer-cli effect layout"),
                bind_group_layouts: &[&self.texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("renderer-cli effect pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: EFFECT_VERTEX_ENTRY_POINT,
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: EFFECT_FRAGMENT_ENTRY_POINT,
                    targets: &[Some(effect_color_target())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });
        if let Some(error) = pollster::block_on(self.device.pop_error_scope()) {
            return Err(RenderError::InvalidEffectShader(error.to_string()));
        }
        Ok(pipeline)
    }
}

/// The GPU render targets one call to [`GpuRenderer::render_frame_with_targets`]
/// composites a single frame into: everything sized only by the scene's
/// declared canvas dimensions and whether it has a post-process `effect`
/// (both scene-level, not animatable), never by a specific frame's node
/// content. Built once by [`GpuRenderer::create_frame_targets`] and reused
/// across every frame of a GIF export -- rather than allocated and torn down
/// `frame_count` times -- since the composite/MSAA textures and the readback
/// buffer are the largest, most expensive-to-allocate resources in the whole
/// render path.
struct FrameTargets {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    // Never read again after `msaa_view` is created from it below, but must
    // stay alive for as long as `msaa_view` does -- kept as a named field
    // (rather than let it drop at the end of `create_frame_targets`) purely
    // for that ownership, not to be used directly.
    #[allow(dead_code)]
    msaa_texture: wgpu::Texture,
    msaa_view: wgpu::TextureView,
    /// Two readback buffers, not one, ping-ponged across frames by
    /// `GpuRenderer::record_frame`/`finish_frame`: a GIF export can submit
    /// frame N+1's GPU work (into the buffer frame N *isn't* using) while
    /// frame N's async `map_async` readback is still pending, instead of
    /// fully blocking the CPU on each frame before starting the next one's.
    /// The single-shot PNG path always uses index 0 and never pipelines --
    /// there's only one frame, nothing to overlap.
    output_buffers: [wgpu::Buffer; 2],
    effect: Option<EffectTargets>,
    width: u32,
    height: u32,
}

/// The extra GPU state a scene's post-process `effect` needs, held on
/// [`FrameTargets`] only when one is present: the compiled shader pipeline
/// (shader compilation is one of the more expensive one-time GPU driver
/// calls, so this alone is worth hoisting out of a GIF's per-frame loop) plus
/// its own target texture/view and the bind group sampling `FrameTargets`'s
/// main composite `view`.
struct EffectTargets {
    pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

/// One frame's GPU work, submitted by [`GpuRenderer::record_frame`] but not
/// yet waited on: the readback its `output_buffer` argument is mid-`map_async`
/// for, plus everything [`GpuRenderer::finish_frame`] needs to turn that
/// readback into declared-size pixels once it's ready (sizing, since the
/// downsample step needs both the oversized and declared dimensions).
struct PendingFrame {
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    unpadded_bytes_per_row: u32,
    padded_bytes_per_row: u32,
    width: u32,
    height: u32,
    declared_width: u32,
    declared_height: u32,
}

/// Analytic (signed-distance-field + `fwidth`) anti-aliasing "shadow mode"
/// entry points. Every method here is net-new and additive: none of them are
/// called by, or share any mutable state with, the pre-existing
/// `render_png`/`render_gif`/`render_rgba` family (or their
/// `_with_asset_root` variants) above, which remain completely untouched and
/// keep producing byte-identical output via the original MSAA+supersampling
/// path (`primitive_pipeline`/`textured_pipeline`, `MSAA_SAMPLE_COUNT`,
/// `SUPERSAMPLE_FACTOR`).
///
/// This path renders vector primitives (rect/ellipse/line) with exact
/// analytic per-fragment distance fields instead of stochastic/regular-grid
/// coverage sampling: no multisampling, no supersampling, a single sample
/// per fragment. See `ANALYTIC_SHADER` for the shader-side technique and
/// `composition_plan_analytic` for how each shape type is handled,
/// including `Path`'s documented fallback to the existing MSAA pipeline.
impl GpuRenderer {
    /// Analytic-AA counterpart to [`Self::render_png`].
    pub fn render_png_analytic_aa(
        &self,
        scene: &SceneV1,
        output: &Path,
    ) -> Result<RenderedImage, RenderError> {
        let root =
            std::env::current_dir().map_err(|error| RenderError::Asset(error.to_string()))?;
        self.render_png_with_asset_root_analytic_aa(scene, output, &root)
    }

    /// Analytic-AA counterpart to [`Self::render_png_with_asset_root`].
    pub fn render_png_with_asset_root_analytic_aa(
        &self,
        scene: &SceneV1,
        output: &Path,
        asset_root: &Path,
    ) -> Result<RenderedImage, RenderError> {
        scene.validate()?;
        ensure_png_output_path(output)?;
        let (pixels, warnings) = self.render_rgba_with_asset_root_analytic_aa(scene, asset_root)?;
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

    /// Analytic-AA counterpart to [`Self::render_gif`].
    pub fn render_gif_analytic_aa(
        &self,
        scene: &SceneV1,
        output: &Path,
    ) -> Result<RenderedImage, RenderError> {
        let root =
            std::env::current_dir().map_err(|error| RenderError::Asset(error.to_string()))?;
        self.render_gif_with_asset_root_analytic_aa(scene, output, &root)
    }

    /// Analytic-AA counterpart to [`Self::render_gif_with_asset_root`].
    /// Structurally identical to that method (same frame count/timing math,
    /// same `AssetCache` sharing across frames, same GIF encoder settings)
    /// except each frame is composited with
    /// `render_composed_rgba_analytic_with_cache` instead of
    /// `render_composed_rgba_with_cache`.
    pub fn render_gif_with_asset_root_analytic_aa(
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
        let mut cache = AssetCache::default();
        {
            let mut encoder = GifEncoder::new_with_speed(
                File::create(output).map_err(RenderError::OutputDirectory)?,
                10,
            );
            encoder
                .set_repeat(Repeat::Infinite)
                .map_err(RenderError::Gif)?;
            for frame_index in 0..frame_count {
                let at_ms = frame_index * 1_000 / fps;
                let animated = scene_at(scene, at_ms);
                animated.validate()?;
                let (pixels, frame_warnings) = self
                    .render_composed_rgba_analytic_with_cache(&animated, asset_root, &mut cache)?;
                warnings.extend(frame_warnings);
                let image = RgbaImage::from_raw(scene.canvas.width, scene.canvas.height, pixels)
                    .expect("validated dimensions match readback length");
                encoder
                    .encode_frame(Frame::from_parts(
                        image,
                        0,
                        0,
                        Delay::from_numer_denom_ms(1_000, fps),
                    ))
                    .map_err(RenderError::Gif)?;
            }
        }
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

    /// Analytic-AA counterpart to [`Self::render_rgba`].
    pub fn render_rgba_analytic_aa(
        &self,
        scene: &SceneV1,
    ) -> Result<(Vec<u8>, Vec<String>), RenderError> {
        let root =
            std::env::current_dir().map_err(|error| RenderError::Asset(error.to_string()))?;
        self.render_rgba_with_asset_root_analytic_aa(scene, &root)
    }

    /// Analytic-AA counterpart to [`Self::render_rgba_with_asset_root`]: same
    /// contract (validates the scene, returns declared-size straight-alpha
    /// RGBA8 bytes plus warnings), but composites with the SDF/`fwidth`
    /// pipeline instead of MSAA+supersampling.
    pub fn render_rgba_with_asset_root_analytic_aa(
        &self,
        scene: &SceneV1,
        asset_root: &Path,
    ) -> Result<(Vec<u8>, Vec<String>), RenderError> {
        scene.validate()?;
        let mut cache = AssetCache::default();
        self.render_composed_rgba_analytic_with_cache(scene, asset_root, &mut cache)
    }

    /// Analytic-AA counterpart to
    /// `GpuRenderer::render_composed_rgba_with_cache`. Renders directly at
    /// the scene's declared canvas size -- no `SUPERSAMPLE_FACTOR` scaling,
    /// no downsample step -- because the SDF/`fwidth` technique produces a
    /// resolution-correct ~1-pixel-wide analytic edge from a single sample,
    /// with no oversized intermediate texture needed.
    ///
    /// Node z-order (draw order) must match document order exactly, same as
    /// the original path, but different node kinds here need different GPU
    /// pipelines with different render-target sample counts (the analytic
    /// rect/ellipse/line pipeline and the single-sampled text/image pipeline
    /// both draw into a single-sampled attachment; the `Path` fallback draws
    /// into a multisampled attachment that resolves into the same target --
    /// see `composition_plan_analytic`'s `AnalyticDrawCommand` grouping).
    /// Since a render pass's attachment sample count is fixed for its
    /// duration, this opens one render pass per contiguous same-technique
    /// run of nodes (all consecutive Analytic commands merged into one
    /// pass -- see `composition_plan_analytic` -- likewise for consecutive
    /// Textured or Path commands), each loading the prior pass's contents
    /// (`LoadOp::Load`) except the very first, which clears the canvas
    /// background. This costs a handful of extra render-pass begin/end calls
    /// versus a single monolithic pass -- see the module's performance
    /// comparison for measured real-world cost -- in exchange for exact
    /// z-order correctness across heterogeneous pipelines.
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn render_composed_rgba_analytic_with_cache(
        &self,
        scene: &SceneV1,
        asset_root: &Path,
        cache: &mut AssetCache,
    ) -> Result<(Vec<u8>, Vec<String>), RenderError> {
        let plan = composition_plan_analytic(scene, asset_root, &self.font, cache)?;
        let width = scene.canvas.width;
        let height = scene.canvas.height;
        let has_effect = scene.effect.is_some();
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("renderer-cli analytic target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: if has_effect {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            } else {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
            },
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let analytic_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("renderer-cli analytic vertices"),
                contents: bytemuck::cast_slice(&plan.analytic_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let path_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("renderer-cli analytic path (msaa fallback) vertices"),
                contents: bytemuck::cast_slice(&plan.path_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let textured_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("renderer-cli analytic textured vertices"),
                contents: bytemuck::cast_slice(&plan.textured_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let mut texture_resources = Vec::with_capacity(plan.textures.len());
        for data in &plan.textures {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&data.label),
                size: wgpu::Extent3d {
                    width: data.width,
                    height: data.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data.pixels,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(data.width * 4),
                    rows_per_image: Some(data.height),
                },
                wgpu::Extent3d {
                    width: data.width,
                    height: data.height,
                    depth_or_array_layers: 1,
                },
            );
            let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&data.label),
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            texture_resources.push((texture, texture_view, bind_group));
        }

        let unpadded_bytes_per_row = width * 4;
        let padded_bytes_per_row =
            align_to(unpadded_bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("renderer-cli analytic readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("renderer-cli analytic commands"),
            });

        // Only allocated when the plan actually contains a `Path` command --
        // most scenes have none -- since it's otherwise wasted memory/setup
        // for a render target this path never touches.
        let needs_path_pass = plan
            .commands
            .iter()
            .any(|command| matches!(command, AnalyticDrawCommand::Path(_)));
        let msaa_texture = needs_path_pass.then(|| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("renderer-cli analytic path msaa target"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: MSAA_SAMPLE_COUNT,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        });
        let msaa_view = msaa_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

        // Clear `view` to the background color exactly once, unconditionally,
        // before any node draws -- every subsequent pass below uses
        // `LoadOp::Load` unconditionally. (This subsumes the old "empty
        // scene" special case too: with zero commands, this is the only
        // pass that runs, producing exactly the cleared-background canvas.)
        //
        // Why this replaced a per-command `cleared`-flag "first pass clears,
        // rest load" scheme: that scheme is correct for `AnalyticDrawCommand
        // ::Analytic`/`::Textured` (both render directly into the
        // single-sampled `view`, where `LoadOp::Load` genuinely means "keep
        // what's already there"), but it silently corrupted output whenever
        // a `Path` command *wasn't* first. `Path` renders into a
        // multisampled `msaa_view` with `resolve_target: Some(&view)` --
        // and `LoadOp::Load` on a resolve-source attachment loads that
        // multisampled texture's *own* prior contents, not the resolve
        // target's. Since `msaa_texture` is freshly created on every call,
        // "loading" it reads back empty/undefined data regardless of what
        // `view` already held, and the end-of-pass resolve then overwrites
        // every pixel of `view` (not just the ones the path itself covers)
        // with that empty data -- silently erasing every node drawn before
        // it. (Caught by rendering a rect followed by a path/text scene and
        // finding the rect's pixels had gone from opaque red to fully
        // transparent after the path pass ran.)
        //
        // The fix: `Path` no longer resolves directly onto `view` at all.
        // It resolves into its own fresh, self-contained `path_overlay`
        // texture (always cleared to transparent -- correct, since that
        // texture has no "prior content" to preserve, it's brand new every
        // time), then a second, ordinary single-sampled pass alpha-blends
        // that overlay onto `view` via the existing textured-quad machinery
        // (`textured_pipeline_single`) with `LoadOp::Load` -- which is
        // correct here because this second pass has no `resolve_target` at
        // all, so `Load` unambiguously means "keep what's in `view`".
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("renderer-cli analytic clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(to_wgpu_color(scene.canvas.background)),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            drop(pass);
        }
        for command in &plan.commands {
            match command {
                AnalyticDrawCommand::Path(range) => {
                    let msaa_view = msaa_view
                        .as_ref()
                        .expect("msaa target is built whenever a Path command exists");
                    let path_overlay_texture =
                        self.device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("renderer-cli analytic path overlay"),
                            size: wgpu::Extent3d {
                                width,
                                height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                | wgpu::TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        });
                    let path_overlay_view =
                        path_overlay_texture.create_view(&wgpu::TextureViewDescriptor::default());
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("renderer-cli analytic path (msaa fallback) pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: msaa_view,
                                resolve_target: Some(&path_overlay_view),
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Discard,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        pass.set_pipeline(&self.primitive_pipeline);
                        pass.set_vertex_buffer(0, path_buffer.slice(..));
                        pass.draw(range.clone(), 0..1);
                    }
                    let mut overlay_vertices = Vec::with_capacity(6);
                    add_textured_rect(
                        &mut overlay_vertices,
                        0.0,
                        0.0,
                        width as f32,
                        height as f32,
                        [1.0; 4],
                        scene,
                    );
                    let overlay_buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("renderer-cli analytic path overlay blit vertices"),
                                contents: bytemuck::cast_slice(&overlay_vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                    let overlay_bind_group =
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("renderer-cli analytic path overlay bind group"),
                            layout: &self.texture_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &path_overlay_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("renderer-cli analytic path overlay blit pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(&self.textured_pipeline_single);
                    pass.set_vertex_buffer(0, overlay_buffer.slice(..));
                    pass.set_bind_group(0, &overlay_bind_group, &[]);
                    pass.draw(0..overlay_vertices.len() as u32, 0..1);
                }
                AnalyticDrawCommand::Analytic(range) => {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("renderer-cli analytic pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(&self.analytic_pipeline);
                    pass.set_vertex_buffer(0, analytic_buffer.slice(..));
                    pass.draw(range.clone(), 0..1);
                }
                AnalyticDrawCommand::Textured {
                    texture_index,
                    vertices,
                } => {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("renderer-cli analytic textured pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(&self.textured_pipeline_single);
                    pass.set_vertex_buffer(0, textured_buffer.slice(..));
                    pass.set_bind_group(0, &texture_resources[*texture_index].2, &[]);
                    pass.draw(vertices.clone(), 0..1);
                }
            }
        }

        let effect_texture;
        let final_texture = if let Some(effect) = &scene.effect {
            let pipeline = self.build_effect_pipeline(&effect.shader)?;
            let created = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("renderer-cli analytic effect target"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let effect_view = created.create_view(&wgpu::TextureViewDescriptor::default());
            let effect_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("renderer-cli analytic effect bind group"),
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("renderer-cli analytic effect pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &effect_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &effect_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            effect_texture = created;
            &effect_texture
        } else {
            &texture
        };

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: final_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| RenderError::Readback)?
            .map_err(|_| RenderError::Readback)?;
        let mapped = slice.get_mapped_range();
        let mut pixels = vec![0; (unpadded_bytes_per_row * height) as usize];
        for (row, target) in pixels
            .chunks_exact_mut(unpadded_bytes_per_row as usize)
            .enumerate()
        {
            let start = row * padded_bytes_per_row as usize;
            target.copy_from_slice(&mapped[start..start + unpadded_bytes_per_row as usize]);
        }
        drop(mapped);
        output_buffer.unmap();
        // Unlike the original path, there is no supersampled intermediate to
        // downsample here: `pixels` is already at the scene's declared size.
        Ok((pixels, Vec::new()))
    }
}

fn create_pipelines(
    device: &wgpu::Device,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::BindGroupLayout,
    wgpu::Sampler,
) {
    let primitive_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("renderer-cli primitives"),
        source: wgpu::ShaderSource::Wgsl(PRIMITIVE_SHADER.into()),
    });
    let primitive_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("renderer-cli primitive layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let primitive_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("renderer-cli primitive pipeline"),
        layout: Some(&primitive_layout),
        vertex: wgpu::VertexState {
            module: &primitive_shader,
            entry_point: "vs_main",
            buffers: &[Vertex::layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &primitive_shader,
            entry_point: "fs_main",
            targets: &[Some(color_target())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLE_COUNT,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });
    let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("renderer-cli texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
    let texture_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("renderer-cli textured quads"),
        source: wgpu::ShaderSource::Wgsl(TEXTURED_SHADER.into()),
    });
    let textured_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("renderer-cli textured layout"),
        bind_group_layouts: &[&texture_bind_group_layout],
        push_constant_ranges: &[],
    });
    let textured_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("renderer-cli textured pipeline"),
        layout: Some(&textured_layout),
        vertex: wgpu::VertexState {
            module: &texture_shader,
            entry_point: "vs_main",
            buffers: &[TexturedVertex::layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &texture_shader,
            entry_point: "fs_main",
            targets: &[Some(color_target())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLE_COUNT,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("renderer-cli texture sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (
        primitive_pipeline,
        textured_pipeline,
        texture_bind_group_layout,
        sampler,
    )
}

/// Builds the two pipelines the analytic-AA "shadow mode" path uses (see the
/// `impl GpuRenderer` block above `create_pipelines`, and `ANALYTIC_SHADER`).
/// Wholly separate from `create_pipelines`, which this function does not
/// call and does not modify in any way -- the existing MSAA+supersampling
/// pipelines it builds are unaffected.
///
/// - `analytic_pipeline` draws rect/ellipse/line primitives with the SDF
///   shader (`ANALYTIC_SHADER`), single-sampled (`multisample.count == 1`),
///   matching the single-sampled render target `render_composed_rgba_
///   analytic_with_cache` uses (no MSAA texture, no resolve).
/// - `textured_pipeline_single` draws text glyphs and images. It uses the
///   exact same shader source (`TEXTURED_SHADER`) and vertex format
///   (`TexturedVertex`) as the existing `textured_pipeline`, but a *new*
///   `wgpu::RenderPipeline` object is required because `textured_pipeline`
///   was built with `multisample.count == MSAA_SAMPLE_COUNT` to match the
///   msaa attachment it is normally drawn into; a pipeline's multisample
///   state must match whatever render pass attachment it is bound within; a
///   4x-multisample pipeline cannot be used inside a single-sampled render
///   pass (`create_render_pipeline` would be fine, but the later
///   `set_pipeline` inside a mismatched pass would be a validation error).
///   Sharing `texture_bind_group_layout` and the (multisample-agnostic)
///   sampler with the original path is safe: neither carries any
///   sample-count information.
fn create_analytic_pipelines(
    device: &wgpu::Device,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> (wgpu::RenderPipeline, wgpu::RenderPipeline) {
    let analytic_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("renderer-cli analytic primitives"),
        source: wgpu::ShaderSource::Wgsl(ANALYTIC_SHADER.into()),
    });
    let analytic_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("renderer-cli analytic layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let analytic_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("renderer-cli analytic pipeline"),
        layout: Some(&analytic_layout),
        vertex: wgpu::VertexState {
            module: &analytic_shader,
            entry_point: "vs_main",
            buffers: &[AnalyticVertex::layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &analytic_shader,
            entry_point: "fs_main",
            targets: &[Some(color_target())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let texture_shader_single = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("renderer-cli textured quads (single-sample)"),
        source: wgpu::ShaderSource::Wgsl(TEXTURED_SHADER.into()),
    });
    let textured_layout_single = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("renderer-cli textured layout (single-sample)"),
        bind_group_layouts: &[texture_bind_group_layout],
        push_constant_ranges: &[],
    });
    let textured_pipeline_single = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("renderer-cli textured pipeline (single-sample)"),
        layout: Some(&textured_layout_single),
        vertex: wgpu::VertexState {
            module: &texture_shader_single,
            entry_point: "vs_main",
            buffers: &[TexturedVertex::layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &texture_shader_single,
            entry_point: "fs_main",
            targets: &[Some(color_target())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    (analytic_pipeline, textured_pipeline_single)
}

fn color_target() -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

/// Color target for the full-canvas effect pass. Unlike `color_target`
/// (used to composite potentially-transparent primitives/text/images over
/// each other and over the canvas background), the effect pass fully
/// replaces every pixel of its target with the wrapped shader's output, so
/// alpha blending must be disabled — otherwise a low- or zero-alpha effect
/// output would blend against the (uninitialized/cleared) target instead of
/// being written directly, silently discarding the effect's result.
fn effect_color_target() -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    }
}

const ANALYTIC_SHAPE_RECT: u32 = 0;
const ANALYTIC_SHAPE_ELLIPSE: u32 = 1;
const ANALYTIC_SHAPE_LINE: u32 = 2;

/// Geometry margin, in scene-pixel units, that every analytic-AA shape
/// builder expands its emitted triangle(s) by beyond the shape's true
/// mathematical boundary (while the SDF math itself still measures distance
/// to the *true*, unexpanded boundary -- see each builder function). The
/// rasterizer only ever runs the fragment shader on pixels actually covered
/// by the submitted geometry; without this margin, pixels just outside the
/// exact edge -- exactly the pixels `fwidth`-based coverage needs to fade
/// smoothly through on its way to 0 -- would never be shaded at all,
/// producing a hard clip at the true edge instead of a smooth fade beside
/// it. 2 scene pixels is comfortably wider than the ~1-pixel coverage band
/// `fs_main` computes for any shape/canvas size this renderer supports.
const ANALYTIC_AA_MARGIN: f32 = 2.0;

/// Same clip-space mapping as `vertex()`/`add_textured_rect`'s `point()`
/// helper above, factored out for the three analytic-AA shape builders.
fn analytic_clip_position(x: f32, y: f32, scene: &SceneV1) -> [f32; 2] {
    [
        x / scene.canvas.width as f32 * 2.0 - 1.0,
        1.0 - y / scene.canvas.height as f32 * 2.0,
    ]
}

/// Resolves the actual RGBA color one vertex at scene-pixel position
/// `(px, py)` should carry for a shape's `fill`, given that shape's bounding
/// center and half-extent (half-width/half-height for a `Rect`, `rx`/`ry`
/// for an `Ellipse`).
///
/// For a solid fill every vertex gets the same color -- exactly today's
/// existing flat-fill behavior, preserved unchanged. For a gradient, each
/// vertex gets its *own* color computed from its own position, and the GPU
/// rasterizer linearly interpolates between a triangle's vertex colors
/// across its interior automatically -- this project's existing flat-color
/// rendering already relies on this same hardware behavior (every vertex of
/// one shape simply happens to receive the same color today). That means a
/// gradient needs no fragment-shader changes in either the default (MSAA,
/// `Vertex`/`PRIMITIVE_SHADER`) or analytic-AA (`AnalyticVertex`/
/// `ANALYTIC_SHADER`) pipeline: both call this same function per emitted
/// vertex and let interpolation do the rest. (Confirmed empirically, not
/// just assumed: a rect built with deliberately different literal per-corner
/// colors was rendered and its raw RGBA output showed a smooth blend across
/// the shape rather than a flat or hard-cut result.)
///
/// Linear gradients project `(px, py) - center` onto the unit direction
/// vector derived from `angle_degrees` (`0` = `+x`; increasing values rotate
/// clockwise in this schema's y-down scene-pixel space), normalized by the
/// shape's bounding half-extent projected onto that same direction, then
/// clamp to `[0, 1]` and mix `from`/`to` by that fraction.
///
/// Radial gradients normalize `(px, py)`'s offset from `center` by an
/// elliptical metric using `half_extent` as the two radii (so the gradient
/// reaches `edge` exactly at the ellipse inscribed in the shape's bounding
/// box -- for a `Rect` this means straight edge midpoints reach `edge`
/// exactly while corners clamp to `edge` slightly before the true corner),
/// then mix `center`/`edge` by that fraction.
fn fill_vertex_color(
    fill: &FillV1,
    px: f32,
    py: f32,
    center: [f32; 2],
    half_extent: [f32; 2],
) -> Color {
    match fill {
        FillV1::Solid(color) => *color,
        FillV1::Gradient(GradientV1::LinearGradient {
            from,
            to,
            angle_degrees,
        }) => {
            let angle = angle_degrees.to_radians();
            let direction = [angle.cos(), angle.sin()];
            let relative = [px - center[0], py - center[1]];
            let projected = relative[0] * direction[0] + relative[1] * direction[1];
            let extent = (half_extent[0] * direction[0].abs()
                + half_extent[1] * direction[1].abs())
            .max(1e-6);
            let fraction = ((projected / extent) + 1.0) / 2.0;
            lerp_color(*from, *to, fraction.clamp(0.0, 1.0))
        }
        FillV1::Gradient(GradientV1::RadialGradient {
            center: stop_center,
            edge,
        }) => {
            let rx = half_extent[0].max(1e-6);
            let ry = half_extent[1].max(1e-6);
            let dx = (px - center[0]) / rx;
            let dy = (py - center[1]) / ry;
            let fraction = (dx * dx + dy * dy).sqrt();
            lerp_color(*stop_center, *edge, fraction.clamp(0.0, 1.0))
        }
    }
}

fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Emits an analytic-AA rect: a single quad expanded by `ANALYTIC_AA_MARGIN`
/// on every side, carrying the *true* (unexpanded) half-extent in `param0`
/// and the *true* corner radius in `param1.x` so `rect_sdf` in
/// `ANALYTIC_SHADER` measures distance to the actual declared (and possibly
/// rounded) rect boundary. Each vertex's color is resolved individually via
/// `fill_vertex_color`, so a gradient `fill` renders correctly here exactly
/// as it does in the default (MSAA) pipeline's `add_rect` -- see that
/// function's sibling doc comment on `fill_vertex_color` above.
// Every parameter here is a genuinely distinct, independently-meaningful
// piece of geometry/styling/rendering context (not accidental duplication);
// bundling them into a params struct for this one internal helper would
// only add indirection, not clarity.
#[allow(clippy::too_many_arguments)]
fn add_rect_analytic(
    vertices: &mut Vec<AnalyticVertex>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    corner_radius: f32,
    fill: &FillV1,
    scene: &SceneV1,
) {
    let half_size = [width / 2.0, height / 2.0];
    let center = [x + half_size[0], y + half_size[1]];
    let margin = ANALYTIC_AA_MARGIN;
    let make = |px: f32, py: f32| AnalyticVertex {
        clip_position: analytic_clip_position(px, py, scene),
        color: fill_vertex_color(fill, px, py, center, half_size),
        shape_kind: ANALYTIC_SHAPE_RECT,
        local: [px - center[0], py - center[1]],
        param0: half_size,
        param1: [corner_radius, 0.0],
        param2: [0.0; 2],
    };
    let a = make(x - margin, y - margin);
    let b = make(x + width + margin, y - margin);
    let c = make(x + width + margin, y + height + margin);
    let d = make(x - margin, y + height + margin);
    vertices.extend([a, b, c, a, c, d]);
}

/// Emits an analytic-AA ellipse as a triangle fan (mirroring `add_ellipse`'s
/// shape), but with the fan's *geometry* radii expanded by
/// `ANALYTIC_AA_MARGIN` (so the rasterizer shades a ring of pixels just
/// outside the true boundary) while `local` -- and therefore `ellipse_sdf`
/// in `ANALYTIC_SHADER` -- is always computed against the *true*,
/// unexpanded `rx`/`ry`. Each vertex's color is resolved individually via
/// `fill_vertex_color` (using the *true*, unexpanded `rx`/`ry` as its
/// half-extent, matching `add_ellipse`), so a gradient `fill` renders
/// correctly here exactly as it does in the default (MSAA) pipeline.
fn add_ellipse_analytic(
    vertices: &mut Vec<AnalyticVertex>,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    fill: &FillV1,
    scene: &SceneV1,
) {
    let geo_rx = rx + ANALYTIC_AA_MARGIN;
    let geo_ry = ry + ANALYTIC_AA_MARGIN;
    let center = [cx, cy];
    let half_extent = [rx, ry];
    let make = |px: f32, py: f32| AnalyticVertex {
        clip_position: analytic_clip_position(px, py, scene),
        color: fill_vertex_color(fill, px, py, center, half_extent),
        shape_kind: ANALYTIC_SHAPE_ELLIPSE,
        local: [(px - cx) / rx, (py - cy) / ry],
        param0: [0.0; 2],
        param1: [0.0; 2],
        param2: [0.0; 2],
    };
    let center = make(cx, cy);
    for index in 0..ELLIPSE_SEGMENTS {
        let start = std::f32::consts::TAU * index as f32 / ELLIPSE_SEGMENTS as f32;
        let end = std::f32::consts::TAU * (index + 1) as f32 / ELLIPSE_SEGMENTS as f32;
        vertices.extend([
            center,
            make(cx + geo_rx * start.cos(), cy + geo_ry * start.sin()),
            make(cx + geo_rx * end.cos(), cy + geo_ry * end.sin()),
        ]);
    }
}

/// Emits an analytic-AA line as a single quad wide/long enough to cover the
/// entire capsule (both rounded ends included) plus `ANALYTIC_AA_MARGIN`,
/// carrying the *true* (unexpanded) endpoints and half-thickness so
/// `capsule_sdf` in `ANALYTIC_SHADER` measures distance to the actual
/// declared stroke.
fn add_line_analytic(
    vertices: &mut Vec<AnalyticVertex>,
    start: [f32; 2],
    end: [f32; 2],
    thickness: f32,
    color: Color,
    scene: &SceneV1,
) {
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let length = (dx * dx + dy * dy).sqrt();
    if length == 0.0 {
        return;
    }
    let dir = [dx / length, dy / length];
    let perp = [-dir[1], dir[0]];
    let half_thickness = thickness / 2.0;
    let extent = half_thickness + ANALYTIC_AA_MARGIN;
    let offset = |along: f32, across: f32| {
        [
            dir[0] * along + perp[0] * across,
            dir[1] * along + perp[1] * across,
        ]
    };
    let corner = |base: [f32; 2], along: f32, across: f32| {
        let delta = offset(along, across);
        [base[0] + delta[0], base[1] + delta[1]]
    };
    let p1 = corner(start, -extent, -extent);
    let p2 = corner(end, extent, -extent);
    let p3 = corner(end, extent, extent);
    let p4 = corner(start, -extent, extent);
    let make = |p: [f32; 2]| AnalyticVertex {
        clip_position: analytic_clip_position(p[0], p[1], scene),
        color,
        shape_kind: ANALYTIC_SHAPE_LINE,
        local: p,
        param0: start,
        param1: end,
        param2: [half_thickness, 0.0],
    };
    let a = make(p1);
    let b = make(p2);
    let c = make(p3);
    let d = make(p4);
    vertices.extend([a, b, c, a, c, d]);
}

enum DrawCommand {
    Primitive(std::ops::Range<u32>),
    Textured {
        texture_index: usize,
        vertices: std::ops::Range<u32>,
    },
}

struct TextureData {
    label: String,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

struct CompositionPlan {
    primitive_vertices: Vec<Vertex>,
    textured_vertices: Vec<TexturedVertex>,
    commands: Vec<DrawCommand>,
    textures: Vec<TextureData>,
}

/// Decoded image pixel data, cached by exactly the key `composition_plan`
/// already used for its function-local `image_textures` dedup map:
/// `(source, upload_width, upload_height)`. Storing the final processed
/// buffer (post raster-decode/SVG-rasterize *and* post-resize) means a cache
/// hit needs no further work beyond a clone into that frame's own
/// `CompositionPlan.textures`.
struct DecodedImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Rasterized glyph pixel data (already alpha-expanded to RGBA, matching
/// what used to be pushed straight into `TextureData::pixels`), cached by
/// exactly the key `composition_plan` already used for its function-local
/// `glyph_textures` dedup map: `(char, size_bits)`.
struct DecodedGlyph {
    metrics: fontdue::Metrics,
    pixels: Vec<u8>,
}

/// Cross-call cache of the *expensive* CPU-side work `composition_plan`
/// performs per image/glyph -- disk read + `image`-crate decode or
/// `resvg`/`usvg`/`tiny-skia` SVG parse+rasterize for images, `fontdue`
/// rasterization for glyphs -- keyed identically to how `composition_plan`
/// already deduped repeats *within* one call before this cache existed.
///
/// `render_composed_rgba` (single-shot renders, including every `render_png`
/// / `render_rgba` call) creates a fresh, single-use `AssetCache` per call,
/// so its behavior -- and therefore the golden-image tests -- is unaffected:
/// every key is a "miss" exactly once, exactly as when the maps were
/// function-local.
///
/// `render_gif_with_asset_root` instead creates ONE `AssetCache` before its
/// per-frame loop and passes it by `&mut` reference into every frame's
/// `composition_plan` call, so a given `(source, width, height)` or `(char,
/// size_bits)` is decoded/rasterized at most once across the whole GIF
/// export, no matter how many frames reference it -- static backgrounds,
/// logos, and labels included.
///
/// This intentionally caches only the decoded *pixel data*, not GPU texture
/// indices: each `composition_plan` call still pushes its own fresh entry
/// (and index) into that frame's `CompositionPlan.textures` even on a cache
/// hit, since GPU upload (`render_composed_rgba`'s `create_texture` /
/// `write_texture` calls) still happens per frame -- see the module-level
/// task notes on why that upload step is out of scope here.
#[derive(Default)]
struct AssetCache {
    images: HashMap<(String, u32, u32), DecodedImage>,
    glyphs: HashMap<(char, u32), DecodedGlyph>,
    /// Test-only instrumentation: counts actual cache-miss decodes/
    /// rasterizations (not lookups), so tests can assert a cache shared
    /// across N `composition_plan` calls performs the expensive work exactly
    /// once instead of N times. Never read outside `#[cfg(test)]` code.
    #[cfg(test)]
    image_decodes: usize,
    #[cfg(test)]
    glyph_rasterizations: usize,
}

#[cfg(test)]
impl AssetCache {
    fn image_decode_count(&self) -> usize {
        self.image_decodes
    }

    fn glyph_rasterization_count(&self) -> usize {
        self.glyph_rasterizations
    }
}

#[cfg(test)]
fn vertices_for_scene(scene: &SceneV1) -> (Vec<Vertex>, Vec<String>) {
    let mut vertices = Vec::new();
    let warnings = Vec::new();
    for node in &scene.nodes {
        add_node_vertices(&mut vertices, node, scene);
    }
    (vertices, warnings)
}

fn add_node_vertices(vertices: &mut Vec<Vertex>, node: &renderer_schema::NodeV1, scene: &SceneV1) {
    let [dx, dy] = node.translate;
    match &node.kind {
        NodeKindV1::Rect {
            x,
            y,
            width,
            height,
            corner_radius,
            fill,
        } => {
            let x = *x + dx;
            let y = *y + dy;
            add_rect(vertices, x, y, *width, *height, *corner_radius, fill, scene);
        }
        NodeKindV1::Ellipse {
            cx,
            cy,
            rx,
            ry,
            fill,
        } => {
            let cx = *cx + dx;
            let cy = *cy + dy;
            add_ellipse(vertices, cx, cy, *rx, *ry, fill, scene);
        }
        NodeKindV1::Line {
            x1,
            y1,
            x2,
            y2,
            thickness,
            fill,
        } => {
            let start = [*x1 + dx, *y1 + dy];
            let end = [*x2 + dx, *y2 + dy];
            add_line(
                vertices,
                start,
                end,
                *thickness,
                fill.resolve_solid(),
                scene,
            );
        }
        NodeKindV1::Path { points, fill } => {
            let translated: Vec<_> = points
                .iter()
                .map(|point| renderer_schema::PointV1 {
                    x: point.x + dx,
                    y: point.y + dy,
                })
                .collect();
            add_path(vertices, &translated, fill.resolve_solid(), scene);
        }
        NodeKindV1::Text { .. } | NodeKindV1::Image { .. } => {}
    }
}

fn composition_plan(
    scene: &SceneV1,
    asset_root: &Path,
    font: &Font,
    cache: &mut AssetCache,
) -> Result<CompositionPlan, RenderError> {
    let mut plan = CompositionPlan {
        primitive_vertices: Vec::new(),
        textured_vertices: Vec::new(),
        commands: Vec::new(),
        textures: Vec::new(),
    };
    // These two maps stay function-local (unlike `cache`, which is shared
    // across calls): they dedup repeated references to the same asset
    // *within this one call* to a single `plan.textures` entry/index, so a
    // glyph or image referenced twice in one frame still only gets pushed
    // into that frame's `CompositionPlan.textures` once -- exactly the
    // within-call behavior this file had before `AssetCache` existed.
    let mut image_textures = HashMap::new();
    let mut glyph_textures = HashMap::new();
    let mut texture_pixels = 0_u64;
    for node in &scene.nodes {
        match &node.kind {
            NodeKindV1::Text {
                x,
                y,
                text,
                size,
                fill,
            } => {
                let x = *x + node.translate[0];
                let y = *y + node.translate[1];
                // Text does not implement true gradient rendering (see
                // `FillV1::resolve_solid`'s doc comment); a gradient fill
                // resolves to a flat representative color here.
                let color = fill.resolve_solid();
                validate_text_raster(node.id.as_str(), text, *size, scene)?;
                let mut cursor_x = x;
                let mut previous = None;
                for character in text.chars() {
                    if let Some(left) = previous {
                        cursor_x += font.horizontal_kern(left, character, *size).unwrap_or(0.0);
                    }
                    let key = (character, size.to_bits());
                    let (texture_index, metrics) = if let Some(value) = glyph_textures.get(&key) {
                        *value
                    } else {
                        // Rasterize only on a cache miss: `cache.glyphs`
                        // never holds a zero-size entry (see the `continue`
                        // below), so a hit here always has real pixel data.
                        let decoded = match cache.glyphs.entry(key) {
                            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                let (metrics, bitmap) = font.rasterize(character, *size);
                                if metrics.width == 0 || metrics.height == 0 {
                                    cursor_x += metrics.advance_width;
                                    previous = Some(character);
                                    continue;
                                }
                                let pixels = bitmap
                                    .into_iter()
                                    .flat_map(|alpha| [255, 255, 255, alpha])
                                    .collect();
                                #[cfg(test)]
                                {
                                    cache.glyph_rasterizations += 1;
                                }
                                entry.insert(DecodedGlyph { metrics, pixels })
                            }
                        };
                        let metrics = decoded.metrics;
                        texture_pixels = reserve_composition_pixels(
                            texture_pixels,
                            (metrics.width * metrics.height) as u64,
                        )?;
                        let index = plan.textures.len();
                        plan.textures.push(TextureData {
                            label: format!("glyph-{}-{}", character as u32, size),
                            width: metrics.width as u32,
                            height: metrics.height as u32,
                            pixels: decoded.pixels.clone(),
                        });
                        glyph_textures.insert(key, (index, metrics));
                        (index, metrics)
                    };
                    let start = plan.textured_vertices.len() as u32;
                    add_textured_rect(
                        &mut plan.textured_vertices,
                        cursor_x + metrics.xmin as f32,
                        y + (*size - metrics.height as f32 - metrics.ymin as f32),
                        metrics.width as f32,
                        metrics.height as f32,
                        color,
                        scene,
                    );
                    plan.commands.push(DrawCommand::Textured {
                        texture_index,
                        vertices: start..start + 6,
                    });
                    cursor_x += metrics.advance_width;
                    previous = Some(character);
                }
            }
            NodeKindV1::Image {
                x,
                y,
                width,
                height,
                source,
            } => {
                let x = *x + node.translate[0];
                let y = *y + node.translate[1];
                let target_width = bounded_image_dimension(*width, "width")?;
                let target_height = bounded_image_dimension(*height, "height")?;
                ensure_target_image_dimensions(target_width, target_height, source)?;
                let upload_width = target_width.min(MAX_GPU_TEXTURE_DIMENSION);
                let upload_height = target_height.min(MAX_GPU_TEXTURE_DIMENSION);
                let cache_key = (source.clone(), upload_width, upload_height);
                let texture_index = if let Some(index) = image_textures.get(&cache_key) {
                    *index
                } else {
                    // Decode/rasterize only on a cache miss.
                    let decoded = match cache.images.entry(cache_key.clone()) {
                        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            // For raster sources this returns the
                            // native-resolution decode, resized below; for
                            // SVG sources `load_image` rasterizes directly
                            // at `upload_width`x`upload_height` (already
                            // equal to `upload_dimensions(..)`'s result, so
                            // the resize below becomes a no-op for SVG).
                            let image =
                                load_image(asset_root, source, upload_width, upload_height)?;
                            let (final_width, final_height) = upload_dimensions(
                                image.width(),
                                image.height(),
                                target_width,
                                target_height,
                            );
                            let image =
                                if image.width() == final_width && image.height() == final_height {
                                    image
                                } else {
                                    image::DynamicImage::ImageRgba8(image)
                                        .resize_exact(
                                            final_width,
                                            final_height,
                                            image::imageops::FilterType::Triangle,
                                        )
                                        .to_rgba8()
                                };
                            #[cfg(test)]
                            {
                                cache.image_decodes += 1;
                            }
                            entry.insert(DecodedImage {
                                width: image.width(),
                                height: image.height(),
                                pixels: image.into_raw(),
                            })
                        }
                    };
                    texture_pixels = reserve_composition_pixels(
                        texture_pixels,
                        u64::from(decoded.width) * u64::from(decoded.height),
                    )?;
                    let index = plan.textures.len();
                    plan.textures.push(TextureData {
                        label: format!("image-{source}"),
                        width: decoded.width,
                        height: decoded.height,
                        pixels: decoded.pixels.clone(),
                    });
                    image_textures.insert(cache_key, index);
                    index
                };
                let start = plan.textured_vertices.len() as u32;
                add_textured_rect(
                    &mut plan.textured_vertices,
                    x,
                    y,
                    *width,
                    *height,
                    [1.0; 4],
                    scene,
                );
                plan.commands.push(DrawCommand::Textured {
                    texture_index,
                    vertices: start..start + 6,
                });
            }
            _ => {
                let start = plan.primitive_vertices.len() as u32;
                add_node_vertices(&mut plan.primitive_vertices, node, scene);
                let end = plan.primitive_vertices.len() as u32;
                if start != end {
                    if let Some(DrawCommand::Primitive(range)) = plan.commands.last_mut() {
                        range.end = end;
                    } else {
                        plan.commands.push(DrawCommand::Primitive(start..end));
                    }
                }
            }
        }
    }
    Ok(plan)
}

/// Draw-command list for the analytic-AA composition plan (see
/// `CompositionPlanAnalytic`/`composition_plan_analytic`). This mirrors
/// `DrawCommand` above but splits its catch-all `Primitive` variant into two
/// -- `Analytic` (Rect/Ellipse/Line, drawn with `AnalyticVertex`/`analytic_
/// pipeline`) and `Path` (drawn with the original `Vertex`/`primitive_
/// pipeline`, as the documented MSAA fallback -- see `composition_plan_
/// analytic`'s doc comment) -- since those two need different GPU pipelines
/// with different render-target sample counts and therefore different
/// render passes (see `render_composed_rgba_analytic_with_cache`).
enum AnalyticDrawCommand {
    Analytic(std::ops::Range<u32>),
    Path(std::ops::Range<u32>),
    Textured {
        texture_index: usize,
        vertices: std::ops::Range<u32>,
    },
}

/// Analytic-AA counterpart to `CompositionPlan`. `textures` and `textured_
/// vertices` are the same `TextureData`/`TexturedVertex` types the original
/// path uses (text/images are handled identically either way -- see
/// `composition_plan_analytic`); `analytic_vertices` holds the new `Analytic
/// Vertex` geometry for Rect/Ellipse/Line, and `path_vertices` holds plain
/// `Vertex` geometry (built with the existing, untouched `add_path`) for the
/// `Path` MSAA fallback.
struct CompositionPlanAnalytic {
    analytic_vertices: Vec<AnalyticVertex>,
    path_vertices: Vec<Vertex>,
    textured_vertices: Vec<TexturedVertex>,
    commands: Vec<AnalyticDrawCommand>,
    textures: Vec<TextureData>,
}

/// Analytic-AA counterpart to `composition_plan`. A separate function
/// (rather than a shared helper `composition_plan` also calls) so that
/// `composition_plan` itself -- and therefore every existing render entry
/// point's behavior -- is not touched at all by this addition. Text and
/// image handling below is intentionally near-identical to
/// `composition_plan`'s (same caching, same validation, same texture
/// upload prep -- none of `AssetCache`/`DecodedImage`/`DecodedGlyph`/
/// `TextureData`/`reserve_composition_pixels`/`validate_text_raster`/
/// `load_image`/`upload_dimensions`/`bounded_image_dimension`/`ensure_
/// target_image_dimensions` are modified, only reused) since this project's
/// SDF/`fwidth` anti-aliasing technique is specifically about vector
/// *shape* edges, not about how text glyphs or images are rasterized/
/// uploaded.
///
/// Rect/Ellipse/Line nodes are built with the new `add_*_analytic` builders
/// into `analytic_vertices`. `Path` nodes are the one documented exception
/// to "fully analytic": implementing a mathematically sound analytic SDF for
/// an arbitrary (potentially concave, potentially self-intersecting)
/// filled polygon is substantially more involved than the closed-form
/// rect/ellipse/capsule SDFs above, so -- as this project's task brief
/// explicitly allows -- `Path` nodes here fall back to exactly the existing
/// MSAA `primitive_pipeline`/`add_path`/`Vertex` machinery, unchanged,
/// executed in its own multisampled render pass that resolves into the same
/// target the analytic passes draw into (see `render_composed_rgba_
/// analytic_with_cache`). This means a scene that uses `Path` nodes gets
/// MSAA-only (no supersampling) quality for just those nodes while every
/// other shape in the same scene still gets full analytic AA -- a
/// consciously narrower guarantee for one shape kind, not a silent gap.
fn composition_plan_analytic(
    scene: &SceneV1,
    asset_root: &Path,
    font: &Font,
    cache: &mut AssetCache,
) -> Result<CompositionPlanAnalytic, RenderError> {
    let mut plan = CompositionPlanAnalytic {
        analytic_vertices: Vec::new(),
        path_vertices: Vec::new(),
        textured_vertices: Vec::new(),
        commands: Vec::new(),
        textures: Vec::new(),
    };
    let mut image_textures = HashMap::new();
    let mut glyph_textures = HashMap::new();
    let mut texture_pixels = 0_u64;
    for node in &scene.nodes {
        match &node.kind {
            NodeKindV1::Text {
                x,
                y,
                text,
                size,
                fill,
            } => {
                let x = *x + node.translate[0];
                let y = *y + node.translate[1];
                // Text does not implement true gradient rendering (see
                // `FillV1::resolve_solid`'s doc comment); a gradient fill
                // resolves to a flat representative color here.
                let color = fill.resolve_solid();
                validate_text_raster(node.id.as_str(), text, *size, scene)?;
                let mut cursor_x = x;
                let mut previous = None;
                for character in text.chars() {
                    if let Some(left) = previous {
                        cursor_x += font.horizontal_kern(left, character, *size).unwrap_or(0.0);
                    }
                    let key = (character, size.to_bits());
                    let (texture_index, metrics) = if let Some(value) = glyph_textures.get(&key) {
                        *value
                    } else {
                        let decoded = match cache.glyphs.entry(key) {
                            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                let (metrics, bitmap) = font.rasterize(character, *size);
                                if metrics.width == 0 || metrics.height == 0 {
                                    cursor_x += metrics.advance_width;
                                    previous = Some(character);
                                    continue;
                                }
                                let pixels = bitmap
                                    .into_iter()
                                    .flat_map(|alpha| [255, 255, 255, alpha])
                                    .collect();
                                #[cfg(test)]
                                {
                                    cache.glyph_rasterizations += 1;
                                }
                                entry.insert(DecodedGlyph { metrics, pixels })
                            }
                        };
                        let metrics = decoded.metrics;
                        texture_pixels = reserve_composition_pixels(
                            texture_pixels,
                            (metrics.width * metrics.height) as u64,
                        )?;
                        let index = plan.textures.len();
                        plan.textures.push(TextureData {
                            label: format!("glyph-{}-{}", character as u32, size),
                            width: metrics.width as u32,
                            height: metrics.height as u32,
                            pixels: decoded.pixels.clone(),
                        });
                        glyph_textures.insert(key, (index, metrics));
                        (index, metrics)
                    };
                    let start = plan.textured_vertices.len() as u32;
                    add_textured_rect(
                        &mut plan.textured_vertices,
                        cursor_x + metrics.xmin as f32,
                        y + (*size - metrics.height as f32 - metrics.ymin as f32),
                        metrics.width as f32,
                        metrics.height as f32,
                        color,
                        scene,
                    );
                    plan.commands.push(AnalyticDrawCommand::Textured {
                        texture_index,
                        vertices: start..start + 6,
                    });
                    cursor_x += metrics.advance_width;
                    previous = Some(character);
                }
            }
            NodeKindV1::Image {
                x,
                y,
                width,
                height,
                source,
            } => {
                let x = *x + node.translate[0];
                let y = *y + node.translate[1];
                let target_width = bounded_image_dimension(*width, "width")?;
                let target_height = bounded_image_dimension(*height, "height")?;
                ensure_target_image_dimensions(target_width, target_height, source)?;
                let upload_width = target_width.min(MAX_GPU_TEXTURE_DIMENSION);
                let upload_height = target_height.min(MAX_GPU_TEXTURE_DIMENSION);
                let cache_key = (source.clone(), upload_width, upload_height);
                let texture_index = if let Some(index) = image_textures.get(&cache_key) {
                    *index
                } else {
                    let decoded = match cache.images.entry(cache_key.clone()) {
                        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            let image =
                                load_image(asset_root, source, upload_width, upload_height)?;
                            let (final_width, final_height) = upload_dimensions(
                                image.width(),
                                image.height(),
                                target_width,
                                target_height,
                            );
                            let image =
                                if image.width() == final_width && image.height() == final_height {
                                    image
                                } else {
                                    image::DynamicImage::ImageRgba8(image)
                                        .resize_exact(
                                            final_width,
                                            final_height,
                                            image::imageops::FilterType::Triangle,
                                        )
                                        .to_rgba8()
                                };
                            #[cfg(test)]
                            {
                                cache.image_decodes += 1;
                            }
                            entry.insert(DecodedImage {
                                width: image.width(),
                                height: image.height(),
                                pixels: image.into_raw(),
                            })
                        }
                    };
                    texture_pixels = reserve_composition_pixels(
                        texture_pixels,
                        u64::from(decoded.width) * u64::from(decoded.height),
                    )?;
                    let index = plan.textures.len();
                    plan.textures.push(TextureData {
                        label: format!("image-{source}"),
                        width: decoded.width,
                        height: decoded.height,
                        pixels: decoded.pixels.clone(),
                    });
                    image_textures.insert(cache_key, index);
                    index
                };
                let start = plan.textured_vertices.len() as u32;
                add_textured_rect(
                    &mut plan.textured_vertices,
                    x,
                    y,
                    *width,
                    *height,
                    [1.0; 4],
                    scene,
                );
                plan.commands.push(AnalyticDrawCommand::Textured {
                    texture_index,
                    vertices: start..start + 6,
                });
            }
            NodeKindV1::Path { points, fill } => {
                let translated: Vec<_> = points
                    .iter()
                    .map(|point| renderer_schema::PointV1 {
                        x: point.x + node.translate[0],
                        y: point.y + node.translate[1],
                    })
                    .collect();
                let start = plan.path_vertices.len() as u32;
                add_path(
                    &mut plan.path_vertices,
                    &translated,
                    fill.resolve_solid(),
                    scene,
                );
                let end = plan.path_vertices.len() as u32;
                if start != end {
                    if let Some(AnalyticDrawCommand::Path(range)) = plan.commands.last_mut() {
                        range.end = end;
                    } else {
                        plan.commands.push(AnalyticDrawCommand::Path(start..end));
                    }
                }
            }
            NodeKindV1::Rect {
                x,
                y,
                width,
                height,
                corner_radius,
                fill,
            } => {
                let x = *x + node.translate[0];
                let y = *y + node.translate[1];
                let start = plan.analytic_vertices.len() as u32;
                add_rect_analytic(
                    &mut plan.analytic_vertices,
                    x,
                    y,
                    *width,
                    *height,
                    *corner_radius,
                    fill,
                    scene,
                );
                push_analytic_range(
                    &mut plan.commands,
                    start,
                    plan.analytic_vertices.len() as u32,
                );
            }
            NodeKindV1::Ellipse {
                cx,
                cy,
                rx,
                ry,
                fill,
            } => {
                let cx = *cx + node.translate[0];
                let cy = *cy + node.translate[1];
                let start = plan.analytic_vertices.len() as u32;
                add_ellipse_analytic(&mut plan.analytic_vertices, cx, cy, *rx, *ry, fill, scene);
                push_analytic_range(
                    &mut plan.commands,
                    start,
                    plan.analytic_vertices.len() as u32,
                );
            }
            NodeKindV1::Line {
                x1,
                y1,
                x2,
                y2,
                thickness,
                fill,
            } => {
                let start_point = [*x1 + node.translate[0], *y1 + node.translate[1]];
                let end_point = [*x2 + node.translate[0], *y2 + node.translate[1]];
                let start = plan.analytic_vertices.len() as u32;
                add_line_analytic(
                    &mut plan.analytic_vertices,
                    start_point,
                    end_point,
                    *thickness,
                    fill.resolve_solid(),
                    scene,
                );
                push_analytic_range(
                    &mut plan.commands,
                    start,
                    plan.analytic_vertices.len() as u32,
                );
            }
        }
    }
    Ok(plan)
}

/// Merges a freshly-emitted `[start, end)` `AnalyticVertex` range into
/// `commands`: extends the last command if it is already an `Analytic` run
/// (matching how `composition_plan`'s catch-all branch merges consecutive
/// `Primitive` commands above), otherwise pushes a new one. A no-op when
/// `start == end` (an unfilled range, matching e.g. `add_line_analytic`'s
/// early return for a zero-length line).
fn push_analytic_range(commands: &mut Vec<AnalyticDrawCommand>, start: u32, end: u32) {
    if start == end {
        return;
    }
    if let Some(AnalyticDrawCommand::Analytic(range)) = commands.last_mut() {
        range.end = end;
    } else {
        commands.push(AnalyticDrawCommand::Analytic(start..end));
    }
}

fn upload_dimensions(
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
) -> (u32, u32) {
    (
        source_width
            .min(target_width)
            .min(MAX_GPU_TEXTURE_DIMENSION),
        source_height
            .min(target_height)
            .min(MAX_GPU_TEXTURE_DIMENSION),
    )
}

fn add_textured_rect(
    vertices: &mut Vec<TexturedVertex>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    tint: Color,
    scene: &SceneV1,
) {
    let point = |x, y, uv| TexturedVertex {
        position: [
            x / scene.canvas.width as f32 * 2.0 - 1.0,
            1.0 - y / scene.canvas.height as f32 * 2.0,
        ],
        uv,
        tint,
    };
    let a = point(x, y, [0.0, 0.0]);
    let b = point(x + width, y, [1.0, 0.0]);
    let c = point(x + width, y + height, [1.0, 1.0]);
    let d = point(x, y + height, [0.0, 1.0]);
    vertices.extend([a, b, c, a, c, d]);
}

fn reserve_composition_pixels(current: u64, additional: u64) -> Result<u64, RenderError> {
    let total = current
        .checked_add(additional)
        .ok_or_else(|| RenderError::Asset("composition texture budget overflowed".into()))?;
    if total > MAX_COMPOSITION_TEXTURE_PIXELS {
        return Err(RenderError::Asset(
            "scene exceeds the composition texture budget".into(),
        ));
    }
    Ok(total)
}

fn validate_text_raster(
    node_id: &str,
    text: &str,
    size: f32,
    scene: &SceneV1,
) -> Result<(), RenderError> {
    if text.len() > MAX_TEXT_BYTES {
        return Err(RenderError::Asset(format!(
            "text node '{node_id}' exceeds 16 KiB"
        )));
    }
    let glyph_count = text.chars().count();
    if size > (scene.canvas.width.max(scene.canvas.height) as f32).min(MAX_GLYPH_SIZE) {
        return Err(RenderError::Asset(format!(
            "text node '{node_id}' has a font size that exceeds the raster limit"
        )));
    }
    let estimate = (size.ceil() as u64)
        .checked_mul(size.ceil() as u64)
        .and_then(|pixels| pixels.checked_mul(glyph_count as u64))
        .ok_or_else(|| {
            RenderError::Asset(format!(
                "text node '{node_id}' exceeds the text raster budget"
            ))
        })?;
    if glyph_count > MAX_TEXT_GLYPHS || estimate > MAX_TEXT_RASTER_PIXELS {
        return Err(RenderError::Asset(format!(
            "text node '{node_id}' exceeds the text raster budget"
        )));
    }
    Ok(())
}

/// Loads the image asset named by `source` (resolved and containment-checked
/// via [`resolve_asset`], exactly like every other asset lookup in this
/// file).
///
/// Raster formats (PNG/JPEG/GIF/WebP, decoded by the `image` crate) are
/// returned at their native resolution; the caller (`composition_plan`)
/// downsamples/upsamples them to the node's declared size with a bilinear
/// GPU-quad resize, same as before this function grew SVG support.
///
/// SVG assets are different: there is no "native resolution" to decode at,
/// so they are rasterized directly at `upload_width`x`upload_height` (the
/// already-`MAX_GPU_TEXTURE_DIMENSION`-clamped size the caller is about to
/// upload) for crisp output, instead of being decoded at some arbitrary size
/// and then bilinearly rescaled.
fn load_image(
    asset_root: &Path,
    source: &str,
    upload_width: u32,
    upload_height: u32,
) -> Result<image::RgbaImage, RenderError> {
    let path = resolve_asset(asset_root, source)?;
    let metadata = fs::metadata(&path).map_err(|error| RenderError::Asset(error.to_string()))?;
    if metadata.len() > MAX_ASSET_BYTES {
        return Err(RenderError::Asset(format!(
            "image '{source}' exceeds 16 MiB"
        )));
    }
    if has_svg_extension(&path) {
        let bytes = fs::read(&path).map_err(|error| RenderError::Asset(error.to_string()))?;
        if !looks_like_svg(&bytes) {
            return Err(RenderError::Asset(format!(
                "asset '{source}' has an .svg extension but its content does not look like SVG"
            )));
        }
        return rasterize_svg(&bytes, asset_root, source, upload_width, upload_height);
    }
    let reader = image::ImageReader::open(&path)
        .map_err(|error| RenderError::Asset(error.to_string()))?
        .with_guessed_format()
        .map_err(|error| RenderError::Asset(error.to_string()))?;
    let (width, height) = reader.into_dimensions()?;
    ensure_source_image_dimensions(width, height, source)?;
    let mut reader = image::ImageReader::open(&path)
        .map_err(|error| RenderError::Asset(error.to_string()))?
        .with_guessed_format()
        .map_err(|error| RenderError::Asset(error.to_string()))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_ASSET_PIXELS as u32);
    limits.max_image_height = Some(MAX_ASSET_PIXELS as u32);
    limits.max_alloc = Some(MAX_ASSET_PIXELS * 4);
    reader.limits(limits);
    Ok(reader.decode()?.to_rgba8())
}

/// Emits a (optionally rounded, optionally gradient-filled) rect for the
/// default MSAA pipeline. `corner_radius <= 0.0` takes the exact same
/// 2-triangle quad path this function always used before rounded corners
/// existed -- byte-identical output to before this feature existed, since
/// `fill_vertex_color` also returns the plain per-vertex `color` unchanged
/// for a `FillV1::Solid` fill (see that function's doc comment). This is
/// verified directly by
/// `rounded_rect_with_zero_radius_matches_the_original_plain_rect_tessellation`
/// below.
///
/// `corner_radius > 0.0` tessellates properly rather than approximating:
/// straight edges plus a small triangle-fan arc at each of the 4 corners,
/// all fanned from the rect's center -- mirroring `add_ellipse`'s
/// triangle-fan-from-center technique and reusing its angular resolution
/// via `RECT_CORNER_SEGMENTS` (`ELLIPSE_SEGMENTS / 4`, i.e. the same
/// degrees-per-segment density for one 90-degree corner as `add_ellipse`
/// uses for a full 360-degree ellipse). A fan from the center to each
/// consecutive pair of perimeter points is correct regardless of whether
/// that pair spans a curved arc segment or a straight edge -- no special
/// casing is needed for the 4 straight edges, since a single triangle
/// between two straight-edge endpoints and the center is already exact.
// See `add_rect_analytic`'s identical justification for this attribute.
#[allow(clippy::too_many_arguments)]
fn add_rect(
    vertices: &mut Vec<Vertex>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    corner_radius: f32,
    fill: &FillV1,
    scene: &SceneV1,
) {
    let center = [x + width / 2.0, y + height / 2.0];
    let half_extent = [width / 2.0, height / 2.0];
    let make = |px: f32, py: f32| {
        vertex(
            px,
            py,
            fill_vertex_color(fill, px, py, center, half_extent),
            scene,
        )
    };
    if corner_radius <= 0.0 {
        let a = make(x, y);
        let b = make(x + width, y);
        let c = make(x + width, y + height);
        let d = make(x, y + height);
        vertices.extend([a, b, c, a, c, d]);
        return;
    }
    let r = corner_radius;
    let center_vertex = make(center[0], center[1]);
    // One (arc_center_x, arc_center_y, start_angle, end_angle) tuple per
    // corner, in clockwise perimeter order starting at the top-right
    // corner (this schema's scene-pixel space is y-down, so angle 0 is
    // `+x` and increasing angle sweeps clockwise on screen).
    let quarter = std::f32::consts::FRAC_PI_2;
    let corners = [
        (x + width - r, y + r, -quarter, 0.0),
        (x + width - r, y + height - r, 0.0, quarter),
        (x + r, y + height - r, quarter, std::f32::consts::PI),
        (
            x + r,
            y + r,
            std::f32::consts::PI,
            std::f32::consts::PI + quarter,
        ),
    ];
    let mut perimeter = Vec::with_capacity(4 * (RECT_CORNER_SEGMENTS + 1));
    for (arc_cx, arc_cy, start, end) in corners {
        for segment in 0..=RECT_CORNER_SEGMENTS {
            let angle = start + (end - start) * segment as f32 / RECT_CORNER_SEGMENTS as f32;
            perimeter.push((arc_cx + r * angle.cos(), arc_cy + r * angle.sin()));
        }
    }
    for pair in perimeter.windows(2) {
        vertices.extend([
            center_vertex,
            make(pair[0].0, pair[0].1),
            make(pair[1].0, pair[1].1),
        ]);
    }
    let first = perimeter[0];
    let last = *perimeter.last().expect("perimeter is never empty");
    vertices.extend([center_vertex, make(last.0, last.1), make(first.0, first.1)]);
}

fn add_ellipse(
    vertices: &mut Vec<Vertex>,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    fill: &FillV1,
    scene: &SceneV1,
) {
    let center = [cx, cy];
    let half_extent = [rx, ry];
    let make = |px: f32, py: f32| {
        vertex(
            px,
            py,
            fill_vertex_color(fill, px, py, center, half_extent),
            scene,
        )
    };
    let center_vertex = make(cx, cy);
    for index in 0..ELLIPSE_SEGMENTS {
        let start = std::f32::consts::TAU * index as f32 / ELLIPSE_SEGMENTS as f32;
        let end = std::f32::consts::TAU * (index + 1) as f32 / ELLIPSE_SEGMENTS as f32;
        vertices.extend([
            center_vertex,
            make(cx + rx * start.cos(), cy + ry * start.sin()),
            make(cx + rx * end.cos(), cy + ry * end.sin()),
        ]);
    }
}

fn add_line(
    vertices: &mut Vec<Vertex>,
    start: [f32; 2],
    end: [f32; 2],
    thickness: f32,
    color: Color,
    scene: &SceneV1,
) {
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let length = (dx * dx + dy * dy).sqrt();
    if length == 0.0 {
        return;
    }
    let offset_x = -dy / length * thickness / 2.0;
    let offset_y = dx / length * thickness / 2.0;
    let a = vertex(start[0] + offset_x, start[1] + offset_y, color, scene);
    let b = vertex(end[0] + offset_x, end[1] + offset_y, color, scene);
    let c = vertex(end[0] - offset_x, end[1] - offset_y, color, scene);
    let d = vertex(start[0] - offset_x, start[1] - offset_y, color, scene);
    vertices.extend([a, b, c, a, c, d]);
}

fn add_path(
    vertices: &mut Vec<Vertex>,
    points: &[renderer_schema::PointV1],
    color: Color,
    scene: &SceneV1,
) {
    let origin = vertex(points[0].x, points[0].y, color, scene);
    for pair in points[1..].windows(2) {
        vertices.extend([
            origin,
            vertex(pair[0].x, pair[0].y, color, scene),
            vertex(pair[1].x, pair[1].y, color, scene),
        ]);
    }
}

fn vertex(x: f32, y: f32, color: Color, scene: &SceneV1) -> Vertex {
    Vertex {
        position: [
            x / scene.canvas.width as f32 * 2.0 - 1.0,
            1.0 - y / scene.canvas.height as f32 * 2.0,
        ],
        color,
    }
}

#[cfg(test)]
fn rasterize_text_and_images(
    pixels: &mut [u8],
    scene: &SceneV1,
    asset_root: &Path,
    font: &Font,
) -> Result<(), RenderError> {
    for node in &scene.nodes {
        match &node.kind {
            NodeKindV1::Text {
                x,
                y,
                text,
                size,
                fill,
            } => {
                let color = fill.resolve_solid();
                if text.len() > MAX_TEXT_BYTES {
                    return Err(RenderError::Asset(format!(
                        "text node '{}' exceeds 16 KiB",
                        node.id
                    )));
                }
                let glyph_count = text.chars().count();
                let canvas_limit = scene.canvas.width.max(scene.canvas.height) as f32;
                if *size > canvas_limit.min(MAX_GLYPH_SIZE) {
                    return Err(RenderError::Asset(format!(
                        "text node '{}' has a font size that exceeds the raster limit",
                        node.id
                    )));
                }
                let estimated_pixels = (*size).ceil() as u64;
                let estimated_pixels = estimated_pixels
                    .checked_mul(estimated_pixels)
                    .and_then(|pixels| pixels.checked_mul(glyph_count as u64))
                    .ok_or_else(|| {
                        RenderError::Asset(format!(
                            "text node '{}' exceeds the text raster budget",
                            node.id
                        ))
                    })?;
                if glyph_count > MAX_TEXT_GLYPHS || estimated_pixels > MAX_TEXT_RASTER_PIXELS {
                    return Err(RenderError::Asset(format!(
                        "text node '{}' exceeds the text raster budget",
                        node.id
                    )));
                }
                let mut cursor_x = *x;
                for character in text.chars() {
                    let (metrics, bitmap) = font.rasterize(character, *size);
                    let glyph_x = cursor_x + metrics.xmin as f32;
                    let glyph_y = *y + (*size - metrics.height as f32 - metrics.ymin as f32);
                    for row in 0..metrics.height {
                        for column in 0..metrics.width {
                            let alpha =
                                bitmap[row * metrics.width + column] as f32 / 255.0 * color[3];
                            blend_pixel(
                                pixels,
                                scene.canvas.width,
                                scene.canvas.height,
                                glyph_x as i32 + column as i32,
                                glyph_y as i32 + row as i32,
                                [color[0], color[1], color[2], alpha],
                            );
                        }
                    }
                    cursor_x += metrics.advance_width;
                }
            }
            NodeKindV1::Image {
                x,
                y,
                width,
                height,
                source,
            } => {
                let path = resolve_asset(asset_root, source)?;
                let metadata =
                    fs::metadata(&path).map_err(|error| RenderError::Asset(error.to_string()))?;
                if metadata.len() > MAX_ASSET_BYTES {
                    return Err(RenderError::Asset(format!(
                        "image '{}' exceeds 16 MiB",
                        source
                    )));
                }
                let reader = image::ImageReader::open(&path)
                    .map_err(|error| RenderError::Asset(error.to_string()))?
                    .with_guessed_format()
                    .map_err(|error| RenderError::Asset(error.to_string()))?;
                let (source_width, source_height) = reader.into_dimensions()?;
                ensure_source_image_dimensions(source_width, source_height, source)?;
                let target_width = bounded_image_dimension(*width, "width")?;
                let target_height = bounded_image_dimension(*height, "height")?;
                ensure_target_image_dimensions(target_width, target_height, source)?;
                let mut reader = image::ImageReader::open(&path)
                    .map_err(|error| RenderError::Asset(error.to_string()))?
                    .with_guessed_format()
                    .map_err(|error| RenderError::Asset(error.to_string()))?;
                let mut limits = image::Limits::default();
                limits.max_image_width = Some(MAX_ASSET_PIXELS as u32);
                limits.max_image_height = Some(MAX_ASSET_PIXELS as u32);
                limits.max_alloc = Some(MAX_ASSET_PIXELS * 4);
                reader.limits(limits);
                let image = reader.decode()?;
                let image = image
                    .resize_exact(
                        target_width,
                        target_height,
                        image::imageops::FilterType::Triangle,
                    )
                    .to_rgba8();
                for (column, row, value) in image.enumerate_pixels() {
                    blend_pixel(
                        pixels,
                        scene.canvas.width,
                        scene.canvas.height,
                        *x as i32 + column as i32,
                        *y as i32 + row as i32,
                        [
                            value[0] as f32 / 255.0,
                            value[1] as f32 / 255.0,
                            value[2] as f32 / 255.0,
                            value[3] as f32 / 255.0,
                        ],
                    );
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn bounded_image_dimension(value: f32, name: &str) -> Result<u32, RenderError> {
    let rounded = value.round();
    if !rounded.is_finite() || rounded <= 0.0 || rounded > 4_096.0 {
        return Err(RenderError::Asset(format!(
            "image {name} must be a finite positive value no greater than 4096"
        )));
    }
    Ok(rounded as u32)
}

fn ensure_source_image_dimensions(
    width: u32,
    height: u32,
    source: &str,
) -> Result<(), RenderError> {
    if u64::from(width) * u64::from(height) > MAX_ASSET_PIXELS {
        return Err(RenderError::Asset(format!(
            "image '{}' exceeds 16 million pixels",
            source
        )));
    }
    Ok(())
}

fn ensure_target_image_dimensions(
    width: u32,
    height: u32,
    source: &str,
) -> Result<(), RenderError> {
    if u64::from(width) * u64::from(height) > MAX_IMAGE_RASTER_PIXELS {
        return Err(RenderError::Asset(format!(
            "image '{}' exceeds the 4 million pixel raster budget",
            source
        )));
    }
    Ok(())
}

fn resolve_asset(root: &Path, source: &str) -> Result<PathBuf, RenderError> {
    let relative = Path::new(source);
    if relative.is_absolute()
        || relative
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return Err(RenderError::Asset(
            "asset paths must be relative and may not traverse parents".into(),
        ));
    }
    let root = root
        .canonicalize()
        .map_err(|error| RenderError::Asset(format!("could not resolve asset root: {error}")))?;
    let path = root.join(relative).canonicalize().map_err(|error| {
        RenderError::Asset(format!("could not resolve asset '{source}': {error}"))
    })?;
    if path.starts_with(&root) {
        Ok(path)
    } else {
        Err(RenderError::Asset("asset path escapes asset root".into()))
    }
}

#[cfg(test)]
fn blend_pixel(pixels: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: Color) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    let offset = (y as usize * width as usize + x as usize) * 4;
    let alpha = color[3];
    let destination_alpha = pixels[offset + 3] as f32 / 255.0;
    let output_alpha = alpha + destination_alpha * (1.0 - alpha);
    for channel in 0..3 {
        let destination = pixels[offset + channel] as f32 / 255.0;
        let value = if output_alpha == 0.0 {
            0.0
        } else {
            (color[channel] * alpha + destination * destination_alpha * (1.0 - alpha))
                / output_alpha
        };
        pixels[offset + channel] = (value * 255.0).round() as u8;
    }
    pixels[offset + 3] = (output_alpha * 255.0).round() as u8;
}

#[cfg(test)]
fn blend_premultiplied_layer(destination: &mut [u8], source: &[u8], width: u32, height: u32) {
    debug_assert_eq!(destination.len(), source.len());
    for offset in (0..width as usize * height as usize * 4).step_by(4) {
        let source_alpha = source[offset + 3] as f32 / 255.0;
        let destination_alpha = destination[offset + 3] as f32 / 255.0;
        let output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha);
        for channel in 0..3 {
            let source_premultiplied = source[offset + channel] as f32 / 255.0;
            let destination_premultiplied =
                destination[offset + channel] as f32 / 255.0 * destination_alpha;
            let output_premultiplied =
                source_premultiplied + destination_premultiplied * (1.0 - source_alpha);
            let value = if output_alpha == 0.0 {
                0.0
            } else {
                output_premultiplied / output_alpha
            };
            destination[offset + channel] = (value * 255.0).round() as u8;
        }
        destination[offset + 3] = (output_alpha * 255.0).round() as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use renderer_schema::{CanvasV1, KeyframeV1, NodeV1, SCENE_VERSION_V1};

    #[test]
    fn compiles_rectangle_vertices() {
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 100,
                height: 100,
                background: [0.0; 4],
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
        };
        let (vertices, warnings) = vertices_for_scene(&scene);
        assert_eq!(vertices.len(), 6);
        assert!(warnings.is_empty());
    }

    #[test]
    fn aligns_copy_rows() {
        assert_eq!(align_to(256, 256), 256);
        assert_eq!(align_to(260, 256), 512);
    }

    #[test]
    fn compiles_all_geometry_and_reports_unrasterized_nodes() {
        let mut scene = test_scene();
        scene.nodes.extend([
            NodeV1 {
                id: "ellipse".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Ellipse {
                    cx: 20.0,
                    cy: 20.0,
                    rx: 5.0,
                    ry: 5.0,
                    fill: FillV1::Solid([0.0, 1.0, 0.0, 1.0]),
                },
            },
            NodeV1 {
                id: "line".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Line {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 20.0,
                    y2: 20.0,
                    thickness: 2.0,
                    fill: FillV1::Solid([0.0, 0.0, 1.0, 1.0]),
                },
            },
            NodeV1 {
                id: "path".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Path {
                    points: vec![
                        renderer_schema::PointV1 { x: 0.0, y: 0.0 },
                        renderer_schema::PointV1 { x: 10.0, y: 0.0 },
                        renderer_schema::PointV1 { x: 0.0, y: 10.0 },
                    ],
                    fill: FillV1::Solid([1.0; 4]),
                },
            },
            NodeV1 {
                id: "text".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Text {
                    x: 0.0,
                    y: 0.0,
                    text: "t".into(),
                    size: 8.0,
                    fill: FillV1::Solid([1.0; 4]),
                },
            },
            NodeV1 {
                id: "vector".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 16.0,
                    y: 1.0,
                    width: 2.0,
                    height: 2.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0; 4]),
                },
            },
            NodeV1 {
                id: "image".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Image {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                    source: "x".into(),
                },
            },
        ]);
        let (vertices, warnings) = vertices_for_scene(&scene);
        assert!(vertices.len() > 100);
        assert!(warnings.is_empty());
    }

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

    /// Direct geometric-movement proof for `Rect` translate keyframes,
    /// through the default MSAA+supersampling `composition_plan` pipeline:
    /// renders the same scene at two `at_ms` values on either side of a
    /// translate keyframe pair, decodes real GPU output at both times, and
    /// asserts the rect's white fill genuinely appears at a *different*
    /// pixel location at each time -- not just that rendering succeeded.
    /// Sample points are chosen well inside each rect position (away from
    /// anti-aliased edges) and far enough apart that the two rect positions
    /// never overlap, so a false pass from coincidental pixel overlap is not
    /// possible.
    #[test]
    fn translate_keyframes_move_a_rect_node_to_a_different_pixel_location_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [0.0, 0.0, 0.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "box".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 4.0,
                    y: 4.0,
                    width: 10.0,
                    height: 10.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0, 1.0, 1.0, 1.0]),
                },
            }],
            timeline: Some(renderer_schema::TimelineV1 {
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
                        property: renderer_schema::AnimatedPropertyV1::Translate([30.0, 30.0]),
                    },
                ],
            }),
            effect: None,
        };
        scene.validate().unwrap();

        let width = scene.canvas.width as usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let start = scene_at(&scene, 0);
        start.validate().unwrap();
        let (start_pixels, start_warnings) = renderer.render_rgba(&start).unwrap();
        assert!(start_warnings.is_empty());

        let end = scene_at(&scene, 1_000);
        end.validate().unwrap();
        let (end_pixels, end_warnings) = renderer.render_rgba(&end).unwrap();
        assert!(end_warnings.is_empty());

        let original_center = (9, 9);
        let translated_center = (39, 39);
        let background = pixel_at(&start_pixels, 0, 0);
        let white = [255, 255, 255, 255];
        assert_eq!(background, [0, 0, 0, 255]);

        assert_eq!(
            pixel_at(&start_pixels, original_center.0, original_center.1),
            white,
            "at at_ms=0 the rect should render at its declared (untranslated) location"
        );
        assert_eq!(
            pixel_at(&start_pixels, translated_center.0, translated_center.1),
            background,
            "at at_ms=0 the translated location should still be background"
        );

        assert_eq!(
            pixel_at(&end_pixels, translated_center.0, translated_center.1),
            white,
            "at at_ms=1000 the rect should have moved to the translated location"
        );
        assert_eq!(
            pixel_at(&end_pixels, original_center.0, original_center.1),
            background,
            "at at_ms=1000 the original location should be background again since the rect moved away"
        );
    }

    /// Same geometric-movement proof as the `Rect` test above, but for a
    /// `Path` node -- whose translate offset is structurally different
    /// (applied to every point in a list, not a single x/y pair), so this
    /// specifically proves that case was not missed.
    #[test]
    fn translate_keyframes_move_a_path_node_to_a_different_pixel_location_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [0.0, 0.0, 0.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "triangle".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Path {
                    points: vec![
                        renderer_schema::PointV1 { x: 10.0, y: 10.0 },
                        renderer_schema::PointV1 { x: 30.0, y: 10.0 },
                        renderer_schema::PointV1 { x: 10.0, y: 30.0 },
                    ],
                    fill: FillV1::Solid([1.0, 1.0, 1.0, 1.0]),
                },
            }],
            timeline: Some(renderer_schema::TimelineV1 {
                fps: 2,
                duration_ms: 1_000,
                keyframes: vec![
                    KeyframeV1 {
                        at_ms: 0,
                        target: "triangle".into(),
                        property: renderer_schema::AnimatedPropertyV1::Translate([0.0, 0.0]),
                    },
                    KeyframeV1 {
                        at_ms: 1_000,
                        target: "triangle".into(),
                        property: renderer_schema::AnimatedPropertyV1::Translate([25.0, 25.0]),
                    },
                ],
            }),
            effect: None,
        };
        scene.validate().unwrap();

        let width = scene.canvas.width as usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let start = scene_at(&scene, 0);
        start.validate().unwrap();
        let (start_pixels, start_warnings) = renderer.render_rgba(&start).unwrap();
        assert!(start_warnings.is_empty());

        let end = scene_at(&scene, 1_000);
        end.validate().unwrap();
        let (end_pixels, end_warnings) = renderer.render_rgba(&end).unwrap();
        assert!(end_warnings.is_empty());

        // Sample points well inside the triangle's interior (away from its
        // anti-aliased hypotenuse edge) for the untranslated and translated
        // positions.
        let original_interior = (14, 14);
        let translated_interior = (39, 39);
        let background = pixel_at(&start_pixels, 0, 0);
        let white = [255, 255, 255, 255];
        assert_eq!(background, [0, 0, 0, 255]);

        assert_eq!(
            pixel_at(&start_pixels, original_interior.0, original_interior.1),
            white,
            "at at_ms=0 the path should render at its declared (untranslated) points"
        );
        assert_eq!(
            pixel_at(&start_pixels, translated_interior.0, translated_interior.1),
            background,
            "at at_ms=0 the translated location should still be background"
        );

        assert_eq!(
            pixel_at(&end_pixels, translated_interior.0, translated_interior.1),
            white,
            "at at_ms=1000 every point in the path should have shifted by the translate offset"
        );
        assert_eq!(
            pixel_at(&end_pixels, original_interior.0, original_interior.1),
            background,
            "at at_ms=1000 the original location should be background again since the path moved away"
        );
    }

    /// Lighter-touch confirmation that the experimental analytic-AA
    /// composition pipeline (`composition_plan_analytic` /
    /// `render_rgba_analytic_aa`) also applies `translate` -- reusing the
    /// same scene/keyframes as the default-pipeline `Rect` proof above so
    /// the two pipelines are checked against literally the same geometry,
    /// without needing the same exhaustive per-shape-kind coverage as the
    /// default path.
    #[test]
    fn translate_keyframes_move_a_rect_node_under_analytic_aa_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [0.0, 0.0, 0.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "box".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 4.0,
                    y: 4.0,
                    width: 10.0,
                    height: 10.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0, 1.0, 1.0, 1.0]),
                },
            }],
            timeline: Some(renderer_schema::TimelineV1 {
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
                        property: renderer_schema::AnimatedPropertyV1::Translate([30.0, 30.0]),
                    },
                ],
            }),
            effect: None,
        };
        scene.validate().unwrap();

        let width = scene.canvas.width as usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let start = scene_at(&scene, 0);
        start.validate().unwrap();
        let (start_pixels, start_warnings) = renderer.render_rgba_analytic_aa(&start).unwrap();
        assert!(start_warnings.is_empty());

        let end = scene_at(&scene, 1_000);
        end.validate().unwrap();
        let (end_pixels, end_warnings) = renderer.render_rgba_analytic_aa(&end).unwrap();
        assert!(end_warnings.is_empty());

        let original_center = (9, 9);
        let translated_center = (39, 39);
        let background = pixel_at(&start_pixels, 0, 0);
        let white = [255, 255, 255, 255];

        assert_eq!(
            pixel_at(&start_pixels, original_center.0, original_center.1),
            white,
            "analytic-AA path: at at_ms=0 the rect should render at its declared location"
        );
        assert_eq!(
            pixel_at(&end_pixels, translated_center.0, translated_center.1),
            white,
            "analytic-AA path: at at_ms=1000 the rect should have moved to the translated location"
        );
        assert_eq!(
            pixel_at(&end_pixels, original_center.0, original_center.1),
            background,
            "analytic-AA path: at at_ms=1000 the original location should be background again"
        );
    }

    #[test]
    fn accepts_only_png_render_paths() {
        assert!(ensure_png_output_path(Path::new("scene.png")).is_ok());
        assert!(ensure_png_output_path(Path::new("scene.PNG")).is_ok());
        assert!(matches!(
            ensure_png_output_path(Path::new("scene.gif")),
            Err(RenderError::InvalidPngOutputPath(_))
        ));
        assert!(matches!(
            ensure_png_output_path(Path::new("scene")),
            Err(RenderError::InvalidPngOutputPath(_))
        ));
    }

    #[test]
    fn covers_geometry_and_animation_edge_cases() {
        let mut scene = test_scene();
        scene.nodes.push(NodeV1 {
            id: "zero".into(),
            translate: [0.0, 0.0],
            kind: NodeKindV1::Line {
                x1: 1.0,
                y1: 1.0,
                x2: 1.0,
                y2: 1.0,
                thickness: 1.0,
                fill: FillV1::Solid([1.0; 4]),
            },
        });
        assert_eq!(vertices_for_scene(&scene).0.len(), 6);
        assert_eq!(scene_at(&scene, 1), scene);
        let variants = [
            NodeKindV1::Ellipse {
                cx: 0.0,
                cy: 0.0,
                rx: 1.0,
                ry: 1.0,
                fill: FillV1::Solid([1.0; 4]),
            },
            NodeKindV1::Line {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
                thickness: 1.0,
                fill: FillV1::Solid([1.0; 4]),
            },
            NodeKindV1::Path {
                points: vec![renderer_schema::PointV1 { x: 0.0, y: 0.0 }; 3],
                fill: FillV1::Solid([1.0; 4]),
            },
            NodeKindV1::Text {
                x: 0.0,
                y: 0.0,
                text: "x".into(),
                size: 1.0,
                fill: FillV1::Solid([1.0; 4]),
            },
            NodeKindV1::Image {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                source: "x".into(),
            },
        ];
        for mut kind in variants {
            let _ = fill_of(&kind);
            let _ = fill_mut(&mut kind);
        }
        assert_eq!(
            interpolate(&[(10, 1.0_f32), (20, 2.0)], 0, |a, b, t| a + (b - a) * t),
            Some(1.0)
        );
        assert_eq!(
            interpolate(&[(10, 1.0_f32), (20, 2.0)], 30, |a, b, t| a + (b - a) * t),
            Some(2.0)
        );
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("output"), b"bytes").unwrap();
        assert!(hash_file(&directory.path().join("output")).is_ok());
        assert!(hash_file(&directory.path().join("missing")).is_err());
    }

    #[test]
    fn renders_png_and_gif_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        {
            let directory = tempfile::tempdir().unwrap();
            let scene = test_scene();
            let png = renderer
                .render_png(&scene, &directory.path().join("scene.png"))
                .unwrap();
            assert_eq!(png.frame_count, 1);
            let mut animated = scene;
            animated.timeline = Some(renderer_schema::TimelineV1 {
                fps: 2,
                duration_ms: 1_000,
                keyframes: vec![],
            });
            let gif = renderer
                .render_gif(&animated, &directory.path().join("scene.gif"))
                .unwrap();
            assert_eq!(gif.frame_count, 2);
        }
    }

    /// Direct anti-aliasing regression test: renders a diagonal (non-axis-
    /// aligned) line and asserts at least one pixel along its edge lands
    /// strictly between the background color and the line color.
    ///
    /// Before MSAA was added, this renderer's vector primitives (rect/
    /// ellipse/line/path) had no anti-aliasing at all: decoding a real
    /// rendered PNG's raw pixel bytes showed a diagonal line's edge
    /// transitioning directly from `(255,255,255,255)` to `(89,89,89,255)`
    /// with no intermediate blended pixel anywhere along a clearly diagonal
    /// edge -- a hard, stair-stepped edge. A binary hard edge can still pass
    /// a tolerance-based golden-image comparison (edges just shift by a
    /// pixel or two), so that alone would not catch a regression back to
    /// hard edges. This test instead checks the actual pixel values along a
    /// known diagonal edge for real intermediate coverage-weighted color,
    /// which only MSAA (or another anti-aliasing scheme) can produce.
    #[test]
    fn anti_aliases_diagonal_primitive_edges_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "diagonal".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Line {
                    x1: 6.0,
                    y1: 6.0,
                    x2: 58.0,
                    y2: 58.0,
                    thickness: 6.0,
                    fill: FillV1::Solid([0.2, 0.2, 0.2, 1.0]),
                },
            }],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };
        // A corner far from the diagonal stroke (solid background) and a
        // point on the line's centerline far from both of its ends (solid
        // line interior) give the two real, GPU-rendered "pure" colors to
        // compare edge pixels against -- more robust than hardcoding
        // expected sRGB-encoded byte values here.
        let background_pixel = pixel_at(0, 0);
        let line_pixel = pixel_at(32, 32);
        assert_ne!(
            background_pixel, line_pixel,
            "sanity check: the sampled background and line-interior points must differ"
        );

        let mut found_blended_pixel = false;
        'scan: for y in 0..scene.canvas.height as usize {
            for x in 0..width {
                let pixel = pixel_at(x, y);
                let strictly_between = (0..3).all(|channel| {
                    let low = background_pixel[channel].min(line_pixel[channel]);
                    let high = background_pixel[channel].max(line_pixel[channel]);
                    pixel[channel] > low && pixel[channel] < high
                });
                if strictly_between {
                    found_blended_pixel = true;
                    break 'scan;
                }
            }
        }
        assert!(
            found_blended_pixel,
            "expected at least one pixel strictly between the background color {background_pixel:?} \
             and the line color {line_pixel:?} along the diagonal edge, proving real \
             coverage-weighted MSAA blending occurred instead of a hard binary edge"
        );
    }

    /// Direct SSAA regression test, analogous to
    /// `anti_aliases_diagonal_primitive_edges_on_an_available_gpu` above but
    /// checking supersampling's *additional* contribution on top of MSAA:
    /// renders the exact same diagonal-line scene as that test and counts
    /// how many genuinely distinct, strictly-intermediate red-channel values
    /// (the line and background are both gray/white, so R=G=B and one
    /// channel suffices) appear anywhere along the line's edge.
    ///
    /// `MSAA_SAMPLE_COUNT`x MSAA alone resolves at most a handful of
    /// coverage fractions per edge pixel (this project's own investigation
    /// that motivated adding `SUPERSAMPLE_FACTOR` found roughly 4-5 such
    /// levels decoding raw MSAA-only output); measured directly against
    /// *this* scene with `SUPERSAMPLE_FACTOR` temporarily forced to 1
    /// (MSAA-only, no supersampling), it produced exactly one distinct
    /// intermediate red value across the whole image. With
    /// `SUPERSAMPLE_FACTOR` at its real value, this same scene measured 7
    /// distinct intermediate values on this machine (Apple M1/Metal) --
    /// deterministic and stable across repeated runs, since both the
    /// geometry and the Lanczos3 downsample are deterministic given fixed
    /// input. `MIN_DISTINCT_EDGE_LEVELS` (6) sits strictly above the MSAA-
    /// only range this project measured (1, and up to ~4-5 by the broader
    /// investigation that motivated this feature) while leaving a small
    /// margin below the 7 measured here, so a regression back to MSAA-only
    /// behavior -- or a supersample factor accidentally forced to 1 -- fails
    /// this test, while ordinary cross-GPU/driver rounding differences in
    /// exactly which byte values appear should not.
    #[test]
    fn supersamples_diagonal_primitive_edges_beyond_msaa_alone_on_an_available_gpu() {
        const MIN_DISTINCT_EDGE_LEVELS: usize = 6;

        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "diagonal".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Line {
                    x1: 6.0,
                    y1: 6.0,
                    x2: 58.0,
                    y2: 58.0,
                    thickness: 6.0,
                    fill: FillV1::Solid([0.2, 0.2, 0.2, 1.0]),
                },
            }],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };
        let background_pixel = pixel_at(0, 0);
        let line_pixel = pixel_at(32, 32);
        assert_ne!(
            background_pixel, line_pixel,
            "sanity check: the sampled background and line-interior points must differ"
        );
        let low = background_pixel[0].min(line_pixel[0]);
        let high = background_pixel[0].max(line_pixel[0]);

        let mut distinct_edge_levels = std::collections::BTreeSet::new();
        for y in 0..scene.canvas.height as usize {
            for x in 0..width {
                let red = pixel_at(x, y)[0];
                if red > low && red < high {
                    distinct_edge_levels.insert(red);
                }
            }
        }
        assert!(
            distinct_edge_levels.len() >= MIN_DISTINCT_EDGE_LEVELS,
            "expected at least {MIN_DISTINCT_EDGE_LEVELS} distinct strictly-intermediate \
             red-channel values along the diagonal edge (found {}: {distinct_edge_levels:?}), \
             proving supersampling contributes a genuinely richer gradient than \
             {MSAA_SAMPLE_COUNT}x MSAA alone can produce",
            distinct_edge_levels.len()
        );
    }

    #[test]
    fn applies_a_full_canvas_effect_shader_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        // `test_scene()`'s canvas background is fully transparent black
        // ([0,0,0,0]) and its only node (a red rect) does not cover pixel
        // (0,0), so pre-supersampling this pixel round-tripped 0.0/1.0
        // exactly through the sRGB transfer function used by the
        // `Rgba8UnormSrgb` intermediate texture, making an exact-byte
        // comparison meaningful there. With `SUPERSAMPLE_FACTOR` supersampling
        // now in the pipeline, that no longer holds exactly: the downsample
        // filter (see `SUPERSAMPLE_FACTOR`'s doc comment) has nonzero support
        // beyond a single output texel, so a hard content edge a few source
        // texels away (the rect's edge, still pixel-aligned in the oversized
        // render) blends a sliver of it into an otherwise-background output
        // pixel near it -- exactly the kind of edge softening this crate's
        // `assert_matches_golden` tolerance elsewhere already accounts for
        // at shape/glyph/image edges. `PIXEL_TOLERANCE` absorbs that
        // (measured 5 of 255 here, with the Triangle filter `SUPERSAMPLE_FACTOR`'s
        // downsample currently uses) while still exercising the real GPU
        // pass and catching an actual regression (wrong composition,
        // dropped alpha, effect not applied): RGB channels invert 0 -> 255
        // and alpha (never gamma-corrected) passes through unchanged.
        const PIXEL_TOLERANCE: i16 = 8;
        let assert_pixel_close = |label: &str, actual: &[u8], expected: [u8; 4]| {
            for channel in 0..4 {
                let delta = (actual[channel] as i16 - expected[channel] as i16).abs();
                assert!(
                    delta <= PIXEL_TOLERANCE,
                    "{label}: channel {channel} was {} (expected close to {}, tolerance {PIXEL_TOLERANCE})",
                    actual[channel],
                    expected[channel]
                );
            }
        };

        let scene = test_scene();
        let (without_effect, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());
        assert_pixel_close("without_effect", &without_effect[0..4], [0, 0, 0, 0]);

        let mut scene = scene;
        scene.effect = Some(renderer_schema::EffectV1 {
            shader: "fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32> {\n\
                     return vec4<f32>(1.0 - color.rgb, color.a);\n\
                     }"
            .into(),
        });
        let (with_effect, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());
        assert_pixel_close("with_effect", &with_effect[0..4], [255, 255, 255, 0]);

        assert_ne!(
            without_effect, with_effect,
            "applying the invert effect must change the composited output"
        );
    }

    #[test]
    fn an_invalid_effect_shader_returns_an_error_instead_of_panicking() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let mut scene = test_scene();

        // References an undefined identifier: must fail WGSL compile
        // validation, not panic the process.
        scene.effect = Some(renderer_schema::EffectV1 {
            shader: "fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32> {\n\
                     return this_identifier_does_not_exist;\n\
                     }"
            .into(),
        });
        let result = renderer.render_rgba(&scene);
        assert!(
            matches!(result, Err(RenderError::InvalidEffectShader(_))),
            "expected InvalidEffectShader, got {result:?}"
        );

        // Missing the required `effect` function signature entirely: the
        // template's fragment stage calls `effect(uv, color)`, which will
        // fail to resolve.
        scene.effect = Some(renderer_schema::EffectV1 {
            shader: "fn not_the_right_name(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32> {\n\
                     return color;\n\
                     }"
            .into(),
        });
        let result = renderer.render_rgba(&scene);
        assert!(
            matches!(result, Err(RenderError::InvalidEffectShader(_))),
            "expected InvalidEffectShader, got {result:?}"
        );

        // The renderer (and process) must still be usable afterwards.
        scene.effect = None;
        assert!(renderer.render_rgba(&scene).is_ok());
    }

    /// Golden-image tolerance for `renders_golden_scenes_within_tolerance_on_an_available_gpu`.
    ///
    /// The renderer anti-aliases vector primitives via `MSAA_SAMPLE_COUNT`x
    /// MSAA (see that constant's doc comment), so shape edges are
    /// coverage-weighted blends rather than exact given identical input; the
    /// sources of legitimate, non-bug pixel drift across GPUs/drivers are:
    /// (1) the MSAA resolve itself, where different GPUs/drivers can place
    /// sample points or weight coverage minutely differently along a shape
    /// edge, (2) fontdue's anti-aliased glyph coverage combined with
    /// sRGB-aware alpha blending on `Rgba8UnormSrgb`, where different GPUs
    /// may round the linear<->sRGB conversion by a few least-significant
    /// bits, and (3) bilinear texture sampling when an image is uploaded
    /// below its target size (as in these fixtures) and stretched by the GPU
    /// sampler, whose interpolation weights can differ minutely by hardware.
    /// None of these should ever move a pixel by more than a handful of
    /// 8-bit levels, and none should affect more than a thin sliver of
    /// pixels along shape/glyph/image edges.
    ///
    /// A genuine regression (wrong placement, dropped alpha blending, wrong
    /// composition order) shifts whole regions of the image by large amounts
    /// and/or moves a large fraction of pixels, which these two independent
    /// checks both catch:
    ///   - `GOLDEN_MAX_MISMATCHED_PIXEL_RATIO`: at most 0.75% of pixels may
    ///     differ by more than `GOLDEN_MAX_CHANNEL_DELTA` in any channel.
    ///   - `GOLDEN_MAX_MEAN_CHANNEL_DELTA`: the average per-channel delta
    ///     across the whole image must stay under 1 of 255 levels, which
    ///     bounds cumulative drift even if it were spread thinly.
    const GOLDEN_MAX_CHANNEL_DELTA: u8 = 8;
    const GOLDEN_MAX_MISMATCHED_PIXEL_RATIO: f64 = 0.0075;
    const GOLDEN_MAX_MEAN_CHANNEL_DELTA: f64 = 1.0;

    fn golden_asset_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/golden")
    }

    fn load_golden_scene(name: &str) -> SceneV1 {
        let path = golden_asset_root().join(name);
        let raw = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read fixture {path:?}: {error}"));
        serde_json::from_str(&raw)
            .unwrap_or_else(|error| panic!("failed to parse fixture {path:?}: {error}"))
    }

    /// Asserts `actual` (raw RGBA8 pixels for a `width`x`height` render)
    /// matches the checked-in golden PNG at `golden_path` within the
    /// documented tolerance. See the comment on the `GOLDEN_*` constants
    /// above for the rationale.
    fn assert_matches_golden(
        label: &str,
        actual: &[u8],
        width: u32,
        height: u32,
        golden_path: &Path,
    ) {
        let golden = image::open(golden_path)
            .unwrap_or_else(|error| panic!("failed to open golden image {golden_path:?}: {error}"))
            .to_rgba8();
        assert_eq!(golden.width(), width, "{label}: golden width mismatch");
        assert_eq!(golden.height(), height, "{label}: golden height mismatch");
        let golden = golden.into_raw();
        assert_eq!(
            actual.len(),
            golden.len(),
            "{label}: pixel buffer length mismatch"
        );

        let pixel_count = width as usize * height as usize;
        let mut mismatched_pixels = 0usize;
        let mut sum_abs_delta: u64 = 0;
        let (actual_pixels, _) = actual.as_chunks::<4>();
        let (golden_pixels, _) = golden.as_chunks::<4>();
        for (actual_pixel, golden_pixel) in actual_pixels.iter().zip(golden_pixels.iter()) {
            let mut pixel_mismatched = false;
            for channel in 0..4 {
                let delta = (actual_pixel[channel] as i16 - golden_pixel[channel] as i16)
                    .unsigned_abs() as u8;
                sum_abs_delta += delta as u64;
                if delta > GOLDEN_MAX_CHANNEL_DELTA {
                    pixel_mismatched = true;
                }
            }
            if pixel_mismatched {
                mismatched_pixels += 1;
            }
        }
        let mismatched_ratio = mismatched_pixels as f64 / pixel_count as f64;
        let mean_channel_delta = sum_abs_delta as f64 / (pixel_count as f64 * 4.0);
        assert!(
            mismatched_ratio <= GOLDEN_MAX_MISMATCHED_PIXEL_RATIO,
            "{label}: {mismatched_pixels}/{pixel_count} pixels ({:.3}%) exceeded the \
             per-channel tolerance of {GOLDEN_MAX_CHANNEL_DELTA}; allowed up to {:.3}%",
            mismatched_ratio * 100.0,
            GOLDEN_MAX_MISMATCHED_PIXEL_RATIO * 100.0
        );
        assert!(
            mean_channel_delta <= GOLDEN_MAX_MEAN_CHANNEL_DELTA,
            "{label}: mean per-channel delta {mean_channel_delta:.3} exceeded {GOLDEN_MAX_MEAN_CHANNEL_DELTA}"
        );
    }

    /// Perceptual golden-image coverage for text/image rasterization and
    /// scene composition, per the approved acceptance plan for the local
    /// text/image rasterization increment ("Use deterministic golden images
    /// to verify text/image placement, alpha blend, node ordering, and
    /// PNG/GIF output on a supported GPU host"). Skips gracefully (rather
    /// than failing) on a host with no GPU adapter, mirroring
    /// `renders_png_and_gif_on_an_available_gpu`.
    #[test]
    fn renders_golden_scenes_within_tolerance_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let asset_root = golden_asset_root();

        // Covers text placement, image placement, and alpha blending across
        // overlapping vector, text, and image nodes.
        let cases = [
            (
                "golden_text_image_placement.scene.json",
                "golden_text_image_placement.expected.png",
            ),
            (
                "golden_alpha_blend.scene.json",
                "golden_alpha_blend.expected.png",
            ),
            (
                "golden_order_image_first.scene.json",
                "golden_order_image_first.expected.png",
            ),
            (
                "golden_order_vector_first.scene.json",
                "golden_order_vector_first.expected.png",
            ),
        ];
        for (scene_file, golden_file) in cases {
            let scene = load_golden_scene(scene_file);
            let (pixels, warnings) = renderer
                .render_rgba_with_asset_root(&scene, &asset_root)
                .unwrap_or_else(|error| panic!("failed to render {scene_file}: {error}"));
            assert!(
                warnings.is_empty(),
                "{scene_file}: unexpected warnings: {warnings:?}"
            );
            assert_matches_golden(
                scene_file,
                &pixels,
                scene.canvas.width,
                scene.canvas.height,
                &asset_root.join(golden_file),
            );
        }

        // `golden_order_image_first.scene.json` and
        // `golden_order_vector_first.scene.json` declare the same
        // partially-transparent image and rect nodes in opposite order.
        // Composition is a strict painter's-algorithm pass over declaration
        // order, so swapping the order must change the blended result.
        let image_first = load_golden_scene("golden_order_image_first.scene.json");
        let vector_first = load_golden_scene("golden_order_vector_first.scene.json");
        let (image_first_pixels, _) = renderer
            .render_rgba_with_asset_root(&image_first, &asset_root)
            .unwrap();
        let (vector_first_pixels, _) = renderer
            .render_rgba_with_asset_root(&vector_first, &asset_root)
            .unwrap();
        assert_ne!(
            image_first_pixels, vector_first_pixels,
            "scene-declaration order must affect the composited output"
        );
    }

    /// Golden-image coverage for a scene-level full-canvas WGSL post-process
    /// effect (a color invert), using the same tolerance-and-skip pattern as
    /// `renders_golden_scenes_within_tolerance_on_an_available_gpu`. Fixture
    /// files use the `golden_effect_` prefix to avoid colliding with that
    /// test's fixtures.
    #[test]
    fn renders_a_golden_effect_scene_within_tolerance_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let asset_root = golden_asset_root();
        let scene_file = "golden_effect_invert.scene.json";
        let golden_file = "golden_effect_invert.expected.png";
        let scene = load_golden_scene(scene_file);
        assert!(scene.effect.is_some(), "{scene_file}: expected an effect");
        let (pixels, warnings) = renderer
            .render_rgba_with_asset_root(&scene, &asset_root)
            .unwrap_or_else(|error| panic!("failed to render {scene_file}: {error}"));
        assert!(
            warnings.is_empty(),
            "{scene_file}: unexpected warnings: {warnings:?}"
        );
        assert_matches_golden(
            scene_file,
            &pixels,
            scene.canvas.width,
            scene.canvas.height,
            &asset_root.join(golden_file),
        );
    }

    /// Separate golden-image test (kept out of
    /// `renders_golden_scenes_within_tolerance_on_an_available_gpu`'s
    /// fixture list to avoid conflicting with concurrent edits to that
    /// list) covering an SVG `Image` node composited alongside vector
    /// shapes and text, generated via `examples/generate_golden_svg.rs`.
    /// `tiny-skia`'s software rasterizer is fully deterministic given fixed
    /// input, so this fixture carries none of the GPU-driver-dependent
    /// pixel drift documented on `GOLDEN_MAX_CHANNEL_DELTA` above -- the
    /// same tolerance is reused anyway for consistency with the other
    /// golden tests (and to absorb the sRGB-blend rounding from the vector
    /// rect/text nodes it's composited with).
    #[test]
    fn renders_svg_golden_scene_within_tolerance_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let asset_root = golden_asset_root();
        let scene = load_golden_scene("golden_svg_placement.scene.json");
        let (pixels, warnings) = renderer
            .render_rgba_with_asset_root(&scene, &asset_root)
            .unwrap_or_else(|error| panic!("failed to render golden_svg_placement: {error}"));
        assert!(
            warnings.is_empty(),
            "golden_svg_placement.scene.json: unexpected warnings: {warnings:?}"
        );
        assert_matches_golden(
            "golden_svg_placement.scene.json",
            &pixels,
            scene.canvas.width,
            scene.canvas.height,
            &asset_root.join("golden_svg_placement.expected.png"),
        );
    }

    #[test]
    fn rasterizes_text_images_and_constrained_assets_without_a_gpu() {
        let directory = tempfile::tempdir().unwrap();
        let asset = directory.path().join("asset.png");
        image::RgbaImage::from_pixel(2, 2, image::Rgba([0, 255, 0, 255]))
            .save(&asset)
            .unwrap();
        let mut scene = test_scene();
        scene.canvas.width = 32;
        scene.canvas.height = 32;
        scene.nodes = vec![
            NodeV1 {
                id: "text".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Text {
                    x: 1.0,
                    y: 1.0,
                    text: "AA".into(),
                    size: 12.0,
                    fill: FillV1::Solid([1.0; 4]),
                },
            },
            NodeV1 {
                id: "vector-between".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 16.0,
                    y: 1.0,
                    width: 2.0,
                    height: 2.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0; 4]),
                },
            },
            NodeV1 {
                id: "image".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Image {
                    x: 20.0,
                    y: 20.0,
                    width: 4.0,
                    height: 4.0,
                    source: "asset.png".into(),
                },
            },
            NodeV1 {
                id: "image-again".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Image {
                    x: 24.0,
                    y: 20.0,
                    width: 4.0,
                    height: 4.0,
                    source: "asset.png".into(),
                },
            },
        ];
        let font = Font::from_bytes(
            include_bytes!("../assets/NotoSans-Regular.ttf") as &[u8],
            fontdue::FontSettings::default(),
        )
        .unwrap();
        let mut pixels = vec![0; 32 * 32 * 4];
        rasterize_text_and_images(&mut pixels, &scene, directory.path(), &font).unwrap();
        assert!(pixels.iter().any(|value| *value != 0));

        let plan =
            composition_plan(&scene, directory.path(), &font, &mut AssetCache::default()).unwrap();
        assert_eq!(plan.commands.len(), 5);
        assert_eq!(plan.textures.len(), 2);
        assert!(matches!(&plan.commands[0], DrawCommand::Textured { .. }));
        assert!(matches!(&plan.commands[1], DrawCommand::Textured { .. }));
        assert!(matches!(&plan.commands[2], DrawCommand::Primitive(_)));
        assert!(matches!(&plan.commands[3], DrawCommand::Textured { .. }));
        assert!(matches!(&plan.commands[4], DrawCommand::Textured { .. }));

        let mut primitives = test_scene();
        primitives.nodes.push(NodeV1 {
            id: "second-box".into(),
            translate: [0.0, 0.0],
            kind: NodeKindV1::Rect {
                x: 12.0,
                y: 1.0,
                width: 2.0,
                height: 2.0,
                corner_radius: 0.0,
                fill: FillV1::Solid([1.0; 4]),
            },
        });
        let primitive_plan = composition_plan(
            &primitives,
            directory.path(),
            &font,
            &mut AssetCache::default(),
        )
        .unwrap();
        assert_eq!(primitive_plan.commands.len(), 1);
        assert!(matches!(
            &primitive_plan.commands[0],
            DrawCommand::Primitive(vertices) if vertices == &(0..12)
        ));
        assert_eq!(upload_dimensions(4_096, 1, 1, 1), (1, 1));
        assert_eq!(
            upload_dimensions(4_096, 4_096, 4_096, 4_096),
            (2_048, 2_048)
        );
        assert!(matches!(
            resolve_asset(directory.path(), "../asset.png"),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            resolve_asset(directory.path(), "/asset.png"),
            Err(RenderError::Asset(_))
        ));
        blend_pixel(&mut pixels, 32, 32, -1, 0, [1.0; 4]);

        let mut destination = vec![255, 255, 255, 255];
        // GPU vector layers are read back in premultiplied-alpha form.
        blend_premultiplied_layer(&mut destination, &[128, 0, 0, 128], 1, 1);
        assert_eq!(destination, vec![255, 127, 127, 255]);

        assert!(matches!(
            bounded_image_dimension(f32::MAX, "width"),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            bounded_image_dimension(4_097.0, "width"),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            ensure_source_image_dimensions(4_001, 4_000, "oversized.png"),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            ensure_target_image_dimensions(4_096, 4_096, "oversized.png"),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            reserve_composition_pixels(MAX_COMPOSITION_TEXTURE_PIXELS, 1),
            Err(RenderError::Asset(_))
        ));

        scene.nodes = vec![NodeV1 {
            id: "oversized-text".into(),
            translate: [0.0, 0.0],
            kind: NodeKindV1::Text {
                x: 0.0,
                y: 0.0,
                text: "A".into(),
                size: 2_000.0,
                fill: FillV1::Solid([1.0; 4]),
            },
        }];
        assert!(matches!(
            rasterize_text_and_images(&mut pixels, &scene, directory.path(), &font),
            Err(RenderError::Asset(_))
        ));
    }

    #[test]
    fn rasterizes_svg_assets_to_declared_dimensions_without_a_gpu() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("badge.svg"),
            br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
<rect width="10" height="10" fill="#ff0000"/>
</svg>"##,
        )
        .unwrap();

        // Rasterizing directly at a non-square declared size (rather than
        // decoding at some source resolution and bilinearly rescaling)
        // should still produce an exact widthxheight buffer with crisp,
        // uniform color -- there is nothing to blur since the whole
        // viewBox is one flat rect.
        let image = load_image(directory.path(), "badge.svg", 40, 20).unwrap();
        assert_eq!(image.width(), 40);
        assert_eq!(image.height(), 20);
        for pixel in image.pixels() {
            assert_eq!(pixel.0, [255, 0, 0, 255]);
        }
    }

    #[test]
    fn rejects_svg_assets_that_escape_the_asset_root() {
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("nested");
        fs::create_dir(&nested).unwrap();

        // Same two escape shapes already covered for raster images by the
        // `resolve_asset` assertions in
        // `rasterizes_text_images_and_constrained_assets_without_a_gpu`
        // above: `..` traversal and an absolute path. `load_image` routes
        // every source (SVG included) through the exact same
        // `resolve_asset` call, so both are rejected before the file is
        // ever opened.
        assert!(matches!(
            load_image(&nested, "../escape.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            load_image(directory.path(), "/absolute-escape.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));
    }

    #[test]
    fn rejects_malformed_and_oversized_svg_assets_without_panicking() {
        let directory = tempfile::tempdir().unwrap();

        fs::write(
            directory.path().join("malformed.svg"),
            b"<svg><unterminated",
        )
        .unwrap();
        assert!(matches!(
            load_image(directory.path(), "malformed.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));

        // Named `.svg` but the content sniff should refuse to hand this to
        // the XML parser at all.
        fs::write(
            directory.path().join("not-svg.svg"),
            b"this has an .svg extension but is not svg content",
        )
        .unwrap();
        assert!(matches!(
            load_image(directory.path(), "not-svg.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));

        // Same 16 MiB cap the raster (`image` crate) path already enforces
        // via `MAX_ASSET_BYTES`, applied before the file is ever parsed.
        let mut oversized = b"<svg xmlns=\"http://www.w3.org/2000/svg\">".to_vec();
        oversized.resize(17 * 1024 * 1024, b' ');
        oversized.extend_from_slice(b"</svg>");
        fs::write(directory.path().join("oversized.svg"), &oversized).unwrap();
        assert!(matches!(
            load_image(directory.path(), "oversized.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));
    }

    #[test]
    fn refuses_to_follow_embedded_svg_image_references_outside_the_asset_root() {
        let directory = tempfile::tempdir().unwrap();

        // A file outside the configured asset root that a hostile SVG will
        // try to pull in via an absolute-path `<image href>`.
        let secret = directory.path().join("secret.png");
        image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]))
            .save(&secret)
            .unwrap();

        let asset_root = directory.path().join("assets");
        fs::create_dir(&asset_root).unwrap();
        fs::write(
            asset_root.join("evil.svg"),
            format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 4 4">
<rect width="4" height="4" fill="#00ff00"/>
<image xlink:href="{}" width="4" height="4"/>
</svg>"##,
                secret.display()
            ),
        )
        .unwrap();

        let image = load_image(&asset_root, "evil.svg", 4, 4).unwrap();
        // The embedded absolute-path href must be refused outright (see the
        // security-posture comment on `rasterize_svg`): only the green
        // background rect should ever be visible, never the referenced
        // file's red pixels.
        for pixel in image.pixels() {
            assert_eq!(pixel.0, [0, 255, 0, 255]);
        }
    }

    /// Portable (no GPU required) proof that `AssetCache` actually caches:
    /// this directly drives `composition_plan` the same way
    /// `render_gif_with_asset_root` does for each frame of a GIF export --
    /// one call per animation frame, `scene_at`-ing the base scene for each
    /// frame's timestamp -- and shows the decode/rasterize call count for a
    /// static image and static glyphs is exactly 1 per distinct asset when a
    /// single `AssetCache` is shared across frames, versus 1 *per frame*
    /// (the pre-change baseline) when each call gets its own fresh cache, as
    /// `composition_plan` built locally before this change.
    #[test]
    fn shares_a_decoded_asset_cache_across_simulated_gif_frames() {
        let directory = tempfile::tempdir().unwrap();
        image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]))
            .save(directory.path().join("logo.png"))
            .unwrap();
        let font = Font::from_bytes(
            include_bytes!("../assets/NotoSans-Regular.ttf") as &[u8],
            fontdue::FontSettings::default(),
        )
        .unwrap();

        let mut scene = test_scene();
        scene.canvas.width = 32;
        scene.canvas.height = 32;
        scene.nodes = vec![
            // Keyframed: this is the only thing that differs frame to frame.
            NodeV1 {
                id: "animated-box".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 1.0,
                    y: 1.0,
                    width: 4.0,
                    height: 4.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0, 0.0, 0.0, 1.0]),
                },
            },
            // Static across every frame: no keyframe targets it.
            NodeV1 {
                id: "label".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Text {
                    x: 2.0,
                    y: 10.0,
                    text: "AB".into(),
                    size: 12.0,
                    fill: FillV1::Solid([1.0, 1.0, 1.0, 1.0]),
                },
            },
            // Static across every frame: no keyframe targets it.
            NodeV1 {
                id: "logo".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Image {
                    x: 20.0,
                    y: 20.0,
                    width: 4.0,
                    height: 4.0,
                    source: "logo.png".into(),
                },
            },
        ];
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 5,
            duration_ms: 1_000,
            keyframes: vec![
                KeyframeV1 {
                    at_ms: 0,
                    target: "animated-box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(0.0),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "animated-box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(1.0),
                },
            ],
        });
        scene.validate().unwrap();

        let timeline = scene.timeline.as_ref().unwrap();
        let frame_count =
            (u64::from(timeline.duration_ms) * u64::from(timeline.fps)).div_ceil(1_000) as u32;
        let fps = u32::from(timeline.fps);
        assert_eq!(frame_count, 5, "sanity check on the fixture's frame math");

        // Baseline: mirrors `composition_plan`'s behavior before this
        // change, where every call got its own fresh, function-local cache
        // -- so every frame independently decodes the static image and
        // rasterizes the static glyphs.
        for frame_index in 0..frame_count {
            let at_ms = frame_index * 1_000 / fps;
            let animated = scene_at(&scene, at_ms);
            let mut fresh_cache = AssetCache::default();
            composition_plan(&animated, directory.path(), &font, &mut fresh_cache).unwrap();
            assert_eq!(
                fresh_cache.image_decode_count(),
                1,
                "a fresh per-frame cache decodes the static image once per frame"
            );
            assert_eq!(
                fresh_cache.glyph_rasterization_count(),
                2,
                "a fresh per-frame cache rasterizes both static glyphs ('A' and 'B') once per frame"
            );
        }

        // Under test: one `AssetCache` shared across every frame -- exactly
        // what `render_gif_with_asset_root` now does -- must decode/
        // rasterize each distinct static asset exactly once for the *whole*
        // multi-frame export, not once per frame.
        let mut shared_cache = AssetCache::default();
        for frame_index in 0..frame_count {
            let at_ms = frame_index * 1_000 / fps;
            let animated = scene_at(&scene, at_ms);
            composition_plan(&animated, directory.path(), &font, &mut shared_cache).unwrap();
        }
        assert_eq!(
            shared_cache.image_decode_count(),
            1,
            "the static image must be decoded exactly once across all {frame_count} frames \
             sharing one AssetCache, not once per frame"
        );
        assert_eq!(
            shared_cache.glyph_rasterization_count(),
            2,
            "each of the 2 distinct static glyphs must be rasterized exactly once across all \
             {frame_count} frames sharing one AssetCache, not once per frame"
        );
    }

    /// GPU smoke coverage (skips gracefully with no adapter, matching this
    /// file's other `_on_an_available_gpu` tests) for the same scenario as
    /// `shares_a_decoded_asset_cache_across_simulated_gif_frames` above, but
    /// driven through the real public `render_gif_with_asset_root` entry
    /// point end to end: a multi-frame export with a keyframed node plus
    /// static image/text nodes must still produce a correct, valid,
    /// warning-free GIF now that the decode/rasterize cache is shared across
    /// frames.
    #[test]
    fn renders_a_gif_with_static_assets_across_frames_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let directory = tempfile::tempdir().unwrap();
        image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]))
            .save(directory.path().join("logo.png"))
            .unwrap();

        let mut scene = test_scene();
        scene.canvas.width = 32;
        scene.canvas.height = 32;
        scene.nodes = vec![
            NodeV1 {
                id: "animated-box".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 1.0,
                    y: 1.0,
                    width: 4.0,
                    height: 4.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0, 0.0, 0.0, 1.0]),
                },
            },
            NodeV1 {
                id: "label".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Text {
                    x: 2.0,
                    y: 10.0,
                    text: "AB".into(),
                    size: 12.0,
                    fill: FillV1::Solid([1.0, 1.0, 1.0, 1.0]),
                },
            },
            NodeV1 {
                id: "logo".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Image {
                    x: 20.0,
                    y: 20.0,
                    width: 4.0,
                    height: 4.0,
                    source: "logo.png".into(),
                },
            },
        ];
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 5,
            duration_ms: 1_000,
            keyframes: vec![
                KeyframeV1 {
                    at_ms: 0,
                    target: "animated-box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(0.0),
                },
                KeyframeV1 {
                    at_ms: 1_000,
                    target: "animated-box".into(),
                    property: renderer_schema::AnimatedPropertyV1::Opacity(1.0),
                },
            ],
        });

        let output = directory.path().join("scene.gif");
        let rendered = renderer
            .render_gif_with_asset_root(&scene, &output, directory.path())
            .unwrap();
        assert_eq!(rendered.frame_count, 5);
        assert!(rendered.warnings.is_empty());
        assert!(
            fs::metadata(&output).unwrap().len() > 0,
            "GIF output must not be empty"
        );
        let decoded = image::open(&output).unwrap_or_else(|error| {
            panic!("render_gif_with_asset_root did not produce a valid, decodable GIF: {error}")
        });
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 32);
    }

    // ---- Analytic (SDF + fwidth) anti-aliasing "shadow mode" tests ----
    //
    // New, separately-named tests only: none of these touch, modify, or
    // regenerate any existing golden image or existing test above, and none
    // of the existing tests above were changed to make room for these.

    /// Basic functional coverage of the analytic-AA path across every node
    /// kind this renderer supports, including the documented `Path` MSAA
    /// fallback (see `composition_plan_analytic`'s doc comment) and an image
    /// asset. Mirrors the spirit of `compiles_all_geometry_and_reports_
    /// unrasterized_nodes` (portable) and `renders_png_and_gif_on_an_
    /// available_gpu` (GPU) above, but actually renders through the new
    /// `_analytic_aa` entry points end to end.
    #[test]
    fn analytic_aa_renders_every_node_kind_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let directory = tempfile::tempdir().unwrap();
        image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]))
            .save(directory.path().join("logo.png"))
            .unwrap();

        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![
                NodeV1 {
                    id: "rect".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Rect {
                        x: 2.0,
                        y: 2.0,
                        width: 10.0,
                        height: 10.0,
                        corner_radius: 0.0,
                        fill: FillV1::Solid([1.0, 0.0, 0.0, 1.0]),
                    },
                },
                NodeV1 {
                    id: "ellipse".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Ellipse {
                        cx: 30.0,
                        cy: 10.0,
                        rx: 6.0,
                        ry: 4.0,
                        fill: FillV1::Solid([0.0, 1.0, 0.0, 1.0]),
                    },
                },
                NodeV1 {
                    id: "line".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Line {
                        x1: 4.0,
                        y1: 30.0,
                        x2: 40.0,
                        y2: 50.0,
                        thickness: 3.0,
                        fill: FillV1::Solid([0.0, 0.0, 1.0, 1.0]),
                    },
                },
                NodeV1 {
                    id: "path".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Path {
                        points: vec![
                            renderer_schema::PointV1 { x: 45.0, y: 5.0 },
                            renderer_schema::PointV1 { x: 60.0, y: 5.0 },
                            renderer_schema::PointV1 { x: 52.0, y: 20.0 },
                        ],
                        fill: FillV1::Solid([0.5, 0.0, 0.5, 1.0]),
                    },
                },
                NodeV1 {
                    id: "text".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Text {
                        x: 4.0,
                        y: 44.0,
                        text: "Hi".into(),
                        size: 12.0,
                        fill: FillV1::Solid([0.0, 0.0, 0.0, 1.0]),
                    },
                },
                NodeV1 {
                    id: "logo".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Image {
                        x: 44.0,
                        y: 44.0,
                        width: 8.0,
                        height: 8.0,
                        source: "logo.png".into(),
                    },
                },
            ],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer
            .render_rgba_with_asset_root_analytic_aa(&scene, directory.path())
            .unwrap();
        assert!(warnings.is_empty());
        assert_eq!(pixels.len(), 64 * 64 * 4);

        // Every non-background color used above must appear somewhere in the
        // output (allowing for AA blending, so this checks for "close to"
        // rather than an exact byte match) -- proof each node kind actually
        // rasterized something instead of silently no-opping.
        let expects = [
            ("rect (red)", [255u8, 0, 0]),
            ("ellipse (green)", [0, 255, 0]),
            ("line (blue)", [0, 0, 255]),
            // Not a naive linear-to-255 mapping (0.5*255=128): this
            // renderer's output is sRGB-gamma-encoded, and gamma-encoding
            // linear 0.5 gives ~0.735, i.e. ~188/255 -- confirmed against
            // the actual rendered pixel value. Every other entry in this
            // list happens to use a pure 0.0/1.0 channel value, which gamma
            // encoding leaves unchanged, so this is the only one affected.
            ("path (purple)", [188, 0, 188]),
            ("image (logo)", [10, 20, 30]),
        ];
        for (label, target) in expects {
            let found = pixels.as_chunks::<4>().0.iter().any(|pixel| {
                (0..3).all(|channel| (pixel[channel] as i16 - target[channel] as i16).abs() <= 12)
            });
            assert!(
                found,
                "{label}: expected a pixel close to {target:?} in the analytic-AA render"
            );
        }

        // A round trip through `render_png_with_asset_root_analytic_aa`
        // works end to end too.
        let png = renderer
            .render_png_with_asset_root_analytic_aa(
                &scene,
                &directory.path().join("out.png"),
                directory.path(),
            )
            .unwrap();
        assert_eq!(png.width, 64);
        assert_eq!(png.height, 64);
        assert_eq!(png.frame_count, 1);
    }

    /// Analytic-AA counterpart to `anti_aliases_diagonal_primitive_edges_
    /// on_an_available_gpu` above: same diagonal-line scene, same "at least
    /// one pixel strictly between background and line color" check, but
    /// rendered through `render_rgba_analytic_aa` instead of `render_rgba`.
    /// Confirms the SDF/`fwidth` pipeline produces real coverage-weighted
    /// blending, not a hard binary edge, exactly like the existing MSAA path
    /// -- from an entirely separate pipeline/shader.
    #[test]
    fn analytic_aa_anti_aliases_diagonal_primitive_edges_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = diagonal_line_scene();
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba_analytic_aa(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };
        let background_pixel = pixel_at(0, 0);
        let line_pixel = pixel_at(32, 32);
        assert_ne!(background_pixel, line_pixel);

        let mut found_blended_pixel = false;
        'scan: for y in 0..scene.canvas.height as usize {
            for x in 0..width {
                let pixel = pixel_at(x, y);
                let strictly_between = (0..3).all(|channel| {
                    let low = background_pixel[channel].min(line_pixel[channel]);
                    let high = background_pixel[channel].max(line_pixel[channel]);
                    pixel[channel] > low && pixel[channel] < high
                });
                if strictly_between {
                    found_blended_pixel = true;
                    break 'scan;
                }
            }
        }
        assert!(
            found_blended_pixel,
            "expected at least one analytically anti-aliased pixel strictly between the \
             background color {background_pixel:?} and the line color {line_pixel:?}"
        );
    }

    /// Confirms `Path` nodes still render via the documented MSAA fallback
    /// (see `composition_plan_analytic`'s doc comment) *and* that draw order
    /// across heterogeneous pipelines is preserved: a `Path` node sandwiched
    /// between two analytic-pipeline `Rect` nodes must still composite in
    /// document order (later nodes drawn on top of earlier ones), even
    /// though the multi-pass design (`render_composed_rgba_analytic_with_
    /// cache`) executes the `Path` node in a separate, differently-sampled
    /// render pass from its analytic-pipeline neighbors.
    #[test]
    fn analytic_aa_preserves_z_order_across_the_path_msaa_fallback_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 32,
                height: 32,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![
                // Bottom: a big opaque red square covering the whole canvas.
                NodeV1 {
                    id: "background-rect".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Rect {
                        x: 0.0,
                        y: 0.0,
                        width: 32.0,
                        height: 32.0,
                        corner_radius: 0.0,
                        fill: FillV1::Solid([1.0, 0.0, 0.0, 1.0]),
                    },
                },
                // Middle: an opaque green path covering the whole canvas --
                // must fully occlude the red rect beneath it.
                NodeV1 {
                    id: "middle-path".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Path {
                        points: vec![
                            renderer_schema::PointV1 { x: 0.0, y: 0.0 },
                            renderer_schema::PointV1 { x: 32.0, y: 0.0 },
                            renderer_schema::PointV1 { x: 32.0, y: 32.0 },
                            renderer_schema::PointV1 { x: 0.0, y: 32.0 },
                        ],
                        fill: FillV1::Solid([0.0, 1.0, 0.0, 1.0]),
                    },
                },
                // Top: a small opaque blue square -- must occlude the green
                // path beneath it, proving the *next* analytic-pipeline pass
                // still loads (not clears) the prior Path pass's output.
                NodeV1 {
                    id: "top-rect".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Rect {
                        x: 8.0,
                        y: 8.0,
                        width: 8.0,
                        height: 8.0,
                        corner_radius: 0.0,
                        fill: FillV1::Solid([0.0, 0.0, 1.0, 1.0]),
                    },
                },
            ],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba_analytic_aa(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };
        // Far from every edge, so no AA blending is in play: a corner (only
        // the green path should be visible -- proving it occluded the red
        // rect) and the small blue square's center (proving it occluded the
        // green path).
        let corner = pixel_at(2, 2);
        let top = pixel_at(12, 12);
        assert!(
            corner[1] > 200 && corner[0] < 40 && corner[2] < 40,
            "expected the green path to occlude the red rect beneath it at a corner far from \
             any edge, got {corner:?}"
        );
        assert!(
            top[2] > 200 && top[0] < 40 && top[1] < 40,
            "expected the blue rect to occlude the green path beneath it at its center, got {top:?}"
        );
    }

    /// GIF export through the analytic-AA path works end to end, mirroring
    /// `renders_png_and_gif_on_an_available_gpu` above but via `render_gif_
    /// analytic_aa`.
    #[test]
    fn analytic_aa_renders_gif_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let directory = tempfile::tempdir().unwrap();
        let mut scene = test_scene();
        scene.timeline = Some(renderer_schema::TimelineV1 {
            fps: 2,
            duration_ms: 1_000,
            keyframes: vec![],
        });
        let gif = renderer
            .render_gif_analytic_aa(&scene, &directory.path().join("scene.gif"))
            .unwrap();
        assert_eq!(gif.frame_count, 2);
        assert!(gif.warnings.is_empty());
        let decoded = image::open(directory.path().join("scene.gif")).unwrap_or_else(|error| {
            panic!("render_gif_analytic_aa did not produce a valid, decodable GIF: {error}")
        });
        assert_eq!(decoded.width(), scene.canvas.width);
        assert_eq!(decoded.height(), scene.canvas.height);
    }

    /// The diagonal-line scene shared by the MSAA/SSAA and analytic-AA
    /// anti-aliasing tests/quality comparison, factored out so both sides of
    /// the comparison render *exactly* the same geometry.
    fn diagonal_line_scene() -> SceneV1 {
        SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "diagonal".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Line {
                    x1: 6.0,
                    y1: 6.0,
                    x2: 58.0,
                    y2: 58.0,
                    thickness: 6.0,
                    fill: FillV1::Solid([0.2, 0.2, 0.2, 1.0]),
                },
            }],
            timeline: None,
            effect: None,
        }
    }

    /// A shallow-angle (not 45°) diagonal line, otherwise matching
    /// `diagonal_line_scene`'s style (same canvas size, thickness, color).
    /// Used only by `analytic_aa_matches_numeric_ground_truth_coverage_
    /// better_than_msaa_supersampling_on_an_available_gpu`, which needs to
    /// sample real *rendered pixels* (necessarily at integer positions)
    /// spanning several distinct true-coverage bands. A 45° line's AA
    /// transition band is only ~1.4px wide in x (each 1px step in x moves
    /// ~0.7px perpendicular to the edge, and the AA band itself is only
    /// about 1px wide), too narrow to contain 5 well-separated integer-pixel
    /// samples -- confirmed numerically: scanning `diagonal_line_scene`'s
    /// 45° line finds only 1 of 5 target coverage bands at integer
    /// resolution, no matter how wide a window is scanned. A shallow slope
    /// spreads that same true 1px-wide transition band across many more
    /// integer x steps, making it actually samplable. Kept as its own
    /// fixture (rather than changing `diagonal_line_scene` itself) so the
    /// pre-existing tests already calibrated against the 45° line are
    /// untouched.
    fn shallow_diagonal_line_scene() -> SceneV1 {
        SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 64,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "shallow-diagonal".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Line {
                    x1: 4.0,
                    y1: 20.0,
                    x2: 60.0,
                    y2: 30.0,
                    thickness: 6.0,
                    fill: FillV1::Solid([0.2, 0.2, 0.2, 1.0]),
                },
            }],
            timeline: None,
            effect: None,
        }
    }

    /// Numerically estimates the *true* fractional pixel-area coverage of a
    /// capsule (2D thick line segment, matching `capsule_sdf` in
    /// `ANALYTIC_SHADER` and `add_line_analytic`'s geometry) at raster pixel
    /// `(pixel_x, pixel_y)`, by regularly subsampling that pixel's
    /// continuous `[pixel_x, pixel_x+1) x [pixel_y, pixel_y+1)` region (in
    /// the same scene-pixel coordinate space `vertex()`/`add_line` use --
    /// scene x/y coordinates map 1:1 onto continuous framebuffer pixel
    /// coordinates, since `vertex()`'s `x / canvas.width * 2 - 1` clip-space
    /// transform is exactly the inverse of the standard NDC-to-viewport
    /// transform) and computing what fraction of subsample points the exact
    /// capsule SDF (no shader approximation, no `fwidth`) classifies as
    /// inside.
    ///
    /// This is deliberately independent of *both* renderers under test: it
    /// does not call `capsule_sdf`/`ANALYTIC_SHADER` (the analytic path's
    /// own formula) or rely on MSAA/supersampling in any way, so comparing
    /// each renderer's actual output against this number is a fair,
    /// non-circular ground-truth check for both.
    /// Standard sRGB EOTF (byte 0-255 -> linear 0.0-1.0), matching the
    /// `Rgba8UnormSrgb` render target format this renderer uses throughout.
    fn srgb_decode_byte(byte: f32) -> f32 {
        let c = byte / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Inverse of `srgb_decode_byte` (linear 0.0-1.0 -> byte 0-255).
    fn srgb_encode_byte(linear: f32) -> f32 {
        let c = if linear <= 0.003_130_8 {
            linear * 12.92
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        c * 255.0
    }

    fn linear_mix(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }

    fn capsule_coverage_numeric(
        pixel_x: i32,
        pixel_y: i32,
        a: [f32; 2],
        b: [f32; 2],
        half_thickness: f32,
        subsamples: u32,
    ) -> f32 {
        let ba = [b[0] - a[0], b[1] - a[1]];
        let ba_len_sq = ba[0] * ba[0] + ba[1] * ba[1];
        let mut inside = 0u32;
        for row in 0..subsamples {
            for column in 0..subsamples {
                let sx = pixel_x as f32 + (column as f32 + 0.5) / subsamples as f32;
                let sy = pixel_y as f32 + (row as f32 + 0.5) / subsamples as f32;
                let pa = [sx - a[0], sy - a[1]];
                let h = ((pa[0] * ba[0] + pa[1] * ba[1]) / ba_len_sq).clamp(0.0, 1.0);
                let dx = pa[0] - ba[0] * h;
                let dy = pa[1] - ba[1] * h;
                let distance = (dx * dx + dy * dy).sqrt() - half_thickness;
                if distance < 0.0 {
                    inside += 1;
                }
            }
        }
        inside as f32 / (subsamples * subsamples) as f32
    }

    /// The core quality-comparison test: renders the exact same diagonal-
    /// line scene (`diagonal_line_scene`) through both the existing
    /// MSAA+supersampling path (`render_rgba`, completely untouched by this
    /// change) and the new analytic SDF/`fwidth` path
    /// (`render_rgba_analytic_aa`), then checks each renderer's output
    /// against a numerically-estimated *ground-truth* coverage
    /// (`capsule_coverage_numeric`, 64x64 subsamples per pixel -- computed
    /// from the exact capsule geometry, independent of either renderer's own
    /// internals) at several sample pixels spanning a range of true coverage
    /// fractions along the line's edge.
    ///
    /// A coverage fraction is turned into an "expected" byte value by
    /// decoding the scene's actual rendered pure background/line-interior
    /// colors from sRGB to linear, interpolating *there*, then re-encoding
    /// (`srgb_decode_byte`/`srgb_encode_byte`/`linear_mix` below) -- not
    /// naive byte-space interpolation (which the simpler pre-existing
    /// `anti_aliases_diagonal_primitive_edges_on_an_available_gpu`/
    /// `supersamples_diagonal_primitive_edges_beyond_msaa_alone_on_an_
    /// available_gpu` tests above use, since they only check "is there any
    /// blending at all", a check loose enough not to care). This one
    /// computes real numeric error against ground truth, and the render
    /// target is `Rgba8UnormSrgb` -- the GPU blends in linear space -- so
    /// byte-space interpolation was measured to disagree with real output
    /// by up to ~23 (of 255) at mid-range coverage, large enough to make
    /// this comparison meaningless without the gamma-correct version.
    ///
    /// Thresholds have real headroom above/below what's actually measured
    /// on this machine's real GPU (Apple M1/Metal) with this scene: analytic
    /// mean absolute error ~4.3 (max ~9.9), MSAA+supersampling mean ~7.3
    /// (max ~16.9) -- so ordinary cross-GPU/driver rounding differences
    /// shouldn't make this flaky, while a real regression in either path's
    /// edge quality, or the two techniques becoming indistinguishable,
    /// still fails it. See `shallow_diagonal_line_scene`'s doc comment for
    /// why this test uses a shallow-angle line rather than
    /// `diagonal_line_scene`'s 45° one.
    #[test]
    fn analytic_aa_matches_numeric_ground_truth_coverage_better_than_msaa_supersampling_on_an_available_gpu()
     {
        const MAX_MEAN_ANALYTIC_ERROR: f32 = 7.0;
        const MIN_MEAN_MSAA_ERROR: f32 = 5.0;

        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = shallow_diagonal_line_scene();
        scene.validate().unwrap();

        let (msaa_pixels, msaa_warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(msaa_warnings.is_empty());
        let (analytic_pixels, analytic_warnings) =
            renderer.render_rgba_analytic_aa(&scene).unwrap();
        assert!(analytic_warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };
        // Pure endpoint colors, measured from the actual GPU output (same
        // approach the pre-existing anti-aliasing tests above use) rather
        // than assumed from the scene's linear input color, since this
        // renderer's actual color-space handling is an internal detail
        // neither this test nor the pre-existing ones above depend on.
        // (32, 25) is deep in the shallow line's interior -- its centerline
        // passes through y=25 at x=32 -- and far from either rounded cap.
        let background_byte = pixel_at(&msaa_pixels, 0, 0)[0] as f32;
        let line_byte = pixel_at(&msaa_pixels, 32, 25)[0] as f32;
        assert_eq!(
            pixel_at(&analytic_pixels, 0, 0)[0] as f32,
            background_byte,
            "both paths must render the exact same flat background color away from any edge"
        );
        assert!(
            (pixel_at(&analytic_pixels, 32, 25)[0] as f32 - line_byte).abs() <= 2.0,
            "both paths must render essentially the same line-interior color far from any edge"
        );

        let line_a = [4.0_f32, 20.0];
        let line_b = [60.0_f32, 30.0];
        let half_thickness = 3.0_f32;

        let mut analytic_errors = Vec::new();
        let mut msaa_errors = Vec::new();
        let mut sampled_coverages = Vec::new();
        // Scan a window straddling the diagonal edge (inset from both the
        // canvas border and the line's rounded end caps) and keep pixels
        // whose true numeric coverage lands in a set of well-separated
        // bands, so the sampled set spans a real range of coverage
        // fractions rather than clustering near one value.
        let mut remaining_bands: Vec<(f32, f32)> =
            vec![(0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95)];
        'scan: for y in 4..40usize {
            for x in 10..56usize {
                let coverage = capsule_coverage_numeric(
                    x as i32,
                    y as i32,
                    line_a,
                    line_b,
                    half_thickness,
                    64,
                );
                if let Some(band_index) = remaining_bands
                    .iter()
                    .position(|(low, high)| coverage >= *low && coverage < *high)
                {
                    // NOT naive byte-space linear interpolation: the render
                    // target is `Rgba8UnormSrgb`, so the GPU blends
                    // `coverage`-weighted colors in *linear* space and then
                    // gamma-encodes the result for storage. Byte-space
                    // interpolation between `background_byte`/`line_byte`
                    // follows a visibly different curve (most divergent
                    // around 40-60% coverage -- confirmed empirically: it
                    // was off by 9-22 bytes here before this fix), so the
                    // "expected" value must decode both endpoints to linear,
                    // interpolate there, then re-encode.
                    let expected = srgb_encode_byte(linear_mix(
                        srgb_decode_byte(background_byte),
                        srgb_decode_byte(line_byte),
                        coverage,
                    ));
                    let analytic_actual = pixel_at(&analytic_pixels, x, y)[0] as f32;
                    let msaa_actual = pixel_at(&msaa_pixels, x, y)[0] as f32;
                    analytic_errors.push((analytic_actual - expected).abs());
                    msaa_errors.push((msaa_actual - expected).abs());
                    sampled_coverages.push(coverage);
                    remaining_bands.remove(band_index);
                    if remaining_bands.is_empty() {
                        break 'scan;
                    }
                }
            }
        }
        assert!(
            sampled_coverages.len() >= 4,
            "expected to find pixels spanning most of the coverage-fraction bands near the \
             diagonal edge (found {} of 5: {sampled_coverages:?})",
            sampled_coverages.len()
        );

        let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
        let analytic_mean_error = mean(&analytic_errors);
        let msaa_mean_error = mean(&msaa_errors);

        assert!(
            analytic_mean_error <= MAX_MEAN_ANALYTIC_ERROR,
            "analytic-AA mean absolute byte error against numeric ground-truth coverage was \
             {analytic_mean_error}, expected <= {MAX_MEAN_ANALYTIC_ERROR} \
             (per-sample errors: {analytic_errors:?}, coverages: {sampled_coverages:?})"
        );
        assert!(
            msaa_mean_error >= analytic_mean_error,
            "expected the analytic-AA path's mean absolute error against numeric ground truth \
             ({analytic_mean_error}) to be no worse than the existing MSAA+supersampling path's \
             ({msaa_mean_error}) on identical geometry -- analytic AA should match ground truth \
             at least as well since it computes an exact per-fragment distance instead of \
             estimating coverage from a fixed sample grid"
        );
        // Loosely confirms MSAA+supersampling's error is in the range this
        // project has already measured for it (see this test's doc comment)
        // rather than accidentally testing two near-identical numbers.
        assert!(
            msaa_mean_error >= MIN_MEAN_MSAA_ERROR,
            "expected the existing MSAA+supersampling path's mean absolute error ({msaa_mean_error}) \
             to be at least {MIN_MEAN_MSAA_ERROR} on this scene, matching this project's prior \
             measurements (see this test's doc comment) -- if not, this comparison may no longer \
             be meaningfully distinguishing the two techniques"
        );
    }

    /// Portable (no-GPU) proof that a zero (or omitted) `corner_radius`
    /// leaves `add_rect`'s output byte-identical to the plain 2-triangle
    /// quad it always emitted before rounded corners existed. Reconstructs
    /// that original tessellation by hand (the same `vertex()` calls in the
    /// same `[a, b, c, a, c, d]` order `add_rect`'s original implementation
    /// used) and compares every emitted `Vertex`'s position and color
    /// field-by-field against `test_scene()`'s zero-radius rect.
    #[test]
    fn rounded_rect_with_zero_radius_matches_the_original_plain_rect_tessellation() {
        let scene = test_scene();
        let NodeKindV1::Rect {
            x,
            y,
            width,
            height,
            corner_radius,
            fill,
        } = &scene.nodes[0].kind
        else {
            panic!("test_scene()'s only node must be a Rect");
        };
        assert_eq!(*corner_radius, 0.0);
        let color = fill.resolve_solid();

        let (actual, _) = vertices_for_scene(&scene);
        let a = vertex(*x, *y, color, &scene);
        let b = vertex(*x + *width, *y, color, &scene);
        let c = vertex(*x + *width, *y + *height, color, &scene);
        let d = vertex(*x, *y + *height, color, &scene);
        let expected = [a, b, c, a, c, d];

        assert_eq!(actual.len(), expected.len());
        for (index, (found, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                found.position, want.position,
                "vertex {index} position mismatch"
            );
            assert_eq!(found.color, want.color, "vertex {index} color mismatch");
        }
    }

    /// Portable (no-GPU) proof that a positive `corner_radius` actually
    /// changes the emitted geometry (more triangles than the flat 2-triangle
    /// quad, since corners are now tessellated as arcs) rather than being
    /// silently ignored by the vertex builder.
    #[test]
    fn rounded_rect_emits_more_triangles_than_a_plain_rect() {
        let mut scene = test_scene();
        let (plain, _) = vertices_for_scene(&scene);

        let NodeKindV1::Rect { corner_radius, .. } = &mut scene.nodes[0].kind else {
            panic!("test_scene()'s only node must be a Rect");
        };
        *corner_radius = 3.0;
        scene.validate().unwrap();
        let (rounded, _) = vertices_for_scene(&scene);

        assert_eq!(
            plain.len(),
            6,
            "a plain (zero-radius) rect is always 2 triangles"
        );
        assert!(
            rounded.len() > plain.len(),
            "expected a rounded rect to tessellate into more triangles ({}) than a plain \
             rect ({})",
            rounded.len(),
            plain.len()
        );
    }

    /// GPU pixel-decode proof (default MSAA pipeline) that `corner_radius`
    /// actually rounds a rect's corners rather than merely not crashing:
    /// renders the same bounding box twice, once sharp (`corner_radius:
    /// 0.0`) and once rounded (`corner_radius: 10.0`), and confirms a pixel
    /// near the bounding box's corner is shape-colored in the sharp render
    /// but background-colored in the rounded render.
    #[test]
    fn rounds_rect_corners_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };

        fn scene_with_radius(corner_radius: f32) -> SceneV1 {
            SceneV1 {
                version: SCENE_VERSION_V1.into(),
                canvas: CanvasV1 {
                    width: 40,
                    height: 40,
                    background: [1.0, 1.0, 1.0, 1.0],
                },
                nodes: vec![NodeV1 {
                    id: "box".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Rect {
                        x: 4.0,
                        y: 4.0,
                        width: 32.0,
                        height: 32.0,
                        corner_radius,
                        fill: FillV1::Solid([0.0, 0.0, 0.0, 1.0]),
                    },
                }],
                timeline: None,
                effect: None,
            }
        }

        let width = 40_usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let sharp = scene_with_radius(0.0);
        sharp.validate().unwrap();
        let (sharp_pixels, sharp_warnings) = renderer.render_rgba(&sharp).unwrap();
        assert!(sharp_warnings.is_empty());

        let rounded = scene_with_radius(10.0);
        rounded.validate().unwrap();
        let (rounded_pixels, rounded_warnings) = renderer.render_rgba(&rounded).unwrap();
        assert!(rounded_warnings.is_empty());

        // (5, 5) sits just inside the shared 4..36 bounding box, near its
        // top-left corner: distance to the top-left arc's center (14, 14)
        // at radius 10 is ~12.7px, i.e. outside the rounded arc but well
        // inside the sharp box.
        let background = pixel_at(&sharp_pixels, 0, 0);
        let sharp_corner = pixel_at(&sharp_pixels, 5, 5);
        let rounded_corner = pixel_at(&rounded_pixels, 5, 5);

        assert_ne!(
            sharp_corner, background,
            "sanity check: a sharp rect's bounding-box corner must be shape-colored"
        );
        assert_eq!(
            rounded_corner, background,
            "expected a corner_radius: 10.0 rect's corner pixel to be background-colored \
             (proving the corner is actually rounded away), got {rounded_corner:?} vs. \
             background {background:?}"
        );
    }

    /// Analytic-AA counterpart to `rounds_rect_corners_on_an_available_gpu`:
    /// same proof, but through `render_rgba_analytic_aa`, confirming the
    /// analytic pipeline's `rect_sdf`/`param1`-carried radius also actually
    /// rounds corners rather than silently ignoring `corner_radius`.
    #[test]
    fn rounds_rect_corners_under_analytic_aa_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };

        fn scene_with_radius(corner_radius: f32) -> SceneV1 {
            SceneV1 {
                version: SCENE_VERSION_V1.into(),
                canvas: CanvasV1 {
                    width: 40,
                    height: 40,
                    background: [1.0, 1.0, 1.0, 1.0],
                },
                nodes: vec![NodeV1 {
                    id: "box".into(),
                    translate: [0.0, 0.0],
                    kind: NodeKindV1::Rect {
                        x: 4.0,
                        y: 4.0,
                        width: 32.0,
                        height: 32.0,
                        corner_radius,
                        fill: FillV1::Solid([0.0, 0.0, 0.0, 1.0]),
                    },
                }],
                timeline: None,
                effect: None,
            }
        }

        let width = 40_usize;
        let pixel_at = |pixels: &[u8], x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let sharp = scene_with_radius(0.0);
        sharp.validate().unwrap();
        let (sharp_pixels, sharp_warnings) = renderer.render_rgba_analytic_aa(&sharp).unwrap();
        assert!(sharp_warnings.is_empty());

        let rounded = scene_with_radius(10.0);
        rounded.validate().unwrap();
        let (rounded_pixels, rounded_warnings) =
            renderer.render_rgba_analytic_aa(&rounded).unwrap();
        assert!(rounded_warnings.is_empty());

        let background = pixel_at(&sharp_pixels, 0, 0);
        let sharp_corner = pixel_at(&sharp_pixels, 5, 5);
        let rounded_corner = pixel_at(&rounded_pixels, 5, 5);

        assert_ne!(
            sharp_corner, background,
            "sanity check: a sharp rect's bounding-box corner must be shape-colored"
        );
        assert_eq!(
            rounded_corner, background,
            "expected a corner_radius: 10.0 rect's corner pixel to be background-colored under \
             analytic AA too, got {rounded_corner:?} vs. background {background:?}"
        );
    }

    /// GPU pixel-decode proof (default MSAA pipeline) that a linear gradient
    /// fill produces real per-pixel interpolation: renders a wide rect with
    /// a pure-red-to-pure-blue horizontal gradient and confirms pixels near
    /// the left edge read close to red, pixels near the right edge read
    /// close to blue, and a pixel at the horizontal midpoint is a genuine
    /// intermediate blend (neither near-red nor near-blue) -- not one flat
    /// color and not a hard cut partway across.
    #[test]
    fn linear_gradient_fill_interpolates_between_stops_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 16,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "gradient".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 64.0,
                    height: 16.0,
                    corner_radius: 0.0,
                    fill: FillV1::Gradient(GradientV1::LinearGradient {
                        from: [1.0, 0.0, 0.0, 1.0],
                        to: [0.0, 0.0, 1.0, 1.0],
                        angle_degrees: 0.0,
                    }),
                },
            }],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let near_left = pixel_at(1, 8);
        let near_right = pixel_at(62, 8);
        let middle = pixel_at(32, 8);

        assert!(
            near_left[0] > 180 && near_left[2] < 80,
            "expected the pixel near the gradient's `from` edge to read close to pure red, \
             got {near_left:?}"
        );
        assert!(
            near_right[2] > 180 && near_right[0] < 80,
            "expected the pixel near the gradient's `to` edge to read close to pure blue, \
             got {near_right:?}"
        );
        assert!(
            middle[0] > 40 && middle[0] < 215 && middle[2] > 40 && middle[2] < 215,
            "expected the midpoint pixel to be a genuine intermediate red/blue blend (neither \
             near-red nor near-blue), got {middle:?}"
        );
    }

    /// Analytic-AA counterpart to
    /// `linear_gradient_fill_interpolates_between_stops_on_an_available_gpu`:
    /// same scene and same proof, but through `render_rgba_analytic_aa`,
    /// confirming `add_rect_analytic` also resolves each vertex's color via
    /// `fill_vertex_color` rather than silently collapsing a gradient `fill`
    /// to one flat color.
    #[test]
    fn linear_gradient_fill_interpolates_under_analytic_aa_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 64,
                height: 16,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "gradient".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 64.0,
                    height: 16.0,
                    corner_radius: 0.0,
                    fill: FillV1::Gradient(GradientV1::LinearGradient {
                        from: [1.0, 0.0, 0.0, 1.0],
                        to: [0.0, 0.0, 1.0, 1.0],
                        angle_degrees: 0.0,
                    }),
                },
            }],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba_analytic_aa(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let near_left = pixel_at(1, 8);
        let near_right = pixel_at(62, 8);
        let middle = pixel_at(32, 8);

        assert!(
            near_left[0] > 180 && near_left[2] < 80,
            "expected the pixel near the gradient's `from` edge to read close to pure red under \
             analytic AA, got {near_left:?}"
        );
        assert!(
            near_right[2] > 180 && near_right[0] < 80,
            "expected the pixel near the gradient's `to` edge to read close to pure blue under \
             analytic AA, got {near_right:?}"
        );
        assert!(
            middle[0] > 40 && middle[0] < 215 && middle[2] > 40 && middle[2] < 215,
            "expected the midpoint pixel to be a genuine intermediate red/blue blend under \
             analytic AA, got {middle:?}"
        );
    }

    /// GPU pixel-decode proof that a radial gradient on an `Ellipse` also
    /// interpolates for real: `center` at the ellipse's middle, `edge` at
    /// its boundary, and a genuine intermediate blend partway out.
    #[test]
    fn radial_gradient_fill_interpolates_between_stops_on_an_available_gpu() {
        let renderer = match GpuRenderer::new() {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("GPU renderer unavailable during this test: {error}");
                return;
            }
        };
        let scene = SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 40,
                height: 40,
                background: [1.0, 1.0, 1.0, 1.0],
            },
            nodes: vec![NodeV1 {
                id: "gradient".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Ellipse {
                    cx: 20.0,
                    cy: 20.0,
                    rx: 18.0,
                    ry: 18.0,
                    fill: FillV1::Gradient(GradientV1::RadialGradient {
                        center: [1.0, 0.0, 0.0, 1.0],
                        edge: [0.0, 0.0, 1.0, 1.0],
                    }),
                },
            }],
            timeline: None,
            effect: None,
        };
        scene.validate().unwrap();
        let (pixels, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());

        let width = scene.canvas.width as usize;
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let index = (y * width + x) * 4;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        };

        let at_center = pixel_at(20, 20);
        let near_edge = pixel_at(20, 3);
        let partway = pixel_at(20, 11);

        assert!(
            at_center[0] > 180 && at_center[2] < 80,
            "expected the ellipse's center pixel to read close to the radial gradient's \
             `center` color (pure red), got {at_center:?}"
        );
        assert!(
            near_edge[2] > 180 && near_edge[0] < 80,
            "expected a pixel near the ellipse's boundary to read close to the radial \
             gradient's `edge` color (pure blue), got {near_edge:?}"
        );
        assert!(
            partway[0] > 40 && partway[0] < 215 && partway[2] > 40 && partway[2] < 215,
            "expected a pixel partway between the ellipse's center and edge to be a genuine \
             intermediate blend, got {partway:?}"
        );
    }

    fn test_scene() -> SceneV1 {
        SceneV1 {
            version: SCENE_VERSION_V1.into(),
            canvas: CanvasV1 {
                width: 32,
                height: 32,
                background: [0.0; 4],
            },
            nodes: vec![NodeV1 {
                id: "box".into(),
                translate: [0.0, 0.0],
                kind: NodeKindV1::Rect {
                    x: 1.0,
                    y: 1.0,
                    width: 10.0,
                    height: 10.0,
                    corner_radius: 0.0,
                    fill: FillV1::Solid([1.0, 0.0, 0.0, 1.0]),
                },
            }],
            timeline: None,
            effect: None,
        }
    }
}
