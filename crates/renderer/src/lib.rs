//! Off-screen wgpu renderer for normalized RendererCli scenes.
#![allow(unexpected_cfgs)] // `cargo llvm-cov` supplies `cfg(coverage)`/`cfg(coverage_nightly)`.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use bytemuck::{Pod, Zeroable};
use fontdue::Font;
use image::{
    Delay, Frame, RgbaImage,
    codecs::gif::{GifEncoder, Repeat},
};
use renderer_schema::{Color, KeyframeV1, NodeKindV1, SceneV1};
use resvg::{tiny_skia, usvg};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use thiserror::Error;
use wgpu::util::DeviceExt;

const ELLIPSE_SEGMENTS: usize = 32;
/// MSAA sample count used to anti-alias vector primitives (rect/ellipse/
/// line/path). 4x is the standard, broadly-supported choice for
/// `Rgba8UnormSrgb` render targets on desktop GPUs (Metal/Vulkan/DX12) and is
/// what this renderer's node-composition pass uses; see
/// `GpuRenderer::new_async` for an adapter-capability check confirming this
/// value is supported before it is relied on. The effect pass intentionally
/// stays single-sampled (see `build_effect_pipeline`): it introduces no new
/// geometric edges, so multisampling it would add cost with no benefit.
const MSAA_SAMPLE_COUNT: u32 = 4;
const MAX_ASSET_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ASSET_PIXELS: u64 = 16_000_000;
const MAX_IMAGE_RASTER_PIXELS: u64 = 4_000_000;
const MAX_TEXT_BYTES: usize = 16 * 1024;
const MAX_TEXT_GLYPHS: usize = 1_024;
const MAX_GLYPH_SIZE: f32 = 1_024.0;
const MAX_TEXT_RASTER_PIXELS: u64 = 4_000_000;
const MAX_COMPOSITION_TEXTURE_PIXELS: u64 = 20_000_000;
const MAX_GPU_TEXTURE_DIMENSION: u32 = 2_048;
/// Wall-clock ceiling for parsing+rasterizing a single SVG asset. `usvg`
/// already refuses documents with more than 1,000,000 XML nodes
/// (`usvg::Error::ElementsLimitReached`, which also bounds `<use>`-expansion
/// style blowups since expansion copies count against the same limit), but
/// pathological filter chains (e.g. many chained `feGaussianBlur`s) can still
/// be expensive without tripping that counter. This budget turns a hang into
/// a fast, actionable `RenderError::Asset` instead.
const SVG_RASTER_TIME_BUDGET: Duration = Duration::from_secs(5);

#[derive(Debug)]
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    font: Font,
    primitive_pipeline: wgpu::RenderPipeline,
    textured_pipeline: wgpu::RenderPipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderedImage {
    pub width: u32,
    pub height: u32,
    pub sha256: String,
    pub frame_count: u32,
    pub warnings: Vec<String>,
}

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("scene validation failed: {0}")]
    InvalidScene(#[from] renderer_schema::SceneValidationError),
    #[error("no compatible GPU adapter was found")]
    NoAdapter,
    #[error("could not create GPU device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("GPU readback failed")]
    Readback,
    #[error("could not write image: {0}")]
    Image(#[from] image::ImageError),
    #[error("PNG render output must use a .png extension: {0}")]
    InvalidPngOutputPath(PathBuf),
    #[error("could not create output directory: {0}")]
    OutputDirectory(#[source] std::io::Error),
    #[error("could not write GIF: {0}")]
    Gif(#[source] image::ImageError),
    #[error("could not hash emitted output: {0}")]
    OutputRead(#[source] std::io::Error),
    #[error("could not load bundled font")]
    Font,
    #[error("asset error: {0}")]
    Asset(String),
    /// The scene's `effect.shader` failed WGSL compile validation once
    /// wrapped in the fixed post-process template. Kept at the end of this
    /// enum to minimize merge-conflict risk with parallel changes elsewhere.
    #[error("invalid effect shader: {0}")]
    InvalidEffectShader(String),
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
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("renderer-cli device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
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
        Ok(Self {
            device,
            queue,
            font,
            primitive_pipeline,
            textured_pipeline,
            texture_bind_group_layout,
            sampler,
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
        {
            let mut encoder =
                GifEncoder::new(File::create(output).map_err(RenderError::OutputDirectory)?);
            encoder
                .set_repeat(Repeat::Infinite)
                .map_err(RenderError::Gif)?;
            for frame_index in 0..frame_count {
                let at_ms = frame_index * 1_000 / fps;
                let animated = scene_at(scene, at_ms);
                animated.validate()?;
                let (pixels, frame_warnings) =
                    self.render_composed_rgba_with_cache(&animated, asset_root, &mut cache)?;
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
        let plan = composition_plan(scene, asset_root, &self.font, cache)?;
        let width = scene.canvas.width;
        let height = scene.canvas.height;
        // When a scene-level effect is present, nodes are composited into
        // this texture as an *intermediate* (sampled, not read back) and a
        // second full-screen pass below writes the final, effect-applied
        // pixels elsewhere. With no effect, this texture is the one and only
        // render target and is read back directly, exactly as before this
        // feature existed: no extra texture or pass is allocated.
        let has_effect = scene.effect.is_some();
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
        // Multisampled intermediate the node-composition pass below actually
        // draws into; it is resolved into `view` (the single-sampled
        // `texture` above) at the end of that pass, which is what performs
        // the anti-aliasing (see `MSAA_SAMPLE_COUNT`'s doc comment). A
        // multisampled texture can only ever be a resolve source, so its
        // usage is restricted to `RENDER_ATTACHMENT` -- it can't be
        // `COPY_SRC` or `TEXTURE_BINDING` -- and it's never read back or
        // sampled directly; `texture`/`view` keep meaning exactly what they
        // meant before this texture existed.
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
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("renderer-cli readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("renderer-cli commands"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("renderer-cli pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &msaa_view,
                    resolve_target: Some(&view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(to_wgpu_color(scene.canvas.background)),
                        // The multisampled contents themselves are never
                        // read -- only the resolve into `view` matters --
                        // so they don't need to be stored.
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
        // This keeps the no-effect path's allocation and pass count
        // unchanged (see `has_effect` above).
        let effect_texture;
        let final_texture = if let Some(effect) = &scene.effect {
            let pipeline = self.build_effect_pipeline(&effect.shader)?;
            let created = self.device.create_texture(&wgpu::TextureDescriptor {
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
            let effect_view = created.create_view(&wgpu::TextureViewDescriptor::default());
            let effect_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("renderer-cli effect pass"),
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
                // Full-screen triangle: the vertex shader derives clip-space
                // position and UV from `vertex_index` alone, so no vertex
                // buffer is bound here.
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
        Ok((pixels, Vec::new()))
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: Color,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TexturedVertex {
    position: [f32; 2],
    uv: [f32; 2],
    tint: Color,
}

impl TexturedVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
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

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
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
    match &node.kind {
        NodeKindV1::Rect {
            x,
            y,
            width,
            height,
            color,
        } => {
            add_rect(vertices, *x, *y, *width, *height, *color, scene);
        }
        NodeKindV1::Ellipse {
            cx,
            cy,
            rx,
            ry,
            color,
        } => {
            add_ellipse(vertices, *cx, *cy, *rx, *ry, *color, scene);
        }
        NodeKindV1::Line {
            x1,
            y1,
            x2,
            y2,
            thickness,
            color,
        } => {
            add_line(vertices, [*x1, *y1], [*x2, *y2], *thickness, *color, scene);
        }
        NodeKindV1::Path { points, color } => add_path(vertices, points, *color, scene),
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
                color,
            } => {
                validate_text_raster(node.id.as_str(), text, *size, scene)?;
                let mut cursor_x = *x;
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
                        *y + (*size - metrics.height as f32 - metrics.ymin as f32),
                        metrics.width as f32,
                        metrics.height as f32,
                        *color,
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
                    *x,
                    *y,
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

/// Extension-based SVG detection, mirroring [`ensure_png_output_path`]'s
/// case-insensitive extension check.
fn has_svg_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("svg"))
}

/// Cheap content sniff so a `.svg`-named file that is not actually SVG (or is
/// empty/binary garbage) fails with a clear diagnostic instead of being
/// handed to the XML parser. Mirrors the spirit of `image`'s own
/// magic-byte format guessing for raster assets.
fn looks_like_svg(bytes: &[u8]) -> bool {
    let prefix_len = bytes.len().min(4096);
    let Ok(prefix) = std::str::from_utf8(&bytes[..prefix_len]) else {
        return false;
    };
    let trimmed = prefix.trim_start_matches('\u{feff}').trim_start();
    trimmed.starts_with("<?xml") || trimmed.starts_with("<svg") || trimmed.contains("<svg")
}

/// Rasterizes an SVG document to an `image::RgbaImage` of exactly
/// `width`x`height` pixels.
///
/// Security posture (see `AGENTS.md`: "Resolve assets locally only; do not
/// introduce implicit remote asset fetching"):
///
/// - `usvg` never performs network I/O of any kind (confirmed by reading the
///   `usvg` 0.48 source: `ImageHrefResolver`'s doc comment states it plainly,
///   and there is no HTTP client anywhere in its dependency tree).
/// - The only way an SVG can reach outside this call is via `<image
///   xlink:href="...">` (or a `<style>`/font-family reference, neither of
///   which `usvg` resolves from the filesystem at all). `usvg`'s *default*
///   string-href resolver treats the href as a filesystem path relative to
///   `Options::resources_dir` -- but critically, `PathBuf::join` treats an
///   *absolute* href as replacing the base entirely, so a default-configured
///   `resources_dir` does NOT stop `<image href="/etc/passwd">` (or a `..`
///   traversal) from escaping the asset root.
///
///   To close that off completely rather than merely "scope it", the
///   `resolve_string` resolver below is replaced with one that returns
///   `None` unconditionally: embedded `<image href="...">` references to
///   *any* local file path are refused, full stop. Only self-contained
///   `data:` URIs (handled by `resolve_data`, which never touches the
///   filesystem) are honored for embedded images. This is strictly more
///   restrictive than scoping to the asset root, so there is no residual
///   path-escape risk from embedded image hrefs.
/// - `fontdb` is left empty (the `system-fonts` cargo feature is disabled and
///   `load_system_fonts()` is never called), so text glyph lookups cannot
///   read arbitrary font files from the host either; SVGs with `<text>` will
///   render without glyphs rather than pulling in system state.
/// - Residual risk: `usvg` cannot be configured to refuse XML parsing
///   entirely (that is the whole point of this function), and a
///   sufficiently adversarial-but-under-the-node-limit document (e.g. many
///   chained blur filters) could still be CPU-expensive to rasterize. That
///   residual is bounded by `SVG_RASTER_TIME_BUDGET` below, on top of
///   `usvg`'s own 1,000,000-node parse limit and the existing
///   `MAX_ASSET_BYTES`/`MAX_IMAGE_RASTER_PIXELS`-derived output-size caps
///   this function's caller already enforces.
fn rasterize_svg(
    data: &[u8],
    asset_root: &Path,
    source: &str,
    width: u32,
    height: u32,
) -> Result<image::RgbaImage, RenderError> {
    let data = Arc::new(data.to_vec());
    let asset_root = asset_root.to_path_buf();
    let source_label = source.to_string();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = rasterize_svg_blocking(&data, &asset_root, &source_label, width, height);
        // The receiver may already be gone if we hit the timeout below; that
        // is fine, the render result is simply dropped.
        let _ = sender.send(result);
    });
    match receiver.recv_timeout(SVG_RASTER_TIME_BUDGET) {
        Ok(result) => result,
        Err(_) => Err(RenderError::Asset(format!(
            "svg '{source}' exceeded the {}s rasterization time budget",
            SVG_RASTER_TIME_BUDGET.as_secs()
        ))),
    }
}

fn rasterize_svg_blocking(
    data: &[u8],
    asset_root: &Path,
    source: &str,
    width: u32,
    height: u32,
) -> Result<image::RgbaImage, RenderError> {
    let image_href_resolver = usvg::ImageHrefResolver {
        resolve_data: usvg::ImageHrefResolver::default_data_resolver(),
        resolve_string: Box::new(|_href: &str, _options: &usvg::Options| {
            // Deliberately refuse every filesystem-path-shaped `href`: see
            // the security-posture comment on `rasterize_svg` above.
            None
        }),
    };
    // `..Default::default()` leaves `fontdb` at `usvg::Options::default()`'s
    // empty `fontdb::Database`: with the `system-fonts` cargo feature
    // disabled and `load_system_fonts()` never called, no host font files
    // are ever read (see the security-posture comment above).
    let options = usvg::Options {
        resources_dir: Some(asset_root.to_path_buf()),
        image_href_resolver,
        ..Default::default()
    };
    let tree = usvg::Tree::from_data(data, &options)
        .map_err(|error| RenderError::Asset(format!("could not parse svg '{source}': {error}")))?;

    let mut pixmap = tiny_skia::Pixmap::new(width, height).ok_or_else(|| {
        RenderError::Asset(format!(
            "svg '{source}' has an invalid raster target size {width}x{height}"
        ))
    })?;
    let tree_size = tree.size();
    let scale_x = if tree_size.width() > 0.0 {
        width as f32 / tree_size.width()
    } else {
        1.0
    };
    let scale_y = if tree_size.height() > 0.0 {
        height as f32 / tree_size.height()
    } else {
        1.0
    };
    let transform = tiny_skia::Transform::from_scale(scale_x, scale_y);
    resvg::render(&tree, transform, &mut pixmap.as_mut());

    // `Pixmap` stores premultiplied alpha internally; the rest of this
    // renderer's textured-quad pipeline (and the `image` crate decode path
    // above) works in straight alpha, so demultiply on the way out.
    let rgba = pixmap.take_demultiplied();
    image::RgbaImage::from_raw(width, height, rgba).ok_or_else(|| {
        RenderError::Asset(format!(
            "failed to assemble rasterized buffer for svg '{source}'"
        ))
    })
}

fn add_rect(
    vertices: &mut Vec<Vertex>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    color: Color,
    scene: &SceneV1,
) {
    let a = vertex(x, y, color, scene);
    let b = vertex(x + width, y, color, scene);
    let c = vertex(x + width, y + height, color, scene);
    let d = vertex(x, y + height, color, scene);
    vertices.extend([a, b, c, a, c, d]);
}

fn add_ellipse(
    vertices: &mut Vec<Vertex>,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    color: Color,
    scene: &SceneV1,
) {
    let center = vertex(cx, cy, color, scene);
    for index in 0..ELLIPSE_SEGMENTS {
        let start = std::f32::consts::TAU * index as f32 / ELLIPSE_SEGMENTS as f32;
        let end = std::f32::consts::TAU * (index + 1) as f32 / ELLIPSE_SEGMENTS as f32;
        vertices.extend([
            center,
            vertex(cx + rx * start.cos(), cy + ry * start.sin(), color, scene),
            vertex(cx + rx * end.cos(), cy + ry * end.sin(), color, scene),
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

fn to_wgpu_color(color: Color) -> wgpu::Color {
    wgpu::Color {
        r: color[0] as f64,
        g: color[1] as f64,
        b: color[2] as f64,
        a: color[3] as f64,
    }
}

fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn hash_file(path: &Path) -> Result<String, RenderError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(fs::read(path).map_err(RenderError::OutputRead)?)
    ))
}

fn ensure_png_output_path(output: &Path) -> Result<(), RenderError> {
    if output
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
    {
        Ok(())
    } else {
        Err(RenderError::InvalidPngOutputPath(output.into()))
    }
}

fn scene_at(scene: &SceneV1, at_ms: u32) -> SceneV1 {
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
        if let (Some(color), Some(value)) = (
            color_mut(&mut node.kind),
            interpolate_color(&color_keyframes, at_ms),
        ) {
            *color = value;
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
        if let (Some(color), Some(opacity)) = (
            color_mut(&mut node.kind),
            interpolate_opacity(&opacity_keyframes, at_ms),
        ) {
            color[3] *= opacity;
        }
    }
    output
}

fn color_mut(kind: &mut NodeKindV1) -> Option<&mut Color> {
    match kind {
        NodeKindV1::Rect { color, .. }
        | NodeKindV1::Ellipse { color, .. }
        | NodeKindV1::Line { color, .. }
        | NodeKindV1::Path { color, .. }
        | NodeKindV1::Text { color, .. } => Some(color),
        NodeKindV1::Image { .. } => None,
    }
}

#[cfg(test)]
fn color_of(kind: &NodeKindV1) -> Option<Color> {
    match kind {
        NodeKindV1::Rect { color, .. }
        | NodeKindV1::Ellipse { color, .. }
        | NodeKindV1::Line { color, .. }
        | NodeKindV1::Path { color, .. }
        | NodeKindV1::Text { color, .. } => Some(*color),
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

fn interpolate<T: Copy>(
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
                color,
            } => {
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

const PRIMITIVE_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@location(0) position: vec2<f32>, @location(1) color: vec4<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.color = color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

const TEXTURED_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
}

@group(0) @binding(0) var image_texture: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tint: vec4<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = uv;
    output.tint = tint;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(image_texture, image_sampler, input.uv) * input.tint;
}
"#;

const EFFECT_VERTEX_ENTRY_POINT: &str = "renderer_cli_effect_vs";
const EFFECT_FRAGMENT_ENTRY_POINT: &str = "renderer_cli_effect_fs";

/// Wraps a scene author's WGSL `effect` function in the fixed post-process
/// template: a full-screen-triangle vertex stage (no vertex buffer; the
/// triangle covers the viewport and is derived purely from
/// `@builtin(vertex_index)`) and a fragment stage that samples the
/// already-composited scene texture, calls `effect(uv, color)`, and writes
/// the result. The user-supplied text is inserted verbatim between the
/// fixed vertex stage and fixed fragment stage; the entry points use
/// distinctive names so they cannot collide with anything a scene author's
/// `effect` function defines.
///
/// This wrapping is the entire security boundary for scene-level effects:
/// authors only ever author a pure per-pixel color transform and never see
/// or control bind group layouts, vertex data, or texture access.
fn wrap_effect_shader(user_shader: &str) -> String {
    format!(
        r#"
struct RendererCliEffectVaryings {{
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}}

@group(0) @binding(0) var renderer_cli_effect_texture: texture_2d<f32>;
@group(0) @binding(1) var renderer_cli_effect_sampler: sampler;

@vertex
fn {EFFECT_VERTEX_ENTRY_POINT}(@builtin(vertex_index) vertex_index: u32) -> RendererCliEffectVaryings {{
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[vertex_index];
    var output: RendererCliEffectVaryings;
    output.clip_position = vec4<f32>(position, 0.0, 1.0);
    output.uv = vec2<f32>(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
    return output;
}}

// ---- begin scene-author effect shader (untrusted; pure color transform only) ----
{user_shader}
// ---- end scene-author effect shader ----

@fragment
fn {EFFECT_FRAGMENT_ENTRY_POINT}(input: RendererCliEffectVaryings) -> @location(0) vec4<f32> {{
    let sampled = textureSample(renderer_cli_effect_texture, renderer_cli_effect_sampler, input.uv);
    return effect(input.uv, sampled);
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use renderer_schema::{CanvasV1, NodeV1, SCENE_VERSION_V1};

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
                kind: NodeKindV1::Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 10.0,
                    height: 10.0,
                    color: [1.0; 4],
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
                kind: NodeKindV1::Ellipse {
                    cx: 20.0,
                    cy: 20.0,
                    rx: 5.0,
                    ry: 5.0,
                    color: [0.0, 1.0, 0.0, 1.0],
                },
            },
            NodeV1 {
                id: "line".into(),
                kind: NodeKindV1::Line {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 20.0,
                    y2: 20.0,
                    thickness: 2.0,
                    color: [0.0, 0.0, 1.0, 1.0],
                },
            },
            NodeV1 {
                id: "path".into(),
                kind: NodeKindV1::Path {
                    points: vec![
                        renderer_schema::PointV1 { x: 0.0, y: 0.0 },
                        renderer_schema::PointV1 { x: 10.0, y: 0.0 },
                        renderer_schema::PointV1 { x: 0.0, y: 10.0 },
                    ],
                    color: [1.0; 4],
                },
            },
            NodeV1 {
                id: "text".into(),
                kind: NodeKindV1::Text {
                    x: 0.0,
                    y: 0.0,
                    text: "t".into(),
                    size: 8.0,
                    color: [1.0; 4],
                },
            },
            NodeV1 {
                id: "vector".into(),
                kind: NodeKindV1::Rect {
                    x: 16.0,
                    y: 1.0,
                    width: 2.0,
                    height: 2.0,
                    color: [1.0; 4],
                },
            },
            NodeV1 {
                id: "image".into(),
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
            color_of(&at_middle.nodes[0].kind),
            Some([0.5, 0.5, 0.5, 0.5])
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
            color_of(&at_middle.nodes[0].kind),
            Some([1.0, 0.0, 0.0, 0.25])
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
            kind: NodeKindV1::Line {
                x1: 1.0,
                y1: 1.0,
                x2: 1.0,
                y2: 1.0,
                thickness: 1.0,
                color: [1.0; 4],
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
                color: [1.0; 4],
            },
            NodeKindV1::Line {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
                thickness: 1.0,
                color: [1.0; 4],
            },
            NodeKindV1::Path {
                points: vec![renderer_schema::PointV1 { x: 0.0, y: 0.0 }; 3],
                color: [1.0; 4],
            },
            NodeKindV1::Text {
                x: 0.0,
                y: 0.0,
                text: "x".into(),
                size: 1.0,
                color: [1.0; 4],
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
            let _ = color_of(&kind);
            let _ = color_mut(&mut kind);
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
                kind: NodeKindV1::Line {
                    x1: 6.0,
                    y1: 6.0,
                    x2: 58.0,
                    y2: 58.0,
                    thickness: 6.0,
                    color: [0.2, 0.2, 0.2, 1.0],
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
        // (0,0). 0.0 and 1.0 round-trip exactly through the sRGB transfer
        // function used by the `Rgba8UnormSrgb` intermediate texture, so an
        // exact-byte comparison at that pixel is meaningful (not sensitive
        // to sRGB rounding) while still exercising the real GPU pass: RGB
        // channels invert 0 -> 255 and alpha (never gamma-corrected) passes
        // through unchanged.
        let scene = test_scene();
        let (without_effect, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(&without_effect[0..4], &[0, 0, 0, 0]);

        let mut scene = scene;
        scene.effect = Some(renderer_schema::EffectV1 {
            shader: "fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32> {\n\
                     return vec4<f32>(1.0 - color.rgb, color.a);\n\
                     }"
            .into(),
        });
        let (with_effect, warnings) = renderer.render_rgba(&scene).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(&with_effect[0..4], &[255, 255, 255, 0]);

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
                kind: NodeKindV1::Text {
                    x: 1.0,
                    y: 1.0,
                    text: "AA".into(),
                    size: 12.0,
                    color: [1.0; 4],
                },
            },
            NodeV1 {
                id: "vector-between".into(),
                kind: NodeKindV1::Rect {
                    x: 16.0,
                    y: 1.0,
                    width: 2.0,
                    height: 2.0,
                    color: [1.0; 4],
                },
            },
            NodeV1 {
                id: "image".into(),
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
            kind: NodeKindV1::Rect {
                x: 12.0,
                y: 1.0,
                width: 2.0,
                height: 2.0,
                color: [1.0; 4],
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
            kind: NodeKindV1::Text {
                x: 0.0,
                y: 0.0,
                text: "A".into(),
                size: 2_000.0,
                color: [1.0; 4],
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
                kind: NodeKindV1::Rect {
                    x: 1.0,
                    y: 1.0,
                    width: 4.0,
                    height: 4.0,
                    color: [1.0, 0.0, 0.0, 1.0],
                },
            },
            // Static across every frame: no keyframe targets it.
            NodeV1 {
                id: "label".into(),
                kind: NodeKindV1::Text {
                    x: 2.0,
                    y: 10.0,
                    text: "AB".into(),
                    size: 12.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                },
            },
            // Static across every frame: no keyframe targets it.
            NodeV1 {
                id: "logo".into(),
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
                kind: NodeKindV1::Rect {
                    x: 1.0,
                    y: 1.0,
                    width: 4.0,
                    height: 4.0,
                    color: [1.0, 0.0, 0.0, 1.0],
                },
            },
            NodeV1 {
                id: "label".into(),
                kind: NodeKindV1::Text {
                    x: 2.0,
                    y: 10.0,
                    text: "AB".into(),
                    size: 12.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                },
            },
            NodeV1 {
                id: "logo".into(),
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
                kind: NodeKindV1::Rect {
                    x: 1.0,
                    y: 1.0,
                    width: 10.0,
                    height: 10.0,
                    color: [1.0, 0.0, 0.0, 1.0],
                },
            }],
            timeline: None,
            effect: None,
        }
    }
}
