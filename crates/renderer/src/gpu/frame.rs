use crate::{
    GpuRenderer, MSAA_SAMPLE_COUNT, SUPERSAMPLE_FACTOR, assets::AssetCache, composition::*,
    error::RenderError, pipelines::effect_color_target, shaders::*, util::*,
};
use image::RgbaImage;
use renderer_schema::{EffectV1, SceneV1};
use std::{path::Path, sync::mpsc};
use wgpu::util::DeviceExt;

/// The GPU render targets one call to [`GpuRenderer::render_frame_with_targets`]
/// composites a single frame into: everything sized only by the scene's
/// declared canvas dimensions and whether it has a post-process `effect`
/// (both scene-level, not animatable), never by a specific frame's node
/// content. Built once by [`GpuRenderer::create_frame_targets`] and reused
/// across every frame of a GIF export -- rather than allocated and torn down
/// `frame_count` times -- since the composite/MSAA textures and the readback
/// buffer are the largest, most expensive-to-allocate resources in the whole
/// render path.
pub(crate) struct FrameTargets {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    // Never read again after `msaa_view` is created from it below, but must
    // stay alive for as long as `msaa_view` does -- kept as a named field
    // (rather than let it drop at the end of `create_frame_targets`) purely
    // for that ownership, not to be used directly.
    #[allow(dead_code)]
    pub(crate) msaa_texture: wgpu::Texture,
    pub(crate) msaa_view: wgpu::TextureView,
    /// Two readback buffers, not one, ping-ponged across frames by
    /// `GpuRenderer::record_frame`/`finish_frame`: a GIF export can submit
    /// frame N+1's GPU work (into the buffer frame N *isn't* using) while
    /// frame N's async `map_async` readback is still pending, instead of
    /// fully blocking the CPU on each frame before starting the next one's.
    /// The single-shot PNG path always uses index 0 and never pipelines --
    /// there's only one frame, nothing to overlap.
    pub(crate) output_buffers: [wgpu::Buffer; 2],
    pub(crate) effect: Option<EffectTargets>,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// The extra GPU state a scene's post-process `effect` needs, held on
/// [`FrameTargets`] only when one is present: the compiled shader pipeline
/// (shader compilation is one of the more expensive one-time GPU driver
/// calls, so this alone is worth hoisting out of a GIF's per-frame loop) plus
/// its own target texture/view and the bind group sampling `FrameTargets`'s
/// main composite `view`.
pub(crate) struct EffectTargets {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) bind_group: wgpu::BindGroup,
}

/// One frame's GPU work, submitted by [`GpuRenderer::record_frame`] but not
/// yet waited on: the readback its `output_buffer` argument is mid-`map_async`
/// for, plus everything [`GpuRenderer::finish_frame`] needs to turn that
/// readback into declared-size pixels once it's ready (sizing, since the
/// downsample step needs both the oversized and declared dimensions).
pub(crate) struct PendingFrame {
    pub(crate) receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    pub(crate) unpadded_bytes_per_row: u32,
    pub(crate) padded_bytes_per_row: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) declared_width: u32,
    pub(crate) declared_height: u32,
}

impl GpuRenderer {
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn render_composed_rgba(
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
    pub(crate) fn render_composed_rgba_with_cache(
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
    pub(crate) fn create_frame_targets(
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
    pub(crate) fn record_frame(
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
    pub(crate) fn finish_frame(
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
    pub(crate) fn build_effect_pipeline(
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
