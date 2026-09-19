use crate::{
    GpuRenderer, MSAA_SAMPLE_COUNT,
    animation::scene_at,
    assets::AssetCache,
    composition::*,
    error::{RenderError, RenderedImage},
    util::*,
};
use image::{
    Delay, Frame, RgbaImage,
    codecs::gif::{GifEncoder, Repeat},
};
use renderer_schema::SceneV1;
use std::{fs, fs::File, path::Path};
use wgpu::util::DeviceExt;

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
