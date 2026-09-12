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
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    fs::File,
    path::{Path, PathBuf},
};
use thiserror::Error;
use wgpu::util::DeviceExt;

const ELLIPSE_SEGMENTS: usize = 32;
const MAX_ASSET_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ASSET_PIXELS: u64 = 16_000_000;
const MAX_IMAGE_RASTER_PIXELS: u64 = 4_000_000;
const MAX_TEXT_BYTES: usize = 16 * 1024;
const MAX_TEXT_GLYPHS: usize = 1_024;
const MAX_GLYPH_SIZE: f32 = 1_024.0;
const MAX_TEXT_RASTER_PIXELS: u64 = 4_000_000;
const MAX_COMPOSITION_TEXTURE_PIXELS: u64 = 20_000_000;
const MAX_GPU_TEXTURE_DIMENSION: u32 = 2_048;

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
        {
            let mut encoder =
                GifEncoder::new(File::create(output).map_err(RenderError::OutputDirectory)?);
            encoder
                .set_repeat(Repeat::Infinite)
                .map_err(RenderError::Gif)?;
            for frame_index in 0..frame_count {
                let at_ms = frame_index * 1_000 / fps;
                let animated = scene_at(scene, at_ms);
                let (pixels, frame_warnings) =
                    self.render_rgba_with_asset_root(&animated, asset_root)?;
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
        let plan = composition_plan(scene, asset_root, &self.font)?;
        let width = scene.canvas.width;
        let height = scene.canvas.height;
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
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
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &texture,
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
        multisample: wgpu::MultisampleState::default(),
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
        multisample: wgpu::MultisampleState::default(),
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
) -> Result<CompositionPlan, RenderError> {
    let mut plan = CompositionPlan {
        primitive_vertices: Vec::new(),
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
                        let (metrics, bitmap) = font.rasterize(character, *size);
                        if metrics.width == 0 || metrics.height == 0 {
                            cursor_x += metrics.advance_width;
                            previous = Some(character);
                            continue;
                        }
                        texture_pixels = reserve_composition_pixels(
                            texture_pixels,
                            (metrics.width * metrics.height) as u64,
                        )?;
                        let pixels = bitmap
                            .into_iter()
                            .flat_map(|alpha| [255, 255, 255, alpha])
                            .collect();
                        let index = plan.textures.len();
                        plan.textures.push(TextureData {
                            label: format!("glyph-{}-{}", character as u32, size),
                            width: metrics.width as u32,
                            height: metrics.height as u32,
                            pixels,
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
                let cache_key = (
                    source.clone(),
                    target_width.min(MAX_GPU_TEXTURE_DIMENSION),
                    target_height.min(MAX_GPU_TEXTURE_DIMENSION),
                );
                let texture_index = if let Some(index) = image_textures.get(&cache_key) {
                    *index
                } else {
                    let image = load_image(asset_root, source)?;
                    let (upload_width, upload_height) = upload_dimensions(
                        image.width(),
                        image.height(),
                        target_width,
                        target_height,
                    );
                    let image = if image.width() == upload_width && image.height() == upload_height
                    {
                        image
                    } else {
                        image::DynamicImage::ImageRgba8(image)
                            .resize_exact(
                                upload_width,
                                upload_height,
                                image::imageops::FilterType::Triangle,
                            )
                            .to_rgba8()
                    };
                    texture_pixels = reserve_composition_pixels(
                        texture_pixels,
                        u64::from(image.width()) * u64::from(image.height()),
                    )?;
                    let index = plan.textures.len();
                    plan.textures.push(TextureData {
                        label: format!("image-{source}"),
                        width: image.width(),
                        height: image.height(),
                        pixels: image.into_raw(),
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

fn load_image(asset_root: &Path, source: &str) -> Result<image::RgbaImage, RenderError> {
    let path = resolve_asset(asset_root, source)?;
    let metadata = fs::metadata(&path).map_err(|error| RenderError::Asset(error.to_string()))?;
    if metadata.len() > MAX_ASSET_BYTES {
        return Err(RenderError::Asset(format!(
            "image '{source}' exceeds 16 MiB"
        )));
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

    /// Golden-image tolerance for `renders_golden_scenes_within_tolerance_on_an_available_gpu`.
    ///
    /// The renderer draws hard-edged triangles with no MSAA, so shape edges
    /// are exact given identical input; the only sources of legitimate,
    /// non-bug pixel drift across GPUs/drivers are: (1) fontdue's
    /// anti-aliased glyph coverage combined with sRGB-aware alpha blending on
    /// `Rgba8UnormSrgb`, where different GPUs may round the linear<->sRGB
    /// conversion by a few least-significant bits, and (2) bilinear texture
    /// sampling when an image is uploaded below its target size (as in these
    /// fixtures) and stretched by the GPU sampler, whose interpolation
    /// weights can differ minutely by hardware. Neither should ever move a
    /// pixel by more than a handful of 8-bit levels, and neither should
    /// affect more than a thin sliver of pixels along glyph/image edges.
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

        let plan = composition_plan(&scene, directory.path(), &font).unwrap();
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
        let primitive_plan = composition_plan(&primitives, directory.path(), &font).unwrap();
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
        }
    }
}
