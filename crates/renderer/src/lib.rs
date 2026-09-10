//! Off-screen wgpu renderer for normalized RendererCli scenes.

use bytemuck::{Pod, Zeroable};
use image::{
    Delay, Frame, RgbaImage,
    codecs::gif::{GifEncoder, Repeat},
};
use renderer_schema::{Color, KeyframeV1, NodeKindV1, SceneV1};
use sha2::{Digest, Sha256};
use std::{
    fs,
    fs::File,
    path::{Path, PathBuf},
};
use thiserror::Error;
use wgpu::util::DeviceExt;

const ELLIPSE_SEGMENTS: usize = 32;

#[derive(Debug)]
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
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
}

impl GpuRenderer {
    pub fn new() -> Result<Self, RenderError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, RenderError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(RenderError::NoAdapter)?;
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
        Ok(Self { device, queue })
    }

    pub fn render_png(&self, scene: &SceneV1, output: &Path) -> Result<RenderedImage, RenderError> {
        scene.validate()?;
        ensure_png_output_path(output)?;
        let (pixels, warnings) = self.render_rgba(scene)?;
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
                let (pixels, frame_warnings) = self.render_rgba(&animated)?;
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
        let (vertices, warnings) = vertices_for_scene(scene);
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
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("renderer-cli primitives"),
                source: wgpu::ShaderSource::Wgsl(PRIMITIVE_SHADER.into()),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("renderer-cli layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("renderer-cli primitives pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[Vertex::layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("renderer-cli vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
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
            pass.set_pipeline(&pipeline);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.draw(0..vertices.len() as u32, 0..1);
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
        Ok((pixels, warnings))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: Color,
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

fn vertices_for_scene(scene: &SceneV1) -> (Vec<Vertex>, Vec<String>) {
    let mut vertices = Vec::new();
    let mut warnings = Vec::new();
    for node in &scene.nodes {
        match &node.kind {
            NodeKindV1::Rect {
                x,
                y,
                width,
                height,
                color,
            } => {
                add_rect(&mut vertices, *x, *y, *width, *height, *color, scene);
            }
            NodeKindV1::Ellipse {
                cx,
                cy,
                rx,
                ry,
                color,
            } => {
                add_ellipse(&mut vertices, *cx, *cy, *rx, *ry, *color, scene);
            }
            NodeKindV1::Line {
                x1,
                y1,
                x2,
                y2,
                thickness,
                color,
            } => {
                add_line(
                    &mut vertices,
                    [*x1, *y1],
                    [*x2, *y2],
                    *thickness,
                    *color,
                    scene,
                );
            }
            NodeKindV1::Path { points, color } => add_path(&mut vertices, points, *color, scene),
            NodeKindV1::Text { .. } | NodeKindV1::Image { .. } => {
                warnings.push(format!(
                    "node '{}' is accepted but not yet rasterized",
                    node.id
                ));
            }
        }
    }
    (vertices, warnings)
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
        assert_eq!(warnings.len(), 2);
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
        GpuRenderer::new().ok().iter().for_each(|renderer| {
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
        });
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
