use crate::{gpu::MSAA_SAMPLE_COUNT, shaders::*, vertex::*};

pub(crate) fn create_pipelines(
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
pub(crate) fn create_analytic_pipelines(
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

pub(crate) fn color_target() -> wgpu::ColorTargetState {
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
pub(crate) fn effect_color_target() -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    }
}
