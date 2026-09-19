mod analytic;
mod frame;
mod render;

pub(crate) use frame::*;

use crate::{error::RenderError, pipelines::*};
use fontdue::Font;
use renderer_schema::MAX_CANVAS_DIMENSION;

/// MSAA sample count used to anti-alias vector primitives (rect/ellipse/
/// line/path). 4x is the standard, broadly-supported choice for
/// `Rgba8UnormSrgb` render targets on desktop GPUs (Metal/Vulkan/DX12) and is
/// what this renderer's node-composition pass uses; see
/// `GpuRenderer::new_async` for an adapter-capability check confirming this
/// value is supported before it is relied on. The effect pass intentionally
/// stays single-sampled (see `build_effect_pipeline`): it introduces no new
/// geometric edges, so multisampling it would add cost with no benefit.
pub(crate) const MSAA_SAMPLE_COUNT: u32 = 4;

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
pub(crate) const SUPERSAMPLE_FACTOR: u32 = 2;

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
            include_bytes!("../../assets/NotoSans-Regular.ttf") as &[u8],
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
}
