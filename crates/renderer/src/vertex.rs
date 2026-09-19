use bytemuck::{Pod, Zeroable};
use renderer_schema::Color;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Vertex {
    pub(crate) position: [f32; 2],
    pub(crate) color: Color,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct TexturedVertex {
    pub(crate) position: [f32; 2],
    pub(crate) uv: [f32; 2],
    pub(crate) tint: Color,
}

impl TexturedVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4];

    pub(crate) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Vertex format for the analytic (signed-distance-field) anti-aliasing
/// pipeline -- see `ANALYTIC_SHADER`. Rect, ellipse, and line SDFs each need
/// genuinely different per-vertex data: a rect's box SDF needs a half-extent;
/// an ellipse's squashed-space SDF needs radii-normalized local coordinates;
/// a line's capsule SDF needs both segment endpoints, a half-thickness, and
/// the fragment's own local position. Two designs were considered: (a) three
/// small, shape-specific vertex formats/pipelines/shaders, or (b) one
/// flexible format wide enough for the most demanding shape (the line
/// capsule), tagged with a `shape_kind` the fragment shader switches on.
/// This uses (b): one pipeline and one shader module is simpler to build,
/// test, and reason about than three near-identical small ones for a
/// renderer this size, at the cost of a few unused `f32`s per vertex on the
/// simpler shapes (`param1`/`param2` go unused for Rect and Ellipse) -- a
/// fine trade here. Each shape's builder function
/// (`add_rect_analytic`/`add_ellipse_analytic`/`add_line_analytic` below)
/// zeroes whatever fields its shape kind does not use.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct AnalyticVertex {
    /// Clip-space position, computed exactly like `vertex()`/`add_textured_
    /// rect`'s `point()` helper above.
    pub(crate) clip_position: [f32; 2],
    pub(crate) color: Color,
    /// Which SDF `fs_main` (in `ANALYTIC_SHADER`) evaluates for this
    /// fragment -- one of `ANALYTIC_SHAPE_RECT`/`ANALYTIC_SHAPE_ELLIPSE`/
    /// `ANALYTIC_SHAPE_LINE`. Constant across every vertex of one shape, so
    /// it is marked `@interpolate(flat)` on the WGSL side (required for
    /// integer varyings regardless).
    pub(crate) shape_kind: u32,
    /// Meaning depends on `shape_kind`:
    /// - Rect: `(fragment - center)`, in scene-pixel units.
    /// - Ellipse: `(fragment - center) / (rx, ry)` -- already in the
    ///   ellipse's "squashed" unit-circle space.
    /// - Line: the fragment's raw scene-pixel position (the capsule SDF
    ///   needs the actual position, not one pre-offset by anything).
    pub(crate) local: [f32; 2],
    /// Rect: `(half_width, half_height)`. Line: segment start `a`. Unused
    /// (zeroed) for Ellipse.
    pub(crate) param0: [f32; 2],
    /// Rect: `.x` is the corner radius (`.y` unused padding) -- see
    /// `rect_sdf` in `ANALYTIC_SHADER`. Line: segment end `b`. Unused
    /// (zeroed) for Ellipse.
    pub(crate) param1: [f32; 2],
    /// Line: `.x` is the half-thickness; `.y` is unused padding. Unused
    /// (zeroed) for Rect and Ellipse.
    pub(crate) param2: [f32; 2],
}

impl AnalyticVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x4,
        2 => Uint32,
        3 => Float32x2,
        4 => Float32x2,
        5 => Float32x2,
        6 => Float32x2,
    ];

    pub(crate) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];

    pub(crate) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}
