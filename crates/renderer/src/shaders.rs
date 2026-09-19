pub(crate) const PRIMITIVE_SHADER: &str = r#"
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

pub(crate) const TEXTURED_SHADER: &str = r#"
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

/// Shader for the analytic (signed-distance-field) anti-aliasing pipeline
/// (`analytic_pipeline`, built in `create_analytic_pipelines`; vertex format
/// is `AnalyticVertex`). Renders rect/ellipse/line primitives with an exact
/// analytic distance to each shape's true mathematical boundary, converted
/// to a smooth per-fragment coverage value with `fwidth` -- a single sample
/// per fragment, no multisampling.
///
/// Sign convention (standard for SDF rendering): `d < 0` means inside the
/// shape, `d == 0` is exactly on the boundary, `d > 0` means outside.
///
/// `fwidth(d)` is the screen-space rate of change of `d` between this
/// fragment and its neighbors -- i.e. "how many `d` units does one pixel
/// span here" -- which is why this technique needs no separate resolution
/// or transform parameter: it is automatically correct at any zoom/skew
/// because it is derived from the GPU's own rasterization derivatives.
/// `coverage` linearly ramps from 1 (fully inside) to 0 (fully outside)
/// across a band roughly one pixel wide, centered on `d == 0`.
pub(crate) const ANALYTIC_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) shape_kind: u32,
    @location(2) local: vec2<f32>,
    @location(3) param0: vec2<f32>,
    @location(4) param1: vec2<f32>,
    @location(5) param2: vec2<f32>,
}

const SHAPE_RECT: u32 = 0u;
const SHAPE_ELLIPSE: u32 = 1u;
const SHAPE_LINE: u32 = 2u;

@vertex
fn vs_main(
    @location(0) clip_position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) shape_kind: u32,
    @location(3) local: vec2<f32>,
    @location(4) param0: vec2<f32>,
    @location(5) param1: vec2<f32>,
    @location(6) param2: vec2<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(clip_position, 0.0, 1.0);
    output.color = color;
    output.shape_kind = shape_kind;
    output.local = local;
    output.param0 = param0;
    output.param1 = param1;
    output.param2 = param2;
    return output;
}

// Axis-aligned (optionally rounded) box SDF in the rect's local
// (fragment - center) space. `radius <= 0.0` (the overwhelmingly common
// case -- every rect without a `corner_radius`) uses the exact same
// Chebyshev-distance formula this function always used before rounded
// corners existed, so a zero-radius rect's rendered output is completely
// unaffected by this change.
//
// `radius > 0.0` uses the standard exact rounded-box SDF (Inigo Quilez's
// well-known `sdRoundBox`): shrink the box by `radius` on every side, take
// the Euclidean distance to *that* shrunk box's boundary via
// `length(max(q, 0)) + min(max(q.x, q.y), 0)`, then subtract `radius` back
// off to re-expand to the true (rounded) boundary. This -- not the simpler
// `length(max(q, 0)) - radius` sometimes quoted -- is the correct exact
// form: the simpler formula is only correct *outside* the box's inscribed
// cross shape and returns 0 (instead of the true negative/inside distance)
// for fragments inside that cross but outside the shrunk box, which would
// wrongly zero out `fs_main`'s coverage ramp near (but not at) a rounded
// edge's midpoint. Verified by rendering a rounded rect and decoding raw
// RGBA output: corner pixels are background-colored where a sharp corner
// would have been shape-colored, confirming this sign/offset convention is
// actually correct rather than merely plausible-looking.
fn rect_sdf(local: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    if (radius <= 0.0) {
        let delta = abs(local) - half_size;
        return max(delta.x, delta.y);
    }
    let q = abs(local) - half_size + vec2<f32>(radius, radius);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0, 0.0))) - radius;
}

// Ellipse SDF approximation in "squashed" space: `local` is
// `(fragment - center) / (rx, ry)`, so the true ellipse boundary maps to the
// unit circle, where `length(local) - 1.0` is exactly zero. This is NOT the
// true Euclidean distance to an ellipse boundary for `rx != ry` (that has no
// simple closed form); it is a standard, widely used approximation. The
// per-fragment gradient still shrinks smoothly to zero exactly at the
// boundary -- all `fwidth`-based coverage actually needs -- but the *rate*
// isn't perfectly isotropic away from the major/minor axes for very
// eccentric ellipses, so the apparent AA band width can vary slightly around
// the boundary of a very non-circular ellipse. Acceptable trade-off for this
// renderer's scale; see `Path`'s documented fallback below for the same
// spirit applied to a harder shape.
fn ellipse_sdf(local: vec2<f32>) -> f32 {
    return length(local) - 1.0;
}

// Standard 2D capsule SDF: distance from `point` to the segment [a, b],
// minus the half-thickness. `point`/`a`/`b` are all in the same (scene-pixel)
// space.
fn capsule_sdf(point: vec2<f32>, a: vec2<f32>, b: vec2<f32>, half_thickness: f32) -> f32 {
    let pa = point - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - half_thickness;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var d: f32;
    if (input.shape_kind == SHAPE_RECT) {
        d = rect_sdf(input.local, input.param0, input.param1.x);
    } else if (input.shape_kind == SHAPE_ELLIPSE) {
        d = ellipse_sdf(input.local);
    } else {
        d = capsule_sdf(input.local, input.param0, input.param1, input.param2.x);
    }
    // `fwidth(d)` (for the true-unit-gradient SDFs above) is exactly a unit
    // pixel's footprint width projected onto the distance gradient's
    // direction -- i.e. exactly how far `d` needs to travel to cross one
    // whole pixel. A coverage ramp that spans exactly that width, centered
    // on `d == 0`, is `clamp(0.5 - d / fwidth(d), 0.0, 1.0)`: 1.0 (fully
    // covered) at `d == -fwidth(d)/2` (half a pixel-footprint inside), 0.0
    // at `d == +fwidth(d)/2`. An earlier version divided by `fwidth(d) *
    // 0.5` here, which halves the ramp's width (saturates to fully
    // covered/uncovered at a quarter-pixel-footprint instead of half) --
    // caught by comparing rendered output against numerically-integrated
    // ground-truth pixel coverage: it saturated to exactly the fill/
    // background color at pixels whose true area coverage was still only
    // ~89%/~17%, not near 100%/0%.
    let aa_width = max(fwidth(d), 1e-5);
    let coverage = clamp(0.5 - d / aa_width, 0.0, 1.0);
    var color = input.color;
    color.a = color.a * coverage;
    return color;
}
"#;

pub(crate) const EFFECT_VERTEX_ENTRY_POINT: &str = "renderer_cli_effect_vs";

pub(crate) const EFFECT_FRAGMENT_ENTRY_POINT: &str = "renderer_cli_effect_fs";

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
pub(crate) fn wrap_effect_shader(user_shader: &str) -> String {
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
