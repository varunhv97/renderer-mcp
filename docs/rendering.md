# Rendering and scenes

## Scene features

The workspace is intentionally CLI and MCP first: agents submit a versioned JSON
scene or drawing-command document, receive a saved image plus structured
metadata, and can keep a named scene alive during a local daemon session.
Scenes compose vector shapes, local raster and SVG images, and embedded-font
text into one ordered GPU pass; PNG and keyframed GIF export both work from
the same scene document. Nodes can be keyframe-animated by opacity, color, or
position (an additive `translate` offset applied on top of the node's own
coordinates, uniform across every node kind). Rects support an optional
`corner_radius` for rounded corners, and every shape's `color` field accepts
either a plain solid RGBA array (unchanged from before) or a gradient object
(`{"kind": "linear_gradient", ...}` or `{"kind": "radial_gradient", ...}`);
rects and ellipses render a true per-pixel gradient, while lines, paths, and
text resolve a gradient fill to one flat color. Vector shapes (rects,
ellipses, lines, paths) are anti-aliased via 4x MSAA plus 2x supersampling
(the whole scene renders at
2x linear resolution and is downsampled with a Lanczos3 filter before
output), matching the anti-aliased edges text and SVG content already had.
Image nodes accept local PNG, JPEG, GIF, WebP, or SVG assets; SVG assets are
rasterized directly at each node's declared size rather than decoded and
rescaled.

An experimental, opt-in analytic (signed-distance-field) anti-aliasing
pipeline is also available for vector shapes, as a shadow-mode alternative to
the default MSAA+supersampling path — a completely separate, additive set of
pipelines/shaders that the default path never touches. Compute a per-fragment
exact distance to each shape's true boundary instead of sampling/averaging;
measured more accurate against numeric ground-truth pixel coverage than
MSAA+supersampling on this project's own test scenes. Try it with:

```sh
cargo run -p renderer-cli -- render --input examples/basic.scene.json --output out.png --experimental-analytic-aa
```

and compare the result against a normal `render` (no flag) of the same
input; `.gif` output works too. Not currently wired into the daemon, named
scenes, or MCP — CLI-only, for local comparison.

## One-shot rendering

```sh
cargo run -p renderer-cli -- render --input examples/basic.scene.json --output out.png
cargo run -p renderer-cli -- render --input examples/pulse.scene.json --output out.gif
cargo run -p renderer-cli -- inspect --input out.png
```

`render` accepts a `SceneV1` JSON document and writes `.png` or `.gif` based on
the output extension; GIF export uses the scene's keyframe timeline. `inspect`
reports dimensions, MIME type, and the SHA-256 of the exact output bytes.
Diagnostics (missing assets, invalid scenes, resource-limit violations) are
returned as structured JSON, not silent failures.

A scene may also include an optional, scene-level full-canvas post-process
`effect`: a WGSL `shader` string defining exactly one function,
`fn effect(uv: vec2<f32>, color: vec4<f32>) -> vec4<f32>`, that transforms
the already-composited color at each pixel (see `examples/effect.scene.json`,
a color-invert effect). The renderer wraps this function in a fixed internal
template (vertex stage, texture bindings) so authors only ever write a pure
color transform; they never supply bindings, vertex data, or a full pipeline.
A shader that fails WGSL validation is rejected with a structured error
rather than crashing the process.

```sh
cargo run -p renderer-cli -- render --input examples/effect.scene.json --output out-effect.png
```
