use super::*;
use crate::{
    analytic_geometry::*, assets::*, error::RenderError, limits::*, tessellation::*, vertex::*,
};
use fontdue::Font;
use renderer_schema::{NodeKindV1, SceneV1};
use std::{collections::HashMap, path::Path};

/// Draw-command list for the analytic-AA composition plan (see
/// `CompositionPlanAnalytic`/`composition_plan_analytic`). This mirrors
/// `DrawCommand` above but splits its catch-all `Primitive` variant into two
/// -- `Analytic` (Rect/Ellipse/Line, drawn with `AnalyticVertex`/`analytic_
/// pipeline`) and `Path` (drawn with the original `Vertex`/`primitive_
/// pipeline`, as the documented MSAA fallback -- see `composition_plan_
/// analytic`'s doc comment) -- since those two need different GPU pipelines
/// with different render-target sample counts and therefore different
/// render passes (see `render_composed_rgba_analytic_with_cache`).
pub(crate) enum AnalyticDrawCommand {
    Analytic(std::ops::Range<u32>),
    Path(std::ops::Range<u32>),
    Textured {
        texture_index: usize,
        vertices: std::ops::Range<u32>,
    },
}

/// Analytic-AA counterpart to `CompositionPlan`. `textures` and `textured_
/// vertices` are the same `TextureData`/`TexturedVertex` types the original
/// path uses (text/images are handled identically either way -- see
/// `composition_plan_analytic`); `analytic_vertices` holds the new `Analytic
/// Vertex` geometry for Rect/Ellipse/Line, and `path_vertices` holds plain
/// `Vertex` geometry (built with the existing, untouched `add_path`) for the
/// `Path` MSAA fallback.
pub(crate) struct CompositionPlanAnalytic {
    pub(crate) analytic_vertices: Vec<AnalyticVertex>,
    pub(crate) path_vertices: Vec<Vertex>,
    pub(crate) textured_vertices: Vec<TexturedVertex>,
    pub(crate) commands: Vec<AnalyticDrawCommand>,
    pub(crate) textures: Vec<TextureData>,
}

/// Analytic-AA counterpart to `composition_plan`. A separate function
/// (rather than a shared helper `composition_plan` also calls) so that
/// `composition_plan` itself -- and therefore every existing render entry
/// point's behavior -- is not touched at all by this addition. Text and
/// image handling below is intentionally near-identical to
/// `composition_plan`'s (same caching, same validation, same texture
/// upload prep -- none of `AssetCache`/`DecodedImage`/`DecodedGlyph`/
/// `TextureData`/`reserve_composition_pixels`/`validate_text_raster`/
/// `load_image`/`upload_dimensions`/`bounded_image_dimension`/`ensure_
/// target_image_dimensions` are modified, only reused) since this project's
/// SDF/`fwidth` anti-aliasing technique is specifically about vector
/// *shape* edges, not about how text glyphs or images are rasterized/
/// uploaded.
///
/// Rect/Ellipse/Line nodes are built with the new `add_*_analytic` builders
/// into `analytic_vertices`. `Path` nodes are the one documented exception
/// to "fully analytic": implementing a mathematically sound analytic SDF for
/// an arbitrary (potentially concave, potentially self-intersecting)
/// filled polygon is substantially more involved than the closed-form
/// rect/ellipse/capsule SDFs above, so -- as this project's task brief
/// explicitly allows -- `Path` nodes here fall back to exactly the existing
/// MSAA `primitive_pipeline`/`add_path`/`Vertex` machinery, unchanged,
/// executed in its own multisampled render pass that resolves into the same
/// target the analytic passes draw into (see `render_composed_rgba_
/// analytic_with_cache`). This means a scene that uses `Path` nodes gets
/// MSAA-only (no supersampling) quality for just those nodes while every
/// other shape in the same scene still gets full analytic AA -- a
/// consciously narrower guarantee for one shape kind, not a silent gap.
pub(crate) fn composition_plan_analytic(
    scene: &SceneV1,
    asset_root: &Path,
    font: &Font,
    cache: &mut AssetCache,
) -> Result<CompositionPlanAnalytic, RenderError> {
    let mut plan = CompositionPlanAnalytic {
        analytic_vertices: Vec::new(),
        path_vertices: Vec::new(),
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
                fill,
            } => {
                let x = *x + node.translate[0];
                let y = *y + node.translate[1];
                // Text does not implement true gradient rendering (see
                // `FillV1::resolve_solid`'s doc comment); a gradient fill
                // resolves to a flat representative color here.
                let color = fill.resolve_solid();
                validate_text_raster(node.id.as_str(), text, *size, scene)?;
                let mut cursor_x = x;
                let mut previous = None;
                for character in text.chars() {
                    if let Some(left) = previous {
                        cursor_x += font.horizontal_kern(left, character, *size).unwrap_or(0.0);
                    }
                    let key = (character, size.to_bits());
                    let (texture_index, metrics) = if let Some(value) = glyph_textures.get(&key) {
                        *value
                    } else {
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
                        y + (*size - metrics.height as f32 - metrics.ymin as f32),
                        metrics.width as f32,
                        metrics.height as f32,
                        color,
                        scene,
                    );
                    plan.commands.push(AnalyticDrawCommand::Textured {
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
                let x = *x + node.translate[0];
                let y = *y + node.translate[1];
                let target_width = bounded_image_dimension(*width, "width")?;
                let target_height = bounded_image_dimension(*height, "height")?;
                ensure_target_image_dimensions(target_width, target_height, source)?;
                let upload_width = target_width.min(MAX_GPU_TEXTURE_DIMENSION);
                let upload_height = target_height.min(MAX_GPU_TEXTURE_DIMENSION);
                let cache_key = (source.clone(), upload_width, upload_height);
                let texture_index = if let Some(index) = image_textures.get(&cache_key) {
                    *index
                } else {
                    let decoded = match cache.images.entry(cache_key.clone()) {
                        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                        std::collections::hash_map::Entry::Vacant(entry) => {
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
                    x,
                    y,
                    *width,
                    *height,
                    [1.0; 4],
                    scene,
                );
                plan.commands.push(AnalyticDrawCommand::Textured {
                    texture_index,
                    vertices: start..start + 6,
                });
            }
            NodeKindV1::Path { points, fill } => {
                let translated: Vec<_> = points
                    .iter()
                    .map(|point| renderer_schema::PointV1 {
                        x: point.x + node.translate[0],
                        y: point.y + node.translate[1],
                    })
                    .collect();
                let start = plan.path_vertices.len() as u32;
                add_path(
                    &mut plan.path_vertices,
                    &translated,
                    fill.resolve_solid(),
                    scene,
                );
                let end = plan.path_vertices.len() as u32;
                if start != end {
                    if let Some(AnalyticDrawCommand::Path(range)) = plan.commands.last_mut() {
                        range.end = end;
                    } else {
                        plan.commands.push(AnalyticDrawCommand::Path(start..end));
                    }
                }
            }
            NodeKindV1::Rect {
                x,
                y,
                width,
                height,
                corner_radius,
                fill,
            } => {
                let x = *x + node.translate[0];
                let y = *y + node.translate[1];
                let start = plan.analytic_vertices.len() as u32;
                add_rect_analytic(
                    &mut plan.analytic_vertices,
                    x,
                    y,
                    *width,
                    *height,
                    *corner_radius,
                    fill,
                    scene,
                );
                push_analytic_range(
                    &mut plan.commands,
                    start,
                    plan.analytic_vertices.len() as u32,
                );
            }
            NodeKindV1::Ellipse {
                cx,
                cy,
                rx,
                ry,
                fill,
            } => {
                let cx = *cx + node.translate[0];
                let cy = *cy + node.translate[1];
                let start = plan.analytic_vertices.len() as u32;
                add_ellipse_analytic(&mut plan.analytic_vertices, cx, cy, *rx, *ry, fill, scene);
                push_analytic_range(
                    &mut plan.commands,
                    start,
                    plan.analytic_vertices.len() as u32,
                );
            }
            NodeKindV1::Line {
                x1,
                y1,
                x2,
                y2,
                thickness,
                fill,
            } => {
                let start_point = [*x1 + node.translate[0], *y1 + node.translate[1]];
                let end_point = [*x2 + node.translate[0], *y2 + node.translate[1]];
                let start = plan.analytic_vertices.len() as u32;
                add_line_analytic(
                    &mut plan.analytic_vertices,
                    start_point,
                    end_point,
                    *thickness,
                    fill.resolve_solid(),
                    scene,
                );
                push_analytic_range(
                    &mut plan.commands,
                    start,
                    plan.analytic_vertices.len() as u32,
                );
            }
        }
    }
    Ok(plan)
}

/// Merges a freshly-emitted `[start, end)` `AnalyticVertex` range into
/// `commands`: extends the last command if it is already an `Analytic` run
/// (matching how `composition_plan`'s catch-all branch merges consecutive
/// `Primitive` commands above), otherwise pushes a new one. A no-op when
/// `start == end` (an unfilled range, matching e.g. `add_line_analytic`'s
/// early return for a zero-length line).
fn push_analytic_range(commands: &mut Vec<AnalyticDrawCommand>, start: u32, end: u32) {
    if start == end {
        return;
    }
    if let Some(AnalyticDrawCommand::Analytic(range)) = commands.last_mut() {
        range.end = end;
    } else {
        commands.push(AnalyticDrawCommand::Analytic(start..end));
    }
}
