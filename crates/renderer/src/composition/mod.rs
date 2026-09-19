mod analytic;

pub(crate) use analytic::*;

#[cfg(test)]
mod tests;

use crate::{assets::*, error::RenderError, limits::*, tessellation::*, vertex::*};
use fontdue::Font;
use renderer_schema::{Color, NodeKindV1, SceneV1};
use std::{collections::HashMap, path::Path};

pub(crate) enum DrawCommand {
    Primitive(std::ops::Range<u32>),
    Textured {
        texture_index: usize,
        vertices: std::ops::Range<u32>,
    },
}

pub(crate) struct TextureData {
    pub(crate) label: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<u8>,
}

pub(crate) struct CompositionPlan {
    pub(crate) primitive_vertices: Vec<Vertex>,
    pub(crate) textured_vertices: Vec<TexturedVertex>,
    pub(crate) commands: Vec<DrawCommand>,
    pub(crate) textures: Vec<TextureData>,
}

pub(crate) fn composition_plan(
    scene: &SceneV1,
    asset_root: &Path,
    font: &Font,
    cache: &mut AssetCache,
) -> Result<CompositionPlan, RenderError> {
    let mut plan = CompositionPlan {
        primitive_vertices: Vec::new(),
        textured_vertices: Vec::new(),
        commands: Vec::new(),
        textures: Vec::new(),
    };
    // These two maps stay function-local (unlike `cache`, which is shared
    // across calls): they dedup repeated references to the same asset
    // *within this one call* to a single `plan.textures` entry/index, so a
    // glyph or image referenced twice in one frame still only gets pushed
    // into that frame's `CompositionPlan.textures` once -- exactly the
    // within-call behavior this file had before `AssetCache` existed.
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
                        // Rasterize only on a cache miss: `cache.glyphs`
                        // never holds a zero-size entry (see the `continue`
                        // below), so a hit here always has real pixel data.
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
                    // Decode/rasterize only on a cache miss.
                    let decoded = match cache.images.entry(cache_key.clone()) {
                        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            // For raster sources this returns the
                            // native-resolution decode, resized below; for
                            // SVG sources `load_image` rasterizes directly
                            // at `upload_width`x`upload_height` (already
                            // equal to `upload_dimensions(..)`'s result, so
                            // the resize below becomes a no-op for SVG).
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

pub(crate) fn add_textured_rect(
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
