use crate::assets::*;
use crate::error::*;
use crate::limits::*;
use fontdue::Font;
use renderer_schema::CanvasV1;
use renderer_schema::Color;
use renderer_schema::FillV1;
use renderer_schema::NodeKindV1;
use renderer_schema::NodeV1;
use renderer_schema::SCENE_VERSION_V1;
use renderer_schema::SceneV1;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

#[cfg(test)]
pub(crate) fn rasterize_text_and_images(
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
                fill,
            } => {
                let color = fill.resolve_solid();
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

#[cfg(test)]
pub(crate) fn blend_pixel(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    color: Color,
) {
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
pub(crate) fn blend_premultiplied_layer(
    destination: &mut [u8],
    source: &[u8],
    width: u32,
    height: u32,
) {
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

/// Golden-image tolerance for `renders_golden_scenes_within_tolerance_on_an_available_gpu`.
///
/// The renderer anti-aliases vector primitives via `MSAA_SAMPLE_COUNT`x
/// MSAA (see that constant's doc comment), so shape edges are
/// coverage-weighted blends rather than exact given identical input; the
/// sources of legitimate, non-bug pixel drift across GPUs/drivers are:
/// (1) the MSAA resolve itself, where different GPUs/drivers can place
/// sample points or weight coverage minutely differently along a shape
/// edge, (2) fontdue's anti-aliased glyph coverage combined with
/// sRGB-aware alpha blending on `Rgba8UnormSrgb`, where different GPUs
/// may round the linear<->sRGB conversion by a few least-significant
/// bits, and (3) bilinear texture sampling when an image is uploaded
/// below its target size (as in these fixtures) and stretched by the GPU
/// sampler, whose interpolation weights can differ minutely by hardware.
/// None of these should ever move a pixel by more than a handful of
/// 8-bit levels, and none should affect more than a thin sliver of
/// pixels along shape/glyph/image edges.
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
pub(crate) const GOLDEN_MAX_CHANNEL_DELTA: u8 = 8;

pub(crate) const GOLDEN_MAX_MISMATCHED_PIXEL_RATIO: f64 = 0.0075;

pub(crate) const GOLDEN_MAX_MEAN_CHANNEL_DELTA: f64 = 1.0;

pub(crate) fn golden_asset_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/golden")
}

pub(crate) fn load_golden_scene(name: &str) -> SceneV1 {
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
pub(crate) fn assert_matches_golden(
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
            let delta =
                (actual_pixel[channel] as i16 - golden_pixel[channel] as i16).unsigned_abs() as u8;
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

pub(crate) fn test_scene() -> SceneV1 {
    SceneV1 {
        version: SCENE_VERSION_V1.into(),
        canvas: CanvasV1 {
            width: 32,
            height: 32,
            background: [0.0; 4],
        },
        nodes: vec![NodeV1 {
            id: "box".into(),
            translate: [0.0, 0.0],
            kind: NodeKindV1::Rect {
                x: 1.0,
                y: 1.0,
                width: 10.0,
                height: 10.0,
                corner_radius: 0.0,
                fill: FillV1::Solid([1.0, 0.0, 0.0, 1.0]),
            },
        }],
        timeline: None,
        effect: None,
    }
}
