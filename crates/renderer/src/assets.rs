use crate::{error::RenderError, limits::*, svg::*};
use renderer_schema::SceneV1;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

/// Decoded image pixel data, cached by exactly the key `composition_plan`
/// already used for its function-local `image_textures` dedup map:
/// `(source, upload_width, upload_height)`. Storing the final processed
/// buffer (post raster-decode/SVG-rasterize *and* post-resize) means a cache
/// hit needs no further work beyond a clone into that frame's own
/// `CompositionPlan.textures`.
pub(crate) struct DecodedImage {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<u8>,
}

/// Rasterized glyph pixel data (already alpha-expanded to RGBA, matching
/// what used to be pushed straight into `TextureData::pixels`), cached by
/// exactly the key `composition_plan` already used for its function-local
/// `glyph_textures` dedup map: `(char, size_bits)`.
pub(crate) struct DecodedGlyph {
    pub(crate) metrics: fontdue::Metrics,
    pub(crate) pixels: Vec<u8>,
}

/// Cross-call cache of the *expensive* CPU-side work `composition_plan`
/// performs per image/glyph -- disk read + `image`-crate decode or
/// `resvg`/`usvg`/`tiny-skia` SVG parse+rasterize for images, `fontdue`
/// rasterization for glyphs -- keyed identically to how `composition_plan`
/// already deduped repeats *within* one call before this cache existed.
///
/// `render_composed_rgba` (single-shot renders, including every `render_png`
/// / `render_rgba` call) creates a fresh, single-use `AssetCache` per call,
/// so its behavior -- and therefore the golden-image tests -- is unaffected:
/// every key is a "miss" exactly once, exactly as when the maps were
/// function-local.
///
/// `render_gif_with_asset_root` instead creates ONE `AssetCache` before its
/// per-frame loop and passes it by `&mut` reference into every frame's
/// `composition_plan` call, so a given `(source, width, height)` or `(char,
/// size_bits)` is decoded/rasterized at most once across the whole GIF
/// export, no matter how many frames reference it -- static backgrounds,
/// logos, and labels included.
///
/// This intentionally caches only the decoded *pixel data*, not GPU texture
/// indices: each `composition_plan` call still pushes its own fresh entry
/// (and index) into that frame's `CompositionPlan.textures` even on a cache
/// hit, since GPU upload (`render_composed_rgba`'s `create_texture` /
/// `write_texture` calls) still happens per frame -- see the module-level
/// task notes on why that upload step is out of scope here.
#[derive(Default)]
pub(crate) struct AssetCache {
    pub(crate) images: HashMap<(String, u32, u32), DecodedImage>,
    pub(crate) glyphs: HashMap<(char, u32), DecodedGlyph>,
    /// Test-only instrumentation: counts actual cache-miss decodes/
    /// rasterizations (not lookups), so tests can assert a cache shared
    /// across N `composition_plan` calls performs the expensive work exactly
    /// once instead of N times. Never read outside `#[cfg(test)]` code.
    #[cfg(test)]
    pub(crate) image_decodes: usize,
    #[cfg(test)]
    pub(crate) glyph_rasterizations: usize,
}

#[cfg(test)]
impl AssetCache {
    pub(crate) fn image_decode_count(&self) -> usize {
        self.image_decodes
    }

    pub(crate) fn glyph_rasterization_count(&self) -> usize {
        self.glyph_rasterizations
    }
}

pub(crate) fn validate_text_raster(
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

/// Loads the image asset named by `source` (resolved and containment-checked
/// via [`resolve_asset`], exactly like every other asset lookup in this
/// file).
///
/// Raster formats (PNG/JPEG/GIF/WebP, decoded by the `image` crate) are
/// returned at their native resolution; the caller (`composition_plan`)
/// downsamples/upsamples them to the node's declared size with a bilinear
/// GPU-quad resize, same as before this function grew SVG support.
///
/// SVG assets are different: there is no "native resolution" to decode at,
/// so they are rasterized directly at `upload_width`x`upload_height` (the
/// already-`MAX_GPU_TEXTURE_DIMENSION`-clamped size the caller is about to
/// upload) for crisp output, instead of being decoded at some arbitrary size
/// and then bilinearly rescaled.
pub(crate) fn load_image(
    asset_root: &Path,
    source: &str,
    upload_width: u32,
    upload_height: u32,
) -> Result<image::RgbaImage, RenderError> {
    let path = resolve_asset(asset_root, source)?;
    let metadata = fs::metadata(&path).map_err(|error| RenderError::Asset(error.to_string()))?;
    if metadata.len() > MAX_ASSET_BYTES {
        return Err(RenderError::Asset(format!(
            "image '{source}' exceeds 16 MiB"
        )));
    }
    if has_svg_extension(&path) {
        let bytes = fs::read(&path).map_err(|error| RenderError::Asset(error.to_string()))?;
        if !looks_like_svg(&bytes) {
            return Err(RenderError::Asset(format!(
                "asset '{source}' has an .svg extension but its content does not look like SVG"
            )));
        }
        return rasterize_svg(&bytes, asset_root, source, upload_width, upload_height);
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

pub(crate) fn bounded_image_dimension(value: f32, name: &str) -> Result<u32, RenderError> {
    let rounded = value.round();
    if !rounded.is_finite() || rounded <= 0.0 || rounded > 4_096.0 {
        return Err(RenderError::Asset(format!(
            "image {name} must be a finite positive value no greater than 4096"
        )));
    }
    Ok(rounded as u32)
}

pub(crate) fn ensure_source_image_dimensions(
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

pub(crate) fn ensure_target_image_dimensions(
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

pub(crate) fn resolve_asset(root: &Path, source: &str) -> Result<PathBuf, RenderError> {
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
