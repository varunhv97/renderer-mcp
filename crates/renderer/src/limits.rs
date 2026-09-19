use std::time::Duration;

pub(crate) const MAX_ASSET_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const MAX_ASSET_PIXELS: u64 = 16_000_000;
pub(crate) const MAX_IMAGE_RASTER_PIXELS: u64 = 4_000_000;
pub(crate) const MAX_TEXT_BYTES: usize = 16 * 1024;
pub(crate) const MAX_TEXT_GLYPHS: usize = 1_024;
pub(crate) const MAX_GLYPH_SIZE: f32 = 1_024.0;
pub(crate) const MAX_TEXT_RASTER_PIXELS: u64 = 4_000_000;
pub(crate) const MAX_COMPOSITION_TEXTURE_PIXELS: u64 = 20_000_000;
pub(crate) const MAX_GPU_TEXTURE_DIMENSION: u32 = 2_048;
/// Wall-clock ceiling for parsing+rasterizing a single SVG asset. `usvg`
/// already refuses documents with more than 1,000,000 XML nodes
/// (`usvg::Error::ElementsLimitReached`, which also bounds `<use>`-expansion
/// style blowups since expansion copies count against the same limit), but
/// pathological filter chains (e.g. many chained `feGaussianBlur`s) can still
/// be expensive without tripping that counter. This budget turns a hang into
/// a fast, actionable `RenderError::Asset` instead.
pub(crate) const SVG_RASTER_TIME_BUDGET: Duration = Duration::from_secs(5);
