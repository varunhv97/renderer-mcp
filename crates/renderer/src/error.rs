use std::path::PathBuf;
use thiserror::Error;

// Only referenced from a doc link on `RenderError::GifEncoding`.
#[cfg(doc)]
use crate::GpuRenderer;

#[derive(Clone, Debug, PartialEq)]
pub struct RenderedImage {
    pub width: u32,
    pub height: u32,
    pub sha256: String,
    pub frame_count: u32,
    pub warnings: Vec<String>,
}

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("scene validation failed: {0}")]
    InvalidScene(#[from] renderer_schema::SceneValidationError),
    #[error("no compatible GPU adapter was found")]
    NoAdapter,
    #[error("could not create GPU device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("GPU readback failed")]
    Readback,
    #[error("could not write image: {0}")]
    Image(#[from] image::ImageError),
    #[error("PNG render output must use a .png extension: {0}")]
    InvalidPngOutputPath(PathBuf),
    #[error("could not create output directory: {0}")]
    OutputDirectory(#[source] std::io::Error),
    #[error("could not write GIF: {0}")]
    Gif(#[source] image::ImageError),
    /// Distinct from [`Self::Gif`] (which wraps `image::ImageError` for the
    /// analytic-AA GIF path, still using the `image` crate's own encoder)
    /// because [`GpuRenderer::render_gif_with_asset_root`] talks to the
    /// lower-level `gif` crate directly -- see that method's doc comment.
    #[error("could not write GIF: {0}")]
    GifEncoding(#[source] gif::EncodingError),
    #[error("could not hash emitted output: {0}")]
    OutputRead(#[source] std::io::Error),
    #[error("could not load bundled font")]
    Font,
    #[error("asset error: {0}")]
    Asset(String),
    /// The scene's `effect.shader` failed WGSL compile validation once
    /// wrapped in the fixed post-process template. Kept at the end of this
    /// enum to minimize merge-conflict risk with parallel changes elsewhere.
    #[error("invalid effect shader: {0}")]
    InvalidEffectShader(String),
}
