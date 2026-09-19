//! Off-screen wgpu renderer for normalized RendererCli scenes.
#![allow(unexpected_cfgs)] // `cargo llvm-cov` supplies `cfg(coverage)`/`cfg(coverage_nightly)`.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

mod analytic_geometry;
mod animation;
mod assets;
mod composition;
mod error;
mod fill;
mod gif;
mod gpu;
mod limits;
mod pipelines;
mod shaders;
mod svg;
mod tessellation;
#[cfg(test)]
mod test_support;
mod util;
mod vertex;

pub use error::{RenderError, RenderedImage};
pub use gpu::GpuRenderer;
