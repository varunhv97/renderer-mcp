//! Versioned, transport-neutral scene contracts for RendererCli.
//!
//! `SceneV1` is the one scene representation shared by every other crate in
//! this workspace: `daemon` stores and renders it, `cli` reads it from JSON
//! files, and `mcp` describes it to LLM clients as a JSON Schema. Keeping
//! the type and its validation rules here, instead of duplicated per crate,
//! is what keeps those surfaces in lockstep. `SceneV1::validate` is the
//! single gate untrusted scene input (from a file, a daemon request, or an
//! MCP tool call) must pass through before it reaches the renderer.

mod error;
mod fill;
mod limits;
mod node;
mod patch;
mod scene;
#[cfg(test)]
mod test_support;
mod timeline;
mod validate;

pub use error::*;
pub use fill::*;
pub use limits::*;
pub use node::*;
pub use patch::*;
pub use scene::*;
pub use timeline::*;
