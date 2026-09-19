//! Versioned, transport-neutral scene contracts for RendererCli.
//!
//! `SceneV1` is the one scene representation shared by every other crate in
//! this workspace: `daemon` stores and renders it, `cli` reads it from JSON
//! files, and `mcp` describes it to LLM clients as a JSON Schema. Keeping
//! the type and its validation rules here, instead of duplicated per crate,
//! is what keeps those surfaces in lockstep. `SceneV1::validate` is the
//! single gate untrusted scene input (from a file, a daemon request, or an
//! MCP tool call) must pass through before it reaches the renderer.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

mod limits;
pub use limits::*;
mod error;
pub use error::*;

mod fill;
pub use fill::*;

mod validate;
use validate::*;

mod timeline;
pub use timeline::*;
mod patch;
pub use patch::*;
#[cfg(test)]
mod test_support;

mod node;
pub use node::*;
mod scene;
pub use scene::*;
