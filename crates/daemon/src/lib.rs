//! Persistent local scene state and its loopback-only daemon protocol.
//!
//! This crate is the stateful named-scene server the `cli` and `mcp` crates
//! both talk to over a line-delimited JSON protocol on a loopback TCP
//! socket ([`serve`] / [`DaemonClient`]): it holds an in-memory,
//! revisioned [`SceneV1`] per `scene_id` (see [`SceneStore`]) so a caller
//! can create a scene once and then `get`/`replace`/`patch`/`render` it by
//! name across many separate requests, instead of resending the whole
//! document every time. It also owns the one [`GpuRenderer`] used for both
//! named-scene and inline (one-shot, unstored) renders, and can be driven
//! in-process (`RendererDaemon`, used by `cli`'s `render`/`show` commands
//! and by `mcp`'s `render_scene` tool) without going over TCP at all.
//!
//! [`SceneV1`]: renderer_schema::SceneV1
//! [`GpuRenderer`]: renderer_core::GpuRenderer

#[cfg(doc)]
use crate::store::SceneStore;
mod endpoint;

mod error;
pub use error::*;
mod protocol;
pub use protocol::*;

mod client;
mod metrics;
mod store;
pub use client::*;

mod server;
pub use server::*;
mod daemon;
pub use daemon::*;

#[cfg(test)]
mod test_support;
