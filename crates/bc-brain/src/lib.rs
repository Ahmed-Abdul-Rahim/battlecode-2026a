//! Decision layer (docs/architecture.md, Phase 3).
//!
//! ```text
//! observe ─► meta-gate ─┬─ routine ─► default heuristic ─┐
//!                       └─ complex ─► MoE router ─► top-1 expert ─┴─► Intent
//! ```
//!
//! v0 gate and router are rule-based; the learned int8 heads ([`int8`]) slot in
//! behind the same interfaces once trained weights exist (`weights/*.bin`).

pub mod champvec;
pub mod context;
pub mod experts;
pub mod features;
#[rustfmt::skip]
pub mod features_gen;
pub mod gate;
pub mod int8;
pub mod policy;
pub mod rl;
pub mod router;
pub mod shop;

pub use context::Context;
pub use experts::{Expert, Intent};
pub use gate::Mode;
