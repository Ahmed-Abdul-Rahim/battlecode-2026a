//! Draft-phase logic. Runs with 10× fuel on the first draft turn and sees the
//! whole map, so it does the per-match adaptation: score champions for this
//! map and opponent, then write θ_match (camp priority, knobs, FiLM, champion
//! vectors) to draft memory for every champion bot.

pub mod map;
pub mod scorer;
pub mod theta;

pub use map::{CampKind, GroupInfo, MapDescriptor, MapInfo};
pub use scorer::{best_pick, DraftState, ScorerWeights};
pub use theta::build_theta;
