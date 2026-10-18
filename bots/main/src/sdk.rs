//! Thin conversions between the cambc SDK and our SDK-free crates.

use crate::bc_core::champion::{builtin_kit, ChampionKind, DraftChoice, KitInfo};
use crate::bc_core::{BitGrid, Pos};
use cambc::{Bitboard, Error, ErrorCode, Result, Vec2};

pub fn pos(v: Vec2) -> Pos {
    Pos::new(v.x, v.y)
}

pub fn vec2(p: Pos) -> Vec2 {
    Vec2::new(p.x, p.y)
}

pub fn fuel() -> u64 {
    cambc::fuel_remaining().unwrap_or(0)
}

pub fn kind(t: cambc::ChampionType) -> ChampionKind {
    ChampionKind::from_u8(t as i32 as u8).unwrap_or(ChampionKind::Ryth)
}

pub fn choice(d: cambc::ChampionDraft) -> Option<DraftChoice> {
    DraftChoice::from_u8(d as i32 as u8)
}

pub fn draft_choice(c: DraftChoice) -> Result<cambc::ChampionDraft> {
    cambc::ChampionDraft::try_from(c as i32)
}

/// Export an SDK mask once into a guest-side bit grid (~1.7k fuel on 100×100).
pub fn grid(bb: &Bitboard, width: i32, height: i32) -> Result<BitGrid> {
    BitGrid::from_linear(width, height, bb.export_linear_buffer()?)
        .ok_or_else(|| local_error("bitboard export size mismatch"))
}

pub fn local_error(message: &str) -> Error {
    Error::Engine {
        raw_code: ErrorCode::InvalidArgument as i32,
        code: Some(ErrorCode::InvalidArgument),
        message: message.into(),
    }
}

/// Kit for a champion. Builtin table for now; reading `champion_designs()`
/// metadata so balance patches flow through automatically is a backlog item.
pub fn kit(kind: ChampionKind) -> KitInfo {
    builtin_kit(kind)
}

/// Damage per round from basic attacks plus the best damaging spell averaged
/// over its 8-turn cooldown.
pub fn dps(strength: f32, focus: f32, kit: &KitInfo) -> f32 {
    use crate::bc_core::champion::AbilityKind as K;
    let spell = kit
        .abilities
        .iter()
        .flatten()
        .filter(|a| matches!(a.kind, K::Damage | K::AoeDamage | K::Cc))
        .map(|a| a.modifier)
        .fold(0.0f32, f32::max);
    strength + focus * spell / 8.0
}
