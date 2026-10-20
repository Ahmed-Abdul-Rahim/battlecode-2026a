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

/// Live design stats per champion kind (base and growth of health, strength, focus; attack
/// range), read once from `champion_designs()` so balance patches flow through.
type LiveStats = [f32; 6];
static LIVE: std::sync::OnceLock<Vec<(ChampionKind, LiveStats, i32)>> = std::sync::OnceLock::new();

fn live_stats() -> &'static [(ChampionKind, LiveStats, i32)] {
    LIVE.get_or_init(|| {
        let Ok(designs) = cambc::champion_designs() else { return Vec::new() };
        designs
            .iter()
            .filter_map(|d| {
                let k = ChampionKind::from_u8(d.champion_name().ok()? as i32 as u8)?;
                let s = d.base_stats().ok()?;
                let v = [
                    s.base_health,
                    s.health_growth,
                    s.base_strength,
                    s.strength_growth,
                    s.base_focus,
                    s.focus_growth,
                ];
                Some((k, v, s.base_attack_range))
            })
            .collect()
    })
}

/// Kit for a champion: the builtin table (abilities, passive) with the engine's live stats.
pub fn kit(kind: ChampionKind) -> KitInfo {
    let mut k = builtin_kit(kind);
    if let Some((_, s, range)) = live_stats().iter().find(|(c, ..)| *c == kind) {
        [
            k.base_health,
            k.health_growth,
            k.base_strength,
            k.strength_growth,
            k.base_focus,
            k.focus_growth,
        ] = *s;
        if *range > 0 {
            k.attack_range = *range;
        }
    }
    k
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

/// Kit has a stun (Cc ability): Tavi, Prism, Cassi, Ryth, Ren.
pub fn has_stun(kind: ChampionKind) -> bool {
    use crate::bc_core::champion::AbilityKind as K;
    kit(kind).abilities.iter().flatten().any(|a| a.kind == K::Cc)
}

/// Fight-horizon damage for stun / attack-reset kits (STUN_THREAT): the spell and the
/// reset's extra hit land in the first rounds of a skirmish, so average them over 6
/// rounds instead of the 8-turn cooldown.
pub fn dps_burst(strength: f32, focus: f32, kit: &KitInfo) -> f32 {
    use crate::bc_core::champion::AbilityKind as K;
    let abilities = || kit.abilities.iter().flatten();
    let bursty = abilities().any(|a| matches!(a.kind, K::Cc | K::AttackReset));
    if !bursty {
        return dps(strength, focus, kit);
    }
    let spell = abilities()
        .filter(|a| matches!(a.kind, K::Damage | K::AoeDamage | K::Cc))
        .map(|a| a.modifier)
        .fold(0.0f32, f32::max);
    let reset = if abilities().any(|a| a.kind == K::AttackReset) { strength } else { 0.0 };
    strength + (focus * spell + reset) / 6.0
}
