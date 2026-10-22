//! Debug telemetry: one compact JSON line per champion turn, prefixed `T `.
//!
//! Everything here runs inside `debug_block!`, which is free in local debug
//! runs (`cambc run`) and compiled out in release / on the platform. tools/
//! bctools/logs.py parses these lines; keep the keys stable.

use crate::bc_core::Pos;

pub struct TurnLog<'a> {
    pub round: u32,
    pub champ: &'a str,
    pub pos: Pos,
    pub hp: f32,
    pub gold: u32,
    pub gems: u32,
    pub lifetime_gold: u32,
    pub ally_tributed: u32,
    pub mode: &'a str,
    pub expert: &'a str,
    pub intent: String,
    pub tier: &'a str,
    pub target: Option<u8>,
    pub fuel_start: u64,
    pub fuel_end: u64,
    pub stages: [u64; crate::bc_core::fuel::STAGES],
    pub leader: bool,
    pub features: Option<String>,
    /// Pre-formatted JSON arrays: visible enemies `[[x,y,hp%],…]`, camps `[[x,y,group,awake],…]`.
    pub enemies: String,
    pub camps: String,
}

fn hex(bytes: &[u8]) -> String {
    use core::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// One-off terrain and camp-group dump for the renderer (debug runs only).
/// Layers are the SDK's linear bit layout, hex encoded.
pub fn map_dump(world: &crate::world::World) {
    cambc::debug_block! {
        let t = &world.terrain;
        let groups: Vec<String> = world
            .groups
            .iter()
            .map(|g| {
                let tiles: Vec<String> = g.tiles.iter().map(|p| format!("[{},{}]", p.x, p.y)).collect();
                format!("{{\"c\":[{},{}],\"gems\":{},\"gold\":{},\"tiles\":[{}]}}", g.center.x, g.center.y, g.gems, g.gold, tiles.join(","))
            })
            .collect();
        println!(
            "M {{\"w\":{},\"h\":{},\"walls\":\"{}\",\"holes\":\"{}\",\"bushes\":\"{}\",\"altars\":\"{}\",\"ally_spawn\":\"{}\",\"enemy_spawn\":\"{}\",\"groups\":[{}]}}",
            world.width, world.height,
            hex(t.walls.as_bytes()), hex(t.holes.as_bytes()), hex(t.bushes.as_bytes()),
            hex(t.altars.as_bytes()), hex(t.ally_spawn.as_bytes()), hex(t.enemy_spawn.as_bytes()),
            groups.join(",")
        );
    }
}

/// Print a turn line. Call only from inside `cambc::debug_block!`.
pub fn turn(log: &TurnLog<'_>) {
    let target = log.target.map_or_else(|| "null".to_string(), |t| t.to_string());
    let feats = log.features.as_ref().map_or_else(String::new, |f| format!(",\"x\":\"{f}\""));
    println!(
        "T {{\"r\":{},\"c\":\"{}\",\"p\":[{},{}],\"hp\":{:.3},\"g\":{},\"gm\":{},\"lg\":{},\"at\":{},\"m\":\"{}\",\"e\":\"{}\",\"i\":\"{}\",\"tier\":\"{}\",\"tg\":{},\"f0\":{},\"f1\":{},\"fs\":{:?},\"ld\":{},\"en\":{},\"cp\":{}{}}}",
        log.round, log.champ, log.pos.x, log.pos.y, log.hp, log.gold, log.gems, log.lifetime_gold,
        log.ally_tributed, log.mode, log.expert, log.intent, log.tier, target, log.fuel_start,
        log.fuel_end, log.stages, u8::from(log.leader), log.enemies, log.camps, feats
    );
}

/// One learned decision (ml/rl): arm, the network's input, the picks per head, their summed
/// log-probability (x1000) and each head's mask of allowed actions (bits).
pub fn rl(round: u32, champ: &str, arm: u8, x: &str, picks: &[usize], logp: f32, masks: &[u32]) {
    cambc::debug_block! {
        println!(
            "R {{\"r\":{round},\"c\":\"{champ}\",\"arm\":{arm},\"x\":\"{x}\",\"a\":{picks:?},\"lp\":{},\"m\":{masks:?}}}",
            (logp * 1000.0).round() as i64
        );
    }
}

pub fn event(round: u32, champ: &str, kind: &str, value: i64) {
    cambc::debug_block! {
        println!("E {{\"r\":{round},\"c\":\"{champ}\",\"k\":\"{kind}\",\"v\":{value}}}");
    }
}

pub fn error(context: &str, error: &cambc::Error) {
    cambc::debug_block! {
        println!("X {{\"ctx\":\"{context}\",\"err\":\"{}\"}}", error.to_string().replace('"', "'"));
    }
}

pub fn draw_path(from: Pos, to: Pos, rgb: u32) {
    cambc::debug_block! {
        let _ = cambc::debug::draw_line(crate::sdk::vec2(from), crate::sdk::vec2(to), rgb);
    }
}
