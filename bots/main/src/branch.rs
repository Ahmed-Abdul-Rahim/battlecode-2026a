//! In-game decision study and the B net (harness/search/branch.py, plan Phase B). Two
//! byte-patched blocks (like settings.rs), never written by the bot:
//!
//! BRANCH_CFG: 8-byte magic, then u32 LE: from round (u32::MAX = off), champion kind, forced
//! choice (0 = keep the rules' choice, 1 = fight the nearest enemy, 2 = retreat), rounds to
//! hold, log flag (1 = print every contact decision and its features as a `D` line).
//!
//! NET_CFG: 8-byte magic, version (0 = off), threshold (i8, gain x 100), then an int8 MLP
//! IN -> HIDDEN (ReLU) -> 2: weights row-major, i32 biases, and one f32 scale per layer.
//! Its outputs are the predicted fitness gain of fight / retreat over the rules' choice; the
//! net overrules the rules only when the best one exceeds the threshold.
//!
//! A contact decision is any turn with an enemy champion within CONTACT tiles.

use crate::act::Visible;
use crate::bc_brain::context::Context;
use crate::bc_brain::experts::Intent;

pub const MAGIC: [u8; 8] = [0x42, 0x52, 0x43, 0x48, 0x02, 0x5A, 0xA5, 0x3C];
pub const NET_MAGIC: [u8; 8] = [0x42, 0x4E, 0x45, 0x54, 0x01, 0x3C, 0xA5, 0x5A];
pub const BLOCK_LEN: usize = 32;
pub const CONTACT: i32 = 6;
pub const IN: usize = 19;
pub const HIDDEN: usize = 16;
pub const OUT: usize = 2;
const NET_PARAMS: usize = IN * HIDDEN + HIDDEN * 4 + HIDDEN * OUT + OUT * 4 + 2 * 4;
pub const NET_LEN: usize = 16 + NET_PARAMS;

const fn block() -> [u8; BLOCK_LEN] {
    let mut b = [0u8; BLOCK_LEN];
    let mut i = 0;
    while i < MAGIC.len() {
        b[i] = MAGIC[i];
        i += 1;
    }
    // from round = u32::MAX (off)
    b[8] = 0xFF;
    b[9] = 0xFF;
    b[10] = 0xFF;
    b[11] = 0xFF;
    b
}

const fn net_block() -> [u8; NET_LEN] {
    let mut b = [0u8; NET_LEN];
    let mut i = 0;
    while i < NET_MAGIC.len() {
        b[i] = NET_MAGIC[i];
        i += 1;
    }
    b
}

#[used]
#[no_mangle]
static mut BRANCH_CFG: [u8; BLOCK_LEN] = block();

#[used]
#[no_mangle]
static mut NET_CFG: [u8; NET_LEN] = net_block();

#[derive(Clone, Copy)]
pub struct Cfg {
    pub from: u32,
    pub kind: u32,
    pub choice: u32,
    pub hold: u32,
    pub log: bool,
}

#[must_use]
pub fn cfg() -> Cfg {
    // SAFETY: never written by the bot; volatile so the patched bytes are read at run time.
    let b = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(BRANCH_CFG)) };
    let w = |i: usize| u32::from_le_bytes([b[8 + 4 * i], b[9 + 4 * i], b[10 + 4 * i], b[11 + 4 * i]]);
    Cfg { from: w(0), kind: w(1), choice: w(2), hold: w(3), log: w(4) == 1 }
}

/// The rules' choice class: 1 fight (engage / kite), 2 flee (retreat / walk home / step back),
/// 0 anything else (farming, tributing, shopping).
#[must_use]
pub fn class(intent: Intent) -> u32 {
    match intent {
        Intent::Engage(_) | Intent::Kite(_) => 1,
        Intent::Retreat | Intent::WalkHome | Intent::StepBack(_) => 2,
        _ => 0,
    }
}

/// Cheap features of a contact decision, each roughly in [-1, 1]; everything here was already
/// computed by the rules this turn.
#[must_use]
pub fn features(ctx: &Context, vis: &Visible, cls: u32) -> [f32; IN] {
    let mut f = [0.0f32; IN];
    let Some(e) = vis.enemies.first() else { return f };
    let near_e = vis.enemies.iter().filter(|o| o.dist <= CONTACT).count() as f32;
    let near_a = vis.allies.iter().filter(|a| a.dist <= CONTACT).count() as f32;
    let adv = ctx.enemy.map_or(1.0, |n| crate::bc_brain::router::fight_advantage(ctx, &n)).max(1e-3);
    f[0] = ctx.round as f32 / 2000.0;
    f[1] = ctx.hp / ctx.max_hp.max(1.0);
    f[2] = (ctx.gems as f32 / 100.0).min(2.0);
    f[3] = e.dist as f32 / CONTACT as f32;
    f[4] = e.hp / e.max_hp.max(1.0);
    f[5] = (e.gems as f32 / 100.0).min(2.0);
    f[6] = f32::from(u8::from(e.ranged));
    f[7] = f32::from(u8::from(ctx.ranged));
    f[8] = near_e / 3.0;
    f[9] = near_a / 3.0;
    f[10] = (adv.ln() / 3.0).clamp(-1.0, 1.0);
    f[11] = ctx.my_dps / (ctx.my_dps + e.dps).max(1.0);
    f[12] = (ctx.ally_dps_near / ctx.my_dps.max(1.0) / 3.0).min(2.0);
    f[13] = (ctx.threat_here / 100.0).min(2.0);
    f[14] = f32::from(u8::from(e.on_spawn));
    f[15] = f32::from(u8::from(e.tributing || e.recalling));
    f[16 + cls.min(2) as usize] = 1.0;
    f
}

fn net_on() -> bool {
    // SAFETY: as in cfg().
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(NET_CFG[8])) != 0 }
}

fn net_choice(f: &[f32; IN]) -> Option<u32> {
    // SAFETY: as in cfg().
    let b = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(NET_CFG)) };
    if b[8] == 0 {
        return None;
    }
    let threshold = f32::from(b[9] as i8) / 100.0;
    let p = &b[16..];
    let w1 = &p[..IN * HIDDEN];
    let at = IN * HIDDEN;
    let i32_at = |o: usize| i32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]]);
    let f32_at = |o: usize| f32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]]);
    let b1 = at;
    let w2 = b1 + HIDDEN * 4;
    let b2 = w2 + HIDDEN * OUT;
    let sc = b2 + OUT * 4;
    let (s1, s2) = (f32_at(sc), f32_at(sc + 4));
    // Inputs quantised to int8 at 1/64 steps.
    let x: [i32; IN] = core::array::from_fn(|i| (f[i] * 64.0).round().clamp(-127.0, 127.0) as i32);
    let mut h = [0f32; HIDDEN];
    for (j, hj) in h.iter_mut().enumerate() {
        let mut acc = i32_at(b1 + 4 * j);
        for i in 0..IN {
            acc += i32::from(w1[j * IN + i] as i8) * x[i];
        }
        *hj = (acc as f32 * s1).max(0.0);
    }
    let mut best = (threshold, None);
    for k in 0..OUT {
        // y = s2 * (b2 + sum w2 * h), h real-valued.
        let mut y = i32_at(b2 + 4 * k) as f32;
        for (j, hj) in h.iter().enumerate() {
            y += f32::from(p[w2 + k * HIDDEN + j] as i8) * hj;
        }
        y *= s2;
        if y > best.0 {
            best = (y, Some(k as u32 + 1));
        }
    }
    best.1
}

/// At a contact decision: log it, apply a branch (study) or the B net (play). Returns true
/// when the intent was forced (the caller then skips its micro overrides).
pub fn apply(intent: &mut Intent, round: u32, kind: u32, ctx: &Context, vis: &Visible) -> bool {
    let Some(e) = vis.enemies.first().filter(|e| e.dist <= CONTACT) else { return false };
    let c = cfg();
    let cls = class(*intent);
    let need_f = c.log || (c.from == u32::MAX && net_on());
    let f = if need_f { features(ctx, vis, cls) } else { [0.0; IN] };
    if c.log {
        let q: Vec<String> = f.iter().map(|v| ((v * 100.0).round() as i32).to_string()).collect();
        println!("D {round} {kind} {cls} {} {}", e.dist, q.join(" "));
    }
    let choice = if c.from != u32::MAX {
        if kind != c.kind || round < c.from || round >= c.from.saturating_add(c.hold) {
            return false;
        }
        c.choice
    } else {
        match net_choice(&f) {
            Some(k) if k != cls => k,
            _ => return false,
        }
    };
    // Only a forced turn prints (a null branch must replay the base game exactly).
    if c.log && choice != 0 {
        println!("B {round} {kind} {choice}");
    }
    match choice {
        1 => *intent = Intent::Engage(e.pos),
        2 => *intent = Intent::Retreat,
        _ => return false,
    }
    true
}
