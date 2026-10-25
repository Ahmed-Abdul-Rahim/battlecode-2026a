//! Branch block for the in-game decision study (harness/search/branch.py, plan Phase B):
//! byte-patched into the compiled file like settings.rs, never written by the bot.
//!
//! Layout: 8-byte magic, then u32 LE words: from round (u32::MAX = off), champion kind,
//! forced choice (0 = keep the rules' choice, 1 = fight the nearest enemy, 2 = retreat),
//! rounds to hold, log flag (1 = print every contact decision as a `D` line).
//!
//! A contact decision is any turn with an enemy champion within CONTACT tiles.

use crate::bc_brain::experts::Intent;
use crate::bc_core::Pos;

pub const MAGIC: [u8; 8] = [0x42, 0x52, 0x43, 0x48, 0x02, 0x5A, 0xA5, 0x3C];
pub const BLOCK_LEN: usize = 32;
pub const CONTACT: i32 = 6;

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

#[used]
#[no_mangle]
static mut BRANCH_CFG: [u8; BLOCK_LEN] = block();

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

/// At a contact decision: log it, and inside the branch window replace the rules' intent.
/// Returns true when the intent was forced (the caller then skips its micro overrides).
pub fn apply(intent: &mut Intent, round: u32, kind: u32, nearest: Option<(Pos, i32)>, gems: u32) -> bool {
    let c = cfg();
    let Some((pos, dist)) = nearest.filter(|&(_, d)| d <= CONTACT) else { return false };
    if c.log {
        println!("D {round} {kind} {} {dist} {gems}", class(*intent));
    }
    if c.from == u32::MAX || kind != c.kind || round < c.from || round >= c.from.saturating_add(c.hold) {
        return false;
    }
    // Only a forced turn prints (a null branch must replay the base game exactly).
    if c.log && c.choice != 0 {
        println!("B {round} {kind} {}", c.choice);
    }
    match c.choice {
        1 => *intent = Intent::Engage(pos),
        2 => *intent = Intent::Retreat,
        _ => return false,
    }
    true
}
