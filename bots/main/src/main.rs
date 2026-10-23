//! Submission bot. Draft turns go to `draft`, game turns to `bot::Bot`.
//! All decision logic lives in the SDK-free crates under crates/; this crate
//! only adapts the SDK and executes.
//!
//! cambc bots must be self-contained, so the crates are vendored into
//! src/bc_*/ by `python -m bctools.sync`. Never edit those copies.

// Vendored copies: formatted at the source in crates/, so rustfmt skips them here.
#[rustfmt::skip]
mod bc_brain;
#[rustfmt::skip]
mod bc_core;
#[rustfmt::skip]
mod bc_draft;
#[rustfmt::skip]
mod bc_path;

mod act;
mod bot;
mod draft;
mod learned;
mod map_plans;
mod pack;
mod plan_build;
mod sdk;
mod skirmish;
mod telemetry;
mod utility;
mod world;

/// A turn right after start-up needs about this much fuel (observe, target, path, act).
const INIT_TURN_MIN_FUEL: u64 = 70_000;

fn main() -> cambc::Result<()> {
    let mut bot: Option<bot::Bot> = None;
    loop {
        if cambc::is_draft() {
            if let Err(e) = draft::play() {
                telemetry::error("draft", &e);
            }
        } else {
            let fresh = bot.is_none();
            if fresh {
                match bot::Bot::new() {
                    Ok(b) => bot = Some(b),
                    Err(e) => telemetry::error("init", &e),
                }
            }
            // Start-up on big maps (map, groups, pack: ~150k+) left the first turn too little
            // for a full turn, which then ran out of fuel and spilled over two turns. At spawn
            // on turn 1 there is nothing to lose: end it and play turn 2 with full fuel.
            let starved = fresh && sdk::fuel() < INIT_TURN_MIN_FUEL;
            if let (false, Some(b)) = (starved, bot.as_mut()) {
                if let Err(e) = b.turn() {
                    telemetry::error("turn", &e);
                }
            }
        }
        cambc::end_turn();
    }
}
