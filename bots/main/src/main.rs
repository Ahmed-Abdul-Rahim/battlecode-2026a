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
mod map_plans;
mod sdk;
mod learned;
mod skirmish;
mod telemetry;
mod world;

fn main() -> cambc::Result<()> {
    let mut bot: Option<bot::Bot> = None;
    loop {
        if cambc::is_draft() {
            if let Err(e) = draft::play() {
                telemetry::error("draft", &e);
            }
        } else {
            if bot.is_none() {
                match bot::Bot::new() {
                    Ok(b) => bot = Some(b),
                    Err(e) => telemetry::error("init", &e),
                }
            }
            if let Some(b) = bot.as_mut() {
                if let Err(e) = b.turn() {
                    telemetry::error("turn", &e);
                }
            }
        }
        cambc::end_turn();
    }
}
