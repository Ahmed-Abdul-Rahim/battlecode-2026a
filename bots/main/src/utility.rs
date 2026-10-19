//! Utility arbiter: one learned score per concrete option instead of a rule cascade.
//!
//! The rules (gate -> router -> intent rewrites) pick one behaviour; a champion near a
//! resource *and* a threat flips between them. Here every turn a few options are scored
//! together, score_k = U_k . phi, over shared features phi (our health, damage we take
//! from camps vs champions, allies / enemies near, an ally in trouble, bag, distance home,
//! ranged). U lives in learned.rs / the policy pack per map and is evolved by ES; all-zero
//! U scores every option 0 and keeps the rule intent, i.e. identity.

use crate::act::Visible;
use crate::bc_brain::{Context, Intent};
use crate::bc_core::Pos;

/// Options, in U row order.
pub const KEEP: usize = 0;
pub const STEP_BACK: usize = 1;
pub const SUPPORT: usize = 2;
pub const FIGHT: usize = 3;
pub const WALK_HOME: usize = 4;
pub const RECALL: usize = 5;
pub const OPTIONS: usize = 6;
pub const PHI: usize = 10;

/// The arbiter's pick: a new intent and the option index (for telemetry), or None to keep.
#[must_use]
pub fn choose(ctx: &Context, vis: &Visible, my_pos: Pos, my_range: i32, u: &[f32], intent: Intent) -> Option<(Intent, usize)> {
    if u.len() < OPTIONS * PHI || u.iter().all(|&w| w == 0.0) {
        return None;
    }
    let max_hp = ctx.max_hp.max(1.0);
    // Awake camps that hit us (no ally closer to them).
    let camp_on_me = vis
        .camps
        .iter()
        .filter(|c| c.awake && c.hp > 0.0 && c.dist <= 3)
        .filter(|c| !vis.allies.iter().any(|a| a.pos.cheb(c.pos) < c.dist))
        .min_by_key(|c| c.dist);
    let camp_dps: f32 = vis
        .camps
        .iter()
        .filter(|c| c.awake && c.hp > 0.0 && c.dist <= 3)
        .filter(|c| !vis.allies.iter().any(|a| a.pos.cheb(c.pos) < c.dist))
        .map(|c| c.strength * 0.7)
        .sum();
    let enemy_dps = vis.reachable_dps(my_pos, 1);
    let enemies_near = vis.enemies.iter().filter(|e| e.dist <= 5 && !e.on_spawn).count();
    let allies_near = vis.allies.iter().filter(|a| a.dist <= 4).count();
    let ally_in_trouble = vis.allies.iter().filter(|a| a.dist <= 8 && a.hp < 0.5 * a.max_hp).find_map(|a| {
        vis.enemies.iter().filter(|e| e.pos.cheb(a.pos) <= e.range + 1 && !e.on_spawn).min_by_key(|e| e.pos.cheb(a.pos))
    });
    let weakest = vis
        .enemies
        .iter()
        .filter(|e| !e.on_spawn && e.dist <= my_range + 2)
        .min_by(|a, b| a.hp.total_cmp(&b.hp));
    let melee_on_me = vis.enemies.iter().filter(|e| !e.ranged && e.dist <= 2 && !e.on_spawn).min_by_key(|e| e.dist);
    let phi = [
        1.0,
        ctx.hp_frac(),
        (camp_dps / max_hp * 10.0).min(3.0),
        (enemy_dps / max_hp * 10.0).min(3.0),
        enemies_near as f32 / 3.0,
        allies_near as f32 / 3.0,
        f32::from(u8::from(ally_in_trouble.is_some())),
        ctx.gems as f32 / 100.0,
        ctx.dist_spawn.map_or(1.0, |d| f32::from(d) / 20.0),
        f32::from(u8::from(ctx.ranged)),
    ];
    let score = |k: usize| -> f32 { u[k * PHI..(k + 1) * PHI].iter().zip(phi).map(|(w, x)| w * x).sum() };
    let step_from = camp_on_me.map(|c| c.pos).or_else(|| melee_on_me.map(|e| e.pos));
    let candidates: [(usize, Option<Intent>); OPTIONS] = [
        (KEEP, Some(intent)),
        (STEP_BACK, step_from.map(Intent::StepBack)),
        (SUPPORT, ally_in_trouble.map(|e| Intent::Engage(e.pos))),
        (FIGHT, weakest.map(|e| Intent::Engage(e.pos))),
        (WALK_HOME, (!ctx.at_spawn).then_some(Intent::WalkHome)),
        (RECALL, (!ctx.at_spawn).then_some(Intent::Retreat)),
    ];
    let keep = score(KEEP);
    let (best_k, best) = candidates
        .iter()
        .filter_map(|(k, i)| i.map(|i| (*k, i)))
        .map(|(k, i)| (k, i, score(k)))
        .fold((KEEP, intent, keep), |acc, x| if x.2 > acc.2 { x } else { acc })
        .into_tuple();
    (best_k != KEEP).then_some((best, best_k))
}

trait IntoTuple {
    fn into_tuple(self) -> (usize, Intent);
}

impl IntoTuple for (usize, Intent, f32) {
    fn into_tuple(self) -> (usize, Intent) {
        (self.0, self.1)
    }
}
