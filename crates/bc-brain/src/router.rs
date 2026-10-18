//! Expert selection: the cheap routine policy, the v0 rule router for complex
//! turns, and a learned top-1 router with FiLM conditioning from θ_match.

use crate::context::{Context, EnemyNear};
use crate::experts::{Expert, N_EXPERTS};
use crate::features::FeatureVec;
use crate::gate::Mode;
use crate::int8::Mlp;
use crate::policy::should_tribute;
use bc_core::belief::GroupState;

/// Default heuristic for routine turns: shop, bank, heal, capture, farm.
#[must_use]
pub fn routine(ctx: &Context) -> Expert {
    if ctx.at_spawn && ctx.can_afford_shard {
        return Expert::RecallShop;
    }
    if should_tribute(ctx) {
        return Expert::Tribute;
    }
    if ctx.low_hp() && !ctx.at_spawn {
        return Expert::RecallShop;
    }
    let far_from_spawn = ctx.dist_spawn.is_none_or(|d| d > 15);
    if ctx.gold >= ctx.knobs.recall_gold && ctx.gems < 10 && far_from_spawn {
        return Expert::RecallShop;
    }
    if ctx.target.is_some_and(|t| t.state == GroupState::RewardAvailable) {
        return Expert::Capture;
    }
    Expert::Farm
}

/// Ratio of how long the enemy needs to kill us to how long we need to kill it.
/// Above 1 we win the duel; allies near us add their damage (Lanchester-style).
#[must_use]
pub fn fight_advantage(ctx: &Context, e: &EnemyNear) -> f32 {
    let our_dps = (ctx.my_dps + ctx.ally_dps_near).max(1.0);
    let their_dps = (e.dps * f32::from(ctx.enemies_in_range.max(1))).max(1.0);
    let time_to_kill_them = e.hp / our_dps;
    let time_to_kill_us = ctx.hp / their_dps;
    time_to_kill_us / time_to_kill_them.max(0.1)
}

/// v0 rule router for complex turns.
#[must_use]
pub fn rule_route(ctx: &Context) -> Expert {
    if let Some(e) = ctx.enemy {
        let advantage = fight_advantage(ctx, &e);
        // Aggression (team channel overrides the draft knob) shrinks the margin.
        let aggression =
            if ctx.team.version > 0 { ctx.team.aggression } else { ctx.knobs.aggression };
        let margin =
            f32::from(ctx.knobs.fight_margin_pct) / 100.0 * (255.0 - f32::from(aggression)) / 127.0;
        // A gem carrier is worth chasing: the killer takes the whole bag.
        let bounty = if e.gems >= 20 { 0.2 } else { 0.0 };
        if advantage >= 1.0 + margin - bounty {
            return Expert::Fight;
        }
        if e.dist <= 4 {
            return if ctx.ranged && !e.ranged && !ctx.low_hp() {
                Expert::Kite
            } else {
                Expert::Retreat
            };
        }
    }
    if ctx.gems > 0 && ctx.threat_here > 0.0 && should_tribute(ctx) {
        return Expert::Tribute;
    }
    routine(ctx)
}

/// Pick the expert for this turn given the gate's mode.
/// What the learned router needs: the network, this turn's features, and the
/// per-match FiLM table from θ_match.
pub type LearnedInputs<'a> = (&'a LearnedRouter, &'a FeatureVec, &'a [(i8, i8)]);

#[must_use]
pub fn select(mode: Mode, ctx: &Context, learned: Option<LearnedInputs<'_>>) -> Expert {
    match mode {
        Mode::Override(e) => e,
        Mode::Routine => routine(ctx),
        Mode::Complex => match learned {
            Some((router, features, film)) => {
                router.route(features, film).unwrap_or_else(|| rule_route(ctx))
            }
            None => rule_route(ctx),
        },
    }
}

/// Learned top-1 router: an int8 MLP with one logit per expert, hidden layers
/// modulated by the per-match FiLM parameters from draft memory.
pub struct LearnedRouter {
    pub mlp: Mlp,
}

impl LearnedRouter {
    #[must_use]
    pub fn route(&self, features: &FeatureVec, film: &[(i8, i8)]) -> Option<Expert> {
        if self.mlp.output_len() != N_EXPERTS {
            return None;
        }
        self.mlp.argmax(features.values(), Some(film)).and_then(Expert::from_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::TargetGroup;
    use bc_core::Pos;

    fn enemy(hp: f32, dps: f32, ranged: bool) -> EnemyNear {
        EnemyNear { pos: Pos::new(13, 10), dist: 3, hp, max_hp: hp, dps, gems: 0, ranged }
    }

    #[test]
    fn routine_priorities() {
        let mut ctx = Context::neutral();
        ctx.at_spawn = true;
        ctx.can_afford_shard = true;
        assert_eq!(routine(&ctx), Expert::RecallShop, "spend gold on spawn");
        ctx.at_spawn = false;
        ctx.gems = 90;
        assert_eq!(routine(&ctx), Expert::Tribute);
        ctx.gems = 0;
        ctx.hp = 200.0;
        assert_eq!(routine(&ctx), Expert::RecallShop, "heal");
        ctx.hp = 1000.0;
        ctx.target = Some(TargetGroup {
            index: 1,
            priority_index: None,
            dist: 3,
            state: GroupState::RewardAvailable,
            hp_left: 0.0,
            gems: 22,
            gold: 90,
            strength: 20.0,
        });
        assert_eq!(routine(&ctx), Expert::Capture);
    }

    #[test]
    fn fights_winnable_duels_and_avoids_bad_ones() {
        let mut ctx = Context::neutral();
        ctx.enemies_in_range = 1;
        ctx.enemy = Some(enemy(300.0, 40.0, false));
        assert_eq!(rule_route(&ctx), Expert::Fight);
        ctx.enemy = Some(enemy(3000.0, 300.0, false));
        ctx.ranged = true;
        assert_eq!(rule_route(&ctx), Expert::Kite, "ranged vs melee: kite");
        ctx.ranged = false;
        assert_eq!(rule_route(&ctx), Expert::Retreat);
    }

    #[test]
    fn allies_tip_the_balance() {
        let mut ctx = Context::neutral();
        ctx.enemies_in_range = 1;
        let strong = enemy(1500.0, 60.0, false);
        ctx.enemy = Some(strong);
        assert!(fight_advantage(&ctx, &strong) < 1.5);
        ctx.ally_dps_near = 150.0;
        assert!(fight_advantage(&ctx, &strong) > 1.5);
        assert_eq!(rule_route(&ctx), Expert::Fight);
    }

    #[test]
    fn select_respects_mode() {
        let ctx = Context::neutral();
        assert_eq!(select(Mode::Override(Expert::Retreat), &ctx, None), Expert::Retreat);
        assert_eq!(select(Mode::Routine, &ctx, None), Expert::Farm);
    }
}
