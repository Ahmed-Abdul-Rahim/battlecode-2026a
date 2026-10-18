//! Meta-gate: decides, before spending fuel on the MoE, whether this turn is
//! routine (cheap default heuristic) or complex (route through the MoE).
//! Adaptive computation in the spirit of early-exit networks (P-053).
//!
//! Hard overrides always win; then a learned head if loaded; else rules.

use crate::context::Context;
use crate::experts::Expert;
use crate::features::FeatureVec;
use crate::int8::Mlp;

/// Below this much fuel left in the turn, never take the expensive path.
pub const LOW_FUEL: u64 = 25_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Routine,
    Complex,
    Override(Expert),
}

/// Safety rules that no learned head may overrule.
#[must_use]
pub fn hard_override(ctx: &Context) -> Option<Mode> {
    if !ctx.alive {
        return Some(Mode::Override(Expert::RecallShop));
    }
    if ctx.low_hp() && ctx.in_danger() {
        return Some(Mode::Override(Expert::Retreat));
    }
    if ctx.fuel_remaining < LOW_FUEL {
        return Some(Mode::Routine);
    }
    None
}

/// v0 rule-based gate.
#[must_use]
pub fn rule_gate(ctx: &Context) -> Mode {
    if let Some(m) = hard_override(ctx) {
        return m;
    }
    if ctx.enemy.is_some() || ctx.threat_here > 0.0 || ctx.team.help {
        Mode::Complex
    } else {
        Mode::Routine
    }
}

/// Learned gate: a 2-logit int8 head (routine, complex), behind the overrides.
pub struct LearnedGate {
    pub mlp: Mlp,
}

impl LearnedGate {
    #[must_use]
    pub fn decide(&self, ctx: &Context, features: &FeatureVec) -> Mode {
        if let Some(m) = hard_override(ctx) {
            return m;
        }
        match self.mlp.argmax(features.values(), None) {
            Some(1) => Mode::Complex,
            Some(_) => Mode::Routine,
            None => rule_gate(ctx),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::EnemyNear;
    use bc_core::Pos;

    #[test]
    fn overrides_take_precedence() {
        let mut ctx = Context::neutral();
        ctx.alive = false;
        assert_eq!(rule_gate(&ctx), Mode::Override(Expert::RecallShop));
        ctx.alive = true;
        ctx.hp = 100.0;
        ctx.camp_threat_here = 60.0;
        assert_eq!(rule_gate(&ctx), Mode::Override(Expert::Retreat));
        ctx.hp = 1000.0;
        ctx.fuel_remaining = 10_000;
        ctx.threat_here = 99.0;
        assert_eq!(rule_gate(&ctx), Mode::Routine, "low fuel beats threat");
    }

    #[test]
    fn enemies_make_turns_complex() {
        let mut ctx = Context::neutral();
        assert_eq!(rule_gate(&ctx), Mode::Routine);
        ctx.enemy = Some(EnemyNear {
            pos: Pos::new(0, 0),
            dist: 6,
            hp: 1.0,
            max_hp: 1.0,
            dps: 1.0,
            gems: 0,
            ranged: false,
        });
        assert_eq!(rule_gate(&ctx), Mode::Complex);
    }
}
