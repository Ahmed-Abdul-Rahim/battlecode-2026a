//! Heuristic experts of the sparse MoE (P-050: top-1 routing means only the
//! chosen expert's code runs). Each turns the [`Context`] into one [`Intent`];
//! the bot's executor turns intents into SDK actions.

use crate::context::Context;
use bc_core::belief::GroupState;
use bc_core::Pos;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Expert {
    Farm = 0,
    Capture = 1,
    Tribute = 2,
    Fight = 3,
    Kite = 4,
    Retreat = 5,
    RecallShop = 6,
    Scout = 7,
    AltarLock = 8,
}

pub const N_EXPERTS: usize = 9;

impl Expert {
    pub const ALL: [Self; N_EXPERTS] = [
        Self::Farm,
        Self::Capture,
        Self::Tribute,
        Self::Fight,
        Self::Kite,
        Self::Retreat,
        Self::RecallShop,
        Self::Scout,
        Self::AltarLock,
    ];

    #[must_use]
    pub fn from_index(i: usize) -> Option<Self> {
        Self::ALL.get(i).copied()
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Farm => "farm",
            Self::Capture => "capture",
            Self::Tribute => "tribute",
            Self::Fight => "fight",
            Self::Kite => "kite",
            Self::Retreat => "retreat",
            Self::RecallShop => "recall_shop",
            Self::Scout => "scout",
            Self::AltarLock => "altar_lock",
        }
    }

    /// What this expert wants to do this turn.
    #[must_use]
    pub fn plan(self, ctx: &Context) -> Intent {
        match self {
            Self::Farm | Self::Scout => farm(ctx),
            Self::Capture => match ctx.target {
                Some(t) => Intent::Capture(t.index),
                None => Intent::Hold,
            },
            Self::Tribute => Intent::Tribute,
            Self::Fight => match ctx.enemy {
                Some(e) => Intent::Engage(e.pos),
                None => farm(ctx),
            },
            Self::Kite => match ctx.enemy {
                Some(e) => Intent::Kite(e.pos),
                None => farm(ctx),
            },
            Self::Retreat => Intent::Retreat,
            Self::RecallShop => Intent::Shop,
            // Altar locks are a backlog item; hold position rather than wander.
            Self::AltarLock => Intent::Hold,
        }
    }
}

/// High-level action for the executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    /// Approach camp group `i` and attack its camps (abilities included).
    ClearCamp(u8),
    /// Stand on a free spawn tile of group `i` until the capture completes.
    Capture(u8),
    /// Walk towards the group's spawn without engaging (it's respawning).
    Approach(u8),
    /// Walk to tribute range of an altar and start tributing.
    Tribute,
    /// Attack the enemy champion at this position.
    Engage(Pos),
    /// Stay at range from the enemy at this position, attacking when possible.
    Kite(Pos),
    /// Get away: recall when nothing can hit us, otherwise walk home.
    Retreat,
    /// Recall (if away from spawn) and buy.
    Shop,
    Hold,
}

fn farm(ctx: &Context) -> Intent {
    let Some(t) = ctx.target else { return Intent::Shop };
    match t.state {
        GroupState::RewardAvailable => Intent::Capture(t.index),
        GroupState::Respawning => Intent::Approach(t.index),
        GroupState::Alive | GroupState::Unknown => Intent::ClearCamp(t.index),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{EnemyNear, TargetGroup};

    fn target(state: GroupState) -> TargetGroup {
        TargetGroup {
            index: 3,
            priority_index: Some(0),
            dist: 10,
            state,
            hp_left: 1350.0,
            gems: 22,
            gold: 90,
            strength: 20.0,
        }
    }

    #[test]
    fn farm_follows_group_state() {
        let mut ctx = Context::neutral();
        ctx.target = Some(target(GroupState::Alive));
        assert_eq!(Expert::Farm.plan(&ctx), Intent::ClearCamp(3));
        ctx.target = Some(target(GroupState::RewardAvailable));
        assert_eq!(Expert::Farm.plan(&ctx), Intent::Capture(3));
        ctx.target = Some(target(GroupState::Respawning));
        assert_eq!(Expert::Farm.plan(&ctx), Intent::Approach(3));
        ctx.target = None;
        assert_eq!(Expert::Farm.plan(&ctx), Intent::Shop, "nothing to farm: go spend gold");
    }

    #[test]
    fn combat_experts_fall_back_to_farming_without_an_enemy() {
        let mut ctx = Context::neutral();
        ctx.target = Some(target(GroupState::Alive));
        assert_eq!(Expert::Fight.plan(&ctx), Intent::ClearCamp(3));
        ctx.enemy = Some(EnemyNear {
            pos: Pos::new(12, 10),
            dist: 2,
            hp: 500.0,
            max_hp: 900.0,
            dps: 40.0,
            gems: 30,
            ranged: true,
        });
        assert_eq!(Expert::Fight.plan(&ctx), Intent::Engage(Pos::new(12, 10)));
        assert_eq!(Expert::Kite.plan(&ctx), Intent::Kite(Pos::new(12, 10)));
    }

    #[test]
    fn expert_indices_round_trip() {
        for (i, e) in Expert::ALL.iter().enumerate() {
            assert_eq!(Expert::from_index(i), Some(*e));
            assert_eq!(*e as usize, i);
        }
        assert_eq!(Expert::from_index(N_EXPERTS), None);
    }
}
