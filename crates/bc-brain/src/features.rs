//! Feature extraction per schema group (ml/features/schema.yaml).
//!
//! The bot fills a [`RawObs`] only for the groups a head needs ([`GroupMask`]),
//! so unselected groups cost no fuel. Values are normalised to about [-1, 1]
//! and quantised with [`q`]. Missing distances encode as 1.0 ("far").

use crate::features_gen::{idx, Group, N_FEATURES};

/// Quantise to int8: round(clamp(v, −1, 1) · 127). Never produces −128.
#[must_use]
pub fn q(v: f32) -> i8 {
    (v.clamp(-1.0, 1.0) * 127.0).round() as i8
}

fn b(v: bool) -> i8 {
    if v {
        127
    } else {
        0
    }
}

fn far(d: Option<i32>, scale: f32) -> i8 {
    d.map_or(127, |d| q(d as f32 / scale))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupMask(pub u16);

impl GroupMask {
    pub const NONE: Self = Self(0);
    pub const ALL: Self = Self((1 << crate::features_gen::N_GROUPS) - 1);

    #[must_use]
    pub const fn with(self, g: Group) -> Self {
        Self(self.0 | 1 << g as u16)
    }

    #[must_use]
    pub const fn contains(self, g: Group) -> bool {
        self.0 >> g as u16 & 1 == 1
    }

    /// Summed per-turn fuel estimate of the selected groups.
    #[must_use]
    pub fn fuel(self) -> u32 {
        Group::ALL.iter().filter(|&&g| self.contains(g)).map(|g| g.fuel()).sum()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureVec {
    v: [i8; N_FEATURES],
    mask: GroupMask,
}

impl Default for FeatureVec {
    fn default() -> Self {
        Self { v: [0; N_FEATURES], mask: GroupMask::NONE }
    }
}

impl FeatureVec {
    #[must_use]
    pub fn values(&self) -> &[i8] {
        &self.v
    }

    #[must_use]
    pub fn mask(&self) -> GroupMask {
        self.mask
    }

    #[must_use]
    pub fn get(&self, i: usize) -> i8 {
        self.v[i]
    }

    /// Hex dump for telemetry (2 chars per feature).
    #[must_use]
    pub fn to_hex(&self) -> String {
        use core::fmt::Write;
        let mut s = String::with_capacity(2 * N_FEATURES);
        for &x in &self.v {
            let _ = write!(s, "{:02x}", x as u8);
        }
        s
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawSelf {
    pub hp: f32,
    pub max_hp: f32,
    pub strength: f32,
    pub focus: f32,
    pub ranged: bool,
    pub attack_range: i32,
    pub cooldowns: [i32; 2],
    pub can_move: bool,
    pub can_attack: bool,
    pub gems: u32,
    pub gold: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawEconomy {
    pub lifetime_gold: u32,
    pub ally_tributed: u32,
    pub enemy_tributed: u32,
    pub carried: u32,
    pub emblems: u32,
    pub has_altar_lock: bool,
    pub has_tp_crystal: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawUnit {
    pub dx: i32,
    pub dy: i32,
    pub dist: i32,
    pub hp_frac: f32,
    pub gems: u32,
    pub ranged: bool,
    pub dps: f32,
    pub stunned: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RawEnemies {
    /// Nearest first.
    pub units: Vec<RawUnit>,
    pub dps_in_range: f32,
    pub in_range3: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawAllies {
    pub n_visible: u32,
    pub nearest_dist: Option<i32>,
    pub nearest_hp_frac: f32,
    pub within3: u32,
    pub dps_near: f32,
    pub same_claim: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawCamps {
    pub n_visible: u32,
    pub nearest_dist: Option<i32>,
    pub nearest_hp_frac: f32,
    pub nearest_awake: bool,
    pub nearest_strength: f32,
    pub target_hp: f32,
    pub target_gems: u32,
    pub target_gold: u32,
    pub threat_here: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawObjectives {
    pub dist_altar: Option<i32>,
    pub dist_spawn: Option<i32>,
    pub dist_target: Option<i32>,
    pub capture_progress: u32,
    pub on_capture_tile: bool,
    pub altar_locked: bool,
    pub gems: u32,
    pub at_spawn: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawBelief {
    pub threat_here: f32,
    pub last_seen_age: Option<u32>,
    pub last_seen_dist: Option<i32>,
    pub unaccounted: u32,
    pub shared_sighting_age: Option<u32>,
    pub target_respawn_eta: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawTeam {
    pub aggression: u8,
    pub tribute_threshold: u8,
    pub help: bool,
    pub enemy_tributing: bool,
    pub am_leader: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawTheta {
    pub knob_aggression: u8,
    pub knob_retreat_hp: u8,
    pub knob_tribute_min: u8,
    pub film_active: bool,
    pub map_cells: u32,
    pub monolith: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawChampVecs {
    pub me: [i8; 16],
    pub enemy: [i8; 16],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawMeta {
    pub round: u32,
    pub fuel: u64,
    pub allies_alive_frac: f32,
    pub just_respawned: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RawObs {
    pub me: RawSelf,
    pub economy: RawEconomy,
    pub enemies: RawEnemies,
    pub allies: RawAllies,
    pub camps: RawCamps,
    pub objectives: RawObjectives,
    pub belief: RawBelief,
    pub team: RawTeam,
    pub theta: RawTheta,
    pub champ_vecs: RawChampVecs,
    pub meta: RawMeta,
}

/// Extract the selected groups. Unselected features stay 0.
#[must_use]
pub fn extract(raw: &RawObs, mask: GroupMask) -> FeatureVec {
    let mut f = FeatureVec { v: [0; N_FEATURES], mask };
    let v = &mut f.v;
    if mask.contains(Group::SelfState) {
        let s = &raw.me;
        v[idx::SELF_STATE_HP_FRAC] = q(if s.max_hp > 0.0 { s.hp / s.max_hp } else { 0.0 });
        v[idx::SELF_STATE_MAX_HP] = q(s.max_hp / 3000.0);
        v[idx::SELF_STATE_STRENGTH] = q(s.strength / 150.0);
        v[idx::SELF_STATE_FOCUS] = q(s.focus / 250.0);
        v[idx::SELF_STATE_IS_RANGED] = b(s.ranged);
        v[idx::SELF_STATE_ATTACK_RANGE] = q(s.attack_range as f32 / 3.0);
        v[idx::SELF_STATE_CD0] = q(s.cooldowns[0] as f32 / 8.0);
        v[idx::SELF_STATE_CD1] = q(s.cooldowns[1] as f32 / 8.0);
        v[idx::SELF_STATE_CAN_MOVE] = b(s.can_move);
        v[idx::SELF_STATE_CAN_ATTACK] = b(s.can_attack);
        v[idx::SELF_STATE_GEMS_CARRIED] = q(s.gems as f32 / 100.0);
        v[idx::SELF_STATE_GOLD] = q(s.gold as f32 / 300.0);
    }
    if mask.contains(Group::Economy) {
        let e = &raw.economy;
        v[idx::ECONOMY_LIFETIME_GOLD] = q(e.lifetime_gold as f32 / 5000.0);
        v[idx::ECONOMY_ALLY_TRIBUTED] = q(e.ally_tributed as f32 / 1000.0);
        v[idx::ECONOMY_ENEMY_TRIBUTED] = q(e.enemy_tributed as f32 / 1000.0);
        v[idx::ECONOMY_TRIBUTED_DIFF] =
            q((e.ally_tributed as f32 - e.enemy_tributed as f32) / 1000.0);
        v[idx::ECONOMY_GEMS_TO_WIN] =
            q(1000u32.saturating_sub(e.ally_tributed + e.carried) as f32 / 1000.0);
        v[idx::ECONOMY_EMBLEMS] = q(e.emblems as f32 / 6.0);
        v[idx::ECONOMY_HAS_ALTAR_LOCK] = b(e.has_altar_lock);
        v[idx::ECONOMY_HAS_TP_CRYSTAL] = b(e.has_tp_crystal);
    }
    if mask.contains(Group::Enemies) {
        let e = &raw.enemies;
        v[idx::ENEMIES_N_VISIBLE] = q(e.units.len() as f32 / 5.0);
        match e.units.first() {
            Some(n) => {
                v[idx::ENEMIES_NEAREST_DX] = q(n.dx as f32 / 8.0);
                v[idx::ENEMIES_NEAREST_DY] = q(n.dy as f32 / 8.0);
                v[idx::ENEMIES_NEAREST_DIST] = q(n.dist as f32 / 8.0);
                v[idx::ENEMIES_NEAREST_HP_FRAC] = q(n.hp_frac);
                v[idx::ENEMIES_NEAREST_GEMS] = q(n.gems as f32 / 100.0);
                v[idx::ENEMIES_NEAREST_RANGED] = b(n.ranged);
                v[idx::ENEMIES_NEAREST_DPS] = q(n.dps / 200.0);
                v[idx::ENEMIES_NEAREST_STUNNED] = b(n.stunned);
            }
            None => v[idx::ENEMIES_NEAREST_DIST] = 127,
        }
        v[idx::ENEMIES_DPS_IN_RANGE] = q(e.dps_in_range / 400.0);
        v[idx::ENEMIES_SECOND_DIST] = far(e.units.get(1).map(|u| u.dist), 8.0);
        v[idx::ENEMIES_IN_RANGE3] = q(e.in_range3 as f32 / 3.0);
    }
    if mask.contains(Group::Allies) {
        let a = &raw.allies;
        v[idx::ALLIES_N_VISIBLE] = q(a.n_visible as f32 / 5.0);
        v[idx::ALLIES_NEAREST_DIST] = far(a.nearest_dist, 8.0);
        v[idx::ALLIES_NEAREST_HP_FRAC] = q(a.nearest_hp_frac);
        v[idx::ALLIES_WITHIN3] = q(a.within3 as f32 / 3.0);
        v[idx::ALLIES_DPS_NEAR] = q(a.dps_near / 400.0);
        v[idx::ALLIES_SAME_CLAIM] = q(a.same_claim as f32 / 3.0);
    }
    if mask.contains(Group::Camps) {
        let c = &raw.camps;
        v[idx::CAMPS_N_VISIBLE] = q(c.n_visible as f32 / 6.0);
        v[idx::CAMPS_NEAREST_DIST] = far(c.nearest_dist, 8.0);
        v[idx::CAMPS_NEAREST_HP_FRAC] = q(c.nearest_hp_frac);
        v[idx::CAMPS_NEAREST_AWAKE] = b(c.nearest_awake);
        v[idx::CAMPS_NEAREST_DPS] = q(c.nearest_strength / 200.0);
        v[idx::CAMPS_TARGET_HP] = q(c.target_hp / 20000.0);
        v[idx::CAMPS_TARGET_GEMS] = q(c.target_gems as f32 / 200.0);
        v[idx::CAMPS_TARGET_GOLD] = q(c.target_gold as f32 / 300.0);
        let value = c.target_gems as f32 + c.target_gold as f32 / 3.0;
        v[idx::CAMPS_TARGET_VALUE_DENSITY] =
            q(if c.target_hp > 0.0 { value / (c.target_hp / 1000.0) / 100.0 } else { 0.0 });
        v[idx::CAMPS_CAMP_THREAT_HERE] = q(c.threat_here / 200.0);
    }
    if mask.contains(Group::Objectives) {
        let o = &raw.objectives;
        v[idx::OBJECTIVES_DIST_ALTAR] = far(o.dist_altar, 100.0);
        v[idx::OBJECTIVES_DIST_SPAWN] = far(o.dist_spawn, 100.0);
        v[idx::OBJECTIVES_DIST_TARGET] = far(o.dist_target, 100.0);
        v[idx::OBJECTIVES_CAPTURE_PROGRESS] = q(o.capture_progress as f32 / 5.0);
        v[idx::OBJECTIVES_ON_CAPTURE_TILE] = b(o.on_capture_tile);
        v[idx::OBJECTIVES_ALTAR_LOCKED] = b(o.altar_locked);
        v[idx::OBJECTIVES_TRIBUTE_TIME] = q(crate::policy::tribute_turns(o.gems) as f32 / 10.0);
        v[idx::OBJECTIVES_AT_SPAWN] = b(o.at_spawn);
    }
    if mask.contains(Group::Belief) {
        let be = &raw.belief;
        v[idx::BELIEF_THREAT_HERE] = q(be.threat_here / 400.0);
        v[idx::BELIEF_LAST_SEEN_AGE] = be.last_seen_age.map_or(127, |a| q(a as f32 / 12.0));
        v[idx::BELIEF_LAST_SEEN_DIST] = far(be.last_seen_dist, 50.0);
        v[idx::BELIEF_UNACCOUNTED] = q(be.unaccounted as f32 / 5.0);
        v[idx::BELIEF_SHARED_SIGHTING_AGE] =
            be.shared_sighting_age.map_or(127, |a| q(a as f32 / 3.0));
        v[idx::BELIEF_TARGET_RESPAWN_ETA] = q(be.target_respawn_eta as f32 / 150.0);
    }
    if mask.contains(Group::TeamParams) {
        let t = &raw.team;
        v[idx::TEAM_PARAMS_AGGRESSION] = q(f32::from(t.aggression) / 255.0);
        v[idx::TEAM_PARAMS_TRIBUTE_THRESHOLD] = q(f32::from(t.tribute_threshold) / 255.0);
        v[idx::TEAM_PARAMS_HELP_FLAG] = b(t.help);
        v[idx::TEAM_PARAMS_ENEMY_TRIBUTING] = b(t.enemy_tributing);
        v[idx::TEAM_PARAMS_AM_LEADER] = b(t.am_leader);
    }
    if mask.contains(Group::ThetaMatch) {
        let t = &raw.theta;
        v[idx::THETA_MATCH_KNOB_AGGRESSION] = q(f32::from(t.knob_aggression) / 255.0);
        v[idx::THETA_MATCH_KNOB_RETREAT_HP] = q(f32::from(t.knob_retreat_hp) / 100.0);
        v[idx::THETA_MATCH_KNOB_TRIBUTE_MIN] = q(f32::from(t.knob_tribute_min) / 100.0);
        v[idx::THETA_MATCH_FILM_ACTIVE] = b(t.film_active);
        v[idx::THETA_MATCH_MAP_SIZE] = q(t.map_cells as f32 / 10000.0);
        v[idx::THETA_MATCH_MAP_MONOLITH] = b(t.monolith);
    }
    if mask.contains(Group::ChampionVectors) {
        let me = idx::CHAMPION_VECTORS_ME_V;
        let en = idx::CHAMPION_VECTORS_ENEMY_V;
        v[me..me + idx::CHAMPION_VECTORS_ME_V_LEN].copy_from_slice(&clamp_vec(raw.champ_vecs.me));
        v[en..en + idx::CHAMPION_VECTORS_ENEMY_V_LEN]
            .copy_from_slice(&clamp_vec(raw.champ_vecs.enemy));
    }
    if mask.contains(Group::Meta) {
        let m = &raw.meta;
        v[idx::META_ROUND_FRAC] = q(m.round as f32 / 2000.0);
        v[idx::META_FUEL_FRAC] = q(m.fuel as f32 / 100_000.0);
        v[idx::META_ALLIES_ALIVE] = q(m.allies_alive_frac);
        v[idx::META_JUST_RESPAWNED] = b(m.just_respawned);
    }
    f
}

/// Keep champion vectors inside the quantiser's range (−128 is reserved).
fn clamp_vec(v: [i8; 16]) -> [i8; 16] {
    v.map(|x| x.max(-127))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features_gen::{FEATURE_NAMES, N_GROUPS, SCHEMA_HASH};

    fn busy_obs() -> RawObs {
        let unit = RawUnit {
            dx: 3,
            dy: -2,
            dist: 3,
            hp_frac: 0.5,
            gems: 40,
            ranged: true,
            dps: 80.0,
            stunned: true,
        };
        RawObs {
            me: RawSelf {
                hp: 500.0,
                max_hp: 1000.0,
                strength: 60.0,
                focus: 110.0,
                ranged: true,
                attack_range: 3,
                cooldowns: [4, 2],
                can_move: true,
                can_attack: true,
                gems: 33,
                gold: 120,
            },
            economy: RawEconomy {
                lifetime_gold: 900,
                ally_tributed: 300,
                enemy_tributed: 250,
                carried: 33,
                emblems: 1,
                has_altar_lock: true,
                has_tp_crystal: true,
            },
            enemies: RawEnemies { units: vec![unit, unit], dps_in_range: 80.0, in_range3: 1 },
            allies: RawAllies {
                n_visible: 2,
                nearest_dist: Some(2),
                nearest_hp_frac: 0.7,
                within3: 1,
                dps_near: 60.0,
                same_claim: 1,
            },
            camps: RawCamps {
                n_visible: 3,
                nearest_dist: Some(4),
                nearest_hp_frac: 0.4,
                nearest_awake: true,
                nearest_strength: 20.0,
                target_hp: 2700.0,
                target_gems: 44,
                target_gold: 180,
                threat_here: 20.0,
            },
            objectives: RawObjectives {
                dist_altar: Some(30),
                dist_spawn: Some(40),
                dist_target: Some(5),
                capture_progress: 2,
                on_capture_tile: true,
                altar_locked: true,
                gems: 33,
                at_spawn: true,
            },
            belief: RawBelief {
                threat_here: 40.0,
                last_seen_age: Some(3),
                last_seen_dist: Some(10),
                unaccounted: 2,
                shared_sighting_age: Some(1),
                target_respawn_eta: 30,
            },
            team: RawTeam {
                aggression: 160,
                tribute_threshold: 40,
                help: true,
                enemy_tributing: true,
                am_leader: true,
            },
            theta: RawTheta {
                knob_aggression: 128,
                knob_retreat_hp: 35,
                knob_tribute_min: 20,
                film_active: true,
                map_cells: 4900,
                monolith: true,
            },
            champ_vecs: RawChampVecs { me: [5; 16], enemy: [-9; 16] },
            meta: RawMeta {
                round: 700,
                fuel: 60_000,
                allies_alive_frac: 0.8,
                just_respawned: true,
            },
        }
    }

    #[test]
    fn every_feature_is_written_by_its_group() {
        let f = extract(&busy_obs(), GroupMask::ALL);
        for (i, name) in FEATURE_NAMES.iter().enumerate() {
            assert_ne!(f.get(i), 0, "{name} (index {i}) was never set");
        }
    }

    #[test]
    fn unselected_groups_stay_zero() {
        let mask = GroupMask::NONE.with(Group::SelfState);
        let f = extract(&busy_obs(), mask);
        let (start, end) = Group::SelfState.range();
        for (i, name) in FEATURE_NAMES.iter().enumerate() {
            assert_eq!(f.get(i) != 0, (start..end).contains(&i), "{name}");
        }
        assert_eq!(mask.fuel(), Group::SelfState.fuel());
    }

    #[test]
    fn quantiser_bounds() {
        assert_eq!(q(2.0), 127);
        assert_eq!(q(-5.0), -127);
        assert_eq!(q(0.5), 64);
        assert_eq!(clamp_vec([-128; 16])[0], -127);
    }

    #[test]
    fn group_ranges_tile_the_vector() {
        let mut next = 0;
        for g in Group::ALL {
            let (s, e) = g.range();
            assert_eq!(s, next);
            assert!(e > s);
            next = e;
        }
        assert_eq!(next, N_FEATURES);
        assert_eq!(GroupMask::ALL.0.count_ones() as usize, N_GROUPS);
        assert_ne!(SCHEMA_HASH, 0);
    }

    #[test]
    fn hex_dump_has_two_chars_per_feature() {
        assert_eq!(extract(&busy_obs(), GroupMask::ALL).to_hex().len(), 2 * N_FEATURES);
    }
}
