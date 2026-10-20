//! Executor: turns an [`Intent`] into SDK actions under the turn rules.
//!
//! Rules that shape the ordering (docs/GameRules.md):
//! * one move, one basic attack and one successful cast per turn;
//! * ranged champions choose walking *or* a basic attack;
//! * "excludes walking" spells must be cast before moving and block moving after;
//! * a move onto an unseen occupant returns `false` without spending the move.

use crate::bc_brain::shop;
use crate::bc_brain::Intent;
use crate::bc_core::champion::{AttackType, KitInfo};
use crate::bc_core::geom::DIRS8;
use crate::bc_core::ledger::{Family, Ledger, Stat, SHARD_COST};
use crate::bc_core::Pos;
use crate::bc_path::danger::{DangerParams, Threat};
use crate::bc_path::planner::{fallback_step, Situation};
use crate::bc_path::{choose_step, DistanceOracle, Goal, Tier};
use crate::sdk::vec2;
use crate::telemetry;
use crate::world::World;
use cambc::{
    AbilityKind, AbilityTargetMask, Camp, Champion, ChampionState, ConsumableType, ItemType, Result,
};

/// Fuel that must remain before the deep (A*) pathing tier may run.
const DEEP_SEARCH_MIN_FUEL: u64 = 75_000;
/// Altar locks: placement range (tiles) and price (gold).
const ALTAR_LOCK_RANGE: i32 = 2;
const ALTAR_LOCK_COST: u32 = 10;

pub struct EnemyView {
    pub handle: Champion,
    pub pos: Pos,
    pub dist: i32,
    pub hp: f32,
    pub max_hp: f32,
    pub dps: f32,
    pub gems: u32,
    pub ranged: bool,
    pub range: i32,
    /// Standing on its own spawn: heals there and we can't step onto it.
    pub on_spawn: bool,
    /// Channelling a tribute: a stun, a kill or a new altar lock cancels the whole bag.
    pub tributing: bool,
    pub kind: crate::bc_core::champion::ChampionKind,
}

pub struct AllyView {
    pub handle: Champion,
    pub pos: Pos,
    pub dist: i32,
    pub hp: f32,
    pub max_hp: f32,
    pub dps: f32,
}

pub struct CampView {
    pub handle: Camp,
    pub pos: Pos,
    pub dist: i32,
    pub hp: f32,
    pub awake: bool,
    pub strength: f32,
    pub group: Option<usize>,
}

#[derive(Default)]
pub struct Visible {
    /// Nearest first.
    pub enemies: Vec<EnemyView>,
    pub allies: Vec<AllyView>,
    pub camps: Vec<CampView>,
    /// Enemies seen in the last few rounds but not now (likely in a bush), widened by age.
    pub ghosts: Vec<Threat>,
}

impl Visible {
    /// Things that could hurt us, for danger-weighted pathing (remembered enemies included).
    pub fn threats(&self, ignore_group: Option<usize>) -> Vec<Threat> {
        let camps = self
            .camps
            .iter()
            .filter(|c| {
                c.awake && c.strength > 0.0 && (ignore_group.is_none() || c.group != ignore_group)
            })
            .map(|c| Threat { pos: c.pos, range: 3, dps: c.strength });
        let enemies =
            self.enemies.iter().map(|e| Threat { pos: e.pos, range: e.range + 1, dps: e.dps });
        camps.chain(enemies).chain(self.ghosts.iter().copied()).collect()
    }

    /// Damage per round that visible and remembered enemies could bring to `p` within `rounds`.
    pub fn reachable_dps(&self, p: Pos, rounds: i32) -> f32 {
        let seen = self.enemies.iter().filter(|e| e.pos.cheb(p) <= e.range + rounds).map(|e| e.dps);
        let ghosts =
            self.ghosts.iter().filter(|g| g.pos.cheb(p) <= g.range + rounds).map(|g| g.dps);
        seen.chain(ghosts).sum()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Outcome {
    pub moved: bool,
    pub attacked: bool,
    pub cast: bool,
    pub tier: Option<Tier>,
    pub spent: u32,
    /// Fuel spent choosing and making steps (the Path stage).
    pub path_fuel: u64,
}

impl Outcome {
    pub fn tier_name(&self) -> &'static str {
        match self.tier {
            None => "-",
            Some(Tier::Static) => "static",
            Some(Tier::LocalRepair) => "repair",
            Some(Tier::Greedy) => "greedy",
            Some(Tier::Weighted) => "weighted",
            Some(Tier::Fallback) => "fallback",
            Some(Tier::Stay) => "stay",
        }
    }
}

pub struct Exec<'a> {
    pub me: &'a Champion,
    pub world: &'a World,
    pub vis: &'a Visible,
    pub kit: &'a KitInfo,
    pub ledger: &'a mut Ledger,
    pub order: &'a [(Family, Stat)],
    pub my_pos: Pos,
    pub occupied: Vec<Pos>,
    pub hurt_this_round: bool,
    /// Walking to a teammate's fight at this position (Engage before the enemy is in view).
    pub assist: Option<Pos>,
    /// Deny enemy tributes (knob DENY_TRIBUTE): stun / lock / attack tributers, carry a lock.
    pub deny_tribute: bool,
    /// DENY_TRIBUTE >= 2: also ground-targeted area stuns (Ryth) on tributers.
    pub deny_ground: bool,
    /// ALTAR_LOCK: keep a lock in the pocket and leave it on altars we tribute at.
    pub altar_lock: bool,
    /// COMBAT_FSM 64|8: combo cast order against champions (Exec::combo_cast).
    pub combo: bool,
    /// COMBAT_FSM 64|16: melee champions dash into attack range.
    pub dash_in: bool,
    /// TAVI_META: Tavi buys shards anywhere on the map (its passive).
    pub field_shop: bool,
    /// RECALL_SAFE: only start a recall nothing can reach before it completes.
    pub recall_safe: bool,
    /// Recall channel length (5, or 3 with a Home emblem).
    pub recall_ticks: i32,
    /// Tavi re-spec: on spawn, sell emblems whose family is not in the build order.
    pub tavi_respec: bool,
    pub out: Outcome,
}

impl Exec<'_> {
    fn ranged(&self) -> bool {
        self.kit.attack == AttackType::Ranged
    }

    fn at_spawn(&self) -> bool {
        self.world.terrain.ally_spawn.get(self.my_pos)
    }

    pub fn execute(&mut self, intent: Intent) -> Result<()> {
        if self.at_spawn() || self.field_shop {
            self.buy()?;
        }
        match intent {
            Intent::ClearCamp(g) => self.clear_camp(g as usize),
            Intent::Capture(g) => self.capture(g as usize),
            Intent::Approach(g) => self.approach(g as usize),
            Intent::Tribute => self.tribute(),
            Intent::Engage(p) => self.engage(p),
            Intent::Kite(p) => self.kite(p),
            Intent::Retreat => self.retreat(),
            Intent::Shop => self.shop(),
            Intent::Hold => self.opportunistic_attack(),
            Intent::StepBack(p) => self.step_back(p),
            Intent::WalkHome => {
                self.cast_support()?;
                let threats = self.vis.threats(None);
                self.walk(Goal::AllySpawn, &threats)
            }
        }
    }

    // ----- shopping -------------------------------------------------------

    /// Buy shards along the build order while gold allows (spawn or dead only).
    pub fn buy(&mut self) -> Result<()> {
        if self.altar_lock {
            self.buy_lock()?;
        }
        if self.tavi_respec && self.at_spawn() {
            self.respec()?;
        }
        let mut gold = self.me.gold_count()?;
        for _ in 0..8 {
            if gold < SHARD_COST {
                break;
            }
            let Some(id) = shop::next_shard(self.order, self.ledger) else { break };
            let item = ItemType::try_from(u32::from(id))?;
            match self.me.buy(item) {
                Ok(()) => {
                    gold -= SHARD_COST;
                    self.ledger.record_spend(SHARD_COST);
                    self.out.spent += SHARD_COST;
                    // Keep the ledger's view current so the order advances this turn.
                    self.ledger.items[id as usize] += 1;
                    if self.ledger.items[id as usize] >= 5 {
                        self.ledger.items[id as usize] = 0;
                        self.ledger.items[id as usize + 1] += 1;
                    }
                }
                Err(e) => {
                    telemetry::error("buy", &e);
                    break;
                }
            }
        }
        // One altar lock in the pocket (10 gold) to cancel enemy tributes.
        if self.deny_tribute && gold >= ALTAR_LOCK_COST {
            self.buy_lock()?;
        }
        Ok(())
    }

    /// Tavi only: sell (full refund on spawn) completed emblems of families the build order
    /// no longer wants, so the gold goes into the current plan.
    fn respec(&mut self) -> Result<()> {
        use crate::bc_core::ledger::{decode_item, ITEM_TYPES};
        if !self.me.can_sell().unwrap_or(false) {
            return Ok(());
        }
        for id in 0..ITEM_TYPES as u8 {
            let Some((family, _, true)) = decode_item(id) else { continue };
            if self.ledger.items[id as usize] == 0 || self.order.iter().any(|&(f, _)| f == family) {
                continue;
            }
            if let Ok(item) = ItemType::try_from(u32::from(id)) {
                if self.me.sell(item).is_ok() {
                    self.ledger.items[id as usize] -= 1;
                    telemetry::event(0, "", "respec_sell", i64::from(id));
                }
            }
        }
        Ok(())
    }

    fn buy_lock(&mut self) -> Result<()> {
        if self.me.gold_count()? >= ALTAR_LOCK_COST
            && self.me.consumables()[ConsumableType::AltarLock].count().unwrap_or(1) == 0
            && self.me.can_buy_consumable(ConsumableType::AltarLock).unwrap_or(false)
        {
            if let Err(e) = self.me.buy_consumable(ConsumableType::AltarLock) {
                telemetry::error("buy_lock", &e);
            }
        }
        Ok(())
    }

    /// Cast a ready crowd-control ability on `enemy` if it is in range. True if cast.
    fn stun(&mut self, enemy: &EnemyView) -> Result<bool> {
        if self.out.cast || !self.me.can_cast()? {
            return Ok(false);
        }
        for ability in self.me.abilities() {
            if !ability.is_cc() || !ability.is_ready()? {
                continue;
            }
            let in_range = ability
                .info()?
                .and_then(|i| i.as_dyn().range().ok())
                .is_some_and(|r| self.my_pos.cheb(enemy.pos) as f32 <= r);
            if in_range && ability.cast_on(&enemy.handle).unwrap_or(false) {
                self.out.cast = true;
                return Ok(true);
            }
            // Area stuns aimed at the ground (Ryth): cast at the tributer's tile.
            if in_range
                && self.deny_ground
                && !self.out.moved
                && ability
                    .info()?
                    .is_some_and(|i| i.as_dyn().valid_targets().contains(AbilityTargetMask::GROUND))
                && ability.cast_at(vec2(enemy.pos)).unwrap_or(false)
            {
                self.out.cast = true;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Place a carried altar lock on the altar next to `near` (cancels tributes there).
    fn lock_altar_near(&mut self, near: Pos) -> Result<bool> {
        let lock = &self.me.consumables()[ConsumableType::AltarLock];
        if lock.count().unwrap_or(0) == 0 {
            return Ok(false);
        }
        let Some(altar) = self.world.altars.iter().copied().find(|a| a.cheb(near) <= 1) else {
            return Ok(false);
        };
        if self.my_pos.cheb(altar) > ALTAR_LOCK_RANGE
            || cambc::vision::has_altar_lock(vec2(altar)).unwrap_or(true)
        {
            return Ok(false);
        }
        Ok(lock.use_at(vec2(altar)).unwrap_or(false))
    }

    fn shop(&mut self) -> Result<()> {
        if self.at_spawn() {
            return Ok(());
        }
        self.recall_or_walk_home()
    }

    // ----- movement -------------------------------------------------------

    fn goal_hint(&self, goal: Goal) -> Pos {
        match goal {
            Goal::Group(i) => self.world.groups.get(i as usize).map_or(self.my_pos, |g| g.center),
            Goal::Altars => self
                .world
                .altars
                .iter()
                .copied()
                .min_by_key(|a| a.cheb(self.my_pos))
                .unwrap_or(self.my_pos),
            Goal::AllySpawn => self.world.spawn_hint,
        }
    }

    /// One step towards `goal` through the pathing ensemble. Retries once if
    /// the chosen tile hides an unseen occupant.
    fn walk(&mut self, goal: Goal, threats: &[Threat]) -> Result<()> {
        let before = crate::sdk::fuel();
        let result = self.walk_inner(goal, threats);
        self.out.path_fuel += before.saturating_sub(crate::sdk::fuel());
        result
    }

    fn walk_inner(&mut self, goal: Goal, threats: &[Threat]) -> Result<()> {
        if !self.me.can_move()? {
            return Ok(());
        }
        for _ in 0..2 {
            let occupied = self.occupied.clone();
            let is_occupied = |p: Pos| occupied.contains(&p);
            // Adaptive: only spend on the deep A* tier when the turn is cheap so far.
            // Measured ~3k fuel per expansion in game (heap + 8 neighbours), so cap it hard.
            let left = crate::sdk::fuel();
            let deep_expansions = if left > DEEP_SEARCH_MIN_FUEL {
                ((left - DEEP_SEARCH_MIN_FUEL) / 3_000).min(8) as u32
            } else {
                0
            };
            let choice = if self.world.fields.is_ready(goal) {
                let sit = Situation {
                    me: self.my_pos,
                    terrain: &self.world.terrain,
                    occupied: &is_occupied,
                    threats,
                    danger_radius: 3,
                    params: DangerParams::default(),
                    deep_expansions,
                };
                choose_step(&sit, goal, &self.world.fields)
            } else {
                fallback_step(self.my_pos, self.goal_hint(goal), &self.world.terrain, &is_occupied)
            };
            self.out.tier = Some(choice.tier);
            if choice.to == self.my_pos {
                return Ok(());
            }
            match self.me.move_absolute(vec2(choice.to)) {
                Ok(true) => {
                    telemetry::draw_path(self.my_pos, choice.to, 0x33_cc_ff);
                    self.my_pos = choice.to;
                    self.out.moved = true;
                    return Ok(());
                }
                Ok(false) => self.occupied.push(choice.to),
                Err(e) => {
                    telemetry::error("move", &e);
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Step to the free neighbour that best satisfies `score` (lower is better).
    fn step_by(&mut self, score: impl Fn(Pos) -> i32) -> Result<()> {
        if !self.me.can_move()? {
            return Ok(());
        }
        let best = DIRS8
            .iter()
            .map(|&d| self.my_pos + d)
            .filter(|&p| self.world.terrain.walkable(p) && !self.occupied.contains(&p))
            .min_by_key(|&p| score(p));
        if let Some(to) = best {
            if score(to) < score(self.my_pos) && self.me.move_absolute(vec2(to)).unwrap_or(false) {
                self.my_pos = to;
                self.out.moved = true;
                self.out.tier = Some(Tier::Fallback);
            }
        }
        Ok(())
    }

    fn recall_or_walk_home(&mut self) -> Result<()> {
        let camps_quiet = self.vis.camps.iter().all(|c| !c.awake || c.dist > 4);
        let in_bush = self.world.terrain.bushes.get(self.my_pos);
        // Outside a bush we need distance; inside one, enemies can't see us, so it
        // is enough that none can reach us before the recall finishes.
        let mut safe = !self.hurt_this_round
            && camps_quiet
            && self.vis.enemies.iter().all(|e| e.dist > if in_bush { e.range + 2 } else { 6 });
        if self.recall_safe {
            // The channel freezes us for `recall_ticks` turns and any hit cancels it.
            let ticks = self.recall_ticks;
            safe = safe
                && self.vis.enemies.iter().all(|e| e.dist > e.range + 1 + ticks)
                && self.vis.ghosts.iter().all(|g| g.pos.cheb(self.my_pos) > g.range + ticks);
        }
        if safe {
            match self.me.recall() {
                Ok(()) => return Ok(()),
                Err(e) => telemetry::error("recall", &e),
            }
        }
        if !self.vis.enemies.is_empty() && !self.hurt_this_round {
            // Hidden with a healthy ally close: hold and let them take the fight.
            let tank_near = self.vis.allies.iter().any(|a| a.dist <= 3 && a.hp >= 0.6 * a.max_hp);
            if in_bush && tank_near {
                return Ok(());
            }
            if !in_bush && self.duck_into_bush()? {
                return Ok(());
            }
        }
        let threats = self.vis.threats(None);
        self.walk(Goal::AllySpawn, &threats)
    }

    /// Step into an adjacent bush that no visible enemy can hit. True if we moved.
    fn duck_into_bush(&mut self) -> Result<bool> {
        if !self.me.can_move()? {
            return Ok(false);
        }
        let enemies = &self.vis.enemies;
        let exposed = |p: Pos| enemies.iter().any(|e| e.pos.cheb(p) <= e.range + 1);
        let best = DIRS8
            .iter()
            .map(|&d| self.my_pos + d)
            .filter(|&p| {
                self.world.terrain.bushes.get(p)
                    && self.world.terrain.walkable(p)
                    && !self.occupied.contains(&p)
                    && !exposed(p)
            })
            .max_by_key(|&p| enemies.iter().map(|e| e.pos.cheb(p)).min().unwrap_or(0));
        let Some(to) = best else { return Ok(false) };
        if self.me.move_absolute(vec2(to)).unwrap_or(false) {
            telemetry::event(0, "", "hide_bush", 1);
            self.my_pos = to;
            self.out.moved = true;
            self.out.tier = Some(Tier::Fallback);
            return Ok(true);
        }
        Ok(false)
    }

    // ----- combat helpers -------------------------------------------------

    fn in_attack_range(&self, p: Pos) -> bool {
        self.my_pos.cheb(p) <= self.kit.attack_range
    }

    fn attack_camp(&mut self, camp: &CampView) -> bool {
        if !self.me.can_attack().unwrap_or(false) {
            return false;
        }
        match self.me.attack(&camp.handle) {
            Ok(()) => {
                self.out.attacked = true;
                true
            }
            Err(e) => {
                telemetry::error("attack_camp", &e);
                false
            }
        }
    }

    fn attack_champion(&mut self, target: &Champion) -> bool {
        if !self.me.can_attack().unwrap_or(false) {
            return false;
        }
        match self.me.attack(target) {
            Ok(()) => {
                self.out.attacked = true;
                true
            }
            Err(e) => {
                telemetry::error("attack_champion", &e);
                false
            }
        }
    }

    /// Cast an attack reset right after a basic attack, then attack again.
    fn reset_and_reattack(&mut self, again: &dyn Fn(&mut Self) -> bool) -> Result<()> {
        let me = self.me;
        if self.out.cast || !self.out.attacked || !me.can_cast()? {
            return Ok(());
        }
        for ability in me.abilities() {
            if ability.is_attack_reset() && ability.is_ready()? && ability.cast_untargeted()? {
                self.out.cast = true;
                again(self);
                break;
            }
        }
        Ok(())
    }

    /// Damage / stun / area spell on a camp or enemy at `target`. Skips spells
    /// that exclude walking when we still intend to move.
    fn cast_offensive(
        &mut self,
        target_pos: Pos,
        camp: Option<&Camp>,
        enemy: Option<&Champion>,
        will_move: bool,
    ) -> Result<bool> {
        let me = self.me;
        if self.out.cast || !me.can_cast()? {
            return Ok(false);
        }
        if let (true, Some(e)) = (self.combo, enemy) {
            if let Some(done) = self.combo_cast(target_pos, e, will_move)? {
                return Ok(done);
            }
        }
        let dist = self.my_pos.cheb(target_pos);
        for ability in me.abilities() {
            let Some(info) = ability.info()? else { continue };
            let meta = info.as_dyn();
            let Ok(range) = meta.range() else { continue };
            if dist as f32 > range
                || (meta.is_exclusive_with_movement() && (will_move || self.out.moved))
            {
                continue;
            }
            let targets = meta.valid_targets();
            let cast = match meta.kind() {
                AbilityKind::Damage | AbilityKind::Cc => {
                    if !ability.is_ready()? {
                        continue;
                    }
                    match (camp, enemy) {
                        (Some(c), _) if targets.contains(AbilityTargetMask::CAMPS) => {
                            ability.cast_on(c)
                        }
                        (_, Some(e)) if targets.contains(AbilityTargetMask::ENEMIES) => {
                            ability.cast_on(e)
                        }
                        _ => continue,
                    }
                }
                AbilityKind::AoeDamage => {
                    if !ability.is_ready()? {
                        continue;
                    }
                    ability.cast_at(vec2(target_pos))
                }
                _ => continue,
            };
            match cast {
                Ok(true) => {
                    self.out.cast = true;
                    return Ok(true);
                }
                Ok(false) => {}
                Err(e) => telemetry::error("cast", &e),
            }
        }
        Ok(false)
    }

    /// Combo order against a champion: stun one that isn't stunned (the opener), burst one that
    /// is (our stun or a teammate's), and never spend a stun on a stunned target (keep it to
    /// chain when the first ends). `Some(cast)` when it decided, `None` = use the plain order.
    fn combo_cast(
        &mut self,
        target_pos: Pos,
        enemy: &Champion,
        will_move: bool,
    ) -> Result<Option<bool>> {
        let me = self.me;
        let dist = self.my_pos.cheb(target_pos);
        let st = enemy.state().unwrap_or_default();
        let channel =
            ChampionState::RECALLING | ChampionState::SUBMITTING_GEMS | ChampionState::TELEPORTING;
        let stunned = st.contains(ChampionState::ALIVE)
            && !st.intersects(ChampionState::CAN_MOVE | ChampionState::CAN_ATTACK)
            && !st.intersects(channel);
        let (mut stun, mut burst) = (None, None);
        for ability in me.abilities() {
            let Some(info) = ability.info()? else { continue };
            let meta = info.as_dyn();
            let Ok(range) = meta.range() else { continue };
            let blocked = meta.is_exclusive_with_movement() && (will_move || self.out.moved);
            if dist as f32 > range || blocked || !ability.is_ready()? {
                continue;
            }
            let foes = meta.valid_targets().contains(AbilityTargetMask::ENEMIES);
            match meta.kind() {
                AbilityKind::Cc if foes && stun.is_none() => stun = Some(ability),
                AbilityKind::Damage if foes && burst.is_none() => burst = Some((ability, false)),
                AbilityKind::AoeDamage if burst.is_none() => burst = Some((ability, true)),
                _ => {}
            }
        }
        let cast = match (stunned, stun, burst) {
            (true, _, Some((b, aoe))) => {
                if aoe {
                    b.cast_at(vec2(target_pos))
                } else {
                    b.cast_on(enemy)
                }
            }
            // Stunned and only the stun is ready: hold it for the chain.
            (true, Some(_), None) => return Ok(Some(false)),
            (false, Some(s), _) => s.cast_on(enemy),
            _ => return Ok(None),
        };
        match cast {
            Ok(true) => {
                self.out.cast = true;
                telemetry::event(0, "", if stunned { "combo_burst" } else { "combo_stun" }, 1);
                Ok(Some(true))
            }
            Ok(false) => Ok(None),
            Err(e) => {
                telemetry::error("combo", &e);
                Ok(None)
            }
        }
    }

    /// Dash onto the tile next to `target` (melee: close a 2-3 tile gap and still attack).
    fn dash_to(&mut self, dest: Pos) -> Result<bool> {
        if self.out.cast || self.out.moved || !self.me.can_cast()? {
            return Ok(false);
        }
        for ability in self.me.abilities() {
            if ability.is_dash() && ability.is_ready()? {
                if let Ok(true) = ability.cast_at(vec2(dest)) {
                    self.out.cast = true;
                    self.out.moved = true;
                    self.my_pos = dest;
                    telemetry::event(0, "", "dash_in", 1);
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Self shield when something can hit us; ally heal when someone is hurt.
    fn cast_support(&mut self) -> Result<()> {
        let (me, vis) = (self.me, self.vis);
        if self.out.cast || !me.can_cast()? {
            return Ok(());
        }
        let under_fire = self.hurt_this_round
            || vis.camps.iter().any(|c| c.awake && c.dist <= 3)
            || vis.enemies.iter().any(|e| e.dist <= e.range + 1);
        for ability in me.abilities() {
            if ability.is_shield() && under_fire && ability.is_ready()? {
                if ability.cast_untargeted().unwrap_or(false) {
                    self.out.cast = true;
                    return Ok(());
                }
            } else if ability.is_heal() && ability.is_ready()? {
                let hurt = vis
                    .allies
                    .iter()
                    .filter(|a| a.dist <= 4 && a.max_hp > 0.0 && a.hp / a.max_hp < 0.75)
                    .min_by(|a, b| (a.hp / a.max_hp).total_cmp(&(b.hp / b.max_hp)));
                if let Some(ally) = hurt {
                    if ability.cast_on(&ally.handle).unwrap_or(false) {
                        self.out.cast = true;
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    /// Attack whatever is worth hitting in range without moving (used while
    /// holding, capturing or as a fallback).
    fn opportunistic_attack(&mut self) -> Result<()> {
        let vis = self.vis;
        if let Some(enemy) = vis.enemies.iter().find(|e| self.in_attack_range(e.pos)) {
            self.attack_champion(&enemy.handle);
            return Ok(());
        }
        // Only already-awake camps: hitting a sleeping camp wakes its whole group.
        if let Some(camp) = vis.camps.iter().find(|c| c.awake && self.in_attack_range(c.pos)) {
            self.attack_camp(camp);
        }
        Ok(())
    }

    // ----- intents --------------------------------------------------------

    fn clear_camp(&mut self, group: usize) -> Result<()> {
        let vis = self.vis;
        let targets: Vec<&CampView> =
            vis.camps.iter().filter(|c| c.group == Some(group) && c.hp > 0.0).collect();
        self.cast_support()?;
        // Focus the weakest camp we can already hit.
        let in_range = targets
            .iter()
            .copied()
            .filter(|c| self.in_attack_range(c.pos))
            .min_by(|a, b| a.hp.total_cmp(&b.hp));
        if let Some(camp) = in_range {
            self.cast_offensive(camp.pos, Some(&camp.handle), None, false)?;
            if self.attack_camp(camp) {
                self.reset_and_reattack(&|s: &mut Self| s.me.attack(&camp.handle).is_ok())?;
            }
            return Ok(());
        }
        // Spell range (4) beats camp range (3): free damage before walking in.
        if let Some(camp) = targets.iter().copied().min_by_key(|c| c.dist) {
            if self.ranged() && self.cast_offensive(camp.pos, Some(&camp.handle), None, false)? {
                return Ok(());
            }
        }
        let threats = vis.threats(Some(group));
        if let Some(camp) = targets.iter().copied().min_by_key(|c| c.dist) {
            // A group's field leads to *any* of its spawn tiles, which can be out
            // of range of the living camps in a long group: close in on a camp.
            let (goal, range) = (camp.pos, self.kit.attack_range);
            let before = crate::sdk::fuel();
            self.step_by(|p| (p.cheb(goal) - range).max(0))?;
            self.out.path_fuel += before.saturating_sub(crate::sdk::fuel());
            if !self.out.moved && !self.in_attack_range(goal) {
                self.walk(Goal::Group(group as u8), &threats)?;
            }
        } else {
            // At the group but nothing alive to hit: it's dead, so don't block its respawn.
            if self.leave_spawn_tiles(group)? {
                return Ok(());
            }
            self.walk(Goal::Group(group as u8), &threats)?;
        }
        if !self.ranged() {
            if let Some(camp) = targets
                .iter()
                .copied()
                .filter(|c| self.in_attack_range(c.pos))
                .min_by(|a, b| a.hp.total_cmp(&b.hp))
            {
                self.attack_camp(camp);
            }
        }
        Ok(())
    }

    fn capture(&mut self, group: usize) -> Result<()> {
        let on_tile = self.world.group_at(self.my_pos) == Some(group);
        if on_tile {
            return self.opportunistic_attack();
        }
        let threats = self.vis.threats(None);
        self.walk(Goal::Group(group as u8), &threats)
    }

    /// Standing on any spawn tile of a dead group blocks its respawn: step off.
    fn leave_spawn_tiles(&mut self, group: usize) -> Result<bool> {
        if self.world.group_at(self.my_pos) != Some(group) {
            return Ok(false);
        }
        let world = self.world;
        let fields = &self.world.fields;
        let goal = Goal::Group(group as u8);
        self.step_by(|p| {
            let on_spawn = world.group_at(p).is_some();
            i32::from(on_spawn) * 1000 + fields.distance(goal, p).map_or(50, i32::from)
        })?;
        Ok(true)
    }

    fn approach(&mut self, group: usize) -> Result<()> {
        if self.leave_spawn_tiles(group)? {
            return Ok(());
        }
        let close = self
            .world
            .fields
            .distance(Goal::Group(group as u8), self.my_pos)
            .is_some_and(|d| d <= 2);
        if close {
            return self.opportunistic_attack();
        }
        let threats = self.vis.threats(None);
        self.walk(Goal::Group(group as u8), &threats)
    }

    fn tribute(&mut self) -> Result<()> {
        if let Some(altar) = self.world.altar_in_range(self.my_pos) {
            // Damage doesn't stop a tribute but death does (and drops the bag to the
            // killer): don't start a channel that enemies can kill us during.
            let ticks = (self.me.gem_count()?.div_ceil(10)).max(1) as i32;
            let danger = self.vis.reachable_dps(self.my_pos, ticks);
            if danger > 0.0 && danger * ticks as f32 >= self.me.health()? {
                telemetry::event(0, "", "tribute_unsafe", 1);
                return self.recall_or_walk_home();
            }
            // Leave our lock on the altar first: it only blocks the enemy.
            if self.altar_lock
                && cambc::vision::altar_lock_at(vec2(altar)).is_err()
                && self.me.consumables()[ConsumableType::AltarLock].count().unwrap_or(0) > 0
                && self.me.consumables()[ConsumableType::AltarLock]
                    .use_at(vec2(altar))
                    .unwrap_or(false)
            {
                telemetry::event(0, "", "lock_placed", 1);
            }
            if cambc::vision::has_altar_lock(vec2(altar)).unwrap_or(false) {
                // Tributing into an enemy lock is converted to attacks on the lock.
                telemetry::event(0, "", "altar_locked", 1);
            }
            match self.me.submit_gems(vec2(altar)) {
                Ok(()) => return Ok(()),
                Err(e) => telemetry::error("submit_gems", &e),
            }
        }
        let threats = self.vis.threats(None);
        self.walk(Goal::Altars, &threats)
    }

    fn engage(&mut self, target: Pos) -> Result<()> {
        let vis = self.vis;
        let Some(enemy) = vis.enemies.iter().find(|e| e.pos == target) else {
            if self.assist == Some(target) {
                return self.step_by(|p| p.cheb(target));
            }
            return self.opportunistic_attack();
        };
        if self.deny_tribute
            && enemy.tributing
            && (self.stun(enemy)? || self.lock_altar_near(enemy.pos)?)
        {
            telemetry::event(0, "", "deny_tribute", 1);
            return Ok(());
        }
        self.cast_support()?;
        // Melee dash-in: a 2-3 tile gap closes in one cast and we still attack this turn.
        let gap = self.my_pos.cheb(enemy.pos);
        if self.dash_in && !self.ranged() && (2..=3).contains(&gap) {
            let dest = enemy.pos.step_towards(self.my_pos);
            if self.dash_to(dest)? && self.in_attack_range(enemy.pos) {
                self.attack_champion(&enemy.handle);
                return Ok(());
            }
        }
        let will_move = !self.in_attack_range(enemy.pos);
        self.cast_offensive(enemy.pos, None, Some(&enemy.handle), will_move)?;
        if self.in_attack_range(enemy.pos) {
            if self.attack_champion(&enemy.handle) {
                self.reset_and_reattack(&|s: &mut Self| s.me.attack(&enemy.handle).is_ok())?;
            }
            return Ok(());
        }
        let goal = enemy.pos;
        self.step_by(|p| p.cheb(goal))?;
        if !self.ranged() && self.in_attack_range(enemy.pos) {
            self.attack_champion(&enemy.handle);
        }
        Ok(())
    }

    fn kite(&mut self, target: Pos) -> Result<()> {
        let vis = self.vis;
        let Some(enemy) = vis.enemies.iter().find(|e| e.pos == target) else {
            return self.opportunistic_attack();
        };
        let d = self.my_pos.cheb(enemy.pos);
        if d <= enemy.range {
            // Too close: back off, preferring tiles towards home.
            let spawn_dist =
                |p: Pos| self.world.fields.distance(Goal::AllySpawn, p).map_or(0, i32::from);
            let goal = enemy.pos;
            return self.step_by(|p| -4 * p.cheb(goal) + spawn_dist(p) / 4);
        }
        self.cast_offensive(enemy.pos, None, Some(&enemy.handle), false)?;
        if self.in_attack_range(enemy.pos) {
            self.attack_champion(&enemy.handle);
            return Ok(());
        }
        let goal = enemy.pos;
        self.step_by(|p| p.cheb(goal))
    }

    /// Aggro swap: move to the edge of `from`'s reach (camps hit range 3; spells reach 4)
    /// preferring tiles near allies, then cast at it if it is a camp still in spell range.
    fn step_back(&mut self, from: Pos) -> Result<()> {
        let vis = self.vis;
        let allies: Vec<Pos> = vis.allies.iter().map(|a| a.pos).collect();
        let reach = vis.enemies.iter().find(|e| e.pos == from).map_or(3, |e| e.range + 1);
        self.step_by(|p| {
            let d = p.cheb(from);
            let edge = if d > reach { (d - reach - 1).abs() } else { 10 * (reach + 1 - d) };
            let near_ally = allies.iter().map(|a| a.cheb(p)).min().unwrap_or(8).min(8);
            edge * 4 + near_ally
        })?;
        if let Some(camp) = vis.camps.iter().find(|c| c.pos == from && c.hp > 0.0) {
            self.cast_offensive(camp.pos, Some(&camp.handle), None, true)?;
        }
        self.opportunistic_attack()
    }

    fn retreat(&mut self) -> Result<()> {
        self.cast_support()?;
        self.recall_or_walk_home()
    }
}

/// Positions of everything visible that blocks a step (excluding ourselves).
pub fn occupied_tiles(vis: &Visible, me: Pos) -> Vec<Pos> {
    vis.enemies
        .iter()
        .map(|e| e.pos)
        .chain(vis.allies.iter().map(|a| a.pos))
        .chain(vis.camps.iter().filter(|c| c.hp > 0.0).map(|c| c.pos))
        .filter(|&p| p != me)
        .collect()
}
