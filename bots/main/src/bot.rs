//! One champion's turn: observe → belief/ledger → team channel → target →
//! meta-gate → router → expert intent → executor → telemetry.

use crate::act::{occupied_tiles, AllyView, CampView, EnemyView, Exec, Outcome, Visible};
use crate::bc_brain::context::{Context, EnemyNear, Knobs, TargetGroup};
use crate::bc_brain::features::{extract, GroupMask, RawEconomy, RawObs, RawSelf};
use crate::bc_brain::{gate, policy, router, shop, Expert, Intent, Mode};
use crate::bc_core::belief::{Belief, GroupObs, GroupState, Sighting};
use crate::bc_core::champion::{AttackType, ChampionKind, KitInfo};
use crate::bc_core::fuel::{FuelGovernor, Stage};
use crate::bc_core::ledger::{Family, Ledger, SelfObs, Stat, ITEM_TYPES};
use crate::bc_core::memory::{TeamChannel, ThetaMatch, TEAM_CHANNEL_BYTES};
use crate::bc_core::Pos;
use crate::bc_path::danger::Threat;
use crate::bc_path::{DistanceOracle, Goal};
use crate::sdk::{self, fuel, pos};
use crate::telemetry::{self, TurnLog};
use crate::world::World;
use cambc::{CampGroupState, Champion, ChampionState, Result, Team};

/// Vision radius for deciding which camp groups we can check this turn.
const VISION: i32 = 8;

/// Assume an unreadable, camp-less group stays down this long before re-checking.
const INFER_DOWN_ROUNDS: i32 = 8;

pub struct Bot {
    world: World,
    kind: ChampionKind,
    kit: KitInfo,
    ledger: Ledger,
    belief: Belief,
    theta: ThetaMatch,
    knobs: Knobs,
    gov: FuelGovernor,
    order: Vec<(Family, Stat)>,
    slot: u8,
    team_size: usize,
    target: Option<usize>,
    last_hp: f32,
    hurt_round: u32,
    last_pos: Pos,
    items_round: u32,
    turns: u32,
    /// Consecutive Engage turns without an attack or cast.
    engage_stall: u32,
    /// Enemy we couldn't reach, ignored as a fight target until a round.
    ignore_enemy: Option<(ChampionKind, u32)>,
}

/// Engage turns without landing anything before we give up on that enemy.
const ENGAGE_STALL_TURNS: u32 = 4;
/// How long to ignore an unreachable enemy.
const IGNORE_ENEMY_ROUNDS: u32 = 40;
/// Remembered enemies (not in vision) count as threats this many rounds after the sighting.
const GHOST_MAX_AGE: u32 = 8;
/// Cap on remembered threats per turn (each costs a pass over the path window).
const MAX_GHOSTS: usize = 5;
/// Teammate sightings further than this are ignored (threat and help alike).
const TEAM_SIGHTING_RADIUS: i32 = 12;
/// Reach and damage assumed for an enemy only a teammate saw (its kind isn't shared).
const TEAM_GHOST_RANGE: i32 = 3;
const TEAM_GHOST_DPS: f32 = 40.0;
/// Only champions above this health fraction answer a call for help.
const ASSIST_MIN_HP: f32 = 0.7;

impl Bot {
    pub fn new() -> Result<Self> {
        let world = World::build()?;
        let me = cambc::self_champion()?.ok_or_else(|| sdk::local_error("no self champion"))?;
        let kind = sdk::kind(me.champion_name()?);
        let kit = sdk::kit(kind);
        let theta = ThetaMatch::decode(cambc::draft_memory().bytes()).unwrap_or_default();
        // Slot = our rank among ally champion kinds; stable for the whole match.
        let mut allies: Vec<u8> = cambc::champions(Some(Team::Ally))?
            .iter()
            .filter_map(|c| c.champion_name().ok())
            .map(|t| sdk::kind(t) as u8)
            .collect();
        allies.sort_unstable();
        let slot = allies.iter().position(|&k| k == kind as u8).unwrap_or(0) as u8;
        let belief = Belief::new(&world.groups.iter().map(|g| g.max_respawn).collect::<Vec<_>>());
        let last_pos = me.position().map(pos).unwrap_or_default();
        if slot == 0 {
            telemetry::map_dump(&world);
        }
        Ok(Self {
            knobs: Knobs::from_theta(&theta),
            team_size: allies.len().max(1),
            world,
            kind,
            kit,
            ledger: Ledger::new(kind),
            belief,
            theta,
            gov: FuelGovernor::default(),
            order: shop::build_order(kind),
            slot,
            target: None,
            last_hp: 0.0,
            hurt_round: 0,
            last_pos,
            items_round: 0,
            turns: 0,
            engage_stall: 0,
            ignore_enemy: None,
        })
    }

    pub fn turn(&mut self) -> Result<()> {
        let fuel_start = fuel();
        self.gov.start_turn(fuel_start);
        self.turns += 1;
        let round = cambc::round_number()?;
        let Some(me) = cambc::self_champion()? else { return Ok(()) };
        let state = me.state()?;
        let alive = state.contains(ChampionState::ALIVE);

        // ---- self observation ---------------------------------------------
        let (my_pos, hp, max_hp) = if alive {
            (pos(me.position()?), me.health()?, me.max_health()?)
        } else {
            (self.last_pos, 0.0, 1.0)
        };
        let at_spawn = self.world.terrain.ally_spawn.get(my_pos);
        let gold = me.gold_count()?;
        let gems = if alive { me.gem_count()? } else { 0 };
        let mut items = self.ledger.items;
        if at_spawn || !alive || round >= self.items_round + 25 {
            items = [0u8; ITEM_TYPES];
            for (item, count) in me.item_count()?.iter() {
                if let Some(slot) = items.get_mut(item as u32 as usize) {
                    *slot = count.min(255) as u8;
                }
            }
            self.items_round = round;
        }
        self.ledger.observe(&SelfObs {
            round,
            alive,
            health: hp,
            max_health: max_hp,
            gold,
            gems,
            ally_tributed: cambc::global_submitted_gems(Team::Ally)?,
            enemy_tributed: cambc::global_submitted_gems(Team::Enemy)?,
            items,
        });
        let hurt_this_round = alive && hp + 0.5 < self.last_hp;
        if hurt_this_round {
            self.hurt_round = round;
        }
        self.last_hp = hp;
        self.last_pos = my_pos;

        // ---- team channel (8 shared bytes) ---------------------------------
        let mut raw = [0u8; TEAM_CHANNEL_BYTES];
        cambc::shared_memory().read(0, &mut raw)?;
        let mut chan = TeamChannel::decode(raw);
        let leader = chan.begin_turn(round, self.slot);
        if leader {
            let (aggression, threshold) = policy::team_params(
                self.ledger.ally_tributed,
                self.ledger.enemy_tributed,
                round,
                cambc::constants().max_rounds,
                self.knobs.aggression,
                self.knobs.tribute_max,
            );
            chan.aggression = aggression;
            chan.tribute_threshold = threshold;
        }

        let mut exec_out = Outcome::default();
        let (mode, expert, mut intent);
        if !alive {
            // Dead champions still get a turn: shop.
            chan.set_claim(self.slot as usize, None);
            let vis = Visible::default();
            let mut exec = self.exec(me, &vis, my_pos, false);
            exec.buy()?;
            exec_out = exec.out;
            (mode, expert, intent) =
                (Mode::Override(Expert::RecallShop), Expert::RecallShop, Intent::Shop);
        } else {
            // ---- surroundings ----------------------------------------------
            let mut mark = fuel();
            self.gov.record(Stage::Observe, fuel_start.saturating_sub(mark));
            let mut vis = self.observe(me, my_pos, round)?;
            mark = self.charge(Stage::Observe, mark);
            // A teammate's sighting (vision isn't shared): treat it as a threat we can't see.
            let shared = chan.sighting(round);
            if let Some((p, age)) = shared {
                let unseen = vis.enemies.iter().all(|e| e.pos.cheb(p) > 2);
                if unseen && vis.ghosts.len() < MAX_GHOSTS && p.cheb(my_pos) <= TEAM_SIGHTING_RADIUS
                {
                    vis.ghosts.push(Threat {
                        pos: p,
                        range: TEAM_GHOST_RANGE + age as i32,
                        dps: TEAM_GHOST_DPS,
                    });
                }
            }
            if let Some(e) = vis.enemies.first() {
                chan.post_sighting(e.pos, round);
            }
            self.choose_target(my_pos, round, &chan, &vis);
            if let Some(t) = self.target {
                self.world.fields.prioritise(Goal::Group(t as u8));
            }
            mark = self.charge(Stage::Brain, mark);
            // First turns: get the fields we need right now.
            let gov = &self.gov;
            self.world.fields.precompute(&|cost| gov.turn_allows(cost + 20_000, fuel()))?;
            mark = self.charge(Stage::Precompute, mark);

            let ctx = self.context(round, my_pos, hp, max_hp, &vis, &chan, state, me)?;
            mode = gate::rule_gate(&ctx);
            expert = router::select(mode, &ctx, None);
            intent = expert.plan(&ctx);
            // Answer a teammate's call for help: healthy, not carrying a big bag, nothing
            // in our own view, and the fight is close. Safety overrides still win (Routine only).
            let mut assist = None;
            if mode == Mode::Routine
                && chan.help
                && vis.enemies.is_empty()
                && ctx.hp_frac() > ASSIST_MIN_HP
                && ctx.gems < router::BIG_BAG_GEMS
            {
                if let Some((p, age)) = shared {
                    if age <= 1 && (2..=TEAM_SIGHTING_RADIUS).contains(&p.cheb(my_pos)) {
                        intent = Intent::Engage(p);
                        assist = Some(p);
                        telemetry::event(round, champ_name(self.kind), "assist", 1);
                    }
                }
            }
            self.charge(Stage::Brain, mark);
            let claim = match intent {
                Intent::ClearCamp(g) | Intent::Capture(g) | Intent::Approach(g) => {
                    self.priority_index(g as usize)
                }
                _ => None,
            };
            chan.set_claim(self.slot as usize, claim);
            if ctx.low_hp() && ctx.in_danger() {
                chan.help = true;
            }

            let channeling = state.intersects(
                ChampionState::RECALLING
                    | ChampionState::SUBMITTING_GEMS
                    | ChampionState::TELEPORTING,
            );
            if !channeling {
                let before = fuel();
                let mut exec = self.exec(me, &vis, my_pos, hurt_this_round);
                exec.assist = assist;
                exec.execute(intent)?;
                exec_out = exec.out;
                let total = before.saturating_sub(fuel());
                self.gov.record(Stage::Path, exec_out.path_fuel);
                self.gov.record(Stage::Act, total.saturating_sub(exec_out.path_fuel));
            }
            if exec_out.moved {
                self.last_pos = pos(me.position()?);
            }
            // Give up on enemies we keep chasing without landing anything.
            if let Intent::Engage(target) = intent {
                if exec_out.attacked || exec_out.cast {
                    self.engage_stall = 0;
                } else {
                    self.engage_stall += 1;
                    if self.engage_stall >= ENGAGE_STALL_TURNS {
                        if let Some(e) = vis.enemies.iter().find(|e| e.pos == target) {
                            self.ignore_enemy = Some((e.kind, round + IGNORE_ENEMY_ROUNDS));
                        }
                        self.engage_stall = 0;
                    }
                }
            } else {
                self.engage_stall = 0;
            }

            // Telemetry (free in debug runs, compiled out in release). Features
            // every 4th turn feed the learned-head datasets.
            cambc::debug_block! {
                let feats = self.turns.is_multiple_of(4)
                    .then(|| self.raw_obs(&ctx, me).ok())
                    .flatten()
                    .map(|r| extract(&r, GroupMask::ALL).to_hex());
                self.log(round, &ctx, &vis, (mode, expert, intent), &exec_out, fuel_start, leader, feats);
            }
        }
        cambc::shared_memory().write(0, &chan.encode())?;

        // Spare fuel → finish the remaining distance fields.
        let gov = &self.gov;
        self.world.fields.precompute(&|cost| gov.turn_allows(cost, fuel()))?;
        if !alive {
            telemetry::event(round, champ_name(self.kind), "dead", i64::from(gold));
        }
        let _ = (mode, expert, intent, exec_out);
        Ok(())
    }

    /// Charge fuel spent since `mark` to `stage`; returns the new mark.
    fn charge(&mut self, stage: Stage, mark: u64) -> u64 {
        let now = fuel();
        self.gov.record(stage, mark.saturating_sub(now));
        now
    }

    fn exec<'a>(
        &'a mut self,
        me: &'a Champion,
        vis: &'a Visible,
        my_pos: Pos,
        hurt: bool,
    ) -> Exec<'a> {
        Exec {
            me,
            world: &self.world,
            vis,
            kit: &self.kit,
            ledger: &mut self.ledger,
            order: &self.order,
            my_pos,
            occupied: occupied_tiles(vis, my_pos),
            hurt_this_round: hurt,
            assist: None,
            out: Outcome::default(),
        }
    }

    fn priority_index(&self, group: usize) -> Option<u8> {
        self.theta.camp_priority.iter().position(|&(g, _)| g as usize == group).map(|i| i as u8)
    }

    // ---- observation ------------------------------------------------------

    fn observe(&mut self, me: &Champion, my_pos: Pos, round: u32) -> Result<Visible> {
        let mut vis = Visible::default();
        for e in cambc::vision::champions(Some(Team::Enemy))? {
            let Ok(p) = e.position() else { continue };
            let p = pos(p);
            let kind = e.champion_name().map(sdk::kind).unwrap_or(ChampionKind::Ryth);
            let kit = sdk::kit(kind);
            let hp = e.health().unwrap_or(0.0);
            if hp <= 0.0 {
                self.belief.forget(kind);
                continue;
            }
            let dps = sdk::dps(
                e.strength().unwrap_or(kit.base_strength),
                e.focus().unwrap_or(kit.base_focus),
                &kit,
            );
            let view = EnemyView {
                pos: p,
                dist: my_pos.cheb(p),
                hp,
                max_hp: e.max_health().unwrap_or(hp),
                dps,
                gems: e.gem_count().unwrap_or(0),
                ranged: kit.attack == AttackType::Ranged,
                range: kit.attack_range,
                on_spawn: self.world.terrain.enemy_spawn.get(p),
                kind,
                handle: e,
            };
            self.belief.see_enemy(Sighting {
                kind,
                pos: p,
                round,
                hp_frac: view.hp / view.max_hp.max(1.0),
                gems: view.gems,
                attack_range: view.range,
                dps,
            });
            vis.enemies.push(view);
        }
        vis.enemies.sort_by_key(|e| e.dist);
        // Enemies seen recently but not now (stepped into a bush or out of sight):
        // keep them as pessimistic threats so pathing avoids where they vanished.
        for s in self.belief.recent(round, GHOST_MAX_AGE) {
            if vis.ghosts.len() >= MAX_GHOSTS || vis.enemies.iter().any(|e| e.kind == s.kind) {
                continue;
            }
            let age = round.saturating_sub(s.round);
            if age == 0 || s.pos.cheb(my_pos) > 12 {
                continue;
            }
            let confidence = 1.0 - age as f32 / (GHOST_MAX_AGE as f32 + 1.0);
            vis.ghosts.push(Threat {
                pos: s.pos,
                range: s.attack_range + 1 + (age as i32 + 1) / 2,
                dps: s.dps * confidence,
            });
        }

        for a in cambc::vision::champions(Some(Team::Ally))? {
            let Ok(p) = a.position() else { continue };
            let p = pos(p);
            if p == my_pos {
                continue;
            }
            let kind = a.champion_name().map(sdk::kind).unwrap_or(ChampionKind::Ryth);
            let kit = sdk::kit(kind);
            vis.allies.push(AllyView {
                pos: p,
                dist: my_pos.cheb(p),
                hp: a.health().unwrap_or(0.0),
                max_hp: a.max_health().unwrap_or(1.0),
                dps: sdk::dps(kit.base_strength, kit.base_focus, &kit),
                handle: a,
            });
        }

        let mut seen_groups = Vec::new();
        for c in cambc::vision::camps()? {
            let Ok(state) = c.state() else { continue };
            if state.is_dead() {
                continue;
            }
            let p = c.position().map(pos).unwrap_or_else(|_| pos(c.spawn_position()));
            let group = self.world.group_at(pos(c.spawn_position()));
            if let Some(g) = group {
                if !seen_groups.contains(&g) {
                    seen_groups.push(g);
                    self.belief.observe_group(g, GroupObs::Alive, round);
                }
            }
            vis.camps.push(CampView {
                pos: p,
                dist: my_pos.cheb(p),
                hp: c.health().unwrap_or(0.0),
                awake: state.is_awake(),
                strength: c.strength().unwrap_or(0.0),
                group,
                handle: c,
            });
        }

        // Groups in sight with no living camp: dead (reward) or respawning.
        // The target is always checked; others when close, at most 2 per turn.
        let mut checked = 0;
        for (i, g) in self.world.groups.iter().enumerate() {
            let near = g.tiles.iter().map(|t| t.cheb(my_pos)).min().unwrap_or(i32::MAX);
            let is_target = self.target == Some(i);
            if seen_groups.contains(&i) || near > VISION || (!is_target && checked >= 2) {
                continue;
            }
            if !is_target {
                checked += 1;
            }
            let state = g.sdk.state();
            match state {
                Ok(CampGroupState::RewardAvailable) => {
                    self.belief.observe_group(i, GroupObs::RewardAvailable, round)
                }
                Ok(CampGroupState::Respawning(t)) => {
                    if self.belief.group(i).is_some_and(|b| b.state == GroupState::RewardAvailable)
                        && self.world.group_at(my_pos) == Some(i)
                    {
                        telemetry::event(round, champ_name(self.kind), "captured", i as i64);
                    }
                    self.belief.observe_group(i, GroupObs::Respawning(t), round);
                }
                Ok(CampGroupState::Alive) => self.belief.observe_group(i, GroupObs::Alive, round),
                // Not readable (spawn tiles hidden), yet we're right next to them
                // and see no living camp: it's down. Re-check in a few rounds.
                Err(_) if near <= 1 => {
                    self.belief.observe_group(i, GroupObs::Respawning(INFER_DOWN_ROUNDS), round)
                }
                Err(_) => {}
            }
        }
        let _ = me;
        Ok(vis)
    }

    // ---- targeting --------------------------------------------------------

    /// Walking distance to a group. Straight-line estimate until its field is
    /// computed; `None` once the field shows it can't be reached on foot
    /// (islands needing a dash or teleport crystal).
    fn group_distance(&self, i: usize, from: Pos) -> Option<u32> {
        let goal = Goal::Group(i as u8);
        if self.world.fields.is_ready(goal) {
            self.world.fields.distance(goal, from).map(u32::from)
        } else {
            Some((self.world.groups[i].center.cheb(from) as u32) * 6 / 5 + 2)
        }
    }

    /// Value per round of effort, with claims, respawn timers and hysteresis.
    fn choose_target(&mut self, my_pos: Pos, round: u32, chan: &TeamChannel, vis: &Visible) {
        let my_dps = sdk::dps(self.kit.base_strength, self.kit.base_focus, &self.kit).max(10.0);
        let team_power = my_dps * self.team_size as f32;
        let mut best: Option<(f32, usize)> = None;
        for (i, g) in self.world.groups.iter().enumerate() {
            if g.has_monolith
                && team_power * 10.0 < f32::from(self.knobs.monolith_min_power) * 100.0
            {
                continue;
            }
            let state = self.belief.expected_state(i, round);
            let Some(dist) = self.group_distance(i, my_pos) else { continue };
            let dist = dist as f32;
            let eta = self.belief.rounds_until_up(i, round) as f32;
            if state == GroupState::Respawning && eta > dist + 15.0 {
                continue;
            }
            let mut value = g.gems as f32 + g.gold as f32 / 3.0;
            let clear = match state {
                GroupState::RewardAvailable => {
                    value *= 1.5;
                    0.0
                }
                _ => g.total_hp / my_dps,
            };
            // A capture needs one champion; an ally already on the tile has it.
            let ally_on_tile = vis.allies.iter().any(|a| self.world.group_at(a.pos) == Some(i));
            if state == GroupState::RewardAvailable && ally_on_tile {
                continue;
            }
            if let Some(pi) = self.priority_index(i) {
                let others = chan.claimants(pi, self.slot as usize);
                if others > 0 {
                    // Big groups go faster together; small ones are wasted on two.
                    value *= if g.total_hp >= 3000.0 { 1.1 } else { 0.35 };
                }
            }
            // Stronger camps hurt: discount by expected damage taken.
            value /= 1.0 + g.max_strength / 100.0;
            let mut score = value / (dist + clear + (eta - dist).max(0.0) + 5.0);
            if self.target == Some(i) {
                score *= 1.3;
            }
            if best.is_none_or(|(b, _)| score > b) {
                best = Some((score, i));
            }
        }
        self.target = best.map(|(_, i)| i);
    }

    // ---- context ----------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn context(
        &self,
        round: u32,
        my_pos: Pos,
        hp: f32,
        max_hp: f32,
        vis: &Visible,
        chan: &TeamChannel,
        state: ChampionState,
        me: &Champion,
    ) -> Result<Context> {
        let fields = &self.world.fields;
        let target = self.target.map(|i| {
            let g = &self.world.groups[i];
            let hp_left: f32 = vis.camps.iter().filter(|c| c.group == Some(i)).map(|c| c.hp).sum();
            TargetGroup {
                index: i as u8,
                priority_index: self.priority_index(i),
                dist: self
                    .group_distance(i, my_pos)
                    .unwrap_or(u32::from(u16::MAX))
                    .min(u32::from(u16::MAX)) as u16,
                state: self.belief.expected_state(i, round),
                hp_left: if hp_left > 0.0 { hp_left } else { g.total_hp },
                gems: g.gems,
                gold: g.gold,
                strength: g.max_strength,
            }
        });
        // Enemies on their own spawn heal and can't be reached: never a fight target.
        // Neither is one we just failed to reach (e.g. across a wall).
        let ignored = |k: ChampionKind| {
            self.ignore_enemy.is_some_and(|(kind, until)| kind == k && round < until)
        };
        let enemy =
            vis.enemies.iter().find(|e| !e.on_spawn && !ignored(e.kind)).map(|e| EnemyNear {
                pos: e.pos,
                dist: e.dist,
                hp: e.hp,
                max_hp: e.max_hp,
                dps: e.dps,
                gems: e.gems,
                ranged: e.ranged,
            });
        let strength = me.strength().unwrap_or(self.kit.base_strength);
        let focus = me.focus().unwrap_or(self.kit.base_focus);
        let next_shard = shop::next_shard(&self.order, &self.ledger);
        Ok(Context {
            round,
            alive: true,
            me: my_pos,
            hp,
            max_hp,
            my_dps: sdk::dps(strength, focus, &self.kit),
            ranged: self.kit.attack == AttackType::Ranged,
            gems: self.ledger.gems,
            gold: self.ledger.gold,
            ally_tributed: self.ledger.ally_tributed,
            enemy_tributed: self.ledger.enemy_tributed,
            winning_gems: cambc::constants().winning_gems,
            at_spawn: self.world.terrain.ally_spawn.get(my_pos),
            dist_altar: fields.distance(Goal::Altars, my_pos),
            dist_spawn: fields.distance(Goal::AllySpawn, my_pos),
            target,
            on_capture_tile: self.target.is_some_and(|t| self.world.group_at(my_pos) == Some(t)),
            enemy,
            enemies_in_range: vis.enemies.iter().filter(|e| e.dist <= e.range + 1).count() as u8,
            ally_dps_near: vis.allies.iter().filter(|a| a.dist <= 3).map(|a| a.dps).sum(),
            camp_threat_here: vis
                .camps
                .iter()
                .filter(|c| c.awake && c.dist <= 3)
                .map(|c| c.strength)
                .sum(),
            threat_here: self.belief.threat_at(my_pos, round),
            fuel_remaining: fuel(),
            can_afford_shard: self.ledger.gold >= crate::bc_core::ledger::SHARD_COST
                && next_shard.is_some(),
            channeling: state.intersects(ChampionState::RECALLING | ChampionState::SUBMITTING_GEMS),
            knobs: self.knobs,
            team: *chan,
        })
    }

    /// Raw observation for telemetry features (cheap groups only; vision groups
    /// are approximated from the context to avoid extra SDK calls).
    fn raw_obs(&self, ctx: &Context, me: &Champion) -> Result<RawObs> {
        let mut raw = RawObs {
            me: RawSelf {
                hp: ctx.hp,
                max_hp: ctx.max_hp,
                strength: me.strength().unwrap_or(0.0),
                focus: me.focus().unwrap_or(0.0),
                ranged: ctx.ranged,
                attack_range: self.kit.attack_range,
                cooldowns: [
                    me.abilities()[0].cooldown().unwrap_or(0),
                    me.abilities()[1].cooldown().unwrap_or(0),
                ],
                can_move: me.can_move().unwrap_or(false),
                can_attack: me.can_attack().unwrap_or(false),
                gems: ctx.gems,
                gold: ctx.gold,
            },
            economy: RawEconomy {
                lifetime_gold: self.ledger.lifetime_gold,
                ally_tributed: ctx.ally_tributed,
                enemy_tributed: ctx.enemy_tributed,
                carried: ctx.gems,
                emblems: Family::ALL.iter().map(|&f| u32::from(self.ledger.emblems(f))).sum(),
                has_altar_lock: false,
                has_tp_crystal: false,
            },
            ..RawObs::default()
        };
        raw.objectives.dist_altar = ctx.dist_altar.map(i32::from);
        raw.objectives.dist_spawn = ctx.dist_spawn.map(i32::from);
        raw.objectives.dist_target = ctx.target.map(|t| i32::from(t.dist));
        raw.objectives.on_capture_tile = ctx.on_capture_tile;
        raw.objectives.gems = ctx.gems;
        raw.objectives.at_spawn = ctx.at_spawn;
        raw.belief.threat_here = ctx.threat_here;
        raw.team.aggression = ctx.team.aggression;
        raw.team.tribute_threshold = ctx.team.tribute_threshold;
        raw.team.help = ctx.team.help;
        raw.theta.knob_aggression = self.knobs.aggression;
        raw.theta.knob_retreat_hp = self.knobs.retreat_hp_pct;
        raw.theta.knob_tribute_min = self.knobs.tribute_min;
        raw.theta.map_cells = (self.world.width * self.world.height) as u32;
        raw.theta.monolith = self.world.groups.iter().any(|g| g.has_monolith);
        if let Some(v) = self.theta.champ_vec(self.kind as u8) {
            raw.champ_vecs.me = *v;
        }
        if let Some(e) = &ctx.enemy {
            raw.enemies.units.push(crate::bc_brain::features::RawUnit {
                dx: e.pos.x - ctx.me.x,
                dy: e.pos.y - ctx.me.y,
                dist: e.dist,
                hp_frac: e.hp / e.max_hp.max(1.0),
                gems: e.gems,
                ranged: e.ranged,
                dps: e.dps,
                stunned: false,
            });
        }
        raw.enemies.in_range3 = u32::from(ctx.enemies_in_range);
        raw.camps.threat_here = ctx.camp_threat_here;
        if let Some(t) = &ctx.target {
            raw.camps.target_hp = t.hp_left;
            raw.camps.target_gems = t.gems;
            raw.camps.target_gold = t.gold;
        }
        raw.allies.dps_near = ctx.ally_dps_near;
        raw.meta.round = ctx.round;
        raw.meta.fuel = ctx.fuel_remaining;
        Ok(raw)
    }

    #[allow(clippy::too_many_arguments)]
    fn log(
        &self,
        round: u32,
        ctx: &Context,
        vis: &Visible,
        (mode, expert, intent): (Mode, Expert, Intent),
        out: &Outcome,
        fuel_start: u64,
        leader: bool,
        features: Option<String>,
    ) {
        let enemies: Vec<String> = vis
            .enemies
            .iter()
            .map(|e| {
                format!("[{},{},{}]", e.pos.x, e.pos.y, (100.0 * e.hp / e.max_hp.max(1.0)) as i32)
            })
            .collect();
        let camps: Vec<String> = vis
            .camps
            .iter()
            .map(|c| {
                format!(
                    "[{},{},{},{}]",
                    c.pos.x,
                    c.pos.y,
                    c.group.map_or(-1, |g| g as i64),
                    u8::from(c.awake)
                )
            })
            .collect();
        let mode_name = match mode {
            Mode::Routine => "routine",
            Mode::Complex => "complex",
            Mode::Override(_) => "override",
        };
        telemetry::turn(&TurnLog {
            round,
            champ: champ_name(self.kind),
            pos: ctx.me,
            hp: ctx.hp_frac(),
            gold: ctx.gold,
            gems: ctx.gems,
            lifetime_gold: self.ledger.lifetime_gold,
            ally_tributed: ctx.ally_tributed,
            mode: mode_name,
            expert: expert.name(),
            intent: format!("{intent:?}"),
            tier: out.tier_name(),
            target: self.target.map(|t| t as u8),
            fuel_start,
            fuel_end: fuel(),
            stages: self.gov.used(),
            leader,
            features,
            enemies: format!("[{}]", enemies.join(",")),
            camps: format!("[{}]", camps.join(",")),
        });
    }
}

fn champ_name(kind: ChampionKind) -> &'static str {
    match kind {
        ChampionKind::Ryth => "Ryth",
        ChampionKind::VesperVale => "VesperVale",
        ChampionKind::Vendetta => "Vendetta",
        ChampionKind::Cassi => "Cassi",
        ChampionKind::Issac => "Issac",
        ChampionKind::Stellar => "Stellar",
        ChampionKind::Prism => "Prism",
        ChampionKind::Marcella => "Marcella",
        ChampionKind::Willow => "Willow",
        ChampionKind::Ren => "Ren",
        ChampionKind::Varg => "Varg",
        ChampionKind::Tavi => "Tavi",
    }
}
