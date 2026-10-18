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
use crate::bc_core::memory::{knob, TeamChannel, ThetaMatch, TEAM_CHANNEL_BYTES};
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
    /// Per enemy kind: ignored as a fight target until this round (we couldn't reach it).
    /// One slot per kind, so several unreachable enemies are ignored together (a single
    /// slot let the target flip between them forever: the rushbait 2000-round stall).
    ignore_until: [u32; 12],
    /// Health lost since our previous turn (what is actually hitting us).
    dmg_taken: f32,
    /// Rounds spent waiting for allies before waking a dangerous group (CAMP_SAFE bit 2).
    gather_wait: u32,
    /// Our current damage per round with items (CAMP_FEASIBLE).
    cur_dps: f32,
}

/// Engage turns without landing anything before we give up on that enemy.
const ENGAGE_STALL_TURNS: u32 = 4;
/// How long to ignore an unreachable enemy.
const IGNORE_ENEMY_ROUNDS: u32 = 40;
/// Remembered enemies (not in vision) count as threats this many rounds after the sighting.
const GHOST_MAX_AGE: u32 = 8;
/// Cap on remembered threats per turn (each costs a pass over the path window).
const MAX_GHOSTS: usize = 5;
/// Fuel that must remain before the combat search runs (measured ~5-25K per search).
const COMBAT_SEARCH_MIN_FUEL: u64 = 60_000;
/// Rollouts stop below this much fuel (the rest of the turn needs it; never over-commit).
const COMBAT_SEARCH_FLOOR: u64 = 45_000;
/// Team-fight evaluation radii and decision thresholds (our strength / theirs).
const SKIRMISH_ALLY_RADIUS: i32 = 4;
const SKIRMISH_ENEMY_RADIUS: i32 = 5;
const SKIRMISH_LOSE: f32 = 0.8;
const SKIRMISH_WIN: f32 = 1.5;
/// Deny tributes within this many tiles, while above this health fraction.
const DENY_RADIUS: i32 = 6;
const DENY_MIN_HP: f32 = 0.35;
/// Answer a teammate's tributer alert from this far (DENY_TRIBUTE >= 3).
const TEAM_DENY_RADIUS: i32 = 14;
/// From this round, bank everything we carry (knob ENDGAME_BANK).
const ENDGAME_ROUND: u32 = 1850;
/// Hunt mode: sightings answered from this far, and the trigger thresholds (mode 2).
const HUNT_SIGHTING_RADIUS: i32 = 20;
const HUNT_LATE_ROUND: u32 = 600;
const HUNT_BEHIND_GEMS: u32 = 300;
const HUNT_CLOSE_GEMS: u32 = 150;
/// Teammate sightings further than this are ignored (threat and help alike).
const TEAM_SIGHTING_RADIUS: i32 = 12;
/// Reach and damage assumed for an enemy only a teammate saw (its kind isn't shared).
const TEAM_GHOST_RANGE: i32 = 3;
const TEAM_GHOST_DPS: f32 = 40.0;
/// Only champions above this health fraction answer a call for help.
const ASSIST_MIN_HP: f32 = 0.7;
/// FIGHT_V6: join fights from this health fraction.
const ASSIST_MIN_HP_V6: f32 = 0.5;
/// TAVI_META: Tavi's field build (shop.rs profile) and when it starts hunting.
const TAVI_BUILD_PROFILE: u8 = 12;
const TAVI_HUNT_EMBLEMS: u32 = 4;
const TAVI_HUNT_MIN_HP: f32 = 0.5;
const TAVI_HUNT_RADIUS: i32 = 6;
/// CAMP_FEASIBLE: a group needing more than this many times our survival time to clear
/// keeps only CAMP_INFEASIBLE_VALUE of its value.
const CAMP_FEASIBLE_SLACK: f32 = 3.0;
const CAMP_INFEASIBLE_VALUE: f32 = 0.1;
/// PASSIVE_ROLES: Vesper farms gold groups until she owns this many emblems.
const VESPER_GOLD_EMBLEMS: u32 = 3;
/// FIGHT_V6 last stand: never with a bag this big (the killer takes it).
const LAST_STAND_MAX_GEMS: u32 = 20;

impl Bot {
    pub fn new() -> Result<Self> {
        let world = World::build()?;
        let me = cambc::self_champion()?.ok_or_else(|| sdk::local_error("no self champion"))?;
        let kind = sdk::kind(me.champion_name()?);
        let kit = sdk::kit(kind);
        let theta = ThetaMatch::decode(cambc::draft_memory().bytes()).unwrap_or_default();
        let build_profile = theta.knob(knob::BUILD_PROFILE);
        let order = if theta.knob(knob::TAVI_META) > 0 && kind == ChampionKind::Tavi {
            shop::build_order_profile(kind, TAVI_BUILD_PROFILE)
        } else {
            shop::build_order_profile(kind, build_profile)
        };
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
            order,
            slot,
            target: None,
            last_hp: 0.0,
            hurt_round: 0,
            last_pos,
            items_round: 0,
            turns: 0,
            engage_stall: 0,
            ignore_until: [0; 12],
            dmg_taken: 0.0,
            gather_wait: 0,
            cur_dps: 0.0,
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
        self.dmg_taken = if alive { (self.last_hp - hp).max(0.0) } else { 0.0 };
        if alive {
            self.cur_dps = sdk::dps(
                me.strength().unwrap_or(self.kit.base_strength),
                me.focus().unwrap_or(self.kit.base_focus),
                &self.kit,
            );
        }
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
            let deny_level = self.theta.knob(knob::DENY_TRIBUTE);
            if deny_level >= 3 {
                if let Some(e) = vis.enemies.iter().find(|e| e.tributing) {
                    chan.post_tributer(e.pos, round);
                }
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

            let mut ctx = self.context(round, my_pos, hp, max_hp, &vis, &chan, state, me)?;
            let hunting = self.hunting(&ctx);
            if hunting {
                // All-in: no fight margin (the router then fights at advantage >= 1).
                ctx.knobs.aggression = 255;
                ctx.team.aggression = 255;
            }
            mode = gate::rule_gate(&ctx);
            expert = router::select(mode, &ctx, None);
            intent = expert.plan(&ctx);
            // Answer a teammate's call for help: healthy, not carrying a big bag, nothing
            // in our own view, and the fight is close. Safety overrides still win (Routine only).
            let mut assist = None;
            // Hunting: every healthy champion answers every sighting, from further away.
            let radius = if hunting { HUNT_SIGHTING_RADIUS } else { TEAM_SIGHTING_RADIUS };
            let fight_v6 = self.theta.knob(knob::FIGHT_V6);
            // FIGHT_V6 bit 2: join the team's fight unless something is already on us.
            let free_to_assist = if fight_v6 & 2 != 0 {
                !matches!(mode, Mode::Override(_))
                    && vis.enemies.iter().all(|e| e.dist > 4)
                    && ctx.hp_frac() > ASSIST_MIN_HP_V6
            } else {
                mode == Mode::Routine && vis.enemies.is_empty() && ctx.hp_frac() > ASSIST_MIN_HP
            };
            if free_to_assist && (chan.help || hunting) && ctx.gems < router::BIG_BAG_GEMS {
                if let Some((p, age)) = shared {
                    if age <= 1 && (2..=radius).contains(&p.cheb(my_pos)) {
                        intent = Intent::Engage(p);
                        assist = Some(p);
                        telemetry::event(round, champ_name(self.kind), "assist", 1);
                    }
                }
            }
            // Deny tributes: an enemy channelling a tribute nearby loses the whole bag to a
            // stun, a kill or a fresh altar lock (damage alone doesn't stop it).
            let deny = self.theta.knob(knob::DENY_TRIBUTE) > 0;
            if deny && !matches!(mode, Mode::Override(_)) && ctx.hp_frac() > DENY_MIN_HP {
                if let Some(e) = vis.enemies.iter().find(|e| e.tributing && e.dist <= DENY_RADIUS) {
                    intent = Intent::Engage(e.pos);
                } else if deny_level >= 3 && chan.enemy_tributing && ctx.gems < router::BIG_BAG_GEMS {
                    // A teammate saw an enemy channelling: go cancel it (long bags take turns).
                    if let Some((p, age)) = chan.sighting(round) {
                        if age <= 1 && p.cheb(my_pos) <= TEAM_DENY_RADIUS {
                            intent = Intent::Engage(p);
                            assist = Some(p);
                            telemetry::event(round, champ_name(self.kind), "team_deny", 1);
                        }
                    }
                }
            }
            // Tavi (ladder meta): once geared, hunt carriers, tributers and weak enemies in reach.
            if self.theta.knob(knob::TAVI_META) > 0
                && self.kind == ChampionKind::Tavi
                && !matches!(mode, Mode::Override(_))
                && ctx.hp_frac() > TAVI_HUNT_MIN_HP
                && ctx.gems < router::BIG_BAG_GEMS
                && self.emblems() >= TAVI_HUNT_EMBLEMS
            {
                let allies_near = vis.allies.iter().filter(|a| a.dist <= 4).count();
                let prey = vis
                    .enemies
                    .iter()
                    .filter(|e| e.dist <= TAVI_HUNT_RADIUS && !e.on_spawn)
                    .filter(|e| e.gems >= 10 || e.tributing || e.hp < 0.5 * e.max_hp)
                    .filter(|e| {
                        vis.enemies.iter().filter(|o| o.pos.cheb(e.pos) <= 4).count() <= allies_near + 1
                    })
                    .min_by(|a, b| a.hp.total_cmp(&b.hp));
                if let Some(e) = prey {
                    intent = Intent::Engage(e.pos);
                    telemetry::event(round, champ_name(self.kind), "tavi_hunt", 1);
                }
            }
            // Fight balance: finish kills, take fights we outnumber, and make a last stand
            // instead of dying in a retreat we can't win; then call the target for the team.
            if fight_v6 & (4 | 8) != 0 {
                if let Some((fix, why)) =
                    fight_fix(fight_v6, &ctx, &vis, intent, mode, self.kit.attack_range)
                {
                    telemetry::event(round, champ_name(self.kind), why, 1);
                    intent = fix;
                }
            }
            if fight_v6 & 1 != 0 {
                if let Intent::Engage(p) = intent {
                    if vis.enemies.iter().any(|e| e.pos == p) {
                        chan.post_sighting(p, round);
                        chan.help = true;
                    }
                }
            }
            // Camp safety: camps hit the nearest champion for their strength every round and
            // don't chase past range 3, so leave while escaping still beats their damage, and
            // don't be the lone champion that wakes a group it can't survive.
            let camp_safe = self.theta.knob(knob::CAMP_SAFE);
            if camp_safe > 0 && !matches!(mode, Mode::Override(_)) {
                let fix = camp_safety(
                    camp_safe,
                    &vis,
                    intent,
                    hp,
                    self.dmg_taken,
                    self.kit_dps(),
                    &mut self.gather_wait,
                );
                if let Some(fix) = fix {
                    telemetry::event(round, champ_name(self.kind), "camp_safe", 1);
                    intent = fix;
                }
            }
            // Team fight (portfolio choice between fight and retreat, scored for the whole
            // skirmish): Lanchester square law, strength = total dps x total hp per side.
            if self.theta.knob(knob::TEAM_FIGHT) > 0 && !matches!(mode, Mode::Override(_)) {
                if let Some(ratio) = skirmish_ratio(&vis, hp, self.kit_dps()) {
                    let weakest = vis
                        .enemies
                        .iter()
                        .filter(|e| e.dist <= SKIRMISH_ENEMY_RADIUS && !e.on_spawn)
                        .min_by(|a, b| a.hp.total_cmp(&b.hp));
                    match intent {
                        Intent::Engage(_) | Intent::Kite(_) if ratio < SKIRMISH_LOSE => {
                            intent = Intent::Retreat
                        }
                        Intent::Retreat | Intent::Engage(_) | Intent::Kite(_)
                            if ratio > SKIRMISH_WIN
                                && ctx.hp_frac() > 0.5
                                && ctx.gems < router::BIG_BAG_GEMS =>
                        {
                            if let Some(e) = weakest {
                                intent = Intent::Engage(e.pos);
                            }
                        }
                        _ => {}
                    }
                }
            }
            // Portfolio combat search: simulate scripted skirmish behaviours, take the best
            // worst case. Only near enemies and with fuel to spare (sized from measured cost).
            if self.theta.knob(knob::COMBAT_SEARCH) > 0
                && !matches!(mode, Mode::Override(_))
                && fuel() > COMBAT_SEARCH_MIN_FUEL
                && vis.enemies.iter().any(|e| e.dist <= 6 && !e.on_spawn)
            {
                let before = fuel();
                let pick = crate::skirmish::choose(
                    &vis,
                    my_pos,
                    hp,
                    self.kit_dps(),
                    self.kit.attack_range,
                    ctx.gems,
                    self.world.spawn_hint,
                    COMBAT_SEARCH_FLOOR,
                );
                telemetry::event(
                    round,
                    champ_name(self.kind),
                    "search_fuel",
                    before.saturating_sub(fuel()) as i64,
                );
                if let Some((plan, target)) = pick {
                    use crate::skirmish::Plan;
                    let carrying = ctx.gems >= router::BIG_BAG_GEMS;
                    intent = match (plan, target) {
                        (Plan::Retreat, _) => Intent::Retreat,
                        _ if carrying => intent,
                        (Plan::Kite, Some(p)) if self.kit.attack_range > 1 => Intent::Kite(p),
                        (_, Some(p)) => Intent::Engage(p),
                        (_, None) => intent,
                    };
                }
            }
            // Endgame: at round 2000 tributed gems decide; never finish holding gems.
            if self.theta.knob(knob::ENDGAME_BANK) > 0
                && round >= ENDGAME_ROUND
                && ctx.gems > 0
                && !matches!(mode, Mode::Override(_))
            {
                intent = Intent::Tribute;
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

            let mut channeling = state.intersects(
                ChampionState::RECALLING
                    | ChampionState::SUBMITTING_GEMS
                    | ChampionState::TELEPORTING,
            );
            // A recall is several frozen turns that damage cancels: abort it while we can run.
            if channeling
                && state.contains(ChampionState::RECALLING)
                && self.theta.knob(knob::RECALL_SAFE) > 0
            {
                let left = me.remaining_channel_time().unwrap_or(5).max(1);
                if vis.enemies.iter().any(|e| !e.on_spawn && e.dist <= e.range + left + 1)
                    && me.cancel_recall().is_ok()
                {
                    channeling = false;
                    telemetry::event(round, champ_name(self.kind), "recall_cancel", 1);
                }
            }
            if !channeling {
                let before = fuel();
                let deny_ground = self.theta.knob(knob::DENY_TRIBUTE) >= 2;
                let altar_lock = self.theta.knob(knob::ALTAR_LOCK) > 0;
                let field_shop =
                    self.theta.knob(knob::TAVI_META) > 0 && self.kind == ChampionKind::Tavi;
                let recall_safe = self.theta.knob(knob::RECALL_SAFE) >= 2;
                let recall_ticks = if self.has_home() { 3 } else { 5 };
                let mut exec = self.exec(me, &vis, my_pos, hurt_this_round);
                exec.field_shop = field_shop;
                exec.recall_safe = recall_safe;
                exec.recall_ticks = recall_ticks;
                exec.assist = assist;
                exec.deny_tribute = deny;
                exec.deny_ground = deny_ground;
                exec.altar_lock = altar_lock;
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
                            if let Some(slot) = self.ignore_until.get_mut(e.kind as usize) {
                                *slot = round + IGNORE_ENEMY_ROUNDS;
                            }
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
            deny_tribute: false,
            deny_ground: false,
            altar_lock: false,
            field_shop: false,
            recall_safe: false,
            recall_ticks: 5,
            out: Outcome::default(),
        }
    }

    /// Hunt mode (knob HUNT_MODE): 1 always; 2 when far behind late, or when the enemy is
    /// close to winning while ahead of us (only kills can stop them then).
    fn hunting(&self, ctx: &Context) -> bool {
        match self.theta.knob(knob::HUNT_MODE) {
            1 => true,
            2 => {
                let (us, them) = (ctx.ally_tributed, ctx.enemy_tributed);
                let behind_late = ctx.round > HUNT_LATE_ROUND && them >= us + HUNT_BEHIND_GEMS;
                let they_close =
                    them + HUNT_CLOSE_GEMS >= ctx.winning_gems && them > us + HUNT_CLOSE_GEMS;
                behind_late || they_close
            }
            _ => false,
        }
    }

    /// Knobs with per-champion adjustments (PASSIVE_ROLES: Prism carries bigger bags).
    fn role_knobs(&self) -> Knobs {
        let mut k = self.knobs;
        if self.theta.knob(knob::PASSIVE_ROLES) > 0 && self.kind == ChampionKind::Prism {
            k.tribute_min = (u32::from(k.tribute_min) * 3 / 2).min(u32::from(k.tribute_max)) as u8;
        }
        k
    }

    /// Completed emblems we own.
    fn emblems(&self) -> u32 {
        self.ledger.items.iter().skip(1).step_by(2).map(|&n| u32::from(n)).sum()
    }

    /// We own a Home emblem (recall takes 3 ticks instead of 5).
    fn has_home(&self) -> bool {
        use crate::bc_core::ledger::item_id;
        [Stat::Strength, Stat::Health, Stat::Focus]
            .iter()
            .any(|&s| self.ledger.items[item_id(Family::Home, s, true) as usize] > 0)
    }

    fn kit_dps(&self) -> f32 {
        sdk::dps(self.kit.base_strength, self.kit.base_focus, &self.kit).max(1.0)
    }

    fn priority_index(&self, group: usize) -> Option<u8> {
        self.theta.camp_priority.iter().position(|&(g, _)| g as usize == group).map(|i| i as u8)
    }

    // ---- observation ------------------------------------------------------

    fn observe(&mut self, me: &Champion, my_pos: Pos, round: u32) -> Result<Visible> {
        let mut vis = Visible::default();
        let stun_threat = self.theta.knob(knob::STUN_THREAT) > 0;
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
            let (strength, focus) =
                (e.strength().unwrap_or(kit.base_strength), e.focus().unwrap_or(kit.base_focus));
            let dps = if stun_threat {
                sdk::dps_burst(strength, focus, &kit)
            } else {
                sdk::dps(strength, focus, &kit)
            };
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
                tributing: self.theta.knob(knob::DENY_TRIBUTE) > 0
                    && e.state().is_ok_and(|s| s.contains(ChampionState::SUBMITTING_GEMS)),
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
            let mut value = match self.theta.knob(knob::VALUE_PROFILE) {
                1 => g.gems as f32 + g.gold as f32 / 10.0,
                2 if self.slot < 2 => g.gems as f32 * 0.3 + g.gold as f32,
                2 => g.gems as f32 + g.gold as f32 * 0.15,
                _ => g.gems as f32 + g.gold as f32 / 3.0,
            };
            // Passives: Vesper's gold bonus applies to her capture share (gear herself first),
            // Prism's Focus grows with the gems she carries (she collects them).
            if self.theta.knob(knob::PASSIVE_ROLES) > 0 {
                match self.kind {
                    ChampionKind::VesperVale if self.emblems() < VESPER_GOLD_EMBLEMS => {
                        value = g.gems as f32 * 0.25 + g.gold as f32 * 1.2;
                    }
                    ChampionKind::Prism => value = g.gems as f32 * 1.4 + g.gold as f32 / 3.0,
                    _ => {}
                }
            }
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
                // Squads: slot pairs (0,1), (2,3) join each other's group (2v1 fights, faster clears).
                let join = self.theta.knob(knob::SQUAD_JOIN_X10);
                let buddy =
                    chan.claims.get((self.slot ^ 1) as usize).copied().flatten() == Some(pi);
                if join > 0 && buddy {
                    value *= f32::from(join) / 10.0;
                } else if others > 0 {
                    // Big groups go faster together; small ones are wasted on two.
                    value *= if g.total_hp >= 3000.0 { 1.1 } else { 0.35 };
                }
            }
            // Stronger camps hurt: discount by expected damage taken.
            value /= 1.0 + g.max_strength / 100.0;
            // Can the champions on it clear it before the tank dies? Gear up on others first.
            if self.theta.knob(knob::CAMP_FEASIBLE) > 0 && state != GroupState::RewardAvailable {
                let helpers =
                    1 + self.priority_index(i).map_or(0, |pi| chan.claimants(pi, self.slot as usize));
                let survive = self.last_hp.max(1.0) / (g.total_strength * CAMP_DMG_FLOOR).max(1.0);
                let kill = g.total_hp / (self.cur_dps.max(my_dps) * helpers as f32).max(1.0);
                if kill > survive * CAMP_FEASIBLE_SLACK {
                    value *= CAMP_INFEASIBLE_VALUE;
                }
            }
            // Remote farmer: the last slot of a 4+ team goes further for quieter groups.
            let remote = self.theta.knob(knob::REMOTE_FARMER) > 0
                && self.team_size >= 4
                && self.slot as usize + 1 == self.team_size;
            let dist_w = if remote { 0.4 } else { 1.0 };
            let mut score = value / (dist * dist_w + clear + (eta - dist).max(0.0) + 5.0);
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
        let ignored =
            |k: ChampionKind| self.ignore_until.get(k as usize).is_some_and(|&until| round < until);
        let nearest = || vis.enemies.iter().find(|e| !e.on_spawn && !ignored(e.kind));
        let enemy = if self.theta.knob(knob::FIGHT_V6) & 1 != 0 {
            // Weakest enemy in reach, the team's called target first.
            let reach = self.kit.attack_range + 2;
            let called = chan.sighting(round).filter(|&(_, age)| age == 0).map(|(p, _)| p);
            vis.enemies
                .iter()
                .filter(|e| !e.on_spawn && !ignored(e.kind) && e.dist <= reach)
                .min_by(|a, b| {
                    (called != Some(a.pos)).cmp(&(called != Some(b.pos))).then(a.hp.total_cmp(&b.hp))
                })
                .or_else(nearest)
        } else {
            nearest()
        };
        let enemy = enemy.map(|e| EnemyNear {
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
            knobs: self.role_knobs(),
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

/// FIGHT_V6: a better intent when we'd otherwise leave a fight we should take.
/// Returns the intent and a telemetry tag.
fn fight_fix(
    bits: u8,
    ctx: &Context,
    vis: &Visible,
    intent: Intent,
    mode: Mode,
    my_range: i32,
) -> Option<(Intent, &'static str)> {
    let e = ctx.enemy?;
    if !matches!(intent, Intent::Retreat | Intent::Shop | Intent::Kite(_)) {
        return None;
    }
    let forced = matches!(mode, Mode::Override(_));
    let our_dps = (ctx.my_dps + ctx.ally_dps_near).max(1.0);
    if bits & 4 != 0 && !forced && ctx.gems < router::BIG_BAG_GEMS {
        // Finish it: two turns of our damage kill the target.
        if e.dist <= my_range + 1 && e.hp <= our_dps * 2.0 {
            return Some((Intent::Engage(e.pos), "finish"));
        }
        // Outnumbering: take the even fight.
        let allies = 1 + vis.allies.iter().filter(|a| a.dist <= 4).count();
        let enemies = vis.enemies.iter().filter(|o| o.dist <= 5 && !o.on_spawn).count();
        if allies > enemies && router::fight_advantage(ctx, &e) >= 1.0 {
            return Some((Intent::Engage(e.pos), "outnumber"));
        }
    }
    // Last stand: something that out-ranges or stuns us is already on us, and walking home
    // would only feed it the kill: trade damage instead.
    if bits & 8 == 0 || ctx.gems >= LAST_STAND_MAX_GEMS {
        return None;
    }
    let on_us: Vec<&EnemyView> =
        vis.enemies.iter().filter(|o| !o.on_spawn && o.dist <= o.range + 1).collect();
    let chased = on_us.iter().any(|o| o.ranged || sdk::has_stun(o.kind));
    let their: f32 = on_us.iter().map(|o| o.dps).sum();
    let turns_home = ctx.dist_spawn.map_or(5.0, |d| d as f32).min(5.0);
    if chased && their * turns_home >= ctx.hp {
        let weakest = on_us
            .iter()
            .filter(|o| o.dist <= my_range + 1)
            .min_by(|a, b| a.hp.total_cmp(&b.hp))
            .or_else(|| on_us.iter().min_by_key(|o| o.dist))?;
        return Some((Intent::Engage(weakest.pos), "last_stand"));
    }
    None
}

/// Damage taken from camps is reduced by Jungle (30%); estimates use this floor.
const CAMP_DMG_FLOOR: f32 = 0.7;
/// Allies this close count towards clearing the camp we are at.
const CAMP_ALLY_RADIUS: i32 = 4;
/// Longest wait for allies before waking a group we can't take alone.
const GATHER_MAX_WAIT: u32 = 8;

/// CAMP_SAFE (bits: 1 escape, 2 gather): a safer intent when camps are about to kill us.
fn camp_safety(
    bits: u8,
    vis: &Visible,
    intent: Intent,
    hp: f32,
    dmg_taken: f32,
    my_dps: f32,
    gather_wait: &mut u32,
) -> Option<Intent> {
    let team_dps =
        my_dps + vis.allies.iter().filter(|a| a.dist <= CAMP_ALLY_RADIUS).map(|a| a.dps).sum::<f32>();
    // Escape: awake camps in range that target us (no ally is closer to them).
    if bits & 1 != 0 {
        let on_me: Vec<&CampView> = vis
            .camps
            .iter()
            .filter(|c| c.awake && c.hp > 0.0 && c.dist <= 3)
            .filter(|c| !vis.allies.iter().any(|a| a.pos.cheb(c.pos) < c.dist))
            .collect();
        if !on_me.is_empty() {
            let est: f32 = on_me.iter().map(|c| c.strength).sum::<f32>() * CAMP_DMG_FLOOR;
            let incoming = dmg_taken.max(est).max(1.0);
            let nearest = on_me.iter().map(|c| c.dist).min().unwrap_or(3);
            // Hits still taken while walking out of range 3, one tile per turn.
            let hits_out = (3 - nearest).max(0) as f32 + 1.0;
            let groups: Vec<Option<usize>> = on_me.iter().map(|c| c.group).collect();
            let group_hp: f32 = vis
                .camps
                .iter()
                .filter(|c| c.hp > 0.0 && groups.contains(&c.group))
                .map(|c| c.hp)
                .sum();
            let kill_turns = group_hp / team_dps.max(1.0);
            let survive = hp / incoming;
            if kill_turns + 1.0 > survive && survive <= hits_out + 1.5 {
                return Some(Intent::Retreat);
            }
        }
    }
    // Gather: we'd be first to hit a sleeping group that kills a lone tank before it dies.
    if bits & 2 != 0 {
        if let Intent::ClearCamp(g) = intent {
            let g = Some(g as usize);
            let group: Vec<&CampView> =
                vis.camps.iter().filter(|c| c.group == g && c.hp > 0.0).collect();
            let asleep = !group.is_empty() && group.iter().all(|c| !c.awake);
            if asleep {
                let strength: f32 = group.iter().map(|c| c.strength).sum::<f32>() * CAMP_DMG_FLOOR;
                let group_hp: f32 = group.iter().map(|c| c.hp).sum();
                let survive = hp / strength.max(1.0);
                let kill_turns = group_hp / team_dps.max(1.0);
                if kill_turns + 2.0 > survive && *gather_wait < GATHER_MAX_WAIT {
                    *gather_wait += 1;
                    return Some(Intent::Approach(g.unwrap_or(0) as u8));
                }
            }
        }
        *gather_wait = 0;
    }
    None
}

/// Our skirmish strength over theirs (Lanchester square law: dps sum x hp sum per side),
/// counting allies within SKIRMISH_ALLY_RADIUS and enemies (seen or remembered) within
/// SKIRMISH_ENEMY_RADIUS. `None` when no enemy is close enough for it to matter.
fn skirmish_ratio(vis: &Visible, my_hp: f32, my_dps: f32) -> Option<f32> {
    let allies = vis.allies.iter().filter(|a| a.dist <= SKIRMISH_ALLY_RADIUS);
    let (mut ours_hp, mut ours_dps) = (my_hp.max(1.0), my_dps);
    for a in allies {
        ours_hp += a.hp.max(0.0);
        ours_dps += a.dps;
    }
    let enemies: Vec<_> =
        vis.enemies.iter().filter(|e| e.dist <= SKIRMISH_ENEMY_RADIUS && !e.on_spawn).collect();
    if enemies.is_empty() {
        return None;
    }
    let theirs_hp: f32 = enemies.iter().map(|e| e.hp.max(1.0)).sum();
    let theirs_dps: f32 = enemies.iter().map(|e| e.dps).sum::<f32>()
        + vis.ghosts.iter().filter(|g| g.range >= 1).map(|g| g.dps * 0.5).sum::<f32>();
    Some((ours_hp * ours_dps) / (theirs_hp * theirs_dps.max(1.0)))
}
