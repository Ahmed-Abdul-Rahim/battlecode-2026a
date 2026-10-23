//! One champion's turn: observe → belief/ledger → team channel → target →
//! meta-gate → router → expert intent → executor → telemetry.

use crate::act::{occupied_tiles, AllyView, CampView, EnemyView, Exec, Outcome, Visible};
use crate::bc_brain::context::{Context, EnemyNear, Knobs, TargetGroup};
use crate::bc_brain::experts::N_EXPERTS;
use crate::bc_brain::features::{extract, GroupMask, RawEconomy, RawObs, RawSelf};
use crate::bc_brain::features_gen::SCHEMA_HASH;
use crate::bc_brain::int8::Mlp;
use crate::bc_brain::rl::{self, TargetMode};
use crate::bc_brain::{gate, policy, router, shop, Expert, Intent, Mode};
use crate::bc_core::belief::{Belief, GroupObs, GroupState, Sighting};
use crate::bc_core::champion::{AttackType, ChampionKind, KitInfo};
use crate::bc_core::combat::{CallIn, Directive, EnemyIn, Fsm, Inputs, Tuning};
use crate::bc_core::fuel::{FuelGovernor, Stage};
use crate::bc_core::ledger::{Family, Ledger, SelfObs, Stat, ITEM_TYPES};
use crate::bc_core::memory::{knob, knob2, TeamChannel, ThetaMatch, TEAM_CHANNEL_BYTES};
use crate::bc_core::Pos;
use crate::bc_path::danger::Threat;
use crate::bc_path::{DistanceOracle, Goal};
use crate::sdk::{self, fuel, pos};
use crate::telemetry::{self, TurnLog};
use crate::world::World;
use cambc::{CampGroupState, Champion, ChampionState, Result, Team};

#[path = "teacher.rs"]
mod teacher;

/// Vision radius for deciding which camp groups we can check this turn.
const VISION: i32 = 8;

/// Camps further than this (from their spawn) only count as alive; no other queries.
const FAR_CAMP: i32 = 7;

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
    /// Fuel the last full target scoring took (~1.5k per group on dense maps).
    target_cost: u64,
    /// Turns the target was kept without rescoring because fuel was short.
    target_kept: u8,
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
    ignore_until: [u32; 13],
    /// Policy-pack death hotspots for this map: (tile, weight 0..1).
    hotspots: Vec<(Pos, f32)>,
    /// Policy-pack opening route for this champion kind and spawn side (group indices).
    opening: Vec<u8>,
    opening_idx: usize,
    /// Health lost since our previous turn (what is actually hitting us).
    dmg_taken: f32,
    /// Rounds spent waiting for allies before waking a dangerous group (CAMP_SAFE bit 2).
    gather_wait: u32,
    /// Our current damage per round with items (CAMP_FEASIBLE, MONOLITH_LIVE).
    cur_dps: f32,
    cur_max_hp: f32,
    /// TEAM_RESPAWN: a group we just saw go down (index, respawn round) to broadcast.
    saw_down: Option<(usize, u32)>,
    /// knob2 CLUSTER: per group, its cluster neighbours with a 0..1 nearness weight and whether
    /// they came from walking distances (exact) or the straight-line estimate.
    cluster: Vec<Option<(Vec<(u8, f32)>, bool)>>,
    /// Our passive (engine PassiveType value), read live at start-up (knob2 PASSIVE_AWARE).
    passive: usize,
    /// knob2 UNSTICK: our last positions (ring) and turns since we last hit or got hit.
    recent: [Pos; UNSTICK_TURNS],
    recent_n: usize,
    idle_turns: usize,
    /// COMBAT_FSM: combat state machine with history (bc_core::combat).
    fsm: Fsm,
    /// COMBAT_FSM: the enemy we watched last turn (kind, hp), to measure damage dealt.
    fsm_seen: Option<(ChampionKind, f32)>,
    /// COMBAT_FSM bit 64: last fight-or-flee choice against the visible enemy.
    fsm_fighting: Option<bool>,
    /// knob2 TEACHER: this champion's previous observations (nn/ pipeline).
    nn_hist: teacher::History,
}

/// Engage turns without landing anything before we give up on that enemy.
/// knob2 DRAFT_PLAN bit 1: frontier groups are worth this much while enemy champions are within
/// PLAN_ENEMY_CLOSE tiles.
const PLAN_FRONTIER_VALUE: f32 = 0.7;
const PLAN_ENEMY_CLOSE: i32 = 10;
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
/// COMBAT_FSM bit 128: answer a help call only within this detour (tiles) on our way, and
/// from at most this far.
const HELP_DETOUR: i32 = 4;
const HELP_ON_WAY_MAX: i32 = 10;
/// CAMP_FEASIBLE bit 8: join an ally's camp fight from this far.
const PVE_ASSIST_RADIUS: i32 = 4;
/// CAMP_FEASIBLE bit 16: deliver at an altar this far off our route, at most this far away.
const ALTAR_DETOUR: i32 = 3;
const ALTAR_ON_WAY_MAX: i32 = 8;
/// COMBAT_FSM 64|2: finish an enemy at <= 50% a teammate reports within this many tiles.
const KILL_ASSIST_RADIUS: i32 = 4;
const KILL_ASSIST_MIN_HP: f32 = 0.4;
/// Engine PassiveType values used by knob2 PASSIVE_AWARE.
const PASSIVE_BONUS_GOLD: usize = 1;
const PASSIVE_SPLASH: usize = 4;
const PASSIVE_MISSING_HEALTH: usize = 10;
/// knob2 PUNISH_RECALL: chase a lone recaller only above this health fraction.
const PUNISH_MIN_HP: f32 = 0.4;
/// knob2 UNSTICK: positions remembered; a loop needs this many idle turns over two tiles.
const UNSTICK_TURNS: usize = 12;
/// knob2 SOLO_CAP: solo a big group only if clearing it takes this much less than surviving it.
const SOLO_GEARED_MARGIN: f32 = 1.5;
/// knob2 EASE_FIRST: each extra monster divides a group's value by this much more.
const EASE_PER_MONSTER: f32 = 0.25;
/// knob2 EARLY_GOLD: extra value per gold on quick groups before 2 emblems.
const EARLY_GOLD_BONUS: f32 = 0.5;
/// knob2 TEAM_UP: join only above this health fraction, and unless the joint fight advantage
/// (allies' damage included) is below this.
const TEAM_UP_MIN_HP: f32 = 0.4;
const TEAM_UP_MIN_ADVANTAGE: f32 = 0.7;
/// knob2 UPGRADE_FIRST: shop first when the fight advantage is below this, or the camp takes
/// more rounds than this to clear alone; only from this far from spawn.
const UPGRADE_HARD_ADVANTAGE: f32 = 1.5;
const UPGRADE_SLOW_CLEAR: f32 = 25.0;
const UPGRADE_MIN_SPAWN_DIST: u16 = 8;
/// knob2 FOCUS: call our target for the team once it is at or below this health fraction.
const FOCUS_CALL_HP: f32 = 0.6;
/// knob2 ENDGAME: from this round (or when a team has 80% of the winning gems); bank bags from
/// this size when ahead.
const ENDGAME_START: u32 = 1700;
const ENDGAME_BANK_MIN: u32 = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Endgame {
    Off,
    Ahead,
    Behind,
}

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
/// CAMP_FEASIBLE bit 2: the early-game ramp (quick camps first) lasts this long, and a camp
/// starts losing value past this many turns to clear.
const EARLY_RAMP_ROUNDS: u32 = 300;
const EARLY_RAMP_KNEE: f32 = 20.0;
const CAMP_INFEASIBLE_VALUE: f32 = 0.1;
/// Fuel the rest of a turn needs after target scoring (path 40k + act 5k + reserve).
const TARGET_REST_FUEL: u64 = 55_000;
/// Below this before acting, only Intent::Hold (attack what is in range) runs: clearing a camp
/// spends ~10k before it walks and the walk ~10k more (measured on yellow-brick-road).
const EXEC_MIN_FUEL: u64 = 25_000;
/// Fuel a turn still needs after a mid-turn field precompute (context, path, act).
const PRECOMPUTE_REST_FUEL: u64 = 25_000;
/// ... and when our own target has no field yet (the fresh field makes pathing cheap).
const URGENT_REST_FUEL: u64 = 12_000;
/// Most turns in a row a target is kept unscored when fuel is short.
const TARGET_KEEP_MAX: u8 = 8;
/// knob2 CLUSTER: neighbours count up to this many walking tiles from a group's centre.
const CLUSTER_PATH: u32 = 14;
/// Debug runs: rounds between `G` lines (each group's target score and what cut it).
const TARGET_DIAG_EVERY: u32 = 100;
/// EN_ROUTE: only for targets this far; detour, solo clear time and damage limits.
const EN_ROUTE_MIN_DIST: i32 = 15;
const EN_ROUTE_DETOUR: i32 = 6;
const EN_ROUTE_MAX_CLEAR: f32 = 25.0;
/// EN_ROUTE 2 (wide, engine 1.0.0 economy): stop for targets this far, camps up to this many turns,
/// and free money (rewards, zero-damage clusters) within this radius and detour.
const EN_ROUTE_WIDE_MIN_DIST: i32 = 8;
const EN_ROUTE_WIDE_MAX_CLEAR: f32 = 40.0;
const EN_ROUTE_FREE_RADIUS: i32 = 4;
/// MONOLITH_LIVE: attack once pooled team health lasts this fraction of the kill time.
const MONOLITH_SLACK: f32 = 1.0;
/// VALUE_PROFILE 4/5: the opening (near camps first: early gold compounds into items).
const OPENING_ROUNDS: u32 = 120;
const OPENING_DIST_WEIGHT: f32 = 2.5;
/// PASSIVE_ROLES: Vesper farms gold groups until she owns this many emblems.
const VESPER_GOLD_EMBLEMS: u32 = 3;
/// Policy-pack opening routes are followed until this round.
const PACK_OPENING_ROUNDS: u32 = 200;
/// Pack hotspots: considered within this radius, at most this many, scaled from this dps.
const HOTSPOT_RADIUS: i32 = 8;
const MAX_HOTSPOTS: usize = 4;
const HOTSPOT_DPS: f32 = 60.0;
/// FIGHT_V6 last stand: never with a bag this big (the killer takes it).
const LAST_STAND_MAX_GEMS: u32 = 20;

impl Bot {
    pub fn new() -> Result<Self> {
        let theta = ThetaMatch::decode(cambc::draft_memory().bytes()).unwrap_or_default();
        let world =
            World::build(theta.knob(knob::SAFE_ALTAR) > 0, theta.knob2(knob2::CLUSTER) > 0)?;
        // Policy pack: this map's learned parameters and hotspots (harness/pack.py). The draft
        // passes the fingerprint (hashing the map here cost ~50-60k fuel on big maps).
        let fp = if theta.map_fp != 0 {
            theta.map_fp
        } else {
            crate::world::fingerprint(
                world.width,
                world.height,
                &world.terrain.walls,
                &world.terrain.holes,
                &world.terrain.altars,
            )
        };
        let entry = crate::pack::lookup(fp);
        crate::learned::init(entry.theta);
        let hotspots: Vec<(Pos, f32)> = entry
            .hotspots
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| (Pos::new(i32::from(c[0]), i32::from(c[1])), f32::from(c[2]) / 255.0))
            .collect();
        let me = cambc::self_champion()?.ok_or_else(|| sdk::local_error("no self champion"))?;
        let kind = sdk::kind(me.champion_name()?);
        let kit = sdk::kit(kind);
        let passive = me.passive().map_or(0, |p| p as i32 as usize);
        let build_profile = theta.knob(knob::BUILD_PROFILE);
        let opening = entry.opening_for((world.spawn_hint.x, world.spawn_hint.y), kind as usize);
        let pack_build = entry.build_for(kind as usize);
        let order = if !pack_build.is_empty() {
            // Policy-pack build (mimicked from top teams / evolved per map).
            pack_build
                .iter()
                .filter_map(|&(f, s)| {
                    Some((*Family::ALL.get(usize::from(f))?, *Stat::ALL.get(usize::from(s))?))
                })
                .collect()
        } else if theta.knob(knob::TAVI_META) > 0 && kind == ChampionKind::Tavi {
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
        let bot = Self {
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
            target_cost: 0,
            target_kept: 0,
            last_hp: 0.0,
            hurt_round: 0,
            last_pos,
            items_round: 0,
            turns: 0,
            engage_stall: 0,
            ignore_until: [0; 13],
            hotspots,
            opening,
            opening_idx: 0,
            dmg_taken: 0.0,
            gather_wait: 0,
            cur_dps: 0.0,
            cur_max_hp: 0.0,
            saw_down: None,
            cluster: Vec::new(),
            recent: [Pos::new(-1, -1); UNSTICK_TURNS],
            recent_n: 0,
            idle_turns: 0,
            passive,
            fsm: Fsm::default(),
            fsm_seen: None,
            fsm_fighting: None,
            nn_hist: teacher::History::default(),
        };
        if bot.teacher_mode() > 0 {
            bot.teacher_header();
        }
        Ok(bot)
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
            self.cur_max_hp = max_hp;
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
        let fsm_bits = self.theta.knob(knob::COMBAT_FSM);
        let fsm_on = fsm_bits > 0;
        // Bit 64: hysteresis only (no states, calls or channel rules): the old mechanisms stay.
        let fsm_full = fsm_on && fsm_bits & 64 == 0;
        // With hysteresis (64): 2 = finish hurt enemies teammates report nearby.
        let kill_assist = fsm_bits & (64 | 2) == 64 | 2;
        let leader = chan.begin_turn(round, self.slot);
        let team_respawn = self.theta.knob(knob::TEAM_RESPAWN) > 0;
        if leader && !team_respawn {
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
            if fsm_full {
                // Dead: back to ECONOMY, keep the fight history.
                let hist = self.fsm.hist;
                self.fsm = Fsm::default();
                self.fsm.hist = hist;
                self.fsm_seen = None;
            }
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
            // Learned death hotspots near us act as standing threats (pathing steers around them).
            let hot_w = crate::learned::hotspot_w();
            if hot_w > 0.0 {
                for &(p, w) in self
                    .hotspots
                    .iter()
                    .filter(|(p, _)| p.cheb(my_pos) <= HOTSPOT_RADIUS)
                    .take(MAX_HOTSPOTS)
                {
                    vis.ghosts.push(Threat { pos: p, range: 1, dps: HOTSPOT_DPS * w * hot_w });
                }
            }
            if let Some(e) = vis.enemies.first() {
                chan.post_sighting_hp(e.pos, round, e.hp / e.max_hp.max(1.0));
            }
            let deny_level = self.theta.knob(knob::DENY_TRIBUTE);
            if deny_level >= 3 {
                if let Some(e) = vis.enemies.iter().find(|e| e.tributing) {
                    chan.post_tributer(e.pos, round);
                }
            }
            if team_respawn {
                self.share_respawns(&mut chan, round);
            }
            self.choose_target(my_pos, round, &chan, &vis);
            if let Some(t) = self.target {
                self.world.fields.prioritise(Goal::Group(t as u8));
            }
            mark = self.charge(Stage::Brain, mark);
            // First turns: get the fields we need right now.
            let gov = &self.gov;
            // Keep what the rest of the turn needs (context, pathing on a fresh field, actions).
            // The out-of-fuel turns came from underestimated field costs (now measured); keeping
            // TARGET_REST_FUEL here instead starved maps like yellow-brick-road of fields
            // altogether (0-15 gems a game, 2026-10-23).
            // No field for our own target yet: that is urgent (without it the bot walks straight
            // lines and gets stuck on maze maps); a fresh field makes the rest of the turn cheap.
            let urgent = self.target.is_some_and(|t| !self.world.fields.is_ready(Goal::Group(t as u8)));
            let rest = if urgent { URGENT_REST_FUEL } else { PRECOMPUTE_REST_FUEL };
            self.world.fields.precompute(&|cost| gov.turn_allows(cost + rest, fuel()))?;
            mark = self.charge(Stage::Precompute, mark);

            let mut ctx = self.context(round, my_pos, hp, max_hp, &vis, &chan, state, me)?;
            let endgame = self.endgame(&ctx);
            if endgame == Endgame::Ahead {
                // Ahead late: take only clearly winning fights (dying hands them gems and time).
                ctx.knobs.fight_margin_pct = ctx.knobs.fight_margin_pct.saturating_mul(2).max(40);
            }
            // knob2 PASSIVE_AWARE: play to the live passive, without champion names. Fight
            // margin, retreat health and bag sizes scale per passive type (learned PS block)
            // and margin / aggression per champion kind (KD block); priors in learned.rs.
            if self.theta.knob2(knob2::PASSIVE_AWARE) > 0 {
                let (pt, k) = (self.passive, self.kind as usize);
                let scale = |v: u8, m: f32| (f32::from(v) * m).round().clamp(0.0, 255.0) as u8;
                let margin = crate::learned::passive(pt, 0) * crate::learned::kind(k, 0);
                ctx.knobs.fight_margin_pct = scale(ctx.knobs.fight_margin_pct, margin);
                ctx.knobs.retreat_hp_pct =
                    scale(ctx.knobs.retreat_hp_pct, crate::learned::passive(pt, 1));
                ctx.knobs.tribute_min =
                    scale(ctx.knobs.tribute_min, crate::learned::passive(pt, 2));
                ctx.knobs.tribute_max =
                    scale(ctx.knobs.tribute_max, crate::learned::passive(pt, 2));
                ctx.knobs.aggression = scale(ctx.knobs.aggression, crate::learned::kind(k, 1));
                // Missing-health damage (Heart of Ruin-like): live strength doesn't show the bonus.
                if pt == PASSIVE_MISSING_HEALTH {
                    ctx.my_dps *= 1.0 + 0.5 * (1.0 - ctx.hp_frac());
                }
            }
            // COMBAT_FSM 64|32: Marcella earns 2 gold per gem delivered, so small bags pay
            // too and carry less risk: halve her delivery thresholds.
            if fsm_bits & (64 | 32) == 64 | 32 && self.kind == ChampionKind::Marcella {
                ctx.knobs.tribute_min /= 2;
                ctx.knobs.tribute_max = (ctx.knobs.tribute_max / 2).max(ctx.knobs.tribute_min + 1);
            }
            let hunting = self.hunting(&ctx) || endgame == Endgame::Behind;
            if hunting {
                // All-in: no fight margin (the router then fights at advantage >= 1).
                ctx.knobs.aggression = 255;
                ctx.team.aggression = 255;
            }
            mode = gate::rule_gate(&ctx);
            ctx.fight_bias = self.fight_bias(&ctx, &vis);
            // knob2 ROUTER with an arm-1 network: it picks the expert and the fight target.
            let rl_pick = self.rl_policy(mode, &ctx, me, round);
            expert = rl_pick.map_or_else(|| self.pick_expert(mode, &ctx, me), |p| p.0);
            intent = expert.plan(&ctx);
            if let Some(tp) = rl_pick.and_then(|p| self.rl_target(p.1, &vis)) {
                intent = match intent {
                    Intent::Engage(_) => Intent::Engage(tp),
                    Intent::Kite(_) => Intent::Kite(tp),
                    other => other,
                };
            }
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
                mode == Mode::Routine
                    && vis.enemies.is_empty()
                    && ctx.hp_frac() > ASSIST_MIN_HP * crate::learned::w(crate::learned::ASSIST_HP)
            };
            // COMBAT_FSM answers calls itself (by intent and time left); hunting still swarms.
            if (!fsm_full || hunting)
                && free_to_assist
                && (chan.help || hunting)
                && ctx.gems < ctx.knobs.big_bag()
            {
                if let Some((p, age)) = shared {
                    // COMBAT_FSM bit 128: only help when the fight is on the way to where we're
                    // already going (small detour), so helping costs no farming time.
                    let on_way =
                        hunting || fsm_bits & 128 == 0 || self.on_the_way(my_pos, p, intent);
                    if on_way && age <= 1 && (2..=radius).contains(&p.cheb(my_pos)) {
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
                } else if deny_level >= 3 && chan.enemy_tributing && ctx.gems < ctx.knobs.big_bag()
                {
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
            // knob2 PUNISH_RECALL: an enemy recalling alone nearby; one hit cancels the recall.
            let punish = i32::from(self.theta.knob2(knob2::PUNISH_RECALL));
            if punish > 0 && !matches!(mode, Mode::Override(_)) && ctx.hp_frac() > PUNISH_MIN_HP {
                let lone = |p: Pos| vis.enemies.iter().filter(|o| o.pos.cheb(p) <= 4).count() <= 1;
                if let Some(e) = vis
                    .enemies
                    .iter()
                    .find(|e| e.recalling && !e.on_spawn && e.dist <= punish && lone(e.pos))
                {
                    intent = Intent::Engage(e.pos);
                    telemetry::event(round, champ_name(self.kind), "punish_recall", 1);
                }
            }
            // Tavi (ladder meta): once geared, hunt carriers, tributers and weak enemies in reach.
            if self.theta.knob(knob::TAVI_META) > 0
                && self.kind == ChampionKind::Tavi
                && !matches!(mode, Mode::Override(_))
                && ctx.hp_frac() > TAVI_HUNT_MIN_HP
                && ctx.gems < ctx.knobs.big_bag()
                && self.emblems() >= TAVI_HUNT_EMBLEMS
            {
                let allies_near = vis.allies.iter().filter(|a| a.dist <= 4).count();
                let prey = vis
                    .enemies
                    .iter()
                    .filter(|e| e.dist <= TAVI_HUNT_RADIUS && !e.on_spawn)
                    .filter(|e| e.gems >= 10 || e.tributing || e.hp < 0.5 * e.max_hp)
                    .filter(|e| {
                        vis.enemies.iter().filter(|o| o.pos.cheb(e.pos) <= 4).count()
                            <= allies_near + 1
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
            if fight_v6 & 1 != 0 && !fsm_full {
                if let Intent::Engage(p) = intent {
                    if vis.enemies.iter().any(|e| e.pos == p) {
                        chan.post_sighting(p, round);
                        chan.help = true;
                    }
                }
            }
            // Utility arbiter (learned, per map): scores keep / step back / support / fight /
            // walk home / recall together. Heuristics remain the fallback when fuel is short.
            let cheap_fuel =
                u64::from(crate::learned::extra(crate::learned::X_F_CHEAP) as u32) * 1000;
            if !matches!(mode, Mode::Override(_)) && fuel() > cheap_fuel.max(gate::LOW_FUEL) {
                if let Some((choice, k)) = crate::utility::choose(
                    &ctx,
                    &vis,
                    my_pos,
                    self.kit.attack_range,
                    crate::learned::utility(),
                    intent,
                ) {
                    telemetry::event(round, champ_name(self.kind), "utility", k as i64);
                    intent = choice;
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
                                && ctx.gems < ctx.knobs.big_bag() =>
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
                    let carrying = ctx.gems >= ctx.knobs.big_bag();
                    intent = match (plan, target) {
                        (Plan::Retreat, _) => Intent::Retreat,
                        _ if carrying => intent,
                        (Plan::Kite, Some(p)) if self.kit.attack_range > 1 => Intent::Kite(p),
                        (_, Some(p)) => Intent::Engage(p),
                        (_, None) => intent,
                    };
                }
            }
            // Finish a hurt enemy (<= 50%) a teammate reports within KILL_ASSIST_RADIUS.
            let calm = !matches!(mode, Mode::Override(_))
                && vis.enemies.iter().all(|e| e.on_spawn || e.dist > 4)
                && ctx.gems < ctx.knobs.big_bag();
            if kill_assist && calm && ctx.hp_frac() > KILL_ASSIST_MIN_HP && !chan.enemy_tributing {
                if let Some((p, age)) = chan.sighting(round) {
                    let near = (1..=KILL_ASSIST_RADIUS).contains(&p.cheb(my_pos));
                    if age <= 1 && near && chan.sighting_hp() <= 0.5 {
                        intent = Intent::Engage(p);
                        assist = Some(p);
                        telemetry::event(round, champ_name(self.kind), "kill_assist", 1);
                    }
                }
            }
            let camp_bits = self.theta.knob(knob::CAMP_FEASIBLE);
            let quiet = !matches!(mode, Mode::Override(_))
                && vis.enemies.iter().all(|e| e.on_spawn || e.dist > 5);
            // CAMP_FEASIBLE bit 8, PvE assist: an ally within PVE_ASSIST_RADIUS is fighting an awake
            // camp: join it on its weakest camp (faster clear, less damage taken by the ally).
            if camp_bits & 8 != 0 && quiet && ctx.hp_frac() > 0.5 && ctx.gems < ctx.knobs.big_bag()
            {
                let busy = matches!(intent, Intent::Capture(_) | Intent::Tribute | Intent::Shop);
                let fight = vis
                    .allies
                    .iter()
                    .filter(|a| a.dist <= PVE_ASSIST_RADIUS)
                    .flat_map(|a| {
                        vis.camps
                            .iter()
                            .filter(move |c| c.awake && c.hp > 0.0 && c.pos.cheb(a.pos) <= 4)
                    })
                    .min_by(|x, y| x.hp.total_cmp(&y.hp));
                if let (false, Some(c)) = (busy, fight) {
                    if let Some(g) = c.group {
                        intent = Intent::ClearCamp(g as u8);
                        telemetry::event(round, champ_name(self.kind), "pve_assist", 1);
                    }
                }
            }
            // CAMP_FEASIBLE bit 16: an altar a few tiles off our route: deliver on the way.
            if camp_bits & 16 != 0 && quiet && ctx.gems > 0 {
                let enough = ctx.gems >= (u32::from(ctx.knobs.tribute_min) / 3).max(3);
                let on_way =
                    self.world.altars.iter().any(|&a| self.altar_on_the_way(my_pos, a, intent));
                if enough && on_way && !matches!(intent, Intent::Tribute | Intent::Capture(_)) {
                    intent = Intent::Tribute;
                    telemetry::event(
                        round,
                        champ_name(self.kind),
                        "altar_on_way",
                        i64::from(ctx.gems),
                    );
                }
            }
            // Combat state machine: memory between turns (alert / engage / disengage / help).
            if fsm_on {
                let (fixed, helping) = self.fsm_turn(
                    round,
                    my_pos,
                    &ctx,
                    &vis,
                    &mut chan,
                    mode,
                    intent,
                    hurt_this_round,
                );
                intent = fixed;
                if helping.is_some() {
                    assist = helping;
                }
            }
            // knob2 TEAM_UP: an ally within N tiles is trading hits with an enemy champion: go
            // help (two separate fights lose both; one shared fight wins it).
            let team_up = i32::from(self.theta.knob2(knob2::TEAM_UP));
            if team_up > 0
                && !matches!(mode, Mode::Override(_))
                && ctx.hp_frac() > TEAM_UP_MIN_HP
                && !matches!(intent, Intent::Engage(_) | Intent::Capture(_) | Intent::Tribute)
            {
                let target = vis
                    .allies
                    .iter()
                    .filter(|a| a.dist <= team_up)
                    .flat_map(|a| {
                        vis.enemies
                            .iter()
                            .filter(move |e| !e.on_spawn && e.pos.cheb(a.pos) <= e.range.max(2))
                    })
                    .min_by(|x, y| x.hp.total_cmp(&y.hp));
                let winnable = ctx.enemy.is_none_or(|n| {
                    crate::bc_brain::router::fight_advantage(&ctx, &n) >= TEAM_UP_MIN_ADVANTAGE
                });
                if let (Some(e), true) = (target, winnable) {
                    intent = Intent::Engage(e.pos);
                    telemetry::event(round, champ_name(self.kind), "team_up", 1);
                }
            }
            // knob2 FOCUS: call the enemy we're attacking once it is hurt and an ally can help.
            if self.theta.knob2(knob2::FOCUS) > 0 {
                if let Intent::Engage(p) = intent {
                    let hurt =
                        vis.enemies.iter().find(|e| e.pos == p && e.hp <= FOCUS_CALL_HP * e.max_hp);
                    if let (Some(e), true) = (hurt, vis.allies.iter().any(|a| a.dist <= 6)) {
                        chan.post_focus(p, round, e.hp / e.max_hp.max(1.0));
                    }
                }
            }
            // knob2 UPGRADE_FIRST: about to start a fight we don't clearly win, or a camp that
            // takes long to clear, with gold in the pocket: buy first (items compound).
            let upgrade_gold = u32::from(self.theta.knob2(knob2::UPGRADE_FIRST)) * 10;
            if upgrade_gold > 0
                && ctx.gold >= upgrade_gold.max(crate::bc_core::ledger::SHARD_COST)
                && !matches!(mode, Mode::Override(_))
                && !ctx.in_danger()
                && ctx.dist_spawn.is_some_and(|d| d > UPGRADE_MIN_SPAWN_DIST)
                && ctx.gems < u32::from(ctx.knobs.tribute_min)
                && crate::bc_brain::shop::next_shard(&self.order, &self.ledger).is_some()
            {
                let hard = match intent {
                    Intent::Engage(_) => ctx.enemy.is_some_and(|e| {
                        crate::bc_brain::router::fight_advantage(&ctx, &e) < UPGRADE_HARD_ADVANTAGE
                    }),
                    Intent::ClearCamp(i) => ctx.target.is_some_and(|t| {
                        t.index == i && t.hp_left / ctx.my_dps.max(1.0) > UPGRADE_SLOW_CLEAR
                    }),
                    _ => false,
                };
                if hard {
                    telemetry::event(round, champ_name(self.kind), "upgrade_first", 1);
                    intent = Intent::Shop;
                }
            }
            // knob2 ENDGAME, ahead late: bank whatever we carry when nothing threatens us.
            if endgame == Endgame::Ahead
                && ctx.gems >= ENDGAME_BANK_MIN
                && !matches!(mode, Mode::Override(_))
                && vis.enemies.iter().all(|e| e.on_spawn || e.dist > 5)
                && !matches!(intent, Intent::Capture(_))
            {
                intent = Intent::Tribute;
            }
            // Endgame: at round 2000 tributed gems decide; never finish holding gems.
            if self.theta.knob(knob::ENDGAME_BANK) > 0
                && round >= ENDGAME_ROUND
                && ctx.gems > 0
                && !matches!(mode, Mode::Override(_))
            {
                intent = Intent::Tribute;
            }
            // knob2 UNSTICK: back and forth over two tiles for a while without fighting: reset.
            if self.theta.knob2(knob2::UNSTICK) > 0
                && !matches!(mode, Mode::Override(_))
                && !ctx.at_spawn
                && !matches!(
                    intent,
                    Intent::Capture(_)
                        | Intent::Approach(_)
                        | Intent::Tribute
                        | Intent::Hold
                        | Intent::Shop
                        | Intent::Retreat
                        | Intent::WalkHome
                )
                && self.recent_n >= UNSTICK_TURNS
                && self.idle_turns >= UNSTICK_TURNS
                && crate::bc_core::tactics::two_tile_loop(&self.recent)
            {
                intent = Intent::Shop;
                self.recent_n = 0;
                telemetry::event(round, champ_name(self.kind), "unstick", 1);
            }
            // knob2 WALK_HOME: close to spawn, walking is as fast as a recall and can't be cancelled.
            let walk_home = self.theta.knob2(knob2::WALK_HOME);
            if walk_home > 0
                && matches!(intent, Intent::Retreat | Intent::Shop)
                && !ctx.at_spawn
                && ctx.dist_spawn.is_some_and(|d| d <= u16::from(walk_home))
            {
                intent = Intent::WalkHome;
                telemetry::event(round, champ_name(self.kind), "walk_home", 1);
            }
            // nn/ teacher (knob2 TEACHER): observe; at 2 the network replaces the rules' choice.
            let teach = self.teacher_mode();
            let busy = state.intersects(
                ChampionState::RECALLING | ChampionState::SUBMITTING_GEMS | ChampionState::TELEPORTING,
            );
            let student_fuel = teach < 5
                || (round >= teacher::STUDENT_FROM_ROUND
                    && fuel() > teacher::STUDENT_MIN_FUEL
                    && self.student_ready());
            let nn_obs = (teach > 0 && !busy && student_fuel).then(|| {
                self.teacher_obs(&ctx, &vis, &chan, me, my_pos, round, hurt_this_round, intent, teach < 5)
            });
            let mut nn_choice = None;
            if let (true, Some(o)) = (teach >= 2, &nn_obs) {
                // 3 = diagnostic: the rules' own goal through the teacher's control path.
                let pick = if teach == 5 {
                    self.student_pick(o)
                } else if teach == 3 {
                    Some(teacher::Choice { a: [o.rule_goal as i32, 0, -1, -1], logp: 0.0 })
                } else if teach == 4 {
                    // 4 = diagnostic: the network's goal, the executor's own moves.
                    self.teacher_pick(o, &self.nn_hist).map(|mut c| {
                        c.a = [c.a[0], 0, -1, -1];
                        c
                    })
                } else {
                    self.teacher_pick(o, &self.nn_hist)
                };
                if let Some(c) = pick {
                    if let Some(i) = teacher::intent_of(o, c.a[0] as usize) {
                        intent = i;
                    }
                    nn_choice = Some(c);
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
            if !fsm_full && ctx.low_hp() && ctx.in_danger() {
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
                let tavi_respec = self.kind == ChampionKind::Tavi
                    && crate::learned::extra(crate::learned::X_TAVI_RESPEC) > 0.5;
                let stun_lock = self.theta.knob2(knob2::STUN_LOCK) > 0;
                let gold_order = self.theta.knob2(knob2::GOLD_ORDER);
                let lock_contested_only = self.theta.knob(knob::ALTAR_LOCK) == 2;
                let los_retreat = self.theta.knob2(knob2::LOS_RETREAT) > 0;
                let tp_crystal = self.theta.knob2(knob2::TP_CRYSTAL) > 0;
                let weak_first = self.theta.knob2(knob2::WEAK_CAMP_FIRST) > 0;
                let micro = self.micro_policy(&ctx, &vis, me, my_pos, intent, round);
                let mut exec = self.exec(me, &vis, my_pos, hurt_this_round);
                exec.gold_order = gold_order;
                exec.lock_contested_only = lock_contested_only;
                exec.los_retreat = los_retreat;
                exec.tp_crystal = tp_crystal;
                exec.weak_first = weak_first;
                exec.field_shop = field_shop;
                exec.recall_safe = recall_safe;
                exec.recall_ticks = recall_ticks;
                exec.tavi_respec = tavi_respec;
                exec.assist = assist;
                exec.deny_tribute = deny;
                exec.deny_ground = deny_ground;
                exec.altar_lock = altar_lock;
                exec.combo = fsm_bits & (64 | 8) == 64 | 8;
                exec.stun_lock = stun_lock;
                exec.dash_in = fsm_bits & (64 | 16) == 64 | 16;
                let nn_micro = nn_choice.filter(|c| c.a[1] > 0).zip(nn_obs.as_ref()).map(|(c, o)| {
                    let step = (c.a[1] >= 2).then(|| o.steps[c.a[1] as usize - 1]);
                    let target = (c.a[2] > 0).then(|| o.enemies[c.a[2] as usize - 1]).flatten();
                    (step, target, c.a[3] == 1)
                });
                // Last safety net: too little fuel for a full move (pathing can cost 10k+): only
                // the cheap hold-and-hit action this turn, instead of running out mid-move.
                if fuel() < EXEC_MIN_FUEL {
                    exec.execute(Intent::Hold)?;
                } else if let Some((step, target, cast)) = nn_micro {
                    exec.micro(step, target, cast)?;
                } else if let Some((step, target, cast)) = micro {
                    exec.micro(step, target, cast)?;
                } else {
                    exec.execute(intent)?;
                }
                exec_out = exec.out;
                let total = before.saturating_sub(fuel());
                self.gov.record(Stage::Path, exec_out.path_fuel);
                self.gov.record(Stage::Act, total.saturating_sub(exec_out.path_fuel));
            }
            if exec_out.moved {
                self.last_pos = pos(me.position()?);
            }
            if let Some(o) = &nn_obs {
                let c = nn_choice.unwrap_or_else(|| Self::teacher_labels(o));
                self.teacher_log(round, o, &c, teach);
                self.nn_hist.push(&o.s);
            }
            // RL training builds: what this turn did (wasted turns cost reward in ml/rl).
            if self.theta.knob2(knob2::RL_SEED) > 0 {
                let did = u8::from(exec_out.moved)
                    | u8::from(exec_out.attacked) << 1
                    | u8::from(exec_out.cast) << 2;
                telemetry::event(round, champ_name(self.kind), "o", i64::from(did));
            }
            // knob2 UNSTICK bookkeeping: where we ended the turn, and whether anything happened.
            self.recent[self.recent_n % UNSTICK_TURNS] =
                if exec_out.moved { self.last_pos } else { my_pos };
            self.recent_n += 1;
            self.idle_turns = if exec_out.attacked || exec_out.cast || hurt_this_round {
                0
            } else {
                self.idle_turns + 1
            };
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

    /// COMBAT_FSM: one step of the combat state machine (bc_core::combat). Returns the intent
    /// (replaced or not) and the ally position we're heading to help, if any.
    #[allow(clippy::too_many_arguments)]
    fn fsm_turn(
        &mut self,
        round: u32,
        my_pos: Pos,
        ctx: &Context,
        vis: &Visible,
        chan: &mut TeamChannel,
        mode: Mode,
        intent: Intent,
        hurt: bool,
    ) -> (Intent, Option<Pos>) {
        let mut tuning = Tuning::from_learned(
            crate::learned::extra(crate::learned::X_FSM_ALERT),
            crate::learned::extra(crate::learned::X_FSM_ASSIST),
        );
        tuning.retreat_hp = f32::from(ctx.knobs.retreat_hp_pct) / 100.0;
        // COMBAT_FSM bits beyond 1: 2 = only go home after a fight when below half health,
        // 4 = channels/camps blocked only by enemies in sight within 2 turns (not remembered
        // ones), 8 = don't answer help calls.
        let bits = self.theta.knob(knob::COMBAT_FSM);
        if bits & 2 != 0 {
            tuning.recover_below_hp = 0.5;
        }
        tuning.answer_calls = bits & 8 == 0;
        // 16 = help only from close by; 32 = when healthy, step out of reach instead of walking home.
        if bits & 16 != 0 {
            tuning.max_help_eta = 4;
        }
        let big_bag = ctx.gems >= ctx.knobs.big_bag();
        // The enemy the rule layers are looking at, with its kind and reach.
        let seen =
            ctx.enemy.and_then(|n| vis.enemies.iter().find(|e| e.pos == n.pos).map(|e| (n, e)));
        let enemy = seen.map(|(n, e)| EnemyIn {
            pos: e.pos,
            dist: e.dist,
            range: e.range,
            kind: e.kind as usize,
            hp_frac: e.hp / e.max_hp.max(1.0),
            advantage: router::fight_advantage(ctx, &n) * ctx.fight_bias.exp(),
        });
        // History: damage we did to the enemy we watched last turn vs what champions did to us.
        if let Some((_, e)) = seen {
            let dealt = match self.fsm_seen {
                Some((k, hp)) if k == e.kind => (hp - e.hp).max(0.0),
                _ => 0.0,
            };
            let champs_hit_us = vis.enemies.iter().any(|x| !x.on_spawn && x.dist <= x.range + 1);
            let taken = if champs_hit_us { self.dmg_taken } else { 0.0 };
            if e.dist <= e.range + 2 {
                self.fsm.hist.record(Some(e.kind as usize), dealt, taken);
            }
        }
        self.fsm_seen = seen.map(|(_, e)| (e.kind, e.hp));
        if self.theta.knob(knob::COMBAT_FSM) & 64 != 0 {
            return (self.fsm_hysteresis(ctx, enemy, mode, intent), None);
        }
        // Turns until something seen or remembered can hit us (enemies move a tile a turn).
        let threat_eta = vis
            .enemies
            .iter()
            .filter(|e| !e.on_spawn)
            .map(|e| (e.dist - e.range).max(0))
            .chain(vis.ghosts.iter().map(|g| (g.pos.cheb(my_pos) - g.range).max(0)))
            .min();
        let incoming: f32 = vis
            .enemies
            .iter()
            .filter(|e| !e.on_spawn && e.dist <= e.range + 2)
            .map(|e| e.dps)
            .sum::<f32>()
            + ctx.camp_threat_here;
        let survive_turns = if incoming > 0.0 { ctx.hp / incoming } else { 99.0 };
        let aggression =
            if ctx.team.version > 0 { ctx.team.aggression } else { ctx.knobs.aggression };
        let margin =
            f32::from(ctx.knobs.fight_margin_pct) / 100.0 * (255.0 - f32::from(aggression)) / 127.0;
        // Team channel v3 carries no help calls: the full machine runs without them.
        let call: Option<CallIn> = None;
        let inp = Inputs {
            round,
            alive: true,
            me: my_pos,
            hp_frac: ctx.hp_frac(),
            big_bag,
            enemy,
            threat_eta,
            hurt,
            survive_turns,
            fight_threshold: 1.0 + margin,
            call,
            recall_ticks: if self.has_home() { 3 } else { 5 },
        };
        let before = self.fsm.state;
        let (dir, out_call) = self.fsm.step(&inp, &tuning);
        if self.fsm.state != before {
            telemetry::event(round, champ_name(self.kind), self.fsm.state.name(), 1);
        }
        let _ = (out_call, chan);
        // A channel (capture 5 turns, tribute 1 per 10 gems) must finish before anything arrives.
        let channel_turns = match intent {
            Intent::Capture(_) => 5,
            Intent::Tribute => ctx.gems.div_ceil(10).max(1) as i32,
            _ => 0,
        };
        let seen_eta =
            vis.enemies.iter().filter(|e| !e.on_spawn).map(|e| (e.dist - e.range).max(0)).min();
        let (block_eta, camp_eta) = if bits & 4 != 0 {
            (seen_eta.map(|t| (t, 2)), seen_eta.map(|t| (t, 2)))
        } else {
            (threat_eta.map(|t| (t, channel_turns + 1)), threat_eta.map(|t| (t, tuning.alert_eta)))
        };
        let channel_safe = block_eta.is_none_or(|(t, limit)| t > limit);
        let camp_safe = camp_eta.is_none_or(|(t, limit)| t > limit);
        let overridden = matches!(mode, Mode::Override(_));
        match dir {
            Directive::Keep => (intent, None),
            Directive::NoChannel => {
                let fixed = match intent {
                    // Waking a camp with an enemy in sight invites a third party.
                    Intent::ClearCamp(g) if !camp_safe => Intent::Approach(g),
                    Intent::Capture(_) | Intent::Tribute if !channel_safe => Intent::Hold,
                    i => i,
                };
                (fixed, None)
            }
            Directive::Engage(p) => {
                // Under the low-health safety override only a last stand (can't escape) fights.
                let stand = !big_bag
                    && enemy.is_some_and(|e| {
                        e.dist <= e.range && e.advantage >= tuning.last_stand_advantage
                    });
                if overridden && !stand {
                    return (intent, None);
                }
                let kite =
                    self.kit.attack_range > 1 && enemy.is_some_and(|e| e.pos == p && e.range <= 1);
                (if kite { Intent::Kite(p) } else { Intent::Engage(p) }, None)
            }
            Directive::Disengage => match enemy {
                Some(e) if bits & 32 != 0 && !big_bag && ctx.hp_frac() > 0.5 => {
                    (Intent::StepBack(e.pos), None)
                }
                _ => (Intent::WalkHome, None),
            },
            Directive::Recover => (Intent::Retreat, None),
            Directive::Assist(p) if !overridden => (Intent::Engage(p), Some(p)),
            Directive::Escort(p) if !overridden => {
                // Fight only the chaser near the carrier; otherwise stand on its way home.
                match vis.enemies.iter().find(|e| !e.on_spawn && e.pos.cheb(p) <= 4) {
                    Some(e) if !big_bag => (Intent::Engage(e.pos), None),
                    _ => (Intent::Engage(p), Some(p)),
                }
            }
            Directive::Assist(_) | Directive::Escort(_) => (intent, None),
        }
    }

    /// COMBAT_FSM bit 64: only hysteresis on the existing fight-or-flee decision. The bot may
    /// switch between fighting and fleeing an enemy only when the 2-bit fight counter agrees
    /// (two turns of evidence); everything else is left to the other layers.
    fn fsm_hysteresis(
        &mut self,
        ctx: &Context,
        enemy: Option<EnemyIn>,
        mode: Mode,
        intent: Intent,
    ) -> Intent {
        let Some(e) = enemy else {
            self.fsm_fighting = None;
            return intent;
        };
        let aggression =
            if ctx.team.version > 0 { ctx.team.aggression } else { ctx.knobs.aggression };
        let margin =
            f32::from(ctx.knobs.fight_margin_pct) / 100.0 * (255.0 - f32::from(aggression)) / 127.0;
        let history_ok = !self.fsm.hist.bad_against(e.kind) && self.fsm.hist.recent_losses(3) < 3;
        self.fsm.fight.update(e.advantage >= 1.0 + margin && history_ok);
        let fights = matches!(intent, Intent::Engage(p) | Intent::Kite(p) if p == e.pos);
        let flees = matches!(intent, Intent::Retreat | Intent::WalkHome | Intent::StepBack(_));
        let kept = match self.fsm_fighting {
            // Was fighting, now told to flee: only if the counter agrees (safety overrides win).
            Some(true) if flees && self.fsm.fight.yes() && !matches!(mode, Mode::Override(_)) => {
                let kite = self.kit.attack_range > 1 && e.range <= 1;
                if kite {
                    Intent::Kite(e.pos)
                } else {
                    Intent::Engage(e.pos)
                }
            }
            // Was fleeing, now told to fight: only if the counter agrees.
            Some(false) if fights && !self.fsm.fight.yes() => Intent::WalkHome,
            _ => intent,
        };
        if matches!(kept, Intent::Engage(p) | Intent::Kite(p) if p == e.pos) {
            self.fsm_fighting = Some(true);
        } else if matches!(kept, Intent::Retreat | Intent::WalkHome | Intent::StepBack(_)) {
            self.fsm_fighting = Some(false);
        }
        kept
    }

    /// Is altar `a` at most ALTAR_DETOUR tiles off the way to a camp target (and close)?
    fn altar_on_the_way(&self, me: Pos, a: Pos, intent: Intent) -> bool {
        let goal = match intent {
            Intent::ClearCamp(g) | Intent::Approach(g) => {
                self.world.groups.get(usize::from(g)).map(|gr| gr.center)
            }
            Intent::Retreat | Intent::WalkHome => Some(self.world.spawn_hint),
            _ => None,
        };
        goal.is_some_and(|g| {
            me.cheb(a) <= ALTAR_ON_WAY_MAX && me.cheb(a) + a.cheb(g) - me.cheb(g) <= ALTAR_DETOUR
        })
    }

    /// Is `p` on the way to where `intent` takes us (camp, altar or home)? At most
    /// HELP_DETOUR extra tiles and HELP_ON_WAY_MAX tiles from here.
    fn on_the_way(&self, me: Pos, p: Pos, intent: Intent) -> bool {
        let goal = match intent {
            Intent::ClearCamp(g) | Intent::Capture(g) | Intent::Approach(g) => {
                self.world.groups.get(usize::from(g)).map(|gr| gr.center)
            }
            Intent::Tribute => self.world.altars.iter().copied().min_by_key(|a| a.cheb(me)),
            Intent::Retreat | Intent::WalkHome | Intent::Shop => Some(self.world.spawn_hint),
            _ => None,
        };
        goal.is_some_and(|g| {
            me.cheb(p) <= HELP_ON_WAY_MAX && me.cheb(p) + p.cheb(g) - me.cheb(g) <= HELP_DETOUR
        })
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
            combo: false,
            stun_lock: false,
            gold_order: 0,
            lock_contested_only: false,
            los_retreat: false,
            tp_crystal: false,
            weak_first: false,
            dash_in: false,
            field_shop: false,
            keep_buying: self.theta.knob2(knob2::KEEP_BUYING) > 0,
            recall_safe: false,
            recall_ticks: 5,
            tavi_respec: false,
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

    /// knob2 ENDGAME: late (round ENDGAME_START+) or when either team is close to winning.
    /// Behind = the enemy leads and is near the win (or far ahead late): hunt. Ahead: bank.
    fn endgame(&self, ctx: &Context) -> Endgame {
        if self.theta.knob2(knob2::ENDGAME) == 0 {
            return Endgame::Off;
        }
        let (us, them, win) = (ctx.ally_tributed, ctx.enemy_tributed, ctx.winning_gems.max(1));
        let close = |x: u32| x * 10 >= win * 8;
        if ctx.round < ENDGAME_START && !close(us) && !close(them) {
            return Endgame::Off;
        }
        if them > us && (close(them) || them >= us + HUNT_BEHIND_GEMS) {
            Endgame::Behind
        } else if us >= them {
            Endgame::Ahead
        } else {
            Endgame::Off
        }
    }

    /// Knobs with per-champion adjustments (PASSIVE_ROLES: Prism carries bigger bags).
    fn role_knobs(&self) -> Knobs {
        use crate::learned as l;
        let mut k = self.knobs;
        let scale = |v: u8, w: f32| (f32::from(v) * w).round().clamp(0.0, 255.0) as u8;
        k.fight_margin_pct = scale(k.fight_margin_pct, l::w(l::FIGHT_MARGIN));
        k.retreat_hp_pct = scale(k.retreat_hp_pct, l::w(l::RETREAT_HP)).min(95);
        k.aggression = scale(k.aggression, l::w(l::AGGRESSION));
        k.tribute_min = scale(k.tribute_min, l::w(l::TRIBUTE_MIN));
        k.tribute_max = scale(k.tribute_max, l::w(l::TRIBUTE_MAX)).max(k.tribute_min);
        k.recall_gold = (k.recall_gold as f32 * l::w(l::RECALL_GOLD)).round().max(30.0) as u32;
        if self.theta.knob(knob::PASSIVE_ROLES) > 0 && self.kind == ChampionKind::Prism {
            k.tribute_min = (u32::from(k.tribute_min) * 3 / 2).min(u32::from(k.tribute_max)) as u8;
        }
        k
    }

    /// The team channel as the brain reads it. Under TEAM_RESPAWN bytes 1-2 carry respawn
    /// timers, so aggression / threshold are computed locally (a pure function of public state).
    fn team_view(&self, chan: &TeamChannel, round: u32) -> TeamChannel {
        let mut team = *chan;
        if self.theta.knob(knob::TEAM_RESPAWN) > 0 {
            (team.aggression, team.tribute_threshold) = policy::team_params(
                self.ledger.ally_tributed,
                self.ledger.enemy_tributed,
                round,
                cambc::constants().max_rounds,
                self.knobs.aggression,
                self.knobs.tribute_max,
            );
        }
        team
    }

    /// Expert for this turn: exploration (RL data builds), the learned router (offline RL),
    /// else the rules.
    fn pick_expert(&self, mode: Mode, ctx: &Context, me: &Champion) -> Expert {
        let epsilon = f32::from(self.theta.knob2(knob2::EXPLORE)) / 100.0;
        if epsilon > 0.0 && !matches!(mode, Mode::Override(_)) && self.turns.is_multiple_of(4) {
            // Deterministic hash: the engine has no RNG and runs must stay reproducible.
            let h = (self.turns.wrapping_mul(2_654_435_761)
                ^ u32::from(self.slot).wrapping_mul(40_503))
            .rotate_left(13)
            .wrapping_mul(2_246_822_519);
            if (h % 10_000) as f32 / 10_000.0 < epsilon {
                let options: Vec<Expert> = (0..N_EXPERTS)
                    .filter_map(Expert::from_index)
                    .filter(|&e| feasible(e, ctx))
                    .collect();
                if !options.is_empty() {
                    return options[(h / 10_000) as usize % options.len()];
                }
            }
        }
        if self.theta.knob2(knob2::ROUTER) > 0
            && !matches!(mode, Mode::Override(_))
            && fuel() > LEARNED_MIN_FUEL
        {
            if let Some(e) = self.learned_expert(ctx, me) {
                return e;
            }
        }
        router::select(mode, ctx, None)
    }

    /// Arm 1 (knob2 ROUTER with a [`rl::RL_OUTPUTS`]-wide network): expert + fight-target
    /// mode, sampled while training (knob2 RL_SEED), argmax otherwise. Logged as an `R` line.
    fn rl_policy(
        &self,
        mode: Mode,
        ctx: &Context,
        me: &Champion,
        round: u32,
    ) -> Option<(Expert, TargetMode)> {
        if self.theta.knob2(knob2::ROUTER) == 0
            || matches!(mode, Mode::Override(_))
            || fuel() <= LEARNED_MIN_FUEL
        {
            return None;
        }
        LEARNED.with(|cell| {
            let mlp = cell.get_or_init(|| Mlp::parse_for_schema(ROUTER_BLOB, SCHEMA_HASH).ok()).as_ref()?;
            if mlp.output_len() != rl::RL_OUTPUTS {
                return None;
            }
            let raw = self.raw_obs(ctx, me).ok()?;
            let features = extract(&raw, GroupMask::ALL);
            let mut logits = [0i32; rl::RL_OUTPUTS];
            mlp.forward(features.values(), Some(&self.theta.film[..]), &mut logits);
            let shift = mlp.layers.last().map_or(0, |l| l.shift);
            let seed = self.theta.knob2(knob2::RL_SEED);
            let u = |head: u8| (seed > 0).then(|| rl::uniform(self.turns, self.slot, seed, head));
            let emask: [bool; 9] = core::array::from_fn(|i| Expert::from_index(i).is_some_and(|e| feasible(e, ctx)));
            let fight = ctx.enemy.is_some();
            let tmask = [true, fight, fight, fight];
            let (ei, lp1) = rl::pick(&logits[rl::EXPERT_HEAD], &emask, shift, u(0))?;
            let (ti, lp2) = rl::pick(&logits[rl::TARGET_HEAD], &tmask, shift, u(1))?;
            cambc::debug_block! {
                telemetry::rl(round, champ_name(self.kind), 1, &features.to_hex(), &[ei, ti], lp1 + lp2, &[bits(&emask), bits(&tmask)]);
            }
            Some((Expert::from_index(ei)?, TargetMode::from_index(ti)))
        })
    }

    /// The enemy a target mode means (None = keep the rules' choice).
    fn rl_target(&self, mode: TargetMode, vis: &Visible) -> Option<Pos> {
        let c = vis.enemies.iter().filter(|e| !e.on_spawn && e.dist <= 8);
        match mode {
            TargetMode::Rule => None,
            TargetMode::Nearest => c.min_by_key(|e| e.dist).map(|e| e.pos),
            TargetMode::Weakest => c.min_by(|a, b| a.hp.total_cmp(&b.hp)).map(|e| e.pos),
            TargetMode::Carrier => c.max_by_key(|e| e.gems).map(|e| e.pos),
        }
    }

    /// Arm 2 (knob2 MICRO_NET): on fight turns the micro network picks (step, attack target,
    /// cast). Its input is the 109 features plus, per step (stay + 8 directions), whether the
    /// tile is free and how many enemies can hit it.
    fn micro_policy(
        &self,
        ctx: &Context,
        vis: &Visible,
        me: &Champion,
        my_pos: Pos,
        intent: Intent,
        round: u32,
    ) -> Option<(Option<Pos>, Option<Pos>, bool)> {
        if self.theta.knob2(knob2::MICRO_NET) == 0
            || !matches!(intent, Intent::Engage(_) | Intent::Kite(_) | Intent::Retreat)
            || !vis.enemies.iter().any(|e| !e.on_spawn && e.dist <= 6)
            || fuel() <= LEARNED_MIN_FUEL
        {
            return None;
        }
        MICRO.with(|cell| {
            let mlp = cell
                .get_or_init(|| {
                    Mlp::parse_for_schema(MICRO_BLOB, rl::MICRO_SCHEMA)
                        .ok()
                        .filter(|m| m.input_len() == rl::MICRO_INPUTS && m.output_len() == rl::MICRO_OUTPUTS)
                })
                .as_ref()?;
            let raw = self.raw_obs(ctx, me).ok()?;
            let mut x: Vec<i8> = extract(&raw, GroupMask::ALL).values().to_vec();
            x.resize(rl::MICRO_INPUTS - 18, 0);
            let occupied = occupied_tiles(vis, my_pos);
            let steps: [Pos; 9] = core::array::from_fn(|i| if i == 0 { my_pos } else { my_pos + crate::bc_core::geom::DIRS8[i - 1] });
            let free: [bool; 9] =
                core::array::from_fn(|i| i == 0 || (self.world.terrain.walkable(steps[i]) && !occupied.contains(&steps[i])));
            for &f in &free {
                x.push(if f { 127 } else { 0 });
            }
            for &p in &steps {
                let n = vis.enemies.iter().filter(|e| !e.on_spawn && e.pos.cheb(p) <= e.range).count();
                x.push((n * 42).min(127) as i8);
            }
            let mut logits = [0i32; rl::MICRO_OUTPUTS];
            mlp.forward(&x, None, &mut logits);
            let shift = mlp.layers.last().map_or(0, |l| l.shift);
            let seed = self.theta.knob2(knob2::RL_SEED);
            let u = |head: u8| (seed > 0).then(|| rl::uniform(self.turns, self.slot, seed, head));
            let fight = vis.enemies.iter().any(|e| !e.on_spawn && e.dist <= 8);
            let tmask = [true, fight, fight, fight];
            let (si, lp1) = rl::pick(&logits[rl::STEP_HEAD], &free, shift, u(2))?;
            let (ti, lp2) = rl::pick(&logits[rl::MICRO_TARGET_HEAD], &tmask, shift, u(3))?;
            let (ci, lp3) = rl::pick(&logits[rl::CAST_HEAD], &[true, true], shift, u(4))?;
            cambc::debug_block! {
                let hex: String = x.iter().map(|b| format!("{:02x}", *b as u8)).collect();
                telemetry::rl(round, champ_name(self.kind), 2, &hex, &[si, ti, ci], lp1 + lp2 + lp3, &[bits(&free), bits(&tmask), 3]);
            }
            let step = (si > 0).then_some(steps[si]);
            let target = self.rl_target(TargetMode::from_index(ti), vis);
            Some((step, target, ci == 1))
        })
    }

    fn learned_expert(&self, ctx: &Context, me: &Champion) -> Option<Expert> {
        LEARNED.with(|cell| {
            let mlp = cell
                .get_or_init(|| Mlp::parse_for_schema(ROUTER_BLOB, SCHEMA_HASH).ok())
                .as_ref()?;
            let raw = self.raw_obs(ctx, me).ok()?;
            let features = extract(&raw, GroupMask::ALL);
            let mut logits = [0i32; N_EXPERTS];
            if mlp.forward(features.values(), Some(&self.theta.film[..]), &mut logits) != N_EXPERTS
            {
                return None;
            }
            let mut order: [usize; N_EXPERTS] = core::array::from_fn(|i| i);
            order.sort_by_key(|&i| core::cmp::Reverse(logits[i]));
            if logits[order[0]] - logits[order[1]] < LEARNED_MARGIN {
                return None;
            }
            Expert::from_index(order[0]).filter(|&e| feasible(e, ctx))
        })
    }

    /// FIGHT . psi (learned.rs): log-multiplier on the fight advantage.
    fn fight_bias(&self, ctx: &Context, vis: &Visible) -> f32 {
        let fight_w = crate::learned::fight();
        if fight_w.iter().all(|&w| w == 0.0) {
            return 0.0;
        }
        let Some(e) = ctx.enemy else { return 0.0 };
        let psi = [
            ctx.hp_frac(),
            e.hp / e.max_hp.max(1.0),
            vis.allies.iter().filter(|a| a.dist <= 4).count() as f32,
            vis.enemies.iter().filter(|o| o.dist <= 5 && !o.on_spawn).count() as f32,
            f32::from(u8::from(ctx.ranged)),
            f32::from(u8::from(e.ranged)),
            ctx.gems as f32 / 100.0,
            ctx.dist_spawn.map_or(1.0, |d| f32::from(d) / 20.0),
            1.0,
        ];
        fight_w.iter().zip(psi).map(|(w, x)| w * x).sum::<f32>().clamp(-3.0, 3.0)
    }

    /// exp(TARGET . phi) (learned.rs): learned correction to a camp group's score. `w` is
    /// [`crate::learned::target`] or None when it is all zero (no bias).
    #[allow(clippy::too_many_arguments)]
    fn target_bias(
        w: Option<&[f32]>,
        g: &crate::world::GroupStatic,
        dist: f32,
        clear: f32,
        eta: f32,
        others: usize,
        last_hp: f32,
        fixed: [f32; 3],
    ) -> f32 {
        let Some(w) = w else { return 1.0 };
        let damage = g.total_strength * clear / last_hp.max(1.0);
        let phi = [
            g.gems as f32 / 100.0,
            g.gold as f32 / 100.0,
            dist / 20.0,
            clear / 50.0,
            eta / 50.0,
            others as f32,
            damage.min(5.0),
            fixed[0],
            fixed[1],
            fixed[2],
        ];
        w.iter().zip(phi).map(|(w, x)| w * x).sum::<f32>().clamp(-4.0, 4.0).exp()
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

    /// Our damage per round: live (items included) once known, else the kit's base stats.
    fn kit_dps(&self) -> f32 {
        let base = sdk::dps(self.kit.base_strength, self.kit.base_focus, &self.kit);
        self.cur_dps.max(base).max(1.0)
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
            let max_hp = e.max_health().unwrap_or(hp);
            let (strength, focus) =
                (e.strength().unwrap_or(kit.base_strength), e.focus().unwrap_or(kit.base_focus));
            let dps = if stun_threat {
                sdk::dps_burst(strength, focus, &kit)
            } else {
                sdk::dps(strength, focus, &kit)
            };
            // Heart of Ruin hits harder the lower its health (up to +50%).
            let dps = dps
                * crate::bc_core::champion::missing_health_bonus(
                    kind,
                    hp / max_hp.max(1.0),
                );
            let view = EnemyView {
                pos: p,
                dist: my_pos.cheb(p),
                hp,
                max_hp,
                dps,
                gems: e.gem_count().unwrap_or(0),
                ranged: kit.attack == AttackType::Ranged,
                range: kit.attack_range,
                on_spawn: self.world.terrain.enemy_spawn.get(p),
                tributing: self.theta.knob(knob::DENY_TRIBUTE) > 0
                    && e.state().is_ok_and(|s| s.contains(ChampionState::SUBMITTING_GEMS)),
                recalling: self.theta.knob2(knob2::PUNISH_RECALL) > 0
                    && e.state().is_ok_and(|s| s.contains(ChampionState::RECALLING)),
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
                kind,
                handle: a,
            });
        }

        let mut seen_groups = Vec::new();
        for c in cambc::vision::camps()? {
            let spawn = pos(c.spawn_position());
            let Ok(state) = c.state() else { continue };
            if state.is_dead() {
                continue;
            }
            let group = self.world.group_at(spawn);
            if let Some(g) = group {
                if !seen_groups.contains(&g) {
                    seen_groups.push(g);
                    self.belief.observe_group(g, GroupObs::Alive, round);
                }
            }
            // Far camps (they never chase more than 3 tiles from their spawn): alive is all we
            // need. Their position/health/strength queries cost ~450 fuel each, and maps like
            // yellow-brick-road show 40+ camps (observe took 60k+ and turns ran out of fuel).
            if spawn.cheb(my_pos) > FAR_CAMP {
                continue;
            }
            let p = c.position().map(pos).unwrap_or(spawn);
            let awake = state.is_awake();
            // Sleeping camps don't hit: their kind's base strength is enough (no host query).
            let strength = if awake {
                c.strength().unwrap_or(0.0)
            } else {
                crate::bc_draft::CampKind::from_engine(c.kind() as i32).map_or(0.0, |k| k.strength())
            };
            vis.camps.push(CampView {
                pos: p,
                dist: my_pos.cheb(p),
                hp: c.health().unwrap_or(0.0),
                awake,
                strength,
                group,
                handle: c,
            });
        }

        // Groups in sight with no living camp: dead (reward) or respawning.
        // The target is always checked; others when close, at most 2 per turn.
        let mut checked = 0;
        for (i, g) in self.world.groups.iter().enumerate() {
            let is_target = self.target == Some(i);
            // Cheap bound first: the centre's distance minus the group's radius.
            if !is_target && g.center.cheb(my_pos) - g.radius > VISION {
                continue;
            }
            let near = g.tiles.iter().map(|t| t.cheb(my_pos)).min().unwrap_or(i32::MAX);
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
                    let fresh =
                        self.belief.group(i).is_none_or(|b| b.state != GroupState::Respawning);
                    if fresh && t > 0 {
                        self.saw_down = Some((i, round + t as u32));
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
        if let Some(g) = self.opening_target(my_pos, round, chan) {
            self.target = Some(g);
            return;
        }
        // Short on fuel: scoring every group costs ~1.5k fuel each (40k+ on dense maps like
        // yellow-brick-road), which ran turns out of fuel and froze the champions. Keep a
        // still-valid target instead, for at most TARGET_KEEP_MAX turns in a row.
        let start = fuel();
        if let Some(t) = self.target {
            let valid = t < self.world.groups.len()
                && self.belief.expected_state(t, round) != GroupState::Respawning;
            if valid
                && self.target_kept < TARGET_KEEP_MAX
                && start < self.target_cost + TARGET_REST_FUEL
            {
                self.target_kept += 1;
                return;
            }
        }
        self.target_kept = 0;
        self.score_targets(my_pos, round, chan, vis);
        self.target_cost = start.saturating_sub(fuel());
    }

    /// Cluster neighbours of group `i`: groups within CLUSTER_PATH walking tiles of its centre,
    /// weighted 1 (next to it) down to ~0 (CLUSTER_PATH away). Exact (cacheable) once every
    /// neighbour's field is ready; a straight-line estimate before that.
    fn cluster_links(&self, i: usize) -> (Vec<(u8, f32)>, bool) {
        let g = &self.world.groups[i];
        let mut exact = true;
        let mut out = Vec::new();
        for &j in &g.near {
            let j = usize::from(j);
            if self.world.groups[j].has_monolith {
                continue;
            }
            exact &= self.world.fields.is_ready(Goal::Group(j as u8));
            if let Some(d) = self.group_distance(j, g.center) {
                if d <= CLUSTER_PATH {
                    out.push((j as u8, 1.0 - d as f32 / (CLUSTER_PATH + 1) as f32));
                }
            }
        }
        (out, exact)
    }

    /// knob2 CLUSTER: per group, the value of the live groups around it (gems + gold / 3,
    /// weighted by nearness), x knob / 10. Empty when the knob is off.
    fn cluster_bonus(&mut self, round: u32) -> Vec<f32> {
        let k = self.theta.knob2(knob2::CLUSTER);
        if k == 0 {
            return Vec::new();
        }
        let n = self.world.groups.len();
        if self.cluster.len() != n {
            self.cluster = vec![None; n];
        }
        // Each group's links are computed once; estimated ones are redone with walking distances
        // once the fields exist, at most one group per call (each costs a few field reads).
        // Round-robin over the estimated ones so each gets its turn.
        let mut retry = true;
        for k in 0..n {
            let i = (k + round as usize) % n;
            let redo = match &self.cluster[i] {
                None => true,
                Some((_, exact)) => !exact && retry,
            };
            if redo {
                if self.cluster[i].is_some() {
                    retry = false;
                }
                self.cluster[i] = Some(self.cluster_links(i));
            }
        }
        let scale = f32::from(k) / 10.0;
        let mut bonus = vec![0.0; n];
        for (i, b) in bonus.iter_mut().enumerate() {
            let Some((links, _)) = &self.cluster[i] else { continue };
            for &(j, w) in links {
                let j = usize::from(j);
                if self.belief.expected_state(j, round) == GroupState::Respawning {
                    continue;
                }
                let gj = &self.world.groups[j];
                *b += w * (gj.gems as f32 + gj.gold as f32 / 3.0);
            }
            *b *= scale;
        }
        bonus
    }

    fn score_targets(&mut self, my_pos: Pos, round: u32, chan: &TeamChannel, vis: &Visible) {
        let cluster = self.cluster_bonus(round);
        // Live damage (items included): base stats alone underrate a geared champion, making
        // every camp look slower to clear than it is.
        let my_dps = sdk::dps(self.kit.base_strength, self.kit.base_focus, &self.kit)
            .max(self.cur_dps)
            .max(10.0);
        let live = self.theta.knob(knob::MONOLITH_LIVE) > 0;
        let team_power =
            if live { self.cur_dps.max(my_dps) } else { my_dps } * self.team_size as f32;
        let value_profile = self.theta.knob(knob::VALUE_PROFILE);
        let plan_bits = self.theta.knob2(knob2::DRAFT_PLAN);
        let enemy_close = vis.enemies.iter().any(|e| !e.on_spawn && e.dist <= PLAN_ENEMY_CLOSE);
        // VALUE_PROFILE 4/5: near camps first in the opening (gold now beats gems later).
        let opening = matches!(value_profile, 4 | 5) && round < OPENING_ROUNDS;
        // Loop invariants, read once (scoring every group used ~1.5k fuel each).
        let slot = self.slot as usize;
        let emblems = self.emblems();
        let passive_aware = self.theta.knob2(knob2::PASSIVE_AWARE) > 0;
        let passive_roles = self.theta.knob(knob::PASSIVE_ROLES) > 0;
        let camp_feasible = self.theta.knob(knob::CAMP_FEASIBLE);
        let squad_join = self.theta.knob(knob::SQUAD_JOIN_X10);
        let buddy_claim = chan.claims.get(slot ^ 1).copied().flatten();
        let solo_cap = usize::from(self.theta.knob2(knob2::SOLO_CAP));
        let ease_first = self.theta.knob2(knob2::EASE_FIRST) > 0;
        let ally_near = vis.allies.iter().any(|a| a.dist <= 6);
        let w_gem = crate::learned::w(crate::learned::GEM);
        let w_gold = crate::learned::w(crate::learned::GOLD);
        let w_reward = crate::learned::w(crate::learned::REWARD);
        let w_shared_small = crate::learned::w(crate::learned::SHARED_SMALL);
        let w_strength = crate::learned::w(crate::learned::STRENGTH_DISCOUNT);
        let target_w = Some(crate::learned::target()).filter(|w| w.iter().any(|&x| x != 0.0));
        let bias_fixed =
            [self.ledger.gems as f32 / 100.0, emblems as f32 / 5.0, round as f32 / 1000.0];
        let early_gold = self.theta.knob2(knob2::EARLY_GOLD) > 0;
        let early = round < EARLY_RAMP_ROUNDS || emblems < 2;
        // Remote farmer: the last slot of a 4+ team goes further for quieter groups.
        let remote = self.theta.knob(knob::REMOTE_FARMER) > 0
            && self.team_size >= 4
            && slot + 1 == self.team_size;
        let dist_w = if remote {
            0.4
        } else if opening {
            OPENING_DIST_WEIGHT * crate::learned::w(crate::learned::OPENING_DIST)
        } else {
            1.0
        } * crate::learned::w(crate::learned::DIST);
        let w_clear = crate::learned::w(crate::learned::CLEAR);
        let w_hysteresis = crate::learned::w(crate::learned::HYSTERESIS);
        let mut best: Option<(f32, usize)> = None;
        // Debug runs: every TARGET_DIAG_EVERY rounds, a `G` line with each group's score and
        // what cut it (debug_block only: free locally, compiled out in release).
        let diag = round % TARGET_DIAG_EVERY == TARGET_DIAG_EVERY / 2;
        let mut rows: Vec<String> = Vec::new();
        macro_rules! why {
            ($($t:tt)*) => { cambc::debug_block! { if diag { $($t)* } } };
        }
        let mut f = String::new();
        let (mut t0, mut t1) = (0u64, 0u64);
        why!(t0 = fuel());
        // Short on fuel (dense maps: ~1.5k per group): nearest groups first, and stop once a
        // candidate exists and the rest of the turn's fuel is reached. With enough fuel every
        // group is scored in index order as before.
        let n_groups = self.world.groups.len();
        // Per-group cost from the last full scoring (it grew to ~6k a group; 1.5k was measured
        // long ago and made `short` miss: one scoring then took 60k).
        let per_group = (self.target_cost / n_groups.max(1) as u64).max(1_500);
        let short = fuel() < n_groups as u64 * per_group + TARGET_REST_FUEL;
        let mut order: Vec<usize> = (0..n_groups).collect();
        if short {
            order.sort_by_key(|&i| self.world.groups[i].center.cheb(my_pos));
        }
        for i in order {
            let g = &self.world.groups[i];
            if best.is_some() && fuel() < TARGET_REST_FUEL {
                break;
            }
            why!(f.clear());
            if g.has_monolith && live {
                // The monolith hits the nearest champion for its strength every round: the
                // team's pooled health must outlast the time its pooled damage needs.
                let n = self.team_size as f32;
                let survive = self.cur_max_hp.max(1.0) * n / g.total_strength.max(1.0);
                let kill = g.total_hp / team_power.max(1.0);
                if kill > survive * MONOLITH_SLACK {
                    why!(rows.push(format!("[{i},\"skip:monolith_survive\"]")));
                    continue;
                }
            } else if g.has_monolith
                && team_power * 10.0 < f32::from(self.knobs.monolith_min_power) * 100.0
            {
                why!(rows.push(format!("[{i},\"skip:monolith_power\"]")));
                continue;
            }
            let state = self.belief.expected_state(i, round);
            let Some(dist) = self.group_distance(i, my_pos) else {
                why!(rows.push(format!("[{i},\"skip:unreachable\"]")));
                continue;
            };
            let dist = dist as f32;
            let eta = self.belief.rounds_until_up(i, round) as f32;
            if state == GroupState::Respawning && eta > dist + 15.0 {
                why!(rows.push(format!("[{i},\"skip:respawning\"]")));
                continue;
            }
            let mut value = match value_profile {
                1 => g.gems as f32 + g.gold as f32 / 10.0,
                4 => g.gems as f32 * w_gem + g.gold as f32 * 0.6 * w_gold,
                5 => g.gems as f32 + g.gold as f32,
                2 if self.slot < 2 => g.gems as f32 * 0.3 + g.gold as f32,
                2 => g.gems as f32 + g.gold as f32 * 0.15,
                _ => g.gems as f32 + g.gold as f32 / 3.0,
            };
            // Passives: Vesper's gold bonus applies to her capture share (gear herself first),
            // Prism's Focus grows with the gems she carries (she collects them).
            if passive_aware {
                // Gold-bonus passives: gold is worth more until geared. Splash attacks clear
                // multi-camp groups faster.
                if self.passive == PASSIVE_BONUS_GOLD && emblems < 3 {
                    value += g.gold as f32 * 0.3;
                }
                if self.passive == PASSIVE_SPLASH {
                    value *= 1.0 + 0.15 * g.tiles.len().saturating_sub(1) as f32;
                }
            }
            if passive_roles {
                match self.kind {
                    ChampionKind::VesperVale if emblems < VESPER_GOLD_EMBLEMS => {
                        value = g.gems as f32 * 0.25 + g.gold as f32 * 1.2;
                    }
                    ChampionKind::Prism => value = g.gems as f32 * 1.4 + g.gold as f32 / 3.0,
                    _ => {}
                }
            }
            // knob2 DRAFT_PLAN bit 1: frontier groups (walk-closer to the enemy spawn, from the
            // draft's match plan) are worth less while enemy champions are close.
            if plan_bits & 1 != 0
                && enemy_close
                && self.theta.plan.groups.get(i).is_some_and(|pg| pg.is(crate::bc_core::plan::class::FRONTIER))
            {
                value *= PLAN_FRONTIER_VALUE;
                why!(f.push_str("\"frontier\":1,"));
            }
            // CAMP_FEASIBLE bit 4: a group whose camps are all in sight costs its live HP
            // (an ally may have half-killed it), not its full HP.
            let group_hp = if camp_feasible & 4 != 0 {
                let (mut n, mut hp) = (0usize, 0.0f32);
                for c in vis.camps.iter().filter(|c| c.group == Some(i)) {
                    n += 1;
                    hp += c.hp.max(0.0);
                }
                if n > 0 && n >= g.tiles.len() {
                    hp
                } else {
                    g.total_hp
                }
            } else {
                g.total_hp
            };
            let clear = match state {
                GroupState::RewardAvailable => {
                    value *= 1.5 * w_reward;
                    0.0
                }
                // A monolith is cleared by the whole team, not one champion.
                _ if g.has_monolith && live => g.total_hp / team_power.max(1.0),
                _ => group_hp / my_dps,
            };
            // A capture needs one champion; an ally already on the tile has it.
            if state == GroupState::RewardAvailable
                && vis.allies.iter().any(|a| self.world.group_at(a.pos) == Some(i))
            {
                why!(rows.push(format!("[{i},\"skip:ally_capturing\"]")));
                continue;
            }
            if let Some(&b) = cluster.get(i) {
                value += b;
                why!(f.push_str(&format!("\"cluster\":{b:.1},")));
            }
            why!(f.push_str(&format!("\"v0\":{value:.1},")));
            let pi = self.priority_index(i);
            let others = pi.map_or(0, |pi| chan.claimants(pi, slot));
            if pi.is_none() {
                why!(f.push_str("\"no_claims\":1,"));
            }
            if let Some(pi) = pi {
                // Squads: slot pairs (0,1), (2,3) join each other's group (2v1 fights, faster clears).
                let join = squad_join;
                let buddy = buddy_claim == Some(pi);
                if join > 0 && buddy {
                    value *= f32::from(join) / 10.0;
                } else if others > 0 {
                    // Big groups go faster together; small ones are wasted on two.
                    let k = if g.total_hp >= 3000.0 { 1.1 } else { 0.35 * w_shared_small };
                    value *= k;
                    why!(f.push_str(&format!("\"shared\":{k:.2},")));
                }
            }
            // Stronger camps hurt: discount by expected damage taken.
            let k = 1.0 + g.max_strength / 100.0 * w_strength;
            value /= k;
            why!(f.push_str(&format!("\"strength\":{:.2},", 1.0 / k)));
            // Can the champions on it clear it before the tank dies? Gear up on others first.
            if camp_feasible & 1 != 0 && state != GroupState::RewardAvailable {
                let helpers = 1 + others;
                let survive = self.last_hp.max(1.0) / (g.total_strength * CAMP_DMG_FLOOR).max(1.0);
                let kill = group_hp / (self.cur_dps.max(my_dps) * helpers as f32).max(1.0);
                if kill > survive * CAMP_FEASIBLE_SLACK {
                    value *= CAMP_INFEASIBLE_VALUE;
                    why!(f.push_str(&format!("\"infeasible\":{CAMP_INFEASIBLE_VALUE},")));
                }
            }
            let monsters = g.tiles.len();
            // knob2 SOLO_CAP: a big group alone only once we're geared enough to outlast it.
            if solo_cap > 0
                && monsters >= solo_cap
                && !ally_near
                && state != GroupState::RewardAvailable
                && others == 0
            {
                let survive = self.last_hp.max(1.0) / (g.total_strength * CAMP_DMG_FLOOR).max(1.0);
                let kill = group_hp / self.cur_dps.max(my_dps).max(1.0);
                if kill * SOLO_GEARED_MARGIN > survive {
                    value *= CAMP_INFEASIBLE_VALUE;
                    why!(f.push_str(&format!("\"solo_cap\":{CAMP_INFEASIBLE_VALUE},")));
                }
            }
            // knob2 EASE_FIRST: alone, fewer monsters first (two-monster groups before packs).
            if ease_first && !ally_near && state != GroupState::RewardAvailable {
                let k = 1.0 + EASE_PER_MONSTER * monsters.saturating_sub(1) as f32;
                value /= k;
                why!(f.push_str(&format!("\"ease\":{:.2},", 1.0 / k)));
            }
            // knob2 EARLY_GOLD: before 2 emblems, gold from quick groups buys the first upgrades.
            if early_gold && emblems < 2 && clear <= EARLY_RAMP_KNEE {
                value += g.gold as f32 * EARLY_GOLD_BONUS;
            }
            // CAMP_FEASIBLE bit 2, early ramp: before EARLY_RAMP_ROUNDS or our 2nd emblem, quick
            // camps first; value falls with clear time past EARLY_RAMP_KNEE turns, so guarded
            // groups (Golem, Elk packs) wait until we've geared up.
            if camp_feasible & 2 != 0 && early {
                let k = 1.0 + (clear - EARLY_RAMP_KNEE).max(0.0) / EARLY_RAMP_KNEE;
                value /= k;
                why!(f.push_str(&format!("\"early_ramp\":{:.2},", 1.0 / k)));
            }
            let mut score = value / (dist * dist_w + clear * w_clear + (eta - dist).max(0.0) + 5.0);
            let bias =
                Self::target_bias(target_w, g, dist, clear, eta, others, self.last_hp, bias_fixed);
            score *= bias;
            if self.target == Some(i) {
                score *= 1.3 * w_hysteresis;
            }
            why!(rows.push(format!(
                "[{i},{{{f}\"v\":{value:.1},\"dist\":{dist:.0},\"clear\":{clear:.0},\"wait\":{:.0},\"bias\":{bias:.2},\"score\":{score:.3}}}]",
                (eta - dist).max(0.0)
            )));
            if best.is_none_or(|(b, _)| score > b) {
                best = Some((score, i));
            }
        }
        why!(t1 = fuel());
        let pick = best.map_or(-1, |(_, i)| i as i64);
        if self.theta.knob(knob::EN_ROUTE) > 0 && fuel() > TARGET_REST_FUEL {
            if let Some((_, far)) = best {
                let quiet = vis.enemies.iter().all(|e| e.on_spawn || e.dist > 6);
                if let Some(stop) = self.en_route_stop(far, my_pos, round, chan, quiet) {
                    best = Some((0.0, stop));
                }
            }
        }
        why!(println!(
            "G {{\"r\":{round},\"c\":\"{}\",\"pick\":{pick},\"route\":{},\"fuel_loop\":{},\"fuel_route\":{},\"groups\":[{}]}}",
            champ_name(self.kind),
            best.map_or(-1, |(_, i)| i as i64),
            t0.saturating_sub(t1),
            t1.saturating_sub(fuel()),
            rows.join(",")
        ));
        self.target = best.map(|(_, i)| i);
    }

    /// Policy-pack opening: the next group of this champion's opening route while it is up,
    /// reachable and not claimed by an ally (pointer moves on once the group goes down).
    fn opening_target(&mut self, my_pos: Pos, round: u32, chan: &TeamChannel) -> Option<usize> {
        if round > PACK_OPENING_ROUNDS {
            return None;
        }
        while let Some(&g) = self.opening.get(self.opening_idx) {
            let g = usize::from(g);
            let gone = g >= self.world.groups.len()
                || self.group_distance(g, my_pos).is_none()
                || matches!(self.belief.expected_state(g, round), GroupState::Respawning);
            let taken =
                self.priority_index(g).is_some_and(|pi| chan.claimants(pi, self.slot as usize) > 0)
                    && self.target != Some(g);
            if gone || taken {
                self.opening_idx += 1;
                continue;
            }
            return Some(g);
        }
        None
    }

    /// EN_ROUTE: a small camp just off the way to `far` that we can clear and survive alone.
    fn en_route_stop(
        &self,
        far: usize,
        my_pos: Pos,
        round: u32,
        chan: &TeamChannel,
        quiet: bool,
    ) -> Option<usize> {
        let to_far = self.group_distance(far, my_pos)? as i32;
        // EN_ROUTE 2 (wide): limits for the engine 1.0.0 economy, plus free money nearby.
        let wide = self.theta.knob(knob::EN_ROUTE) >= 2;
        let (min_dist, max_clear) = if wide {
            (EN_ROUTE_WIDE_MIN_DIST, EN_ROUTE_WIDE_MAX_CLEAR)
        } else {
            (EN_ROUTE_MIN_DIST, EN_ROUTE_MAX_CLEAR)
        };
        if to_far <= min_dist && !(wide && quiet) {
            return None;
        }
        let dps = self.cur_dps.max(10.0);
        let mut pick: Option<(f32, usize)> = None;
        for (i, g) in self.world.groups.iter().enumerate() {
            if i == far || g.has_monolith || g.gold + g.gems == 0 {
                continue;
            }
            let state = self.belief.expected_state(i, round);
            // Free money: a reward waiting to be captured, or a camp that can't hurt us.
            let free = wide
                && quiet
                && (state == GroupState::RewardAvailable
                    || (g.total_strength <= 0.0
                        && matches!(state, GroupState::Alive | GroupState::Unknown)));
            if !free
                && (to_far <= min_dist || !matches!(state, GroupState::Alive | GroupState::Unknown))
            {
                continue;
            }
            if self.priority_index(i).is_some_and(|pi| chan.claimants(pi, self.slot as usize) > 0) {
                continue;
            }
            let (Some(d1), Some(d2)) =
                (self.group_distance(i, my_pos), self.group_distance(far, g.center))
            else {
                continue;
            };
            let detour = d1 as i32 + d2 as i32 - to_far;
            let clear = if state == GroupState::RewardAvailable { 0.0 } else { g.total_hp / dps };
            if free {
                if d1 as i32 > EN_ROUTE_FREE_RADIUS || detour > EN_ROUTE_FREE_RADIUS {
                    continue;
                }
            } else {
                let damage = g.total_strength * CAMP_DMG_FLOOR * clear;
                if detour as f32
                    > EN_ROUTE_DETOUR as f32 * crate::learned::w(crate::learned::EN_DETOUR)
                    || clear > max_clear * crate::learned::w(crate::learned::EN_CLEAR)
                    || damage > self.last_hp * 0.7
                {
                    continue;
                }
            }
            let cost = detour as f32 + clear;
            if pick.is_none_or(|(c, _)| cost < c) {
                pick = Some((cost, i));
            }
        }
        pick.map(|(_, i)| i)
    }

    /// TEAM_RESPAWN: read the team's last "group went down" and post ours.
    /// Bytes 1-2 (aggression / tribute_threshold fields) carry group + 1 and respawn round mod 256.
    fn share_respawns(&mut self, chan: &mut TeamChannel, round: u32) {
        let g = usize::from(chan.aggression);
        if g > 0 && g <= self.world.groups.len() {
            // Respawn rounds lie within [round - 100, round + 155]: unwrap the low byte there.
            let base = round.saturating_sub(100);
            let at = base + (u32::from(chan.tribute_threshold).wrapping_sub(base) & 0xFF);
            let i = g - 1;
            let known = self.belief.group(i).is_some_and(|b| b.last_seen == Some(round));
            if at > round && !known {
                self.belief.observe_group(i, GroupObs::Respawning((at - round) as i32), round);
            }
        }
        if let Some((i, at)) = self.saw_down.take() {
            chan.aggression = (i + 1).min(255) as u8;
            chan.tribute_threshold = (at & 0xFF) as u8;
        }
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
        // knob2 FOCUS: the team's called target, if it is in reach (it may have moved a tile).
        let focused = (self.theta.knob2(knob2::FOCUS) > 0)
            .then(|| chan.focus(round))
            .flatten()
            .and_then(|f| {
                let reach = self.kit.attack_range + 2;
                vis.enemies.iter().find(|e| !e.on_spawn && e.pos.cheb(f) <= 1 && e.dist <= reach)
            });
        let enemy = if let Some(e) = focused {
            Some(e)
        } else if self.theta.knob(knob::FIGHT_V6) & 1 != 0 {
            // Weakest enemy in reach, the team's called target first.
            let reach = self.kit.attack_range + 2;
            let called = chan.sighting(round).filter(|&(_, age)| age == 0).map(|(p, _)| p);
            vis.enemies
                .iter()
                .filter(|e| !e.on_spawn && !ignored(e.kind) && e.dist <= reach)
                .min_by(|a, b| {
                    (called != Some(a.pos))
                        .cmp(&(called != Some(b.pos)))
                        .then(a.hp.total_cmp(&b.hp))
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
            fight_bias: 0.0,
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
            team: self.team_view(chan, round),
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
        ChampionKind::HeartOfRuin => "HeartOfRuin",
    }
}

/// Learned router weights (BCW1 blob, ml/distill / harness/league_train.py offline RL).
static ROUTER_BLOB: &[u8] = include_bytes!("weights/router.bin");
/// Arm 2 network (ml/rl); empty until one is trained, which disables MICRO_NET.
static MICRO_BLOB: &[u8] = include_bytes!("weights/micro.bin");
/// Logit gap (int32 accumulator units) the learned router needs over its runner-up.
const LEARNED_MARGIN: i32 = 1_700;
/// Below this much fuel left, skip feature extraction + inference (rule router only).
const LEARNED_MIN_FUEL: u64 = 65_000;

thread_local! {
    static LEARNED: std::cell::OnceCell<Option<Mlp>> = const { std::cell::OnceCell::new() };
    static MICRO: std::cell::OnceCell<Option<Mlp>> = const { std::cell::OnceCell::new() };
}

/// An expert the situation allows (exploration and the learned router never pick nonsense).
/// A head's mask as bits (for the `R` log line).
fn bits(mask: &[bool]) -> u32 {
    mask.iter().enumerate().fold(0, |b, (i, &m)| b | u32::from(m) << i)
}

fn feasible(e: Expert, ctx: &Context) -> bool {
    match e {
        Expert::Fight => ctx.enemy.is_some() && ctx.gems < ctx.knobs.big_bag(),
        Expert::Kite => ctx.enemy.is_some(),
        Expert::Tribute => ctx.gems > 0,
        Expert::Capture => ctx.target.is_some_and(|t| t.state == GroupState::RewardAvailable),
        Expert::Scout | Expert::AltarLock => false,
        _ => true,
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
    if bits & 4 != 0 && !forced && ctx.gems < ctx.knobs.big_bag() {
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
    let team_dps = my_dps
        + vis.allies.iter().filter(|a| a.dist <= CAMP_ALLY_RADIUS).map(|a| a.dps).sum::<f32>();
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
                    return Some(Intent::Approach(g.map_or(0, |v| v as u8)));
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
