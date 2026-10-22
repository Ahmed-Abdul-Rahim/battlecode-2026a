//! nn/ teacher: observation, decision and the `N` log line (docs/architecture.md, loop 3).
//!
//! knob2 TEACHER = 1 logs the observation with the rules' own choice (imitation data);
//! TEACHER = 2 lets the network (weights/teacher.bin) pick the goal, the step, the attack
//! target and the cast. Layout constants are mirrored in nn/spec.py (SPEC must match).
//!
//! Heads:
//! - goal (15): 4 camp-group slots, tribute, shop, retreat, walk home, hold, engage enemy
//!   slot 0-2, kite enemy 0, step back from the nearest threat, go to the team's sighting;
//! - step (10): auto (the executor walks to the goal), stay, 8 directions;
//! - attack (4): none, enemy slot 0-2 (only with a manual step);
//! - cast (2): no / yes at the attack target (only with a manual step).

use super::Bot;
use crate::act::{occupied_tiles, Outcome, Visible};
use crate::bc_brain::nn::{self, Net};
use crate::bc_brain::rl;
use crate::bc_brain::{Context, Intent};
use crate::bc_core::belief::GroupState;
use crate::bc_core::geom::DIRS8;
use crate::bc_core::ledger::Family;
use crate::bc_core::memory::{knob2, TeamChannel};
use crate::bc_core::Pos;
use crate::bc_path::{DistanceOracle, Goal};
use cambc::Champion;

/// Bump with every layout change (nn/spec.py checks it).
pub const SPEC: u32 = 0x4E4E_0001;
pub const S: usize = 310;
pub const HIST: usize = 3;
pub const R: i32 = 7;
pub const SIDE: usize = 15;
pub const CH: usize = 5;
pub const GRID_BITS: usize = SIDE * SIDE * CH;
pub const GRID_BYTES: usize = GRID_BITS.div_ceil(8);
pub const INPUT: usize = S * (1 + HIST) + GRID_BITS;
pub const HEADS: [usize; 4] = [15, 10, 4, 2];
const KINDS: usize = 13;
const GROUP_SLOTS: usize = 4;
const ENEMY_SLOTS: usize = 3;
/// Enemies this close (not on their spawn) allow a manual step.
const FIGHT_RADIUS: i32 = 6;

pub const G_TRIBUTE: usize = 4;
pub const G_SHOP: usize = 5;
pub const G_RETREAT: usize = 6;
pub const G_WALK_HOME: usize = 7;
pub const G_HOLD: usize = 8;
pub const G_ENGAGE: usize = 9;
pub const G_KITE: usize = 12;
pub const G_STEP_BACK: usize = 13;
pub const G_SIGHTING: usize = 14;

static TEACHER_BLOB: &[u8] = include_bytes!("weights/teacher.bin");

thread_local! {
    static NET: std::cell::OnceCell<Option<Net>> = const { std::cell::OnceCell::new() };
}

/// Previous observations of this champion (newest first), as the network sees them.
#[derive(Default)]
pub struct History {
    prev: Vec<[i8; S]>,
}

pub struct Obs {
    pub s: [i8; S],
    pub grid: [u8; GRID_BYTES],
    /// What each goal means this turn (None = not available).
    pub goals: [Option<Intent>; HEADS[0]],
    pub enemies: [Option<Pos>; ENEMY_SLOTS],
    /// Stay, then the 8 directions.
    pub steps: [Pos; 9],
    pub masks: [u32; 4],
    pub rule_goal: usize,
    pub fight_near: bool,
}

/// The network's (or, logged, the rules') choice per head; -1 = head not used.
#[derive(Clone, Copy)]
pub struct Choice {
    pub a: [i32; 4],
    pub logp: f32,
}

struct Put<'a> {
    s: &'a mut [i8; S],
    at: usize,
}

impl Put<'_> {
    fn v(&mut self, v: f32) {
        if let Some(x) = self.s.get_mut(self.at) {
            *x = (v.clamp(-1.0, 1.0) * 127.0).round() as i8;
        }
        self.at += 1;
    }

    fn b(&mut self, b: bool) {
        self.v(if b { 1.0 } else { 0.0 });
    }

    fn onehot(&mut self, i: Option<usize>, n: usize) {
        for k in 0..n {
            self.b(i == Some(k));
        }
    }
}

fn sign(a: Option<u16>, b: Option<u16>) -> f32 {
    match (a, b) {
        (Some(a), Some(b)) => (f32::from(b) - f32::from(a)).signum(),
        _ => 0.0,
    }
}

impl Bot {
    pub(super) fn teacher_mode(&self) -> u8 {
        self.theta.knob2(knob2::TEACHER)
    }

    /// Build this turn's observation. `rule` is the rules' final intent.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn teacher_obs(
        &self,
        ctx: &Context,
        vis: &Visible,
        chan: &TeamChannel,
        me: &Champion,
        my_pos: Pos,
        round: u32,
        hurt: bool,
        rule: Intent,
    ) -> Obs {
        let mut s = [0i8; S];
        let mut p = Put { s: &mut s, at: 0 };
        let fields = &self.world.fields;
        let (w, h) = (self.world.width.max(1) as f32, self.world.height.max(1) as f32);

        // ---- self (24) + champion (13) ----
        let cd = |i: usize| me.abilities().get(i).and_then(|a| a.cooldown().ok()).unwrap_or(0) as f32;
        p.v(ctx.hp_frac());
        p.v(ctx.max_hp / 3000.0);
        p.v(me.strength().unwrap_or(0.0) / 150.0);
        p.v(me.focus().unwrap_or(0.0) / 250.0);
        p.b(ctx.ranged);
        p.v(self.kit.attack_range as f32 / 3.0);
        p.v(cd(0) / 8.0);
        p.v(cd(1) / 8.0);
        p.b(me.can_move().unwrap_or(false));
        p.b(me.can_attack().unwrap_or(false));
        p.v(ctx.gems as f32 / 100.0);
        p.v(ctx.gold as f32 / 500.0);
        p.v(self.ledger.lifetime_gold as f32 / 5000.0);
        p.v(Family::ALL.iter().map(|&f| f32::from(self.ledger.emblems(f))).sum::<f32>() / 6.0);
        p.v(my_pos.x as f32 / w * 2.0 - 1.0);
        p.v(my_pos.y as f32 / h * 2.0 - 1.0);
        p.b(ctx.at_spawn);
        p.v(ctx.dist_spawn.map_or(1.0, |d| f32::from(d) / 100.0));
        p.v(ctx.dist_altar.map_or(1.0, |d| f32::from(d) / 100.0));
        p.v(round as f32 / 2000.0);
        p.v(self.dmg_taken / 300.0);
        p.b(hurt);
        p.b(ctx.on_capture_tile);
        p.v(ctx.threat_here / 400.0);
        p.onehot(Some(self.kind as usize), KINDS);

        // ---- economy (6) + map (4) ----
        p.v(ctx.ally_tributed as f32 / 1000.0);
        p.v(ctx.enemy_tributed as f32 / 1000.0);
        p.v((ctx.ally_tributed as f32 - ctx.enemy_tributed as f32) / 1000.0);
        p.v(ctx.winning_gems.saturating_sub(ctx.ally_tributed + ctx.gems) as f32 / 1000.0);
        p.v(f32::from(ctx.team.aggression) / 255.0);
        p.v(f32::from(ctx.team.tribute_threshold) / 255.0);
        p.v(w / 100.0);
        p.v(h / 100.0);
        p.v(self.team_size as f32 / 5.0);
        p.v(self.world.groups.len() as f32 / 30.0);

        // ---- allies (2 x 18) ----
        let mut allies: Vec<_> = vis.allies.iter().collect();
        allies.sort_by_key(|a| a.dist);
        for k in 0..2 {
            let a = allies.get(k);
            p.b(a.is_some());
            p.v(a.map_or(0.0, |a| (a.pos.x - my_pos.x) as f32 / 16.0));
            p.v(a.map_or(0.0, |a| (a.pos.y - my_pos.y) as f32 / 16.0));
            p.v(a.map_or(0.0, |a| a.dist as f32 / 16.0));
            p.v(a.map_or(0.0, |a| a.hp / a.max_hp.max(1.0)));
            p.onehot(a.map(|a| a.kind as usize), KINDS);
        }

        // ---- enemies (3 x 26), nearest first ----
        let mut enemies = [None; ENEMY_SLOTS];
        for k in 0..ENEMY_SLOTS {
            let e = vis.enemies.get(k);
            enemies[k] = e.map(|e| e.pos);
            p.b(e.is_some());
            p.v(e.map_or(0.0, |e| (e.pos.x - my_pos.x) as f32 / 16.0));
            p.v(e.map_or(0.0, |e| (e.pos.y - my_pos.y) as f32 / 16.0));
            p.v(e.map_or(0.0, |e| e.dist as f32 / 16.0));
            p.v(e.map_or(0.0, |e| e.hp / e.max_hp.max(1.0)));
            p.v(e.map_or(0.0, |e| e.max_hp / 3000.0));
            p.v(e.map_or(0.0, |e| e.dps / 400.0));
            p.v(e.map_or(0.0, |e| e.gems as f32 / 100.0));
            p.b(e.is_some_and(|e| e.ranged));
            p.v(e.map_or(0.0, |e| e.range as f32 / 3.0));
            p.b(e.is_some_and(|e| e.on_spawn));
            p.b(e.is_some_and(|e| e.tributing));
            p.b(e.is_some_and(|e| e.recalling));
            p.onehot(e.map(|e| e.kind as usize), KINDS);
        }

        // ---- remembered enemies (4) ----
        let ghost = vis.ghosts.iter().min_by_key(|g| g.pos.cheb(my_pos));
        p.v(vis.ghosts.len() as f32 / 5.0);
        p.v(ghost.map_or(0.0, |g| (g.pos.x - my_pos.x) as f32 / 16.0));
        p.v(ghost.map_or(0.0, |g| (g.pos.y - my_pos.y) as f32 / 16.0));
        p.v(ghost.map_or(0.0, |g| g.dps / 400.0));

        // ---- camps (4 x 6), nearest first ----
        let mut camps: Vec<_> = vis.camps.iter().filter(|c| c.hp > 0.0).collect();
        camps.sort_by_key(|c| c.dist);
        for k in 0..4 {
            let c = camps.get(k);
            p.b(c.is_some());
            p.v(c.map_or(0.0, |c| (c.pos.x - my_pos.x) as f32 / 16.0));
            p.v(c.map_or(0.0, |c| (c.pos.y - my_pos.y) as f32 / 16.0));
            p.v(c.map_or(0.0, |c| c.hp / 2000.0));
            p.b(c.is_some_and(|c| c.awake));
            p.v(c.map_or(0.0, |c| c.strength / 200.0));
        }

        // ---- camp-group slots (4 x 13): the nearest reachable groups, plus the rules' ----
        let rule_group = match rule {
            Intent::ClearCamp(g) | Intent::Capture(g) | Intent::Approach(g) => Some(g as usize),
            _ => None,
        };
        let mut near: Vec<(u32, usize)> = (0..self.world.groups.len())
            .filter_map(|i| self.group_distance(i, my_pos).map(|d| (d, i)))
            .collect();
        near.sort_unstable();
        let mut slots: Vec<(u32, usize)> = near.iter().copied().take(GROUP_SLOTS).collect();
        if let Some(g) = rule_group {
            if !slots.iter().any(|&(_, i)| i == g) {
                let d = self.group_distance(g, my_pos).unwrap_or(u32::from(u16::MAX));
                if slots.len() == GROUP_SLOTS {
                    slots.pop();
                }
                slots.push((d, g));
                slots.sort_unstable();
            }
        }
        let mut goals: [Option<Intent>; HEADS[0]] = [None; HEADS[0]];
        let claims = |i: usize| {
            self.priority_index(i).map_or(0, |pi| chan.claimants(pi, self.slot as usize))
        };
        for k in 0..GROUP_SLOTS {
            let slot = slots.get(k).copied();
            let state = slot.map(|(_, i)| self.belief.expected_state(i, round));
            let g = slot.map(|(_, i)| &self.world.groups[i]);
            p.b(slot.is_some());
            p.v(slot.map_or(1.0, |(d, _)| d as f32 / 100.0));
            p.b(state == Some(GroupState::Alive));
            p.b(state == Some(GroupState::RewardAvailable));
            p.b(state == Some(GroupState::Respawning));
            p.b(state == Some(GroupState::Unknown));
            p.v(g.map_or(0.0, |g| g.gems as f32 / 200.0));
            p.v(g.map_or(0.0, |g| g.gold as f32 / 300.0));
            let eta = slot
                .and_then(|(_, i)| self.belief.group(i))
                .and_then(|b| b.respawn_at)
                .map_or(0.0, |at| at.saturating_sub(round) as f32 / 150.0);
            p.v(eta);
            p.v(slot.map_or(0.0, |(_, i)| claims(i) as f32 / 3.0));
            p.v(g.map_or(0.0, |g| g.total_hp / 20000.0));
            p.b(slot.is_some_and(|(_, i)| Some(i) == rule_group));
            p.b(g.is_some_and(|g| g.has_monolith));
            if let (Some((_, i)), Some(st)) = (slot, state) {
                let i = i as u8;
                goals[k] = Some(match st {
                    GroupState::RewardAvailable => Intent::Capture(i),
                    GroupState::Respawning => Intent::Approach(i),
                    _ => Intent::ClearCamp(i),
                });
            }
        }

        // ---- the other goals ----
        let fight_near = vis.enemies.iter().any(|e| !e.on_spawn && e.dist <= FIGHT_RADIUS);
        if ctx.gems > 0 && !self.world.altars.is_empty() {
            goals[G_TRIBUTE] = Some(Intent::Tribute);
        }
        goals[G_SHOP] = Some(Intent::Shop);
        if !ctx.at_spawn {
            goals[G_RETREAT] = Some(Intent::Retreat);
            goals[G_WALK_HOME] = Some(Intent::WalkHome);
        }
        goals[G_HOLD] = Some(Intent::Hold);
        for k in 0..ENEMY_SLOTS {
            if let Some(e) = vis.enemies.get(k).filter(|e| !e.on_spawn) {
                goals[G_ENGAGE + k] = Some(Intent::Engage(e.pos));
            }
        }
        if let Some(e) = vis.enemies.first().filter(|e| !e.on_spawn && ctx.ranged) {
            goals[G_KITE] = Some(Intent::Kite(e.pos));
        }
        let threat = vis
            .camps
            .iter()
            .filter(|c| c.awake && c.hp > 0.0 && c.dist <= 3)
            .map(|c| (c.dist, c.pos))
            .chain(vis.enemies.iter().filter(|e| !e.on_spawn && e.dist <= 3).map(|e| (e.dist, e.pos)))
            .min_by_key(|t| t.0);
        if let Some((_, t)) = threat {
            goals[G_STEP_BACK] = Some(Intent::StepBack(t));
        }
        if let Some((sp, age)) = chan.sighting(round) {
            if age <= 1 && sp.cheb(my_pos) >= 2 {
                goals[G_SIGHTING] = Some(Intent::Engage(sp));
            }
        }
        let rule_goal = goal_of(rule, &goals, vis);
        // The rules' suggestion (14 + 1 spare = 15).
        p.onehot(Some(rule_goal), HEADS[0]);

        // ---- per step (9 x 6): free, enemy reach, camp reach, closer to spawn / altar / target ----
        let occupied = occupied_tiles(vis, my_pos);
        let steps: [Pos; 9] =
            core::array::from_fn(|i| if i == 0 { my_pos } else { my_pos + DIRS8[i - 1] });
        let here = |g: Goal| fields.distance(g, my_pos);
        let target_goal = self.target.map(|t| Goal::Group(t as u8));
        let mut free = [false; 9];
        for (i, &sp) in steps.iter().enumerate() {
            free[i] = i == 0 || (self.world.terrain.walkable(sp) && !occupied.contains(&sp));
            p.b(free[i]);
            let n = vis.enemies.iter().filter(|e| !e.on_spawn && e.pos.cheb(sp) <= e.range).count();
            p.v(n as f32 / 3.0);
            let c = vis.camps.iter().filter(|c| c.awake && c.hp > 0.0 && c.pos.cheb(sp) <= 3).count();
            p.v(c as f32 / 3.0);
            p.v(sign(here(Goal::AllySpawn), fields.distance(Goal::AllySpawn, sp)));
            p.v(sign(here(Goal::Altars), fields.distance(Goal::Altars, sp)));
            p.v(target_goal.map_or(0.0, |g| sign(here(g), fields.distance(g, sp))));
        }
        debug_assert_eq!(p.at, S);

        // ---- local grid: blocked, bush, enemy, ally, camp ----
        let mut grid = [0u8; GRID_BYTES];
        let mut set = |c: usize, dx: i32, dy: i32| {
            if dx.abs() <= R && dy.abs() <= R {
                let bit = (c * SIDE + (dy + R) as usize) * SIDE + (dx + R) as usize;
                grid[bit / 8] |= 1 << (bit % 8);
            }
        };
        for dy in -R..=R {
            for dx in -R..=R {
                let t = Pos::new(my_pos.x + dx, my_pos.y + dy);
                if !self.world.terrain.walkable(t) {
                    set(0, dx, dy);
                }
                if t.x >= 0 && t.y >= 0 && t.x < self.world.width && t.y < self.world.height && self.world.terrain.bushes.get(t) {
                    set(1, dx, dy);
                }
            }
        }
        for e in &vis.enemies {
            set(2, e.pos.x - my_pos.x, e.pos.y - my_pos.y);
        }
        for a in &vis.allies {
            set(3, a.pos.x - my_pos.x, a.pos.y - my_pos.y);
        }
        for c in vis.camps.iter().filter(|c| c.hp > 0.0) {
            set(4, c.pos.x - my_pos.x, c.pos.y - my_pos.y);
        }

        // ---- masks ----
        let mut gm = bits(goals.iter().map(Option::is_some));
        gm |= 1 << rule_goal;
        let sm = if fight_near { 1 | (bits(free.iter().copied()) << 1) } else { 1 };
        let am = 1 | bits((0..ENEMY_SLOTS).map(|k| vis.enemies.get(k).is_some_and(|e| !e.on_spawn))) << 1;
        Obs { s, grid, goals, enemies, steps, masks: [gm, sm, am, 0b11], rule_goal, fight_near }
    }

    /// The network's choice (TEACHER = 2), or None without a usable network.
    pub(super) fn teacher_pick(&self, obs: &Obs, hist: &History) -> Option<Choice> {
        NET.with(|cell| {
            let net = cell
                .get_or_init(|| Net::parse(TEACHER_BLOB).filter(|n| n.spec == SPEC && n.input_len() == INPUT))
                .as_ref()?;
            let x = input(obs, hist);
            let logits = net.forward(&x);
            let seed = self.theta.knob2(knob2::RL_SEED);
            let u = |head: u8| (seed > 0).then(|| rl::uniform(self.turns, self.slot, seed, 10 + head));
            let mask = |h: usize| -> Vec<bool> { (0..HEADS[h]).map(|i| obs.masks[h] >> i & 1 == 1).collect() };
            let head = |h: usize| net.head(h).map(|r| &logits[r]);
            let (g, lg) = nn::pick(head(0)?, &mask(0), u(0))?;
            let (st, ls) = nn::pick(head(1)?, &mask(1), u(1))?;
            let mut c = Choice { a: [g as i32, st as i32, -1, -1], logp: lg + ls };
            if st > 0 {
                let (at, la) = nn::pick(head(2)?, &mask(2), u(2))?;
                let (ca, lc) = nn::pick(head(3)?, &mask(3), u(3))?;
                c.a[2] = at as i32;
                c.a[3] = ca as i32;
                c.logp += la + lc;
            }
            Some(c)
        })
    }

    /// The rules' choice as labels, from the rules' intent and what the executor did.
    pub(super) fn teacher_labels(obs: &Obs, my_pos: Pos, out: &Outcome, new_pos: Pos) -> Choice {
        // Manual steps only when the rules were fighting a champion: other intents (farming next
        // to an enemy, say) need the executor's camp attacks, which the micro heads can't name.
        let fighting = (G_ENGAGE..=G_KITE).contains(&obs.rule_goal);
        let step = if !obs.fight_near || !fighting {
            0
        } else if out.moved {
            obs.steps.iter().position(|&s| s == new_pos).map_or(1, |i| i as i32 + 1)
        } else {
            1
        };
        let _ = my_pos;
        let (attack, cast) = if step == 0 {
            (-1, -1)
        } else {
            let slot = out
                .attacked_at
                .and_then(|p| obs.enemies.iter().position(|&e| e == Some(p)))
                .map_or(0, |k| k as i32 + 1);
            (slot, i32::from(out.cast))
        };
        Choice { a: [obs.rule_goal as i32, step, attack, cast], logp: 0.0 }
    }

    /// Print the `N` line (debug runs only).
    pub(super) fn teacher_log(&self, round: u32, obs: &Obs, c: &Choice, mode: u8) {
        cambc::debug_block! {
            let s: String = obs.s.iter().map(|b| format!("{:02x}", *b as u8)).collect();
            let g: String = obs.grid.iter().map(|b| format!("{b:02x}")).collect();
            // Labels must be allowed by the masks (the rules can do what the heads can't name).
            let mut m = obs.masks;
            for (h, &a) in c.a.iter().enumerate() {
                if a >= 0 {
                    m[h] |= 1 << a;
                }
            }
            println!(
                "N {{\"r\":{round},\"c\":\"{}\",\"t\":{mode},\"s\":\"{s}\",\"g\":\"{g}\",\"a\":{:?},\"m\":{:?},\"lp\":{}}}",
                super::champ_name(self.kind), c.a, m, (c.logp * 1000.0).round() as i64
            );
        }
    }

    /// One-off header with the layout (nn/spec.py checks it).
    pub(super) fn teacher_header(&self) {
        cambc::debug_block! {
            println!(
                "NH {{\"spec\":{SPEC},\"c\":\"{}\",\"S\":{S},\"hist\":{HIST},\"side\":{SIDE},\"ch\":{CH},\"heads\":{HEADS:?}}}",
                super::champ_name(self.kind)
            );
        }
    }
}

impl History {
    pub fn push(&mut self, s: &[i8; S]) {
        self.prev.insert(0, *s);
        self.prev.truncate(HIST);
    }
}

/// The network input: this turn, the previous HIST turns (zeros before the first), the grid.
fn input(obs: &Obs, hist: &History) -> Vec<f32> {
    let mut x = Vec::with_capacity(INPUT);
    x.extend(obs.s.iter().map(|&v| f32::from(v) / 127.0));
    for k in 0..HIST {
        match hist.prev.get(k) {
            Some(s) => x.extend(s.iter().map(|&v| f32::from(v) / 127.0)),
            None => x.extend(core::iter::repeat_n(0.0, S)),
        }
    }
    x.extend((0..GRID_BITS).map(|b| f32::from(obs.grid[b / 8] >> (b % 8) & 1)));
    x
}

fn bits(it: impl Iterator<Item = bool>) -> u32 {
    it.enumerate().fold(0, |m, (i, b)| m | (u32::from(b) << i))
}

/// The goal index of an intent (the rules' choice).
fn goal_of(intent: Intent, goals: &[Option<Intent>; HEADS[0]], vis: &Visible) -> usize {
    let group_slot = |g: u8| {
        (0..GROUP_SLOTS).find(|&k| {
            matches!(goals[k], Some(Intent::ClearCamp(i) | Intent::Capture(i) | Intent::Approach(i)) if i == g)
        })
    };
    match intent {
        Intent::ClearCamp(g) | Intent::Capture(g) | Intent::Approach(g) => group_slot(g).unwrap_or(G_HOLD),
        Intent::Tribute => G_TRIBUTE,
        Intent::Shop => G_SHOP,
        Intent::Retreat => G_RETREAT,
        Intent::WalkHome => G_WALK_HOME,
        Intent::Hold => G_HOLD,
        Intent::Kite(_) => G_KITE,
        Intent::StepBack(_) => G_STEP_BACK,
        Intent::Engage(p) => vis
            .enemies
            .iter()
            .take(ENEMY_SLOTS)
            .position(|e| e.pos == p)
            .map_or(G_SIGHTING, |k| G_ENGAGE + k),
    }
}

/// The intent a goal index means, for the network's pick.
pub fn intent_of(obs: &Obs, goal: usize) -> Option<Intent> {
    obs.goals.get(goal).copied().flatten()
}
