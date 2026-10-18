//! Portfolio combat search: pick a skirmish behaviour by simulating it.
//!
//! When enemies are close, roll the fight forward a few rounds for each of a small set
//! of scripted team behaviours (ours) against each scripted response (theirs) and take
//! the behaviour with the best worst case (1-ply max-min over a 4 x 3 matrix). This is
//! the "portfolio" approach to real-time strategy micro: search over scripts, not moves.
//!
//! The model is deliberately coarse (no walls, one step per round, damage = dps per
//! round once in range) and allocation-free (fixed arrays). The caller passes a fuel
//! floor: rollouts stop when fuel drops below it and the best plan so far is used, so
//! the search can never over-commit the turn. Every ally runs the same search on nearly
//! the same view, so the team converges on one behaviour without shared memory.

use crate::act::Visible;
use crate::bc_core::Pos;
use crate::sdk::fuel;

/// Our scripted behaviours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// Everyone focuses the weakest enemy.
    FocusWeak,
    /// Everyone focuses the richest enemy (gem carrier), falling back to the weakest.
    FocusCarrier,
    /// Hold range: step back when the target is closer than our range, attack otherwise.
    Kite,
    /// Walk away from the enemy (towards home), hit only what is adjacent.
    Retreat,
}

const OURS: [Plan; 4] = [Plan::FocusWeak, Plan::Retreat, Plan::FocusCarrier, Plan::Kite];
/// Their responses: 0 focus our weakest, 1 hit the nearest, 2 retreat.
const THEIRS: usize = 3;
const ROUNDS: usize = 3;
const N: usize = 4;

#[derive(Clone, Copy, Default)]
struct Unit {
    pos: Pos,
    hp: f32,
    dps: f32,
    range: i32,
    gems: f32,
}

#[derive(Clone, Copy, Default)]
struct Side {
    u: [Unit; N],
    n: usize,
}

impl Side {
    fn live(&self, i: usize) -> bool {
        i < self.n && self.u[i].hp > 0.0
    }
    fn any(&self) -> bool {
        (0..self.n).any(|i| self.live(i))
    }
    fn weakest(&self) -> Option<usize> {
        (0..self.n).filter(|&i| self.live(i)).min_by(|&a, &b| self.u[a].hp.total_cmp(&self.u[b].hp))
    }
    fn carrier(&self) -> Option<usize> {
        (0..self.n)
            .filter(|&i| self.live(i) && self.u[i].gems >= 20.0)
            .max_by(|&a, &b| self.u[a].gems.total_cmp(&self.u[b].gems))
    }
    fn nearest(&self, from: Pos) -> Option<usize> {
        (0..self.n).filter(|&i| self.live(i)).min_by_key(|&i| self.u[i].pos.cheb(from))
    }
    /// Value lost (dead units, weighted by their bags).
    fn lost(&self) -> f32 {
        (0..self.n).filter(|&i| self.u[i].hp <= 0.0).map(|i| 1.0 + self.u[i].gems / 50.0).sum()
    }
}

/// The behaviour to use and the enemy it targets (`None` target = retreat), or `None`
/// when there is no skirmish worth searching or no fuel to finish one rollout.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn choose(
    vis: &Visible,
    me: Pos,
    my_hp: f32,
    my_dps: f32,
    my_range: i32,
    my_gems: u32,
    home: Pos,
    fuel_floor: u64,
) -> Option<(Plan, Option<Pos>)> {
    let mut ours = Side::default();
    ours.u[0] = Unit { pos: me, hp: my_hp, dps: my_dps, range: my_range, gems: my_gems as f32 };
    ours.n = 1;
    for a in vis.allies.iter().filter(|a| a.dist <= 4 && a.hp > 0.0) {
        if ours.n == N - 1 {
            break;
        }
        ours.u[ours.n] = Unit { pos: a.pos, hp: a.hp, dps: a.dps, range: 1, gems: 0.0 };
        ours.n += 1;
    }
    let mut theirs = Side::default();
    for e in vis.enemies.iter().filter(|e| e.dist <= 6 && !e.on_spawn && e.hp > 0.0) {
        if theirs.n == N {
            break;
        }
        theirs.u[theirs.n] =
            Unit { pos: e.pos, hp: e.hp, dps: e.dps, range: e.range, gems: e.gems as f32 };
        theirs.n += 1;
    }
    if theirs.n == 0 {
        return None;
    }
    let mut best: Option<(f32, Plan)> = None;
    'plans: for plan in OURS {
        let mut worst = f32::INFINITY;
        for response in 0..THEIRS {
            if fuel() < fuel_floor {
                break 'plans;
            }
            worst = worst.min(rollout(ours, theirs, plan, response, home));
        }
        if best.is_none_or(|(v, _)| worst > v) {
            best = Some((worst, plan));
        }
    }
    let (_, plan) = best?;
    let target = match plan {
        Plan::Retreat => None,
        Plan::FocusCarrier => theirs.carrier().or_else(|| theirs.weakest()),
        Plan::FocusWeak | Plan::Kite => theirs.weakest(),
    }
    .map(|i| theirs.u[i].pos);
    Some((plan, target))
}

/// One step from `p` towards (`towards = true`) or away from `goal`.
fn step(p: Pos, goal: Pos, towards: bool) -> Pos {
    let (dx, dy) = ((goal.x - p.x).signum(), (goal.y - p.y).signum());
    if towards {
        Pos::new(p.x + dx, p.y + dy)
    } else {
        Pos::new(p.x - dx, p.y - dy)
    }
}

/// Score for us after ROUNDS rounds: kills (and bags won) minus deaths (and bags lost),
/// plus a small damage-trade term.
fn rollout(mut ours: Side, mut theirs: Side, plan: Plan, response: usize, home: Pos) -> f32 {
    let mut trade = 0.0;
    for _ in 0..ROUNDS {
        if !ours.any() || !theirs.any() {
            break;
        }
        let focus = match plan {
            Plan::FocusCarrier => theirs.carrier().or_else(|| theirs.weakest()),
            _ => theirs.weakest(),
        };
        for i in 0..ours.n {
            if !ours.live(i) {
                continue;
            }
            let u = ours.u[i];
            let tgt = if plan == Plan::Retreat { theirs.nearest(u.pos) } else { focus };
            let Some(ti) = tgt else { continue };
            let goal = theirs.u[ti].pos;
            let d = u.pos.cheb(goal);
            let pos = match plan {
                Plan::Retreat => step(u.pos, home, true),
                Plan::Kite if d < u.range => step(u.pos, goal, false),
                _ if d > u.range => step(u.pos, goal, true),
                _ => u.pos,
            };
            ours.u[i].pos = pos;
            let reach = if plan == Plan::Retreat { 1 } else { u.range };
            if pos.cheb(goal) <= reach {
                theirs.u[ti].hp -= u.dps;
                trade += u.dps;
            }
        }
        let their_focus = ours.weakest();
        for j in 0..theirs.n {
            if !theirs.live(j) {
                continue;
            }
            let e = theirs.u[j];
            let tgt = if response == 0 { their_focus } else { ours.nearest(e.pos) };
            let Some(oi) = tgt else { break };
            let goal = ours.u[oi].pos;
            if response == 2 {
                theirs.u[j].pos = step(e.pos, goal, false);
                continue;
            }
            let pos = if e.pos.cheb(goal) > e.range { step(e.pos, goal, true) } else { e.pos };
            theirs.u[j].pos = pos;
            if pos.cheb(goal) <= e.range {
                ours.u[oi].hp -= e.dps;
                trade -= e.dps;
            }
        }
    }
    theirs.lost() - 1.2 * ours.lost() + trade / 2000.0
}
