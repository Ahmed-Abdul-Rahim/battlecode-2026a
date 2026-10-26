//! Draft side of the match plan (bc_core::plan): the map analysis on the first draft turn
//! (10M fuel since engine 2.0.0, whole map in view) and the roster-dependent part on the last pick (100k).
//! Champions read the result from draft memory instead of computing it at 100k per turn.

use crate::bc_core::champion::ChampionKind;
use crate::bc_core::plan::{class, MatchPlan, PlanGroup, BODIES, FIELD_ORDER_LEN, MAX_ALTARS, MAX_CLUSTERS, MAX_GROUPS, NEIGHBOURS, NONE, OPENING_LEN, UNKNOWN};
use crate::bc_core::Pos;
use crate::bc_draft::CampKind;
use crate::sdk::{self, fuel, pos, vec2};
use cambc::{Bitboard, DistanceField, Result, Team};

/// Fuel the first draft turn keeps back after the analysis (it still has to pick and write).
const KEEP_FUEL: u64 = 400_000;
/// Neighbour search radius (walking steps) from each group. Engine 2.0.0 gives the first draft
/// turn 10M fuel (the analysis used ~0.8M of it), so the search reaches every group and goes
/// further than the 40 steps that fitted in 1M.
const NEIGHBOUR_STEPS: u16 = 96;
/// Groups at most this many steps apart (by walking) join one cluster.
const CLUSTER_STEPS: u8 = 6;
/// COLD: this many steps (/2) farther from the enemy spawn than from ours.
const COLD_MARGIN: u8 = 6;
/// FRONTIER: this many steps (/2) closer to the enemy spawn than to ours.
const FRONTIER_MARGIN: u8 = 2;

/// Per-group static facts the last pick needs again (no SDK calls there: 100k fuel).
pub struct GroupFacts {
    pub gems: u32,
    pub gold: u32,
    /// Summed camp health and strength (the last pick only does arithmetic on these).
    pub hp: f32,
    pub hit: f32,
}

pub struct MapAnalysis {
    pub plan: MatchPlan,
    pub facts: Vec<GroupFacts>,
    /// Fuel the analysis cost (debug log).
    pub spent: u64,
}

fn steps2(d: u16) -> u8 {
    if d == DistanceField::UNREACHABLE {
        UNKNOWN
    } else {
        (d / 2).min(254) as u8
    }
}

fn field(walls: &Bitboard, roots: &Bitboard, max: u16) -> Result<DistanceField> {
    let mut f = DistanceField::new()?;
    f.compute(walls, roots, &mut Bitboard::new()?, max)?;
    Ok(f)
}

fn min_over(f: &DistanceField, tiles: &[Pos]) -> u16 {
    tiles.iter().filter_map(|&t| f.get(vec2(t)).ok()).min().unwrap_or(DistanceField::UNREACHABLE)
}

/// First draft turn: everything that depends only on the map.
pub fn analyse() -> Result<MapAnalysis> {
    let start = fuel();
    let (w, h) = (cambc::map::width()?, cambc::map::height()?);
    let mut walls = cambc::map::walls()?.clone();
    walls.union_assign(cambc::map::holes()?)?;
    walls.union_assign(cambc::map::altars()?)?;
    let free = walls.complement()?;
    let mut ours = cambc::map::spawn_region(Team::Ally)?.clone();
    ours.intersect_assign(&free)?;
    let mut theirs = cambc::map::spawn_region(Team::Enemy)?.clone();
    theirs.intersect_assign(&free)?;
    let f_us = field(&walls, &ours, DistanceField::UNREACHABLE)?;
    let f_them = field(&walls, &theirs, DistanceField::UNREACHABLE)?;

    let mut plan = MatchPlan {
        symmetry: cambc::map::symmetry().map_or(NONE, |s| s as i32 as u8),
        map_class: (w.max(h) / 25).min(15) as u8,
        ..MatchPlan::default()
    };

    // Altars: one field each (tribute tiles around it), safety = their walk minus ours.
    let range = cambc::constants().submit_gems_range as f32;
    let altars: Vec<Pos> = cambc::map::altars()?.positions()?.into_iter().map(pos).collect();
    let mut altar_fields = Vec::new();
    for &a in altars.iter().take(MAX_ALTARS) {
        let mut roots = Bitboard::new()?;
        roots.draw_chebyshev(vec2(a), range)?;
        roots.intersect_assign(&free)?;
        let us = steps2(min_over(&f_us, &[a]).saturating_sub(1));
        let them = steps2(min_over(&f_them, &[a]).saturating_sub(1));
        let safety = (128 + i32::from(them) - i32::from(us)).clamp(0, 254) as u8;
        plan.altars.push((safety, us));
        altar_fields.push(field(&walls, &roots, DistanceField::UNREACHABLE)?);
    }

    let bushes = cambc::map::bushes()?;
    let mut facts = Vec::new();
    let mut tiles_of = Vec::new();
    for g in cambc::map::camp_groups()?.into_iter().take(MAX_GROUPS) {
        let camps: Vec<CampKind> =
            g.camps().iter().filter_map(|c| CampKind::from_engine(c.kind() as i32)).collect();
        let tiles: Vec<Pos> = g.camps().iter().map(|c| pos(c.spawn_position())).collect();
        let (gems, gold) = (g.gem_value(), g.gold_value());
        let mut pg = PlanGroup {
            from_us: steps2(min_over(&f_us, &tiles)),
            from_them: steps2(min_over(&f_them, &tiles)),
            to_altar: UNKNOWN,
            altar: NONE,
            cluster: NONE,
            ..PlanGroup::default()
        };
        // Nearest safe altar (safety >= 128: no closer to them than to us), else the nearest.
        let mut best: Option<(bool, u8, usize)> = None;
        for (k, f) in altar_fields.iter().enumerate() {
            let d = steps2(min_over(f, &tiles));
            let safe = plan.altars[k].0 >= 128;
            if d != UNKNOWN && best.is_none_or(|(s, bd, _)| (safe, std::cmp::Reverse(d)) > (s, std::cmp::Reverse(bd))) {
                best = Some((safe, d, k));
            }
        }
        if let Some((_, d, k)) = best {
            pg.to_altar = d;
            pg.altar = k as u8;
        }
        let mut c = 0u8;
        if gems > 0 {
            c |= class::GEMS;
        }
        if gold > 0 {
            c |= class::GOLD;
        }
        for k in &camps {
            c |= match k {
                CampKind::Elk => class::ELK,
                CampKind::Golem => class::GOLEM,
                CampKind::Monolith => class::MONOLITH,
                _ => 0,
            };
        }
        let mut near = Bitboard::new()?;
        for &t in &tiles {
            near.draw_chebyshev(vec2(t), 3.0)?;
        }
        near.intersect_assign(bushes)?;
        if near.count()? > 0 {
            c |= class::BUSH;
        }
        if pg.from_us != UNKNOWN && pg.from_them != UNKNOWN {
            if u16::from(pg.from_them) >= u16::from(pg.from_us) + u16::from(COLD_MARGIN) {
                c |= class::COLD;
            }
            if u16::from(pg.from_them) + u16::from(FRONTIER_MARGIN) < u16::from(pg.from_us) {
                c |= class::FRONTIER;
            }
        }
        pg.class = c;
        plan.groups.push(pg);
        facts.push(GroupFacts {
            gems,
            gold,
            hp: camps.iter().map(|c| c.health()).sum(),
            hit: camps.iter().map(|c| c.strength()).sum(),
        });
        tiles_of.push(tiles);
    }
    let n = plan.groups.len();

    // Field order: value per walk from our spawn (the groups champions will want first).
    let value = |i: usize| facts[i].gems as f32 + facts[i].gold as f32 / 3.0;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let s = |i: usize| value(i) / (f32::from(plan.groups[i].from_us.min(254)) + 5.0);
        s(b).total_cmp(&s(a))
    });
    plan.field_order = order.iter().take(FIELD_ORDER_LEN).map(|&i| i as u8).collect();

    // Neighbours: a bounded field from each group (most valuable first, while fuel lasts),
    // read at one tile of every other group.
    plan.neighbours = vec![[(NONE, UNKNOWN); NEIGHBOURS]; n];
    for &i in &order {
        if fuel() < KEEP_FUEL + 30_000 {
            break;
        }
        let mut roots = Bitboard::new()?;
        for &t in &tiles_of[i] {
            roots.set(vec2(t), true)?;
        }
        let f = field(&walls, &roots, NEIGHBOUR_STEPS)?;
        // Only groups the bounded search can have reached (each get costs ~74 fuel).
        let reach = i32::from(NEIGHBOUR_STEPS);
        let here = tiles_of[i][0];
        let mut near: Vec<(u8, u8)> = (0..n)
            .filter(|&j| j != i && tiles_of[j][0].cheb(here) <= reach)
            .filter_map(|j| {
                let d = f.get(vec2(tiles_of[j][0])).ok()?;
                (d != DistanceField::UNREACHABLE).then(|| (j as u8, steps2(d)))
            })
            .collect();
        near.sort_by_key(|&(_, d)| d);
        for (k, &e) in near.iter().take(NEIGHBOURS).enumerate() {
            plan.neighbours[i][k] = e;
        }
    }

    // Clusters: groups linked by a neighbour edge of at most CLUSTER_STEPS (union-find).
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for i in 0..n {
        for &(j, d) in &plan.neighbours[i] {
            if j != NONE && d <= CLUSTER_STEPS {
                let (a, b) = (root(&mut parent, i), root(&mut parent, usize::from(j)));
                parent[a] = b;
            }
        }
    }
    let mut sets: Vec<(f32, usize, Vec<usize>)> = Vec::new();
    for i in 0..n {
        let r = root(&mut parent, i);
        match sets.iter_mut().find(|s| s.1 == r) {
            Some(s) => {
                s.0 += value(i);
                s.2.push(i);
            }
            None => sets.push((value(i), r, vec![i])),
        }
    }
    sets.retain(|s| s.2.len() >= 2);
    sets.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (cid, (v, _, members)) in sets.iter().take(MAX_CLUSTERS).enumerate() {
        let entry = *members.iter().min_by_key(|&&m| plan.groups[m].from_us).unwrap_or(&members[0]);
        plan.clusters.push(((v / 8.0).min(254.0) as u8, entry as u8));
        for &m in members {
            plan.groups[m].cluster = cid as u8;
        }
    }
    let spent = start.saturating_sub(fuel());
    Ok(MapAnalysis { plan, facts, spent })
}

/// The parts that depend on our final roster: clear rounds, need, monolith round, opening split.
/// Runs on the second-to-last pick with the predicted roster (the last pick already uses ~80k of
/// its 100k in release: building theta). No SDK map calls.
pub fn finish_roster(a: &MapAnalysis, allies: &[ChampionKind]) -> MatchPlan {
    let mut plan = a.plan.clone();
    if plan.is_empty() || allies.is_empty() {
        return plan;
    }
    let kits: Vec<_> = allies.iter().map(|&k| sdk::kit(k)).collect();
    let team_dps: f32 = kits.iter().map(|k| sdk::dps(k.base_strength, k.base_focus, k)).sum();
    let mean_dps = team_dps / allies.len() as f32;
    let tank = kits.iter().map(|k| k.base_health).fold(0.0, f32::max);
    for (g, f) in plan.groups.iter_mut().zip(&a.facts) {
        let (hp, hit) = (f.hp, f.hit);
        g.clear = (hp / team_dps.max(1.0) / 4.0).ceil().min(254.0) as u8;
        g.need = if hit <= 0.0 {
            1
        } else {
            (1..=4u8).find(|&m| hit * hp / (f32::from(m) * mean_dps.max(1.0)) <= tank * 0.8).unwrap_or(5)
        };
    }
    // Monolith: the round our base roster takes one without losing the tank, assuming damage
    // grows ~25% per 100 rounds with items (rough; champions keep their live checks).
    plan.monolith_round8 = if plan.groups.iter().any(|g| g.is(class::MONOLITH)) {
        let (hp, hit) = (20_000.0f32, 200.0f32);
        (0..=250u16)
            .find(|&r8| {
                let grow = 1.0 + 0.25 * f32::from(r8) * 8.0 / 100.0;
                hit * hp / (team_dps * grow) <= tank * grow * 0.8 * allies.len() as f32
            })
            .map_or(NONE, |r8| r8.min(254) as u8)
    } else {
        NONE
    };

    // Opening split: bodies in slot order (champion kind rank, as bot.rs computes the slot).
    let mut bodies: Vec<ChampionKind> = allies.to_vec();
    bodies.sort_unstable_by_key(|&k| k as u8);
    let mut taken = vec![false; plan.groups.len()];
    let values: Vec<f32> = a.facts.iter().map(|f| f.gems as f32 + f.gold as f32 / 3.0).collect();
    let value = |i: usize| values[i];
    let solo_ok = |g: &PlanGroup| g.need <= 2 && !g.is(class::MONOLITH) && !g.is(class::FRONTIER);
    for s in 0..bodies.len().min(BODIES) {
        let mut at: Option<usize> = None;
        for k in 0..OPENING_LEN {
            let next = match at {
                // First target: best value per walk from spawn.
                None => (0..plan.groups.len())
                    .filter(|&i| !taken[i] && solo_ok(&plan.groups[i]) && plan.groups[i].from_us != UNKNOWN)
                    .max_by(|&x, &y| {
                        let sc = |i: usize| value(i) / (f32::from(plan.groups[i].from_us) + 5.0);
                        sc(x).total_cmp(&sc(y))
                    }),
                // Then the best untaken neighbour of the previous target.
                Some(p) => plan.neighbours[p]
                    .iter()
                    .filter(|&&(j, d)| j != NONE && d != UNKNOWN)
                    .map(|&(j, _)| usize::from(j))
                    .filter(|&j| !taken[j] && solo_ok(&plan.groups[j]))
                    .max_by(|&x, &y| value(x).total_cmp(&value(y))),
            };
            let Some(g) = next else { break };
            taken[g] = true;
            plan.opening[s][k] = g as u8;
            at = Some(g);
        }
    }
    plan
}

/// Last pick: the cheap parts that need the enemy's picks.
pub fn finish_enemies(plan: &mut MatchPlan, enemies: &[ChampionKind], fp: u32) {
    // Counter profile 1: the comp that beats us most (Stellar, Cassi & Issac, Varg: 0.26 win).
    let hunters = enemies
        .iter()
        .filter(|k| matches!(k, ChampionKind::Stellar | ChampionKind::Varg | ChampionKind::Cassi))
        .count();
    plan.counter = u8::from(hunters >= 2);
    // Variety seed: same map + same enemy picks = same seed (deterministic per pairing).
    let mut hsh = fp ^ 0x9E37_79B9;
    for &e in enemies {
        hsh = (hsh ^ e as u32).wrapping_mul(0x0100_0193);
    }
    plan.seed = (hsh >> 13) as u8;
}
