//! Draft turns: score champions for this map with bc-draft, pick, then write
//! θ_match (camp priority, knobs, champion vectors) to draft memory.
//!
//! Fuel: the first draft turn has 10× fuel (1M) and pays for map analysis once;
//! later picks get only 100k, so everything reusable is cached and θ_match is
//! (re)written only on the first and last pick (~40k each).

use crate::bc_brain::champvec::{from_kit, ChampVec};
use crate::bc_core::champion::{ChampionKind, DraftChoice};
use crate::bc_draft::{
    best_pick, build_theta, CampKind, DraftState, GroupInfo, MapDescriptor, MapInfo, ScorerWeights,
};
use crate::sdk::{self, pos, vec2};
use cambc::{Bitboard, DistanceField, Result, Team};
use std::cell::RefCell;

struct DraftCache {
    map: MapInfo,
    desc: MapDescriptor,
    team_size: usize,
    vecs: [ChampVec; 12],
    weights: ScorerWeights,
}

thread_local! {
    static CACHE: RefCell<Option<DraftCache>> = const { RefCell::new(None) };
}

fn map_info() -> Result<MapInfo> {
    let width = cambc::map::width()?;
    let height = cambc::map::height()?;
    let cells = (width * height).max(1) as f32;
    let mut blocked = cambc::map::walls()?.clone();
    blocked.union_assign(cambc::map::holes()?)?;
    blocked.union_assign(cambc::map::altars()?)?;
    blocked.union_assign(cambc::map::spawn_region(Team::Enemy)?)?;
    let mut roots = cambc::map::spawn_region(Team::Ally)?.clone();
    roots.intersect_assign(&blocked.complement()?)?;
    let mut field = DistanceField::new()?;
    field.compute(&blocked, &roots, &mut Bitboard::new()?, DistanceField::UNREACHABLE)?;

    let mut groups = Vec::new();
    for g in cambc::map::camp_groups()? {
        let camps: Vec<CampKind> =
            g.camps().iter().filter_map(|c| CampKind::from_engine(c.kind() as i32)).collect();
        let tiles: Vec<_> = g.camps().iter().map(|c| pos(c.spawn_position())).collect();
        let n = tiles.len().max(1) as i32;
        let center = crate::bc_core::Pos::new(
            tiles.iter().map(|p| p.x).sum::<i32>() / n,
            tiles.iter().map(|p| p.y).sum::<i32>() / n,
        );
        let dist_from_spawn = tiles
            .iter()
            .filter_map(|&t| field.get(vec2(t)).ok())
            .filter(|&d| d != DistanceField::UNREACHABLE)
            .min();
        groups.push(GroupInfo {
            camps,
            center,
            dist_from_spawn,
            gems: g.gem_value(),
            gold: g.gold_value(),
            max_respawn: g.max_respawn_timer().max(0) as u32,
        });
    }
    Ok(MapInfo {
        width,
        height,
        groups,
        n_altars: cambc::map::altars()?.count()? as u32,
        hole_frac: cambc::map::holes()?.count()? as f32 / cells,
        bush_frac: cambc::map::bushes()?.count()? as f32 / cells,
    })
}

fn build_cache() -> Result<DraftCache> {
    let map = map_info()?;
    Ok(DraftCache {
        desc: MapDescriptor::from_map(&map),
        map,
        team_size: cambc::draft::team_size()?,
        vecs: ChampionKind::ALL.map(|k| from_kit(&sdk::kit(k))),
        weights: ScorerWeights::default(),
    })
}

fn picks(team: Team) -> Result<Vec<DraftChoice>> {
    Ok(cambc::draft::current_draft(team)?.into_iter().filter_map(sdk::choice).collect())
}

fn kinds(choices: &[DraftChoice]) -> Vec<ChampionKind> {
    choices.iter().flat_map(|c| c.champions().iter().copied()).collect()
}

/// Rough team damage per round, used to decide whether monoliths are viable.
fn team_dps(allies: &[ChampionKind]) -> f32 {
    allies
        .iter()
        .map(|&k| {
            let kit = sdk::kit(k);
            sdk::dps(kit.base_strength, kit.base_focus, &kit)
        })
        .sum()
}

pub fn play() -> Result<()> {
    CACHE.with(|cell| -> Result<()> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(build_cache()?);
        }
        let guard = cell.borrow();
        let c = guard.as_ref().expect("cache initialised above");
        let vec_of = |k: ChampionKind| c.vecs[k as usize];

        let mut state = DraftState {
            ally: picks(Team::Ally)?,
            enemy: picks(Team::Enemy)?,
            team_size: c.team_size,
        };
        if let Some(choice) = best_pick(&c.desc, &state, &vec_of, &c.weights) {
            cambc::draft::pick(sdk::draft_choice(choice)?)?;
            state.ally.push(choice);
        }
        let first = state.ally.len() == 1;
        let last = state.ally.len() >= c.team_size;
        if first || last {
            let allies = kinds(&state.ally);
            let enemies = kinds(&state.enemy);
            let theta =
                build_theta(&c.map, c.team_size, &allies, &enemies, &vec_of, team_dps(&allies));
            cambc::draft_memory().submit(&theta.encode())?;
        }
        Ok(())
    })
}
