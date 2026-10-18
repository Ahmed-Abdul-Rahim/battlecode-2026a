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
    plan: Option<&'static crate::map_plans::MapPlan>,
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

/// Exact map identity (see world::fingerprint), from the same grids the champions build.
fn map_fingerprint() -> Result<u32> {
    let (w, h) = (cambc::map::width()?, cambc::map::height()?);
    let walls = sdk::grid(cambc::map::walls()?, w, h)?;
    let holes = sdk::grid(cambc::map::holes()?, w, h)?;
    let altars = sdk::grid(cambc::map::altars()?, w, h)?;
    Ok(crate::world::fingerprint(w, h, &walls, &holes, &altars))
}

fn build_cache() -> Result<DraftCache> {
    let map = map_info()?;
    let fp = map_fingerprint()?;
    let plan = crate::map_plans::lookup(fp);
    cambc::debug_block! {
        println!("F {{\"fingerprint\":{fp},\"plan\":\"{}\"}}", plan.map_or("-", |p| p.name));
    }
    Ok(DraftCache {
        desc: MapDescriptor::from_map(&map),
        map,
        team_size: cambc::draft::team_size()?,
        vecs: ChampionKind::ALL.map(|k| from_kit(&sdk::kit(k))),
        weights: ScorerWeights::default(),
        plan,
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
        let mut prefs = draft_prefs(c.plan.map_or(0, |p| p.draft), &state);
        if prefs.is_empty() && default_knob(crate::bc_core::memory::knob::DRAFT_BEST) > 0 {
            prefs = BEST.to_vec();
        } else if prefs.is_empty() && meta_draft() {
            prefs = META.to_vec();
        }
        let forced = prefs.iter().copied().find(|ch| !state.ally.contains(ch));
        if let Some(choice) = forced.or_else(|| best_pick(&c.desc, &state, &vec_of, &c.weights)) {
            cambc::draft::pick(sdk::draft_choice(choice)?)?;
            state.ally.push(choice);
        }
        let first = state.ally.len() == 1;
        let last = state.ally.len() >= c.team_size;
        if first || last {
            let allies = kinds(&state.ally);
            let enemies = kinds(&state.enemy);
            let mut theta =
                build_theta(&c.map, c.team_size, &allies, &enemies, &vec_of, team_dps(&allies));
            for &(k, v) in c.plan.map_or(&[][..], |p| p.knobs) {
                if let Some(slot) = theta.knobs.get_mut(k) {
                    *slot = v;
                }
            }
            cambc::draft_memory().submit(&theta.encode())?;
        }
        Ok(())
    })
}

/// Ladder meta (every team that beat v5): Varg, field-shopping Tavi, Cassi & Issac (two
/// bodies for one pick: 4 champions on 3-pick maps), then Stellar and Prism.
const META: [DraftChoice; 5] = [
    DraftChoice::Varg,
    DraftChoice::Tavi,
    DraftChoice::CassiAndIssac,
    DraftChoice::Stellar,
    DraftChoice::Prism,
];

/// The best ladder bot's draft (best_bot_dump/): two Power-Focus casters, the double pick,
/// then Ryth and Varg. 3-pick maps get Stellar, Ren, Cassi & Issac.
const BEST: [DraftChoice; 5] = [
    DraftChoice::Stellar,
    DraftChoice::Ren,
    DraftChoice::CassiAndIssac,
    DraftChoice::Ryth,
    DraftChoice::Varg,
];

fn default_knob(index: usize) -> u8 {
    crate::bc_core::memory::ThetaMatch::default().knob(index)
}

fn meta_draft() -> bool {
    crate::bc_core::memory::ThetaMatch::default().knob(crate::bc_core::memory::knob::TAVI_META) > 0
}

const RANGED: [DraftChoice; 5] = [
    DraftChoice::Stellar,
    DraftChoice::Prism,
    DraftChoice::VesperVale,
    DraftChoice::Tavi,
    DraftChoice::Willow,
];
const MELEE: [DraftChoice; 5] = [
    DraftChoice::Varg,
    DraftChoice::Vendetta,
    DraftChoice::Ren,
    DraftChoice::Ryth,
    DraftChoice::Marcella,
];

/// Preferred picks for a map plan's draft profile (empty = the scorer decides).
fn draft_prefs(profile: u8, state: &DraftState) -> Vec<DraftChoice> {
    match profile {
        1 => RANGED.to_vec(),
        2 => MELEE.to_vec(),
        3 => vec![DraftChoice::CassiAndIssac],
        // RedDragon's draft (ladder): bruisers plus Tavi and Willow.
        5 => vec![
            DraftChoice::Varg,
            DraftChoice::Ryth,
            DraftChoice::Tavi,
            DraftChoice::Ren,
            DraftChoice::Willow,
        ],
        // Ladder mimics (teams that beat v5, runs/ranked_v5).
        6 => vec![
            DraftChoice::Stellar,
            DraftChoice::Tavi,
            DraftChoice::Varg,
            DraftChoice::Prism,
            DraftChoice::Vendetta,
        ],
        7 => vec![
            DraftChoice::Vendetta,
            DraftChoice::Varg,
            DraftChoice::Ryth,
            DraftChoice::Willow,
            DraftChoice::CassiAndIssac,
        ],
        4 => {
            // Counter: they lean ranged -> divers; they lean melee -> kiters.
            let n_ranged = state.enemy.iter().filter(|ch| RANGED.contains(ch)).count();
            let n_melee = state.enemy.len() - n_ranged;
            if n_ranged >= 2 && n_ranged > n_melee {
                MELEE.to_vec()
            } else if n_melee >= 2 && n_melee > n_ranged {
                RANGED.to_vec()
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}
