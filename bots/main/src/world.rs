//! Static map knowledge built once, plus the distance-field cache (tier 1 of
//! the pathing ensemble). Host `DistanceField::compute` costs ~19k fuel on a
//! 100×100 map, so fields are computed lazily and on a precompute queue that
//! uses spare fuel; lookups cost ~74 fuel per tile.

use crate::bc_core::geom::DIRS8;
use crate::bc_core::grid::BitGrid;
use crate::bc_core::{Pos, Terrain};
use crate::bc_draft::CampKind;
use crate::bc_path::{DistanceOracle, Goal};
use crate::sdk::{self, pos, vec2};
use cambc::{Bitboard, CampGroup, DistanceField, Result, Team};
use std::collections::VecDeque;

/// An altar this close to the enemy spawn is skipped by SAFE_ALTAR (if another exists).
const UNSAFE_ALTAR_SPAWN_DIST: i32 = 6;

pub struct GroupStatic {
    pub sdk: CampGroup,
    pub center: Pos,
    /// Spawn tiles (capture tiles), from the SDK's group mask.
    pub tiles: Vec<Pos>,
    pub gems: u32,
    pub gold: u32,
    pub max_respawn: u32,
    pub total_hp: f32,
    pub max_strength: f32,
    /// Sum of camp strengths: what a lone tank takes per round once the group is awake.
    pub total_strength: f32,
    pub has_monolith: bool,
    /// Other groups whose centre is within [`CLUSTER_RADIUS`] (knob2 CLUSTER).
    pub near: Vec<u8>,
}

/// Groups whose centres are this close (Chebyshev) may form a cluster; the walking distance
/// between them decides (bot.rs CLUSTER_PATH).
pub const CLUSTER_RADIUS: i32 = 10;

/// Dense tile → group index. A `HashMap<Pos, _>` costs ~150 fuel per lookup in
/// wasm (SipHash); this is a single byte read.
pub struct TileIndex {
    width: i32,
    height: i32,
    data: Vec<u8>,
}

const NO_GROUP: u8 = u8::MAX;

impl TileIndex {
    fn new(width: i32, height: i32) -> Self {
        Self { width, height, data: vec![NO_GROUP; (width * height).max(0) as usize] }
    }

    fn set(&mut self, p: Pos, group: usize) {
        if p.x >= 0 && p.y >= 0 && p.x < self.width && p.y < self.height {
            self.data[(p.y * self.width + p.x) as usize] = group.min(254) as u8;
        }
    }

    pub fn get(&self, p: Pos) -> Option<usize> {
        if p.x < 0 || p.y < 0 || p.x >= self.width || p.y >= self.height {
            return None;
        }
        let g = self.data[(p.y * self.width + p.x) as usize];
        (g != NO_GROUP).then_some(g as usize)
    }
}

pub struct World {
    pub width: i32,
    pub height: i32,
    pub terrain: Terrain,
    pub groups: Vec<GroupStatic>,
    pub tile_group: TileIndex,
    pub altars: Vec<Pos>,
    /// One own-spawn tile, as a straight-line hint before fields exist.
    pub spawn_hint: Pos,
    pub fields: FieldCache,
}

/// Map identity: FNV-1a over the size and the wall / hole / altar layers. The map is
/// fully visible from the draft, so this is exact (no estimation needed); the same
/// bytes are in the debug `M` dump, so harness/map_plans.py computes the same value.
#[must_use]
pub fn fingerprint(
    width: i32,
    height: i32,
    walls: &BitGrid,
    holes: &BitGrid,
    altars: &BitGrid,
) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    let mut eat = |b: u8| {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    };
    for v in [width as u16, height as u16] {
        v.to_le_bytes().into_iter().for_each(&mut eat);
    }
    for layer in [walls, holes, altars] {
        layer.as_bytes().iter().copied().for_each(&mut eat);
    }
    h
}

impl World {
    /// `safe_altar`: drop altars next to the enemy spawn from tribute routing when another
    /// altar exists (knob SAFE_ALTAR).
    pub fn build(safe_altar: bool) -> Result<Self> {
        let width = cambc::map::width()?;
        let height = cambc::map::height()?;
        let walls = cambc::map::walls()?;
        let holes = cambc::map::holes()?;
        let altars = cambc::map::altars()?;
        let ally_spawn = cambc::map::spawn_region(Team::Ally)?;
        let enemy_spawn = cambc::map::spawn_region(Team::Enemy)?;
        let terrain = Terrain::new(
            sdk::grid(walls, width, height)?,
            sdk::grid(holes, width, height)?,
            sdk::grid(cambc::map::bushes()?, width, height)?,
            sdk::grid(altars, width, height)?,
            sdk::grid(ally_spawn, width, height)?,
            sdk::grid(enemy_spawn, width, height)?,
        );

        let mut blocked = walls.clone();
        blocked.union_assign(holes)?;
        blocked.union_assign(altars)?;
        blocked.union_assign(enemy_spawn)?;

        let mut groups = Vec::new();
        let mut tile_group = TileIndex::new(width, height);
        let mut roots = Vec::new();
        for (i, g) in cambc::map::camp_groups()?.into_iter().enumerate() {
            let camps: Vec<(CampKind, Pos)> = g
                .camps()
                .iter()
                .map(|c| {
                    (
                        CampKind::from_engine(c.kind() as i32).unwrap_or(CampKind::Wolf),
                        pos(c.spawn_position()),
                    )
                })
                .collect();
            let spawn_mask = g.spawn_tiles()?;
            let mut tiles: Vec<Pos> = spawn_mask.positions()?.into_iter().map(pos).collect();
            for &(_, p) in &camps {
                if !tiles.contains(&p) {
                    tiles.push(p);
                }
            }
            for &t in &tiles {
                tile_group.set(t, i);
            }
            let n = tiles.len().max(1) as i32;
            let center = Pos::new(
                tiles.iter().map(|p| p.x).sum::<i32>() / n,
                tiles.iter().map(|p| p.y).sum::<i32>() / n,
            );
            roots.push(spawn_mask);
            groups.push(GroupStatic {
                gems: g.gem_value(),
                gold: g.gold_value(),
                max_respawn: g.max_respawn_timer().max(0) as u32,
                total_hp: camps.iter().map(|(k, _)| k.health()).sum(),
                max_strength: camps.iter().map(|(k, _)| k.strength()).fold(0.0, f32::max),
                total_strength: camps.iter().map(|(k, _)| k.strength()).sum(),
                has_monolith: camps.iter().any(|(k, _)| *k == CampKind::Monolith),
                near: Vec::new(),
                center,
                tiles,
                sdk: g,
            });
        }

        let centers: Vec<Pos> = groups.iter().map(|g| g.center).collect();
        for (i, g) in groups.iter_mut().enumerate() {
            g.near = (0..centers.len())
                .filter(|&j| j != i && centers[j].cheb(g.center) <= CLUSTER_RADIUS)
                .map(|j| j as u8)
                .collect();
        }

        // SDK positions(): ~600 fuel. BitGrid::positions() would scan every tile (~150k).
        let mut altar_list: Vec<Pos> = altars.positions()?.into_iter().map(pos).collect();
        if safe_altar {
            let enemy: Vec<Pos> = enemy_spawn.positions()?.into_iter().map(pos).collect();
            let near_enemy = |a: &Pos| enemy.iter().any(|e| e.cheb(*a) <= UNSAFE_ALTAR_SPAWN_DIST);
            if altar_list.iter().any(|a| !near_enemy(a)) {
                altar_list.retain(|a| !near_enemy(a));
            }
        }
        let spawn_hint = ally_spawn.positions()?.first().copied().map(pos).unwrap_or_default();
        let mut altar_roots = Bitboard::new()?;
        for &a in &altar_list {
            altar_roots.draw_chebyshev(vec2(a), cambc::constants().submit_gems_range as f32)?;
        }
        let free = blocked.complement()?;
        altar_roots.intersect_assign(&free)?;
        let mut spawn_roots = ally_spawn.clone();
        spawn_roots.intersect_assign(&free)?;

        let fields = FieldCache::new(blocked, spawn_roots, altar_roots, roots, width, height)?;
        Ok(Self {
            width,
            height,
            terrain,
            groups,
            tile_group,
            altars: altar_list,
            spawn_hint,
            fields,
        })
    }

    pub fn group_at(&self, p: Pos) -> Option<usize> {
        self.tile_group.get(p)
    }

    /// Tribute tile: Chebyshev distance ≤ submit range to some altar.
    pub fn altar_in_range(&self, p: Pos) -> Option<Pos> {
        let range = cambc::constants().submit_gems_range;
        self.altars.iter().copied().find(|a| a.cheb(p) <= range)
    }
}

/// Distance fields keyed by goal, computed on demand.
pub struct FieldCache {
    blocked: Bitboard,
    roots: Vec<Bitboard>,
    fields: Vec<Option<DistanceField>>,
    queue: VecDeque<Goal>,
    width: i32,
    height: i32,
    /// Fuel estimate for one full-map compute.
    pub compute_cost: u64,
}

fn slot(goal: Goal) -> usize {
    match goal {
        Goal::AllySpawn => 0,
        Goal::Altars => 1,
        Goal::Group(i) => 2 + i as usize,
    }
}

impl FieldCache {
    fn new(
        blocked: Bitboard,
        spawn: Bitboard,
        altars: Bitboard,
        groups: Vec<Bitboard>,
        width: i32,
        height: i32,
    ) -> Result<Self> {
        let mut roots = vec![spawn, altars];
        let n_groups = groups.len();
        roots.extend(groups);
        let mut queue = VecDeque::from([Goal::AllySpawn, Goal::Altars]);
        queue.extend((0..n_groups).map(|i| Goal::Group(i as u8)));
        Ok(Self {
            blocked,
            fields: (0..roots.len()).map(|_| None).collect(),
            roots,
            queue,
            width,
            height,
            // Measured: 19.4k for 100×100; scales roughly with tile count.
            compute_cost: 2_000 + (width * height) as u64 * 2,
        })
    }

    /// Put a goal at the front of the precompute queue.
    pub fn prioritise(&mut self, goal: Goal) {
        if self.is_ready(goal) {
            return;
        }
        self.queue.retain(|&g| g != goal);
        self.queue.push_front(goal);
    }

    pub fn compute(&mut self, goal: Goal) -> Result<()> {
        let s = slot(goal);
        if s >= self.roots.len() || self.fields[s].is_some() {
            return Ok(());
        }
        let mut field = DistanceField::new()?;
        let mut visited = Bitboard::new()?;
        field.compute(&self.blocked, &self.roots[s], &mut visited, DistanceField::UNREACHABLE)?;
        self.fields[s] = Some(field);
        self.queue.retain(|&g| g != goal);
        Ok(())
    }

    /// Compute queued fields while `can_spend(cost)` allows. Returns how many.
    pub fn precompute(&mut self, can_spend: &dyn Fn(u64) -> bool) -> Result<usize> {
        let mut done = 0;
        while let Some(&goal) = self.queue.front() {
            if !can_spend(self.compute_cost) {
                break;
            }
            self.compute(goal)?;
            self.queue.retain(|&g| g != goal);
            done += 1;
        }
        Ok(done)
    }

    fn get(&self, goal: Goal, p: Pos) -> Option<u16> {
        if p.x < 0 || p.y < 0 || p.x >= self.width || p.y >= self.height {
            return None;
        }
        let field = self.fields.get(slot(goal))?.as_ref()?;
        field.get(vec2(p)).ok().filter(|&d| d != DistanceField::UNREACHABLE)
    }
}

impl DistanceOracle for FieldCache {
    fn distance(&self, goal: Goal, p: Pos) -> Option<u16> {
        self.get(goal, p)
    }

    /// Eight `get`s (~600 fuel) beat `next_moves_from` (~1.5k).
    fn next_steps(&self, goal: Goal, p: Pos) -> Vec<Pos> {
        let Some(here) = self.get(goal, p) else { return Vec::new() };
        DIRS8
            .iter()
            .copied()
            .filter(|&d| self.get(goal, p + d).is_some_and(|n| n + 1 == here))
            .collect()
    }

    fn is_ready(&self, goal: Goal) -> bool {
        self.fields.get(slot(goal)).is_some_and(Option::is_some)
    }
}
