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
    /// Largest Chebyshev distance from the centre to a spawn tile.
    pub radius: i32,
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
    /// `cluster`: also link each group to its cluster neighbours (only knob2 CLUSTER reads them;
    /// ~40k fuel on maps with many groups).
    pub fn build(safe_altar: bool, cluster: bool) -> Result<Self> {
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

        // One pass per group, no intermediate vectors and no regrowth: start-up on maps with many
        // camps ran past the 200k first-turn budget (~83k in this loop on yellow-brick-road).
        let sdk_groups = cambc::map::camp_groups()?;
        let mut groups = Vec::with_capacity(sdk_groups.len());
        let mut tile_group = TileIndex::new(width, height);
        let mut roots = Vec::with_capacity(sdk_groups.len());
        for (i, g) in sdk_groups.into_iter().enumerate() {
            let mut tiles: Vec<Pos> = Vec::with_capacity(g.camps().len());
            let (mut total_hp, mut max_strength, mut total_strength) = (0.0f32, 0.0f32, 0.0f32);
            let mut has_monolith = false;
            for c in g.camps() {
                let k = CampKind::from_engine(c.kind() as i32).unwrap_or(CampKind::Wolf);
                let (hp, st) = (k.health(), k.strength());
                total_hp += hp;
                total_strength += st;
                max_strength = max_strength.max(st);
                has_monolith |= k == CampKind::Monolith;
                tiles.push(pos(c.spawn_position()));
            }
            // The spawn tiles are the camps' spawn positions (SDK spawn_tiles builds its mask from
            // them), sorted like Bitboard::positions(): no mask or positions() call per group.
            tiles.sort_unstable_by_key(|p| p.y * width + p.x);
            tiles.dedup();
            let (mut sx, mut sy) = (0, 0);
            for &t in &tiles {
                tile_group.set(t, i);
                sx += t.x;
                sy += t.y;
            }
            let n = tiles.len().max(1) as i32;
            let center = Pos::new(sx / n, sy / n);
            let radius = tiles.iter().map(|t| t.cheb(center)).max().unwrap_or(0);
            roots.push(tiles.clone());
            groups.push(GroupStatic {
                gems: g.gem_value(),
                gold: g.gold_value(),
                max_respawn: g.max_respawn_timer().max(0) as u32,
                total_hp,
                max_strength,
                total_strength,
                has_monolith,
                near: Vec::new(),
                radius,
                center,
                tiles,
                sdk: g,
            });
        }

        let centers: Vec<Pos> = groups.iter().map(|g| g.center).collect();
        for (i, g) in groups.iter_mut().enumerate().filter(|_| cluster) {
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
    /// Spawn region, altar tribute tiles; group masks are built when their field is computed.
    roots: Vec<Bitboard>,
    group_tiles: Vec<Vec<Pos>>,
    fields: Vec<Option<DistanceField>>,
    queue: VecDeque<Goal>,
    width: i32,
    height: i32,
    /// Fuel estimate for one full-map compute: a conservative guess until the first compute is
    /// measured, then the most expensive one measured (+1/8). The size-based guess alone was
    /// too low on big maps and turns ran out of fuel after a precompute (2026-10-23).
    pub compute_cost: u64,
    measured: bool,
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
        groups: Vec<Vec<Pos>>,
        width: i32,
        height: i32,
    ) -> Result<Self> {
        let roots = vec![spawn, altars];
        let n_groups = groups.len();
        let mut queue = VecDeque::from([Goal::AllySpawn, Goal::Altars]);
        queue.extend((0..n_groups).map(|i| Goal::Group(i as u8)));
        Ok(Self {
            blocked,
            fields: (0..roots.len() + n_groups).map(|_| None).collect(),
            roots,
            group_tiles: groups,
            queue,
            width,
            height,
            // First guess until a compute is measured: measured 19-37k (100x100 up to the biggest
            // maps). Capped: an uncapped size-based guess never fit a turn on the biggest maps, so
            // no field was ever computed (8 gems in a game, 2026-10-23).
            compute_cost: (3_000 + (width * height) as u64 * 3).min(45_000),
            measured: false,
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
        if s >= self.fields.len() || self.fields[s].is_some() {
            return Ok(());
        }
        let before = crate::sdk::fuel();
        let group_mask;
        let root = if s < self.roots.len() {
            &self.roots[s]
        } else {
            let mut m = Bitboard::new()?;
            for &t in &self.group_tiles[s - self.roots.len()] {
                m.set(vec2(t), true)?;
            }
            group_mask = m;
            &group_mask
        };
        let mut field = DistanceField::new()?;
        let mut visited = Bitboard::new()?;
        field.compute(&self.blocked, root, &mut visited, DistanceField::UNREACHABLE)?;
        self.fields[s] = Some(field);
        let spent = before.saturating_sub(crate::sdk::fuel());
        self.compute_cost =
            if self.measured { self.compute_cost.max(spent + spent / 8) } else { spent + spent / 8 };
        self.measured = true;
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
