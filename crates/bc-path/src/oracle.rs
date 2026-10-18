//! Distance-field abstraction. In game it's backed by SDK `DistanceField`s
//! (host-computed, ~19k fuel for a full 100×100 field); in tests by a BFS.

use bc_core::geom::DIRS8;
use bc_core::{Pos, Terrain};
use std::collections::{HashMap, VecDeque};

/// What a precomputed field leads to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Goal {
    /// Tiles from which any altar can be tributed (Chebyshev 1).
    Altars,
    /// Our spawn region (shop, heal).
    AllySpawn,
    /// Spawn tiles of camp group `i` (capture tiles; attack from nearby).
    Group(u8),
}

pub trait DistanceOracle {
    /// Static-terrain distance from `p` to `goal`, if that field is ready and reaches `p`.
    fn distance(&self, goal: Goal, p: Pos) -> Option<u16>;

    /// Optimal first steps (unit deltas) from `p` towards `goal`.
    fn next_steps(&self, goal: Goal, p: Pos) -> Vec<Pos>;

    fn is_ready(&self, goal: Goal) -> bool;
}

/// Pure-Rust oracle: BFS fields over [`Terrain`]. Used in tests and as a
/// reference for the SDK-backed oracle. Far too expensive for in-game use on
/// big maps (a full-map guest BFS overruns a 100k turn).
#[derive(Clone, Debug, Default)]
pub struct BfsOracle {
    width: i32,
    fields: HashMap<Goal, Vec<u16>>,
}

pub const UNREACHABLE: u16 = u16::MAX;

impl BfsOracle {
    #[must_use]
    pub fn new(width: i32) -> Self {
        Self { width, fields: HashMap::new() }
    }

    /// Multi-source BFS from `roots` over walkable terrain.
    pub fn add_field(&mut self, terrain: &Terrain, goal: Goal, roots: &[Pos]) {
        let (w, h) = (terrain.width(), terrain.height());
        let mut dist = vec![UNREACHABLE; (w * h) as usize];
        let mut queue = VecDeque::new();
        for &r in roots {
            if terrain.walkable(r) {
                dist[(r.y * w + r.x) as usize] = 0;
                queue.push_back(r);
            }
        }
        while let Some(p) = queue.pop_front() {
            let d = dist[(p.y * w + p.x) as usize];
            for dir in DIRS8 {
                let n = p + dir;
                if !terrain.walkable(n) {
                    continue;
                }
                let i = (n.y * w + n.x) as usize;
                if dist[i] == UNREACHABLE {
                    dist[i] = d + 1;
                    queue.push_back(n);
                }
            }
        }
        self.fields.insert(goal, dist);
    }

    fn get(&self, goal: Goal, p: Pos) -> Option<u16> {
        let field = self.fields.get(&goal)?;
        if p.x < 0 || p.y < 0 || p.x >= self.width {
            return None;
        }
        field.get((p.y * self.width + p.x) as usize).copied().filter(|&d| d != UNREACHABLE)
    }
}

impl DistanceOracle for BfsOracle {
    fn distance(&self, goal: Goal, p: Pos) -> Option<u16> {
        self.get(goal, p)
    }

    fn next_steps(&self, goal: Goal, p: Pos) -> Vec<Pos> {
        let Some(here) = self.get(goal, p) else { return Vec::new() };
        DIRS8
            .iter()
            .copied()
            .filter(|&d| self.get(goal, p + d).is_some_and(|n| n + 1 == here))
            .collect()
    }

    fn is_ready(&self, goal: Goal) -> bool {
        self.fields.contains_key(&goal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfs_routes_around_walls() {
        let t = Terrain::from_ascii(&[
            ".....", //
            ".###.", //
            ".....",
        ]);
        let mut o = BfsOracle::new(t.width());
        o.add_field(&t, Goal::AllySpawn, &[Pos::new(2, 2)]);
        // (2,0) → (1,0) → (0,1) → (1,2) → (2,2): diagonals let it hug the wall.
        assert_eq!(o.distance(Goal::AllySpawn, Pos::new(2, 0)), Some(4));
        let steps = o.next_steps(Goal::AllySpawn, Pos::new(2, 0));
        assert!(!steps.is_empty());
        for s in steps {
            let n = Pos::new(2, 0) + s;
            assert_eq!(o.distance(Goal::AllySpawn, n), Some(3));
        }
        assert!(o.is_ready(Goal::AllySpawn) && !o.is_ready(Goal::Altars));
        assert_eq!(o.distance(Goal::AllySpawn, Pos::new(2, 1)), None, "wall");
    }
}
