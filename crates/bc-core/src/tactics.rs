//! Small pure decisions used by bot behaviours (knob2 UNSTICK, ALTAR_LOCK 2), tested here.

use crate::geom::Pos;

/// The recent positions cover at most two tiles: walking back and forth without progress.
#[must_use]
pub fn two_tile_loop(recent: &[Pos]) -> bool {
    let mut seen: Vec<Pos> = Vec::with_capacity(3);
    for &p in recent {
        if !seen.contains(&p) {
            seen.push(p);
            if seen.len() > 2 {
                return false;
            }
        }
    }
    recent.len() >= 4 && seen.len() == 2
}

/// An altar about as close to the enemy spawn as to ours: both teams will use it.
#[must_use]
pub fn contested(dist_ours: i32, dist_theirs: i32, slack: i32) -> bool {
    (dist_ours - dist_theirs).abs() <= slack
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_needs_exactly_two_tiles_back_and_forth() {
        let a = Pos::new(3, 3);
        let b = Pos::new(4, 3);
        assert!(two_tile_loop(&[a, b, a, b, a, b, a, b]));
        assert!(!two_tile_loop(&[a; 8]), "standing still (waiting) is not a loop");
        assert!(!two_tile_loop(&[a, b, Pos::new(5, 3), b, a, b, a, b]), "made progress");
        assert!(!two_tile_loop(&[a, b]), "too short to tell");
    }

    #[test]
    fn contested_altars() {
        assert!(contested(10, 12, 3));
        assert!(!contested(5, 20, 3));
    }
}
