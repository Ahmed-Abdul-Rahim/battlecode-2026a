//! A fixed square window around the champion for guest-side search.

use bc_core::Pos;

/// 11×11 = 121 cells. Each cell may cost one `DistanceField::get` (~74 fuel)
/// in the weighted tier, so the window size bounds that tier at ~9k fuel.
pub const SIDE: i32 = 11;
pub const CELLS: usize = (SIDE * SIDE) as usize;
/// Cost marking an impassable cell.
pub const BLOCKED: u16 = u16::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub origin: Pos,
}

impl Window {
    /// Window centred on `center`, clamped to the map so it never hangs off an edge
    /// (on maps at least `SIDE` wide).
    #[must_use]
    pub fn around(center: Pos, map_width: i32, map_height: i32) -> Self {
        let half = SIDE / 2;
        let clamp = |v: i32, size: i32| (v - half).clamp(0, (size - SIDE).max(0));
        Self { origin: Pos::new(clamp(center.x, map_width), clamp(center.y, map_height)) }
    }

    #[must_use]
    pub fn contains(&self, p: Pos) -> bool {
        let d = p - self.origin;
        d.x >= 0 && d.y >= 0 && d.x < SIDE && d.y < SIDE
    }

    #[must_use]
    pub fn index(&self, p: Pos) -> Option<usize> {
        self.contains(p).then(|| {
            let d = p - self.origin;
            (d.y * SIDE + d.x) as usize
        })
    }

    #[must_use]
    pub fn pos(&self, index: usize) -> Pos {
        self.origin + Pos::new(index as i32 % SIDE, index as i32 / SIDE)
    }

    #[must_use]
    pub fn on_edge(&self, p: Pos) -> bool {
        let d = p - self.origin;
        self.contains(p) && (d.x == 0 || d.y == 0 || d.x == SIDE - 1 || d.y == SIDE - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_clamps_to_map_and_indexes_round_trip() {
        let w = Window::around(Pos::new(2, 98), 100, 100);
        assert_eq!(w.origin, Pos::new(0, 100 - SIDE));
        for i in [0, 7, CELLS / 2, CELLS - 1] {
            assert_eq!(w.index(w.pos(i)), Some(i));
        }
        assert!(w.on_edge(Pos::new(0, 95)));
        assert!(!w.on_edge(Pos::new(5, 95)));
        assert_eq!(w.index(Pos::new(SIDE, 100 - SIDE)), None);
        let centred = Window::around(Pos::new(50, 50), 100, 100);
        assert_eq!(centred.origin, Pos::new(50 - SIDE / 2, 50 - SIDE / 2));
    }
}
