//! Champion vectors (docs/champions.md; P-040, P-041, P-042).
//!
//! 16 int8 dims in three blocks:
//!
//! | dims | block | initial value |
//! |---|---|---|
//! | 0–5 | stats: durability, basic damage, spell power, reach, mobility, utility | projected from the kit descriptor |
//! | 6–11 | strengths / weaknesses | 0, learned from self-play |
//! | 12–15 | synergy keys | 0, learned from self-play |
//!
//! The initial projection is the model's own representation of each champion;
//! `ml/champions/finetune.py` refines all three blocks from self-play outcomes
//! (the stats block regularised towards this projection). The Python mirror in
//! `ml/champions/descriptor.py` must stay identical — a test checks parity.

use bc_core::champion::{KitInfo, DESCRIPTOR_LEN};

pub const LEN: usize = 16;
pub type ChampVec = [i8; LEN];

pub const STATS: core::ops::Range<usize> = 0..6;
pub const TRAITS: core::ops::Range<usize> = 6..12;
pub const SYNERGY: core::ops::Range<usize> = 12..16;

fn q(v: f32) -> i8 {
    (v.clamp(-1.0, 1.0) * 127.0).round() as i8
}

/// The six stats-block values in 0..1 from a kit descriptor.
#[must_use]
pub fn stats_block(d: &[f32; DESCRIPTOR_LEN]) -> [f32; 6] {
    [
        0.6 * d[0] + 0.4 * d[1],
        0.5 * d[2] + 0.3 * d[3] + 0.2 * d[8],
        0.4 * d[4] + 0.3 * d[5] + 0.3 * d[15],
        0.5 * d[6] + 0.25 * d[7] + 0.25 * d[16],
        d[13].min(1.0),
        (0.5 * d[11] + 0.5 * d[12] + 0.5 * d[14]).min(1.0),
    ]
}

/// Initial vector: stats block mapped from 0..1 to −1..1, other blocks zero.
#[must_use]
pub fn from_descriptor(d: &[f32; DESCRIPTOR_LEN]) -> ChampVec {
    let mut v = [0i8; LEN];
    for (slot, x) in v[STATS].iter_mut().zip(stats_block(d)) {
        *slot = q(2.0 * x - 1.0);
    }
    v
}

#[must_use]
pub fn from_kit(kit: &KitInfo) -> ChampVec {
    from_descriptor(&kit.descriptor())
}

/// Bilinear interaction `aᵀ M b / 127`, used for synergy (ally pairs) and
/// counters (ally vs enemy) in the draft scorer.
#[must_use]
pub fn bilinear(a: &ChampVec, m: &[[i8; LEN]; LEN], b: &ChampVec) -> i32 {
    let mut total = 0i32;
    for i in 0..LEN {
        let mut row = 0i32;
        for j in 0..LEN {
            row += i32::from(m[i][j]) * i32::from(b[j]);
        }
        total += i32::from(a[i]) * row;
    }
    total / 127
}

#[must_use]
pub fn cosine(a: &ChampVec, b: &ChampVec) -> f32 {
    let dot: i32 = a.iter().zip(b).map(|(&x, &y)| i32::from(x) * i32::from(y)).sum();
    let na: i32 = a.iter().map(|&x| i32::from(x) * i32::from(x)).sum();
    let nb: i32 = b.iter().map(|&x| i32::from(x) * i32::from(x)).sum();
    if na == 0 || nb == 0 {
        return 0.0;
    }
    dot as f32 / ((na as f32).sqrt() * (nb as f32).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bc_core::champion::{builtin_kit, ChampionKind};

    #[test]
    fn initial_vectors_reflect_kits() {
        let ryth = from_kit(&builtin_kit(ChampionKind::Ryth));
        let willow = from_kit(&builtin_kit(ChampionKind::Willow));
        let vesper = from_kit(&builtin_kit(ChampionKind::VesperVale));
        assert!(ryth[0] > willow[0], "Ryth is the tankier");
        assert!(willow[5] >= ryth[5], "Willow heals");
        assert!(vesper[3] > ryth[3], "Vesper is ranged");
        assert!(vesper[4] > willow[4], "Vesper has a dash");
        assert!(ryth[TRAITS].iter().chain(&ryth[SYNERGY]).all(|&x| x == 0));
    }

    #[test]
    fn bilinear_identity_is_scaled_dot_product() {
        let mut eye = [[0i8; LEN]; LEN];
        for (i, row) in eye.iter_mut().enumerate() {
            row[i] = 127;
        }
        let a = [10i8; LEN];
        let b = [3i8; LEN];
        assert_eq!(bilinear(&a, &eye, &b), 10 * 3 * 16);
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        assert_eq!(cosine(&a, &[0; LEN]), 0.0);
    }

    #[test]
    fn python_parity_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/champion_vectors.json"))
                .unwrap();
        for kind in ChampionKind::ALL {
            let expected: Vec<i8> = fixture[format!("{kind:?}")]
                .as_array()
                .unwrap_or_else(|| panic!("{kind:?} missing from fixture"))
                .iter()
                .map(|v| v.as_i64().unwrap() as i8)
                .collect();
            assert_eq!(from_kit(&builtin_kit(kind)).to_vec(), expected, "{kind:?}");
        }
    }
}
