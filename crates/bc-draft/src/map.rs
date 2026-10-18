//! What the draft knows about the map.

use bc_core::Pos;

/// Mirrors `cambc::CampType`, with the stats from docs/GameRules.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CampKind {
    Wolf,
    Elk,
    GemCluster,
    GoldCluster,
    Monolith,
}

impl CampKind {
    #[must_use]
    pub fn from_engine(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Wolf),
            1 => Some(Self::Elk),
            2 => Some(Self::GemCluster),
            3 => Some(Self::GoldCluster),
            4 => Some(Self::Monolith),
            _ => None,
        }
    }

    #[must_use]
    pub fn health(self) -> f32 {
        match self {
            Self::Wolf => 1350.0,
            Self::Elk => 2400.0,
            Self::GemCluster | Self::GoldCluster => 500.0,
            Self::Monolith => 20000.0,
        }
    }

    #[must_use]
    pub fn strength(self) -> f32 {
        match self {
            Self::Wolf => 20.0,
            Self::Elk => 60.0,
            Self::GemCluster | Self::GoldCluster => 0.0,
            Self::Monolith => 200.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupInfo {
    pub camps: Vec<CampKind>,
    pub center: Pos,
    /// Path distance from our spawn (None if unknown/unreachable).
    pub dist_from_spawn: Option<u16>,
    pub gems: u32,
    pub gold: u32,
    pub max_respawn: u32,
}

impl GroupInfo {
    #[must_use]
    pub fn total_health(&self) -> f32 {
        self.camps.iter().map(|c| c.health()).sum()
    }

    #[must_use]
    pub fn max_strength(&self) -> f32 {
        self.camps.iter().map(|c| c.strength()).fold(0.0, f32::max)
    }

    #[must_use]
    pub fn has_monolith(&self) -> bool {
        self.camps.contains(&CampKind::Monolith)
    }

    /// Reward in gem-equivalents (gold buys power, 3 gold ≈ 1 gem).
    #[must_use]
    pub fn value(&self) -> f32 {
        self.gems as f32 + self.gold as f32 / 3.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MapInfo {
    pub width: i32,
    pub height: i32,
    pub groups: Vec<GroupInfo>,
    pub n_altars: u32,
    /// Fraction of tiles that are holes / bushes.
    pub hole_frac: f32,
    pub bush_frac: f32,
}

/// Normalised map summary (written into θ_match's map cache).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapDescriptor {
    pub size: f32,
    pub n_groups: f32,
    pub monolith: f32,
    /// Fraction of groups with several camps (splash/AoE value).
    pub clustered: f32,
    pub elk: f32,
    pub clusters: f32,
    pub mean_dist: f32,
    pub holes: f32,
    pub bushes: f32,
    pub altars: f32,
}

impl MapDescriptor {
    #[must_use]
    pub fn from_map(m: &MapInfo) -> Self {
        let n = m.groups.len().max(1) as f32;
        let camps: Vec<CampKind> = m.groups.iter().flat_map(|g| g.camps.iter().copied()).collect();
        let n_camps = camps.len().max(1) as f32;
        let count = |k: CampKind| camps.iter().filter(|&&c| c == k).count() as f32;
        let dists: Vec<f32> =
            m.groups.iter().filter_map(|g| g.dist_from_spawn.map(f32::from)).collect();
        let mean_dist = if dists.is_empty() {
            0.5
        } else {
            dists.iter().sum::<f32>() / dists.len() as f32 / 100.0
        };
        Self {
            size: (m.width * m.height) as f32 / 10000.0,
            n_groups: n / 30.0,
            monolith: f32::from(u8::from(m.groups.iter().any(GroupInfo::has_monolith))),
            clustered: m.groups.iter().filter(|g| g.camps.len() >= 2).count() as f32 / n,
            elk: count(CampKind::Elk) / n_camps,
            clusters: (count(CampKind::GemCluster) + count(CampKind::GoldCluster)) / n_camps,
            mean_dist,
            holes: m.hole_frac,
            bushes: m.bush_frac,
            altars: m.n_altars as f32 / 8.0,
        }
    }

    #[must_use]
    pub fn to_array(self) -> [f32; 10] {
        [
            self.size,
            self.n_groups,
            self.monolith,
            self.clustered,
            self.elk,
            self.clusters,
            self.mean_dist,
            self.holes,
            self.bushes,
            self.altars,
        ]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn sample_map() -> MapInfo {
        let g = |camps: Vec<CampKind>, d: u16, gems, gold| GroupInfo {
            camps,
            center: Pos::new(d as i32, 0),
            dist_from_spawn: Some(d),
            gems,
            gold,
            max_respawn: 40,
        };
        MapInfo {
            width: 60,
            height: 60,
            groups: vec![
                g(vec![CampKind::Wolf, CampKind::Wolf], 12, 44, 180),
                g(vec![CampKind::GemCluster], 8, 24, 0),
                g(vec![CampKind::Elk], 30, 36, 160),
                g(vec![CampKind::Monolith], 40, 200, 300),
            ],
            n_altars: 2,
            hole_frac: 0.05,
            bush_frac: 0.1,
        }
    }

    #[test]
    fn descriptor_summarises_map() {
        let d = MapDescriptor::from_map(&sample_map());
        assert!((d.size - 0.36).abs() < 1e-6);
        assert_eq!(d.monolith, 1.0);
        assert!((d.clustered - 0.25).abs() < 1e-6);
        assert!((d.elk - 0.2).abs() < 1e-6, "1 elk of 5 camps");
        assert!((d.mean_dist - 0.225).abs() < 1e-6);
    }

    #[test]
    fn group_value_and_health() {
        let m = sample_map();
        assert_eq!(m.groups[0].total_health(), 2700.0);
        assert_eq!(m.groups[3].max_strength(), 200.0);
        assert!((m.groups[0].value() - (44.0 + 60.0)).abs() < 1e-6);
    }
}
