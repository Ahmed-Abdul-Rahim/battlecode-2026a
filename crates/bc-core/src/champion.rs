//! Champion identities, kits and the kit descriptor used to seed champion vectors.
//!
//! At runtime the bot fills [`KitInfo`] from SDK metadata (`champion_designs()`,
//! `base_stats()`, `AbilityInfo`, `passive()`), so balance patches flow through
//! automatically. [`builtin_kit`] mirrors docs/GameRules.md and is the fallback
//! and the fixture for tests.

/// Mirrors `cambc::ChampionType` (engine values 0..=11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum ChampionKind {
    Ryth = 0,
    VesperVale = 1,
    Vendetta = 2,
    Cassi = 3,
    Issac = 4,
    Stellar = 5,
    Prism = 6,
    Marcella = 7,
    Willow = 8,
    Ren = 9,
    Varg = 10,
    Tavi = 11,
    /// Engine 1.0.0: melee, damage grows with missing health, shield + stunning dash.
    HeartOfRuin = 12,
}

impl ChampionKind {
    pub const ALL: [Self; 13] = [
        Self::Ryth,
        Self::VesperVale,
        Self::Vendetta,
        Self::Cassi,
        Self::Issac,
        Self::Stellar,
        Self::Prism,
        Self::Marcella,
        Self::Willow,
        Self::Ren,
        Self::Varg,
        Self::Tavi,
        Self::HeartOfRuin,
    ];

    #[must_use]
    pub fn from_u8(v: u8) -> Option<Self> {
        Self::ALL.get(v as usize).copied()
    }

    /// The draft choice that puts this champion on the team.
    #[must_use]
    pub fn draft_choice(self) -> DraftChoice {
        match self {
            Self::Ryth => DraftChoice::Ryth,
            Self::VesperVale => DraftChoice::VesperVale,
            Self::Vendetta => DraftChoice::Vendetta,
            Self::Cassi | Self::Issac => DraftChoice::CassiAndIssac,
            Self::Stellar => DraftChoice::Stellar,
            Self::Prism => DraftChoice::Prism,
            Self::Marcella => DraftChoice::Marcella,
            Self::Willow => DraftChoice::Willow,
            Self::Ren => DraftChoice::Ren,
            Self::Varg => DraftChoice::Varg,
            Self::Tavi => DraftChoice::Tavi,
            Self::HeartOfRuin => DraftChoice::HeartOfRuin,
        }
    }
}

/// Mirrors `cambc::ChampionDraft` (engine values 0..=11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum DraftChoice {
    Ryth = 0,
    VesperVale = 1,
    Vendetta = 2,
    CassiAndIssac = 3,
    Stellar = 4,
    Prism = 5,
    Marcella = 6,
    Willow = 7,
    Ren = 8,
    Varg = 9,
    Tavi = 10,
    HeartOfRuin = 11,
}

impl DraftChoice {
    pub const ALL: [Self; 12] = [
        Self::Ryth,
        Self::VesperVale,
        Self::Vendetta,
        Self::CassiAndIssac,
        Self::Stellar,
        Self::Prism,
        Self::Marcella,
        Self::Willow,
        Self::Ren,
        Self::Varg,
        Self::Tavi,
        Self::HeartOfRuin,
    ];

    #[must_use]
    pub fn from_u8(v: u8) -> Option<Self> {
        Self::ALL.get(v as usize).copied()
    }

    /// Champions this choice adds (Cassi & Issac is one pick, two bodies).
    #[must_use]
    pub fn champions(self) -> &'static [ChampionKind] {
        match self {
            Self::Ryth => &[ChampionKind::Ryth],
            Self::VesperVale => &[ChampionKind::VesperVale],
            Self::Vendetta => &[ChampionKind::Vendetta],
            Self::CassiAndIssac => &[ChampionKind::Cassi, ChampionKind::Issac],
            Self::Stellar => &[ChampionKind::Stellar],
            Self::Prism => &[ChampionKind::Prism],
            Self::Marcella => &[ChampionKind::Marcella],
            Self::Willow => &[ChampionKind::Willow],
            Self::Ren => &[ChampionKind::Ren],
            Self::Varg => &[ChampionKind::Varg],
            Self::Tavi => &[ChampionKind::Tavi],
            Self::HeartOfRuin => &[ChampionKind::HeartOfRuin],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttackType {
    Melee,
    Ranged,
}

/// Mirrors `cambc::AbilityKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbilityKind {
    AttackReset,
    Damage,
    AoeDamage,
    Cc,
    Heal,
    Dash,
    Shield,
}

/// Target categories, bit-compatible with `cambc::AbilityTargetMask`.
pub mod targets {
    pub const CAMPS: u8 = 1 << 0;
    pub const ENEMIES: u8 = 1 << 1;
    pub const ALLIES: u8 = 1 << 2;
    pub const SELF: u8 = 1 << 3;
    pub const GROUND: u8 = 1 << 4;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbilitySpec {
    pub kind: AbilityKind,
    /// Focus multiplier for damage, heal or shield; 0 when not applicable.
    pub modifier: f32,
    /// Targeting range in tiles; 0 for self/attack-reset abilities.
    pub range: f32,
    /// Area radius; 0 for single-target.
    pub radius: f32,
    /// Stun or shield duration in rounds; 0 if none.
    pub duration: i32,
    pub excludes_walking: bool,
    pub targets: u8,
}

/// Mirrors `cambc::PassiveType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Passive {
    None,
    BonusGold,
    RemoteShop,
    GemFocus,
    SplashAttack,
    HealthStrength,
    ResetCooldownsOnKill,
    Retaliation,
    StunImmunity,
    /// Heart of Ruin: up to +50% damage, linear in missing health.
    MissingHealthDamage,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KitInfo {
    pub kind: ChampionKind,
    pub attack: AttackType,
    pub attack_range: i32,
    pub base_health: f32,
    pub health_growth: f32,
    pub base_strength: f32,
    pub strength_growth: f32,
    pub base_focus: f32,
    pub focus_growth: f32,
    pub abilities: [Option<AbilitySpec>; 2],
    pub passive: Passive,
}

/// Length of [`KitInfo::descriptor`].
pub const DESCRIPTOR_LEN: usize = 24;

impl KitInfo {
    /// Normalised kit features (roughly 0..1). Dimensions:
    ///
    /// | idx | meaning |
    /// |---|---|
    /// | 0–5 | base health, health growth, strength, strength growth, focus, focus growth |
    /// | 6–7 | ranged flag, basic attack range |
    /// | 8–14 | ability kind counts: reset, damage, aoe, cc, heal, dash, shield |
    /// | 15 | burst: summed damage modifiers × base focus |
    /// | 16 | longest ability range |
    /// | 17 | fraction of abilities that exclude walking |
    /// | 18–23 | passive class: economy, scaling, offensive, defensive, linked body, none |
    #[must_use]
    pub fn descriptor(&self) -> [f32; DESCRIPTOR_LEN] {
        let mut d = [0.0f32; DESCRIPTOR_LEN];
        d[0] = self.base_health / 1200.0;
        d[1] = self.health_growth / 100.0;
        d[2] = self.base_strength / 70.0;
        d[3] = self.strength_growth / 5.0;
        d[4] = self.base_focus / 105.0;
        d[5] = self.focus_growth / 15.0;
        d[6] = f32::from(u8::from(self.attack == AttackType::Ranged));
        d[7] = self.attack_range as f32 / 3.0;
        let mut burst = 0.0;
        let mut max_range = 0.0f32;
        let mut exclusive = 0.0;
        let mut count = 0.0;
        for a in self.abilities.iter().flatten() {
            count += 1.0;
            let slot = match a.kind {
                AbilityKind::AttackReset => 8,
                AbilityKind::Damage => 9,
                AbilityKind::AoeDamage => 10,
                AbilityKind::Cc => 11,
                AbilityKind::Heal => 12,
                AbilityKind::Dash => 13,
                AbilityKind::Shield => 14,
            };
            d[slot] += 1.0;
            if matches!(a.kind, AbilityKind::Damage | AbilityKind::AoeDamage | AbilityKind::Cc) {
                burst += a.modifier * self.base_focus;
            }
            max_range = max_range.max(a.range);
            if a.excludes_walking {
                exclusive += 1.0;
            }
        }
        d[15] = burst / 300.0;
        d[16] = max_range / 4.0;
        d[17] = if count > 0.0 { exclusive / count } else { 0.0 };
        let class = match self.passive {
            Passive::BonusGold | Passive::RemoteShop => 18,
            Passive::GemFocus | Passive::HealthStrength => 19,
            Passive::SplashAttack
            | Passive::ResetCooldownsOnKill
            | Passive::Retaliation
            | Passive::MissingHealthDamage => 20,
            Passive::StunImmunity => 21,
            Passive::None if matches!(self.kind, ChampionKind::Cassi | ChampionKind::Issac) => 22,
            Passive::None => 23,
        };
        d[class] = 1.0;
        d
    }

    /// Stats after buying `shards` matching shards of each stat (health, strength, focus).
    #[must_use]
    pub fn stats_with(&self, shards: [u32; 3]) -> (f32, f32, f32) {
        (
            self.base_health + self.health_growth * shards[0] as f32,
            self.base_strength + self.strength_growth * shards[1] as f32,
            self.base_focus + self.focus_growth * shards[2] as f32,
        )
    }
}

fn ability(
    kind: AbilityKind,
    modifier: f32,
    range: f32,
    radius: f32,
    duration: i32,
    excludes_walking: bool,
    targets: u8,
) -> Option<AbilitySpec> {
    Some(AbilitySpec { kind, modifier, range, radius, duration, excludes_walking, targets })
}

/// Kit table transcribed from docs/GameRules.md (engine 2.0.0, cambc 4.0.0 balance patch). The bot overlays
/// the engine's live base stats at start-up (bots/main/src/sdk.rs), so this is the fallback.
#[must_use]
pub fn builtin_kit(kind: ChampionKind) -> KitInfo {
    use targets::{ALLIES, CAMPS, ENEMIES, GROUND, SELF};
    use AbilityKind::{AoeDamage, AttackReset, Cc, Damage, Dash, Heal, Shield};
    use AttackType::{Melee, Ranged};
    let foe = CAMPS | ENEMIES;
    let reset = ability(AttackReset, 0.0, 0.0, 0.0, 0, false, SELF);
    let dash = ability(Dash, 0.0, 2.0, 0.0, 0, false, GROUND);
    let k = |attack, range, h, hg, s, sg, f, fg, a0, a1, passive| KitInfo {
        kind,
        attack,
        attack_range: range,
        base_health: h,
        health_growth: hg,
        base_strength: s,
        strength_growth: sg,
        base_focus: f,
        focus_growth: fg,
        abilities: [a0, a1],
        passive,
    };
    match kind {
        ChampionKind::Cassi => k(
            Ranged,
            3,
            600.0,
            65.0,
            30.0,
            3.0,
            100.0,
            13.0,
            ability(Damage, 2.0, 4.0, 0.0, 0, true, foe),
            ability(Cc, 0.0, 4.0, 0.0, 1, false, foe),
            Passive::None,
        ),
        ChampionKind::Issac => {
            k(Melee, 1, 900.0, 80.0, 45.0, 3.0, 100.0, 10.0, reset, dash, Passive::None)
        }
        ChampionKind::Ryth => k(
            Melee,
            1,
            1200.0,
            100.0,
            45.0,
            4.5,
            100.0,
            10.0,
            ability(AoeDamage, 1.0, 4.0, 1.0, 1, true, GROUND),
            reset,
            Passive::HealthStrength,
        ),
        ChampionKind::VesperVale => {
            k(Ranged, 3, 900.0, 100.0, 45.0, 5.0, 100.0, 10.0, reset, dash, Passive::BonusGold)
        }
        ChampionKind::Prism => k(
            Ranged,
            3,
            700.0,
            65.0,
            35.0,
            2.5,
            80.0,
            8.0,
            ability(Cc, 1.5, 4.0, 0.0, 1, true, foe),
            ability(AoeDamage, 1.0, 4.0, 1.0, 0, true, GROUND),
            Passive::GemFocus,
        ),
        ChampionKind::Stellar => k(
            Ranged,
            3,
            750.0,
            75.0,
            30.0,
            3.5,
            85.0,
            8.5,
            ability(Damage, 2.0, 4.0, 0.0, 0, true, foe),
            ability(AoeDamage, 1.4, 4.0, 1.0, 0, true, GROUND),
            Passive::None,
        ),
        ChampionKind::Vendetta => k(
            Melee,
            1,
            800.0,
            100.0,
            70.0,
            6.0,
            100.0,
            10.0,
            reset,
            dash,
            Passive::ResetCooldownsOnKill,
        ),
        ChampionKind::Marcella => k(
            Melee,
            1,
            1200.0,
            100.0,
            45.0,
            4.5,
            60.0,
            14.0,
            reset,
            ability(Damage, 2.0, 4.0, 0.0, 0, true, foe),
            Passive::StunImmunity,
        ),
        ChampionKind::Tavi => k(
            Ranged,
            3,
            750.0,
            75.0,
            45.0,
            5.0,
            100.0,
            10.0,
            reset,
            ability(Cc, 1.6, 4.0, 0.0, 1, true, foe),
            Passive::RemoteShop,
        ),
        ChampionKind::Willow => k(
            Ranged,
            3,
            700.0,
            50.0,
            55.0,
            3.0,
            140.0,
            5.0,
            ability(Heal, 3.0, 4.0, 0.0, 0, false, ALLIES),
            reset,
            Passive::None,
        ),
        ChampionKind::Ren => k(
            Melee,
            1,
            900.0,
            100.0,
            35.0,
            5.0,
            100.0,
            9.0,
            ability(Cc, 1.1, 4.0, 0.0, 1, true, foe),
            dash,
            Passive::Retaliation,
        ),
        ChampionKind::Varg => k(
            Melee,
            1,
            1000.0,
            100.0,
            45.0,
            5.0,
            100.0,
            10.0,
            reset,
            ability(Shield, 2.5, 0.0, 0.0, 5, false, SELF),
            Passive::SplashAttack,
        ),
        ChampionKind::HeartOfRuin => k(
            Melee,
            1,
            1000.0,
            100.0,
            40.0,
            3.0,
            100.0,
            10.0,
            ability(Shield, 1.5, 0.0, 0.0, 5, false, SELF),
            // Dash 2 tiles, 1.5 x Focus to radius 1 on landing, one-turn stun.
            ability(Cc, 1.5, 2.0, 1.0, 1, false, GROUND),
            Passive::MissingHealthDamage,
        ),
    }
}

/// Heart of Ruin's passive: damage x (1 + 0.5 x missing health fraction).
#[must_use]
pub fn missing_health_bonus(kind: ChampionKind, hp_frac: f32) -> f32 {
    if kind == ChampionKind::HeartOfRuin {
        1.0 + 0.5 * (1.0 - hp_frac.clamp(0.0, 1.0))
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_choices_cover_every_champion_once() {
        let mut seen = Vec::new();
        for choice in DraftChoice::ALL {
            seen.extend_from_slice(choice.champions());
        }
        seen.sort();
        assert_eq!(seen, ChampionKind::ALL.to_vec());
        for kind in ChampionKind::ALL {
            assert!(kind.draft_choice().champions().contains(&kind));
        }
    }

    #[test]
    fn engine_values_round_trip() {
        for kind in ChampionKind::ALL {
            assert_eq!(ChampionKind::from_u8(kind as u8), Some(kind));
        }
        for choice in DraftChoice::ALL {
            assert_eq!(DraftChoice::from_u8(choice as u8), Some(choice));
        }
        assert_eq!(ChampionKind::from_u8(13), None);
        assert_eq!(DraftChoice::from_u8(12), None);
    }

    #[test]
    fn descriptors_are_normalised_and_distinct() {
        let descriptors: Vec<_> =
            ChampionKind::ALL.iter().map(|&k| builtin_kit(k).descriptor()).collect();
        for d in &descriptors {
            assert!(d.iter().all(|v| (0.0..=2.0).contains(v)), "{d:?}");
            assert_eq!(d[18..].iter().filter(|&&v| v == 1.0).count(), 1, "one passive class");
        }
        for i in 0..descriptors.len() {
            for j in i + 1..descriptors.len() {
                assert_ne!(descriptors[i], descriptors[j], "{i} vs {j}");
            }
        }
    }

    #[test]
    fn kit_table_matches_rules_examples() {
        let stellar = builtin_kit(ChampionKind::Stellar);
        assert_eq!(stellar.base_focus, 85.0);
        assert_eq!(stellar.abilities[0].unwrap().modifier, 2.0, "2× focus burst = 170 damage");
        let ryth = builtin_kit(ChampionKind::Ryth);
        assert_eq!(ryth.stats_with([2, 0, 0]).0, 1400.0);
        assert_eq!(builtin_kit(ChampionKind::Varg).passive, Passive::SplashAttack);
        assert_eq!(builtin_kit(ChampionKind::Tavi).abilities[1].unwrap().modifier, 1.6, "engine 2.0.0 patch");
        assert_eq!(missing_health_bonus(ChampionKind::HeartOfRuin, 0.5), 1.25);
        assert_eq!(missing_health_bonus(ChampionKind::Varg, 0.1), 1.0);
    }
}
