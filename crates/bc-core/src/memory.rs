//! Codecs for the two team memories, both treated as model parameters.
//!
//! * **Draft memory (1024 B, θ_match):** written once by the draft bot, read-only
//!   in game. Holds per-match parameters adapted to the map: FiLM scale/shift for
//!   the learned heads (P-032), expert knobs, champion vectors, camp priority.
//! * **Shared memory (8 B, team channel):** read/write all match. Bytes 1–2 are
//!   live team strategy parameters set by the round's first actor ("leader");
//!   the rest are messages. Allies act in order within a round, so later allies
//!   see what earlier ones wrote.
//!
//! Byte layouts are documented in docs/memory-layout.md; keep them in sync.

use crate::crc::crc16;
use crate::geom::Pos;

pub const DRAFT_MEMORY_BYTES: usize = 1024;
pub const THETA_VERSION: u8 = 1;
pub const MAX_PRIORITY_GROUPS: usize = 15;
pub const FILM_UNITS: usize = 48;
pub const KNOB_COUNT: usize = 32;
pub const CHAMP_SLOTS: usize = 10;
pub const CHAMP_VEC_LEN: usize = 16;
pub const MAP_CACHE_BYTES: usize = 192;
pub const NO_CHAMPION: u8 = 0xFF;

mod off {
    pub const MAGIC: usize = 0;
    pub const VERSION: usize = 2;
    pub const TEAM_SIZE: usize = 3;
    pub const SCHEMA_HASH: usize = 4;
    pub const WEIGHTS_HASH: usize = 6;
    pub const ROLES: usize = 8;
    pub const N_PRIORITY: usize = 16;
    pub const PRIORITY: usize = 18;
    pub const FILM: usize = 48;
    pub const KNOBS: usize = 144;
    pub const CHAMP_VECS: usize = 176;
    pub const CHAMP_KINDS: usize = 336;
    pub const MAP_CACHE: usize = 352;
    pub const CRC: usize = 1022;
}

const MAGIC: [u8; 2] = *b"BC";

/// Indices into [`ThetaMatch::knobs`] and their neutral defaults.
pub mod knob {
    /// 0 = passive, 128 = neutral, 255 = all-in.
    pub const AGGRESSION: usize = 0;
    /// Retreat below this health, in percent.
    pub const RETREAT_HP_PCT: usize = 1;
    /// Don't walk to an altar for fewer gems than this unless forced.
    pub const TRIBUTE_MIN_GEMS: usize = 2;
    /// Always tribute at or above this many carried gems.
    pub const TRIBUTE_MAX_GEMS: usize = 3;
    /// Extra gems required per tile of distance to the altar, ×0.1.
    pub const TRIBUTE_DISTANCE_SCALE: usize = 4;
    /// Minimum team strength (×100 effective HP) before engaging a monolith.
    pub const MONOLITH_MIN_POWER: usize = 5;
    /// Recall to shop once gold reaches this ×10.
    pub const RECALL_GOLD: usize = 6;
    /// Required fight advantage (percent over 100) to engage an enemy champion.
    pub const FIGHT_MARGIN_PCT: usize = 7;
    /// Emblem build profile (bc_brain::shop::build_order_profile).
    pub const BUILD_PROFILE: usize = 8;
    /// Squad pairs (slots 2k, 2k+1) join each other's camp group: value ×(this/10); 0 = off.
    pub const SQUAD_JOIN_X10: usize = 9;
    /// Camp value formula: 0 gems + gold/3, 1 gems + gold/10, 2 slots 0-1 gold / rest gems,
    /// 4 gems + gold x0.6, 5 gems + gold (4/5 also prefer near camps in the opening).
    pub const VALUE_PROFILE: usize = 10;
    /// 1 = the last slot of a 4+ team farms far, quiet groups (distance weight 0.4).
    pub const REMOTE_FARMER: usize = 11;
    /// Hunt mode: 0 off, 1 always (whole team hunts and pillages), 2 triggered when far
    /// behind late or when the enemy is close to winning (only kills can stop them).
    pub const HUNT_MODE: usize = 12;
    /// 1 = stun / altar-lock / attack enemies channelling a tribute (stun, death or a new
    /// lock cancels it; damage doesn't), and carry an altar lock (10 gold).
    pub const DENY_TRIBUTE: usize = 13;
    /// 1 = bank every carried gem from ENDGAME_ROUND on (at round 2000 gems decide, then gold).
    pub const ENDGAME_BANK: usize = 14;
    /// 1 = decide skirmishes team-vs-team (Lanchester square law over all nearby allies and
    /// enemies) instead of me-vs-one-enemy: retreat from losing fights, focus-fire winning ones.
    pub const TEAM_FIGHT: usize = 15;
    /// 1 = portfolio combat search (skirmish.rs) picks the fight behaviour when enemies are close.
    pub const COMBAT_SEARCH: usize = 16;
    /// Camp safety bits: 1 = leave an awake camp while escape still beats its damage (unless
    /// we out-race it), 2 = don't wake a group a lone champion can't survive until the
    /// allies nearby can clear it (or a short wait runs out).
    pub const CAMP_SAFE: usize = 17;
    /// 1 = carry an altar lock (10 gold, bought before shards) and leave it on every altar we
    /// tribute at: it blocks only the enemy (3 of their attacks to clear).
    pub const ALTAR_LOCK: usize = 18;
    /// Ladder meta (v7): draft Varg/Tavi/Cassi&Issac/Stellar/Prism; Tavi buys shards in the
    /// field (its passive), builds Strength on every passive, and hunts once geared.
    pub const TAVI_META: usize = 19;
    /// 1 = cancel a recall when an enemy closes in (recall is a free kill otherwise);
    /// 2 = also only start one when nothing seen or remembered can reach us during it.
    pub const RECALL_SAFE: usize = 20;
    /// Fight bits: 1 focus the weakest enemy in reach and call it for the team, 2 join team
    /// fights from 50% hp, 4 finish kills / fight when outnumbering, 8 last stand.
    pub const FIGHT_V6: usize = 21;
    /// 1 = value stun / attack-reset kits (Tavi, Prism, Cassi, Ryth) on a 6-round fight horizon.
    pub const STUN_THREAT: usize = 22;
    /// 1 = discount camp groups the team can't survive yet (farm gold and gear up first).
    pub const CAMP_FEASIBLE: usize = 23;
    /// 1 = play to passives: Vesper Vale (+20/40% gold) farms gold groups until geared,
    /// Prism (+0.2 Focus per carried gem) collects gems and carries bigger bags.
    pub const PASSIVE_ROLES: usize = 24;
    /// 1 = draft like the best ladder bot: Stellar, Ren, Cassi & Issac, Ryth, Varg.
    pub const DRAFT_BEST: usize = 25;
    /// 1 = never route tributes to an altar next to the enemy spawn when another exists
    /// (enemies there make every tribute "unsafe": we looped for whole games on swirl).
    pub const SAFE_ALTAR: usize = 26;
    /// 1 = judge monolith readiness on current (itemised) damage, not base stats
    /// (the base-stat gate never opened: 0 gems on contest in 2000 rounds).
    pub const MONOLITH_LIVE: usize = 27;
    /// 1 = shop before banking: recall to spend gold even while carrying gems (recall keeps
    /// them), so items come early and one trip home does both; tribute stays bag-driven.
    pub const SHOP_FIRST: usize = 28;

    pub const DEFAULTS: [(usize, u8); 29] = [
        (AGGRESSION, 128),
        (RETREAT_HP_PCT, 35),
        (TRIBUTE_MIN_GEMS, 20),
        (TRIBUTE_MAX_GEMS, 120),
        (TRIBUTE_DISTANCE_SCALE, 10),
        (MONOLITH_MIN_POWER, 40),
        (RECALL_GOLD, 15),
        (FIGHT_MARGIN_PCT, 25),
        (BUILD_PROFILE, 2),
        (SQUAD_JOIN_X10, 0),
        (VALUE_PROFILE, 4),
        (REMOTE_FARMER, 0),
        (HUNT_MODE, 0),
        (DENY_TRIBUTE, 1),
        (ENDGAME_BANK, 0),
        (TEAM_FIGHT, 0),
        (COMBAT_SEARCH, 0),
        (CAMP_SAFE, 0),
        (ALTAR_LOCK, 0),
        (TAVI_META, 1),
        (RECALL_SAFE, 1),
        (FIGHT_V6, 0),
        (STUN_THREAT, 0),
        (CAMP_FEASIBLE, 0),
        (PASSIVE_ROLES, 0),
        (DRAFT_BEST, 0),
        (SAFE_ALTAR, 1),
        (MONOLITH_LIVE, 1),
        (SHOP_FIRST, 0),
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    Empty,
    BadMagic,
    UnsupportedVersion(u8),
    BadChecksum,
    TooShort,
}

/// Per-match parameter vector carried in draft memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThetaMatch {
    pub team_size: u8,
    pub schema_hash: u16,
    pub weights_hash: u16,
    /// Role per draft pick index.
    pub roles: [u8; 8],
    /// (camp-group index, priority score), best first. At most 15 entries.
    pub camp_priority: Vec<(u8, u8)>,
    /// FiLM (γ, β) per hidden unit. Scale is `1 + γ/64`; (0, 0) is identity.
    pub film: [(i8, i8); FILM_UNITS],
    pub knobs: [u8; KNOB_COUNT],
    /// Champion kind per slot: allies 0..5, enemies 5..10; `NO_CHAMPION` if empty.
    pub champ_kinds: [u8; CHAMP_SLOTS],
    pub champ_vecs: [[i8; CHAMP_VEC_LEN]; CHAMP_SLOTS],
    pub map_cache: [u8; MAP_CACHE_BYTES],
}

impl Default for ThetaMatch {
    fn default() -> Self {
        let mut knobs = [0u8; KNOB_COUNT];
        for (i, v) in knob::DEFAULTS {
            knobs[i] = v;
        }
        Self {
            team_size: 0,
            schema_hash: 0,
            weights_hash: 0,
            roles: [0; 8],
            camp_priority: Vec::new(),
            film: [(0, 0); FILM_UNITS],
            knobs,
            champ_kinds: [NO_CHAMPION; CHAMP_SLOTS],
            champ_vecs: [[0; CHAMP_VEC_LEN]; CHAMP_SLOTS],
            map_cache: [0; MAP_CACHE_BYTES],
        }
    }
}

impl ThetaMatch {
    #[must_use]
    pub fn encode(&self) -> [u8; DRAFT_MEMORY_BYTES] {
        let mut b = [0u8; DRAFT_MEMORY_BYTES];
        b[off::MAGIC..off::MAGIC + 2].copy_from_slice(&MAGIC);
        b[off::VERSION] = THETA_VERSION;
        b[off::TEAM_SIZE] = self.team_size;
        b[off::SCHEMA_HASH..off::SCHEMA_HASH + 2].copy_from_slice(&self.schema_hash.to_le_bytes());
        b[off::WEIGHTS_HASH..off::WEIGHTS_HASH + 2]
            .copy_from_slice(&self.weights_hash.to_le_bytes());
        b[off::ROLES..off::ROLES + 8].copy_from_slice(&self.roles);
        let n = self.camp_priority.len().min(MAX_PRIORITY_GROUPS);
        b[off::N_PRIORITY] = n as u8;
        for (i, &(group, score)) in self.camp_priority.iter().take(n).enumerate() {
            b[off::PRIORITY + 2 * i] = group;
            b[off::PRIORITY + 2 * i + 1] = score;
        }
        for (i, &(g, beta)) in self.film.iter().enumerate() {
            b[off::FILM + 2 * i] = g as u8;
            b[off::FILM + 2 * i + 1] = beta as u8;
        }
        b[off::KNOBS..off::KNOBS + KNOB_COUNT].copy_from_slice(&self.knobs);
        for (s, v) in self.champ_vecs.iter().enumerate() {
            for (j, &x) in v.iter().enumerate() {
                b[off::CHAMP_VECS + s * CHAMP_VEC_LEN + j] = x as u8;
            }
        }
        b[off::CHAMP_KINDS..off::CHAMP_KINDS + CHAMP_SLOTS].copy_from_slice(&self.champ_kinds);
        b[off::MAP_CACHE..off::MAP_CACHE + MAP_CACHE_BYTES].copy_from_slice(&self.map_cache);
        let crc = crc16(&b[..off::CRC]);
        b[off::CRC..off::CRC + 2].copy_from_slice(&crc.to_le_bytes());
        b
    }

    /// Decode draft memory. All-zero memory (draft bot crashed or never wrote)
    /// is reported as `Empty` so the caller can fall back to defaults.
    pub fn decode(b: &[u8]) -> Result<Self, DecodeError> {
        if b.len() < DRAFT_MEMORY_BYTES {
            return Err(DecodeError::TooShort);
        }
        if b.iter().all(|&x| x == 0) {
            return Err(DecodeError::Empty);
        }
        if b[off::MAGIC..off::MAGIC + 2] != MAGIC {
            return Err(DecodeError::BadMagic);
        }
        if b[off::VERSION] != THETA_VERSION {
            return Err(DecodeError::UnsupportedVersion(b[off::VERSION]));
        }
        let stored = u16::from_le_bytes([b[off::CRC], b[off::CRC + 1]]);
        if stored != crc16(&b[..off::CRC]) {
            return Err(DecodeError::BadChecksum);
        }
        let mut t = Self {
            team_size: b[off::TEAM_SIZE],
            schema_hash: u16::from_le_bytes([b[off::SCHEMA_HASH], b[off::SCHEMA_HASH + 1]]),
            weights_hash: u16::from_le_bytes([b[off::WEIGHTS_HASH], b[off::WEIGHTS_HASH + 1]]),
            ..Self::default()
        };
        t.roles.copy_from_slice(&b[off::ROLES..off::ROLES + 8]);
        let n = (b[off::N_PRIORITY] as usize).min(MAX_PRIORITY_GROUPS);
        t.camp_priority =
            (0..n).map(|i| (b[off::PRIORITY + 2 * i], b[off::PRIORITY + 2 * i + 1])).collect();
        for (i, slot) in t.film.iter_mut().enumerate() {
            *slot = (b[off::FILM + 2 * i] as i8, b[off::FILM + 2 * i + 1] as i8);
        }
        t.knobs.copy_from_slice(&b[off::KNOBS..off::KNOBS + KNOB_COUNT]);
        for (s, v) in t.champ_vecs.iter_mut().enumerate() {
            for (j, x) in v.iter_mut().enumerate() {
                *x = b[off::CHAMP_VECS + s * CHAMP_VEC_LEN + j] as i8;
            }
        }
        t.champ_kinds.copy_from_slice(&b[off::CHAMP_KINDS..off::CHAMP_KINDS + CHAMP_SLOTS]);
        t.map_cache.copy_from_slice(&b[off::MAP_CACHE..off::MAP_CACHE + MAP_CACHE_BYTES]);
        Ok(t)
    }

    #[must_use]
    pub fn knob(&self, index: usize) -> u8 {
        self.knobs.get(index).copied().unwrap_or(0)
    }

    /// Champion vector for a champion kind, if the draft recorded it.
    #[must_use]
    pub fn champ_vec(&self, kind: u8) -> Option<&[i8; CHAMP_VEC_LEN]> {
        self.champ_kinds.iter().position(|&k| k == kind).map(|i| &self.champ_vecs[i])
    }
}

pub const TEAM_CHANNEL_BYTES: usize = 8;
pub const CHANNEL_VERSION: u8 = 1;
pub const MAX_CLAIM_SLOTS: usize = 6;

/// Decoded 8-byte team channel.
///
/// | byte | bits | field |
/// |---|---|---|
/// | 0 | 0–1 | version (0 = never written) |
/// | 0 | 2 | parity of the round the leader last wrote |
/// | 0 | 3–5 | leader slot |
/// | 0 | 6 | flag: the sighting is an enemy channelling a tribute (lives with the sighting) |
/// | 0 | 7 | flag: help requested |
/// | 1 | – | team aggression parameter |
/// | 2 | – | team tribute-threshold parameter (gems) |
/// | 3–5 | 4 bits × 6 | claim per ally slot: priority index + 1, 0 = none |
/// | 6–7 | x:7, y:7, r:2 | latest enemy sighting; r = round mod 4; (0,0) = none |
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TeamChannel {
    pub version: u8,
    pub parity: bool,
    pub leader: u8,
    pub enemy_tributing: bool,
    pub help: bool,
    pub aggression: u8,
    pub tribute_threshold: u8,
    /// Per ally slot: `Some(index into θ_match.camp_priority)`.
    pub claims: [Option<u8>; MAX_CLAIM_SLOTS],
    sighting: Option<(Pos, u8)>,
}

impl TeamChannel {
    #[must_use]
    pub fn decode(b: [u8; TEAM_CHANNEL_BYTES]) -> Self {
        let claims_bits = u32::from(b[3]) | u32::from(b[4]) << 8 | u32::from(b[5]) << 16;
        let mut claims = [None; MAX_CLAIM_SLOTS];
        for (i, c) in claims.iter_mut().enumerate() {
            let v = (claims_bits >> (4 * i)) & 0xF;
            *c = (v != 0).then(|| (v - 1) as u8);
        }
        let s = u16::from_le_bytes([b[6], b[7]]);
        let (x, y, r) = (i32::from(s & 0x7F), i32::from(s >> 7 & 0x7F), (s >> 14) as u8);
        Self {
            version: b[0] & 0b11,
            parity: b[0] >> 2 & 1 == 1,
            leader: b[0] >> 3 & 0b111,
            enemy_tributing: b[0] >> 6 & 1 == 1,
            help: b[0] >> 7 & 1 == 1,
            aggression: b[1],
            tribute_threshold: b[2],
            claims,
            sighting: (x != 0 || y != 0).then_some((Pos::new(x, y), r)),
        }
    }

    #[must_use]
    pub fn encode(&self) -> [u8; TEAM_CHANNEL_BYTES] {
        let mut b = [0u8; TEAM_CHANNEL_BYTES];
        b[0] = (self.version & 0b11)
            | u8::from(self.parity) << 2
            | (self.leader & 0b111) << 3
            | u8::from(self.enemy_tributing) << 6
            | u8::from(self.help) << 7;
        b[1] = self.aggression;
        b[2] = self.tribute_threshold;
        let mut claims_bits = 0u32;
        for (i, c) in self.claims.iter().enumerate() {
            if let Some(v) = c {
                claims_bits |= (u32::from(*v).min(14) + 1) << (4 * i);
            }
        }
        b[3..6].copy_from_slice(&claims_bits.to_le_bytes()[..3]);
        let s = match self.sighting {
            Some((p, r)) => {
                (p.x.clamp(0, 127) as u16)
                    | (p.y.clamp(0, 127) as u16) << 7
                    | u16::from(r & 0b11) << 14
            }
            None => 0,
        };
        b[6..8].copy_from_slice(&s.to_le_bytes());
        b
    }

    /// Call at the start of each turn. Returns `true` if this bot is the first
    /// ally to act this round and therefore the leader: it clears the per-round
    /// flags and ages out the shared sighting before anyone else reads them.
    pub fn begin_turn(&mut self, round: u32, my_slot: u8) -> bool {
        let parity = round & 1 == 1;
        let leader = self.version == 0 || self.parity != parity;
        if leader {
            self.version = CHANNEL_VERSION;
            self.parity = parity;
            self.leader = my_slot & 0b111;
            self.help = false;
            if let Some((_, r)) = self.sighting {
                // Age 3 is indistinguishable from age −1 next round: drop it now.
                if (round.wrapping_sub(u32::from(r)) & 0b11) == 3 {
                    self.sighting = None;
                    self.enemy_tributing = false;
                }
            }
        }
        leader
    }

    pub fn post_sighting(&mut self, p: Pos, round: u32) {
        // A fresh tributer alert outranks an ordinary sighting.
        if self.enemy_tributing && self.sighting(round).is_some_and(|(_, age)| age <= 1) {
            return;
        }
        self.sighting = Some((p, (round & 0b11) as u8));
        self.enemy_tributing = false;
    }

    /// An enemy is channelling a tribute at `p`: a stun, kill or lock cancels the bag.
    pub fn post_tributer(&mut self, p: Pos, round: u32) {
        self.sighting = Some((p, (round & 0b11) as u8));
        self.enemy_tributing = true;
    }

    /// Shared sighting and its age in rounds (0..=2).
    #[must_use]
    pub fn sighting(&self, round: u32) -> Option<(Pos, u32)> {
        self.sighting.map(|(p, r)| (p, round.wrapping_sub(u32::from(r)) & 0b11))
    }

    pub fn set_claim(&mut self, slot: usize, priority_index: Option<u8>) {
        if let Some(c) = self.claims.get_mut(slot) {
            *c = priority_index.map(|v| v.min(14));
        }
    }

    /// How many *other* allies currently claim this priority index.
    #[must_use]
    pub fn claimants(&self, priority_index: u8, my_slot: usize) -> usize {
        self.claims
            .iter()
            .enumerate()
            .filter(|&(i, c)| i != my_slot && *c == Some(priority_index))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_theta() -> ThetaMatch {
        let mut t = ThetaMatch {
            team_size: 5,
            schema_hash: 0xBEEF,
            weights_hash: 0x1234,
            roles: [1, 2, 3, 4, 5, 0, 0, 0],
            camp_priority: vec![(3, 200), (0, 150), (7, 90)],
            ..ThetaMatch::default()
        };
        t.film[0] = (-12, 7);
        t.film[47] = (127, -128);
        t.champ_kinds[0] = 10;
        t.champ_vecs[0] = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16];
        t.map_cache[191] = 0xAA;
        t
    }

    #[test]
    fn theta_round_trips() {
        let t = sample_theta();
        let decoded = ThetaMatch::decode(&t.encode()).unwrap();
        assert_eq!(decoded, t);
        assert_eq!(decoded.champ_vec(10).unwrap()[15], -16);
        assert!(decoded.champ_vec(3).is_none());
    }

    #[test]
    fn theta_rejects_corruption_and_reports_empty() {
        let mut bytes = sample_theta().encode();
        bytes[500] ^= 1;
        assert_eq!(ThetaMatch::decode(&bytes), Err(DecodeError::BadChecksum));
        assert_eq!(ThetaMatch::decode(&[0u8; 1024]), Err(DecodeError::Empty));
        let mut wrong = sample_theta().encode();
        wrong[2] = 9;
        assert_eq!(ThetaMatch::decode(&wrong), Err(DecodeError::UnsupportedVersion(9)));
        assert_eq!(ThetaMatch::decode(b"hello world"), Err(DecodeError::TooShort));
    }

    #[test]
    fn default_knobs_are_set() {
        let t = ThetaMatch::default();
        assert_eq!(t.knob(knob::RETREAT_HP_PCT), 35);
        assert_eq!(t.knob(knob::AGGRESSION), 128);
        assert_eq!(t.knob(999), 0);
    }

    #[test]
    fn channel_round_trips() {
        let mut c = TeamChannel {
            version: 1,
            parity: true,
            leader: 4,
            enemy_tributing: true,
            help: false,
            aggression: 200,
            tribute_threshold: 33,
            ..TeamChannel::default()
        };
        c.set_claim(0, Some(2));
        c.set_claim(5, Some(14));
        c.post_sighting(Pos::new(99, 87), 41);
        let d = TeamChannel::decode(c.encode());
        assert_eq!(d, c);
        assert_eq!(d.claimants(2, 3), 1);
        assert_eq!(d.claimants(2, 0), 0, "own claim not counted");
        assert_eq!(d.sighting(42), Some((Pos::new(99, 87), 1)));
    }

    #[test]
    fn zeroed_channel_means_nothing_written() {
        let c = TeamChannel::decode([0; 8]);
        assert_eq!(c.version, 0);
        assert!(c.claims.iter().all(Option::is_none));
        assert_eq!(c.sighting(10), None);
    }

    #[test]
    fn first_actor_each_round_becomes_leader_and_ages_sightings() {
        let mut c = TeamChannel::decode([0; 8]);
        assert!(c.begin_turn(1, 2), "first write ever");
        c.help = true;
        c.post_sighting(Pos::new(10, 10), 1);
        assert!(!c.begin_turn(1, 0), "same round: not leader");
        assert!(c.help, "flags survive within the round");
        assert!(c.begin_turn(2, 0), "new round");
        assert!(!c.help, "leader clears flags");
        assert_eq!(c.sighting(2), Some((Pos::new(10, 10), 1)));
        c.begin_turn(3, 0);
        assert!(c.sighting(3).is_some());
        c.begin_turn(4, 0);
        assert_eq!(c.sighting(4), None, "age 3 dropped");
    }
}
