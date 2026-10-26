//! Per-map plan settings (harness/search): which draft profile to pick and which knob / knob2
//! values to write into draft memory. One record format serves both sources:
//!
//! - the `SEARCH_CFG` block, byte-patched into the compiled file per candidate by the offline
//!   search (no rebuild per candidate; the magic finds it);
//! - pack tag 6 (`pack::TAG_PLAN`), the searched table keyed by map fingerprint.
//!
//! Record (little-endian): version u8 (0 = none), draft profile u8 (0 = keep; see
//! `draft::draft_prefs`), override count u8, reserved u8, then (slot u8, value u8) × count,
//! with slots 0..32 = knob and 32..64 = knob2. The patch block wins over the pack.
//!
//! Version 2 (pack only; maps where the best plan depends on the side): team A's spawn tile x u8,
//! y u8, the length of team A's record u8, then team A's record and team B's record (both v1).
//! The draft picks the one whose spawn tile is in its own spawn region.

use crate::bc_core::memory::{ThetaMatch, KNOB2_COUNT, KNOB_COUNT};

pub const MAGIC: [u8; 8] = [0x50, 0x53, 0x43, 0x46, 0x47, 0x01, 0xA5, 0x5A];
pub const BLOCK_LEN: usize = 128;


const fn block() -> [u8; BLOCK_LEN] {
    let mut b = [0u8; BLOCK_LEN];
    let mut i = 0;
    while i < MAGIC.len() {
        b[i] = MAGIC[i];
        i += 1;
    }
    b
}

#[used]
#[no_mangle]
static mut SEARCH_CFG: [u8; BLOCK_LEN] = block();

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Settings {
    pub draft: u8,
    pub overrides: Vec<(u8, u8)>,
}

impl Settings {
    /// None for an empty or malformed record.
    #[must_use]
    pub fn decode(b: &[u8]) -> Option<Self> {
        let (&version, &draft, &n) = (b.first()?, b.get(1)?, b.get(2)?);
        if version == 0 {
            return None;
        }
        let pairs = b.get(4..4 + 2 * usize::from(n))?;
        Some(Self { draft, overrides: pairs.chunks_exact(2).map(|p| (p[0], p[1])).collect() })
    }

    /// Write the overrides into the knob and knob2 bytes of draft memory.
    pub fn apply(&self, theta: &mut ThetaMatch) {
        for &(slot, value) in &self.overrides {
            let s = usize::from(slot);
            if s < KNOB_COUNT {
                theta.knobs[s] = value;
            } else if s - KNOB_COUNT < KNOB2_COUNT {
                theta.knobs2[s - KNOB_COUNT] = value;
            }
        }
    }
}

fn patched() -> Option<Settings> {
    // SAFETY: never written by the bot; volatile so the patched bytes are read at run time.
    let b = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(SEARCH_CFG)) };
    Settings::decode(&b[MAGIC.len()..])
}

/// A pack entry: v1 as is; v2 picks team A's or team B's record by our spawn region.
fn decode_entry(b: &[u8]) -> Option<Settings> {
    if b.first() != Some(&2) {
        return Settings::decode(b);
    }
    let (&x, &y, &len_a) = (b.get(1)?, b.get(2)?, b.get(3)?);
    let (rec_a, rec_b) = b.get(4..)?.split_at_checked(usize::from(len_a))?;
    let ours = cambc::map::spawn_region(cambc::Team::Ally).ok()?;
    let team_a = ours.is_set(crate::sdk::vec2(crate::bc_core::Pos::new(i32::from(x), i32::from(y)))).unwrap_or(false);
    Settings::decode(if team_a { rec_a } else { rec_b })
}

/// This map's settings: the patched block, else the pack's table entry, else none.
#[must_use]
pub fn for_map(fingerprint: u32) -> Option<Settings> {
    patched().or_else(|| decode_entry(crate::pack::lookup(fingerprint).plan))
}

