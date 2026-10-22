//! Learned policies that are trained by playing (ml/rl/ppo.py), not by copying the rules.
//!
//! Two experiment arms share this:
//! - arm 1 "experts+" (knob2 ROUTER, `weights/router.bin` with [`RL_OUTPUTS`] outputs): picks the
//!   expert (9) and, for fights, which enemy to target (4);
//! - arm 2 "micro" (knob2 MICRO_NET, `weights/micro.bin`): on fight turns picks the step (9),
//!   the attack target (4) and whether to cast (2) itself.
//!
//! Logits are the network's last-layer accumulators; the exporter stores the power-of-two scale
//! in that layer's (otherwise unused) `shift` byte, so `logit = acc / 2^shift`. During training
//! (knob2 RL_SEED > 0) every head samples from its masked softmax with a deterministic hash (the
//! engine has no RNG and games must stay reproducible); otherwise it takes the argmax.

/// Arm 1 outputs: 9 experts, then 4 fight-target modes.
pub const RL_OUTPUTS: usize = 13;
pub const EXPERT_HEAD: core::ops::Range<usize> = 0..9;
pub const TARGET_HEAD: core::ops::Range<usize> = 9..13;
/// Arm 2: the 109 features + 9 walkable + 9 threat bytes in; step, target and cast heads out.
pub const MICRO_INPUTS: usize = 127;
pub const MICRO_OUTPUTS: usize = 15;
pub const STEP_HEAD: core::ops::Range<usize> = 0..9;
pub const MICRO_TARGET_HEAD: core::ops::Range<usize> = 9..13;
pub const CAST_HEAD: core::ops::Range<usize> = 13..15;
/// Schema hash of micro blobs (the feature schema's, so a feature change invalidates both).
pub const MICRO_SCHEMA: u16 = crate::features_gen::SCHEMA_HASH ^ 0x4D49;

/// Target modes of the target heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetMode {
    /// Keep the rules' choice (arm 1) / don't attack (arm 2).
    Rule,
    Nearest,
    Weakest,
    Carrier,
}

impl TargetMode {
    #[must_use]
    pub fn from_index(i: usize) -> Self {
        match i {
            1 => Self::Nearest,
            2 => Self::Weakest,
            3 => Self::Carrier,
            _ => Self::Rule,
        }
    }
}

/// A uniform number in [0, 1) from the turn, champion slot, training seed and head.
#[must_use]
pub fn uniform(turn: u32, slot: u8, seed: u8, head: u8) -> f32 {
    let mut h = turn.wrapping_mul(2_654_435_761)
        ^ u32::from(slot).wrapping_mul(40_503)
        ^ u32::from(seed).wrapping_mul(0x9E37_79B9)
        ^ u32::from(head).wrapping_mul(0x85EB_CA6B);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Pick from a head: masked softmax over `logits / 2^shift`. `u` = None takes the argmax.
/// Returns (index within the head, log-probability of that pick under the softmax).
#[must_use]
pub fn pick(logits: &[i32], mask: &[bool], shift: u8, u: Option<f32>) -> Option<(usize, f32)> {
    let scale = (1u64 << u32::from(shift.min(30))) as f32;
    let vals: Vec<(usize, f32)> = logits
        .iter()
        .zip(mask)
        .enumerate()
        .filter(|(_, (_, &ok))| ok)
        .map(|(i, (&l, _))| (i, l as f32 / scale))
        .collect();
    let max = vals.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }
    let z: f32 = vals.iter().map(|&(_, v)| (v - max).exp()).sum();
    let chosen = match u {
        None => vals.iter().copied().fold((usize::MAX, f32::NEG_INFINITY), |b, x| {
            if x.1 > b.1 {
                x
            } else {
                b
            }
        }),
        Some(u) => {
            let mut acc = 0.0;
            let mut pick = *vals.last()?;
            for &(i, v) in &vals {
                acc += (v - max).exp() / z;
                if u < acc {
                    pick = (i, v);
                    break;
                }
            }
            pick
        }
    };
    Some((chosen.0, chosen.1 - max - z.ln()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_and_sampling_respect_the_mask() {
        let logits = [400, 100, 900];
        let mask = [true, true, false];
        assert_eq!(pick(&logits, &mask, 8, None).map(|p| p.0), Some(0));
        let (i, lp) = pick(&logits, &mask, 8, Some(0.99)).unwrap();
        assert_eq!(i, 1, "high u lands on the last allowed action");
        assert!(lp < 0.0);
        assert!(pick(&logits, &[false; 3], 8, None).is_none());
    }

    #[test]
    fn log_probs_sum_to_one() {
        let logits = [256, 0, -256];
        let mask = [true; 3];
        let total: f32 = (0..3)
            .map(|k| {
                let u = [0.0, 0.8, 0.99][k];
                pick(&logits, &mask, 8, Some(u)).unwrap()
            })
            .map(|(_, lp)| lp.exp())
            .sum();
        // logits 1, 0, -1 -> probabilities 0.665, 0.245, 0.090
        assert!((total - 1.0).abs() < 0.01, "{total}");
    }

    #[test]
    fn uniform_is_deterministic_and_spread() {
        assert_eq!(uniform(10, 2, 7, 0), uniform(10, 2, 7, 0));
        assert_ne!(uniform(10, 2, 7, 0), uniform(10, 2, 8, 0));
        let mean: f32 = (0..1000).map(|t| uniform(t, 1, 3, 0)).sum::<f32>() / 1000.0;
        assert!((mean - 0.5).abs() < 0.05);
    }
}
