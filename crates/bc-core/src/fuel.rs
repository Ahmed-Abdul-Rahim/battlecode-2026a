//! Per-turn fuel budgeting.
//!
//! A bot that runs out of fuel is suspended mid-code and resumes next round,
//! effectively losing a turn. The governor gives each stage a budget, keeps a
//! reserve, and lets optional work (learned heads, local search) be skipped
//! when the turn is already expensive.

/// Something that reports remaining fuel (the SDK in game, a counter in tests).
pub trait FuelSource {
    fn remaining(&self) -> u64;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Stage {
    Observe = 0,
    Gate = 1,
    Brain = 2,
    Path = 3,
    Act = 4,
    Precompute = 5,
}

pub const STAGES: usize = 6;

/// Default per-stage budgets for a normal 100k turn (see docs/fuel-budget.md).
pub const DEFAULT_BUDGETS: [u64; STAGES] = [15_000, 1_000, 15_000, 40_000, 5_000, 10_000];
pub const DEFAULT_RESERVE: u64 = 10_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuelGovernor {
    budgets: [u64; STAGES],
    used: [u64; STAGES],
    reserve: u64,
    turn_start: u64,
}

impl Default for FuelGovernor {
    fn default() -> Self {
        Self::new(DEFAULT_BUDGETS, DEFAULT_RESERVE)
    }
}

impl FuelGovernor {
    #[must_use]
    pub fn new(budgets: [u64; STAGES], reserve: u64) -> Self {
        Self { budgets, used: [0; STAGES], reserve, turn_start: 0 }
    }

    pub fn start_turn(&mut self, remaining: u64) {
        self.used = [0; STAGES];
        self.turn_start = remaining;
    }

    /// Whether `estimate` more fuel can be spent in `stage` without breaking its
    /// budget or eating the reserve.
    #[must_use]
    pub fn allows(&self, stage: Stage, estimate: u64, remaining: u64) -> bool {
        let s = stage as usize;
        self.used[s] + estimate <= self.budgets[s] && remaining > self.reserve + estimate
    }

    /// Whether `estimate` fits in what's left of the turn, ignoring stage budgets.
    /// Used by start-up precomputation, which may borrow from idle stages.
    #[must_use]
    pub fn turn_allows(&self, estimate: u64, remaining: u64) -> bool {
        remaining > self.reserve + estimate
    }

    pub fn record(&mut self, stage: Stage, spent: u64) {
        self.used[stage as usize] += spent;
    }

    /// Run `f`, charging the fuel it used to `stage`.
    pub fn measure<T>(
        &mut self,
        stage: Stage,
        source: &impl FuelSource,
        f: impl FnOnce() -> T,
    ) -> T {
        let before = source.remaining();
        let out = f();
        self.record(stage, before.saturating_sub(source.remaining()));
        out
    }

    #[must_use]
    pub fn used(&self) -> [u64; STAGES] {
        self.used
    }

    #[must_use]
    pub fn turn_start(&self) -> u64 {
        self.turn_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct Counter(Cell<u64>);
    impl FuelSource for Counter {
        fn remaining(&self) -> u64 {
            self.0.get()
        }
    }

    #[test]
    fn stage_budget_and_reserve_both_apply() {
        let mut g = FuelGovernor::default();
        g.start_turn(100_000);
        assert!(g.allows(Stage::Brain, 10_000, 90_000));
        g.record(Stage::Brain, 10_000);
        assert!(!g.allows(Stage::Brain, 6_000, 80_000), "brain budget is 15k");
        assert!(g.allows(Stage::Path, 30_000, 80_000));
        assert!(!g.allows(Stage::Path, 5_000, 12_000), "would eat the 10k reserve");
        assert!(g.turn_allows(50_000, 70_000));
    }

    #[test]
    fn measure_charges_the_stage() {
        let fuel = Counter(Cell::new(50_000));
        let mut g = FuelGovernor::default();
        g.start_turn(fuel.remaining());
        let v = g.measure(Stage::Observe, &fuel, || {
            fuel.0.set(fuel.0.get() - 1_234);
            7
        });
        assert_eq!(v, 7);
        assert_eq!(g.used()[Stage::Observe as usize], 1_234);
    }
}
