# nn pipeline: phase notes

What's done, what's next, and what each later phase must not forget. Short version of the plan in
[architecture.md](architecture.md).

## Status (2026-10-23)

| Phase | State |
|---|---|
| 0 Cleanup, `nn` branch | done |
| 1 Inputs/actions/logging | done (`bots/main/src/teacher.rs`, `nn/spec.py`) |
| 2 Proof: teacher vs rules | running: imitation at parity (0.50 train maps, 0.625 held-out, -8 gems); PPO sweep in `nn/run.py` |
| 3 Simulator | not started |
| 4 Zoo + student | not started (fuel measurements below) |
| 5 Patch watcher | done (`nn/patch.py`, every 6 h from the loop) |

Resume training: `scripts\train.ps1 ml` (state in `runs/nn/state.json`; see `scripts\train.ps1 ml status`).

## Lessons so far

- Imitating the rules' fight moves through the manual heads lost ~120 gems a game: the executor
  does more than the heads can name (camp attacks, combos, support casts). Imitation now copies
  goals only; manual control is a smoothed option PPO can learn to use (`nn/train.py STEP_SMOOTH`).
- `TEACHER=3` (rules' goal through the teacher's path) and `TEACHER=4` (network goal, executor
  moves) are the diagnostics for "is the control path lossy?".
- Games are deterministic: only 80 distinct teacher-vs-rules games exist (40 maps x 2 sides).
- Fuel chaos: any change reshuffles some games; judge on many games, never on one.

## Phase 2: verdict rules (unchanged)

Training to a plateau before any verdict: >= 20k PPO games per run, flat over 5 evaluations,
3 seeds, both widths, shaped and win-only rewards. Curves still rising at the end = throughput
problem -> build the simulator, don't stop.

## Phase 3: simulator

- Clean-room from `docs/GameRules.md` + SDK constants; implement the host interface of
  `sdk/rust/src/sys.rs` so the unchanged wasm bots run in it.
- Validate against cambc with identical bots (per-round visible-state logs + final stats).
- Ghidra/REA only for specific mismatches, and only after the competition rules are checked.
- Batched games / GPU kernels only after the CPU version is validated; N chosen by benchmark.

## Phase 4: zoo, student, shipping

**Overfitting to our own setup (must check before any upload):**
- Held-out maps (`nn/run.py HELD_OUT`) in every evaluation; a student must not be worse there.
- A held-out opponent set: mimics never used in training, scored against the rule bot's own
  results vs the same mimics (paired).
- Diversity in training opponents: rule bot, anchors, mimics, frozen snapshots (already in PPO).
- Fresh ladder mimics before release (`harness/mimic.py` from recent top-10 games).
- Platform check: unranked matches before ranked use; automatic revert if the ladder drops.

**Fuel (release build, cambc 3.0.5):**
- Rule bot: ~29k mean per turn, p95 ~36k, of 100k.
- Teacher observation as built today: ~47k extra (grid ~21k, per-step ~9k, groups ~8k, rest ~9k)
  -> far too much for the shipped bot. The student needs a cheap observation: no grid, fewer
  field lookups, no history copies, single SDK calls, `round_ties_even` instead of `round`.
- Network maths: int8 scalar 4.4 fuel per multiply-add, int8 SIMD 1.8, f32 scalar 7.5,
  f32 SIMD 3.9 (docs/fuel-budget.md). Next: i16 pre-widened weights, 4 rows per input load.
- **Unknown: does the ladder engine accept `simd128`?** Test with an unranked match first.
- Budget target: network <= ~35k fuel per turn on top of the rules (p95 under ~70k total, 0 out-of-fuel turns).
- Measured (release, vs baseline): ambush rules 29k mean / 0 out-of-fuel; + student-style observation
  (no grid) 55k / 11; + full teacher observation 73k / 21. yellow-brick-road: rules alone 54k mean,
  p95 80k, 31 out-of-fuel turns; + student observation 3,417 out-of-fuel turns.
- Consequence: the student gets its own small feature set (logged next to the teacher's in teacher
  games, so it can be trained), runs only on turns with fuel to spare (rules decide otherwise), and
  probably outputs the goal head only unless PPO shows the teacher relies on manual fighting.
- Rule bot itself runs out of fuel on dense maps (yellow-brick-road: 31 turns): worth fixing anyway.

**Zoo:** MLP 512/1024, GRU, LSTM, S4/S6, transformer over entity tokens; same inputs and training
budget; each needs in-bot inference (recurrent state carried between turns) to play rollouts.

**Freeze-and-grow:** best frozen into the pool, next candidate trains against the pool, promoted
only if it beats the best; stop when the student stops gaining.

**Submission:** hook the student into the existing `harness/league.py` submit + auto-revert path
(lead PC, team-agreed rules). Ask the user before the first real upload.

## Phase 5: patch watcher

- `nn/patch.py check|apply`; `NN_AUTO_UPGRADE=1` also upgrades cambc (rolled back if the bot breaks).
- On a patch: new SDK/maps copied in, smoke match, the runs continue from their best on the new
  rules. Qwen summary only if its server is already up (it shares the GPU).
