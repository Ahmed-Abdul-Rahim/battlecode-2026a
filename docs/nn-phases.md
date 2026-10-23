# nn pipeline: phase notes

What's done, what's next, and what each later phase must not forget. Short version of the plan in
[architecture.md](architecture.md).

## Status (2026-10-23)

| Phase | State |
|---|---|
| 0 Cleanup, `nn` branch | done |
| 1 Inputs/actions/logging | done (`bots/main/src/teacher.rs`, `nn/spec.py`) |
| 2 Proof: teacher vs rules | imitation at parity; PPO flat after 78 iterations (paused), GRPO ~0.47; expert iteration running (`nn/expert.py`) |
| 3 Simulator | GPU engine `nn/gpusim/` validated against official replays (see below) |
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

## Expert iteration (`nn/expert.py`, 2026-10-23)

Why PPO/GRPO stayed flat: one result per ~1,800 decisions, the actions only reselect the rules'
goals (already an input), deterministic rollouts, a saturated critic and +-0.06 evaluation noise.
Search uses determinism directly instead:

- Base games (current policy, logging decisions) -> at sampled decisions, branch games force
  another goal (held 20 rounds while available). The bot reads a 32-byte block (magic
  `e5c4a1b7 0df0ed5e`) that `patch()` rewrites in the compiled file: no rebuild per branch. A
  pool of 32 prebuilt bots runs one batch.
- Value after H = 150 rounds from the bot's own "S" score line (submitted ours - theirs + half
  carried + earned gold / 10). cambc writes no statistics for `--rounds`-truncated games.
- Null branches (forced = own choice, held 0 rounds) must give exactly 0: checks alignment.
- Improved decisions (gain > 4) are relabelled, oversampled x8, distilled into the teacher with
  the imitation data; accepted only if it beats the best evaluation vs the rules.
- Next (chess-engine ideas that transfer): branch on the network's top choices (move ordering),
  a learned value instead of 150-round playouts (network evaluation), resume from snapshots in
  the GPU engine instead of replaying the shared prefix (hash table), skip one-option decisions
  (futility), per-map opening book. Alpha-beta itself only for small fights in the GPU engine.

## Phase 3: simulator

GPU engine status (00fragment replay, first round each statistic diverges): gems 127, team B
submitted gems exact for 400 rounds, camp damage Issac 228 / Prism 264 / Marcella 237, damage
taken Ren 36 / Cassi 30. Learned from replays: live ability values differ from the handbook
(Ren stun 0.75, Heart of Ruin dash 1.5, Cassi's stun allows walking; debug "A" dump), a capture
pays at the sixth phase, replays record failed moves (onto occupied tiles), camps target the
nearest champion by Chebyshev distance and sleep after two rounds with nobody in range (aggro,
no-damage and Euclidean variants all diverge earlier). Open: a wolf at round 30 skips an
in-range champion; kill gold from champion fights; vision.


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

**Order (decided 2026-10-23):** train the teacher to a plateau, then distil, then verify the student vs
the rule bot (80 map x side games), vs mimics incl. the 8 held-out ones, vs the teacher (gap =
what distillation lost), on held-out maps, and in release at 100k fuel (no out-of-fuel turns).
No further teacher compression unless training time becomes the bottleneck (rank-64 already
halved game time at no measured loss).

**Short compression check (2026-10-23, reference = full teacher it2, which mostly copies the rules):**

| Model | Multiply-adds | Goal agreement | Agreement where teacher overrules rules (249 states) | vs rules (80 games) | vs teacher (80 games) |
|---|---|---|---|---|---|
| teacher (full 512) | 1,489k | - | - | 0.542, +50 | - |
| teacher + tiny noise (noise band) | 1,489k | - | - | 0.556, +53 | 0.500 |
| rank 64 (in use) | 462k | 99.95% | 85% | 0.431, +17 | 0.550 |
| rank 16 | 324k | 99.94% | 84% | 0.486, +52 | 0.475 |
| width 64 | 157k | 99.93% | 89% | 0.458, +10 | 0.537 |
| width 16 (too-small sanity case) | 39k | 99.89% | 53% | 0.514, +28 | 0.537 |

- All sizes are lossless *for this teacher*: it overrules the rules' suggestion on only ~0.1% of
  states, and copying one input needs no capacity. Not informative about a real learned teacher.
- The discriminating measure is agreement on the states where the teacher overrules the rules:
  width 16 already drops to 53% there. Use it (plus games and plasticity) in the full study once
  the teacher wins by its own decisions.
- Noise band for 80-game evaluations: about +/-0.06 win rate.

**Student MVP (2026-10-23):** `nn/student.py` distils the teacher's goal head into an int8 BCW1
network (256 inputs the bot already computes -> 8 hidden -> 15 goals), quantised with
`ml/distill/quant.py`; the bot runs it with knob2 TEACHER = 5 (`teacher.rs student_pick`, lean
kernel bit-exact with the Python mirror: 10,153/10,153 decisions). It decides only from round 10,
with >= 70k fuel left at the decision point, and loads its weights on a turn of its own; otherwise
the rules decide.
- Agreement with the teacher: 99.86% (float and int8 identical).
- Release, 100k fuel, out-of-fuel turns (rules vs student): ambush 0/0, expanse 0/0, big-world 6/6,
  yellow-brick-road 58/22.
- vs the rule bot over 80 games, three builds: 0.487 (+21), 0.475 (+7), 0.412 (-58): parity
  within noise, as expected while the teacher copies the rules.
- Observation cost (debug): goal slots 7.4k of ~17k (distances to every camp group): first target
  if the student must run on more turns.

**LoRA / adapters:** for game patches (adapter on the frozen teacher, fast and can't drift far),
per-map or per-opponent specialists (merged into the weights: no extra fuel), and student
fine-tuning with PPO after distillation if the teacher-student gap is large. Not for the main
teacher run (full update takes ~4 s on the GPU).

**Architecture comparison (2026-10-23, `nn/zoo.py`, 1,411 train / 157 held-out champion-games
of logged play, same budget: predict the win and the return from each decision):**

| Model | Log-loss | Accuracy | Return error |
|---|---|---|---|
| base rate | 0.498 | 0.80 | - |
| MLP + 3 turns of history (current teacher) | **0.287** | 0.873 | 0.442 |
| MLP | 0.293 | 0.869 | 0.445 |
| LSTM | 0.329 | 0.869 | 0.445 |
| encoder-decoder | 0.337 | 0.866 | 0.446 |
| entity transformer | 0.346 | 0.866 | **0.419** |
| selective SSM (Mamba-style) | 0.352 | 0.876 | 0.448 |
| time transformer | 0.367 | 0.878 | 0.430 |
| GRU | 0.376 | 0.868 | 0.453 |

No architecture extracts clearly more from the state: sequence models get similar accuracy but
are overconfident; the entity transformer predicts returns best (small gain). Caveats: only
rule-bot games, a small held-out set, one budget without early stopping. Conclusion: the
bottleneck is the action space (the network only picks among the rules' goals) and the
learning signal, not the network type.

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
