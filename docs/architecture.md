# Architecture

One page: what runs where, what learns what, and the commands. Branch: `nn` (archived history: tags `archive/*`).

## Commands

```
scripts\train.ps1 start | stop | status | logs   # loop 1: per-map tuning daemon (harness/league.py)
scripts\train.ps1 experiments [list|add|stop]    # loop 2: experiment autopilot (harness/experiments.py)
scripts\train.ps1 ml [status|stop]               # loop 3: nn pipeline (nn/run.py, this PC only)
```

## The three loops

| Loop | Learns | Against | Output | Shared via |
|---|---|---|---|---|
| 1. Per-map tuning | θ: 102 numbers per map (multipliers on the rules' constants) | Mimics of the live top 10 | `league/champion.json` → `pack.bin` | git (branch `league`) |
| 2. Experiments | Yes/no on queued code ideas (paired screen, then gate) | Mimics | Merged or rejected branches | git |
| 3. nn pipeline | A policy that chooses the actual actions | Current bot, mimics, frozen baselines | `nn/baselines/<patch>/…`, `bots/main/src/weights/` | This PC; only frozen baselines are committed |

Loops 1 and 2 tune the rule-based bot. Loop 3 is meant to replace its decisions.

## The bot (bots/main, wasm, 100k fuel per turn)

```
crates/ (SDK-free, host-tested, vendored into bots/main/src/bc_*/ by `bctools sync`)
  bc-core   geometry, terrain, champion kits, belief (unseen enemies), shared memory, fuel governor
  bc-path   distance fields, greedy / windowed A* steps
  bc-brain  features, gate, router, 9 experts, int8 kernel, shop
  bc-draft  map descriptor, draft scorer

one turn:  observe → belief → choose target → gate → expert → intent → SDK actions
map id:    world::fingerprint (exact; the draft sees the whole map) → per-map pack entry
```

Fuel facts: [fuel-budget.md](fuel-budget.md). Local matches can raise the limit with `cambc run --fuel N`; the ladder can't.

## Loop 3: teacher → student (plan: docs in `nn/`, status below)

```
              GPU (no fuel limit)                                  bot (100k fuel)
inputs ─► big teacher (512/1024 MLP, GRU, LSTM, SSM, transformer) ─► distil ─► small int8 student
 │            trained by imitation, then PPO vs frozen baselines          (MLP/GRU, or trees)
 │                                                                         gate on official cambc
 └ entity tokens (champion IDs, HP, cooldowns, buffs), 15×15 local grid,
   precomputed map channels per fingerprint (chokepoints, openness, distances), last 4 turns
```

1. **Proof:** the teacher plays inside the bot on the official engine with a raised `--fuel`. It's trained to a plateau (several seeds, both sizes) before any verdict.
2. **Simulator** (only after the proof): `crates/bc-sim`, a clean-room engine written from GameRules + SDK constants that runs the same wasm bots. It's checked against cambc, then batched and moved to the GPU.
3. **Freeze-and-grow:** the best model is frozen into the pool and the next, bigger or different, model trains against the pool. Each winner is distilled. The loop stops when the student stops gaining.
4. **Patches:** `nn/patch.py` diffs the SDK constants, the rules and the map pool. Changed maps get recomputed; changed champions get a fine-tune from the current best plus a new baseline folder.

| Step | Status (2026-10-23) |
|---|---|
| Cleanup, `nn` branch | done |
| Inputs/actions spec, logging | next |
| Teacher proof | – |
| Simulator | – |
| Zoo / freeze-and-grow | – |
| Patch watcher | – |

## Where things live

| Path | What |
|---|---|
| `bots/main` | The shipped bot |
| `bots/baseline`, `claude-*`, `exploit-qwen-*` | Fixed sparring anchors (loop 1) |
| `bots/mimic-*` | Rebuilt from `data/mimic` + `runs/ladder_*` (gitignored) |
| `harness/` | Loops 1 and 2 |
| `ml/` | Old learned router (arm 1, still in the bot), int8 export, feature schema; replaced by `nn/` in Phase 1 |
| `nn/` | Loop 3 |
| `tools/bctools` | Match runner, gate, replay tools, MCP server |
| `runs/` | Local output (ladder downloads are kept; everything else can be deleted) |
