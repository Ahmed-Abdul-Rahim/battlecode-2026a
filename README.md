# Battlecode 2026a — team bot

Our entry for Cambridge Battlecode **cambc2026a** (season ends ~18 Nov 2026). Rust bot, Python tooling.

**Status (2026-10-19):** heuristic engine plus a learned layer: per-map policy packs, a utility arbiter and a router, trained by self-running daemons (co-evolution, ES, offline RL). The latest packaged versions are in `submission/` (gitignored, see the hand-off section). The engine is on branch `exp/v6-features`; read [docs/training.md](docs/training.md) first.

## Quick start (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # git, gh, Python 3.12, Rust, MSVC, venv, cambc, hooks
.\.venv\Scripts\Activate.ps1
bctools gate main                 # all maps, both sides vs bots/baseline (~20 s)
cargo test --workspace            # 79 host tests for the logic crates
pytest -q                         # Python tooling tests
```

Linux / WSL / GPU box: `bash scripts/setup.sh [--ml]`.

## Hand-off: setting up on another PC

Git carries the code and the **trained knowledge**:
- `bots/main/src/learned.rs`, `bots/main/src/pack/pack.bin`, `bots/main/src/weights/router.bin` and `bots/main/src/map_plans.rs`;
- the extracted top-team clones in `data/mimic/`.

Some things are gitignored on purpose and have to be handed over by hand:

| What | Why it's needed | How |
| --- | --- | --- |
| `submission/*.zip` | Every packaged version. The trainers' frozen opponents (`bots/main-v4` … `main-v13b`) and the lineage champions are unzipped from these. **Required** | `handoff.zip` |
| `runs/coevo/state.json`, `runs/league/state.json` | Continue the current training run (lineages, Elo, ladder history). Optional: without them, training restarts from the committed champion | `handoff.zip` |
| `bots/lin-b`, `bots/lin-c` | Source of co-evolution lineages B and C. Only needed to continue a run | `handoff.zip` |
| `runs/mimic/`, `best_bot_dump/` | Top-team clones (also in `data/mimic/`) and the best-bot games | `handoff.zip` |
| `runs/ladder_*`, `runs/ranked_v5` | Downloaded ladder summaries (~150 MB), for the ladder study and mimic re-extraction. Optional | `handoff.zip --with-ladder` |
| `.venv`, `target/`, `bots/*/.cambc` | Build products | recreated by `scripts/setup.ps1` / builds |

**On this PC:**

```powershell
python scripts\handoff.py pack                 # -> handoff.zip (~3 MB); add --with-ladder for match data
```

**On the new PC**, after `git clone` and `git checkout exp/v6-features`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
python scripts\handoff.py restore handoff.zip  # unpacks state + rebuilds bots/main-v*, co*-v* from the zips
scripts\train.ps1 start                        # resume training; scripts\train.ps1 status to check
```

Without `handoff.zip`, ask for the `submission/` zips. Run `python scripts\handoff.py refs` to unzip them into `bots/main-v*`, then `scripts\train.ps1 start` begins a fresh run from the committed champion.

Uploading stays manual and needs the team's agreement (`docs/cambc-cli.md`).

## Layout

| Path | What |
| --- | --- |
| `bots/main/` | Submission bot (cambc layout). `src/*.rs` is the SDK adapter; `src/bc_*/` are **generated** copies of `crates/` |
| `bots/baseline/` | Frozen starter bot, the gate's default opponent |
| `bots/fuelbench/` | Measures fuel cost of SDK calls → docs/fuel-budget.md |
| `crates/bc-core` | Geometry, terrain, champion kits, ledger (gold/gems/items), belief, memory codecs, fuel governor |
| `crates/bc-path` | Pathing ensemble: cached fields → repair → greedy/danger → windowed A* → fallback |
| `crates/bc-brain` | Features (generated), meta-gate, router, experts, int8 kernel + FiLM, champion vectors, shop |
| `crates/bc-draft` | Map descriptor, champion scorer, θ_match builder |
| `tools/bctools/` | `bctools` CLI: run, gate, summary, render, timeline, dataset, sync |
| `tools/mcp_server/` | MCP server for Claude Code / the Qwen harness (registered in `.mcp.json`) |
| `ml/features/` | Feature schema (source of truth), codegen, group backward selection |
| `ml/champions/` | Champion descriptors, fine-tuning from self-play, heatmap report |
| `ml/distill/`, `ml/uda/` | int8 blobs + teacher→student distillation; DANN scaffold (GPU box) |
| `maps/`, `maps/heldout/` | Platform map pool; held-out copies for the generalisation check |
| `docs/` | Rules, SDK reference, architecture, fuel budget, memory layout, papers registry, … |
| `harness/` | Training and evaluation: `coevo.py`, `league_train.py` (daemons), `speedrun.py`, `map_tune.py`, `mimic.py`, `ladder_study.py`, `params.py`, `pack.py`, `status.py` (see [harness/README.md](harness/README.md)) |
| `scripts/` | `setup.ps1`/`setup.sh`, `train.ps1`/`train.sh` (start/stop/status/logs), `handoff.py` |
| `data/mimic/` | Extracted top-team macro clones (draft, builds, openings, cadence) |

## Everyday commands

```powershell
bctools run main baseline --maps ambush lanes     # targeted matches
bctools gate main --best baseline                 # pass/fail with Wilson interval + held-out rate
bctools summary runs\<dir>\<match>.repl26a        # experts, tiers, fuel, errors
bctools render runs\<dir>\<match>.repl26a 300 renders\r300.png
bctools timeline runs\<dir>\<match>.repl26a renders\timeline.png
bctools dataset runs\<dir> datasets\sp.npz        # features + labels for ML
bctools sync                                      # after editing crates/ — vendor into the bot
cambc view runs\<dir>\<match>.repl26a             # official visualiser
```

In Claude Code: `/improve [focus]` runs one gated self-improvement iteration (docs/self-improvement.md).

## Training (learned layer)

```powershell
scripts\train.ps1 start     # league trainer + co-evolution daemons (resumable)
scripts\train.ps1 status    # champions, Elo, acceptances, ladder, which zip to upload
scripts\train.ps1 stop
```

What is learned (θ, policy pack, router), how the daemons work and how to extend them: [docs/training.md](docs/training.md).

## Rules of the road

- **Never `cambc submit` (or `match-groups`, `login`, `ladder`) without the team agreeing.** All tooling is local-only.
- Edit `crates/`, not `bots/main/src/bc_*/`; run `bctools sync`. CI fails on drift.
- Every technique cites a paper ID from [docs/papers.md](docs/papers.md).
- See [CONTRIBUTING.md](CONTRIBUTING.md) for branches, PRs and the backlog.

## Docs

[training](docs/training.md) · [architecture](docs/architecture.md) · [fuel budget](docs/fuel-budget.md) · [memory layout](docs/memory-layout.md) · [feature selection](docs/feature-selection.md) · [champion vectors](docs/champions.md) · [self-improvement loop](docs/self-improvement.md) · [cambc CLI notes](docs/cambc-cli.md) · [papers](docs/papers.md) · [game rules](docs/GameRules.md) · [Rust SDK](docs/BotAPIRust.md)
