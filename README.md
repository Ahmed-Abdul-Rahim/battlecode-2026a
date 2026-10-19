# Battlecode 2026a — team bot

Our entry for Cambridge Battlecode **cambc2026a** (season ends ~18 Nov 2026). Rust bot, Python tooling.

**Status:** v0 heuristic core + sparse mixture-of-experts wins **34/34** against the starter bot (17 maps × both sides), mean ~27k fuel/turn, 0 exhausted turns. Learned heads, feature selection, champion vectors and UDA are scaffolded and tested; they need self-play data and the GPU box.

## Quick start (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # git, gh, Python 3.12, Rust, MSVC, venv, cambc, hooks
.\.venv\Scripts\Activate.ps1
bctools gate main                 # all maps, both sides vs bots/baseline (~20 s)
cargo test --workspace            # 79 host tests for the logic crates
pytest -q                         # Python tooling tests
```

Linux / WSL / GPU box: `bash scripts/setup.sh [--ml]`.

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
| `harness/` | Contract for the teammate's local-LLM harness |

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
scripts	rain.ps1 start     # league trainer + co-evolution daemons (resumable)
scripts	rain.ps1 status    # champions, Elo, acceptances, ladder, which zip to upload
scripts	rain.ps1 stop
```

What is learned (θ, policy pack, router), how the daemons work and how to extend them: [docs/training.md](docs/training.md).

## Rules of the road

- **Never `cambc submit` (or `match-groups`, `login`, `ladder`) without the team agreeing.** All tooling is local-only.
- Edit `crates/`, not `bots/main/src/bc_*/`; run `bctools sync`. CI fails on drift.
- Every technique cites a paper ID from [docs/papers.md](docs/papers.md).
- See [CONTRIBUTING.md](CONTRIBUTING.md) for branches, PRs and the backlog.

## Docs

[training](docs/training.md) · [architecture](docs/architecture.md) · [fuel budget](docs/fuel-budget.md) · [memory layout](docs/memory-layout.md) · [feature selection](docs/feature-selection.md) · [champion vectors](docs/champions.md) · [self-improvement loop](docs/self-improvement.md) · [cambc CLI notes](docs/cambc-cli.md) · [papers](docs/papers.md) · [game rules](docs/GameRules.md) · [Rust SDK](docs/BotAPIRust.md)
