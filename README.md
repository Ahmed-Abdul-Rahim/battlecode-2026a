# Battlecode 2026a: Patient Wolves

Our entry for Cambridge Battlecode **cambc2026a** (season ends ~18 Nov 2026). Rust bot, Python tooling.

This is the **`league` branch**. It holds the bot and one self-improvement loop: track the live top 10, clone each team into a *mimic*, train our bot against the mimics, and ship a new version when we lose on the ladder. Teammates run the same loop on their own PCs and share the weights through this branch.

Two mechanisms keep training useful over time:
- **Evolved mimics.** Every third cycle trains an *opponent* instead of us. It keeps a team's draft, builds and openings, but its micro evolves to beat our champion, so training continues when the real teams stop updating.
- **Map pool.** The lead follows the platform's current pool and adds new maps automatically.

Local games run the ladder's engine: **cambc ≥ 3.0.1** (engine 1.0.0).

**Start here:** [docs/league.pdf](docs/league.pdf) covers the architecture, every command, and debugging. The LaTeX source is `docs/league.tex`.

## Quick start (any PC)

```powershell
git clone -b league https://github.com/Ahmed-Abdul-Rahim/battlecode-2026a.git
cd battlecode-2026a
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
cambc login                                                  # once; read-only platform access for the lead
scripts\train.ps1 train                                      # worker: trains θ, tests queued experiments, pushes
scripts\train.ps1 status
```

Linux / macOS, same commands:

```bash
git clone -b league https://github.com/Ahmed-Abdul-Rahim/battlecode-2026a.git && cd battlecode-2026a
bash scripts/setup.sh                                        # --gpu also makes .venv-gpu for gpusim/
scripts/train.sh train
scripts/train.sh status
```

Exactly **one** PC runs as the lead. It reads the ladder, rebuilds mimics, writes the human review, and is the only PC that may upload. `train` remembers its flags, so a plain `train` later restarts the same role:

```powershell
scripts\train.ps1 train --lead                 # reads the platform, never uploads
scripts\train.ps1 train --lead --auto-submit   # also uploads, under the team-agreed rules (docs/league.pdf §4)
```

## What lives where

| Path | What |
| --- | --- |
| `bots/main/` | The bot (cambc layout). `src/pack/pack.bin` holds the **shared trained weights**; never edit it by hand |
| `league/champion.json` | Champion θ per map, promotion history, benchmarks (written by every PC, merged on push) |
| `league/ledger.json` | Top-10 snapshots, our ladder games, loss queue, uploads (written by the lead) |
| `league/review/<date>.md` | Human review: epistemic gaps and losses the trainer can't fix. Write in "Reviewer notes" |
| `data/mimic/` | Mimic specs of the current top 10 (and teams we recently lost to). Built as `bots/mimic-<slug>` (faithful, never trained) |
| `league/mimics/<slug>.json` | Evolved mimics: the team's θ trained to beat our champion, built as `bots/mimx-<slug>` |
| `maps/`, `data/fingerprints.json`, `data/map_geometry.json` | The platform's current map pool (`00…`/`01…` files) and each map's pack key and geometry. Retired maps are in `maps/retired/` |
| `harness/league.py` | The loop. `mimic.py` clones a team, `params.py` is the θ registry, `pack.py` writes `pack.bin` |
| `crates/` | Decision logic, pathing, brain, draft (SDK-free, host-tested). `bctools sync` vendors it into the bot |
| `gpusim/` | GPU engine (torch, batched games): replay validator, skirmish search. Runs with the CUDA venv (`.venv-gpu`, or `D:\Code\ml-gpu-venv` on the lead PC) |
| `tools/bctools/` | `bctools` CLI: run, gate, summary, render, timeline, dataset, sync. `tools/mcp_server/` exposes it over MCP |
| `bots/baseline`, `bots/claude-*`, `bots/exploit-*` | Fixed sparring bots (anchors) so old fixes stay fixed |
| `docs/` | Game rules, SDK reference, architecture, fuel budget, memory layout, papers, cambc notes |

## Everyday commands

```powershell
scripts\train.ps1 train | stop | status | logs    # the daemon (scripts/train.sh on Linux)
scripts\train.ps1 exp-status                      # experiments: tried, merged, rejected, up next
scripts\train.ps1 experiments add --title "Loot guard" --area economy --ref origin/feat/loot-guard --knobs LOOT_GUARD=1
python harness\league.py bench                    # champion vs every mimic, all maps
python harness\league.py review                   # write the human review now
python harness\league.py maps                     # lead: follow the platform map pool now
python harness\patch.py check                     # did the engine / SDK / rules / maps change?
bctools gate main                                 # pass/fail vs bots/baseline
bctools run main mimx-nyx --maps 00lanes          # vs an evolved mimic (mimic-nyx = faithful)
bctools summary runs\<dir>\<match>.repl26a        # experts, tiers, fuel, errors
cambc view runs\<dir>\<match>.repl26a             # official visualiser
cargo test --workspace; pytest -q                 # tests
```

Queued experiments run inside the daemon between training cycles, one at a time, on every PC that runs `train` (claims keep two PCs off the same one). A passed experiment is merged into `league` by a human, and the daemons pick it up on their next pull.

## Rules of the road

- Uploading happens only through `--auto-submit` on the lead PC, or `python harness\league.py submit` by a human. The team agreed to this. Never run `cambc submit` or `match-groups` by hand.
- Behaviour changes (new features or knobs found in a review) go on `feat/*` branches. Edit `crates/`, run `bctools sync`, gate, then merge into `league`. The daemons pick up the change on their next pull.
- Every technique cites a paper ID from [docs/papers.md](docs/papers.md). See [CONTRIBUTING.md](CONTRIBUTING.md).
- The old experiment history (co-evolution, ES, league trainer, Gemini's auto_train) stays on `exp/v6-features`. The ML pipeline (teacher/student, PPO/GRPO, MuZero, expert iteration) is archived at tag `archive/nn`.
