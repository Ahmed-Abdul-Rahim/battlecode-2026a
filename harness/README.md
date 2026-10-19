# Qwen harness (owner: teammate with the RTX 5060 Ti)

This is a placeholder and interface contract for the local-LLM harness. The harness itself is the teammate's to build.

## Hard constraints

- **Bots can't call an LLM.** Every cambc bot (Rust or Python) runs in a Wasmtime sandbox with **no network**, about 100k fuel per turn, and 8 bytes of shared team memory. An LLM can't play live from inside a bot.
- **Speed.** A 2,000-round match with 5–6 champions is about 10k decisions. A 27B model on a 16 GB card at roughly Q4 (partly offloaded) would take hours per match.
- **Platform.** Nothing in the harness may submit to or play on the Battlecode platform.

## Roles that fit those constraints

1. **Exploiter author** (AlphaStar-style league exploiters, P-006): the model reads our replays through MCP and writes rival bots under `bots/exploit-*/` that target our weaknesses. Python bots get 25× fuel, which is handy for quick experiments. We then gate `main` against them.
2. **Labeller:** the model labels logged game states (`bctools dataset` rows) with yes/no or expert choices. Those labels train the distilled heads (`ml/distill/train.py`).
3. **Reviewer:** the model looks at `render_frame` / `render_timeline` PNGs (if it's multimodal) and drafts experiment issues.

## Interface: the bctools MCP server

Start the server: `.venv/Scripts/python.exe -m mcp_server.server` (stdio). Any MCP client works, for example the `mcp` Python SDK's `ClientSession`, or a tool-calling wrapper around an OpenAI-compatible local server (llama.cpp, vLLM). Tools are listed in docs/self-improvement.md. `draft_issue` needs `confirm=true` to file.

## League + knob tuning (built 2026-10-18)

Two scripts put roles 1 and 2 into practice. Both are local only.

- **`harness/qwen_league.py`**: Qwen writes Python exploiter bots, one per strategy (`rush`, `hunter`, `stealer`, `banker`, `pack`). Each one is copied from `bots/starter-python` into `bots/exploit-qwen-<strategy>/`, smoke-tested against `main`, and repaired from its logged errors (`--repairs`). `--refine <s>` asks Qwen for a stronger version of an existing bot. The script needs the qwen llama-server on `127.0.0.1:11435`, which the qwen MCP server starts on its first `ask`. Its log is `runs/qwen-league/log.jsonl`.
- **`harness/tune_knobs.py`**: coordinate descent over `main`'s θ_match knob defaults (`knob::DEFAULTS` in `crates/bc-core/src/memory.rs`). It plays against the pool `baseline`, `starter-python` and `exploit-qwen-*`, on the non-held-out maps only, both sides. The objective is the win score plus 0.25 × the normalised gem margin. The trace is in `runs/tune/<ts>/`. `--apply runs/tune/<ts>/best.json` writes the winning knobs into `crates/` and runs `bctools sync`. After that, gate as usual.

- **`harness/claude_bots/`**: hand-written exploiters `claude-ambusher`, `claude-pillager` and `claude-focus` (shared-memory focus fire). `python harness/claude_bots/make.py` regenerates `bots/claude-*`.
- **`harness/league_gate.py <bot> [--release] [--pool ...]`**: the candidate against the whole league (fixed bots, Qwen and claude exploiters, `main-v*` self-play snapshots, which are gitignored copies of past `main`). It plays all maps on both sides and reports per-opponent win rate, held-out win rate and fuel-suspended turns.
- **`ml/distill/train_np.py`**: numpy-only router distillation, so no PyTorch is needed. It writes a BCW1 int8 blob and checks the int8 accuracy. `bots/main-learned` loads it, but as of 2026-10-18 it isn't shipped because it failed non-inferiority: 47% vs main, and 9× the suspended turns.

What the league found (2026-10-18): pillagers beat `main` by killing a hurt big-bag carrier near the altar. The fixes that passed the league gate:
- hide in a bush when retreating;
- remembered ("ghost") threats;
- no unsafe tribute channel;
- a big bag at low HP recalls first;
- a big bag avoids even fights;
- reading the team's shared sighting, and answering help calls.

Tried and rejected by the gate:
- an adaptive deep-search fuel threshold (over-commits; suspended turns ×100);
- escape dash and chase dash;
- melee peel and the sacrifice rule;
- wards;
- the learned router.

Teleport crystals were skipped: only `ambush` has unreachable groups (96 gems).

## Deferred (decided in planning, 2026-10-18)

- OpenJev / small-model yes/no tool (licence: CC BY-NC 4.0, so check before any non-research use).
- Mamba / SSM belief model.

## Learned layer and training daemons (2026-10-19)

See [docs/training.md](../docs/training.md) for the full hand-off guide. Scripts:

| Script | What it does |
| --- | --- |
| `coevo.py` | Co-evolution: lineages A (ours), B (top-team comp), C (mimic-initialised) evolve per-map θ by ES against each other's hall of fame + mimics; confirmed acceptances become `bots/co<L>-vN` + zip; Elo in `runs/coevo/report.md` |
| `league_train.py` | Global ES / offline-RL router / both (UCB bandit), gate vs champion + league, auto-promotion, ladder watcher, optional `--auto-submit` |
| `params.py` | Registry of the 102 trainable parameters (layout, scale, bounds); `--check` vs `learned.rs` |
| `pack.py` | Policy-pack writer (θ, draft, builds, openings, hotspots) and hotspot miner |
| `mimic.py` | Clone a ladder team's macro (draft, builds, openings, cadence) into a pack spec |
| `ladder_study.py` | Per-team tempo / survival / comp / build table from match summaries |
| `speedrun.py` | Rounds to 1000 gems vs `bots/idle` (absolute farming speed) |
| `map_tune.py` | Per-map knob plans (`map_plans.rs`) |
| `status.py` | One-screen status; `scripts/train.ps1 status` |
