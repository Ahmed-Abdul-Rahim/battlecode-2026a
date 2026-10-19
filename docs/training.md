# Training the bot: the learned layer and the self-improvement daemons

This guide is for whoever takes over training. It explains:
- what is learned and where it lives;
- how the two training daemons work;
- how to run, read and extend them;
- what has already been tried.

Everything here runs locally. Nothing uploads to the platform unless you pass `--auto-submit`. Don't do that without the team agreeing (see [cambc-cli.md](cambc-cli.md)).

## TL;DR

```powershell
scripts\train.ps1 start     # start both daemons in the background (hours, days; resumable)
scripts\train.ps1 status    # one screen: champions, Elo, acceptances, ladder, which zip to upload
scripts\train.ps1 logs      # follow the logs
scripts\train.ps1 stop
```

Linux/macOS: use `scripts/train.sh` with the same subcommands.

Each improvement the daemons accept lands in four places:
- a new champion snapshot under `bots/`;
- a zip in `submission/`;
- a commit of the trained files (`bots/main/src/pack/pack.bin`, `learned.rs`, `weights/router.bin`);
- a line in the logs.

## What is learned, and where it lives

The engine is still the hand-written bot (`bots/main`, crates in `crates/`): pathing, combat, camps and the team channel. The learning sits on top, in **data the bot reads at startup**. Learned behaviour is always an extra term on the rules. With default values it is a no-op, so the bot plays exactly as the rules would. Learned parts can therefore always be switched on per map, and a broken one can be zeroed.

| Piece | File | What it holds | Trained by |
| --- | --- | --- | --- |
| Parameter vector θ (102 floats) | `bots/main/src/learned.rs` (defaults) | see the next table | `harness/es_tune.py` (global), `harness/coevo.py` (per map) |
| Policy pack | `bots/main/src/pack/pack.bin` (written by `harness/pack.py`) | **per map** (keyed by map fingerprint): θ, draft, emblem builds per champion kind, opening routes, hotspots | `coevo.py`, `mimic.py`, `map_tune.py` |
| Learned router | `bots/main/src/weights/router.bin` | int8 MLP: features → expert | `harness/league_train.py` RL track (`ml/rl/awr.py`) |
| Per-map knob plans | `bots/main/src/map_plans.rs` | knob overrides per map | `harness/map_tune.py` |

The **θ layout** is defined once, in `harness/params.py`. The Python trainers read it from there, and `learned.rs` holds the matching defaults.

| Block | Size | Meaning (used in) | Search scale |
| --- | --- | --- | --- |
| W | 18 | multipliers on hand-set constants: camp scoring weights, en-route limits, fight margin, retreat/assist HP, aggression, tribute and shop thresholds (`bot.rs`) | log |
| TARGET | 10 | camp-target scorer: score × exp(TARGET·φ) (`bot.rs target_bias`) | linear |
| FIGHT | 9 | fight-advantage bias: advantage × exp(FIGHT·ψ) (`bot.rs fight_bias`) | linear |
| HOT | 1 | death-hotspot threat weight. **Off**: tested harmful | fixed |
| U | 60 | **utility arbiter**: 6 options × 10 features (`utility.rs`) | linear |
| X | 4 | Tavi re-spec flag, arbiter fuel floor (k fuel), 2 reserved | fixed |

**The utility arbiter** (`bots/main/src/utility.rs`) is the "think about several things at once" layer. Every turn it scores six options together, with score = U_k · φ:
- keep the rule intent;
- **step back** (aggro swap: leave a camp's range so it turns on an ally, and keep casting from range 4);
- **support** an ally in trouble;
- fight the weakest enemy;
- **walk home** without the frozen recall;
- recall.

The features φ are: our HP, camp vs. enemy damage on us, allies and enemies near, an ally in trouble, bag size, distance home, and whether we're ranged. With all-zero U the arbiter keeps the rule intent. A hand-set aggro-swap row cut camp deaths per game from 4.1 to 3.0 at the same win rate.

**Compute safety.** The learned parts run only while fuel allows: the arbiter above `X.arbiter_fuel_floor_k` × 1000 fuel, and the router above 65k. Below that the rules decide, and below 25k (`gate::LOW_FUEL`) the cheapest rules always decide. Training also penalises suspended (out-of-fuel) turns in the fitness, so a learned policy can't buy strength by freezing champions.

## The daemons

### `harness/coevo.py`: co-evolution with per-map ES (main producer)

- **Three lineages evolve per-map θ against each other:**
  - **A**, our engine (`bots/main`);
  - **B**, seeded with the top teams' comp and Power build (`bots/lin-b`);
  - **C**, imitation-initialised from a top-team mimic: its draft, builds and openings (`bots/lin-c`).
- **Each cycle:**
  1. One lineage, one map (round-robin).
  2. 8 antithetic ES samples around that map's θ.
  3. Each sample plays that map only (both sides) against 5 opponents. The opponents are PFSP-sampled from the other lineages' halls of fame, the frozen bots and the mimics.
- **Fitness:** win + 0.25 × gem margin − 0.01 × deaths/game − 0.002 × suspended turns/game.
- **Acceptance:**
  - better fitness than the champion on the same games (by ≥ 0.05);
  - confirmed on 4 *fresh* opponents (the engine is chaotic, see the pitfalls below);
  - doesn't lose the head-to-head against its own champion on that map.
- **Then:** a new snapshot `bots/co<L>-vN` and `submission/co<L>-vN.zip`. For lineage A the pack is committed.
- **Every 10 cycles:** a cross-check between all champions and v13b, and Bradley-Terry Elo in `runs/coevo/report.md`.
- **Every 30 cycles:** the mimics are rebuilt from newly downloaded ladder games.
- **State:** `runs/coevo/state.json`.

### `harness/league_train.py`: global ES, RL router, ladder watcher

- **A UCB bandit picks one of three tracks per cycle:**
  - `es`: sep-ES on W/TARGET/FIGHT against a PFSP league;
  - `rl`: exploration games plus advantage-weighted regression of the router (`ml/rl/awr.py`);
  - `es+rl`: both together.
- **Gate:** the candidate must beat the champion head to head on all 17 maps (≥ 0.56) and must not lose to the pool. A promotion produces `bots/auto-vN` plus a zip and a commit.
- **Ladder watcher:** every cycle it downloads our new ladder games (read-only) into `runs/ladder_live`, tracks win rate and deaths, and raises the training death penalty while ladder deaths stay high.
- **`--auto-submit`** (off by default): uploads promotions and reverts versions that do worse on the ladder.
- **State:** `runs/league/state.json`.

### One-shot tools
- `harness/speedrun.py`: rounds to 1,000 gems against `bots/idle` (absolute farming speed).
- `harness/v6_trials.py --sanity`: a variant against the submitted bots (v4–v8).
- `harness/map_tune.py`: per-map knob plans.
- `harness/ladder_study.py`: tempo, survival, comp and build per team from match summaries.
- `harness/mimic.py`: clone a team's macro policy (draft, builds, openings, cadence) into a pack spec.

## Common tasks

**Add a trainable parameter.**
1. Append it to a block in `harness/params.py`, and to the matching array in `learned.rs` (bump the block's length const, for example `U_N`).
2. Use it in Rust through `learned::w/target/fight/utility/extra`.
3. Run `python harness/params.py --check`.

Old packs stay valid, because a shorter θ fills a prefix and the rest keeps the defaults.

**Add a learned behaviour.** Implement it behind a parameter whose default reproduces today's play. Then check identity: `bots/main` against the previous champion should give the same winners (expect a few fuel-chaos flips), then let ES turn it on per map.

**Clone a new strong team.**
1. Download its games: `cambc matches list --team <id> --json`, then `matches summary <id> --json -o runs/ladder_top/<id>.json`.
2. Run `python harness/mimic.py --team "<name>"`.
3. The co-evolution daemon picks up `bots/mimic-*` on its next refresh. You can also build them by hand: copy `bots/main`, then run `harness/pack.py write runs/mimic/<slug>.spec.json --out bots/mimic-<slug>/src/pack/pack.bin`.

**Resume on another PC.** The trained knowledge is in committed files: `learned.rs`, `pack.bin`, `router.bin` and `map_plans.rs`. `runs/` is not committed. Starting with an empty `runs/` re-seeds the daemons from the committed champion.
- Frozen reference bots (`bots/main-v*`) come from the zips in `submission/`, which are gitignored too: copy them over or unzip them.
- The `bots/mimic-*` bots are rebuilt from `runs/mimic/*.spec.json` (or re-extracted from the ladder).

**Pick what to upload.**
1. `scripts\train.ps1 status` names the lineage A champion's zip.
2. Check `runs/coevo/report.md`: the champion should beat v13b in the cross-check.

## Pitfalls and dead ends (measured, don't repeat blindly)

- **The engine is deterministic but chaotic.** A code change that does nothing (one extra function call) flips about 9/34 game winners, because pathing and precompute are fuel-gated. Never trust 1–2-game differences. That's why the co-evolution gate confirms on fresh opponents.
- **Local self-play hides absolute weaknesses** (our economy was half the top teams' speed). Always check `speedrun.py` and the ladder study too.
- **Top teams' strength is execution, not macro.** Mimics with their exact draft, Treasure-first builds, openings and cadence *lose* to our champion (0.27–0.38). Copying their visible decisions isn't enough.
- **Tested and rejected:**
  - hotspot threats (death spots are our farm camps; win 0.54 → 0.15);
  - the learned router so far (dies less, loses head to head);
  - skirmish search;
  - camp feasibility as a global rule (fine on some maps only, hence the per-map plans);
  - shop-first (globally worse; per map it's a candidate).

## Files at a glance

- **Bot:** `bots/main/src/` (`bot.rs` intent pipeline, `act.rs` execution, `utility.rs`, `learned.rs`, `pack.rs`, `map_plans.rs`)
- **Harness:** `harness/`
  - daemons: `coevo.py`, `league_train.py`
  - tools: `speedrun.py`, `map_tune.py`, `mimic.py`, `ladder_study.py`, `pack.py`, `params.py`, `status.py`, `es_tune.py`, `v6_trials.py`
- **Offline RL:** `ml/rl/awr.py`, which uses `ml/distill/train_np.py`
- **State and logs:** `runs/coevo/`, `runs/league/`, `runs/mimic/`, `runs/ladder_*`
