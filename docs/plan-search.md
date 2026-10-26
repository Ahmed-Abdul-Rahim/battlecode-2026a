# Plan search (harness/search)

Per-map plans for the rule bot, found offline: a style, a roster and knob values per map, written
by the draft into draft memory (bots/main/src/settings.rs, pack tag 6). Submitted as platform
version 22 on 2026-10-26.

## Results (2026-10-26)

| | |
|---|---|
| Maps with a plan | 37 of 40 (corners and gemfight kept the defaults; yellow-brick-road's best plan failed the fuel gate) |
| Honest gain (accept on one half of the scoring opponents, score on the other) | +0.214 ± 0.026 win rate |
| Submission build vs the same bot without the table, our rule bots + fresh top 10, all maps, both sides | +0.286 ± 0.016 win rate; 0.863 vs our rule bot (was 0.487) |
| Generalisation to unseen maps (1-NN, gated / weighted kNN, best-on-average, mix and match) | none beats the defaults: unseen maps use the plain rule bot |
| Side (team A vs B) | only broken-curve; its team-A plan tied the map plan on fresh opponents, so not used |
| In-game nets (fight/retreat, tribute timing) | no learnable gain; A+B was −0.155 vs A+rules |

Summaries: data/search/*.json (table, splits, audit, crossmap, mvp, verify). The plan table:
league/plan_table.json.

## Engine 2.0.0 / cambc 4.0.0 (2026-10-26, cloud session)

Every result above was measured on engine 1.x (snapshot ba24f051). The engine, the balance patch
and the bot changes below make them stale: re-measure before trusting the table.

Bot / tooling changes:

| Change | Where | Measured |
|---|---|---|
| SDK refresh: 84 item types, 14 emblem families, camp capture state, `get_item_enum` | sdk/, ledger.rs | builds, tests pass |
| Balance patch in the fallback kit table (live stats still overlay at start-up) | champion.rs | Ren S1 1.1, Tavi S2 1.6 are used directly (dps) |
| Build profiles 14-17: Skirmish/Hunter, Capture/Prospector, Greed, Collectors | shop.rs | in the search space, not yet searched |
| JUMP_HOLES (knob2 30): dash champions path over 1-wide hole gaps | world.rs, act.rs | 48 groups on 5 maps were unreachable on foot (islands 38/46); A/B: runs/search/ab-jump.json |
| TRIBUTE_DETOUR (knob2 31): rewards / weak camps just off the way to an altar | bot.rs | A/B: runs/search/ab-detour.json |
| simd128 int8 kernel | int8.rs | 64x128 layer 20.8k fuel vs 51.1k (only the router / micro / teacher nets use it, all off by default) |
| Draft: neighbour search 40 -> 96 steps | plan_build.rs | first draft turn now has ~10M fuel (draft memory is still 1024 B); the analysis used ~0.8M. K2_DRAFT_PLAN is 0 on every table map, so the plan is unread until the re-audit picks it |
| Harness memory: fuel-only replay reads, streamed games.jsonl, per-candidate index | replay.py, common.py, branch.py | bot memory is flat (23-27 wasm pages over whole games): no leak in the bot |

Order of work (each step invalidates the measurements of the steps after it):

1. code changes to the bot (above), each checked with a paired A/B;
2. `setup` (new snapshot), `baseline`, `audit` (the new knobs and profiles are probes);
3. `verify` for the loss table (absolute win rate per map and per opponent, `--maps` for a subset);
4. `search` on every map, starfish and yellow-brick-road included (their v22 plans: starfish
   -0.13 vs fresh opponents, yellow-brick-road +0.37 with ~9 extra out-of-fuel turns a game);
5. `mvp`, `export`, `verify` on the submission build.

New map pool (70 maps): `python -m harness.league maps` pulls the pool, registers each map's
fingerprint and geometry and retires the old maps; it needs `cambc login` and network access to
2026a.battlecode.co.uk (blocked in the cloud session of 2026-10-26). Then `splits` and the steps
above.

## Commands

    python -m harness.search status | table | sstatus
    python -m harness.search search [--maps a,b] [--hours H]   # per-map search (resumes)
    python -m harness.search crossmap run|report                # generalisation study
    python -m harness.search mvp                                # honest gain + head-to-head
    python -m harness.search export                             # league/plan_table.json + pack.bin
    python -m harness.search verify                             # the submission build vs no table
    python -m harness.search side --map M --side a|b            # side-specific plan

On Windows, scripts\search.ps1 run|status|stop|results|candidate drives all of it (and keeps the
PC awake).

## Running it in a Claude Code cloud session (Linux)

    git checkout feat/plan-search
    bash scripts/cloud_setup.sh          # cambc, opponents, map cache, states, slot pool
    export BC_MAIN=$PWD OPENBLAS_NUM_THREADS=1
    python3 -m harness.search table      # the 40-map table from the saved states

- The opponents (75 mimics / exploiters + our rule bot + the fresh top 10) are gitignored bot
  folders; scripts/cloud_setup.sh unpacks their sources from branch `search-data` and cambc builds
  them on first use.
- Game rows (runs/search/*/games.jsonl, ~190 MB) are not uploaded: any game the cloud needs is
  played again. Games are deterministic, so the same build reproduces them. If the cloud's build
  of the slots differs from the saved snapshot, `check_snapshot` asks for `setup` and every result
  is recomputed under the new snapshot.
- A cloud box has fewer cores than the 16-thread PC: 4 cores ran ~0.8 games/s (5 s a game), so
  a 6k-game map search takes about 2 hours. Set `BCTOOLS_JOBS` to the core count.
- The fresh top-10 mimics and lg-v23-fuel build on first use (`verify` builds any that are
  missing).
