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
- A cloud box has fewer cores than the 16-thread PC: expect a few games per second at most.
