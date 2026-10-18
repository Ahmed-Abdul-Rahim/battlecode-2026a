# Feature engineering and selection

## The schema

`ml/features/schema.yaml` is the single source of truth: **109 features in 11 groups**. Each group is aligned to one SDK query, so dropping a group saves that query's fuel; the measured cost is stored with the group. `python ml/features/gen.py` generates `crates/bc-brain/src/features_gen.rs` and `ml/features/schema_gen.py`, and CI fails if they're stale. The schema hash goes into θ_match, so a bot never applies weights trained on a different schema.

| Group | Fuel | Source |
| --- | ---: | --- |
| self_state | 1,300 | `self_champion()` queries |
| economy | 300 | `global_submitted_gems` + ledger |
| enemies | 2,000 | `vision::champions(Enemy)` + per-enemy queries |
| allies | 1,700 | `vision::champions(Ally)` |
| camps | 2,000 | `vision::camps()` + cached group rewards |
| objectives | 400 | cached distance fields |
| belief | 300 | guest-side belief map |
| team_params | 100 | 8-byte shared memory |
| theta_match | 20 | draft memory |
| champion_vectors | 60 | θ_match champion vectors |
| meta | 100 | round, fuel |

**Baselines come from the literature.** These are the unit-state and relative-position observation designs of the A* Honor of Kings papers (P-001, P-003, P-004). The net-worth and team-sum features follow Hodge et al. (P-007), which is in a Q2 journal and marked as below the bar in papers.md. Features are tagged `inv` (egocentric, map-size invariant, so they transfer from other MOBAs for UDA) or `map` (specific to this game).

## Selection: multi-resolution group backward elimination

This is the approach agreed in the planning review, implemented in `ml/features/select.py`:

```
units ← schema groups
repeat
  for b in bootstraps:  S_b ← floating backward elimination(units)   # SBFS, P-021 / P-020
                         J(S) = score(S) − λ · fuel(S) / fuel(all)  # cost-aware, P-026
  stability ← Nogueira index over {S_b}                              # P-024
  if stability ≥ τ and J still improving:  halve the group size (split survivors, mRMR order, P-022)
  else:                                    stop; the previous level is the stable point
```

- **`score` is pluggable.** `cv_scorer()` is the offline proxy: cross-validated accuracy on logged telemetry. A full 34-match `bctools gate` takes about 20 s, so coarse levels (few units, few evaluations) can use real win rate instead. Write a scorer that rebuilds the bot with a group mask, or stubs the group's features to zero, and calls `bctools.gate.gate`.
- **Fuel at finer levels is shared out by feature count.** The SDK call is really paid once per group, so the in-game confirmation step is what keeps that honest.

## Running it

```powershell
bctools run main baseline --out runs/sp1          # self-play with debug telemetry
bctools dataset runs/sp1 datasets/sp1.npz         # X (int8 features), expert, complex, won
python -m ml.features.select datasets/sp1.npz --target expert --lam 0.2
```

The output is `ml/features/selected/report.json`: the selected indices, plus a per-level table of group size, objective, stability and evaluations. The MCP `feature_selection_report` tool returns it. One run per head: router (`--target expert`) and gate (`--target complex`).

Tests (`tests/test_select.py`) plant an informative group, a noise group, a redundant copy and an expensive noise group. They check the right ones survive, that the search refines below group level, and that it backs off when stability is forced low.
