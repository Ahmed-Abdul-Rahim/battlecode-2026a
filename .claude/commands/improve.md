---
description: One self-improvement iteration on the bot (gate → diagnose → patch on exp/* → gate → PR)
argument-hint: "[focus, e.g. 'tribute timing' or a map name]"
---

Run one iteration of the evaluator-gated self-improvement loop described in
docs/self-improvement.md (FunSearch P-080, Eureka P-081, Reflexion P-082).
Focus (optional): $ARGUMENTS

Rules (non-negotiable):
- Never run `cambc submit`, `cambc match-groups`, `cambc login` or anything that touches the platform.
- Work on a new branch `exp/<short-topic>`; never commit to `main`.
- A human merges. You may open a PR only if the gate passed.
- Edit the canonical code in `crates/` or `bots/main/src/*.rs`, never the vendored `bots/main/src/bc_*/` copies;
  run `bctools sync` after changing `crates/`.

Steps:
1. Baseline: call the `bctools` MCP tool `gate` with candidate `main` (or reuse the newest `list_runs` entry if
   `main` hasn't changed). Note win rate, held-out win rate and the losses.
2. Diagnose the worst losses (and the focus, if given): `match_summary`, `fuel_report`, `render_timeline`,
   then `render_frame` at the rounds where tributes stall or deaths cluster. Look at the images.
3. Check `list_backlog` and open experiment issues so you don't repeat a tried idea.
4. Write a one-paragraph hypothesis: what is wrong, the change, why it should help, expected fuel impact.
5. Implement the smallest change that tests it. Keep per-turn fuel under the budgets in docs/fuel-budget.md.
6. Verify locally: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
   `bctools sync`, `pytest -q`, `ruff check .`.
7. Gate the candidate (`gate` tool). Passing means the rule in docs/self-improvement.md.
8. If it passed: commit, push the branch and open a PR using .github/pull_request_template.md with the gate
   numbers. If it failed: don't open a PR; record the result.
9. Either way, draft an experiment issue (`draft_issue`, show it to the user, file only with their OK) with the
   hypothesis, result and what you learned, citing papers.md IDs where relevant.
