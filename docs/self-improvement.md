# Self-improvement loop

An LLM proposes changes. An automatic evaluator keeps only the ones that win. This is the FunSearch pattern (P-080), with Eureka-style reading of feedback (P-081) and written reflections kept between attempts (P-082). Our evaluator is the local match runner, which plays a 34-match gate in about 20 s. The LLM *sees* matches through rendered frames, the multimodal part (cf. MLES, P-083).

```
gate(main) ──► losses ──► match_summary / fuel_report / render_timeline / render_frame
     ▲                                                  │
     │                                     hypothesis + smallest change (branch exp/*)
     │                                                  │
     └──── PR (human merges) ◄── gate passes ◄── tests / clippy / sync / ruff
                                      │
                                      └── fails → experiment issue with what we learned
```

## Autonomy rule

- **The agent works on an `exp/*` branch.** It may push and open a PR only if the gate passed. **A human always merges.**
- **Nothing touches the platform.** No `cambc submit`, `match-groups`, `login` or `ladder`, from any tool, ever, unless the team decides.
- **Filing issues needs a human OK.** The MCP `draft_issue` tool returns a draft and files only with `confirm=true`.

## Gate

`bctools gate <candidate> --best <bot>` plays every map on both sides. Team B wins full ties, so sides matter. It passes when all three hold:

- win rate ≥ 55%, and
- Wilson 95% lower bound > 50%, and
- held-out win rate ≥ 50% (`maps/heldout/`: contest, rushbait-v2, swirl, packs, monolith). **Never tune against these.**

Matches are deterministic per map and side, so more maps, not more repeats, tighten the interval. Generating extra maps with `cambc mapeditor` is a backlog item.

## Running it with Claude Code

`.mcp.json` registers the `bctools` MCP server for everyone who opens the repo. Then use `/improve [focus]`, which runs one iteration of the loop above (`.claude/commands/improve.md`).

| MCP tool | Use |
| --- | --- |
| `gate`, `run_matches`, `list_runs` | evaluate |
| `match_summary`, `get_logs`, `fuel_report` | read telemetry |
| `render_frame`, `render_timeline`, `champion_report` | look at it (PNG) |
| `feature_selection_report` | feature-selection curve |
| `list_backlog`, `draft_issue` | backlog (confirm before filing) |

## What the loop found on day one

These came from the same tools, applied by hand while building v0:

1. **Spawn-blocked respawn:** standing on a dead group's spawn tile blocks its respawn, and `Respawning(0)` was read as "back next round". This caused a 60-turn stall.
2. **Unreachable groups:** walled islands looked close in straight-line distance. They're now skipped until teleport crystals are implemented.
3. **Long camp groups:** a group's field leads to *any* of its spawn tiles, which can be out of range of the living camps.
4. **Enemies on spawn:** chasing enemies standing on their own spawn (rushbait-v2) is useless, because they heal and can't be reached.
5. **Wall-separated enemies:** chasing an enemy across a wall needed a stall detector.

Win rate against the starter went from 1 of 1 lost (938 vs 1000 gems) to **34 of 34 won**.
