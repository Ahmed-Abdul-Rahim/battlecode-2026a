# Architecture

```
crates/ (SDK-free, host-tested)                     bots/main (cambc bot, wasm32-wasip1)
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   ┌───────────────────────────────────────┐
│ bc-core  │ │ bc-path  │ │ bc-brain │ │ bc-draft │   │ src/bc_*/  ← vendored by `bctools sync` │
│ geometry │ │ fields   │ │ features │ │ map desc │   │ sdk.rs     SDK ⇄ our types            │
│ terrain  │ │ greedy   │ │ gate     │ │ scorer   │   │ world.rs   terrain, groups, fields    │
│ kits     │ │ A* window│ │ router   │ │ θ_match  │   │ bot.rs     observe → decide → log     │
│ ledger   │ │ planner  │ │ experts  │ │          │   │ act.rs     intents → SDK actions      │
│ belief   │ │          │ │ int8+FiLM│ │          │   │ draft.rs   picks + θ_match            │
│ memory   │ │          │ │ champvec │ │          │   │ telemetry  debug_block JSON lines     │
│ fuel     │ │          │ │ shop     │ │          │   └───────────────────────────────────────┘
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**Why vendoring:** cambc requires every bot's `Cargo.toml` to match its template, so a bot can't depend on our crates. `crates/` is canonical; `bots/main/src/bc_*/` are generated copies; CI fails if they drift.

## One champion turn

```
observe (self, vision, camp groups, team channel)          ~13k fuel
  → belief + ledger update; leader writes team params (8-byte shared memory)
  → choose target group: value / (walk + clear time + respawn wait), claims, hysteresis
  → Context
  → meta-gate:  hard overrides (dead, low HP in danger, low fuel) │ routine │ complex
  → router:     routine heuristic, or rule router (v0) / learned int8 head with FiLM (later)
  → top-1 expert → Intent (ClearCamp / Capture / Approach / Tribute / Engage / Kite / Retreat / Shop)
  → executor:   casts before moves (exclude-walking rule), ranged move-or-attack, attack resets, buy order
  → telemetry (debug only): T/E/X/M lines, features every 4th turn
```

## Pathing ensemble (bc-path)

| Tier | When | Cost |
| --- | --- | --- |
| Static | cached host `DistanceField`, free optimal step | ~2k |
| Local repair | optimal steps blocked → best non-worsening neighbour | ~4k |
| Greedy | threat nearby → neighbour minimising distance·10 + danger | ~2k |
| Weighted A* | threat nearby **and** spare fuel → 11×11 window, field as heuristic, ≤ 8 expansions | ≤ 25k |
| Fallback | field not computed yet → straight line | <1k |

Fields (ally spawn, altars, every camp group) cost about 19k each on 100×100. They're computed on the first turn (200k budget), then from spare fuel, with the current target first.

## Decision layer (bc-brain)

- **Meta-gate:** chooses, ahead of time, between the cheap heuristic and the MoE. Rule-based in v0; a `LearnedGate` slot is ready behind the same hard overrides.
- **Sparse MoE:** nine heuristic experts, and the top-1 router runs only one of them (P-050). `LearnedRouter` is an int8 MLP whose hidden units are FiLM-modulated by θ_match (P-032).
- **int8 kernel:** `BCW1` blobs with the schema hash and a CRC. Bit-exact with `ml/distill/quant.py`, checked by a shared fixture.
- **Policies:** tribute control (bag size vs. distance, danger, endgame) and leader team parameters.

## Status (2026-10-18)

- Against the starter bot: **34/34 wins** over 17 maps × 2 sides. Mean fuel about 27k per turn; 0 exhausted turns.
- Learned heads, champion fine-tuning, feature selection and DANN are scaffolded and tested; they're waiting on self-play data and the GPU box.
- See the GitHub backlog for what's next.
