# Memory layouts (both memories are model parameters)

Codecs live in `crates/bc-core/src/memory.rs`, with round-trip and corruption tests. Change the code and this file together, and bump the version byte whenever a layout changes.

## Draft memory: θ_match (1024 bytes, written in draft, read-only in game)

The draft bot sees the whole map with 10× fuel. It writes a per-match parameter vector that every champion bot reads once on its first turn. This is where map-specific adaptation happens (the UDA angle: one model, conditioned per map).

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 2 | magic `BC` |
| 2 | 1 | layout version (1) |
| 3 | 1 | team size |
| 4 | 2 | feature schema hash (u16 LE): must match `features_gen::SCHEMA_HASH` |
| 6 | 2 | weights hash (u16 LE) |
| 8 | 8 | role per draft pick: 0 flex, 1 frontline, 2 ranged, 3 support |
| 16 | 1 | number of camp-priority entries (≤ 15) |
| 18 | 30 | camp priority: (group index, score 1–255) × 15, best first |
| 48 | 96 | **FiLM** (γ, β) int8 × 48 hidden units; scale = 1 + γ/64, identity = (0, 0) (P-032) |
| 144 | 32 | expert knobs (`memory::knob`): aggression, retreat HP %, tribute min/max/distance scale, monolith power, recall gold, fight margin |
| 176 | 160 | **champion vectors** 10 × 16 int8: allies in slots 0–4, enemies in slots 5–9 |
| 336 | 10 | champion kind per slot (0xFF = empty) |
| 352 | 192 | map-descriptor cache (normalised map features × 255) |
| 544 | 478 | reserved (zero) |
| 1022 | 2 | CRC-16/CCITT-FALSE of bytes 0..1022 |

All-zero memory, which happens if the draft bot crashed or never wrote, decodes as `Empty`. Bots then fall back to `ThetaMatch::default()`.

## Shared memory: team channel (8 bytes, read/write all match)

Allies act one after another within a round, in priority order, so later allies see what earlier ones wrote this round.

Layout version **2** (feat/combat, 2026-10-20). Bytes 3–7 are one 40-bit little-endian field.

| Byte | Bits | Field |
| ---: | --- | --- |
| 0 | 0–1 | version (0 = never written; 2 = this layout) |
| 0 | 2 | parity of the round the leader last wrote |
| 0 | 3–5 | leader slot |
| 0 | 6 | flag: the sighting is an enemy tributing |
| 0 | 7 | flag: help requested. With COMBAT_FSM the sighting **is** a help call (bits 36–39) |
| 1 | – | **team aggression parameter** (0–255) |
| 2 | – | **team tribute threshold** (gems) |
| 3–7 | 0–17 | camp claim per ally slot, **3 bits** × 6: θ_match priority index + 1 (0 = none; only the 7 best camps can be claimed) |
| 3–7 | 18–33 | latest enemy sighting x:7 y:7 r:2; r = round mod 4; (0,0) = none |
| 3–7 | 34–35 | **sighted enemy health** in quarters (0 < 25 %, 3 ≥ 75 % or unknown) |
| 3–7 | 36–37 | **help-call intent**: 0 FIGHT (come attack), 1 ESCAPE (retreating with loot: cover my way home), 2 FINISH (enemy almost dead), 3 ABORT (lost or dead: don't come) |
| 3–7 | 38–39 | **help-call time left**: 3 = 10+ turns, 2 = 5–9, 1 = 2–4, 0 = none |

**Help calls (knob COMBAT_FSM, `bc_core::combat`).** A champion's state machine writes the call: FIGHT while engaged and not clearly winning, FINISH when the enemy is under 25 %, ESCAPE when disengaging while chased, ABORT when it died or can't be saved. Calls live with the sighting's age (2 rounds) instead of being cleared by the leader, so allies acting earlier in the round see them. A fresh call is only replaced by a more urgent one that can still be saved, and an ABORT only cancels a call at its own position. Helpers go only if their travel time fits the time left (FINISH: within 3 turns), never join a FIGHT/FINISH with a big bag, and cover an ESCAPE by standing on the caller's way home and fighting only the chaser. With the knob off, the old 1-bit behaviour is unchanged (the leader clears the flag each round).

**Leader protocol.** The first ally to act in a round sees a stale parity bit and becomes the leader. It clears the per-round flags, ages out sightings older than 2 rounds, and writes bytes 1–2 from `bc_brain::policy::team_params`. When we're ahead it raises aggression less and banks more often; when we're behind late in the match it pushes harder. Every bot reads all 8 bytes as features (`team_params` group) and as inputs to the gate and router.

**TEAM_RESPAWN (knob 30, on since v12).** Bytes 1–2 instead carry a *respawn broadcast*: byte 1 = camp group + 1 that someone just saw go down (0 = none), byte 2 = its respawn round mod 256 (unwrapped to the window [round − 100, round + 155]). Allies load it into their respawn belief, so they stop walking to camps a teammate cleared. Aggression and tribute threshold are then computed locally by every bot (`Bot::team_view`); they are a pure function of public state.

Bytes 3–7 go through `TeamChannel`. A learned discrete protocol (DIAL P-030, with an IMAC entropy budget P-031) can replace this hand-written codec later.
