# Autonomous Self-Improvement & Push System

This guide explains the autonomous continuous improvement harness designed for Battlecode. It automates benchmark tracking against the Top 5 teams, ingests losses from the live ladder, runs targeted evolutionary & micro-optimization cycles, and automatically packages, commits, and pushes new best models after every loss.

---

## Quick Reference (How to Run)

The system is automated and simple to run using `train.bat` (or `scripts\train.ps1`):

```cmd
:: 1. Start continuous improvement daemon in the background (with automated platform submission)
train start

:: 2. View live status (current champion, rank, benchmark vs top 5, pending losses, submissions)
train status

:: 3. Run instant benchmark against Top 5 Competitors
train benchmark

:: 4. One-click manual push of current champion (bundles, commits, and submits to Battlecode)
train push

:: 5. Follow live training logs
train logs

:: 6. Run a single cycle in foreground
train run-once

:: 7. Stop background trainers cleanly
train stop
```

*(On Linux / macOS, use `./scripts/train.sh <command>` with the same subcommands.)*

---

## Architecture & How It Works

```
   ┌────────────────────────────────────────────────────────┐
   │                  Ladder Watcher Loop                   │
   │  - Polls Battlecode API for team matches               │
   │  - Detects match losses in real time                   │
   │  - Enqueues loss (map, opponent, timestamps)           │
   └──────────────────────────┬─────────────────────────────┘
                              │
                    Loss detected on map M
                    (or periodic loss sweep)
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │            Targeted Improvement Engine                 │
   │  1. Top-5 Benchmarking:                                │
   │     - Clones macro of live top 5 teams via mimics      │
   │     - Tests candidate against top 5 sparring pool      │
   │  2. Targeted Block-Structured Antithetic ES:           │
   │     - Perturbs θ (102 parameters) on target map M:     │
   │       * Multipliers W (retreat HP, aggression, clear)  │
   │       * Micro Utility Arbiter U (aggro swap, support)  │
   │       * Combat Bias FIGHT & Camp Scorer TARGET         │
   │  3. Adaptive mutation scale (sigma) tracking           │
   └──────────────────────────┬─────────────────────────────┘
                              │
                    Candidate Evaluated
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │               Strict Multi-Gate Verification           │
   │  - Fitness gain over champion: >= +0.04                │
   │  - Head-to-head match vs champion on map M: >= 50%     │
   │  - Out-of-fuel suspended turns: <= 2x champ            │
   │  - Confirmation check on fresh pool opponents          │
   └──────────────────────────┬─────────────────────────────┘
                              │
                 Passed & Proven Superior!
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │              Automated Model Promotion & Push          │
   │  1. Snapshot new champion (bots/auto-vN)               │
   │  2. Update main policy pack (pack.bin) & learned.rs    │
   │  3. Verify release build compiles cleanly              │
   │  4. Bundle submission ZIP (submission/auto-vN.zip)     │
   │  5. Git commit & push                                  │
   │  6. Push to Battlecode platform (`cambc submit`)       │
   │  7. Mark loss resolved in state queue                  │
   └────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Dynamic Top 5 Competitor Benchmarking
- Top 5 teams on the leaderboard (`Intercontinental Fish`, `Nyx`, `Sneaky Bush Campers`, `Univardisty of Shingtonwash`, `Mite`) are tracked dynamically.
- `mimic.py` extracts their draft picks, emblem build paths, and tribute cadences into sparring bots under `bots/mimic-*`.
- The current champion is evaluated against these mimics, and any lost matchup is cataloged to prevent blind spots.

### 2. Loss Ingestion & Queue
- Matches on the Battlecode ladder are polled with exponential backoff to handle HTTP 429 rate limits safely.
- When our team loses a match on the ladder or in top-5 benchmark sparring:
  - The map and opponent strategy are logged in `runs/auto_train/state.json`.
  - The queue prioritizes training compute directly on lost maps.
  - Losses rotate round-robin across cycles so the trainer doesn't get stuck on one map.

### 3. Block-Structured Antithetic Evolutionary Strategy (ES)
Rather than unconstrained random mutations across 102 dimensions:
- **Block 1 (Macro Multipliers $W$)**: Detour limits, retreat HP, fight margins, aggression thresholds.
- **Block 2 (Micro Utility Arbiter $U$)**: Real-time decision scoring (aggro swap away from camps, supporting allies in distress, walking home safely vs frozen recall).
- **Block 3 (Combat & Camp Scorers $FIGHT$, $TARGET$)**: Damage advantage and camp selection biases.
- **Block 4 (Joint $W + U$)**: Macro/micro alignment.

Fitness objective:
$$\text{Fitness} = \text{Win Rate} + 0.25 \times \text{Gem Margin} - 0.01 \times \text{Deaths} - 0.002 \times \text{Suspended Turns}$$

### 4. Automated Push & Platform Submission
Whenever a candidate beats the champion and passes confirmation:
1. Creates snapshot: `bots/auto-v{N}`.
2. Updates `bots/main/src/pack/pack.bin`.
3. Creates release bundle: `submission/auto-v{N}.zip`.
4. Commits and pushes to git branch.
5. Invokes `cambc submit auto-v{N}` to update the active bot on Battlecode.
6. Updates `runs/auto_train/report.md` with timestamps and status.

---

## State & Logs

- **State JSON**: `runs/auto_train/state.json` (champion, pending losses, benchmark scores, submission history)
- **Markdown Report**: `runs/auto_train/report.md` (human-readable dashboard)
- **Log Files**: `runs/auto_train/train.log` and `runs/auto_train/train.err`
