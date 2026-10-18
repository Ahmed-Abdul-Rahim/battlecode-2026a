"""Create the team's labels, seed backlog issues and the Project board with `gh`.

Idempotent: existing labels are updated, issues with an existing title are skipped,
and the board is reused if it exists. Needs `gh auth login` with the `project` scope.

    python scripts/seed_backlog.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

BOARD = "Battlecode 2026a"

LABELS = {
    "type:feature": ("1d76db", "New behaviour, tool or capability"),
    "type:bug": ("d73a4a", "Crash, fuel exhaustion, illegal action or wrong behaviour"),
    "type:experiment": ("5319e7", "Hypothesis tested with the gate or a training run"),
    "type:research": ("0e8a16", "Reading / analysis; record in docs/papers.md"),
    "P0": ("b60205", "Do next"),
    "P1": ("d93f0b", "Important"),
    "P2": ("fbca04", "Nice to have"),
    "deferred": ("cccccc", "Decided to postpone"),
}
for area in [
    "pathing", "combat", "economy", "draft", "brain", "features",
    "champions", "memory", "ml", "tooling", "mcp", "harness",
]:  # fmt: skip
    LABELS[f"area:{area}"] = ("c5def5", f"{area}")

# (title, labels, body)
ISSUES: list[tuple[str, list[str], str]] = [
    (
        "Reach walled camp islands with teleport crystals and dashes",
        ["type:feature", "area:pathing", "P0"],
        "Some camp groups can't be reached on foot (ambush group 8: field distance is None everywhere). "
        "They're skipped for now (bot.rs `group_distance`). Add TP-crystal (3 tiles, 3-turn channel, ignores terrain) "
        "and dash (2 steps, crosses holes) macro-edges: buy a crystal at spawn when the target is an island, then "
        "channel from the nearest reachable tile.\n\nPapers: P-062. Done when: island groups get captured on ambush.",
    ),
    (
        "bots/best snapshot and `bctools promote`",
        ["type:feature", "area:tooling", "P0"],
        "The gate compares against the starter (`bots/baseline`), which we now beat 34/34. Add `bctools promote` "
        "that snapshots `bots/main` into `bots/best` after a merged gate pass, and default the gate to `--best best`.",
    ),
    (
        "More maps for the gate: generate tuning and held-out maps",
        ["type:feature", "area:tooling", "P1"],
        "Matches are deterministic per map and side, so the gate's sample is maps x 2 (34 now). Use `cambc mapeditor` "
        "(or a generator) to add varied maps; keep a disjoint held-out set (Procgen protocol, P-012).",
    ),
    (
        "rushbait-v2 from side B: 454 vs 770 gems",
        ["type:bug", "area:combat", "P1"],
        "After the enemy-on-spawn and engage-stall fixes we win rushbait-v2 from side A but lose from side B. "
        "Diagnose with `render_timeline` / `render_frame` (MCP) on `bctools run main baseline --maps rushbait-v2`.",
    ),
    (
        "Fight/kite tuning against stronger opponents",
        ["type:experiment", "area:combat", "P1"],
        "Combat rules (router.rs `fight_advantage`, margins, kite) are only validated against the starter, which "
        "barely fights. Build or obtain a fighting bot (or an exploiter from the harness) and tune.",
    ),
    (
        "Altar locks: deny enemy tributes, clear enemy locks",
        ["type:feature", "area:economy", "P1"],
        "AltarLock expert is a stub (Hold). Buy (10g) and place locks on altars the enemy tributes at "
        "(team-channel flag 'enemy tributing'); clear enemy locks with 3 basic attacks when we need to tribute.",
    ),
    (
        "Build-order experiments per champion",
        ["type:experiment", "area:economy", "P1"],
        "bc-brain/src/shop.rs: Jungle(Health) first, then Monster(primary), Power... Test Home (3-turn recall), "
        "Durability earlier for melee, Treasure for more than Vesper. Gate each variant.",
    ),
    (
        "First group-SBS feature-selection run per head",
        ["type:experiment", "area:features", "P1"],
        "Self-play with telemetry -> `bctools dataset` -> `python -m ml.features.select ... --target expert` and "
        "`--target complex`. Confirm the top 2 subsets in-game. Papers: P-020, P-021, P-024, P-026.",
    ),
    (
        "Draft exploration temperature for self-play",
        ["type:feature", "area:draft", "P1"],
        "Drafts are deterministic per map, so champion-vector fine-tuning sees too little variety. Add a "
        "temperature/epsilon knob used only in self-play runs.",
    ),
    (
        "Champion-vector fine-tune v1",
        ["type:experiment", "area:champions", "P1"],
        "Needs varied drafts. `python -m ml.champions.finetune runs/`, review with `champion_report`, then compile "
        "vectors + S/C into the draft scorer. Kit-twins (Cassi/Prism, Vendetta/Issac, Ryth/Marcella) must separate. "
        "Papers: P-040, P-041, P-042.",
    ),
    (
        "Tune the leader's team parameters",
        ["type:experiment", "area:memory", "P1"],
        "policy.rs `team_params` (aggression, tribute threshold vs score difference and round) is hand-set. "
        "Sweep it with the gate.",
    ),
    (
        "Distillation on the GPU box and wire learned heads into the bot",
        ["type:feature", "area:ml", "P1"],
        "`ml/distill/train.py` (teacher -> student -> BCW1, untested without torch). Load weights with "
        "`include_bytes!` in bc-brain, use `LearnedGate`/`LearnedRouter` behind the hard overrides, keep the Brain "
        "stage under 15k fuel. Papers: P-051, P-052.",
    ),
    (
        "SIMD int8 kernel (wasm simd128)",
        ["type:feature", "area:brain", "P1"],
        "int8 layers cost ~3 fuel per MAC (64x32 = 5.9k). The engine ships wasm_simd128.h for C++, so simd128 is "
        'supported: use i16x8 dot products via #[target_feature(enable = "simd128")]. Measure with bots/fuelbench.',
    ),
    (
        "Learn FiLM parameters per map in the draft",
        ["type:feature", "area:ml", "P2"],
        "θ_match carries 48 FiLM (γ, β) pairs (identity now). Train a small map-descriptor -> FiLM regressor so "
        "the draft writes map-conditioned parameters. Papers: P-032.",
    ),
    (
        "Learned team-channel protocol for the 6 message bytes",
        ["type:research", "area:memory", "P2"],
        "Replace the hand codec behind `TeamChannel` with a learned discrete protocol. Papers: P-030 (DIAL), P-031 (IMAC).",
    ),
    (
        "UDA: Hokoff ingestion into the feature schema",
        ["type:feature", "area:ml", "P2"],
        "Map Hokoff (P-005) HoK frames onto our `inv`-tagged features (`ml/uda/dann.py: hokoff_to_schema`).",
    ),
    (
        "UDA: DANN alignment HoK -> cambc, then distil",
        ["type:experiment", "area:ml", "P2"],
        "Train with gradient reversal (P-010) on Hokoff (labelled) + our telemetry (unlabelled); distil to int8.",
    ),
    (
        "Read champion kits from SDK metadata",
        ["type:feature", "area:brain", "P2"],
        "sdk.rs `kit()` uses the builtin table from GameRules.md. Build KitInfo from `champion_designs()`, "
        "`base_stats()`, `AbilityInfo` and `passive()` so balance patches flow through.",
    ),
    (
        "Confirm the CI smoke match and make it required",
        ["type:feature", "area:tooling", "P1"],
        "The `smoke` job runs cambc on GitHub's Linux runners with continue-on-error. Once green, drop that flag.",
    ),
    (
        "Opt-in auto-filing of crash / fuel-exhaustion issues from gate runs",
        ["type:feature", "area:tooling", "P2"],
        "When a gate run sees bot faults or exhausted turns, draft a bug issue (map, side, round, bundle) via the "
        "MCP `draft_issue` flow; file only with confirmation.",
    ),
    (
        "Speed up the windowed A* tier",
        ["type:feature", "area:pathing", "P2"],
        "Measured ~3k fuel per expansion; currently capped at 8 and gated on spare fuel. Try a bucket queue (costs "
        "are small integers) instead of BinaryHeap.",
    ),
    (
        "Mamba / SSM belief model",
        ["type:research", "area:brain", "deferred"],
        "Deferred in planning (2026-10-18). Candidates: offline Mamba-2 critic (P-071), in-bot LRU-style recurrence (P-073).",
    ),
    (
        "OpenJev small-model yes/no tool",
        ["type:research", "area:harness", "deferred"],
        "Deferred in planning. Flash 9B fits a 16 GB GPU; weights are CC BY-NC 4.0 (check licence before use).",
    ),
    (
        "Qwen-27B harness: exploiter author + labeller via MCP",
        ["type:feature", "area:harness", "P1"],
        "Owned by the teammate with the 5060 Ti. Contract in harness/README.md: bots can't call an LLM (sandbox, "
        "no network); use the MCP tools to write exploiter bots and label datasets.",
    ),
]


def gh(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8")
    if check and proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.dry_run:
        print(f"{len(LABELS)} labels, {len(ISSUES)} issues, board '{BOARD}'")
        return 0
    for name, (color, desc) in LABELS.items():
        gh("label", "create", name, "--color", color, "--description", desc, "--force")
    existing = {
        i["title"]
        for i in json.loads(gh("issue", "list", "--state", "all", "--limit", "500", "--json", "title"))
    }
    owner = json.loads(gh("api", "user"))["login"]
    boards = json.loads(gh("project", "list", "--owner", owner, "--format", "json"))["projects"]
    board = next((b for b in boards if b["title"] == BOARD), None)
    if board is None:
        board = json.loads(gh("project", "create", "--owner", owner, "--title", BOARD, "--format", "json"))
    repo = json.loads(gh("repo", "view", "--json", "nameWithOwner"))["nameWithOwner"]
    gh("project", "link", str(board["number"]), "--owner", owner, "--repo", repo, check=False)
    created = 0
    for title, labels, body in ISSUES:
        if title in existing:
            continue
        args = ["issue", "create", "--title", title, "--body", body]
        for label in labels:
            args += ["--label", label]
        url = gh(*args).strip().splitlines()[-1]
        gh("project", "item-add", str(board["number"]), "--owner", owner, "--url", url)
        created += 1
    print(json.dumps({"labels": len(LABELS), "issues_created": created, "board": board.get("url")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
