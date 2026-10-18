"""Strategy-variant tournament: copies of `main` with text patches, played vs a fixed pool.

Each variant is `bots/main` copied to `bots/_var-<name>/` with exact-string patches
applied to its sources (vendored `bc_*` copies included, so crate logic can be
varied without touching `crates/`). Every variant plays the same opponents on all
maps, both sides, in release mode, and lands in one ranked table.

    python harness/variants.py --set builds              # emblem build orders
    python harness/variants.py --set builds --only base treasure-first
    python harness/variants.py --set roles

Score = win (1/0, 0.5 unfinished) + 0.25 × normalised gem margin, averaged over all
matches; self-play vs the `--best` snapshot (default main-v2) is reported separately.
Local only: nothing here touches the Battlecode platform.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from league_gate import suspended

from bctools import ROOT, gate, runner

MARGIN_WEIGHT = 0.25
POOL = ["baseline", "claude-focus", "claude-pillager"]
#: Bot the variants are copied from (--src).
SRC = "main"
SHOP = "src/bc_brain/shop.rs"

# ----- emblem builds ---------------------------------------------------------

BUILD_HEAD = "pub fn build_order(kind: ChampionKind) -> Vec<(Family, Stat)> {\n"
BUILD_BODY = """    let primary = primary_stat(kind);
    let mut order = vec![(Family::Jungle, Stat::Health)];
    if kind == ChampionKind::VesperVale {
        // +20% gold stacks with her passive for +40%.
        order.push((Family::Treasure, Stat::Strength));
    }
    order.push((Family::Monster, primary));
    order.push((Family::Power, primary));
    order.push((Family::Durability, Stat::Health));
    order.extend([(Family::Power, primary); 6]);
    order
}"""


def build(first: list[str]) -> list[tuple[str, str, str]]:
    """Put these emblems (Rust expressions) first, then the current order."""
    pushes = "".join(f"    order.push({e});\n" for e in first)
    new = (
        "    let primary = primary_stat(kind);\n"
        "    let mut order = Vec::new();\n" + pushes + "    order.push((Family::Jungle, Stat::Health));\n"
        "    order.push((Family::Monster, primary));\n"
        "    order.push((Family::Power, primary));\n"
        "    order.push((Family::Durability, Stat::Health));\n"
        "    order.extend([(Family::Power, primary); 6]);\n"
        "    order\n}"
    )
    return [(SHOP, BUILD_HEAD + BUILD_BODY, BUILD_HEAD + new)]


BOT = "src/bot.rs"
DRAFT = "src/draft.rs"
ROUTER = "src/bc_brain/router.rs"

# ----- gold/gem roles (target value in choose_target) --------------------------

VALUE = "            let mut value = g.gems as f32 + g.gold as f32 / 3.0;"


def value(expr: str) -> list[tuple[str, str, str]]:
    """Replace the camp-group value with `expr` (may use g, round, self.slot)."""
    return [(BOT, VALUE, f"            let mut value = {expr};")]


def phase(r: int) -> str:
    return f"if round < {r} {{ g.gems as f32 * 0.3 + g.gold as f32 }} else {{ g.gems as f32 + g.gold as f32 * 0.1 }}"


# ----- draft ------------------------------------------------------------------

PICK = "        if let Some(choice) = best_pick(&c.desc, &state, &vec_of, &c.weights) {"
RANGED = "[DraftChoice::Stellar, DraftChoice::Prism, DraftChoice::VesperVale, DraftChoice::Tavi, DraftChoice::Willow]"
MELEE = (
    "[DraftChoice::Varg, DraftChoice::Vendetta, DraftChoice::Ren, DraftChoice::Ryth, DraftChoice::Marcella]"
)


def prefer(list_expr: str) -> list[tuple[str, str, str]]:
    """Pick the first not-yet-taken champion from a list, else the scorer's choice."""
    new = f"""        let prefs: &[DraftChoice] = &{list_expr};
        let forced = prefs.iter().copied().find(|ch| !state.ally.contains(ch));
        if let Some(choice) = forced.or_else(|| best_pick(&c.desc, &state, &vec_of, &c.weights)) {{"""
    return [(DRAFT, PICK, new)]


COUNTER = f"""        let is_ranged = |ch: &DraftChoice| {{
            matches!(
                ch,
                DraftChoice::Stellar | DraftChoice::Prism | DraftChoice::VesperVale | DraftChoice::Tavi | DraftChoice::Willow
            )
        }};
        let n_ranged = state.enemy.iter().filter(|ch| is_ranged(ch)).count();
        let n_melee = state.enemy.len() - n_ranged;
        let prefs: &[DraftChoice] = if n_ranged >= 2 && n_ranged > n_melee {{
            &{MELEE}
        }} else if n_melee >= 2 && n_melee > n_ranged {{
            &{RANGED}
        }} else {{
            &[]
        }};
        let forced = prefs.iter().copied().find(|ch| !state.ally.contains(ch));
        if let Some(choice) = forced.or_else(|| best_pick(&c.desc, &state, &vec_of, &c.weights)) {{"""

# ----- squads (pairs join each other's camp group) -----------------------------

CLAIM = """                if others > 0 {
                    // Big groups go faster together; small ones are wasted on two."""


def squads(join: float) -> list[tuple[str, str, str]]:
    new = f"""                let buddy = chan.claims.get((self.slot ^ 1) as usize).copied().flatten() == Some(pi);
                if buddy {{
                    value *= {join};
                }} else if others > 0 {{
                    // Big groups go faster together; small ones are wasted on two."""
    return [(BOT, CLAIM, new)]


P2_ON = (
    "    if ctx.gems > 0 && ctx.gold >= ctx.knobs.recall_gold && !ctx.at_spawn && spawn_on_the_way && quiet {"
)


GATE = "src/bc_brain/gate.rs"

# ----- macro: stay with allies (R1), contest penalty (R2), remote farmer (R3) ----

R1 = [
    (
        GATE,
        """    if ctx.low_hp() && ctx.in_danger() {
        return Some(Mode::Override(Expert::Retreat));
    }""",
        """    if ctx.low_hp() && ctx.in_danger() {
        // Covered by allies and winning the fight: let the router decide (fight / step back).
        let covered = ctx.hp_frac() > 0.2
            && ctx.ally_dps_near > 0.0
            && ctx.enemy.is_none_or(|e| crate::bc_brain::router::fight_advantage(ctx, &e) >= 1.5);
        if !covered {
            return Some(Mode::Override(Expert::Retreat));
        }
    }""",
    ),
    (
        ROUTER,
        "    if ctx.low_hp() && !ctx.at_spawn {",
        "    if ctx.low_hp() && !ctx.at_spawn && (ctx.ally_dps_near <= 0.0 || ctx.hp_frac() < 0.2) {",
    ),
    (
        ROUTER,
        "    if ctx.gold >= ctx.knobs.recall_gold && ctx.gems < 10 && far_from_spawn {",
        """    let mid_camp = ctx.target.is_some_and(|t| t.state == GroupState::Alive && t.dist <= 3);
    if ctx.gold >= ctx.knobs.recall_gold && ctx.gems < 10 && far_from_spawn && !mid_camp {""",
    ),
]

STRENGTH = "            value /= 1.0 + g.max_strength / 100.0;"


def contest(per_sighting: float) -> list[tuple[str, str, str]]:
    return [
        (
            BOT,
            STRENGTH,
            STRENGTH
            + f"""
            // Contested: enemies keep showing up here. Spread out to quieter groups.
            let contest = self.belief.recent(round, 40).filter(|s| s.pos.cheb(g.center) <= 8).count();
            value /= 1.0 + {per_sighting} * contest as f32;""",
        )
    ]


SCORE = "            let mut score = value / (dist + clear + (eta - dist).max(0.0) + 5.0);"
R3 = [
    (
        BOT,
        SCORE,
        """            // Remote farmer: the last slot of a 4+ team goes further for quieter groups.
            let remote = self.team_size >= 4 && self.slot as usize + 1 == self.team_size;
            let dist_w = if remote { 0.4 } else { 1.0 };
            let mut score = value / (dist * dist_w + clear + (eta - dist).max(0.0) + 5.0);""",
    )
]


def full_build(entries: list[str]) -> list[tuple[str, str, str]]:
    """Replace the whole build order with these emblems (Rust expressions)."""
    nl = chr(10)
    pushes = "".join(f"    order.push({e});{nl}" for e in entries)
    head = f"    let primary = primary_stat(kind);{nl}    let mut order = Vec::new();{nl}"
    new = head + pushes + f"    order{nl}}}"
    return [(SHOP, BUILD_HEAD + BUILD_BODY, BUILD_HEAD + new)]


P, PW, J, M, BR = (
    "(Family::Power, primary)",
    "(Family::Power, primary)",
    "(Family::Jungle, primary)",
    "(Family::Monster, primary)",
    "(Family::Bruiser, primary)",
)

THETA = "src/bc_draft/theta.rs"
MEMORY = "src/bc_core/memory.rs"
TRIB_MIN = "    k[knob::TRIBUTE_MIN_GEMS] = (15.0 + walk * 0.25).round() as u8;"
TRIB_MAX = "        (TRIBUTE_MAX_GEMS, 80),"


def tribute(base: float, per_walk: float, cap: int) -> list[tuple[str, str, str]]:
    """Fewer, bigger altar trips: higher minimum bag (per map walk) and always-bank cap."""
    return [
        (THETA, TRIB_MIN, f"    k[knob::TRIBUTE_MIN_GEMS] = ({base} + walk * {per_walk}).round() as u8;"),
        (MEMORY, TRIB_MAX, f"        (TRIBUTE_MAX_GEMS, {cap}),"),
    ]


def defaults(**knobs: int) -> list[tuple[str, str, str]]:
    """Set strategy-knob defaults in the vendored memory.rs (bots/main-maps has the knobs)."""
    return [(MEMORY, f"        ({k}, 0),", f"        ({k}, {v}),") for k, v in knobs.items()]


SETS: dict[str, dict[str, list[tuple[str, str, str]]]] = {
    "builds": {
        "base": [],
        "treasure-first": build(["(Family::Treasure, primary)"]),
        "monster-first": build(["(Family::Monster, primary)"]),
        "power-first": build(["(Family::Power, primary)", "(Family::Power, primary)"]),
        "durable-first": build(["(Family::Durability, Stat::Health)"]),
        "home-first": build(["(Family::Home, Stat::Health)"]),
        "bruiser-first": build(["(Family::Bruiser, primary)"]),
    },
    "roles": {
        "base": [],
        "all-gems": value("g.gems as f32 + g.gold as f32 / 10.0"),
        "gold-heavy": value("g.gems as f32 + g.gold as f32 / 1.5"),
        "gold-then-gems-500": value(phase(500)),
        "gold-then-gems-800": value(phase(800)),
        "split-2-gold": value(
            "if self.slot < 2 { g.gems as f32 * 0.3 + g.gold as f32 } else { g.gems as f32 + g.gold as f32 * 0.15 }"
        ),
    },
    "macro": {
        "base": [],
        "r1-stay": R1,
        "r2-contest": contest(0.5),
        "r2-contest-strong": contest(1.5),
        "r3-remote": R3,
        "r1-r2": R1 + contest(0.5),
        "r1-r2-r3": R1 + contest(0.5) + R3,
    },
    "builds2": {
        "base": [],
        "all-primary": full_build([J, M] + [P] * 9),
        "outlast-like": full_build([P, BR, P, P, J] + [P] * 6),
        "cache-like": full_build([P, J, M] + [P] * 8),
        "monster-primary": full_build([M, J] + [P] * 9),
    },
    "tribute": {
        "base": [],
        "bag-30": tribute(25.0, 0.4, 120),
        "bag-45": tribute(40.0, 0.5, 160),
        "bag-small": tribute(10.0, 0.15, 60),
    },
    "combo": {
        "base": [],
        "c-squad": defaults(SQUAD_JOIN_X10=16),
        "c-power": defaults(BUILD_PROFILE=5),
        "c-squad-power": defaults(SQUAD_JOIN_X10=16, BUILD_PROFILE=5),
        "c-squad-power-gems": defaults(SQUAD_JOIN_X10=16, BUILD_PROFILE=5, VALUE_PROFILE=1),
        "c-squad-power-remote": defaults(SQUAD_JOIN_X10=16, BUILD_PROFILE=5, REMOTE_FARMER=1),
        "c-squad-allprimary": defaults(SQUAD_JOIN_X10=16, BUILD_PROFILE=2),
        "c-squad-outlast": defaults(SQUAD_JOIN_X10=16, BUILD_PROFILE=3),
        "c-hunt-always": defaults(HUNT_MODE=1),
        "c-hunt-trigger": defaults(HUNT_MODE=2),
        "c-squad-power-hunt2": defaults(SQUAD_JOIN_X10=16, BUILD_PROFILE=5, HUNT_MODE=2),
        "c-bag30": tribute(25.0, 0.4, 120),
        "c-squad-power-bag30": defaults(SQUAD_JOIN_X10=16, BUILD_PROFILE=5) + tribute(25.0, 0.4, 120),
        "c-squad-power-bag45": defaults(SQUAD_JOIN_X10=16, BUILD_PROFILE=5) + tribute(40.0, 0.5, 160),
        "c-all-bag30-hunt2": defaults(SQUAD_JOIN_X10=16, BUILD_PROFILE=5, HUNT_MODE=2) + tribute(25.0, 0.4, 120),
    },
    "new": {
        "base": [],
        "n-deny": defaults(DENY_TRIBUTE=1),
        "n-endgame": defaults(ENDGAME_BANK=1),
        "n-deny-endgame": defaults(DENY_TRIBUTE=1, ENDGAME_BANK=1),
        "n-deny-endgame-bag30": defaults(DENY_TRIBUTE=1, ENDGAME_BANK=1) + tribute(25.0, 0.4, 120),
        "n-deny-endgame-hunt2": defaults(DENY_TRIBUTE=1, ENDGAME_BANK=1, HUNT_MODE=2),
    },
    "fight": {
        "base": [],
        "f-teamfight": defaults(TEAM_FIGHT=1),
        "f-teamfight-deny-endgame": defaults(TEAM_FIGHT=1, DENY_TRIBUTE=1, ENDGAME_BANK=1),
        "f-teamfight-hunt2": defaults(TEAM_FIGHT=1, HUNT_MODE=2),
    },
    "search": {
        "base": [],
        "s-search": defaults(COMBAT_SEARCH=1),
        "s-search-bag30": defaults(COMBAT_SEARCH=1) + tribute(25.0, 0.4, 120),
        "s-search-power-bag30": defaults(COMBAT_SEARCH=1, BUILD_PROFILE=5) + tribute(25.0, 0.4, 120),
        "s-reddragon-build": defaults(BUILD_PROFILE=10),
        "s-power-bag30": defaults(BUILD_PROFILE=5) + tribute(25.0, 0.4, 120),
    },
    "draft": {
        "base": [],
        "all-ranged": prefer(RANGED),
        "all-melee": prefer(MELEE),
        "double-body": prefer("[DraftChoice::CassiAndIssac]"),
        "counter": [(DRAFT, PICK, COUNTER)],
    },
    "squads": {
        "base": [],
        "squads-1.6": squads(1.6),
        "squads-2.5": squads(2.5),
    },
}


def make_variant(name: str, patches: list[tuple[str, str, str]]) -> str:
    bot = f"_var-{name}"
    d = ROOT / "bots" / bot
    if d.exists():
        shutil.rmtree(d / "src", ignore_errors=True)
        shutil.copytree(ROOT / "bots" / SRC / "src", d / "src")
    else:
        shutil.copytree(ROOT / "bots" / SRC, d, ignore=shutil.ignore_patterns(".cambc", "target"))
    for rel, old, new in patches:
        f = d / rel
        text = f.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise ValueError(f"variant {name}: patch anchor not found exactly once in {rel}")
        f.write_text(text.replace(old, new), encoding="utf-8")
    return bot


def score(results: list[runner.MatchResult], bot: str) -> float:
    s = 0.0
    for r in results:
        ours, theirs = (r.gems_a, r.gems_b) if r.bot_a == bot else (r.gems_b, r.gems_a)
        s += r.score_for(bot) + MARGIN_WEIGHT * (ours - theirs) / max(ours + theirs, 1)
    return s / max(len(results), 1)


def farming_speed(results: list[runner.MatchResult], bot: str) -> dict:
    """How fast we farm: mean rounds to win, team monster damage and gold per round."""
    won = [r.rounds for r in results if r.winner == bot]
    mon = gold = 0.0
    for r in results:
        try:
            s = json.loads(Path(r.stats).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        us = "teamA" if r.bot_a == bot else "teamB"
        ours = [c["metrics"] for c in s.get("champions", []) if c["team"] == us]
        mon += sum(m["monsterDamageDone"] for m in ours) / max(r.rounds, 1)
        gold += sum(float(m["earnedGold"]) for m in ours) / max(r.rounds, 1)
    n = max(len(results), 1)
    return {
        "rounds_to_win": round(sum(won) / len(won)) if won else None,
        "monster_per_round": round(mon / n, 1),
        "gold_per_round": round(gold / n, 2),
    }


def evaluate(bot: str, best: str, pool: list[str], out: Path) -> dict:
    held = gate.heldout_names()
    row: dict = {"variant": bot}
    allres = []
    for opp in [best, *pool]:
        res = runner.run_matches(bot, opp, out_dir=out / opp, swap=True, release=True)
        allres += res
        h = [r.score_for(bot) for r in res if r.map in held]
        row[opp] = {
            "win": round(sum(r.score_for(bot) for r in res) / len(res), 3),
            "heldout": round(sum(h) / len(h), 3) if h else None,
            "score": round(score(res, bot), 4),
        }
        if opp == "baseline":
            row["speed"] = farming_speed(res, bot)
    row["score_all"] = round(score(allres, bot), 4)
    row["suspended"] = suspended(allres, bot)
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=list(SETS))
    p.add_argument("--sets", nargs="*", choices=list(SETS), help="run several sets in a row")
    p.add_argument("--only", nargs="*")
    p.add_argument("--best", default="main-v3")
    p.add_argument("--src", default="main", help="bot the variants are copied from")
    p.add_argument("--pool", nargs="*", default=POOL)
    a = p.parse_args(argv)
    global SRC
    SRC = a.src
    for name in a.sets or [a.set]:
        run_set(name, a)
    return 0


def run_set(set_name: str, a: argparse.Namespace) -> None:
    out = ROOT / "runs" / "variants" / f"{set_name}-{time.strftime('%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for name, patches in SETS[set_name].items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        row = evaluate(make_variant(name, patches), a.best, a.pool, out / name)
        row["secs"] = round(time.time() - t0)
        table.append(row)
        print(
            f"[{set_name}] {name:20s} all {row['score_all']:.4f}  vs {a.best} win {row[a.best]['win']:.3f} "
            f"held {row[a.best]['heldout']}  speed {row.get('speed')}  susp {row['suspended'][0]}",
            flush=True,
        )
        (out / "table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    table.sort(key=lambda r: -r["score_all"])
    (out / "table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps([(r["variant"], r["score_all"], r[a.best]["win"]) for r in table], indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
