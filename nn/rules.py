"""Search over the rule bot's priority order (which override rule wins), per opponent too.

    python nn/rules.py run        # loop (state in runs/nn/rules/)
    python nn/rules.py status

The bot's override rules (bots/main/src/bot.rs, `RULE_PRIO`) may only replace the goal when
their priority is at least that of the rule that set it; the default (code order) reproduces
the current bot exactly. A candidate is an order of the 20 rules (rule 0 = the expert's plan:
rules placed below it never fire) plus rules switched off. It is byte-patched into prebuilt
bots, so a candidate costs no rebuild.

One generation:
1. Proposals: mutations of the best order (swap, move, switch off/on) and, when the local Qwen
   server answers, orders it proposes from the rule list and the results so far.
2. Stage 1: each candidate plays STAGE1 setups (opponent, map, side) paired with the current
   best on the same setups (deterministic games: the difference is the candidate's effect).
3. Stage 2: the top TOP_K play STAGE2 fresh setups. Stage 3: the best confirms on STAGE3 more.
   Accepted if its paired gain over all its games clears ACCEPT (win-rate points) and the gem
   margin doesn't drop.
Per-opponent paired gains are kept for every evaluated candidate (runs/nn/rules/per_opp.json):
the data for choosing an order per opponent.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bctools import runner  # noqa: E402

from harness import experiments as X  # noqa: E402
from nn import run as R  # noqa: E402

OUT = R.OUT / "rules"
STATE = OUT / "state.json"
MAGIC = bytes([0xC7, 0xD1, 0xE2, 0xF3, 0x0A, 0x0B, 0x0C, 0x0D])
RULES = ["expert plan", "answer a teammate's call for help", "deny an enemy tribute", "punish a lone recall",
         "Tavi hunts carriers", "fight fix (finish kills, last stand)", "utility arbiter (keep/step back/fight/recall)",
         "camp safety (leave camps we can't survive)", "team fight (Lanchester: fight or retreat)",
         "combat search (simulated skirmish)", "kill assist (finish a hurt enemy)", "PvE assist (help an ally's camp)",
         "deliver at an altar on the way", "combat state machine", "team up (help an ally trading hits)",
         "upgrade before a hard fight", "endgame ahead: bank gems", "endgame round: bank gems",
         "unstick (two-tile loop)", "walk home instead of recall"]
N = len(RULES)
POOL = 24
STAGE1, STAGE2, STAGE3 = 16, 48, 96
TOP_K = 4
ACCEPT = 0.03
QWEN_URL = "http://127.0.0.1:11435/v1/chat/completions"
QWEN_SHARE = 0.25  # fraction of proposals asked of Qwen (when it answers)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "rules.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"gen": 0, "best": {"order": list(range(N)), "off": []}, "accepted": [], "tried": 0}


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def prio_bytes(c: dict) -> bytes:
    p = [255] * 32
    for rank, rule in enumerate(c["order"]):
        p[rule] = rank
    for rule in c["off"]:
        if rule != 0:
            p[rule] = 255
    for i in range(N, 32):
        p[i] = i
    return MAGIC + bytes(p)


def patch(bot: str, c: dict) -> None:
    blob, n = prio_bytes(c), 0
    for f in (runner.bot_dir(bot) / ".cambc").rglob("*.wasm"):
        b = bytearray(f.read_bytes())
        i = b.find(MAGIC)
        if i >= 0:
            b[i : i + len(blob)] = blob
            f.write_bytes(b)
            n += 1
    if not n:
        raise RuntimeError(f"{bot}: rule priority block not found")


def key(c: dict) -> str:
    return json.dumps([c["order"], sorted(c["off"])])


def describe(c: dict) -> str:
    base = c["order"].index(0)
    live = [r for r in c["order"][base + 1 :] if r not in c["off"]]
    return " < ".join(str(r) for r in live) + (f" | off {sorted(c['off'])}" if c["off"] else "") + \
        (f" | below plan {c['order'][:base]}" if base else "")


# ---------------------------------------------------------------- proposals


def mutate(c: dict, rng: random.Random, active: list[int]) -> dict:
    """Swap / move / switch off rules among `active` ones (those whose removal changes games;
    reordering rules that never fire together changes nothing). Rule 0 (the plan) may move."""
    o, off = list(c["order"]), set(c["off"])
    movable = [0] + [r for r in active if r != 0]
    for _ in range(rng.choice((1, 1, 2, 3))):
        op = rng.random()
        if op < 0.4 and len(movable) >= 2:  # swap two rules
            a, b = rng.sample(movable, 2)
            i, j = o.index(a), o.index(b)
            o[i], o[j] = o[j], o[i]
        elif op < 0.8:  # move one rule elsewhere
            r = rng.choice(movable)
            o.remove(r)
            o.insert(rng.randrange(N), r)
        elif op < 0.9 and len(off) < 4 and active:
            off.add(rng.choice(active))
        elif off:
            off.discard(rng.choice(sorted(off)))
    return {"order": o, "off": sorted(off)}


def qwen(st: dict, n: int) -> list[dict]:
    """Up to n orders proposed by the local Qwen model; [] when it doesn't answer."""
    rules = "\n".join(f"{i}: {r}" for i, r in enumerate(RULES))
    hist = "\n".join(f"- {a['desc']}: gain {a['gain']:+.3f}" for a in st.get("log", [])[-12:])
    task = (
        "A game bot picks each champion's goal every turn: rule 0 (the expert's plan) proposes a goal, then "
        "override rules may replace it. A rule may replace the goal only if its priority is at least the priority "
        "of the rule that set it; rules ranked below rule 0 never fire. Rules:\n" + rules +
        f"\n\nCurrent best order, lowest to highest priority: {st['best']['order']} (off: {st['best']['off']}).\n"
        "Recent candidates (order of live rules above the plan, lowest first) and their win-rate gain vs the "
        f"best:\n{hist or '- none yet'}\n\n"
        f"Propose {n} new orders likely to win more (e.g. safety before greed, team fights before solo hunts). "
        "Answer ONLY with JSON: a list of objects {\"order\": [20 distinct ints 0-19], \"off\": [ints]}.")
    payload = {"model": "qwen", "stream": False, "temperature": 0.8, "max_tokens": 1500,
               "messages": [{"role": "user", "content": task}]}
    try:
        req = urllib.request.Request(QWEN_URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
        m = re.search(r"\[.*\]", text, re.S)
        out = []
        for c in json.loads(m.group(0)) if m else []:
            o = [int(x) for x in c.get("order", [])]
            if sorted(o) == list(range(N)):
                out.append({"order": o, "off": sorted({int(x) for x in c.get("off", []) if 0 < int(x) < N})[:4]})
        return out[:n]
    except Exception:  # noqa: BLE001 - Qwen is optional
        return []


# ---------------------------------------------------------------- games


def bots() -> tuple[str, list[str]]:
    names = ["_rp_ref"] + [f"_rp{i}" for i in range(POOL)]
    for nm in names:
        X.knob_variant(nm, {})
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(runner.build, names))
    return names[0], names[1:]


def setups(rng: random.Random, n: int) -> list[tuple[str, Path, bool]]:
    train, _ = R.maps()
    opps = R.mimics()
    return [(rng.choice(opps), rng.choice(train), rng.random() < 0.5) for _ in range(n)]


def play(cands: list[dict], sets: list[tuple[str, Path, bool]], ref: str, pool: list[str], best: dict,
         tag: str) -> list[list[tuple[str, float, float]]]:
    """Per candidate (at most len(pool)), per setup: (opponent, score diff vs the best order,
    gem-margin diff). One batch: the best order on _rp_ref, each candidate on its own pool bot."""
    assert len(cands) <= len(pool)
    patch(ref, best)
    bots_ = [ref]
    for c, bot in zip(cands, pool):
        patch(bot, c)
        bots_.append(bot)
    games = [((b, o, m) if a else (o, b, m)) for b in bots_ for o, m, a in sets]
    out = OUT / "games" / tag
    res = runner.run_games(games, out, log_limit=0, timeout=R.GAME_TIMEOUT)
    shutil.rmtree(out, ignore_errors=True)
    val = {}
    for r in res:
        a = r.bot_a in bots_
        bot = r.bot_a if a else r.bot_b
        o = r.bot_b if a else r.bot_a
        if r.winner is None and r.rounds == 0:
            continue  # failed game
        val[(bot, o, r.map, a)] = (r.score_for(bot), (r.gems_a - r.gems_b) * (1 if a else -1))
    out_ = []
    for bot in bots_[1:]:
        row = []
        for o, m, a in sets:
            c, b = val.get((bot, o, m.stem, a)), val.get((ref, o, m.stem, a))
            if c and b:
                row.append((o, c[0] - b[0], c[1] - b[1]))
        out_.append(row)
    return out_


def play_all(cands, sets, ref, pool, best, tag):
    """play() handles at most len(pool) candidates per call."""
    out = []
    for i in range(0, len(cands), len(pool)):
        out += play(cands[i : i + len(pool)], sets, ref, pool, best, f"{tag}-{i}")
    return out


def cmd_run(_: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    st = load_state()
    st.setdefault("log", [])
    per_opp_path = OUT / "per_opp.json"
    per_opp = json.loads(per_opp_path.read_text()) if per_opp_path.exists() else {}
    ref, pool = bots()
    seen = {key(st["best"])}
    if "active" not in st:  # probe: which rules change games when switched off alone
        probe = [{"order": list(range(N)), "off": [r]} for r in range(1, N)]
        sets = setups(random.Random(5), STAGE1)
        rows = play_all(probe, sets, ref, pool, st["best"], "probe")
        # a rule without compared games counts as active (unknown), never as "no effect"
        st["active"] = [r for r, row in zip(range(1, N), rows) if not row or any(x[2] != 0 for x in row)]
        st["probe_games"] = [len(row) for row in rows]
        st["probe"] = {RULES[r]: round(sum(x[1] for x in row) / max(len(row), 1), 3)
                       for r, row in zip(range(1, N), rows)}
        log(f"probe ({sum(st['probe_games'])} paired games): active rules {st['active']} ({[RULES[r] for r in st['active']]}); "
            f"win change when off: {st['probe']}")
        save_state(st)
    while True:
        g = st["gen"]
        rng = random.Random(900 + g)
        best = st["best"]
        n_q = int(POOL * QWEN_SHARE)
        cands = [c for c in qwen(st, n_q) if key(c) not in seen]
        src = ["qwen"] * len(cands)
        while len(cands) < POOL:
            c = mutate(best, rng, st["active"])
            if key(c) not in seen:
                cands.append(c)
                src.append("mut")
                seen.add(key(c))
        t0 = time.time()
        s1 = setups(rng, STAGE1)
        r1 = play_all(cands, s1, ref, pool, best, f"g{g}s1")
        gain = lambda rows: sum(x[1] for x in rows) / max(len(rows), 1)  # noqa: E731
        order = sorted(range(len(cands)), key=lambda i: -gain(r1[i]))
        top = order[:TOP_K]
        s2 = setups(rng, STAGE2)
        r2 = play_all([cands[i] for i in top], s2, ref, pool, best, f"g{g}s2")
        rows = {i: r1[i] + r2[k] for k, i in enumerate(top)}
        champ = max(top, key=lambda i: gain(rows[i]))
        msg = f"gen {g}: stage 1 gains " + " ".join(f"{gain(r1[i]):+.2f}{'q' if src[i] == 'qwen' else ''}" for i in order[:8])
        msg += f" | stage 2 best {gain(rows[champ]):+.3f} ({len(rows[champ])} games)"
        accepted = False
        if gain(rows[champ]) > ACCEPT:
            s3 = setups(rng, STAGE3)
            r3 = play_all([cands[champ]], s3, ref, pool, best, f"g{g}s3")[0]
            allr = rows[champ] + r3
            gm = sum(x[2] for x in allr) / len(allr)
            msg += f" | confirm {gain(r3):+.3f}, all {gain(allr):+.3f} over {len(allr)}, margin {gm:+.1f}"
            if gain(allr) > ACCEPT and gm >= 0:
                accepted = True
                st["best"] = cands[champ]
                st["accepted"].append({"gen": g, "cand": cands[champ], "gain": gain(allr), "games": len(allr),
                                       "src": src[champ], "desc": describe(cands[champ])})
        log(msg + (f" ACCEPTED ({src[champ]}): {describe(cands[champ])}" if accepted else "") +
            f" [{time.time() - t0:.0f}s]")
        for i, c in enumerate(cands):
            rr = r1[i] + (rows[i][len(r1[i]):] if i in rows else [])
            st["log"].append({"gen": g, "desc": describe(c), "gain": gain(rr), "games": len(rr), "src": src[i]})
            d = per_opp.setdefault(key(c), {})
            for o, sd, _ in rr:
                d.setdefault(o, []).append(sd)
        st["log"] = st["log"][-400:]
        st["gen"] = g + 1
        st["tried"] += len(cands)
        save_state(st)
        per_opp_path.write_text(json.dumps(per_opp))


def cmd_status(_: argparse.Namespace) -> int:
    st = load_state()
    print(json.dumps({"gen": st["gen"], "tried": st["tried"], "best": describe(st["best"]),
                      "accepted": st["accepted"][-5:]}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
