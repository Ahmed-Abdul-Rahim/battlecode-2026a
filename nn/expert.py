"""Expert iteration: search with high fuel, then distill the search into the teacher.

    python nn/expert.py run       # loop (state in runs/nn/expert/)
    python nn/expert.py status

One iteration:
1. Base games: the current policy (the rules, then the latest accepted teacher) plays full games
   against the opponent pool and logs every goal decision.
2. Search: at sampled decisions, branch games force a different goal (the bot reads a block
   that this script byte-patches into the compiled file, so a branch costs no rebuild). The
   game is deterministic, so a branch replays the base game exactly up to that decision.
   A branch plays on for H rounds; its value is the team gem difference (submitted + half
   of carried) at that round minus the base game's at the same round.
3. Labels: where an alternative beat the policy's own choice by more than MARGIN, the
   decision is relabelled to that alternative (and oversampled).
4. Distill: fine-tune the teacher on the policy's own decisions + the improved labels.
5. Evaluate against the rule bot (80 games); accept if better than the best so far. The next
   iteration searches around the accepted teacher (policy iteration).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import struct
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bctools import runner  # noqa: E402

from nn import data  # noqa: E402
from nn import run as R  # noqa: E402

import os

# EXPERT_TAG / EXPERT_H / EXPERT_VALUE: a separate run (nn/gemcheck.py: 150 rounds of the blend
# barely predicted the final result, rank corr 0.12; 500 rounds of submitted gems 0.52)
OUT = R.OUT / os.environ.get("EXPERT_TAG", "expert")
STATE = OUT / "state.json"
MAGIC = bytes.fromhex("e5c4a1b70df0ed5e")
POOL = 32  # branch bots (one batch of branch games)
BASE_GAMES = 24
DECISIONS = 4  # searched decisions per base game
ALTS = 3  # alternatives per decision
H = int(os.environ.get("EXPERT_H", 150))  # rounds a branch plays after its decision
COMMIT = 20  # rounds the forced intent is kept (while available)
VERIFY_COMMIT = 10  # the re-check of a winner holds it this long instead
GOLD_PER_GEM = 0.1
MARGIN = 4.0  # value units (gems) an alternative must win by
OVERSAMPLE = 8
NULL_EVERY = 6  # every 6th base game also gets a null branch (forced = own choice): must give 0
KIND = {"ryth": 0, "vespervale": 1, "vendetta": 2, "cassi": 3, "issac": 4, "stellar": 5, "prism": 6,
        "marcella": 7, "willow": 8, "ren": 9, "varg": 10, "tavi": 11, "heartofruin": 12}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "expert.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"it": 0, "weights": None, "best": None, "evals": [], "search": []}


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


# ---------------------------------------------------------------- bots


def policy_bots(weights: str | None) -> tuple[str, list[str]]:
    """The base bot and the branch pool, all the same policy (rules: TEACHER 1, teacher: 2)."""
    mode, w = (1, None) if weights is None else (2, Path(weights))
    base = R.variant("_xbase", mode, 0, w)
    pool = [R.variant(f"_xb{i}", mode, 0, w) for i in range(POOL)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(runner.build, [base, *pool]))
    return base, pool


def patch(bot: str, cfg: tuple[int, ...]) -> None:
    words = struct.pack("<8I", 0xB7A1C4E5, 0x5EEDF00D, *cfg, *([0] * (6 - len(cfg))))
    n = 0
    for f in (runner.bot_dir(bot) / ".cambc").rglob("*.wasm"):
        b = bytearray(f.read_bytes())
        i = b.find(MAGIC)
        if i >= 0:
            b[i : i + 32] = words
            f.write_bytes(b)
            n += 1
    if n == 0:
        raise RuntimeError(f"{bot}: branch block not found in the compiled file")


# ---------------------------------------------------------------- games


def decisions(bundle: Path, side: str) -> list[dict]:
    out = []
    for ln in data._lines(bundle, side):
        if ln.startswith("N "):
            try:
                o = json.loads(ln[2:])
            except json.JSONDecodeError:
                continue
            out.append({"r": o["r"], "c": data._norm(o["c"]), "a": o["a"][0], "m": o["m"][0]})
    return out


def scores(bundle: Path, side: str) -> list[tuple[int, float]]:
    """(round, team gem difference after that turn) from our bot's own score lines ("S", every
    turn): submitted gems (ours - theirs) + half of our carried gems + our earned gold / 10 (a
    wolf pays 40 gold and 4 gems; early branches rarely change gems within H rounds). Truncated games have no
    statistics file, so the bot's log is the source; base and branch games use the same one."""
    out: list[tuple[int, float]] = []
    carried: dict[str, float] = {}
    gold: dict[str, float] = {}
    for ln in data._lines(bundle, side):
        if ln.startswith("S "):
            o = json.loads(ln[2:])
            carried[o["c"]] = float(o["g"])
            gold[o["c"]] = float(o.get("lg", 0))
            v = float(o["us"]) - float(o["them"])
            if os.environ.get("EXPERT_VALUE", "blend") == "blend":
                v += 0.5 * sum(carried.values()) + GOLD_PER_GEM * sum(gold.values())
            out.append((int(o["r"]), v))
    return out


def value_at(sc: list[tuple[int, float]], t: int) -> float:
    v = 0.0
    for r, x in sc:
        if r > t:
            break
        v = x
    return v


def value(bundle: Path, side: str, t: int) -> float:
    return value_at(scores(bundle, side), t)


def base_games(st: dict, base: str, rng: random.Random) -> list[dict]:
    train, _ = R.maps()
    opps = ["main", *R.ANCHORS, *R.mimics()]
    setups = []
    for _ in range(BASE_GAMES):
        setups.append((rng.choice(opps), rng.choice(train), rng.random() < 0.5))
    games = [(base, o, m) if a else (o, base, m) for o, m, a in setups]
    out = OUT / "games" / f"base-{st['it']}"
    res = runner.run_games(games, out, log_limit=R.LOG_LIMIT, fuel=R.FUEL[512], timeout=R.GAME_TIMEOUT)
    games_ = []
    for (o, m, a), r in zip(setups, res):
        side = "a" if a else "b"
        if r.winner is None and r.rounds == 0:
            continue
        try:
            dec = decisions(Path(r.replay), side)
        except (OSError, KeyError, zipfile.BadZipFile):
            continue
        games_.append({"opp": o, "map": m, "a": a, "side": side, "res": r, "dec": dec, "rounds": r.rounds})
    return games_


def search(st: dict, games: list[dict], pool: list[str], rng: random.Random) -> list[dict]:
    """Branch specs, played in batches of POOL; returns one record per searched decision."""
    specs = []
    for gi, g in enumerate(games):
        cand = [d for d in g["dec"] if 20 <= d["r"] <= g["rounds"] - H - 5
                and bin(d["m"] & ~(1 << d["a"])).count("1") >= 1 and d["c"] in KIND]
        for d in rng.sample(cand, min(DECISIONS, len(cand))):
            alts = [i for i in range(15) if d["m"] >> i & 1 and i != d["a"]]
            for alt in rng.sample(alts, min(ALTS, len(alts))):
                specs.append((gi, d, alt))
        if cand and gi % NULL_EVERY == 0:
            d = rng.choice(cand)
            specs.append((gi, d, d["a"]))
    specs.sort(key=lambda s: s[1]["r"])
    vals: dict[int, float] = {}
    missing = 0
    out_root = OUT / "games" / f"branch-{st['it']}"
    for b0 in range(0, len(specs), POOL):
        batch = specs[b0 : b0 + POOL]
        games_ = []
        for bot, (gi, d, alt) in zip(pool, batch):
            g = games[gi]
            # a null branch (forced = own choice) must not hold the goal: then it replays the base
            patch(bot, (d["r"], KIND[d["c"]], alt, COMMIT if alt != d["a"] else 0))
            games_.append((bot, g["opp"], g["map"]) if g["a"] else (g["opp"], bot, g["map"]))
        rounds = max(d["r"] for _, d, _ in batch) + H + 1
        out = out_root / f"b{b0 // POOL}"
        res = runner.run_games(games_, out, rounds=rounds, log_limit=R.LOG_LIMIT, fuel=R.FUEL[512],
                               timeout=R.GAME_TIMEOUT)
        for k, ((gi, d, alt), r) in enumerate(zip(batch, res)):
            g = games[gi]
            try:
                # only a branch whose forced goal fired counts (its "branch" event is in the log)
                lines = data._lines(Path(r.replay), g["side"])
                if not any(ln.startswith("E ") and '"k":"branch"' in ln for ln in lines):
                    missing += 1
                    continue
                vals[b0 + k] = value(Path(r.replay), g["side"], d["r"] + H)
            except (OSError, json.JSONDecodeError, KeyError, zipfile.BadZipFile):
                pass
        shutil.rmtree(out, ignore_errors=True)
    if missing:
        log(f"  {missing} of {len(specs)} branches never fired (ignored)")
    recs: dict[tuple[int, int, str], dict] = {}
    base_sc: dict[int, list] = {}
    for k, (gi, d, alt) in enumerate(specs):
        if k not in vals:
            continue
        g = games[gi]
        key = (gi, d["r"], d["c"])
        if gi not in base_sc:
            base_sc[gi] = scores(Path(g["res"].replay), g["side"])
        if key not in recs:
            recs[key] = {"gi": gi, "r": d["r"], "c": d["c"], "a": d["a"],
                         "base": value_at(base_sc[gi], d["r"] + H), "alts": {}}
        recs[key]["alts"][alt] = vals[k]
    out = list(recs.values())
    verify(st, games, pool, out, base_sc)
    return out


def verify(st: dict, games: list[dict], pool: list[str], recs: list[dict], base_sc: dict) -> None:
    """Best-of-several noisy branches looks good by luck: re-run each apparent winner holding the
    goal VERIFY_COMMIT rounds instead of COMMIT; its gain is the smaller of the two."""
    todo = []
    for rc in recs:
        dq = {a: v - rc["base"] for a, v in rc["alts"].items() if a != rc["a"]}
        if dq and max(dq.values()) > MARGIN:
            todo.append((rc, max(dq, key=dq.get)))
    todo.sort(key=lambda t: t[0]["r"])
    kept = 0
    for b0 in range(0, len(todo), POOL):
        batch = todo[b0 : b0 + POOL]
        games_ = []
        for bot, (rc, alt) in zip(pool, batch):
            g = games[rc["gi"]]
            patch(bot, (rc["r"], KIND[rc["c"]], alt, VERIFY_COMMIT))
            games_.append((bot, g["opp"], g["map"]) if g["a"] else (g["opp"], bot, g["map"]))
        out = OUT / "games" / f"verify-{st['it']}-{b0 // POOL}"
        res = runner.run_games(games_, out, rounds=max(rc["r"] for rc, _ in batch) + H + 1, log_limit=R.LOG_LIMIT,
                               fuel=R.FUEL[512], timeout=R.GAME_TIMEOUT)
        for (rc, alt), r in zip(batch, res):
            g = games[rc["gi"]]
            try:
                v2 = value(Path(r.replay), g["side"], rc["r"] + H)
            except (OSError, json.JSONDecodeError, KeyError, zipfile.BadZipFile):
                v2 = rc["base"]
            rc["alts"][alt] = min(rc["alts"][alt], v2)
            kept += v2 - rc["base"] > MARGIN
        shutil.rmtree(out, ignore_errors=True)
    log(f"  verify: {kept} of {len(todo)} winners hold with a {VERIFY_COMMIT}-round hold")


# ---------------------------------------------------------------- labels and training


def relabel(games: list[dict], recs: list[dict], it: int) -> tuple[Path, dict]:
    """Samples of the base games (our side) with improved decisions relabelled + oversampled."""
    by_game: dict[int, dict[tuple[int, str], int]] = {}
    null, gains, n_imp = [], [], 0
    for rc in recs:
        dq = {a: v - rc["base"] for a, v in rc["alts"].items()}
        if rc["a"] in dq:
            null.append(dq.pop(rc["a"]))
        if not dq:
            continue
        best = max(dq, key=dq.get)
        gains.append(dq[best])
        if dq[best] > MARGIN:
            by_game.setdefault(rc["gi"], {})[(rc["r"], rc["c"])] = best
            n_imp += 1
    parts = []
    qrecs = []  # searched decisions: row in this iteration's data, own choice, gain per alternative
    rec_at = {(rc["gi"], rc["r"], rc["c"]): rc for rc in recs}
    for gi, g in enumerate(games):
        r = g["res"]
        won = r.score_for(r.bot_a if g["a"] else r.bot_b)
        margin = ((r.gems_a - r.gems_b) if g["a"] else (r.gems_b - r.gems_a)) / 1000.0
        try:
            d = data.game_samples(Path(r.replay), Path(r.stats), g["side"], won, margin)
        except (OSError, KeyError, zipfile.BadZipFile, ValueError):
            continue
        if not d:
            continue
        # Row order of game_samples: champions sorted by name, then rounds.
        keys = sorted(((dd["c"], dd["r"]) for dd in g["dec"]), key=lambda x: (x[0], x[1]))
        if len(keys) != len(d["S"]):
            continue
        imp = by_game.get(gi, {})
        rows = [i for i, (c, rr) in enumerate(keys) if (rr, c) in imp]
        for i in rows:
            c, rr = keys[i]
            d["A"][i, 0] = imp[(rr, c)]
            d["M"][i, 0] |= 1 << imp[(rr, c)]
        off = sum(len(p["S"]) for p in parts)
        for i, (c, rr) in enumerate(keys):
            rc = rec_at.get((gi, rr, c))
            if rc:
                qrecs.append({"row": off + i, "own": rc["a"],
                              "gains": {str(a): v - rc["base"] for a, v in rc["alts"].items()}})
        parts.append(d)
        if rows:
            extra = {k: v[rows] for k, v in d.items()}
            extra["PREV"] = np.full_like(extra["PREV"], -1)
            extra["TRAJ"] = np.zeros_like(extra["TRAJ"])
            for _ in range(OVERSAMPLE - 1):
                parts.append(dict(extra))
    path = OUT / "data" / f"it{it:03d}.npz"
    data.save(path, data.concat(parts))
    path.with_suffix(".q.json").write_text(json.dumps(qrecs))
    stats = {"searched": len(recs), "improved": n_imp,
             "improved_frac": round(n_imp / max(len(recs), 1), 3),
             "mean_best_gain": round(float(np.mean(gains)), 2) if gains else None,
             "median_best_gain": round(float(np.median(gains)), 2) if gains else None,
             "null_abs_max": round(float(np.max(np.abs(null))), 3) if null else None}
    return path, stats


def evaluate(weights: Path, tag: str) -> dict:
    train, held = R.maps()
    bot = R.variant("_xeval", 2, 0, weights)
    out = OUT / "games" / f"eval-{tag}"
    res = runner.run_pairs([(bot, "main"), ("main", bot)], train + held, out, log_limit=R.LOG_LIMIT,
                           fuel=R.FUEL[512], timeout=R.GAME_TIMEOUT)
    shutil.rmtree(out, ignore_errors=True)
    tr = [r for r in res if r.map not in R.HELD_OUT]
    ho = [r for r in res if r.map in R.HELD_OUT]

    def sc(rs: list) -> float:
        return round(sum(r.score_for(bot) for r in rs) / max(len(rs), 1), 3)

    margin = [((r.gems_a - r.gems_b) if r.bot_a == bot else (r.gems_b - r.gems_a)) for r in res]
    return {"win": sc(res), "train": sc(tr), "held_out": sc(ho), "games": len(res),
            "gem_margin": round(sum(margin) / max(len(margin), 1), 1)}


def cmd_run(_: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    st = load_state()
    base_ck = R.ensure_rank(R.OUT / "ckpt" / "bc512")
    while True:
        it = st["it"]
        rng = random.Random(4242 + it)
        t0 = time.time()
        base, pool = policy_bots(st["weights"])
        games = base_games(st, base, rng)
        t1 = time.time()
        recs = search(st, games, pool, rng)
        t2 = time.time()
        path, sstats = relabel(games, recs, it)
        shutil.rmtree(OUT / "games" / f"base-{it}", ignore_errors=True)
        shutil.rmtree(OUT / "games" / f"branch-{it}", ignore_errors=True)
        st["search"].append({"it": it, **sstats})
        log(f"it {it}: {len(games)} base games ({t1 - t0:.0f}s), search {sstats} ({t2 - t1:.0f}s)")
        # Distill: the imitation data (keeps the rules' skills) + every iteration's search data.
        shards = sorted(str(p) for p in (R.OUT / "data" / "imitation").glob("*.npz"))
        shards += sorted(str(p) for p in (OUT / "data").glob("it*.npz"))
        init = Path(st["best"]) if st["best"] else base_ck
        ck = OUT / "ckpt" / f"it{it:03d}"
        m = R.gpu(["bc", "--data", *shards, "--init", str(init), "--out", str(ck), "--width", "512",
                   "--rank", str(R.RANK), "--lr", "3e-4", "--epochs", "6"])
        e = evaluate(ck / "teacher.bin", f"it{it}")
        st["evals"].append({"it": it, **e, "bc_goal_acc": m["held_out"][-1]["goal"],
                            "adopted": m.get("off_rule_labels")})
        best = max((x for x in st["evals"] if x.get("accepted")), key=lambda x: x["win"], default=None)
        bar = best["win"] if best else 0.5
        if e["win"] > bar:
            st["evals"][-1]["accepted"] = True
            st["best"], st["weights"] = str(ck), str(ck / "teacher.bin")
        log(f"eval it {it}: {e} adopted {m.get('off_rule_labels')} (bar {bar}){' ACCEPTED' if st['evals'][-1].get('accepted') else ''} "
            f"| bc {time.time() - t2:.0f}s")
        st["it"] = it + 1
        save_state(st)


def cmd_status(_: argparse.Namespace) -> int:
    st = load_state()
    print(json.dumps({"it": st["it"], "best": st["best"], "search": st["search"][-3:], "evals": st["evals"][-5:]},
                     indent=1))
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
