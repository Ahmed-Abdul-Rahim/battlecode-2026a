"""B (plan Phase B): labels from branch games, then the B net.

At sampled contact decisions (an enemy champion within 6 tiles) of games played with A's
searched plans active, the game is replayed with the choice forced to fight or retreat for
HOLD rounds (bots/main/src/branch.rs) and played to the end; the fitness difference to the
base game is the label. Null branches (choice 0) must reproduce the base game exactly.
Port of archive nn/expert.py's branch games.

Each game gets its own patched copy of the compiled wasm (cambc runs a .wasm directly, with
no build step), so batches are large.

    python -m harness.search branch setup        # build the B wasm (one slot)
    python -m harness.search branch labels [--hours H]
    python -m harness.search branch train        # MLP on the labels, int8, CV report
    python -m harness.search branch eval [--hours H]   # rules+A+B vs rules+A, paired
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from harness.search import common as C
from harness.search import search as S

MAGIC = bytes([0x42, 0x52, 0x43, 0x48, 0x02, 0x5A, 0xA5, 0x3C])
NET_MAGIC = bytes([0x42, 0x4E, 0x45, 0x54, 0x01, 0x3C, 0xA5, 0x5A])
BLOCK = 32
IN, HIDDEN, OUT = 19, 16, 2
NET_LEN = 16 + IN * HIDDEN + HIDDEN * 4 + HIDDEN * OUT + OUT * 4 + 8
OFF = 0xFFFFFFFF
HOLD = 10
VERIFY_HOLD = 6
#: decision kind -> (log prefix, branch-block id, the two forced choices)
DECISIONS = {"contact": ("D ", 0, (1, 2)), "tribute": ("T ", 1, (3, 4))}
DECISIONS_PER_GAME = 8
MIN_GAP = 25
NULL_EVERY = 10
N_OPPS = 32
BATCH = 600
LOG_LIMIT = 4_000_000
DIR = C.RUNS / "b"
SEARCH_RUNS = C.RUNS


def block(frm: int = OFF, kind: int = 0, choice: int = 0, hold: int = 0, log: int = 1, decision: int = 0) -> bytes:
    words = b"".join(int(x).to_bytes(4, "little") for x in (frm, kind, choice, hold, log, decision))
    return (MAGIC + words).ljust(BLOCK, b"\0")


def _put(b: bytearray, magic: bytes, data: bytes) -> None:
    i = b.find(magic)
    if i < 0 or b.find(magic, i + 1) >= 0:
        raise RuntimeError("block not found exactly once")
    b[i : i + len(data)] = data


def base_wasm() -> bytes:
    return C.injected(C.slot_dir(0)).read_bytes()


def setup() -> None:
    C.SLOT, C.RUNS = "_pb{}", DIR
    snap = C.build_slots(1)
    print(f"B wasm {snap['wasm'][:12]}")


def _load_plans() -> dict[str, bytes]:
    C.RUNS = SEARCH_RUNS
    snap_a = C.load_json(SEARCH_RUNS / "snapshot.json", {})["wasm"]
    table = {m: r for m, r in S.table().items() if r["accepted"]}
    plans = {m: bytes.fromhex(C.Store(f"search/{m}", snap_a).cands[r["accepted"]]) for m, r in table.items()}
    C.SLOT, C.RUNS = "_pb{}", DIR
    return plans


# ---------------------------------------------------------------- games


def _parse(args) -> dict:
    r, side, keep_d, prefix = args
    us = r["bot_a"] if side == "a" else r["bot_b"]
    res = C.runner.MatchResult(**r)
    row = C._row(res, C.Game("x", "", "", side), "b", us)
    text = ""
    try:
        with zipfile.ZipFile(r["replay"]) as z:
            text = z.read(f"team-{side}.stdout").decode("utf-8", "replace").replace("\x00", "")
    except (OSError, KeyError, zipfile.BadZipFile):
        pass
    lines = text.splitlines()
    if keep_d:
        row["D"] = [list(map(int, ln.split()[1:])) for ln in lines if ln.startswith(prefix)]
    row["fired"] = sum(1 for ln in lines if ln.startswith("B "))
    return row


def _key(j: dict) -> str:
    return f"{j['map']}|{j['opp']}|{j['side']}|{j['tag']}"


class Log:
    def __init__(self, name: str) -> None:
        DIR.mkdir(parents=True, exist_ok=True)
        self.path = DIR / f"{name}.jsonl"
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for ln in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                self.rows[_key(r)] = r["row"]

    def get(self, j: dict) -> dict | None:
        return self.rows.get(_key(j))

    def play(self, jobs: list[dict], plans: dict[str, bytes], deadline) -> bool:
        """jobs: {map, opp, side, tag, blk (bytes), net (bytes|None), keep_d}. One `cambc run`
        per game on its own patched wasm copy (run-batch only takes project directories)."""
        from concurrent.futures import ThreadPoolExecutor

        todo = [j for j in jobs if _key(j) not in self.rows]
        wasm = base_wasm()
        opps, maps = C.opponents(), C.maps()
        for o in {j["opp"] for j in todo}:
            C.runner.build(opps[o], release=True)
        jobs_n = max(1, (os.cpu_count() or 2) - 2)
        for i in range(0, len(todo), BATCH):
            if deadline and time.time() > deadline:
                return False
            chunk = todo[i : i + BATCH]
            tmp = DIR / "tmp" / f"b{int(time.time() * 1000)}"
            tmp.mkdir(parents=True, exist_ok=True)

            def one(n: int) -> dict:
                j = chunk[n]
                d = tmp / f"j{n}"
                d.mkdir()
                b = bytearray(wasm)
                _put(b, C.MAGIC, C.block(plans.get(j["map"], C.record())))
                _put(b, MAGIC, j["blk"])
                if j.get("net"):
                    _put(b, NET_MAGIC, j["net"])
                w = d / "bot.wasm"
                w.write_bytes(b)
                a, bb = (str(w), opps[j["opp"]]) if j["side"] == "a" else (opps[j["opp"]], str(w))
                stats = d / "stats.json"
                import subprocess

                subprocess.run([str(C.runner.CAMBC), "run", a, bb, "--map", str(maps[j["map"]]), "--release",
                                "-o", str(d), "--log-limit", str(LOG_LIMIT), "--statistics", str(stats), "--json"],
                               cwd=C.ROOT, env=C.runner._env(), capture_output=True, timeout=900)
                rep = next(d.glob("*.repl26a"), d / "missing.repl26a")
                res = C.runner._parse(stats, j["map"], a, bb, rep)
                return res.to_dict()

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=jobs_n) as ex:
                res = list(ex.map(one, range(len(chunk))))
            with ProcessPoolExecutor(max_workers=jobs_n) as ex:
                rows = list(ex.map(_parse, [(r, j["side"], j.get("keep_d", False), j.get("prefix", "D ")) for r, j in zip(res, chunk)], chunksize=8))
            with open(self.path, "a", encoding="utf-8") as f:
                for j, row in zip(chunk, rows):
                    rec = {k: j[k] for k in ("map", "opp", "side", "tag")}
                    f.write(json.dumps({**rec, "row": row}, separators=(",", ":")) + chr(10))
                    self.rows[_key(rec)] = row
                f.flush()
                os.fsync(f.fileno())
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"  {len(chunk)} games in {time.time() - t0:.0f}s ({len(self.rows)} logged)", flush=True)
        return True


def _setups(splits: dict, plans: dict) -> list[tuple[str, str, str]]:
    rng = random.Random(2026)
    opps = rng.sample(splits["opps_A"], N_OPPS)
    return [(m, o, s) for m in plans for o in opps for s in "ab"]


def labels(hours: float | None = None, decision: str = "contact") -> None:
    prefix, did, choices = DECISIONS[decision]
    C.SLOT, C.RUNS = "_pb{}", DIR
    C.check_snapshot()
    plans = _load_plans()
    splits = json.loads((SEARCH_RUNS / "splits.json").read_text(encoding="utf-8"))
    deadline = C.deadline_from(hours)
    log = Log("labels" if decision == "contact" else f"labels_{decision}")
    # The base games take the same code path as a branch (a kind that never plays), so a null
    # branch replays them exactly.
    base = [{"map": m, "opp": o, "side": s, "tag": "base", "blk": block(0, 999, 0, 0, 1, did),
             "keep_d": True, "prefix": prefix} for m, o, s in _setups(splits, plans)]
    if not log.play(base, plans, deadline):
        return print("stopped: run labels again to continue")
    decisions, branches = sample(base, log, choices), []
    for d in decisions:
        for c in d["alts"]:
            for hold in (HOLD, VERIFY_HOLD) if c else (HOLD,):
                branches.append({**{k: d[k] for k in ("map", "opp", "side")},
                                 "tag": f"{d['round']}:{d['kind']}:{c}:{hold}",
                                 "blk": block(d["round"], d["kind"], c, hold, 1, did)})
    print(f"{len(decisions)} {decision} decisions, {len(branches)} branch games", flush=True)
    if not log.play(branches, plans, deadline):
        return print("stopped: run labels again to continue")
    export(decisions, log, decision, choices)


def sample(base: list[dict], log: Log, choices: tuple[int, int]) -> list[dict]:
    out, n = [], 0
    for j in base:
        row = log.get(j)
        if not row:
            continue
        ds = list(row.get("D", []))
        random.Random(_key(j)).shuffle(ds)
        picked: list[list[int]] = []
        for d in ds:
            if d[0] < 20 or d[0] > row["rounds"] - 50:
                continue
            if all(abs(d[0] - p[0]) >= MIN_GAP for p in picked):
                picked.append(d)
            if len(picked) >= DECISIONS_PER_GAME:
                break
        for d in picked:
            rnd, kind, cls = d[0], d[1], d[2]
            alts = [c for c in choices if c != cls]
            if n % NULL_EVERY == 0:
                alts.append(0)
            n += 1
            out.append({"map": j["map"], "opp": j["opp"], "side": j["side"], "round": rnd, "kind": kind,
                        "cls": cls, "f": d[4:], "alts": alts})
    return out


def export(decisions: list[dict], log: Log, decision: str, choices: tuple[int, int]) -> dict:
    """labels_<decision>.json: features + the verified gain of each forced choice over the
    rules': both hold lengths must agree in sign (the smaller one counts), else 0."""
    rows, nulls_bad, fired = [], 0, 0
    for d in decisions:
        b = log.get({**d, "tag": "base"})
        if not b:
            continue
        g = {}
        for c in d["alts"]:
            r = log.get({**d, "tag": f"{d['round']}:{d['kind']}:{c}:{HOLD}"})
            if not r:
                continue
            if c == 0:
                nulls_bad += C.fitness(r) != C.fitness(b)
                continue
            v = log.get({**d, "tag": f"{d['round']}:{d['kind']}:{c}:{VERIFY_HOLD}"})
            if not r["fired"] or not v:
                continue
            g1, g2 = C.fitness(r) - C.fitness(b), C.fitness(v) - C.fitness(b)
            if g1 > 0 and g2 > 0:
                g[c] = round(min(g1, g2), 4)
            elif g1 < 0 and g2 < 0:
                g[c] = round(max(g1, g2), 4)
            else:
                g[c] = 0.0
        if not g:
            continue
        fired += 1
        rows.append({"map": d["map"], "game": f"{d['map']}|{d['opp']}|{d['side']}", "cls": d["cls"],
                     "f": [v / 100 for v in d["f"]], "a": g.get(choices[0], 0.0), "b": g.get(choices[1], 0.0)})
    C.save_json(DIR / f"labels_{decision}.json", rows)
    gains = [x for r in rows for x, c in ((r["a"], choices[0]), (r["b"], choices[1])) if c != r["cls"]]
    pos, neg = sum(x > 0.05 for x in gains), sum(x < -0.05 for x in gains)
    print(f"labels ({decision}): {len(rows)} decisions ({fired} fired), null mismatches {nulls_bad}, "
          f"verified better {pos} / worse {neg} of {len(gains)}")
    return {"n": len(rows), "null_bad": nulls_bad}


# ---------------------------------------------------------------- the net


def train(decision: str = "contact") -> dict:
    import numpy as np

    _, did, choices = DECISIONS[decision]
    rows = C.load_json(DIR / f"labels_{decision}.json", [])
    X = np.array([r["f"] for r in rows], dtype=np.float32)
    Y = np.array([[r["a"], r["b"]] for r in rows], dtype=np.float32)
    # The rules' own class has gain 0 by definition (masked out of the loss).
    M = np.array([[r["cls"] != choices[0], r["cls"] != choices[1]] for r in rows], dtype=np.float32)
    games = sorted({r["game"] for r in rows})
    gid = np.array([games.index(r["game"]) for r in rows])
    rng = np.random.default_rng(0)
    fold = rng.permutation(len(games)) % 5

    def fit(Xt, Yt, Mt, epochs=400, wd=1e-3, lr=0.01, seed=0):
        r = np.random.default_rng(seed)
        W1 = r.normal(0, 0.3, (IN, HIDDEN)).astype(np.float32)
        b1 = np.zeros(HIDDEN, np.float32)
        W2 = r.normal(0, 0.3, (HIDDEN, OUT)).astype(np.float32)
        b2 = np.zeros(OUT, np.float32)
        params = [W1, b1, W2, b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for t in range(1, epochs + 1):
            H = np.maximum(Xt @ W1 + b1, 0)
            P = H @ W2 + b2
            E = (P - Yt) * Mt / max(Mt.sum(), 1)
            gW2 = H.T @ E + wd * W2
            gb2 = E.sum(0)
            dH = (E @ W2.T) * (H > 0)
            gW1 = Xt.T @ dH + wd * W1
            gb1 = dH.sum(0)
            for k, g in enumerate((gW1, gb1, gW2, gb2)):
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g * g
                params[k] -= lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v[k] / (1 - 0.999**t)) + 1e-8)
        return params

    def predict(p, Xs):
        W1, b1, W2, b2 = p
        return np.maximum(Xs @ W1 + b1, 0) @ W2 + b2

    def captured(P, Y, M, tau):
        """Mean realised gain per decision when the net overrules above tau (0 = keep rules)."""
        Pm = np.where(M > 0, P, -9)
        k = Pm.argmax(1)
        take = Pm.max(1) > tau
        return float(np.where(take, Y[np.arange(len(Y)), k], 0).mean()), float(take.mean())

    oracle = float(np.where(M > 0, np.maximum(Y, 0), 0).max(1).mean())
    taus = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
    cv = {t: [0.0, 0.0] for t in taus}
    for f in range(5):
        tr, te = np.isin(gid, np.where(fold != f)[0]), np.isin(gid, np.where(fold == f)[0])
        p = fit(X[tr], Y[tr], M[tr])
        P = predict(p, X[te])
        for t in taus:
            g, share = captured(P, Y[te], M[te], t)
            cv[t][0] += g * te.sum() / len(X)
            cv[t][1] += share * te.sum() / len(X)
    best_tau = max(taus, key=lambda t: cv[t][0])
    p = fit(X, Y, M)
    net = quantise(p, best_tau, did)
    (DIR / f"net_{decision}.bin").write_bytes(net)
    rep = {
        "n": len(rows),
        "oracle_gain_per_decision": round(oracle, 4),
        "cv_gain_per_decision": {str(t): round(cv[t][0], 4) for t in taus},
        "cv_overrule_share": {str(t): round(cv[t][1], 3) for t in taus},
        "tau": best_tau,
    }
    C.save_json(DIR / f"train_{decision}.json", rep)
    print(json.dumps(rep, indent=1))
    return rep


def quantise(p, tau: float, did: int = 0) -> bytes:
    import numpy as np

    W1, b1, W2, b2 = p
    s_w1 = max(np.abs(W1).max(), 1e-6) / 127
    s1 = s_w1 / 64  # inputs are int8 at 1/64
    q1 = np.clip(np.round(W1.T / s_w1), -127, 127).astype(np.int8)  # [HIDDEN][IN]
    qb1 = np.round(b1 / s1).astype(np.int32)
    s2 = max(np.abs(W2).max(), 1e-6) / 127
    q2 = np.clip(np.round(W2.T / s2), -127, 127).astype(np.int8)  # [OUT][HIDDEN]
    qb2 = np.round(b2 / s2).astype(np.int32)
    head = NET_MAGIC + bytes([1, int(np.clip(round(tau * 100), -127, 127)) & 0xFF, did]) + b"\0" * 5
    body = q1.tobytes() + qb1.astype("<i4").tobytes() + q2.tobytes() + qb2.astype("<i4").tobytes()
    body += np.array([s1, s2], dtype="<f4").tobytes()
    out = head + body
    assert len(out) == NET_LEN, (len(out), NET_LEN)
    return out


def evaluate(hours: float | None = None, decision: str = "contact") -> dict:
    """rules+A+B vs rules+A on the searched maps, B opponents (never used to choose), both sides."""
    C.SLOT, C.RUNS = "_pb{}", DIR
    C.check_snapshot()
    plans = _load_plans()
    splits = json.loads((SEARCH_RUNS / "splits.json").read_text(encoding="utf-8"))
    net = (DIR / f"net_{decision}.bin").read_bytes()
    log = Log(f"eval_{decision}")
    opps = splits["opps_B"] + [o for o in splits["head_to_head"]] + sorted(C.fresh_top())
    jobs = []
    for m in plans:
        for o in opps:
            for s in "ab":
                jobs.append({"map": m, "opp": o, "side": s, "tag": "A", "blk": block()})
                jobs.append({"map": m, "opp": o, "side": s, "tag": "AB", "blk": block(), "net": net})
    if not log.play(jobs, plans, C.deadline_from(hours)):
        print("stopped: run eval again to continue")
        return {}
    out = {}
    for name, group in (("scoring", splits["opps_B"]), ("our rules", splits["head_to_head"]),
                        ("fresh top 10", sorted(C.fresh_top()))):
        df, dw = [], []
        for m in plans:
            for o in group:
                for s in "ab":
                    a, ab = log.get({"map": m, "opp": o, "side": s, "tag": "A"}), log.get({"map": m, "opp": o, "side": s, "tag": "AB"})
                    if a and ab:
                        df.append(C.fitness(ab) - C.fitness(a))
                        dw.append(ab["res"] - a["res"])
        out[name] = {"n": len(df), **C._mean_se(df, "fit"), **C._mean_se(dw, "win")}
    fired = sum(r.get("fired", 0) for k, r in log.rows.items() if k.endswith("|AB"))
    out["net_overrules_logged"] = fired
    C.save_json(DIR / f"eval_{decision}.json", out)
    print(json.dumps(out, indent=1))
    return out
