"""The league: train bots/main against mimics of the live top 10 and share the weights through git.

One loop, run on every teammate's PC (architecture and commands: docs/league.pdf):

  pull     rebase on origin/league; adopt the shared champion (league/champion.json -> pack.bin)
  sync     lead only, every SYNC_EVERY: ladder top 10 (+ teams we just lost to) -> download
           their games -> mimic.extract -> data/mimic/<slug>.spec.json
  watch    lead only: our new ladder games -> ledger; every loss joins the queue
  maps     lead only, with sync: follow the platform's map pool (pull + register new maps)
  train    per-map antithetic ES on θ (harness/params.py) against the mimic pool (PFSP) + anchors
           (map order: never-trained maps first, then ladder losses, then our worst map)
  evolve   every MIMIC_EVERY-th cycle: ES on an *evolved mimic* (league/mimics/<slug>.json) to
           beat our champion, so training goes on when the real teams stop updating
  promote  accepted θ -> champion.json + pack.bin, benchmark vs every mimic, commit, push
  release  lead with --auto-submit only: `cambc submit` after a ladder loss or a benchmark jump,
           at most once per SUBMIT_GAP, reverted if the ladder says the new version is worse
  review   lead only: league/review/<date>.md for humans (epistemic gaps, losses ES can't fix)

Shared state lives in git on branch `league`:
  league/champion.json   written by any PC: per-map θ, promotion history, benchmarks, try counts;
                         a push race is resolved by a 3-way merge (merge_champion)
  league/ledger.json     written by the lead only: top-10 snapshots, ladder games, loss queue,
                         submissions
  data/mimic/            mimic specs (lead); every PC builds bots/mimic-<slug> from them
  league/review/         review reports (lead); humans edit the "Reviewer notes" section
Per-PC scratch (sigma, PFSP win rates, try counts, logs, lock): runs/league/.

    python harness/league.py run --forever                         # worker
    python harness/league.py run --forever --lead                  # the PC that reads the platform
    python harness/league.py run --forever --lead --auto-submit    # ... and uploads (one PC only)
    python harness/league.py run --cycles 1 --no-push              # smoke test, no git
    python harness/league.py status | sync | maps | bench | review | submit
    python harness/league.py replay [nyx] [lanes] [--evolved]     # watch the champion play
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mimic
import pack as packlib
import params

from bctools import CAMBC, ROOT, replay, runner

OUR_TEAM = "1ec259ba-7596-4a78-9fd6-090c77f543bf"  # Patient Wolves
BRANCH = "league"
LEAGUE = ROOT / "league"
CHAMPION = LEAGUE / "champion.json"
LEDGER = LEAGUE / "ledger.json"
REVIEW = LEAGUE / "review"
MIMICS = ROOT / "data" / "mimic"
EVOLVED = LEAGUE / "mimics"  # evolved mimics: per-team θ trained to beat our champion
FINGERPRINTS = ROOT / "data" / "fingerprints.json"
GEOMETRY = ROOT / "data" / "map_geometry.json"
PACK = ROOT / "bots" / "main" / "src" / "pack" / "pack.bin"
SCRATCH = ROOT / "runs" / "league"
LOCAL = SCRATCH / "local.json"
LOCK = SCRATCH / "lock"
GAMES = ROOT / "runs" / "ladder_top"  # downloaded summaries + match lists (mimic.SUMMARY_DIRS)
OURS = ROOT / "runs" / "ladder_live"  # summaries of our own ladder games
HOST = os.environ.get("LEAGUE_HOST") or socket.gethostname()
NOTES = "## Reviewer notes"
#: What the lead commits after a sync: platform data, mimic specs, evolved-mimic resets, maps.
LEAD_FILES = [LEDGER, REVIEW, MIMICS, EVOLVED, ROOT / "maps", FINGERPRINTS, GEOMETRY]

# Training
LAMBDA = 8  # ES candidates per cycle (antithetic pairs)
N_OPPS = 4  # mimics per cycle, PFSP-weighted towards the ones that beat us
N_ANCHORS = 1  # plus fixed exploiter bots, so old fixes are not forgotten
N_CONFIRM = 3  # fresh opponents an accepted candidate must also beat the champion on
ACCEPT_GAIN = 0.04
MARGIN_W, DEATH_W, SUSP_W = 0.25, 0.01, 0.002
ANCHORS = [
    "claude-pillager",
    "claude-ambusher",
    "claude-focus",
    "exploit-qwen-rush",
    "exploit-qwen-pillager",
    "exploit-qwen-spawn_camper",
    "baseline",
]
# Platform / release
TOP_N = 10
LOSS_MIMICS = 3  # also mimic the most recent teams we lost to that are outside the top 10
GAMES_PER_TEAM = 10  # newest finished games downloaded per team per sync
SYNC_EVERY = 3 * 3600
WATCH_EVERY = 600
STUCK_TRIES = 6  # ES cycles on a lost map without a fix before a human is asked
BOOTSTRAP_TRIES = 4  # rounds every never-trained (new) map gets before losses and top-10 work
SUBMIT_GAP = 3 * 3600
BENCH_JUMP = 0.05
# Revert only on REVERT_GAMES+ ranked games of the new version, against the previous version's
# latest REVERT_RECENT games, and a drop bigger than both REVERT_DROP and two standard errors.
REVERT_GAMES, REVERT_DROP, REVERT_RECENT = 30, 0.10, 60
HEARTBEAT = 1800  # push try counts at least this often, so the lead sees stuck maps
#: Map pressure (map_pressure): a loss's weight falls to 1/e after a day; every map keeps a
#: floor of this fraction of the top pressure, so no map is starved.
PRESSURE_TAU = 24 * 3600
PRESSURE_FLOOR = 0.1
# Evolved mimics: every MIMIC_EVERY-th cycle trains an opponent instead of the champion.
MIMIC_EVERY = int(os.environ.get("LEAGUE_MIMIC_EVERY", "3"))
#: θ entries an evolved mimic may not change: the team's observed cadence (W recall_gold,
#: tribute_min, tribute_max), so it keeps the real team's macro while its micro evolves.
LOCKED = frozenset(params.W_NAMES.index(n) for n in ("recall_gold", "tribute_min", "tribute_max"))


def log(msg: str) -> None:
    try:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    except Exception:
        safe = msg.encode("ascii", "backslashreplace").decode("ascii")
        print(f"[{time.strftime('%H:%M:%S')}] {safe}", flush=True)


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ts(iso: str | None) -> float:
    """ISO time (ours or the platform's '...816Z') -> epoch seconds; 0 if missing."""
    if not iso:
        return 0.0
    try:
        return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0  # malformed platform data: treat as very old


def load_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(p)


def sh(args: list[str], env: dict | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout,
    )


def git(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return sh(["git", *args], env=env)


map_name = mimic.map_name


def local_maps() -> dict[str, Path]:
    """Every map file in maps/, keyed by normalized name ('01big-world' -> 'big-world'); a
    platform-pulled (digit-prefixed) file wins over an older unprefixed copy."""
    files = sorted(runner.list_maps(), key=lambda p: p.stem[:1].isdigit())
    return {map_name(p.stem): p for p in files}


def active_maps(led: dict) -> dict[str, Path]:
    """The platform's current pool (ledger 'maps') that we have locally; all local maps until
    the lead's first map sync. Retired maps stay in maps/ but are not trained on."""
    local = local_maps()
    pool = led.get("maps") or []
    return {m: local[m] for m in pool if m in local} or local


# ----------------------------------------------------------------- shared state


def empty_champion() -> dict:
    return {"version": 0, "maps": {}, "history": [], "bench": {}, "bench_hard": {}, "tries": {}}


def empty_ledger() -> dict:
    return {"top": [], "pool": [], "maps": [], "map_files": {}, "ladder": {}, "losses": [], "submissions": []}


def load_champion() -> dict:
    return {**empty_champion(), **load_json(CHAMPION, {})}


def load_ledger() -> dict:
    return {**empty_ledger(), **load_json(LEDGER, {})}


def total_tries(ch: dict, mp: str) -> int:
    return sum((ch["tries"].get(mp) or {}).values())


def merge_champion(base: dict, ours: dict, theirs: dict, decide) -> dict:
    """3-way merge of two champion.json versions that both moved on from `base`.

    θ: per map, take whichever side changed it; if both did, decide(map, a, b) picks one
    (a head-to-head). History: union, renumbered by time, so versions stay unique. Bench:
    union. Tries: per map and host, the larger count (they only grow)."""
    base, ours, theirs = (({**empty_champion(), **(x or {})}) for x in (base, ours, theirs))
    maps = {}
    for m in sorted(set(ours["maps"]) | set(theirs["maps"])):
        a, b, o = ours["maps"].get(m), theirs["maps"].get(m), base["maps"].get(m)
        if a == b or b == o:
            v = a
        elif a == o:
            v = b
        else:
            v = decide(m, a, b)
        if v is not None:
            maps[m] = v
    seen, hist = set(), []
    for h in sorted(ours["history"] + theirs["history"], key=lambda h: (h["time"], h["host"], h["map"])):
        key = (h["time"], h["host"], h["map"])
        if key not in seen:
            seen.add(key)
            hist.append(dict(h))
    for i, h in enumerate(hist):
        h["version"] = i + 1
    tries = {}
    for side in (ours["tries"], theirs["tries"]):
        for m, per in side.items():
            for host, n in per.items():
                tries.setdefault(m, {})[host] = max(n, tries.get(m, {}).get(host, 0))
    known = ("version", "maps", "history", "bench", "bench_hard", "tries")
    extras = {k: v for k, v in {**ours, **theirs}.items() if k not in known}  # seed, base, slug
    return {
        **extras,
        "version": len(hist),
        "maps": maps,
        "history": hist,
        "bench": {**ours["bench"], **theirs["bench"]},
        "bench_hard": {**ours["bench_hard"], **theirs["bench_hard"]},
        "tries": tries,
    }


STATUS_RANK = {
    "queued": 0,
    "screening": 1,
    "running": 1,
    "full": 2,
    "passed": 3,
    "rejected": 3,
    "done": 3,
    "failed": 3,
    "merged": 4,
}


def merge_experiments(ours: dict | None, theirs: dict | None, key: str = "experiments") -> dict:
    """Experiment registry from two PCs: per id the later run, then the further-along entry wins;
    two claims on the same id keep the earlier one. Order (= queue priority) follows `theirs`, then new ids."""
    ours, theirs = ours or {key: []}, theirs or {key: []}
    mine = {e["id"]: e for e in ours.get(key, [])}

    def pick(a: dict, b: dict) -> dict:
        # A deliberate re-run (run + 1) beats the older run's result; then the further status.
        if a.get("run", 1) != b.get("run", 1):
            return a if a.get("run", 1) > b.get("run", 1) else b
        ra, rb = STATUS_RANK.get(a.get("status"), 0), STATUS_RANK.get(b.get("status"), 0)
        if ra != rb:
            return a if ra > rb else b
        return a if a.get("claimed", "~") <= b.get("claimed", "~") else b

    def same(a: dict, b: dict) -> bool:  # the same entry, not two PCs adding the same new id
        return (a.get("title"), a.get("kind"), a.get("added")) == (
            b.get("title"),
            b.get("kind"),
            b.get("added"),
        )

    out, extra = [], []
    for e in theirs.get(key, []):
        m = mine.pop(e["id"], None)
        if m is None:
            out.append(e)
        elif same(m, e):
            out.append(pick(m, e))
        else:
            out.append(e)
            extra.append(m)
    extra += list(mine.values())
    top = max((e["id"] for e in out), default=0)
    for m in extra:
        if any(e["id"] == m["id"] for e in out):
            top += 1
            m = {**m, "id": top}
        top = max(top, m["id"])
        out.append(m)
    return {**theirs, key: out}


def merge_ledger(ours: dict, theirs: dict) -> dict:
    """Union of two ledgers (only the lead writes it, so this is a safety net)."""
    ours, theirs = {**empty_ledger(), **(ours or {})}, {**empty_ledger(), **(theirs or {})}
    top = {s["time"]: s for s in ours["top"] + theirs["top"]}
    losses = {x["id"]: x for x in ours["losses"]}
    for x in theirs["losses"]:
        if losses.get(x["id"], {}).get("status", "open") == "open":
            losses[x["id"]] = x
    subs = {(s["version"], s["time"]): s for s in ours["submissions"] + theirs["submissions"]}
    newer = (
        theirs
        if (theirs["top"] or [{}])[-1].get("time", "") >= (ours["top"] or [{}])[-1].get("time", "")
        else ours
    )
    return {
        "top": [top[k] for k in sorted(top)][-500:],
        "pool": newer["pool"],
        "maps": newer["maps"],
        "map_files": {**ours["map_files"], **theirs["map_files"]},
        "ladder": {**ours["ladder"], **theirs["ladder"]},
        "losses": sorted(losses.values(), key=lambda x: x["time"]),
        "submissions": [subs[k] for k in sorted(subs, key=lambda k: k[1])],
    }


# ----------------------------------------------------------------- θ and bots


def base_theta() -> list[float]:
    return params.to_search(params.defaults(ROOT / "bots" / "main" / "src" / "learned.rs"))


def clip_search(theta: list[float]) -> list[float]:
    """Keep θ inside the registry bounds; fixed parameters (lo == hi) keep their default."""
    out, i, base = [], 0, base_theta()
    for _, ns, _, lo, hi, _ in params.BLOCKS:
        for _ in ns:
            out.append(base[i] if lo == hi else min(max(theta[i], lo), hi))
            i += 1
    return out


def full_theta(theta: list[float] | None) -> list[float]:
    base = base_theta()
    return list(theta or base) + base[len(theta or base) :]


def pack_bytes_for(maps: dict[str, list[float]], base_spec: dict | None = None) -> bytes:
    spec = {m: dict(e) for m, e in ((base_spec or {}).get("maps") or {}).items()}
    dflt = params.defaults(ROOT / "bots" / "main" / "src" / "learned.rs")
    for m, th in maps.items():
        spec.setdefault(m, {})["theta"] = params.to_bot(th, dflt)
    tmp = SCRATCH / f"pack-{os.getpid()}.bin"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    packlib.write({"maps": spec}, tmp)
    data = tmp.read_bytes()
    tmp.unlink()
    return data


def write_if_changed(p: Path, data: bytes) -> bool:
    if p.exists() and p.read_bytes() == data:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return True


def apply_champion(ch: dict) -> bool:
    """bots/main's pack.bin from champion.json (only rewritten when it changes, so no rebuild)."""
    return write_if_changed(PACK, pack_bytes_for(ch["maps"]))


def engine_hash() -> str:
    """Hash of bots/main's code without the pack: mimics rebuild only when the engine changes."""
    h = hashlib.sha1()
    src = ROOT / "bots" / "main"
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src).as_posix()
        if p.is_file() and not rel.startswith((".cambc", "target", "src/pack/")):
            h.update(rel.encode() + p.read_bytes())
    return h.hexdigest()


def make_bot(name: str, pack_data: bytes) -> str:
    """bots/<name> = bots/main's code + the given pack (target dir kept: incremental builds)."""
    d, src = ROOT / "bots" / name, ROOT / "bots" / "main"
    if d.exists():
        shutil.rmtree(d / "src", ignore_errors=True)
        shutil.copytree(src / "src", d / "src")
        for f in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml"):
            if (src / f).exists():
                shutil.copy2(src / f, d / f)
    else:
        shutil.copytree(src, d, ignore=shutil.ignore_patterns(".cambc", "target"))
    (d / "src" / "pack" / "pack.bin").write_bytes(pack_data)
    return name


def macro_hash(clone: dict) -> str:
    """What the real team does at the macro level (draft, builds, cadence). Unchanged between
    syncs = the team hasn't updated its bot, so evolving our mimic of it is the only way on."""
    keep = {k: clone.get(k) for k in ("draft", "build", "cadence")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def load_evolved(slug: str, base: str | None) -> dict:
    """league/mimics/<slug>.json (champion.json schema); reset when the team's macro changed."""
    mx = {**empty_champion(), **load_json(EVOLVED / f"{slug}.json", {})}
    if base and mx.get("base") != base:
        mx = {**empty_champion(), "base": base, "slug": slug}
    return mx


def spec_theta(spec: dict, mp: str) -> list[float]:
    """The faithful mimic's θ on a map, in search space (spec θ is bot space, maybe shorter)."""
    th = ((spec.get("maps") or {}).get(mp) or {}).get("theta")
    if not th:
        return base_theta()
    dflt = params.defaults(ROOT / "bots" / "main" / "src" / "learned.rs")
    return params.to_search(list(th) + dflt[len(th) :])


def build_stamped(name: str, key: bytes, data) -> None:
    """make_bot only when the inputs (key) changed since the last build (.league-stamp)."""
    stamp = hashlib.sha1(key + engine_hash().encode()).hexdigest()
    stamp_p = ROOT / "bots" / name / ".league-stamp"
    if not stamp_p.exists() or stamp_p.read_text() != stamp:
        make_bot(name, data())
        stamp_p.write_text(stamp)


def ensure_mimics(led: dict) -> tuple[list[str], list[str]]:
    """Build the pool's opponents: faithful bots/mimic-<slug> (the team's spec from its games,
    never trained) and evolved bots/mimx-<slug> (same macro, θ evolved against our champion)."""
    faithful, evolved = [], []
    for p in led["pool"]:
        spec_p = MIMICS / f"{p['slug']}.spec.json"
        if not spec_p.exists():
            continue
        raw = spec_p.read_bytes()
        spec = json.loads(raw)
        build_stamped(f"mimic-{p['slug']}", raw, lambda spec=spec: pack_bytes_for({}, spec))
        faithful.append(f"mimic-{p['slug']}")
        mx = load_evolved(p["slug"], p.get("macro_hash"))
        if mx["version"] >= 1:
            key = raw + json.dumps(mx["maps"], sort_keys=True).encode()
            build_stamped(f"mimx-{p['slug']}", key, lambda spec=spec, mx=mx: pack_bytes_for(mx["maps"], spec))
            evolved.append(f"mimx-{p['slug']}")
    return faithful, evolved


# ----------------------------------------------------------------- matches and fitness


def suspended(results: list[runner.MatchResult], bot: str) -> int:
    """Out-of-fuel (suspended) turns of `bot` across the matches."""
    bad = 0
    for r in results:
        if bot not in (r.bot_a, r.bot_b):
            continue
        try:
            bad += replay.exhausted_turns(replay.load(r.replay).team("a" if r.bot_a == bot else "b"))
        except (OSError, KeyError, ValueError):
            continue
    return bad


def fitness(results: list[runner.MatchResult], bot: str) -> tuple[float, float, int]:
    """(fitness, win rate, games) over the games `bot` played: win + 0.25 gem margin - deaths."""
    rs = [r for r in results if bot in (r.bot_a, r.bot_b)]
    n = max(len(rs), 1)
    win = sum(r.score_for(bot) for r in rs) / n
    margin = deaths = 0.0
    for r in rs:
        ours, theirs = (r.gems_a, r.gems_b) if r.bot_a == bot else (r.gems_b, r.gems_a)
        margin += (ours - theirs) / max(ours + theirs, 1)
        deaths += r.deaths_a if r.bot_a == bot else r.deaths_b
    return win + MARGIN_W * margin / n - DEATH_W * deaths / n, win, len(rs)


def score(results: list[runner.MatchResult], bot: str, opp: str) -> float:
    rs = [r for r in results if {r.bot_a, r.bot_b} == {bot, opp}]
    return sum(r.score_for(bot) for r in rs) / max(len(rs), 1)


def play(bots: list[str], opps: list[str], maps: list[Path], out: Path) -> list[runner.MatchResult]:
    """Every bot against every opponent, both sides, one batch."""
    pairs = [p for b in bots for o in opps if b != o for p in ((b, o), (o, b))]
    return runner.run_pairs(pairs, maps, out, release=True) if pairs else []


def pfsp(local: dict, cands: list[str], k: int, rng: random.Random) -> list[str]:
    """Prioritised fictitious self-play: opponents we lose to are picked more often."""
    cands, out = list(cands), []
    w = [(1.0 - local["pwin"].get(b, 0.5)) ** 2 + 0.05 for b in cands]
    while cands and len(out) < k:
        i = rng.choices(range(len(cands)), weights=w)[0]
        out.append(cands.pop(i))
        w.pop(i)
    return out


def benchmark(bot: str, mimics: list[str], maps: list[Path], out: Path) -> dict:
    """`bot` against every mimic on every active map, both sides; keyed by team slug."""
    res = play([bot], mimics, maps, out)
    grid: dict[str, dict[str, list[float]]] = {}
    for r in res:
        opp = (r.bot_b if r.bot_a == bot else r.bot_a).split("-", 1)[1]  # mimic-/mimx-<slug>
        grid.setdefault(opp, {}).setdefault(map_name(r.map), []).append(r.score_for(bot))
    per_team = {
        t: round(sum(sum(v) for v in g.values()) / max(sum(len(v) for v in g.values()), 1), 3)
        for t, g in grid.items()
    }
    maps: dict[str, list[float]] = {}
    for g in grid.values():
        for m, v in g.items():
            maps.setdefault(m, []).extend(v)
    return {
        "time": now_iso(),
        "games": len(res),
        "overall": round(sum(r.score_for(bot) for r in res) / max(len(res), 1), 3),
        "per_team": per_team,
        "per_map": {m: round(sum(v) / len(v), 3) for m, v in sorted(maps.items())},
        "grid": {t: {m: round(sum(v) / len(v), 2) for m, v in sorted(g.items())} for t, g in grid.items()},
    }


# ----------------------------------------------------------------- git


def git_ready() -> bool:
    """Only commit/push from the league branch with an origin remote."""
    # The experiment autopilot and the ml job runner run in their own worktrees (branches
    # 'autopilot' / 'mlpilot'; league is the daemon's); they rebase on and push to origin/league.
    if git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() not in (BRANCH, "autopilot", "mlpilot"):
        log(f"not on branch '{BRANCH}': git sync disabled")
        return False
    return git("remote", "get-url", "origin").returncode == 0


def show(rev: str, path: str) -> dict | None:
    p = git("show", f"{rev}:{path}")
    return json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None


def splice_notes(body: str, old: str | None) -> str:
    """A regenerated review keeps the humans' 'Reviewer notes' section from the old file."""
    if old and NOTES in old:
        return body.split(NOTES)[0] + NOTES + old.split(NOTES, 1)[1]
    return body


def resolve_conflicts(decide) -> bool:
    """Resolve a rebase stop on our own files; False (and abort) on anything else."""
    files = [f for f in git("diff", "--name-only", "--diff-filter=U").stdout.split() if f]
    for f in files:
        if f == "league/champion.json":
            # In a rebase ':2' is upstream, ':3' is our commit being replayed.
            merged = merge_champion(show(":1", f), show(":2", f), show(":3", f), decide)
            save_json(CHAMPION, merged)
        elif f.startswith("league/mimics/"):
            # Evolved mimic raced on the same map: keep ours (both are fine sparring partners).
            save_json(
                ROOT / f, merge_champion(show(":1", f), show(":2", f), show(":3", f), lambda m, a, b: b)
            )
        elif f == "league/ledger.json":
            save_json(LEDGER, merge_ledger(show(":2", f), show(":3", f)))
        elif f == "league/experiments.json":
            save_json(ROOT / f, merge_experiments(show(":3", f), show(":2", f)))
        elif f == "league/ml_jobs.json":
            save_json(ROOT / f, merge_experiments(show(":3", f), show(":2", f), key="jobs"))
        elif f.startswith("league/review/"):
            ours, up = git("show", f":3:{f}"), git("show", f":2:{f}")
            (ROOT / f).write_text(
                splice_notes(ours.stdout, up.stdout if up.returncode == 0 else None),
                encoding="utf-8",
                newline="\n",
            )
        elif f.startswith("data/mimic/"):
            git("checkout", "--theirs", "--", f)
        elif f != "bots/main/src/pack/pack.bin":
            log(f"merge needs a human: {files}")
            git("rebase", "--abort")
            return False
    if files:
        apply_champion(load_champion())  # pack.bin always follows champion.json
        git("add", "--", *files, "bots/main/src/pack/pack.bin")
    return True


def rebasing() -> bool:
    return any(
        (ROOT / git("rev-parse", "--git-path", d).stdout.strip()).exists()
        for d in ("rebase-merge", "rebase-apply")
    )


def integrate(decide) -> bool:
    """Rebase local commits on origin/league; our JSON files are merged semantically."""
    if git("fetch", "-q", "origin", BRANCH).returncode != 0:
        log("git fetch failed (offline?)")
        return False
    # pack.bin is regenerated from champion.json after every pull (apply_champion), so the local
    # copy never needs keeping; stashing it was what left 'unmerged pack.bin' behind.
    if not rebasing():
        git("reset", "-q", "--", PACK.relative_to(ROOT).as_posix())
        git("checkout", "-q", "--", PACK.relative_to(ROOT).as_posix())
    p = git("rebase", "--autostash", f"origin/{BRANCH}")
    env = {**os.environ, "GIT_EDITOR": "true"}
    for _ in range(50):
        if p.returncode == 0:
            return True
        if not rebasing():
            log(f"rebase failed: {p.stderr.strip()[:300]}")
            return False
        if not resolve_conflicts(decide):
            return False
        p = git("rebase", "--continue", env=env)
    git("rebase", "--abort")
    return False


def publish(paths: list[Path], msg: str, decide) -> bool:
    rel = [p.relative_to(ROOT).as_posix() for p in paths if p.exists()]
    if not rel:
        return True
    git("add", "-A", "--", *rel)
    if git("diff", "--cached", "--quiet", "--", *rel).returncode == 0:
        return True
    if git("commit", "-q", "-m", f"{msg}\n\nleague host: {HOST}", "--", *rel).returncode != 0:
        log("git commit failed")
        return False
    for _ in range(3):
        if git("push", "-q", "origin", f"HEAD:{BRANCH}").returncode == 0:
            log(f"pushed: {msg}")
            return True
        if not integrate(decide):
            return False
    log("push failed 3 times; the commit stays local and goes out with the next push")
    return False


# ----------------------------------------------------------------- platform (read-only + submit)


def cambc_json(args: list[str], out: Path | None = None) -> dict | None:
    """A read-only cambc platform call with backoff on HTTP 429."""
    delay = 2.0
    for _ in range(5):
        try:
            p = sh([str(CAMBC), *args, "--json", *(["-o", str(out)] if out else [])], timeout=120)
        except subprocess.TimeoutExpired:
            p = None
        if p is not None and p.returncode == 0:
            if out:
                return {"ok": True}
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(p.stdout)
            return None
        if p is not None and "429" not in p.stderr and "busy" not in p.stderr:
            log(f"cambc {' '.join(args[:2])} failed: {p.stderr.strip()[:200]}")
            return None
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return None


def download_games(team_id: str, slug: str, limit: int) -> int:
    """Newest finished games of a team: match list (meta for mimic.load_games) + summaries."""
    got = cambc_json(["matches", "list", "--team", team_id, "--limit", str(max(limit, 10))])
    if not got:
        return 0
    GAMES.mkdir(parents=True, exist_ok=True)
    lst = GAMES / f"list_{slug}.json"
    known = {m["matchId"]: m for m in load_json(lst, {}).get("matches", [])}
    new = 0
    for m in got.get("matches", []):
        known[m["matchId"]] = m
        f = GAMES / f"{m['matchId']}.json"
        fresh = m.get("status") == "finished" and not f.exists() and new < limit
        if fresh and cambc_json(["matches", "summary", m["matchId"]], out=f):
            new += 1
    save_json(lst, {"matches": sorted(known.values(), key=lambda m: m.get("createdAt", ""))})
    return new


def sync(led: dict) -> str:
    """Lead: refresh the top 10 and the teams we lost to, and their mimic specs."""
    lad = cambc_json(["ladder", "--limit", str(TOP_N + 5)])
    if not lad:
        return "sync: ladder unavailable"
    teams = [t for t in lad.get("teams", []) if t.get("teamId") != OUR_TEAM][:TOP_N]
    snap = [
        {
            "rank": t["rank"],
            "name": t["name"],
            "teamId": t["teamId"],
            "rating": round(t["rating"], 1),
            "slug": mimic.slug(t["name"]),
        }
        for t in teams
    ]
    led["top"] = (led["top"] + [{"time": now_iso(), "teams": snap}])[-500:]
    top_ids = {t["teamId"] for t in snap}
    recent = {}
    for x in sorted(led["losses"], key=lambda x: x["time"], reverse=True):
        if x.get("oppId") and x["oppId"] not in top_ids and len(recent) < LOSS_MIMICS:
            recent.setdefault(x["oppId"], x["opp"])
    wanted = [(t["name"], t["teamId"], f"top{t['rank']}") for t in snap]
    wanted += [(name, tid, "loss") for tid, name in recent.items()]
    geo = mimic.map_geometry()
    prev = {p["slug"]: p for p in led["pool"]}
    pool, new, changed = [], 0, []
    MIMICS.mkdir(parents=True, exist_ok=True)
    for name, tid, why in wanted:
        slug = mimic.slug(name) or tid[:8]
        new += download_games(tid, slug, GAMES_PER_TEAM)
        clone = mimic.extract(name, geo)
        if not clone["games"]:
            log(f"sync: no games for {name}, no mimic")
            continue
        save_json(MIMICS / f"{slug}.json", clone)
        save_json(MIMICS / f"{slug}.spec.json", mimic.pack_spec(clone, sorted(geo)))
        entry = {"slug": slug, "name": name, "teamId": tid, "why": why, "games": clone["games"]}
        pool.append({**entry, **static_state(prev.get(slug), macro_hash(clone), now_iso())})
        if prev.get(slug, {}).get("macro_hash") not in (None, pool[-1]["macro_hash"]):
            changed.append(slug)
            (EVOLVED / f"{slug}.json").unlink(missing_ok=True)  # the real team moved: restart
    keep = {p["slug"] for p in pool}
    for f in MIMICS.glob("*.json"):
        if f.name.split(".")[0] not in keep:
            f.unlink()
    led["pool"] = pool
    return f"sync: {len(pool)} mimics ({new} new games{', changed code: ' + ', '.join(changed) if changed else ''})"


def static_state(prev: dict | None, h: str, now: str) -> dict:
    """macro_hash, since (when this macro was first seen) and static (unchanged since the last
    sync: the team has not updated its bot, so our evolved mimic of it carries the training)."""
    if prev and prev.get("macro_hash") == h:
        return {"macro_hash": h, "since": prev.get("since") or now, "static": True}
    return {"macro_hash": h, "since": now, "static": False}


# ----------------------------------------------------------------- map pool


def parse_fingerprint(lines: list[str]) -> int | None:
    """The bot's debug line `F {"fingerprint": <u32>, ...}` (draft.rs) -> the fingerprint."""
    for line in lines:
        if line.startswith("F "):
            with contextlib.suppress(json.JSONDecodeError, KeyError, TypeError):
                return int(json.loads(line[2:])["fingerprint"])
    return None


def register_map(name: str, path: Path) -> bool:
    """Fingerprint (pack lookup key) and geometry (mimic openings) of a new map, from one debug
    match of our bot: it prints `F` (fingerprint) and `M` (terrain/group dump) lines."""
    res = runner.run_matches("main", "baseline", maps=[path], out_dir=SCRATCH / "register" / name, swap=False)
    try:
        log_a = replay.load(res[0].replay).team("a")
    except (OSError, KeyError, ValueError, IndexError) as ex:
        log(f"maps: {name}: no replay ({ex!r})")
        return False
    fp = parse_fingerprint(log_a.other)
    if fp is None or not log_a.map:
        log(f"maps: {name}: no F/M debug lines in the replay")
        return False
    fps, geo = load_json(FINGERPRINTS, {}), load_json(GEOMETRY, {})
    fps[name] = fp
    geo[name] = mimic.geometry_from_dump(log_a.map)
    save_json(FINGERPRINTS, fps)
    save_json(GEOMETRY, geo)
    return True


def tidy_maps(pool: set[str]) -> None:
    """maps/ holds exactly the platform pool: cambc names pulled files '00contest', so an older
    unprefixed copy of a pool map is stale and goes; maps that left the pool move to
    maps/retired/ (kept for reference, not trained or gated on)."""
    files = runner.list_maps()
    prefixed = {map_name(p.stem) for p in files if p.stem[:1].isdigit()}
    for p in files:
        n = map_name(p.stem)
        if not p.stem[:1].isdigit() and n in prefixed:
            p.unlink()
        elif n not in pool:
            (ROOT / "maps" / "retired").mkdir(exist_ok=True)
            p.replace(ROOT / "maps" / "retired" / p.name)


def refresh_maps(led: dict, ch: dict) -> str:
    """Lead: follow the platform's current map pool: pull new maps, register them, reopen the
    losses that were waiting for them."""
    names, page = [], 1
    while page:
        got = cambc_json(["maps", "list", "--limit", "100", "--page", str(page)])
        if not got:
            return "maps: list unavailable"
        names += [map_name(m["name"]) for m in got.get("maps", [])]
        page = got.get("nextPage")
    p = sh([str(CAMBC), "maps", "pull"], timeout=300)  # keeps local files; needs cambc >= 3.0.1
    if p.returncode != 0:
        log(f"maps pull failed (cambc >= 3.0.1 needed): {p.stderr.strip()[:200]}")
    tidy_maps(set(names))
    local = local_maps()
    fps, geo = load_json(FINGERPRINTS, {}), load_json(GEOMETRY, {})
    hashes = led.setdefault("map_files", {})
    added = []
    for n in names:
        if n not in local:
            continue
        h = hashlib.sha1(local[n].read_bytes()).hexdigest()[:12]
        if (n not in fps or n not in geo or hashes.get(n) != h) and register_map(n, local[n]):
            hashes[n] = h
            added.append(n)
    led["maps"] = sorted(n for n in names if n in local)
    missing = sorted(set(names) - set(local))
    for x in led["losses"]:
        if x["status"] == "needs-human" and x.get("hint", "").startswith("map not in") and x["map"] in local:
            x.update(status="open", tries_at=total_tries(ch, x["map"]))
            x.pop("hint", None)
    return f"maps: pool {len(names)}, registered {len(added)} new" + (
        f", missing {missing}" if missing else ""
    )


def live_version(led: dict, at: float) -> int | None:
    subs = [s for s in led["submissions"] if s.get("ok") and ts(s["time"]) <= at]
    return subs[-1]["version"] if subs else None


def death_summary(path: Path, our_side: str) -> dict:
    d = load_json(path, {})
    ctx: Counter = Counter()
    deaths = {"teamA": 0, "teamB": 0}
    for e in d.get("events", []):
        if e["kind"] != "death":
            continue
        team = (e.get("owner") or {}).get("team")
        deaths[team] = deaths.get(team, 0) + len(e["entries"])
        if team == our_side:
            ctx.update(en.get("context", "?") for en in e["entries"])
    them = "teamB" if our_side == "teamA" else "teamA"
    gems = {t["team"]: t.get("submittedGems") for t in d.get("teams", [])}
    return {
        "deaths_us": deaths[our_side],
        "deaths_them": deaths[them],
        "contexts": dict(ctx.most_common(4)),
        "gems_us": gems.get(our_side),
        "gems_them": gems.get(them),
    }


def watch(led: dict, ch: dict) -> str:
    """Lead: record our new ladder games; losses join the queue with their death summary."""
    got = cambc_json(["matches", "list", "--team", OUR_TEAM, "--limit", "50"])
    if not got:
        return "watch: unavailable"
    maps, new, lost = active_maps(led), 0, 0
    for m in got.get("matches", []):
        mid = m["matchId"]
        if m.get("status") != "finished" or mid in led["ladder"]:
            continue
        us = next((t for t in m["teams"] if t["teamId"] == OUR_TEAM), None)
        opp = next((t for t in m["teams"] if t["teamId"] != OUR_TEAM), None)
        if not us or not opp:
            continue
        mp, t = map_name(m["map"]), m.get("createdAt") or now_iso()
        led["ladder"][mid] = {
            "time": t,
            "kind": m.get("kind"),
            "map": mp,
            "opp": opp["name"],
            "oppId": opp["teamId"],
            "win": us["result"] == "win",
            "live": live_version(led, ts(t)),
        }
        new += 1
        if us["result"] != "loss":
            continue
        lost += 1
        loss = {
            "id": mid,
            "time": t,
            "map": mp,
            "opp": opp["name"],
            "oppId": opp["teamId"],
            "status": "open",
            "tries_at": total_tries(ch, mp),
        }
        OURS.mkdir(parents=True, exist_ok=True)
        f = OURS / f"{mid}.json"
        if f.exists() or cambc_json(["matches", "summary", mid], out=f):
            loss.update(death_summary(f, "team" + us["side"].upper()))
        if mp not in maps:
            loss.update(status="needs-human", hint="map not in maps/ yet: the next map sync registers it")
        led["losses"].append(loss)
    return f"watch: {new} new games, {lost} losses"


def reconcile(led: dict, ch: dict) -> None:
    """Close losses whose map got a promotion; hand stuck ones to humans."""
    for x in led["losses"]:
        if x["status"] != "open":
            continue
        fix = next((h for h in ch["history"] if h["map"] == x["map"] and ts(h["time"]) > ts(x["time"])), None)
        if fix:
            x.update(status="fixed", fixed_in=fix["version"])
        elif total_tries(ch, x["map"]) - x.get("tries_at", 0) >= STUCK_TRIES:
            x["status"] = "needs-human"
            x["hint"] = classify(x, ch)


def classify(loss: dict, ch: dict) -> str:
    """Whose model is wrong? If our champion beats the opponent's mimic on that map locally but
    lost the real game, the mimic is missing something (micro). Otherwise ES could not fix it
    within θ: the engine needs a new feature or knob."""
    b = ch["bench"].get(str(ch["version"])) or {}
    local = (b.get("grid") or {}).get(mimic.slug(loss["opp"]), {}).get(loss["map"])
    if local is None:
        return "no mimic of this team: check their replay by hand"
    if local >= 0.5:
        return f"mimic can't express it (we win {local:.0%} vs their mimic here): improve the mimic"
    return f"model can't express it (still {local:.0%} vs their mimic after ES): new feature/knob in crates/"


def release_decision(ch: dict, led: dict, now: float) -> tuple[int, str] | None:
    """(version, reason) to submit now, or None. Needs a newer champion, SUBMIT_GAP since the
    last upload, and a reason: a ladder loss since then or a benchmark jump of BENCH_JUMP."""
    subs = [s for s in led["submissions"] if s.get("ok")]
    cur = ch["version"]
    # Before the league's first upload, compare against v0 (the seed champion).
    last = subs[-1] if subs else {"version": 0, "time": "1970-01-01T00:00:00Z"}
    if cur <= last["version"] or now - ts(last["time"]) < SUBMIT_GAP:
        return None
    if any(ts(x["time"]) > ts(last["time"]) for x in led["losses"]):
        return cur, "ladder loss since the last upload"
    b_cur, b_live = ch["bench"].get(str(cur)), ch["bench"].get(str(last["version"]))
    if b_cur and b_live and b_cur["overall"] >= b_live["overall"] + BENCH_JUMP:
        return cur, f"benchmark {b_live['overall']:.2f} -> {b_cur['overall']:.2f}"
    return None


def revert_decision(led: dict) -> int | None:
    """The previous version, if the live one has REVERT_GAMES ranked games and does worse."""
    subs = [s for s in led["submissions"] if s.get("ok")]
    if len(subs) < 2 or subs[-1].get("reason", "").startswith("revert"):
        return None

    def rate(v: int, recent: int | None = None) -> tuple[float, int]:
        g = sorted(
            (x for x in led["ladder"].values() if x.get("live") == v and x.get("kind") == "ranked"),
            key=lambda x: x.get("time", ""),
        )
        w = [x["win"] for x in g[-recent:]] if recent else [x["win"] for x in g]
        return (sum(w) / len(w) if w else 0.0), len(w)

    (wl, nl), (wp, np_) = rate(subs[-1]["version"]), rate(subs[-2]["version"], REVERT_RECENT)
    if nl < REVERT_GAMES or np_ < REVERT_GAMES:
        return None
    noise = 2 * math.sqrt(max(wp * (1 - wp), 0.05) / nl)
    return subs[-2]["version"] if wl < wp - max(REVERT_DROP, noise) else None


def submit(led: dict, version: int, reason: str, bot: str | None = None) -> bool:
    """Upload a version: snapshot bots/lg-vN (kept for reverts), bundle the zip, cambc submit."""
    bot = bot or f"lg-v{version}"
    if not (ROOT / "bots" / bot).exists():
        if version != load_champion()["version"]:
            log(f"submit: bots/{bot} is missing and v{version} is not the current champion")
            return False
        make_bot(bot, PACK.read_bytes())
    zip_path = ROOT / "submission" / f"{bot}.zip"
    zip_path.parent.mkdir(exist_ok=True)
    sh([str(CAMBC), "bundle", bot, "--output", str(zip_path)])
    p = sh([str(CAMBC), "submit", bot], timeout=300)
    ok = p.returncode == 0
    led["submissions"].append(
        {
            "version": version,
            "time": now_iso(),
            "reason": reason,
            "ok": ok,
            "bot": bot,
            "host": HOST,
            "output": (p.stdout + p.stderr).strip()[:300],
        }
    )
    log(f"submit v{version} ({reason}): {'ok' if ok else 'FAILED ' + p.stderr.strip()[:200]}")
    if ok:
        n = supersede_losses(led, version)
        log(f"loss queue: {n} open losses superseded by v{version} (they still weigh in map pressure)")
    return ok


def supersede_losses(led: dict, version: int) -> int:
    """A new upload starts a fresh queue: open losses (of older versions) leave it but stay in
    the ledger and the review, and keep steering training through map_pressure."""
    n = 0
    for x in led["losses"]:
        if x["status"] == "open":
            x.update(status="superseded", superseded_by=version)
            n += 1
    return n


def map_pressure(led: dict, now: float) -> dict[str, float]:
    """Recent ladder losses per map, decaying with time constant PRESSURE_TAU; losses
    of the version live now count double."""
    subs = [s for s in led["submissions"] if s.get("ok")]
    live = subs[-1]["version"] if subs else None
    out: dict[str, float] = {}
    for x in led["losses"]:
        if x["status"] == "needs-human":
            continue
        age = max(0.0, now - ts(x["time"]))
        w = math.exp(-age / PRESSURE_TAU)
        if live is not None and (led["ladder"].get(x["id"]) or {}).get("live") == live:
            w *= 2.0
        out[x["map"]] = out.get(x["map"], 0.0) + w
    return out


# ----------------------------------------------------------------- review report


def review_text(ch: dict, led: dict) -> str:
    snaps = led["top"]
    top = snaps[-1]["teams"] if snaps else []
    day_ago = (
        next((s for s in reversed(snaps) if ts(s["time"]) <= ts(snaps[-1]["time"]) - 86400), None)
        if snaps
        else None
    )
    old = {t["teamId"]: t for t in (day_ago or {}).get("teams", [])}
    b = ch["bench"].get(str(ch["version"])) or {}
    pool = {p["teamId"]: p for p in led["pool"]}
    subs = [s for s in led["submissions"] if s.get("ok")]
    L = [
        f"# League review {time.strftime('%Y-%m-%d')}",
        "",
        f"Champion **v{ch['version']}** ({len(ch['maps'])} maps tuned) · live on the ladder: "
        f"**{'v' + str(subs[-1]['version']) if subs else 'none uploaded by the league'}** · "
        f"last sync {snaps[-1]['time'] if snaps else 'never'} · benchmark vs mimics {b.get('overall', '—')}",
        "",
        f"Map pool: {len(led['maps'])} maps ({', '.join(led['maps']) or 'not synced: all local maps'}). "
        f"Trained: {map_report(ch, led)['trained']}/{map_report(ch, led)['active']}; "
        f"still on default numbers: {', '.join(map_report(ch, led)['untrained']) or 'none'}.",
        "",
        "## Top 10",
        "",
        "`gap` = local win rate vs the team's mimic − real ladder win rate vs the team. A large positive gap "
        "means the mimic is easier than the real team: its macro was cloned, its micro was not. "
        "Fix the mimic before trusting training against it. `vs evolved` = our win rate vs the evolved "
        "mimic (same macro, micro trained to beat us); `code` = whether the team's macro changed since "
        "the last sync (static teams are carried on by their evolved mimic).",
        "",
        "| # | team | rating (24h Δ) | ladder W-L vs us | local WR vs mimic | gap | vs evolved (gen) "
        "| code | mimic games |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    hard = ch["bench_hard"].get(str(ch["version"])) or {}
    flags = []
    for t in top:
        games = [x["win"] for x in led["ladder"].values() if x["oppId"] == t["teamId"]]
        w = sum(games)
        lw = sum(games) / len(games) if games else None
        local = (b.get("per_team") or {}).get(t["slug"])
        gap = local - lw if local is not None and lw is not None and len(games) >= 3 else None
        d = t["rating"] - old[t["teamId"]]["rating"] if t["teamId"] in old else None
        pe = pool.get(t["teamId"], {})
        mg = pe.get("games")
        gen = load_evolved(t["slug"], pe.get("macro_hash"))["version"]
        hw = (hard.get("per_team") or {}).get(t["slug"])
        code = "—" if not pe else f"static since {pe['since'][:10]}" if pe.get("static") else "changed"
        L.append(
            f"| {t['rank']} | {t['name']} | {t['rating']:.0f} ({'—' if d is None else f'{d:+.0f}'}) | "
            f"{w}-{len(games) - w} | {'—' if local is None else f'{local:.2f}'} | "
            f"{'—' if gap is None else f'{gap:+.2f}'} | {'—' if hw is None else f'{hw:.2f}'} (g{gen}) | "
            f"{code} | {mg if mg is not None else '—'} |"
        )
        if gap is not None and gap >= 0.3:
            flags.append(
                f"- **{t['name']}**: mimic looks too weak (gap {gap:+.2f}); watch their replays for micro we don't model."
            )
        if mg is not None and mg < 5:
            flags.append(f"- **{t['name']}**: mimic built from only {mg} games; treat its numbers as noise.")
        if t["teamId"] not in pool:
            flags.append(f"- **{t['name']}**: no mimic (no downloadable games).")
    L += ["", "## Epistemic flags", "", *(flags or ["- none"]), ""]
    stuck = [x for x in led["losses"] if x["status"] == "needs-human"]
    open_ = [x for x in led["losses"] if x["status"] == "open"]
    L += [f"## Losses ES could not fix ({len(stuck)}; {len(open_)} still being trained)", ""]
    if stuck:
        L += [
            "| date | map | opponent | deaths us/them | our deaths by | hint | replay |",
            "|---|---|---|---|---|---|---|",
        ]
        for x in stuck[-20:]:
            ctx = ", ".join(f"{k} {v}" for k, v in (x.get("contexts") or {}).items()) or "—"
            L.append(
                f"| {x['time'][:10]} | {x['map']} | {x['opp']} | {x.get('deaths_us', '?')}/{x.get('deaths_them', '?')} | "
                f"{ctx} | {x.get('hint', '')} | `cambc matches download {x['id']} -o runs/review/{x['id'][:8]}.repl26a` |"
            )
    else:
        L.append("- none")
    L += [
        "",
        "## Recent promotions",
        "",
        "| v | map | gain | h2h | host | benchmark |",
        "|---|---|---|---|---|---|",
    ]
    for h in ch["history"][-10:][::-1]:
        bb = ch["bench"].get(str(h["version"]), {}).get("overall", "—")
        L.append(
            f"| {h['version']} | {h['map']} | {h['gain']:+.3f} | {h.get('h2h', '—')} | {h['host']} | {bb} |"
        )
    L += ["", "## Ladder by uploaded version", "", "| version | reason | ranked W-L |", "|---|---|---|"]
    for s in subs[-8:][::-1]:
        g = [
            x["win"]
            for x in led["ladder"].values()
            if x.get("live") == s["version"] and x.get("kind") == "ranked"
        ]
        L.append(f"| v{s['version']} | {s.get('reason', '')} | {sum(g)}-{len(g) - sum(g)} |")
    L += [
        "",
        NOTES,
        "",
        "_Humans: write findings here (feature ideas, mimic fixes). The harness keeps this section._",
        "",
    ]
    return "\n".join(L)


def write_review(ch: dict, led: dict) -> Path:
    p = REVIEW / f"{time.strftime('%Y-%m-%d')}.md"
    old = p.read_text(encoding="utf-8") if p.exists() else None
    if old is None:
        prev = sorted(REVIEW.glob("*.md"))
        old = prev[-1].read_text(encoding="utf-8") if prev else None  # notes carry over to the next day
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(splice_notes(review_text(ch, led), old), encoding="utf-8", newline="\n")
    return p


# ----------------------------------------------------------------- the cycle


def load_local() -> dict:
    return {
        "sigma": {},
        "msigma": {},  # evolved-mimic ES steps, per "slug|map"
        "pwin": {},
        "tries": {},
        "rr": 0,
        "mrr": 0,
        "mlast": {},
        "cycle": 0,
        "last_sync": 0,
        "last_watch": 0,
        "last_heartbeat": 0,
        **load_json(LOCAL, {}),
    }


def fold_tries(ch: dict, local: dict) -> None:
    for m, n in local["tries"].items():
        ch["tries"].setdefault(m, {})[HOST] = max(n, ch["tries"].get(m, {}).get(HOST, 0))


def tries_now(ch: dict, local: dict, mp: str) -> int:
    """ES rounds on a map by all PCs: the shared counts plus this PC's not yet pushed ones."""
    per = dict(ch["tries"].get(mp) or {})
    per[HOST] = max(per.get(HOST, 0), local["tries"].get(mp, 0))
    return sum(per.values())


def trained_maps(ch: dict) -> set[str]:
    """Maps whose θ differs from the compiled defaults (a promotion or the old seed)."""
    base = base_theta()
    return {m for m, th in ch["maps"].items() if full_theta(th) != base}


def trainable_maps(led: dict) -> dict[str, Path]:
    """Active maps the bot can recognise: without a fingerprint a map's θ never reaches
    pack.bin, so training it would compare identical bots."""
    fps = load_json(FINGERPRINTS, {})
    return {m: p for m, p in active_maps(led).items() if m in fps}


def seed_maps(ch: dict, led: dict) -> list[str]:
    """Give every trainable map an explicit θ (the defaults), so champion.json, pack.bin and
    the map pool always have the same maps. Defaults play exactly like no entry."""
    new = [m for m in trainable_maps(led) if m not in ch["maps"]]
    for m in new:
        ch["maps"][m] = base_theta()
    return new


def map_report(ch: dict, led: dict) -> dict:
    """The per-map numbers that must agree: active pool vs fingerprints, geometry, pack, trained."""
    active = set(active_maps(led))
    fps, geo = set(load_json(FINGERPRINTS, {})), set(load_json(GEOMETRY, {}))
    trained = trained_maps(ch) & active
    return {
        "active": len(active),
        "fingerprint": len(active & fps),
        "geometry": len(active & geo),
        "pack": len(active & set(ch["maps"]) & fps),
        "trained": len(trained),
        "untrained": sorted(active - trained),
        "missing": sorted(active - (fps & geo)),
        "dropped": sorted(set(ch["maps"]) - fps),  # θ that pack.bin can't carry
    }


def pick_map(ch: dict, led: dict, local: dict) -> tuple[str, dict | None]:
    """1. maps never trained (new maps), until each had BOOTSTRAP_TRIES rounds; maps with
    ladder losses first. 2. the oldest open ladder loss not stuck yet. 3. our worst
    benchmark map / round-robin."""
    maps = trainable_maps(led)
    open_losses = sorted(
        (x for x in led["losses"] if x["status"] == "open" and x["map"] in maps), key=lambda x: x["time"]
    )
    trained = trained_maps(ch)
    fresh = [m for m in maps if m not in trained and tries_now(ch, local, m) < BOOTSTRAP_TRIES]
    if fresh:
        lost_on = {x["map"] for x in open_losses}
        mp = min(fresh, key=lambda m: (m not in lost_on, tries_now(ch, local, m), m))
        return mp, next((x for x in open_losses if x["map"] == mp), None)
    for x in open_losses:
        if tries_now(ch, local, x["map"]) - x.get("tries_at", 0) < STUCK_TRIES:
            return x["map"], x
    per_map = {
        m: v
        for m, v in ((ch["bench"].get(str(ch["version"])) or {}).get("per_map") or {}).items()
        if m in maps
    }
    # Queue empty (e.g. just after an upload): train where we recently lost on the ladder and
    # still lose locally; every map keeps a small chance.
    pressure = {m: v for m, v in map_pressure(led, time.time()).items() if m in maps}
    if pressure:
        names = sorted(maps)
        floor = PRESSURE_FLOOR * max(pressure.values()) + 0.05
        weights = [(pressure.get(m, 0.0) + floor) * (1.1 - per_map.get(m, 0.5)) for m in names]
        return random.Random(local["rr"]).choices(names, weights=weights)[0], None
    if per_map and local["rr"] % 2 == 0:
        return min(per_map, key=per_map.get), None
    names = sorted(maps)
    return names[local["rr"] % len(names)], None


def h2h_decider(cyc: Path, maps_now: dict):
    """For merge_champion: two promotions of the same map are settled by a head-to-head there."""

    def decide(m: str, a: list[float], b: list[float]) -> list[float]:
        ba = make_bot("_lg-merge-a", pack_bytes_for({**maps_now, m: a}))
        bb = make_bot("_lg-merge-b", pack_bytes_for({**maps_now, m: b}))
        res = runner.run_matches(ba, bb, maps=[local_maps()[m]], out_dir=cyc / "merge", release=True)
        win = sum(r.score_for(ba) for r in res) / max(len(res), 1)
        log(
            f"merge: both sides promoted {m}; head-to-head {win:.2f} -> keep {'upstream' if win >= 0.5 else 'ours'}"
        )
        return a if win >= 0.5 else b

    return decide


def train(ch: dict, led: dict, local: dict, rng: random.Random, cyc: Path) -> dict:
    mp, loss = pick_map(ch, led, local)
    local["rr"] += 1
    mpath = trainable_maps(led)[mp]
    faithful, evolved = ensure_mimics(led)
    mimics = faithful + evolved  # PFSP over both: the evolved ones keep pressure on static teams
    anchors = [a for a in ANCHORS if (ROOT / "bots" / a).exists()]
    opps = pfsp(local, mimics, N_OPPS, rng) + rng.sample(anchors, min(N_ANCHORS, len(anchors)))
    if not mimics:
        log("no mimics yet (the lead's first sync builds them): training against anchors only")
        opps = rng.sample(anchors, min(N_OPPS, len(anchors)))
    theta = full_theta(ch["maps"].get(mp))
    s0 = params.sigma0()
    sigma = list(local["sigma"].get(mp, s0)) + s0[len(local["sigma"].get(mp, s0)) :]
    cands = []
    for k in range(LAMBDA // 2):
        eps = [rng.gauss(0, 1) for _ in theta]
        for sgn in (1, -1):
            th = clip_search([t + sgn * s * e for t, s, e in zip(theta, sigma, eps, strict=False)])
            cands.append(
                (make_bot(f"_lg-c{k}{'p' if sgn > 0 else 'n'}", pack_bytes_for({**ch["maps"], mp: th})), th)
            )
    res = play(["main"] + [b for b, _ in cands], opps, [mpath], cyc / "es")
    champ_f = fitness(res, "main")[0]
    for o in opps:  # PFSP win rates: an exponential average of the champion's score
        local["pwin"][o] = round(0.7 * local["pwin"].get(o, 0.5) + 0.3 * score(res, "main", o), 3)
    scored = sorted(((fitness(res, b)[0], b, th) for b, th in cands), key=lambda x: -x[0])
    best_f, best, best_th = scored[0]
    mult = 1.15 if best_f > champ_f else 0.85
    local["sigma"][mp] = [min(max(s * mult, 0.25 * z), 2.0 * z) for s, z in zip(sigma, s0, strict=False)]
    local["tries"][mp] = local["tries"].get(mp, 0) + 1
    entry = {
        "map": mp,
        "loss": loss["id"][:8] if loss else None,
        "opps": opps,
        "champ": round(champ_f, 3),
        "best": round(best_f, 3),
        "gain": round(best_f - champ_f, 3),
    }
    if best_f < champ_f + ACCEPT_GAIN:
        return entry
    # Confirm on fresh opponents (the engine is chaotic: few games are noisy), head-to-head, fuel.
    fresh = [b for b in mimics + anchors if b not in opps]
    fresh = pfsp(local, fresh, N_CONFIRM, rng)
    conf = play(["main", best], fresh, [mpath], cyc / "confirm") if fresh else []
    h2h = runner.run_matches(best, "main", maps=[mpath], out_dir=cyc / "h2h", release=True)
    hw = sum(r.score_for(best) for r in h2h) / max(len(h2h), 1)
    cf, chf = (fitness(conf, best)[0], fitness(conf, "main")[0]) if conf else (best_f, champ_f)
    es_best = [r for r in res if best in (r.bot_a, r.bot_b)]
    es_main = [r for r in res if "main" in (r.bot_a, r.bot_b)]
    susp_b, susp_c = suspended(es_best, best), suspended(es_main, "main")
    entry.update(h2h=round(hw, 2), confirm=round(cf - chf, 3), susp=[susp_b, susp_c])
    if hw < 0.5 or cf < chf or susp_b > max(2 * susp_c, 20):
        return entry
    ch["maps"][mp] = [round(x, 5) for x in best_th]
    ch["history"].append(
        {
            "version": len(ch["history"]) + 1,
            "host": HOST,
            "time": now_iso(),
            "map": mp,
            "gain": round(best_f - champ_f, 3),
            "h2h": round(hw, 2),
            "loss": loss["id"] if loss else None,
            "opps": opps,
        }
    )
    ch["version"] = len(ch["history"])
    apply_champion(ch)
    runner.build("main", release=True, force=True)  # raises if the champion doesn't build
    maps = list(active_maps(led).values())
    if faithful:  # honest tracking: the faithful mimics only
        ch["bench"][str(ch["version"])] = benchmark("main", faithful, maps, cyc / "bench")
    if evolved:
        ch["bench_hard"][str(ch["version"])] = benchmark("main", evolved, maps, cyc / "bench-hard")
    entry["promoted"] = ch["version"]
    return entry


def mimic_target(ch: dict, led: dict, local: dict) -> tuple[dict, str] | None:
    """(pool entry, map) for an evolved-mimic cycle: the static team our champion beats most
    (the opponent we've 'solved'), on the map where we beat it most. Without a benchmark:
    round-robin over the pool."""
    pool = [p for p in led["pool"] if (MIMICS / f"{p['slug']}.spec.json").exists()]
    maps = trainable_maps(led)
    if not pool or not maps:
        return None
    static = [p for p in pool if p.get("static")] or pool
    b = ch["bench"].get(str(ch["version"])) or {}
    per_team = b.get("per_team") or {}
    last = local["mlast"]  # slug / "slug|map" -> cycle it was last evolved
    # Among the 3 teams we beat most, the one evolved least recently (no single-team rut).
    best = sorted(static, key=lambda p: -per_team.get(p["slug"], 0.0))[:3] if per_team else static
    team = min(best, key=lambda p: (last.get(p["slug"], -1), p["slug"]))
    grid = {m: v for m, v in ((b.get("grid") or {}).get(team["slug"]) or {}).items() if m in maps}
    # Among the 5 maps we beat them most on (all maps without a benchmark), least recent first.
    cands = sorted(grid, key=lambda m: -grid[m])[:5] if grid else sorted(maps)
    mp = min(cands, key=lambda m: (last.get(f"{team['slug']}|{m}", -1), m))
    last[team["slug"]] = last[f"{team['slug']}|{mp}"] = local["cycle"]
    return team, mp


def evolve_mimic(ch: dict, led: dict, local: dict, rng: random.Random, cyc: Path) -> dict:
    """Train an opponent instead of ourselves: ES on one team's evolved-mimic θ for one map, so
    that it beats our champion. Draft, builds, openings and cadence stay the team's (LOCKED);
    only micro/targeting weights move. Faithful mimics are never touched."""
    tgt = mimic_target(ch, led, local)
    local["mrr"] += 1
    if not tgt:
        return {"mimic": None, "skipped": "no mimic pool"}
    team, mp = tgt
    slug, mpath = team["slug"], trainable_maps(led)[mp]
    spec = json.loads((MIMICS / f"{slug}.spec.json").read_text(encoding="utf-8"))
    mx = load_evolved(slug, team.get("macro_hash"))
    _faithful, evolved = ensure_mimics(led)
    cur = f"mimx-{slug}" if f"mimx-{slug}" in evolved else f"mimic-{slug}"
    theta = full_theta(mx["maps"].get(mp) or spec_theta(spec, mp))
    key = f"{slug}|{mp}"
    s0 = [0.0 if i in LOCKED else s for i, s in enumerate(params.sigma0())]
    sigma = list(local["msigma"].get(key, s0))
    cands = []
    for k in range(LAMBDA // 2):
        eps = [rng.gauss(0, 1) for _ in theta]
        for sgn in (1, -1):
            th = clip_search([t + sgn * s * e for t, s, e in zip(theta, sigma, eps, strict=False)])
            name = f"_lg-m{k}{'p' if sgn > 0 else 'n'}"
            cands.append((make_bot(name, pack_bytes_for({**mx["maps"], mp: th}, spec)), th))
    res = play(["main"], [cur] + [b for b, _ in cands], [mpath], cyc / "mimic")
    cur_f = fitness(res, cur)[0]
    scored = sorted(((fitness(res, b)[0], b, th) for b, th in cands), key=lambda x: -x[0])
    best_f, best, best_th = scored[0]
    local["msigma"][key] = [
        min(max(s * (1.15 if best_f > cur_f else 0.85), 0.25 * z), 2.0 * z)
        for s, z in zip(sigma, s0, strict=False)
    ]
    entry = {
        "mimic": slug,
        "map": mp,
        "cur": round(cur_f, 3),
        "best": round(best_f, 3),
        "gain": round(best_f - cur_f, 3),
    }
    if best_f < cur_f + ACCEPT_GAIN:
        return entry
    h2h = runner.run_matches(best, cur, maps=[mpath], out_dir=cyc / "mimic-h2h", release=True)
    hw = sum(r.score_for(best) for r in h2h) / max(len(h2h), 1)
    susp_b, susp_c = suspended(res, best), suspended(res, cur)
    entry.update(h2h=round(hw, 2), susp=[susp_b, susp_c])
    if hw < 0.5 or susp_b > max(2 * susp_c, 20):
        return entry
    mx["maps"][mp] = [round(x, 5) for x in best_th]
    mx["history"].append(
        {
            "version": len(mx["history"]) + 1,
            "host": HOST,
            "time": now_iso(),
            "map": mp,
            "gain": round(best_f - cur_f, 3),
            "h2h": round(hw, 2),
        }
    )
    mx.update(version=len(mx["history"]), base=team.get("macro_hash"), slug=slug)
    save_json(EVOLVED / f"{slug}.json", mx)
    entry["evolved"] = mx["version"]
    return entry


#: Scratch that is safe to delete (results live in git), pruned oldest-first when the disk is low.
PRUNABLE = ("experiments", "ml", "cycles", "bench", "bench-hard", "manual", "replay")
MIN_FREE_GB, HARD_FREE_GB = 15.0, 5.0


def free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def prune_scratch(need_gb: float, free=free_gb, scratch: Path | None = None) -> list[str]:
    """Delete regenerable scratch folders, oldest first, until `need_gb` is free."""
    scratch = scratch or SCRATCH
    entries = [p for d in PRUNABLE if (scratch / d).is_dir() for p in (scratch / d).iterdir() if p.is_dir()]
    removed = []
    for p in sorted(entries, key=lambda p: p.stat().st_mtime):
        if free() >= need_gb:
            break
        shutil.rmtree(p, ignore_errors=True)
        removed.append(p.name)
    return removed


def ensure_space(free=free_gb, sleep=time.sleep, scratch: Path | None = None) -> None:
    """Keep the disk from filling: prune scratch below MIN_FREE_GB; below HARD_FREE_GB even
    after pruning, wait (and say so) instead of failing every build and git write."""
    if free() >= MIN_FREE_GB:
        return
    removed = prune_scratch(MIN_FREE_GB, free, scratch)
    if removed:
        log(f"disk low: pruned {len(removed)} scratch folders, {free():.1f} GB free")
    while free() < HARD_FREE_GB:
        log(f"disk almost full ({free():.1f} GB free): waiting 10 min; free space on the drive")
        sleep(600)


def cycle(a: argparse.Namespace, rng: random.Random) -> None:
    ensure_space()
    local = load_local()
    local["cycle"] += 1
    cyc = SCRATCH / "cycles" / f"{local['cycle'] % 20:02d}"
    shutil.rmtree(cyc, ignore_errors=True)
    use_git = a.git and git_ready()
    ch = load_champion()
    if use_git:
        integrate(h2h_decider(cyc, ch["maps"]))
        ch = load_champion()
    errs = params.check()
    if errs:
        raise SystemExit(f"bots/main/src/learned.rs and harness/params.py disagree: {errs}")
    led = load_ledger()
    now = time.time()
    rep = map_report(ch, led)
    if rep["missing"] or rep["dropped"]:
        log(
            f"maps out of sync: no fingerprint/geometry for {rep['missing']}, theta not in pack for {rep['dropped']}"
        )
        if a.lead and rep["missing"]:
            local["last_sync"] = 0  # register them now, not in up to 3 h
    seeded = seed_maps(ch, led)
    apply_champion(ch)
    maybe_release(a, ch, cyc, use_git)
    led = load_ledger()  # an upload just now is in the file, not in `led`: don't save over it
    if seeded:
        log(f"new maps get default theta until trained: {', '.join(seeded)}")
        save_json(CHAMPION, ch)
        if use_git:
            publish(
                [CHAMPION, PACK],
                f"league: default theta for {len(seeded)} new maps",
                h2h_decider(cyc, ch["maps"]),
            )
            ch = load_champion()
    if a.lead:
        msgs = []
        if now - local["last_watch"] >= WATCH_EVERY:
            msgs.append(watch(led, ch))
            local["last_watch"] = now
        if now - local["last_sync"] >= SYNC_EVERY:
            msgs.append(refresh_maps(led, ch))
            msgs.append(sync(led))
            local["last_sync"] = now
        if msgs:
            fold_tries(ch, local)
            reconcile(led, ch)
            rv = revert_decision(led) if a.auto_submit else None
            if rv is not None:
                submit(led, rv, f"revert: v{led['submissions'][-1]['version']} did worse on the ladder")
            write_review(ch, led)
            save_json(LEDGER, led)
            log("; ".join(msgs))
            if use_git:
                publish(LEAD_FILES, "league: " + "; ".join(msgs), h2h_decider(cyc, ch["maps"]))
    t0 = time.time()
    # New maps come first: no mimic rounds until every never-trained map had its bootstrap rounds.
    bootstrap = any(
        m not in trained_maps(ch) and tries_now(ch, local, m) < BOOTSTRAP_TRIES for m in trainable_maps(led)
    )
    if MIMIC_EVERY and local["cycle"] % MIMIC_EVERY == 0 and led["pool"] and not bootstrap:
        entry = evolve_mimic(ch, led, local, rng, cyc)
        log(f"cycle {local['cycle']} mimic ({round(time.time() - t0)}s): {json.dumps(entry)}")
        if "evolved" in entry and use_git:
            msg = f"league: mimx-{entry['mimic']} g{entry['evolved']} {entry['map']} {entry['gain']:+.3f}"
            publish([EVOLVED], msg, h2h_decider(cyc, ch["maps"]))
        save_json(LOCAL, local)
        return
    entry = train(ch, led, local, rng, cyc)
    log(f"cycle {local['cycle']} ({round(time.time() - t0)}s): {json.dumps(entry)}")
    promoted = "promoted" in entry
    if promoted or now - local["last_heartbeat"] >= HEARTBEAT:
        fold_tries(ch, local)
        save_json(CHAMPION, ch)
        local["last_heartbeat"] = now
        if use_git:
            msg = (
                f"league: v{ch['version']} {entry['map']} {entry['gain']:+.3f}"
                if promoted
                else f"league: try counts ({HOST})"
            )
            publish([CHAMPION, PACK], msg, h2h_decider(cyc, ch["maps"]))
            ch = load_champion()  # a merge may have renumbered the version
    maybe_release(a, ch, cyc, use_git)
    save_json(LOCAL, local)


def maybe_release(a: argparse.Namespace, ch: dict, cyc: Path, use_git: bool) -> None:
    """Lead with --auto-submit: upload if due. Runs at the start and the end of every cycle, so a
    restarted daemon uploads at once instead of after a full (first: slow) round."""
    if not (a.lead and a.auto_submit):
        return
    led = load_ledger()
    dec = release_decision(ch, led, time.time())
    if dec:
        submit(led, *dec)
        save_json(LEDGER, led)
        write_review(ch, led)
        if use_git:
            publish([LEDGER, REVIEW], f"league: uploaded v{dec[0]} ({dec[1]})", h2h_decider(cyc, ch["maps"]))


def next_upload(ch: dict, led: dict, now: float) -> str:
    """When auto-submit will upload next (from the ledger, the same on every PC)."""
    dec = release_decision(ch, led, now)
    if dec:
        return f"due now: v{dec[0]} ({dec[1]})"
    subs = [s for s in led["submissions"] if s.get("ok")]
    live = subs[-1]["version"] if subs else 0
    if ch["version"] <= live:
        return f"waiting for a version newer than v{live}"
    wait = SUBMIT_GAP - (now - ts(subs[-1]["time"])) if subs else 0
    if wait > 0:
        return f"v{ch['version']} in {int(wait // 3600)}h {int(wait % 3600 // 60)}m (if a loss or +{BENCH_JUMP:.2f} benchmark by then)"
    return f"v{ch['version']}: waiting for a ladder loss or a +{BENCH_JUMP:.2f} benchmark gain"


# ----------------------------------------------------------------- commands


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def running() -> int | None:
    with contextlib.suppress(OSError, ValueError):
        pid = int(LOCK.read_text().strip())
        if pid_alive(pid):
            return pid
    return None


def cmd_run(a: argparse.Namespace) -> int:
    if a.auto_submit and not a.lead:
        print("--auto-submit needs --lead (only the lead PC talks to the platform)")
        return 2
    pid = running()
    if pid:
        print(f"the league is already running here (pid {pid}); `train stop` first")
        return 1
    SCRATCH.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()))
    if os.name == "nt":
        # Keep Windows from idle-sleeping while the league runs (2026-10-21: the PC slept 4
        # minutes after a restart and the daemon was dead for 16 h). ES_CONTINUOUS |
        # ES_SYSTEM_REQUIRED; ends with the process, manual sleep still works.
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    log(
        f"league on {HOST}: {'lead' if a.lead else 'worker'}, auto-submit {'ON' if a.auto_submit else 'off'}, "
        f"git {'on' if a.git else 'off'}"
    )
    rng = random.Random(f"{HOST}-{time.time()}")
    try:
        n = 0
        while a.forever or n < a.cycles:
            n += 1
            try:
                cycle(a, rng)
            except SystemExit:
                raise
            except Exception as ex:  # keep the daemon alive; the log has the error
                try:
                    log(f"cycle error: {ex!r}")
                except Exception:
                    log("cycle error: [unprintable exception]")
                time.sleep(30)
    finally:
        LOCK.unlink(missing_ok=True)
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    ch, led, local = load_champion(), load_ledger(), load_local()
    pid = running()
    print(
        f"daemon      {'running, pid ' + str(pid) if pid else 'stopped'} on {HOST} (cycle {local['cycle']})"
    )
    print(f"champion    v{ch['version']}, trained on {', '.join(sorted(trained_maps(ch))) or '-'}")
    b = ch["bench"].get(str(ch["version"]))
    if b:
        print(f"benchmark   {b['overall']:.3f} over {b['games']} games")
        for t, wr in sorted(b["per_team"].items(), key=lambda kv: kv[1]):
            print(f"  vs {t:32s} {wr:.3f}")
    top = led["top"][-1] if led["top"] else None
    print(
        f"top 10      {'as of ' + top['time'] if top else 'never synced'}: "
        + ", ".join(f"{t['rank']}. {t['name']}" for t in (top or {}).get("teams", []))
    )
    print(
        f"maps        {len(active_maps(led))} active ({'platform pool' if led['maps'] else 'all local, not synced'})"
    )
    rep = map_report(ch, led)
    ok = rep["fingerprint"] == rep["geometry"] == rep["active"] and not rep["dropped"]
    print(
        f"            fingerprint {rep['fingerprint']}/{rep['active']}, geometry {rep['geometry']}/{rep['active']}, "
        f"in pack {rep['pack']}/{rep['active']}, trained {rep['trained']}/{rep['active']}"
        + ("" if ok else f"  OUT OF SYNC: missing {rep['missing']}, dropped {rep['dropped']}")
    )
    if rep["untrained"]:
        lc = load_local()
        print(
            "  new maps  "
            + ", ".join(f"{m} {tries_now(ch, lc, m)}/{BOOTSTRAP_TRIES}" for m in rep["untrained"])
        )
    print(f"pool        {', '.join(p['slug'] + ' (' + p['why'] + ')' for p in led['pool']) or '-'}")
    for p in led["pool"]:
        mx = load_evolved(p["slug"], p.get("macro_hash"))
        if mx["version"]:
            print(f"  evolved   mimx-{p['slug']}: g{mx['version']} on {', '.join(sorted(mx['maps']))}")
    c = Counter(x["status"] for x in led["losses"])
    print(
        f"losses      open {c['open']}, fixed {c['fixed']}, needs-human {c['needs-human']}, "
        f"superseded {c['superseded']}"
    )
    top = sorted(map_pressure(led, time.time()).items(), key=lambda kv: -kv[1])[:5]
    if top:
        print("  pressure  " + ", ".join(f"{m} {v:.1f}" for m, v in top))
    for x in [x for x in led["losses"] if x["status"] == "open"][:5]:
        print(
            f"  open: {x['map']} vs {x['opp']} ({x['time'][:16]}), tries {total_tries(ch, x['map']) - x.get('tries_at', 0)}/{STUCK_TRIES}"
        )
    for s in led["submissions"][-3:]:
        print(
            f"upload      v{s['version']} {s['time']} {'ok' if s['ok'] else 'FAILED'}: {s.get('reason', '')}"
        )
    for h in ch["history"][-5:]:
        print(f"promotion   v{h['version']} {h['time']} {h['map']} {h['gain']:+.3f} ({h['host']})")
    if led["submissions"] or ch["version"]:
        print(f"next upload {next_upload(ch, led, time.time())}")
    rev = sorted(REVIEW.glob("*.md"))
    print(f"review      {rev[-1].relative_to(ROOT) if rev else '-'}")
    return 0


def cmd_sync(a: argparse.Namespace) -> int:
    ch, led = load_champion(), load_ledger()
    print(watch(led, ch))
    print(refresh_maps(led, ch))
    print(sync(led))
    reconcile(led, ch)
    save_json(LEDGER, led)
    print(f"review: {write_review(ch, led).relative_to(ROOT)}")
    if a.git and git_ready():
        publish(LEAD_FILES, "league: manual sync", h2h_decider(SCRATCH / "manual", ch["maps"]))
    return 0


def cmd_maps(a: argparse.Namespace) -> int:
    ch, led = load_champion(), load_ledger()
    print(refresh_maps(led, ch))
    save_json(LEDGER, led)
    print(f"active: {', '.join(sorted(active_maps(led)))}")
    if a.git and git_ready():
        publish(LEAD_FILES, "league: map pool", h2h_decider(SCRATCH / "manual", ch["maps"]))
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    ch, led = load_champion(), load_ledger()
    apply_champion(ch)
    faithful, evolved = ensure_mimics(led)
    if not faithful:
        print("no mimics: run `league.py sync` on the lead (or pull data/mimic) first")
        return 1
    maps = list(active_maps(led).values())
    b = benchmark("main", faithful, maps, SCRATCH / "bench")
    ch["bench"][str(ch["version"])] = b
    if evolved:
        ch["bench_hard"][str(ch["version"])] = benchmark("main", evolved, maps, SCRATCH / "bench-hard")
    save_json(CHAMPION, ch)
    print(json.dumps({k: b[k] for k in ("overall", "games", "per_team", "per_map")}, indent=1))
    if evolved:
        print("vs evolved:", json.dumps(ch["bench_hard"][str(ch["version"])]["per_team"]))
    if a.git and git_ready():
        publish(
            [CHAMPION], f"league: benchmark v{ch['version']}", h2h_decider(SCRATCH / "manual", ch["maps"])
        )
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    """Watch the current champion play: vs a mimic (default: the team we do worst against) on a
    map (default: our worst map vs that team), both sides, then open the replays in cambc view."""
    ch, led = load_champion(), load_ledger()
    apply_champion(ch)
    faithful, evolved = ensure_mimics(led)
    b = ch["bench"].get(str(ch["version"])) or {}
    per_team = {t: v for t, v in (b.get("per_team") or {}).items() if f"mimic-{t}" in faithful}
    slug = (a.opponent or "").removeprefix("mimic-").removeprefix("mimx-") or (
        min(per_team, key=per_team.get)
        if per_team
        else (faithful[0].removeprefix("mimic-") if faithful else "")
    )
    opp = f"mimx-{slug}" if a.evolved or (a.opponent or "").startswith("mimx-") else f"mimic-{slug}"
    if opp not in faithful + evolved:
        print(f"no bot {opp}; have: {', '.join(faithful + evolved) or 'none (run sync on the lead first)'}")
        return 1
    maps = active_maps(led)
    grid = {m: v for m, v in ((b.get("grid") or {}).get(slug) or {}).items() if m in maps}
    mp = map_name(a.map) if a.map else (min(grid, key=grid.get) if grid else sorted(maps)[0])
    if mp not in maps:
        print(f"unknown map {mp}; active: {', '.join(sorted(maps))}")
        return 1
    out = SCRATCH / "replay" / f"{time.strftime('%Y%m%d-%H%M%S')}-{opp}-{mp}"
    print(f"champion v{ch['version']} vs {opp} on {mp} (both sides)...")
    replays = []
    for r in runner.run_matches("main", opp, maps=[maps[mp]], out_dir=out, release=True):
        res = "WIN " if r.winner == "main" else "LOSS" if r.winner else "DRAW"
        side = "A" if r.bot_a == "main" else "B"
        ours, theirs = (r.gems_a, r.gems_b) if side == "A" else (r.gems_b, r.gems_a)
        print(f"  {res} as team {side}: gems {ours}-{theirs}, rounds {r.rounds}")
        replays.append(r.replay)
    for rp in replays[1:]:
        print(f"  other side: cambc view {rp}")
    if replays and not a.no_view:
        print(f"opening the viewer: cambc view {replays[0]}  (stop it with Ctrl+C)")
        subprocess.Popen([str(CAMBC), "view", replays[0]], cwd=ROOT)
    return 0


def cmd_review(_: argparse.Namespace) -> int:
    print(write_review(load_champion(), load_ledger()).relative_to(ROOT))
    return 0


def cmd_submit(a: argparse.Namespace) -> int:
    ch, led = load_champion(), load_ledger()
    v = a.version if a.version is not None else ch["version"]
    ok = submit(led, v, "manual (league.py submit)")
    save_json(LEDGER, led)
    if a.git and git_ready():
        publish([LEDGER], f"league: uploaded v{v} (manual)", h2h_decider(SCRATCH / "manual", ch["maps"]))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the training loop")
    r.add_argument("--forever", action="store_true")
    r.add_argument("--cycles", type=int, default=1)
    r.add_argument(
        "--lead", action="store_true", help="this PC reads the ladder, builds mimics, writes reviews"
    )
    r.add_argument(
        "--auto-submit", action="store_true", help="lead only: upload new champions (team-agreed rules)"
    )
    for name in ("run", "sync", "maps", "bench", "submit"):
        sp = r if name == "run" else sub.add_parser(name)
        sp.add_argument("--no-push", dest="git", action="store_false", help="no git pull/commit/push")
    sub.choices["submit"].add_argument("--version", type=int, help="default: the current champion")
    sub.add_parser("status")
    rp = sub.add_parser("replay", help="watch the champion vs a mimic in cambc view")
    rp.add_argument("opponent", nargs="?", help="team slug, e.g. nyx (default: the one we do worst against)")
    rp.add_argument("map", nargs="?", help="map, e.g. lanes (default: our worst map vs that team)")
    rp.add_argument("--evolved", action="store_true", help="vs the evolved mimic (mimx-) instead")
    rp.add_argument("--no-view", action="store_true", help="only print the results")
    sub.add_parser("review")
    a = p.parse_args(argv)
    return {
        "run": cmd_run,
        "status": cmd_status,
        "sync": cmd_sync,
        "maps": cmd_maps,
        "bench": cmd_bench,
        "review": cmd_review,
        "replay": cmd_replay,
        "submit": cmd_submit,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
