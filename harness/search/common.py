"""Shared layer of the offline plan search: settings records, the patchable slot pool, the
checkpointed game store and batch play.

Candidates are byte-patched into prebuilt copies of one bot ("slots", bots/_ps<k>), so every
candidate runs the same wasm and games differ only through the candidate's settings (no fuel
chaos from rebuilds). The block is bots/main/src/settings.rs SEARCH_CFG.

Every finished game is one appended row in <job>/games.jsonl, keyed by
(snapshot, candidate, map, opponent, side). Games are deterministic, so an interrupted batch
replays identically and finished rows are never played again.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from bctools import replay, runner  # noqa: E402

#: The main checkout: mimic bots and the live pack.bin come from there (they're gitignored).
MAIN = Path(os.environ.get("BC_MAIN", r"D:\Code\battlecode"))
RUNS = ROOT / "runs" / "search"
MAPS_DIR = ROOT / "maps"

MAGIC = bytes([0x50, 0x53, 0x43, 0x46, 0x47, 0x01, 0xA5, 0x5A])
BLOCK = 128
N_SLOTS = 16
SLOT = "_ps{}"
#: Our side's fitness, as league.fitness: win + 0.25 gem margin - 0.01 deaths.
MARGIN_W, DEATH_W = 0.25, 0.01
#: Games per batch: about 2-3 minutes at 14 jobs, so a stop loses little.
BATCH_GAMES = 600


# ---------------------------------------------------------------- settings records


def record(draft: int = 0, overrides: dict[int, int] | None = None) -> bytes:
    """settings.rs record: version 1, draft profile, count, reserved, (slot, value) pairs.
    Slots 0..32 are knobs, 32..64 knob2. No draft and no overrides = the empty record."""
    ov = sorted((overrides or {}).items())
    if not draft and not ov:
        return b""
    if len(ov) > (BLOCK - len(MAGIC) - 4) // 2:
        raise ValueError(f"too many overrides: {len(ov)}")
    return bytes([1, draft, len(ov), 0]) + b"".join(bytes([s, v]) for s, v in ov)


def block(rec: bytes) -> bytes:
    return MAGIC + rec.ljust(BLOCK - len(MAGIC), b"\0")


def knob_slots() -> dict[str, int]:
    """Knob name -> settings slot, parsed from the vendored memory.rs (knob2 slots + 32)."""
    src = (ROOT / "bots" / "main" / "src" / "bc_core" / "memory.rs").read_text(encoding="utf-8")
    out: dict[str, int] = {}
    for mod, base in (("knob", 0), ("knob2", 32)):
        start = src.index(f"pub mod {mod} {{")
        end = src.index("\n}", start)
        for line in src[start:end].splitlines():
            line = line.strip()
            if line.startswith("pub const ") and ": usize = " in line:
                name, val = line[len("pub const ") :].split(": usize = ")
                out[name if base == 0 else f"K2_{name}"] = base + int(val.rstrip(";"))
    return out


def defaults() -> dict[int, int]:
    """Compiled default value per settings slot (knob::DEFAULTS, knob2::DEFAULTS)."""
    src = (ROOT / "bots" / "main" / "src" / "bc_core" / "memory.rs").read_text(encoding="utf-8")
    slots = knob_slots()
    out = {s: 0 for s in slots.values()}
    for mod, prefix in (("knob", ""), ("knob2", "K2_")):
        start = src.index(f"pub mod {mod} {{")
        end = src.index("\n}", start)
        body = src[start:end]
        d = body.index("pub const DEFAULTS")
        for line in body[d:].splitlines()[1:]:
            line = line.strip()
            if line.startswith("]"):
                break
            if line.startswith("("):
                name, val = line.strip("(),").split(",")
                out[slots[prefix + name.strip()]] = int(val.strip())
    return out


# ---------------------------------------------------------------- slots


def slot_dir(k: int) -> Path:
    return ROOT / "bots" / SLOT.format(k)


def wasm_files(bot: Path) -> list[Path]:
    return sorted((bot / ".cambc").rglob("*.wasm"))


def injected(bot: Path) -> Path:
    return bot / ".cambc" / "build-release" / "wasm32-wasip1" / "release" / "bot-injected.wasm"


def patch(bot: Path, rec: bytes) -> None:
    blk, n = block(rec), 0
    for f in wasm_files(bot):
        b = bytearray(f.read_bytes())
        i = b.find(MAGIC)
        if i < 0:
            continue
        if b.find(MAGIC, i + 1) >= 0:
            raise RuntimeError(f"{f}: two settings blocks")
        if b[i : i + BLOCK] != blk:
            b[i : i + BLOCK] = blk
            f.write_bytes(b)
        n += 1
    if n == 0:
        raise RuntimeError(f"{bot}: settings block not found (built without settings.rs?)")


def wasm_sha(bot: Path) -> str:
    """sha1 of the release wasm's code and data with the settings payload zeroed: the same for
    every slot. Custom sections are skipped: the `name` section holds the crate hash, which
    depends on the bot's directory, and doesn't affect execution or fuel."""
    b = bytearray(injected(bot).read_bytes())
    i = b.find(MAGIC)
    if i >= 0:
        b[i : i + BLOCK] = block(b"")
    h, at = hashlib.sha1(b[:8]), 8
    while at < len(b):
        sid, size, n, shift = b[at], 0, at + 1, 0
        while True:
            byte = b[n]
            size |= (byte & 0x7F) << shift
            n += 1
            shift += 7
            if byte < 0x80:
                break
        if sid != 0:
            h.update(b[at : n + size])
        at = n + size
    return h.hexdigest()


def _copy_source(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".gitignore"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    for name in ("src", ".cargo"):
        if (dst / name).exists():
            shutil.rmtree(dst / name)
        shutil.copytree(src / name, dst / name)


def build_slots(n: int = N_SLOTS, pack: Path | None = None) -> dict:
    """Build slot 0 from this worktree's bots/main (with the live pack.bin), copy it to the other
    slots, warm every slot up with a 1-round game (the first run after a build rewrites the
    wasm, which would undo a patch), and check every slot has the same wasm."""
    pack = pack or MAIN / "bots" / "main" / "src" / "pack" / "pack.bin"
    s0 = slot_dir(0)
    _copy_source(ROOT / "bots" / "main", s0)
    shutil.copy2(pack, s0 / "src" / "pack" / "pack.bin")
    runner.build(str(s0), release=True)
    for k in range(1, n):
        d = slot_dir(k)
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s0, d)
    slots = [slot_dir(k) for k in range(n)]
    warm_up(slots)
    # Each copy rebuilt itself (Cargo's crate hash includes the directory, which reorders some
    # functions), so give every slot slot 0's wasm files, then warm up again to show nothing is
    # rebuilt: every slot then runs byte-identical code.
    for d in slots[1:]:
        for f in wasm_files(s0):
            g = d / f.relative_to(s0)
            if g.exists():
                shutil.copy2(f, g)
    warm_up(slots)
    full = {hashlib.sha1(injected(d).read_bytes()).hexdigest() for d in slots}
    shas = {wasm_sha(d) for d in slots}
    if len(full) != 1 or len(shas) != 1:
        raise RuntimeError(f"slots differ after the copy: {full}")
    snap = {
        "wasm": shas.pop(),
        "slots": n,
        "pack_sha1": hashlib.sha1(pack.read_bytes()).hexdigest(),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain", "bots/main") != "",
        "cambc": _cambc_version(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_json(RUNS / "snapshot.json", snap)
    return snap


def warm_up(bots: list[Path]) -> None:
    m = sorted(MAPS_DIR.glob("*.map26a"))[0]
    opp = str(MAIN / "bots" / "baseline")
    out = RUNS / "tmp" / "warmup"
    runner.run_games([(str(b), opp, m) for b in bots], out, rounds=1, release=True, log_limit=0)
    shutil.rmtree(out, ignore_errors=True)


def check_snapshot() -> dict:
    """The snapshot, after checking every slot still has its wasm (else rebuild with setup)."""
    snap = load_json(RUNS / "snapshot.json", None)
    if not snap:
        raise SystemExit("no snapshot: run `python -m harness.search setup` first")
    for k in range(snap["slots"]):
        if not injected(slot_dir(k)).exists() or wasm_sha(slot_dir(k)) != snap["wasm"]:
            raise SystemExit(f"slot {k} differs from the snapshot: run setup again")
    return snap


def _git(*args: str) -> str:
    import subprocess

    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _cambc_version() -> str:
    import subprocess

    p = subprocess.run([str(runner.CAMBC), "--version"], capture_output=True, text=True)
    return p.stdout.strip()


# ---------------------------------------------------------------- files


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, obj) -> None:
    """Atomic: write a temp file, fsync, rename (a power cut leaves the old or the new file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- opponents and maps


def maps() -> dict[str, Path]:
    """Map name (without the 00/01 prefix) -> file."""
    return {p.stem[2:]: p for p in sorted(MAPS_DIR.glob("*.map26a"))}


def opponents() -> dict[str, str]:
    """Short name -> bot path: every built mimic and exploiter in the main checkout, the
    baseline, the live ladder version and our rule bot as it is on league ("rules")."""
    out = {}
    for d in sorted((MAIN / "bots").iterdir()):
        n = d.name
        if not (n.startswith(("mimic-", "xpl-")) or n in ("baseline", "lg-v23-fuel")):
            continue
        if injected(d).exists():
            out[n] = str(d)
    rules = ROOT / "bots" / "_rules"
    if injected(rules).exists():
        out["rules"] = str(rules)
    out.update(fresh_top())
    return out


def fresh_top() -> dict[str, str]:
    """The ladder top 10 (+ recent losses) refreshed into this worktree (league sync +
    ensure_mimics, 2026-10-25): "top-<slug>", used only as the final MVP test, never to choose."""
    led = load_json(ROOT / "league" / "ledger.json", {})
    out = {}
    for p in led.get("pool", []):
        d = ROOT / "bots" / f"mimic-{p['slug']}"
        if injected(d).exists():
            out[f"top-{p['slug']}"] = str(d)
    return out


def build_rules() -> None:
    """Our rule bot as it is on league today (the main checkout's bots/main), as an opponent."""
    d = ROOT / "bots" / "_rules"
    _copy_source(MAIN / "bots" / "main", d)
    runner.build(str(d), release=True)


# ---------------------------------------------------------------- the game store


@dataclass(frozen=True)
class Game:
    cand: str
    map: str
    opp: str
    side: str  # "a": we are team A

    def key(self, snap: str) -> str:
        return f"{snap}|{self.cand}|{self.map}|{self.opp}|{self.side}"


class Store:
    """Append-only rows of finished games (one JSON per line), plus candidate records."""

    def __init__(self, job: str, snap: str):
        self.dir = RUNS / job
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snap = snap
        self.path = self.dir / "games.jsonl"
        self.rows: dict[str, dict] = {}
        #: Rows per candidate: select() and paired() are called per candidate in every ranking
        #: (stage A sorts arms by score), so a full scan each time was O(rows) per call.
        self.by_cand: dict[str, list[dict]] = {}
        if self.path.exists():
            # Streamed: games.jsonl reaches ~190 MB, and read_text().splitlines() held the file
            # twice over (text + line list) on top of the parsed rows.
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a line cut by a power loss
                    if r.get("snap") == snap:
                        self._put(r)
        self.cands: dict[str, str] = load_json(self.dir / "cands.json", {})

    def _put(self, r: dict) -> None:
        key = Game(r["cand"], r["map"], r["opp"], r["side"]).key(self.snap)
        old = self.rows.get(key)
        if old is not None:
            self.by_cand[r["cand"]].remove(old)
        self.rows[key] = r
        self.by_cand.setdefault(r["cand"], []).append(r)

    def has(self, g: Game) -> bool:
        return g.key(self.snap) in self.rows

    def get(self, g: Game) -> dict | None:
        return self.rows.get(g.key(self.snap))

    def register(self, cand: str, rec: bytes) -> None:
        old = self.cands.get(cand)
        if old is not None and old != rec.hex():
            raise ValueError(f"candidate {cand} already registered with a different record")
        if old is None:
            self.cands[cand] = rec.hex()
            save_json(self.dir / "cands.json", self.cands)

    def add(self, rows: list[dict]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
                self._put(r)
            f.flush()
            os.fsync(f.fileno())

    def select(self, cand: str | None = None, **eq) -> list[dict]:
        rows = self.rows.values() if cand is None else self.by_cand.get(cand, [])
        return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def stalls(r: dict) -> int:
    """Out-of-fuel turns after start-up (rows without oof_init: start-up included)."""
    return max(r.get("oof", 0), 0) - r.get("oof_init", 0)


def fitness(r: dict) -> float:
    margin = (r["ours"] - r["theirs"]) / max(r["ours"] + r["theirs"], 1)
    return r["res"] + MARGIN_W * margin - DEATH_W * r["deaths"]


# ---------------------------------------------------------------- playing


def _row(res: runner.MatchResult, g: Game, snap: str, us: str) -> dict:
    a = res.bot_a == us
    win = 0.5 if res.winner is None else (1.0 if res.winner == us else 0.0)
    row = {
        "snap": snap,
        "cand": g.cand,
        "map": g.map,
        "opp": g.opp,
        "side": g.side,
        "res": win,
        "ours": res.gems_a if a else res.gems_b,
        "theirs": res.gems_b if a else res.gems_a,
        "deaths": res.deaths_a if a else res.deaths_b,
        "rounds": res.rounds,
    }
    team = "teamA" if a else "teamB"
    try:
        s = replay.load_stats(res.stats)
        draft = s.get("roster", {}).get("draft", [])
        row["picks"] = [d["champion"] for d in draft if d.get("team") == team]
    except (OSError, ValueError, KeyError):
        row["picks"] = []
    try:
        log = replay.load_fuel(res.replay, "a" if a else "b")
        k = len(row["picks"])
        champ = log.fuel[k:]
        row["draft_fuel"] = log.fuel[:k]
        row["oof"] = replay.exhausted_turns(log)
        # Each body's first turn runs start-up (Bot::new: ~200k on maps with many camps). Losing
        # that turn at spawn costs nothing (league main skips it on purpose when start-up leaves
        # under 70k), so it is counted apart from real stalls.
        bodies = len(row["picks"]) + ("Cassi & Issac" in row["picks"])
        row["oof_init"] = sum(1 for f in champ[:bodies] if f in replay.EXHAUSTED)
        if champ:
            srt = sorted(champ)
            row["fmean"] = round(sum(champ) / len(champ))
            row["fp95"] = srt[int(len(srt) * 0.95)]
            row["fmax"] = srt[-1]
    except (OSError, KeyError, ValueError):
        row["oof"] = -1
    return row


def play(
    games: list[Game],
    recs: dict[str, bytes],
    store: Store,
    opps: dict[str, str],
    deadline: float | None = None,
    log=print,
) -> bool:
    """Play every game not yet in the store, in batches of at most N_SLOTS candidates and
    BATCH_GAMES games. Returns False if it stopped at the deadline with games left."""
    snap = store.snap
    todo = [g for g in dict.fromkeys(games) if not store.has(g)]
    for c in {g.cand for g in todo}:
        store.register(c, recs[c])
    maps_ = maps()
    while todo:
        if deadline and time.time() > deadline:
            return False
        cands: list[str] = []
        batch: list[Game] = []
        for g in todo:
            if g.cand not in cands:
                if len(cands) >= N_SLOTS:
                    continue
                cands.append(g.cand)
            batch.append(g)
            if len(batch) >= BATCH_GAMES:
                break
        slot_of = {c: k for k, c in enumerate(cands)}
        for c, k in slot_of.items():
            patch(slot_dir(k), recs[c])
        triples, meta = [], []
        for g in batch:
            us = str(slot_dir(slot_of[g.cand]))
            them = opps[g.opp]
            a, b = (us, them) if g.side == "a" else (them, us)
            triples.append((a, b, maps_[g.map]))
            meta.append((g, us))
        out = RUNS / "tmp" / f"b{int(time.time() * 1000)}"
        t0 = time.time()
        results = runner.run_games(triples, out, release=True, log_limit=0)
        rows = [_row(r, g, snap, us) for r, (g, us) in zip(results, meta)]
        store.add(rows)
        shutil.rmtree(out, ignore_errors=True)
        done = set(batch)
        todo = [g for g in todo if g not in done]
        log(f"  batch {len(batch)} games, {len(cands)} cands, {time.time() - t0:.0f}s, {len(todo)} left")
    return True


def paired(store: Store, cand: str, base: str, **eq) -> dict:
    """Mean and SE of the per-game fitness and win difference cand - base over the same
    (map, opponent, side)."""
    b = {(r["map"], r["opp"], r["side"]): r for r in store.select(base, **eq)}
    df, dw = [], []
    for r in store.select(cand, **eq):
        o = b.get((r["map"], r["opp"], r["side"]))
        if o is not None:
            df.append(fitness(r) - fitness(o))
            dw.append(r["res"] - o["res"])
    return {"n": len(df), **_mean_se(df, "fit"), **_mean_se(dw, "win")}


def _mean_se(x: list[float], name: str) -> dict:
    n = len(x)
    m = sum(x) / n if n else 0.0
    v = sum((y - m) ** 2 for y in x) / (n - 1) if n > 1 else 0.0
    return {name: round(m, 4), f"{name}_se": round((v / n) ** 0.5 if n else 0.0, 4)}


def deadline_from(hours: float | None) -> float | None:
    return time.time() + hours * 3600 if hours else None

