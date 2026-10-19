"""Build the policy pack (bots/main/src/pack/pack.bin) and mine its per-map data.

    python harness/pack.py mine-hotspots runs/ runs/ladder_live runs/ladder_v8 --out runs/pack/hotspots.json
    python harness/pack.py write runs/pack/spec.json [--out bots/main/src/pack/pack.bin]

spec.json: {"global": {"theta": [P_N floats]}, "maps": {"<map>": {"theta": [...],
"hotspots": [[x, y, w0..255], ...], "opening": [u8 * 6 * 4]}}}. Map names resolve to
fingerprints through runs/variants/fingerprints.json (world::fingerprint values).
Hotspots: tiles where our champions died most (local league games of our engine and
ladder summaries), 3x3-smoothed, non-maximum-suppressed, weights scaled to 0..255.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINTS = ROOT / "runs" / "variants" / "fingerprints.json"
PACK = ROOT / "bots" / "main" / "src" / "pack" / "pack.bin"
OUR_ENGINE = ("main", "_lg-", "auto-v", "_es-", "_v6-", "_sr-", "_mt-", "_oos-", "_var-", "_tune")
TAG_THETA, TAG_HOTSPOTS, TAG_OPENING, TAG_DRAFT, TAG_BUILD = 1, 2, 3, 4, 5
KINDS, OPENING_LEN, DRAFT_LEN, BUILD_LEN = 12, 4, 5, 12


def _opening_bytes(sides: list[dict]) -> bytes:
    out = bytearray([len(sides)])
    for s in sides:
        out += bytes([s["spawn"][0] & 0xFF, s["spawn"][1] & 0xFF])
        for k in range(KINDS):
            route = (s["kinds"].get(str(k)) or s["kinds"].get(k) or [])[:OPENING_LEN]
            out += bytes(route + [255] * (OPENING_LEN - len(route)))
    return bytes(out)


def _draft_bytes(draft: dict) -> bytes:
    out = bytearray()
    for n in (3, 4, 5):
        picks = (draft.get(str(n)) or draft.get(n) or [])[:DRAFT_LEN]
        out += bytes(picks + [255] * (DRAFT_LEN - len(picks)))
    return bytes(out)


def _build_bytes(build: dict) -> bytes:
    out = bytearray()
    for k in range(KINDS):
        seq = (build.get(str(k)) or build.get(k) or [])[:BUILD_LEN]
        vals = [f * 3 + s for f, s in seq]
        out += bytes(vals + [255] * (BUILD_LEN - len(vals)))
    return bytes(out)
K_HOTSPOTS = 24


def write(spec: dict, out: Path = PACK) -> int:
    fps = json.loads(FINGERPRINTS.read_text(encoding="utf-8"))
    entries = []
    if "global" in spec:
        entries.append((0, spec["global"]))
    for name, e in sorted(spec.get("maps", {}).items()):
        key = name.replace("_", "-")
        if key in fps:
            entries.append((fps[key], e))
    blob = bytearray(b"BCPK" + struct.pack("<HH", 1, len(entries)))
    for fp, e in entries:
        sections = []
        if e.get("theta"):
            sections.append((TAG_THETA, struct.pack(f"<{len(e['theta'])}f", *e["theta"])))
        if e.get("hotspots"):
            sections.append((TAG_HOTSPOTS, bytes(v for h in e["hotspots"] for v in (h[0], h[1], h[2]))))
        if e.get("opening"):
            sections.append((TAG_OPENING, _opening_bytes(e["opening"])))
        if e.get("draft"):
            sections.append((TAG_DRAFT, _draft_bytes(e["draft"])))
        if e.get("build"):
            sections.append((TAG_BUILD, _build_bytes(e["build"])))
        blob += struct.pack("<IH", fp, len(sections))
        for tag, data in sections:
            blob += struct.pack("<BI", tag, len(data)) + data
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(blob))
    return len(blob)


def _deaths_local(stats: Path, results: list[dict]) -> list[tuple[str, int, int, int, int]]:
    """(map, w, h, x, y) for every death of our engine's champions in local games."""
    out = []
    for r in results:
        sides = [s for s, b in (("teamA", r["bot_a"]), ("teamB", r["bot_b"])) if b.startswith(OUR_ENGINE)]
        if not sides:
            continue
        try:
            d = json.loads(Path(r["stats"]).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        w, h = d["match"]["width"], d["match"]["height"]
        for e in d.get("events", []):
            if e["kind"] == "death" and (e.get("owner") or {}).get("team") in sides:
                for en in e["entries"]:
                    out.append((r["map"], w, h, en["x"], en["y"]))
    return out


def mine_hotspots(dirs: list[Path]) -> dict:
    deaths = []
    for d in dirs:
        for res in d.rglob("results.json"):
            try:
                deaths += _deaths_local(res, json.loads(res.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        # Ladder summaries: our team is "Patient Wolves" (match lists sit next to them).
        for f in d.glob("*-*-*-*-*.json"):
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if "events" not in s or "match" not in s:
                continue
            ours = _ladder_side(f.stem, d)
            if not ours:
                continue
            w, h = s["match"]["width"], s["match"]["height"]
            mp = _ladder_map(f.stem, d)
            for e in s["events"]:
                if e["kind"] == "death" and (e.get("owner") or {}).get("team") == ours:
                    for en in e["entries"]:
                        deaths.append((mp, w, h, en["x"], en["y"]))
    grids: dict[str, dict] = {}
    for mp, w, h, x, y in deaths:
        if mp is None:
            continue
        g = grids.setdefault(mp.replace("_", "-"), {"w": w, "h": h, "n": 0, "c": defaultdict(float)})
        g["n"] += 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                g["c"][(x + dx, y + dy)] += 1.0 if dx == dy == 0 else 0.5
    out = {}
    for mp, g in grids.items():
        picks = []
        for (x, y), v in sorted(g["c"].items(), key=lambda kv: -kv[1]):
            if not (0 <= x < g["w"] and 0 <= y < g["h"]):
                continue
            if any(max(abs(x - px), abs(y - py)) <= 2 for px, py, _ in picks):
                continue
            picks.append((x, y, v))
            if len(picks) >= K_HOTSPOTS:
                break
        top = max((v for *_, v in picks), default=1.0)
        out[mp] = {"deaths": g["n"], "hotspots": [[x, y, max(1, round(255 * v / top))] for x, y, v in picks]}
    return out


_LISTS: dict[Path, dict] = {}


def _ladder_index(d: Path) -> dict:
    if d not in _LISTS:
        idx = {}
        for f in d.glob("*.json"):
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            ms = j.get("matches") if isinstance(j, dict) else j if isinstance(j, list) else None
            for m in ms or []:
                if isinstance(m, dict) and "matchId" in m:
                    idx[m["matchId"]] = m
        _LISTS[d] = idx
    return _LISTS[d]


def _ladder_side(match_id: str, d: Path) -> str | None:
    m = _ladder_index(d).get(match_id) or _ladder_index(ROOT / "runs" / "ladder_v8").get(match_id)
    if not m:
        return None
    us = [t for t in m["teams"] if t["name"] == "Patient Wolves"]
    return "team" + us[0]["side"].upper() if us else None


def _ladder_map(match_id: str, d: Path) -> str | None:
    m = _ladder_index(d).get(match_id) or _ladder_index(ROOT / "runs" / "ladder_v8").get(match_id)
    return m["map"] if m else None


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine-hotspots")
    m.add_argument("dirs", nargs="+", type=Path)
    m.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("write")
    w.add_argument("spec", type=Path)
    w.add_argument("--out", type=Path, default=PACK)
    a = p.parse_args(argv)
    if a.cmd == "mine-hotspots":
        hs = mine_hotspots(a.dirs)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(hs, indent=1), encoding="utf-8")
        print({k: v["deaths"] for k, v in hs.items()})
    else:
        print(f"wrote {write(json.loads(a.spec.read_text(encoding='utf-8')), a.out)} bytes to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
