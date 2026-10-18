"""Render match frames and timelines to PNG from our bot's debug telemetry.

The engine replay format is not documented, so frames are drawn from what our
bot logs in debug runs: a one-off terrain dump (`M ` line) plus per-turn lines
(`T `) with positions, HP, intent, visible enemies and camps. These images are
what the MCP `render_frame` tool hands to a multimodal model.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from bctools import replay

COLORS = {
    "floor": (236, 233, 224),
    "wall": (70, 70, 78),
    "hole": (25, 25, 30),
    "bush": (120, 170, 100),
    "altar": (218, 165, 32),
    "ally_spawn": (170, 200, 245),
    "enemy_spawn": (245, 180, 180),
    "group": (150, 110, 70),
    "camp_awake": (230, 90, 30),
    "camp_asleep": (240, 200, 60),
    "enemy": (200, 30, 40),
    "text": (20, 20, 20),
}
CHAMP_COLORS = [(30, 90, 200), (20, 150, 120), (140, 60, 180), (0, 130, 200), (60, 60, 160), (200, 120, 0)]


def _bits(hex_str: str, w: int, h: int) -> list[bool]:
    data = bytes.fromhex(hex_str)
    return [bool(data[i >> 3] >> (i & 7) & 1) for i in range(w * h)]


def _terrain(m: dict, scale: int) -> Image.Image:
    w, h = m["w"], m["h"]
    img = Image.new("RGB", (w * scale, h * scale), COLORS["floor"])
    draw = ImageDraw.Draw(img)
    for layer in ("ally_spawn", "enemy_spawn", "bushes", "holes", "walls", "altars"):
        key = {"bushes": "bush", "holes": "hole", "walls": "wall", "altars": "altar"}.get(layer, layer)
        for i, on in enumerate(_bits(m[layer], w, h)):
            if on:
                x, y = i % w, i // w
                # Image y grows down; the game's y grows north.
                py = (h - 1 - y) * scale
                draw.rectangle([x * scale, py, x * scale + scale - 1, py + scale - 1], fill=COLORS[key])
    for gi, g in enumerate(m.get("groups", [])):
        for x, y in g["tiles"]:
            py = (h - 1 - y) * scale
            draw.rectangle([x * scale, py, x * scale + scale - 1, py + scale - 1], outline=COLORS["group"])
        cx, cy = g["c"]
        draw.text((cx * scale + 2, (h - 1 - cy) * scale - 10), str(gi), fill=COLORS["group"])
    return img


def frame(bundle_path: str | Path, round_no: int, out: str | Path, scale: int = 6) -> Path:
    """Draw the state around `round_no` from our side's telemetry."""
    b = replay.load(bundle_path)
    _, log = b.ours()
    if not log.map:
        raise ValueError("no map dump in telemetry (run in debug mode: `cambc run` without --release)")
    m = log.map
    h = m["h"]
    img = _terrain(m, scale)
    side_w = 360
    canvas = Image.new("RGB", (img.width + side_w, max(img.height, 300)), (250, 250, 250))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # Latest line per champion at or before the round.
    latest: dict[str, dict] = {}
    for t in log.turns:
        if t["r"] <= round_no:
            latest[t["c"]] = t
    seen_enemies, seen_camps = set(), {}
    for t in log.turns:
        if round_no - 1 <= t["r"] <= round_no:
            seen_enemies.update(tuple(e) for e in t.get("en", []))
            for c in t.get("cp", []):
                seen_camps[(c[0], c[1])] = c

    def px(x: int, y: int) -> tuple[int, int]:
        return x * scale + scale // 2, (h - 1 - y) * scale + scale // 2

    for (x, y), c in seen_camps.items():
        cx, cy = px(x, y)
        r = scale
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=COLORS["camp_awake" if c[3] else "camp_asleep"],
            outline=(0, 0, 0),
        )
    for x, y, hp in seen_enemies:
        cx, cy = px(x, y)
        r = scale + 1
        draw.line([cx - r, cy - r, cx + r, cy + r], fill=COLORS["enemy"], width=3)
        draw.line([cx - r, cy + r, cx + r, cy - r], fill=COLORS["enemy"], width=3)
        draw.text((cx + r, cy - r), f"{hp}%", fill=COLORS["enemy"])
    groups = m.get("groups", [])
    for i, (name, t) in enumerate(sorted(latest.items())):
        color = CHAMP_COLORS[i % len(CHAMP_COLORS)]
        cx, cy = px(*t["p"])
        if t.get("tg") is not None and t["tg"] < len(groups):
            gx, gy = px(*groups[t["tg"]]["c"])
            draw.line([cx, cy, gx, gy], fill=color, width=1)
        r = scale + 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(255, 255, 255), width=2)
        draw.text((cx - 4, cy - 5), name[:2], fill=(255, 255, 255))
        bar = int(2 * r * max(0.0, min(1.0, t["hp"])))
        draw.rectangle([cx - r, cy + r + 2, cx - r + bar, cy + r + 4], fill=(40, 180, 40))

    x0, y = img.width + 10, 10
    tributed = max((t["at"] for t in latest.values()), default=0)
    draw.text((x0, y), f"round {round_no}   tributed {tributed}/1000", fill=COLORS["text"])
    y += 20
    for name, t in sorted(latest.items()):
        used = t["f0"] - t["f1"]
        draw.text(
            (x0, y), f"{name:<10} hp {t['hp']:.2f} gems {t['gm']:>3} gold {t['g']:>4}", fill=COLORS["text"]
        )
        y += 14
        draw.text((x0 + 10, y), f"{t['m']}/{t['e']} {t['i']} [{t['tier']}] fuel {used}", fill=(90, 90, 90))
        y += 18
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def timeline(bundle_path: str | Path, out: str | Path, width: int = 900, height: int = 360) -> Path:
    """Tributed gems, carried gems and mean fuel per round for our side."""
    b = replay.load(bundle_path)
    _, log = b.ours()
    per_round: dict[int, list[dict]] = defaultdict(list)
    for t in log.turns:
        per_round[t["r"]].append(t)
    rounds = sorted(per_round)
    img = Image.new("RGB", (width, height), (252, 252, 252))
    draw = ImageDraw.Draw(img)
    if not rounds:
        draw.text((10, 10), "no telemetry", fill=COLORS["text"])
        img.save(out)
        return Path(out)
    last = rounds[-1] or 1
    pad = 40
    plot_w, plot_h = width - 2 * pad, height - 2 * pad

    def series(values: list[tuple[int, float]], vmax: float, color: tuple[int, int, int]) -> None:
        pts = [(pad + plot_w * r / last, height - pad - plot_h * min(v, vmax) / vmax) for r, v in values]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)

    tributed = [(r, max(t["at"] for t in per_round[r])) for r in rounds]
    carried = [(r, sum(t["gm"] for t in per_round[r])) for r in rounds]
    fuel = [(r, sum(t["f0"] - t["f1"] for t in per_round[r]) / len(per_round[r])) for r in rounds]
    draw.rectangle([pad, pad, width - pad, height - pad], outline=(180, 180, 180))
    series(tributed, 1000, (30, 90, 200))
    series(carried, 200, (218, 165, 32))
    series(fuel, 100_000, (200, 30, 40))
    deaths = [e for e in log.events if e.get("k") == "dead"]
    for r in sorted({e["r"] for e in deaths}):
        x = pad + plot_w * r / last
        draw.line([x, height - pad, x, height - pad + 6], fill=(0, 0, 0))
    draw.text(
        (pad, 10),
        "blue: tributed (/1000)   gold: carried (/200)   red: mean fuel/turn (/100k)   ticks: deaths",
        fill=COLORS["text"],
    )
    draw.text((pad, height - pad + 8), f"round 0 .. {last}", fill=COLORS["text"])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
