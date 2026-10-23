"""Decoder for official replays (.repl26a inside a cambc bundle), black-box from our own files.

Layout: "a62per" + 8 header bytes, the embedded .map26a file, a u32 (final input count), then
records: a u16 tag (two bytes as written) and a fixed-length payload. Every champion turn ends
with END_TURN; the draft comes first. Verified on a full game against its statistics file:
end-turns = rounds x champions, tribute starts = gem submissions, buys = buyItem events,
consumable buys = buyConsumable events.

Turn order (docs/GameRules.md, confirmed on replays): champions of the two teams interleave one
by one (A1 B1 A2 B2 ...; the bigger team finishes the round), team A starts round 0 and the
starting team swaps every round; within a team higher priority first, then draft order;
set_priority takes effect from the next round.
"""

from __future__ import annotations

import json
import struct
from itertools import zip_longest
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# tag -> (name, payload bytes)
TAGS = {
    (4, 1): ("draft_pick", 5),  # u32 champion (cambc enum), u8
    (0, 0): ("set_priority", 4),  # u32
    (0, 1): ("move", 8),  # u32 x, u32 y
    (0, 2): ("attack", 9),  # u8 target kind (1 champion, 2 camp), u32 target id, 4 bytes
    (0, 3): ("cast", 17),  # u8 slot, 3 bytes, u8 target kind (0 ground, 1 champion, 2 camp),
    #                         u32 target id, i32 x, i32 y (-1 for unit targets)
    (0, 4): ("recall", 0),
    (0, 5): ("tribute", 8),  # u32 x, u32 y (the champion's tile)
    (0, 6): ("buy", 8),  # u32 item, u32 count
    (0, 7): ("cancel", 0),
    (0, 8): ("buy_consumable", 8),  # u32 consumable, u32 count
    (0, 9): ("use_consumable", 9),  # u8 consumable, u32 x, u32 y
    (3, 0): ("end_turn", 0),
}
# cambc champion enum (draft picks) -> names in the statistics roster
CHAMPION_ENUM = {1: "Vesper Vale", 3: "Cassi & Issac", 4: "Stellar", 5: "Prism", 6: "Marcella", 8: "Ren"}


@dataclass
class Record:
    name: str
    payload: bytes

    def u32(self, i: int) -> int:
        return struct.unpack_from("<I", self.payload, i)[0]


@dataclass
class Turn:
    round: int
    champion: int  # roster id
    records: list[Record] = field(default_factory=list)


def records(bundle: Path | str, map_bytes: bytes) -> list[Record]:
    r = zipfile.ZipFile(bundle).read("replay.repl26a")
    if r[:6] != b"a62per":
        raise ValueError("not a replay")
    at = r.find(map_bytes[:16])
    if at < 0 or r[at : at + len(map_bytes)] != map_bytes:
        raise ValueError("map not embedded where expected")
    at += len(map_bytes) + 4
    out = []
    while at < len(r):
        t = (r[at], r[at + 1])
        if t not in TAGS:
            raise ValueError(f"unknown record {t} at {at}")
        name, n = TAGS[t]
        out.append(Record(name, r[at + 2 : at + 2 + n]))
        at += 2 + n
    return out


def turns(bundle: Path | str, stats: Path | str, map_bytes: bytes) -> list[Turn]:
    """Records grouped into champion turns, attributed with the official turn order."""
    s = json.loads(Path(stats).read_text(encoding="utf-8-sig"))
    roster = s["roster"]["champions"]
    team = {c["id"]: 0 if c["team"] == "teamA" else 1 for c in roster}
    pick = {c["id"]: c["draftPick"] for c in roster}
    prio = {c["id"]: 0 for c in roster}
    pending: dict[int, int] = {}
    recs = [x for x in records(bundle, map_bytes) if x.name != "draft_pick"]
    # the draft's own turns end with end_turn records too: skip one per pick
    n_picks = len(s["roster"]["draft"])
    out: list[Turn] = []
    i, skipped = 0, 0
    while skipped < n_picks and i < len(recs):
        if recs[i].name == "end_turn":
            skipped += 1
        i += 1
    rnd = 0
    while i < len(recs):
        prio.update(pending)
        pending = {}
        first = rnd % 2
        lists = []
        for t in (first, 1 - first):
            ids = [cid for cid in team if team[cid] == t]
            ids.sort(key=lambda cid: (-prio[cid], pick[cid], cid))
            lists.append(ids)
        # champions of the two teams interleave one by one; the bigger team finishes the round
        order = [cid for pair in zip_longest(*lists) for cid in pair if cid is not None]
        for cid in order:
            turn = Turn(rnd, cid)
            while i < len(recs) and recs[i].name != "end_turn":
                rec = recs[i]
                if rec.name == "set_priority":
                    pending[cid] = rec.u32(0)
                turn.records.append(rec)
                i += 1
            i += 1  # end_turn
            out.append(turn)
            if i >= len(recs):
                break
        rnd += 1
    return out


def check_moves(ts: list[Turn]) -> tuple[int, int]:
    """(moves adjacent to the champion's previous move target, moves checked)."""
    last: dict[int, tuple[int, int]] = {}
    ok = n = 0
    for t in ts:
        for rec in t.records:
            if rec.name != "move":
                continue
            p = (rec.u32(0), rec.u32(4))
            if t.champion in last:
                n += 1
                ok += max(abs(p[0] - last[t.champion][0]), abs(p[1] - last[t.champion][1])) <= 2
            last[t.champion] = p
    return ok, n
