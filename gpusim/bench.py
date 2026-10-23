"""Throughput of the GPU engine with random actions.

    python -m gpusim.bench [map] [rounds]
"""

from __future__ import annotations

import sys
import time

import torch

from gpusim import maps
from gpusim.engine import ACT_ATTACK, ACT_MOVE, ACT_SPECIAL, ACT_SPELL, Engine

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "00ambush"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    md = maps.load(name)
    kinds = [[0, 1, 3, 5, 10], [11, 10, 6, 4, 9]]
    print(f"{name} {md.w}x{md.h}, {sum(len(g.camps) for g in md.groups)} camps, GPU {torch.cuda.get_device_name(0)}")
    for B in (256, 1024, 4096, 16384):
        e = Engine(md, B, kinds, "cuda")
        g = torch.Generator(device="cuda").manual_seed(0)
        hi = torch.tensor([ACT_MOVE, ACT_ATTACK, ACT_SPELL, ACT_SPECIAL], device="cuda")
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(rounds):
            a = (torch.rand(B, 2, e.T, 4, generator=g, device="cuda") * hi).long()
            e.step(a)
        torch.cuda.synchronize()
        dt = time.time() - t
        per_game_round = dt / (B * rounds)
        games_per_hour = 3600 / (per_game_round * 2000)
        print(f"  B={B:6d}: {rounds} rounds in {dt:6.1f}s -> {B * rounds / dt:9.0f} game-rounds/s = "
              f"{games_per_hour:9.0f} full games/hour  (mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)", flush=True)
        del e
        torch.cuda.empty_cache()
