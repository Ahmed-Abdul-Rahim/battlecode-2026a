"""ml/rl: reward parsing, PPO learning, int8 export matching the bot's probabilities."""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.distill import quant
from ml.rl import ppo, reward


def test_export_keeps_probabilities():
    rng = np.random.default_rng(0)
    arm = ppo.ARMS[1]
    p = ppo.init_params(arm.n_in, arm.hidden, arm.n_out, rng, out_std=0.3)
    x = rng.integers(-127, 128, (400, arm.n_in)).astype(np.int8)
    schema, layers = quant.decode(ppo.export(p, x, arm))
    assert schema == arm.schema
    k = layers[-1].shift
    _, _, want = ppo.mlp(p, x.astype(np.float32) / 127.0)
    got = np.stack([quant.forward(layers, row) for row in x]).astype(np.float64) / 2.0**k
    e = slice(0, 9)
    agree = (want[:, e].argmax(1) == got[:, e].argmax(1)).mean()
    assert agree > 0.9, agree
    assert np.abs(want - got).mean() < 0.1 * np.abs(want).mean() + 0.05


def test_ppo_learns_a_contextual_bandit():
    rng = np.random.default_rng(1)
    arm = ppo.ARMS[1]
    policy = ppo.init_params(arm.n_in, arm.hidden, arm.n_out, rng)
    value = ppo.init_params(arm.n_in, 64, 1, rng)
    masks = np.tile([(1 << 9) - 1, 0b1111], (4000, 1)).astype(np.int64)

    def rollout():
        x = rng.integers(-127, 128, (4000, arm.n_in)).astype(np.int8)
        _, _, logits = ppo.mlp(policy, x.astype(np.float32) / 127.0)
        acts, lps = [], []
        for lsm, pr, _ in ppo.head_stats(logits, masks, arm):
            a = np.array([rng.choice(len(r), p=r / r.sum()) for r in pr])
            acts.append(a)
            lps.append(lsm[np.arange(len(a)), a])
        acts = np.stack(acts, axis=1)
        good = np.where(x[:, 0] > 0, 2, 5)  # the best expert depends on the input
        r = np.where(acts[:, 0] == good, 1.0, 0.0).astype(np.float32)
        n = len(x)
        data = {
            "X": x,
            "A": acts,
            "LP": sum(lps).astype(np.float32),
            "M": masks,
            "R": r,
            "DONE": np.ones(n, bool),
            "TRAJ": np.arange(n),
        }
        return data, r.mean()

    _, before = rollout()
    for _ in range(12):
        data, _ = rollout()
        ppo.update(policy, value, data, arm, rng)
    _, after = rollout()
    assert after > before + 0.25, (before, after)


def test_gae_single_trajectory():
    _adv, ret = ppo.gae(
        np.array([0.0, 1.0], np.float32),
        np.zeros(2, np.float32),
        np.array([False, True]),
        np.zeros(2, np.int64),
    )
    assert ret[1] == 1.0 and abs(ret[0] - ppo.GAMMA * ppo.LAMBDA) < 1e-6


def test_reward_from_a_logged_game(tmp_path):
    x = "00" * 109
    lines = [
        'T {"r":1,"c":"VesperVale","lg":0,"at":0,"i":"ClearCamp(2)","f0":100000,"f1":50000}',
        'R {"r":1,"c":"VesperVale","arm":1,"x":"' + x + '","a":[3,0],"lp":-1500,"m":[511,1]}',
        'E {"r":2,"c":"VesperVale","k":"o","v":0}',
        'T {"r":2,"c":"VesperVale","lg":100,"at":10,"i":"ClearCamp(2)","f0":100000,"f1":90000}',
        'R {"r":3,"c":"VesperVale","arm":1,"x":"' + x + '","a":[0,0],"lp":-200,"m":[511,1]}',
        'T {"r":3,"c":"VesperVale","lg":100,"at":10,"i":"Tribute","f0":100000,"f1":90000}',
    ]
    bundle = tmp_path / "g.repl26a"
    with zipfile.ZipFile(bundle, "w") as z:
        z.writestr("team-a.stdout", "\n".join(lines))
    stats = tmp_path / "g.json"
    stats.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "kind": "kill",
                        "owner": {"kind": "champion", "name": "Vesper Vale", "team": "teamA"},
                        "entries": [{"timestamp": {"round": 2}}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    s = reward.game_samples(bundle, stats, "a", arm=1, won=1.0, margin=0.5)
    w = reward.WEIGHTS
    first = w["gold"] * 100 + w["deliver"] * 10 + w["kill"] - w["waste"] - w["fuel"] * 0.2  # rounds 2 and 3
    assert len(s["R"]) == 2 and abs(s["R"][0] - first) < 1e-4
    assert s["DONE"].tolist() == [False, True]
    assert abs(s["R"][1] - (w["win"] + w["margin"] * 0.5)) < 1e-4
    assert abs(s["LP"][0] + 1.5) < 1e-6 and s["A"].tolist() == [[3, 0], [0, 0]]


def test_thinning_keeps_total_reward(tmp_path):
    x = "00" * 109
    lines = []
    for r in range(1, 9):
        lines.append(
            json.dumps({"r": r, "c": "Ryth", "lg": 10 * r, "at": 0, "i": "Hold", "f0": 0, "f1": 0}).join(
                ["T ", ""]
            )
        )
        lines.append(
            json.dumps({"r": r, "c": "Ryth", "arm": 1, "x": x, "a": [0, 0], "lp": -100, "m": [511, 1]}).join(
                ["R ", ""]
            )
        )
    bundle = tmp_path / "g.repl26a"
    with zipfile.ZipFile(bundle, "w") as z:
        z.writestr("team-a.stdout", "\n".join(lines))
    stats = tmp_path / "g.json"
    stats.write_text('{"events": []}', encoding="utf-8")
    full = reward.game_samples(bundle, stats, "a", 1, 1.0, 0.0)
    thin = reward.game_samples(bundle, stats, "a", 1, 1.0, 0.0, every=4)
    assert len(full["R"]) == 8 and len(thin["R"]) == 3  # decisions 0, 4 and the last
    assert abs(full["R"].sum() - thin["R"].sum()) < 1e-4


def test_broken_shards_are_skipped(tmp_path):
    good = tmp_path / "a.npz"
    np.savez_compressed(good, X=np.zeros((3, 109), np.int8), y=np.zeros(3, int))
    (tmp_path / "b.npz").write_bytes(b"PK truncated")
    np.savez_compressed(tmp_path / "c.npz", X=np.zeros((2, 109), np.int8))  # no y
    got = [reward.read_shard(p, ("X", "y")) for p in sorted(tmp_path.glob("*.npz"))]
    assert got[0] is not None and len(got[0]["y"]) == 3 and got[1] is None and got[2] is None
