"""ml/rl/ppo_gpu.py: GAE on the device and PPO learning (needs torch; skipped without it).

The main .venv has no torch; run these with the GPU Python, e.g.
    <ml-gpu-venv python> -m pytest tests/test_rl_gpu.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.rl import ppo, ppo_gpu  # noqa: E402

CPU = [] if torch.cuda.is_available() else ["--cpu"]


def test_gae_matches_numpy():
    rng = np.random.default_rng(3)
    lens = rng.integers(1, 60, 40)
    traj = np.repeat(np.arange(40) * 7, lens)  # ids need not be 0..n-1
    n = len(traj)
    rew = rng.normal(0, 1, n).astype(np.float32)
    val = rng.normal(0, 1, n).astype(np.float32)
    ends = np.cumsum(lens) - 1
    done = np.zeros(n, bool)
    done[ends[rng.random(40) < 0.7]] = True  # some trajectories end unfinished
    want_adv, want_ret = ppo.gae(rew, val, done, traj)
    t = [torch.from_numpy(a) for a in (rew, val, done, traj)]
    adv, ret = ppo_gpu.gae(*t)
    assert np.abs(adv.numpy() - want_adv).max() < 1e-4
    assert np.abs(ret.numpy() - want_ret).max() < 1e-4


def test_ppo_learns_a_contextual_bandit(tmp_path):
    rng = np.random.default_rng(1)
    arm = ppo.ARMS[1]
    base = tmp_path / "rl" / "arm1"
    base.mkdir(parents=True)
    ppo.save(base / "policy-0000.npz", ppo.init_params(arm.n_in, arm.hidden, arm.n_out, rng))
    masks = np.tile([(1 << 9) - 1, 0b1111], (4000, 1)).astype(np.int64)

    def rollout(it):
        policy = ppo.load(base / f"policy-{it:04d}.npz")
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
        np.savez_compressed(
            base / f"it{it:04d}-p0-test.npz",
            X=x,
            A=acts,
            LP=sum(lps).astype(np.float32),
            M=masks,
            R=r,
            DONE=np.ones(n, bool),
            TRAJ=np.arange(n),
        )
        return r.mean()

    before = rollout(0)
    for it in range(12):
        if it:
            rollout(it)
        assert ppo_gpu.main(["--arm", "1", "--it", str(it), "--store", str(tmp_path), *CPU]) == 0
    after = rollout(12)
    assert after > before + 0.25, (before, after)
    assert (base / "policy-0012.bin").exists() and (base / "teacher-0012.pt").exists()


def test_init_arm2_writes_a_network(tmp_path):
    assert ppo_gpu.main(["--arm", "2", "--init", "--store", str(tmp_path), *CPU]) == 0
    assert (tmp_path / "rl" / "arm2" / "policy-0000.bin").exists()
