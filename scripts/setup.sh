#!/usr/bin/env bash
# Setup on Linux / WSL / macOS. Idempotent. Then: scripts/train.sh train (a worker; see README).
#   bash scripts/setup.sh          # tooling only
#   bash scripts/setup.sh --gpu    # + .venv-gpu with PyTorch for the GPU simulator (CUDA 12.8+ for RTX 50-series)
set -euo pipefail

if ! command -v rustup >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
  source "$HOME/.cargo/env"
fi
rustup show >/dev/null   # installs the toolchain pinned in rust-toolchain.toml

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install --upgrade "cambc>=3.0.1"
pip install -e ".[dev]"
pre-commit install --hook-type pre-commit --hook-type pre-push

python - <<'PY'
import subprocess
v = subprocess.run(["cambc", "--version"], capture_output=True, text=True).stdout.split()[-1]
assert tuple(map(int, v.split(".")[:3])) >= (3, 0, 1), f"cambc {v}: the ladder engine needs >= 3.0.1"
print("cambc", v)
PY
git ls-remote --exit-code origin league >/dev/null || echo "warning: no access to origin/league (the daemon pushes there)"

if [[ "${1:-}" == "--gpu" ]]; then
  # Its own venv (torch is large); gpusim scripts run with GPU_PY=.venv-gpu/bin/python.
  python3 -m venv .venv-gpu
  .venv-gpu/bin/pip install --upgrade pip
  .venv-gpu/bin/pip install -r gpusim/requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
  .venv-gpu/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
fi

echo "Done. Try: bctools gate main"
