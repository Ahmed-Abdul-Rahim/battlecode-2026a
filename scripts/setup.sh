#!/usr/bin/env bash
# Setup on Linux / WSL (e.g. the GPU training box). Idempotent.
#   bash scripts/setup.sh          # tooling only
#   bash scripts/setup.sh --ml     # + PyTorch (CUDA 12.8+, needed for RTX 50-series)
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

if [[ "${1:-}" == "--ml" ]]; then
  pip install -r ml/requirements-ml.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
fi

echo "Done. Try: bctools gate main"
