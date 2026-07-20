#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

test -e /dev/dxg
grep -q '6.4.2' /opt/rocm/.info/version
ROCMINFO_OUTPUT="$(rocminfo 2>/dev/null)"
grep -q 'gfx1101' <<<"$ROCMINFO_OUTPUT"

if [[ ! -x .tools/uv ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/.tools" sh
fi
.tools/uv venv .venv-rocm --python /usr/bin/python3 --clear
.tools/uv pip install --python .venv-rocm/bin/python -r requirements-rocm.txt
.tools/uv pip install --python .venv-rocm/bin/python --no-deps -e .

export LLVM_PATH=/opt/rocm/llvm
.venv-rocm/bin/muscle-research accelerator-doctor
