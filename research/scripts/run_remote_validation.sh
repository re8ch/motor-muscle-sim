#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/remote

export LLVM_PATH=/opt/rocm/llvm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

PY=.venv-rocm/bin/python
CLI=.venv-rocm/bin/muscle-research

"$CLI" accelerator-doctor --output results/remote/device.json \
  2>&1 | tee results/remote/device.log
"$PY" -m pytest tests \
  2>&1 | tee results/remote/tests.log
"$CLI" parity --output results/remote/consistency.json \
  2>&1 | tee results/remote/consistency.log
"$CLI" rollout --backend mjx --batch-size 64 --steps 100 --motors 20000 \
  2>&1 | tee results/remote/smoke.log
"$CLI" mjx-benchmark --steps 20 --motors 20000 \
  --output results/remote/mjx_benchmark.json \
  2>&1 | tee results/remote/benchmark.log

