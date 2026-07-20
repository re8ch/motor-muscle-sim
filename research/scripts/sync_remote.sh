#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-amdwsl-zt}"
REMOTE_DIR="${REMOTE_DIR:-/root/projects/motor-muscle-sim}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.tools/' \
  --exclude '.venv/' \
  --exclude 'research/.venv/' \
  --exclude 'research/.venv-rocm/' \
  --exclude 'research/.tools/' \
  --exclude 'research/artifacts/' \
  --exclude 'research/results/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.egg-info/' \
  "$ROOT/" "$REMOTE:$REMOTE_DIR/"

echo "Synced $ROOT -> $REMOTE:$REMOTE_DIR"
