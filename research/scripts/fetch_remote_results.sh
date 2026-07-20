#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-amdwsl-zt}"
REMOTE_DIR="${REMOTE_DIR:-/root/projects/motor-muscle-sim}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_DIR="$ROOT/research/results/remote"

mkdir -p "$LOCAL_DIR"
rsync -az "$REMOTE:$REMOTE_DIR/research/results/remote/" "$LOCAL_DIR/"
echo "Fetched results into $LOCAL_DIR"

