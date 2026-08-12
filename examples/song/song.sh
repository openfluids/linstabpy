#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${LINSTABPY_PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[song.sh] missing Python runtime: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${SONG2023_CONFIG:-}" ]]; then
  exec "$PYTHON_BIN" -m examples.song.song --config "$SONG2023_CONFIG"
fi

exec "$PYTHON_BIN" -m examples.song.song
