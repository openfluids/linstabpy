#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${LINSTAB2D_PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[song_2023.sh] missing Python runtime: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${SONG2023_CONFIG:-}" ]]; then
  export LINSTAB2D_SONG2023_CONFIG="$SONG2023_CONFIG"
fi

exec "$PYTHON_BIN" -m validation.song_2023.song_2023
