#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 <config-jsonc> <output-dir> [case-module]" >&2
  exit 2
fi

CONFIG_JSONC="$1"
OUTPUT_DIR="$2"
CASE_MODULE="${3:-case2}"

ROOT="/home/ghamada/linstabpy"
PYTHON_EXE="$ROOT/.venv/bin/python"
HARNESS="$ROOT/studies/case2/perf_memory.py"
EXIT_FILE="$OUTPUT_DIR/queue_exit_code.txt"

mkdir -p "$OUTPUT_DIR"
rm -f "$EXIT_FILE"

export LINSTAB2D_CASE_MODULE="$CASE_MODULE"
export LINSTAB2D_CASE2_PERF_CONFIG="$CONFIG_JSONC"

set +e
"$PYTHON_EXE" "$HARNESS"
STATUS=$?
set -e
printf "%s\n" "$STATUS" > "$EXIT_FILE"
exit "$STATUS"
