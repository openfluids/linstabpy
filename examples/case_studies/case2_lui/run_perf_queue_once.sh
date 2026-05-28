#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 <config-jsonc> <output-dir> [case-module]" >&2
  exit 2
fi

CONFIG_JSONC="$1"
OUTPUT_DIR="$2"
CASE_MODULE="${3:-case2}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LINSTABPY_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_EXE="$ROOT/.venv/bin/python"
HARNESS="$SCRIPT_DIR/perf_memory.py"
EXIT_FILE="$OUTPUT_DIR/queue_exit_code.txt"

mkdir -p "$OUTPUT_DIR"
rm -f "$EXIT_FILE"

export LINSTABPY_CASE_MODULE="$CASE_MODULE"
export LINSTABPY_CASE2_PERF_CONFIG="$CONFIG_JSONC"

set +e
"$PYTHON_EXE" "$HARNESS"
STATUS=$?
set -e
printf "%s\n" "$STATUS" > "$EXIT_FILE"
exit "$STATUS"
