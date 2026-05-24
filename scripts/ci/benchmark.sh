#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${LINSTAB2D_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [ ! -f "${REPO_ROOT}/benchmark.sh" ]; then
    printf '[install_tester] ERROR: missing canonical benchmark.sh at %s\n' "${REPO_ROOT}" >&2
    printf '[install_tester] Run from a full checkout or export LINSTAB2D_REPO_ROOT=/path/to/linstabpy.\n' >&2
    exit 1
fi

exec bash "${REPO_ROOT}/benchmark.sh" "$@"
