#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${LINSTABPY_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [ ! -f "${REPO_ROOT}/diagnose.sh" ]; then
    printf '[install_tester] ERROR: missing canonical diagnose.sh at %s\n' "${REPO_ROOT}" >&2
    printf '[install_tester] Run from a full checkout or export LINSTABPY_REPO_ROOT=/path/to/linstabpy.\n' >&2
    exit 1
fi

exec bash "${REPO_ROOT}/diagnose.sh" "$@"
