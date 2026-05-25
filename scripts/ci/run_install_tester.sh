#!/usr/bin/env bash
# This script tests the INSTALLED linstabpy package. It does not bundle a copy
# of the library sources; all test targets run against the pip-installed package.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${LINSTABPY_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_LABEL="${1:-$(hostname 2>/dev/null || echo unknown-host)}"
RUN_ID="$(date -u +"%Y%m%dT%H%M%SZ")_${RUN_LABEL}"

export LINSTABPY_INSTALL_RUN_ID="${RUN_ID}"
export LINSTABPY_INSTALL_LOG_DIR="${SCRIPT_DIR}/install_runs"
export LINSTABPY_REPO_ROOT="${REPO_ROOT}"

mkdir -p "${LINSTABPY_INSTALL_LOG_DIR}/${RUN_ID}"
exec > >(tee -a "${LINSTABPY_INSTALL_LOG_DIR}/${RUN_ID}/install_tester.log") 2>&1

echo "[install_tester] run_id=${RUN_ID}"
echo "[install_tester] logs_dir=${LINSTABPY_INSTALL_LOG_DIR}/${RUN_ID}"
echo "[install_tester] repo_root=${REPO_ROOT}"

if [ ! -f "${REPO_ROOT}/pyproject.toml" ] || [ ! -d "${REPO_ROOT}/src/linstabpy" ] || [ ! -d "${REPO_ROOT}/examples" ]; then
    echo "[install_tester] ERROR: repository checkout incomplete at ${REPO_ROOT}"
    echo "[install_tester] Required: pyproject.toml + src/linstabpy/ + examples/"
    echo "[install_tester] Export LINSTABPY_REPO_ROOT=/path/to/linstabpy if the harness was copied elsewhere."
    exit 1
fi

echo "[install_tester] step=diagnose"
if ! bash "${SCRIPT_DIR}/diagnose.sh"; then
    echo "[install_tester] diagnose.sh reported issues. Fix them and re-run."
    exit 1
fi

echo "[install_tester] step=preflight"
bash "${SCRIPT_DIR}/install.sh" --preflight

echo "[install_tester] step=install"
bash "${SCRIPT_DIR}/install.sh"

echo "[install_tester] step=check"
bash "${SCRIPT_DIR}/install.sh" --check

echo "[install_tester] done"
