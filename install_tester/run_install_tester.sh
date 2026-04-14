#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_LABEL="${1:-$(hostname 2>/dev/null || echo unknown-host)}"
RUN_ID="$(date -u +"%Y%m%dT%H%M%SZ")_${RUN_LABEL}"

export LINSTAB2D_INSTALL_RUN_ID="${RUN_ID}"
export LINSTAB2D_INSTALL_LOG_DIR="${SCRIPT_DIR}/install_runs"

echo "[install_tester] run_id=${RUN_ID}"
echo "[install_tester] logs_dir=${LINSTAB2D_INSTALL_LOG_DIR}/${RUN_ID}"

if [ ! -f "${SCRIPT_DIR}/install.sh" ] || [ ! -d "${SCRIPT_DIR}/src" ] || [ ! -d "${SCRIPT_DIR}/examples" ]; then
    echo "[install_tester] ERROR: bundle incomplete at ${SCRIPT_DIR}"
    echo "[install_tester] Required: install.sh + src/ + examples/"
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
