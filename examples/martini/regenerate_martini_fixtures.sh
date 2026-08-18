#!/usr/bin/env bash
# Regenerate Martini MATLAB fixtures: LinStab2D run -> .mat -> NPZ -> LinStab2DResults/.
#
# Contract (Example 3):
#   temporal: omega (50) + V (50 modes on mesh), plot_grid, idx, metadata — no floquet_ext*
#   spatial:  alphas (50) + plot_grid, idx, metadata — no spatial mode vectors
#
# Prerequisites:
#   LINSTAB2D_ROOT, MATLAB on PATH
#
# Usage (repo root):
#   bash examples/martini/regenerate_martini_fixtures.sh
#   bash examples/martini/regenerate_martini_fixtures.sh --only example_3 --force

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAGING="${REPO_ROOT}/martini_examples/_extract_staging"
RESULTS="${REPO_ROOT}/examples/martini/LinStab2DResults"

PYTHON="${LINSTABPY_PYTHON:-python3}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
fi

RUNNER_ARGS=()
SPLIT_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --split-only)
      SPLIT_ONLY=1
      shift
      ;;
    --staging-dir)
      STAGING="${2:-}"
      shift 2
      ;;
    *)
      RUNNER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${SPLIT_ONLY}" -eq 0 ]]; then
  echo "=== Step 1/3: MATLAB export (linstab2D_scripts) ==="
  bash "${SCRIPT_DIR}/linstab2D_scripts/run_martini_exports.sh" "${RUNNER_ARGS[@]}"

  echo "=== Step 2/3: Extract .mat -> staging NPZ ==="
  rm -rf "${STAGING}"
  mkdir -p "${STAGING}"
  "${PYTHON}" "${SCRIPT_DIR}/extract_martini_to_npz.py" \
    --input-dir "${REPO_ROOT}/martini_examples" \
    --output-dir "${STAGING}"
else
  echo "=== Split-only mode (skip MATLAB + extract) ==="
  if [[ ! -d "${STAGING}" ]]; then
    echo "ERROR: staging dir missing: ${STAGING}" >&2
    exit 1
  fi
fi

echo "=== Step 3/3: Split staging -> LinStab2DResults ==="
"${PYTHON}" "${SCRIPT_DIR}/split_martini_fixtures.py" \
  --input-dir "${STAGING}" \
  --output-dir "${RESULTS}"

echo "Done."
echo "  staging:  ${STAGING}"
echo "  fixtures: ${RESULTS}"
echo "Optional: ${PYTHON} examples/martini/compare_martini_examples.py"
