#!/usr/bin/env bash
# Run Martini parity MATLAB drivers from examples/martini/linstab2D_scripts/.
#
# Prerequisites:
#   LINSTAB2D_ROOT  — LinStab2D checkout (cases/Martini2024/LinStab2D or upstream clone)
#   MATLAB on PATH (or set MATLAB_BIN)
#
# Outputs (gitignored):
#   ${MARTINI2024_OUTPUT_ROOT:-<repo>/martini_examples/}
#
# Usage:
#   export LINSTAB2D_ROOT=/path/to/LinStab2D
#   bash examples/martini/linstab2D_scripts/run_martini_exports.sh
#   bash .../run_martini_exports.sh --only example_1
#   bash .../run_martini_exports.sh --from example_3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
POSTPROCHAMADA_ROOT="${POSTPROCHAMADA_ROOT:-}"

if [[ -z "${LINSTAB2D_ROOT:-}" ]]; then
  if [[ -n "${POSTPROCHAMADA_ROOT}" && -d "${POSTPROCHAMADA_ROOT}/cases/Martini2024/LinStab2D" ]]; then
    export LINSTAB2D_ROOT="${POSTPROCHAMADA_ROOT}/cases/Martini2024/LinStab2D"
  elif [[ -d "/media/hamada/PostProcHamada/cases/Martini2024/LinStab2D" ]]; then
    export LINSTAB2D_ROOT="/media/hamada/PostProcHamada/cases/Martini2024/LinStab2D"
  fi
fi

export MARTINI2024_OUTPUT_ROOT="${MARTINI2024_OUTPUT_ROOT:-${REPO_ROOT}/martini_examples}"
export MARTINI2024_BATCH=1

MATLAB_BIN="${MATLAB_BIN:-matlab}"
ONLY=""
FROM=""
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="${2:-}"; shift 2 ;;
    --from) FROM="${2:-}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${LINSTAB2D_ROOT:-}" || ! -d "${LINSTAB2D_ROOT}" ]]; then
  echo "ERROR: set LINSTAB2D_ROOT to a LinStab2D checkout." >&2
  exit 1
fi

mkdir -p "${MARTINI2024_OUTPUT_ROOT}"

artifact_exists() {
  [[ -f "${MARTINI2024_OUTPUT_ROOT}/$1.mat" ]]
}

example_index() {
  case "$1" in
    example_0) echo 0 ;;
    example_1) echo 1 ;;
    example_2) echo 2 ;;
    example_3) echo 3 ;;
    example_4) echo 4 ;;
    example_5) echo 5 ;;
    *) echo "ERROR: unknown example '$1'" >&2; exit 1 ;;
  esac
}

should_run() {
  local key="$1"
  [[ -z "${ONLY}" || "${ONLY}" == "${key}" ]] || return 1
  if [[ -n "${FROM}" && "$(example_index "${key}")" -lt "$(example_index "${FROM}")" ]]; then
    return 1
  fi
  return 0
}

run_matlab() {
  local driver="$1"
  local label="$2"
  echo "=== ${label} ==="
  (
    cd "${SCRIPT_DIR}"
    "${MATLAB_BIN}" -nodisplay -nosplash -r \
      "try, run('${driver}'); catch ME, disp(getReport(ME,'extended')), exit(1), end, exit(0);"
  )
}

run_if_missing() {
  local driver="$1"
  local label="$2"
  local stem="$3"
  if [[ "${FORCE}" -eq 0 ]] && artifact_exists "${stem}"; then
    echo "=== Skipping ${label} (${stem}.mat exists) ==="
    return 0
  fi
  run_matlab "${driver}" "${label}"
}

if should_run example_0; then
  run_if_missing example_0_TestCurvilinearDerivative.m "Example 0" \
    example_0_TestCurvilinearDerivative_results
fi
if should_run example_1; then
  run_if_missing example_1_Blasius.m "Example 1" example_1_Blasius_results
fi
if should_run example_2; then
  run_if_missing example_2_Jet.m "Example 2" example_2_Jet_results
fi
if should_run example_3; then
  run_if_missing run_ex3_bundled_parity.m "Ex3 bundled parity (compare gate)" \
    example_3_Streaky_BL_results_temporal
  run_if_missing run_ex3_paper_temporal_Lx2pi.m "Ex3 paper temporal Lx2pi" \
    example_3_paper_temporal_Lx2pi_nP2_results_temporal
  run_if_missing run_ex3_paper_temporal_Lxpi_nfloq0.m "Ex3 paper temporal nfloq0" \
    example_3_paper_temporal_Lxpi_nfloq0_results_temporal
  run_if_missing run_ex3_paper_temporal_Lxpi_nfloq2.m "Ex3 paper temporal nfloq2" \
    example_3_paper_temporal_Lxpi_nfloq2_results_temporal
  run_if_missing run_ex3_paper_spatial.m "Ex3 paper spatial (+ temporal block)" \
    example_3_paper_spatial_Lxpi_nP2_results_spatial
  run_if_missing run_ex3_paper_spatial.m "Ex3 paper spatial temporal block" \
    example_3_paper_spatial_Lxpi_nP2_results_temporal
fi
if should_run example_4; then
  run_if_missing example_4_Cavity.m "Example 4" example_4_Cavity_results_direct
fi
if should_run example_5; then
  run_if_missing example_5_Paraboloid.m "Example 5" example_5_Paraboloid_results
fi

echo "Done. Raw .mat -> ${MARTINI2024_OUTPUT_ROOT}"
