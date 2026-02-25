#!/usr/bin/env bash
set -euo pipefail

# Example-6 Hamada full pipeline runner.
#
# Goal:
# - Always run one reproducible pipeline for the single Hamada reference case:
#   1) solve with PETSc/SLEPc (MUMPS backend),
#   2) compare spectrum to MATLAB reference,
#   3) generate summary figures.
#
# Design choices:
# - Single case directory by default: `example_6_hamada/`
# - MPI is forced to 1 for now (current SciPy->PETSc conversion is serial CSR).
# - Performance comes from threaded BLAS/OpenMP/MUMPS on local cores.
# - Every run writes a timestamped log under `artifacts/logs/`.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# -------------------------------------------------------------------
# 0) Logging/session metadata
# -------------------------------------------------------------------
LOG_DIR="${LINSTAB2D_LOG_DIR:-artifacts/logs}"
mkdir -p "$LOG_DIR"
TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LINSTAB2D_LOG_FILE:-$LOG_DIR/example_6_hamada_${TS}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[hamada] log file: $LOG_FILE"
SCRIPT_START_TS="$(date +%s)"
SCRIPT_START_HUMAN="$(date +"%Y-%m-%d %H:%M:%S %Z")"
echo "[hamada] start: $SCRIPT_START_HUMAN"

# Basic runtime prerequisite: repo-local virtual environment.
if [[ ! -x ".venv/bin/python" ]]; then
  echo "[hamada] missing .venv/bin/python. Run install first."
  exit 1
fi

# -------------------------------------------------------------------
# 1) Verbosity and monitor options
# -------------------------------------------------------------------
# LINSTAB2D_VERBOSE levels:
# 0 = heartbeat only
# 1 = SLEPc EPS monitor + converged reason (default)
# 2 = add ST/KSP converged reasons and shorter heartbeat interval
VERBOSE="${LINSTAB2D_VERBOSE:-1}"
HEARTBEAT_SEC="${LINSTAB2D_HEARTBEAT_SEC:-20}"
declare -a PETSC_VERBOSE_OPTS=()
if [[ "$VERBOSE" -ge 1 ]]; then
  PETSC_VERBOSE_OPTS+=(
    -eps_monitor
    -eps_converged_reason
  )
fi
if [[ "$VERBOSE" -ge 2 ]]; then
  PETSC_VERBOSE_OPTS+=(
    -st_ksp_converged_reason
    -ksp_converged_reason
  )
  HEARTBEAT_SEC=10
fi
echo "[hamada] verbosity level: $VERBOSE"
echo "[hamada] heartbeat interval: ${HEARTBEAT_SEC}s"
if [[ "${#PETSC_VERBOSE_OPTS[@]}" -gt 0 ]]; then
  echo "[hamada] PETSc verbose opts: ${PETSC_VERBOSE_OPTS[*]}"
fi
echo "[hamada] monitor legend: ram_mb=resident physical memory (RSS), cpu_pct=process CPU usage, mem_pct=percent of system RAM"
echo "[hamada] state legend: R=running, S=sleeping/waiting, D=I/O wait, T=stopped, Z=zombie"

# -------------------------------------------------------------------
# 2) Resource discovery and environment setup
# -------------------------------------------------------------------
detect_cores() {
  local n
  n="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  if [[ -z "${n}" ]] && command -v sysctl >/dev/null 2>&1; then
    n="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
  fi
  if [[ -z "${n}" ]]; then
    n=1
  fi
  printf "%s" "$n"
}

CORES="$(detect_cores)"
MPI_PROCS="${LINSTAB2D_MPI_PROCS:-1}"

# Safety cleanup for unattended runs: clear stale python jobs.
echo "[hamada] killing existing python processes"
KILLED=0
for pid in $(ps -ax -o pid=,comm= | awk '$2 ~ /python/ {print $1}'); do
  if kill -9 "$pid" 2>/dev/null; then
    KILLED=$((KILLED + 1))
  fi
done
echo "[hamada] killed python processes: $KILLED"

export OMP_NUM_THREADS="$CORES"
export OPENBLAS_NUM_THREADS="$CORES"
export MKL_NUM_THREADS="$CORES"
export NUMEXPR_NUM_THREADS="$CORES"
export VECLIB_MAXIMUM_THREADS="$CORES"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LINSTAB2D_HAMADA_CASE_DIR="${LINSTAB2D_HAMADA_CASE_DIR:-example_6_hamada}"

echo "[hamada] using logical cores: $CORES"
# Current matrix import path is single-rank CSR; keep MPI fixed at 1.
if [[ "$MPI_PROCS" -ne 1 ]]; then
  echo "[hamada] requested MPI processes: $MPI_PROCS"
  echo "[hamada] forcing MPI processes to 1 (current SciPy->PETSc path is single-rank CSR)."
  MPI_PROCS=1
fi
echo "[hamada] using MPI processes: $MPI_PROCS"
echo "[hamada] using threaded cores (OMP/BLAS): $CORES"
echo "[hamada] case dir: ${LINSTAB2D_HAMADA_CASE_DIR}"

# -------------------------------------------------------------------
# 3) Clean generated outputs from previous run
# -------------------------------------------------------------------
echo "[hamada] cleaning generated Hamada artifacts"

rm -f \
  example_6_hamada.npz \
  example_6_hamada_preview.png \
  example_6_hamada_vs_matlab_spectrum.png \
  example_6_hamada_vs_matlab_mode7.png \
  artifacts/hamada_spectrum_parity.json \
  artifacts/hamada_diagnostics_step1_4.json \
  artifacts/hamada_mapping_sweep.json

# -------------------------------------------------------------------
# 4) Run solver (PETSc/SLEPc) with live heartbeat
# -------------------------------------------------------------------
echo "[hamada] recomputing example_6_hamada"
if ! command -v mpiexec >/dev/null 2>&1; then
  echo "[hamada] mpiexec not found; install MPI to run PETSc/SLEPc in parallel."
  exit 1
fi
echo "[hamada] launching solver (heartbeat every ${HEARTBEAT_SEC}s)"
START_TS="$(date +%s)"
if [[ "${#PETSC_VERBOSE_OPTS[@]}" -gt 0 ]]; then
  mpiexec -n "$MPI_PROCS" .venv/bin/python -u -c "from example_6_hamada import compute; compute()" "${PETSC_VERBOSE_OPTS[@]}" &
else
  mpiexec -n "$MPI_PROCS" .venv/bin/python -u -c "from example_6_hamada import compute; compute()" &
fi
SOLVER_PID=$!

# Resolve a useful monitoring PID (prefer python worker over mpiexec/hydra wrappers).
resolve_monitor_pid() {
  local root="$1"
  local cur="$root"
  local next
  local children
  local line
  while true; do
    children="$(ps -ax -o pid=,ppid=,command= | awk -v p="$cur" '$2==p {print $1 "|" substr($0, index($0, $3))}')"
    if [[ -z "$children" ]]; then
      break
    fi
    next="$(printf "%s\n" "$children" | awk -F'|' '/python/ {print $1; exit}')"
    if [[ -z "$next" ]]; then
      line="$(printf "%s\n" "$children" | head -n 1)"
      next="${line%%|*}"
    fi
    if [[ -z "$next" ]]; then
      break
    fi
    cur="$next"
  done
  printf "%s" "$cur"
}

LAST_HEARTBEAT_TS=0
while kill -0 "$SOLVER_PID" 2>/dev/null; do
  NOW_TS="$(date +%s)"
  if (( LAST_HEARTBEAT_TS == 0 || NOW_TS - LAST_HEARTBEAT_TS >= HEARTBEAT_SEC )); then
    ELAPSED="$((NOW_TS - START_TS))"
    MONITOR_PID="$(resolve_monitor_pid "$SOLVER_PID")"
    STAT_LINE="$(ps -p "$MONITOR_PID" -o etime=,%cpu=,%mem=,rss=,state=,command= 2>/dev/null | xargs || true)"
    if [[ -n "$STAT_LINE" ]]; then
      set -- $STAT_LINE
      P_ETIME="${1:-?}"
      P_CPU="${2:-?}"
      P_MEM="${3:-?}"
      P_RSS_KB="${4:-0}"
      P_STATE="${5:-?}"
      shift 5
      P_CMD="$*"
      case "$P_RSS_KB" in
        ''|*[!0-9]*)
          P_RSS_MB="?"
          ;;
        *)
          P_RSS_MB="$((P_RSS_KB / 1024))"
          ;;
      esac
      echo "[hamada] running t=${ELAPSED}s | pid=${MONITOR_PID} elapsed=${P_ETIME} cpu_pct=${P_CPU} mem_pct=${P_MEM} ram_mb=${P_RSS_MB} proc_state=${P_STATE} cmd=${P_CMD}"
    else
      echo "[hamada] running t=${ELAPSED}s | solver pid=${SOLVER_PID}"
    fi
    LAST_HEARTBEAT_TS="$NOW_TS"
  fi
  sleep 1
done
wait "$SOLVER_PID"
echo "[hamada] solver finished in $(( $(date +%s) - START_TS ))s"

# -------------------------------------------------------------------
# 5) Compare computed spectrum against MATLAB reference
# -------------------------------------------------------------------
echo "[hamada] writing spectrum parity artifact"
set +e
.venv/bin/python tools/parity/check_hamada_spectrum.py \
  --python-npz example_6_hamada.npz \
  --matlab-spectrum example_6_hamada/MATLAB_results/spectrum.mat \
  --artifact artifacts/hamada_spectrum_parity.json
PARITY_EXIT=$?
set -e
if [[ "$PARITY_EXIT" -ne 0 ]]; then
  echo "[hamada] parity check failed (exit=${PARITY_EXIT}) - continuing to generate figures"
fi

# -------------------------------------------------------------------
# 6) Generate quick diagnostic figures for visual comparison
# -------------------------------------------------------------------
echo "[hamada] generating figures"
.venv/bin/python tools/parity/generate_hamada_figures.py \
  --python-npz example_6_hamada.npz \
  --matlab-spectrum example_6_hamada/MATLAB_results/spectrum.mat \
  --matlab-mode example_6_hamada/MATLAB_results/eigenvector_7.mat

# -------------------------------------------------------------------
# 7) End-of-run summary
# -------------------------------------------------------------------
echo "[hamada] done"
SCRIPT_END_TS="$(date +%s)"
SCRIPT_END_HUMAN="$(date +"%Y-%m-%d %H:%M:%S %Z")"
SCRIPT_ELAPSED="$((SCRIPT_END_TS - SCRIPT_START_TS))"
echo "[hamada] end: $SCRIPT_END_HUMAN"
echo "[hamada] total elapsed: ${SCRIPT_ELAPSED}s | cores=${CORES} | mpi=${MPI_PROCS}"
