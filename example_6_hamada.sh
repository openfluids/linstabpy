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
# - MPI rank count is user-configurable (`LINSTAB2D_MPI_PROCS`).
# - Threading per rank is configurable (`LINSTAB2D_THREADS_PER_RANK`).
# - PETSc matrix conversion mode is configurable (`LINSTAB2D_PETSC_MATRIX_MODE`).
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
HOST_NAME="$(hostname 2>/dev/null || echo unknown)"
OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
ARCH_NAME="$(uname -m 2>/dev/null || echo unknown)"

IS_LINUX=0
if [[ "$OS_NAME" == "Linux" ]]; then
  IS_LINUX=1
fi

CPU_MODEL="unknown"
SOCKETS=1
CORES_PER_SOCKET=1
THREADS_PER_CORE=1
PHYSICAL_CORES=1
TOTAL_MEM_GIB=0

if [[ "$IS_LINUX" -eq 1 && -x "$(command -v lscpu 2>/dev/null || true)" ]]; then
  CPU_MODEL="$(lscpu | awk -F: '/Model name/ {print $2; exit}' | xargs || echo unknown)"
  SOCKETS_RAW="$(lscpu | awk -F: '/Socket\\(s\\)/ {print $2; exit}' | xargs || echo 1)"
  CORES_PER_SOCKET_RAW="$(lscpu | awk -F: '/Core\\(s\\) per socket/ {print $2; exit}' | xargs || echo 1)"
  THREADS_PER_CORE_RAW="$(lscpu | awk -F: '/Thread\\(s\\) per core/ {print $2; exit}' | xargs || echo 1)"
  [[ "$SOCKETS_RAW" =~ ^[0-9]+$ ]] && SOCKETS="$SOCKETS_RAW"
  [[ "$CORES_PER_SOCKET_RAW" =~ ^[0-9]+$ ]] && CORES_PER_SOCKET="$CORES_PER_SOCKET_RAW"
  [[ "$THREADS_PER_CORE_RAW" =~ ^[0-9]+$ ]] && THREADS_PER_CORE="$THREADS_PER_CORE_RAW"
fi

PHYSICAL_CORES="$(( SOCKETS * CORES_PER_SOCKET ))"
if [[ "$PHYSICAL_CORES" -lt 1 ]]; then
  PHYSICAL_CORES=1
fi

if [[ "$IS_LINUX" -eq 1 && -x "$(command -v free 2>/dev/null || true)" ]]; then
  MEM_BYTES="$(free -b | awk '/^Mem:/ {print $2; exit}' || echo 0)"
  if [[ "$MEM_BYTES" =~ ^[0-9]+$ ]]; then
    TOTAL_MEM_GIB="$(( MEM_BYTES / 1024 / 1024 / 1024 ))"
  fi
fi

MPI_LAUNCHER=""
if command -v mpiexec >/dev/null 2>&1; then
  MPI_LAUNCHER="mpiexec"
elif command -v mpirun >/dev/null 2>&1; then
  MPI_LAUNCHER="mpirun"
fi
MPI_VERSION="not found"
if [[ -n "$MPI_LAUNCHER" ]]; then
  MPI_VERSION="$("$MPI_LAUNCHER" --version 2>&1 | head -n 1 || true)"
fi

if [[ -n "${LINSTAB2D_MPI_PROCS:-}" ]]; then
  MPI_PROCS="${LINSTAB2D_MPI_PROCS}"
else
  if [[ "$IS_LINUX" -eq 1 ]]; then
    MPI_PROCS="$PHYSICAL_CORES"
  else
    MPI_PROCS=1
  fi
fi
if ! [[ "$MPI_PROCS" =~ ^[0-9]+$ ]] || [[ "$MPI_PROCS" -lt 1 ]]; then
  echo "[hamada] invalid LINSTAB2D_MPI_PROCS=${MPI_PROCS}; expected positive integer."
  exit 1
fi

if [[ -n "${LINSTAB2D_THREADS_PER_RANK:-}" ]]; then
  THREADS_PER_RANK="${LINSTAB2D_THREADS_PER_RANK}"
else
  if [[ "$IS_LINUX" -eq 1 ]]; then
    THREADS_PER_RANK=1
  else
    THREADS_PER_RANK="$(( CORES / MPI_PROCS ))"
  fi
fi
if [[ "$THREADS_PER_RANK" -lt 1 ]]; then
  THREADS_PER_RANK=1
fi

# Hard safety gate: do not allow accidental serial execution.
# Non-serial means at least one parallel dimension is active:
# - MPI ranks > 1, and/or
# - threaded BLAS/OpenMP per rank > 1.
if [[ "$MPI_PROCS" -eq 1 && "$THREADS_PER_RANK" -eq 1 ]]; then
  echo "[hamada] serial configuration detected (MPI_PROCS=1 and THREADS_PER_RANK=1)."
  echo "[hamada] refusing to run in serial mode."
  echo "[hamada] set LINSTAB2D_MPI_PROCS>1 and/or LINSTAB2D_THREADS_PER_RANK>1."
  exit 1
fi

# Safety cleanup for unattended runs: clear stale python jobs.
echo "[hamada] killing existing python processes"
KILLED=0
for pid in $(ps -ax -o pid=,comm= | awk '$2 ~ /python/ {print $1}'); do
  if kill -9 "$pid" 2>/dev/null; then
    KILLED=$((KILLED + 1))
  fi
done
echo "[hamada] killed python processes: $KILLED"

export OMP_NUM_THREADS="$THREADS_PER_RANK"
export OPENBLAS_NUM_THREADS="$THREADS_PER_RANK"
export MKL_NUM_THREADS="$THREADS_PER_RANK"
export NUMEXPR_NUM_THREADS="$THREADS_PER_RANK"
export VECLIB_MAXIMUM_THREADS="$THREADS_PER_RANK"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LINSTAB2D_HAMADA_CASE_DIR="${LINSTAB2D_HAMADA_CASE_DIR:-example_6_hamada}"
# MATLAB export uses baseFlow.kappa=1.4 in case_inputs_used.mat.
export LINSTAB2D_HAMADA_KAPPA="${LINSTAB2D_HAMADA_KAPPA:-1.4}"
# MATLAB export path uses uppercase dMUdT/d2MUdT2 fields (stored as zeros).
export LINSTAB2D_HAMADA_ZERO_MU_T_DERIVS="${LINSTAB2D_HAMADA_ZERO_MU_T_DERIVS:-1}"
# Matrix conversion mode:
#   auto         -> distributed when MPI_PROCS > 1
#   distributed  -> force ownership-aware PETSc AIJ path
#   local        -> force legacy local CSR->AIJ path
if [[ -n "${LINSTAB2D_PETSC_MATRIX_MODE:-}" ]]; then
  export LINSTAB2D_PETSC_MATRIX_MODE="${LINSTAB2D_PETSC_MATRIX_MODE}"
else
  if [[ "$IS_LINUX" -eq 1 ]]; then
    export LINSTAB2D_PETSC_MATRIX_MODE="distributed"
  else
    export LINSTAB2D_PETSC_MATRIX_MODE="auto"
  fi
fi
EIGEN_ST_SOLVER="${LINSTAB2D_EIGEN_ST_SOLVER:-mumps}"
# Root-owned SciPy->PETSc transfer mode for MPI iterative eigensolve.
# When MPI>1, default to assembling the SciPy operator only on rank 0 and
# scattering row-block CSR payloads to owning ranks inside PETSc conversion.
if [[ "$MPI_PROCS" -gt 1 ]]; then
  if [[ "$EIGEN_ST_SOLVER" == "slepc" || "$EIGEN_ST_SOLVER" == "iterative" || "$EIGEN_ST_SOLVER" == "krylov" ]]; then
    export LINSTAB2D_HAMADA_ROOT_ASSEMBLY="${LINSTAB2D_HAMADA_ROOT_ASSEMBLY:-1}"
    export LINSTAB2D_PETSC_ROOT_SCATTER="${LINSTAB2D_PETSC_ROOT_SCATTER:-1}"
  else
    export LINSTAB2D_HAMADA_ROOT_ASSEMBLY="${LINSTAB2D_HAMADA_ROOT_ASSEMBLY:-0}"
    export LINSTAB2D_PETSC_ROOT_SCATTER="${LINSTAB2D_PETSC_ROOT_SCATTER:-0}"
  fi
else
  export LINSTAB2D_HAMADA_ROOT_ASSEMBLY="${LINSTAB2D_HAMADA_ROOT_ASSEMBLY:-0}"
  export LINSTAB2D_PETSC_ROOT_SCATTER="${LINSTAB2D_PETSC_ROOT_SCATTER:-0}"
fi
# Keep MATLAB convention by default: omega = eig(i*L0). Set to 1 only for
# diagnostics on the raw operator eig(L0).
export LINSTAB2D_HAMADA_USE_RAW_EIGS="${LINSTAB2D_HAMADA_USE_RAW_EIGS:-0}"

echo "[hamada] using logical cores: $CORES"
echo "[hamada] host: ${HOST_NAME}"
echo "[hamada] platform: ${OS_NAME} ${ARCH_NAME}"
echo "[hamada] cpu_model: ${CPU_MODEL}"
echo "[hamada] sockets=${SOCKETS} cores_per_socket=${CORES_PER_SOCKET} threads_per_core=${THREADS_PER_CORE}"
echo "[hamada] physical_cores=${PHYSICAL_CORES} logical_cores=${CORES}"
echo "[hamada] total_memory_gib=${TOTAL_MEM_GIB}"
echo "[hamada] mpi_launcher: ${MPI_LAUNCHER:-none}"
echo "[hamada] mpi_version: ${MPI_VERSION}"
if [[ "$IS_LINUX" -eq 1 && "$ARCH_NAME" == "x86_64" && -n "$MPI_LAUNCHER" ]]; then
  echo "[hamada] machine_test: PASS (linux/x86_64 with MPI launcher detected)"
elif [[ "$IS_LINUX" -eq 1 ]]; then
  echo "[hamada] machine_test: WARN (linux detected but MPI launcher missing)"
else
  echo "[hamada] machine_test: WARN (non-linux node; production scaling validation should run on Linux/Xeon)"
fi
echo "[hamada] using MPI processes: $MPI_PROCS"
echo "[hamada] using threaded cores per rank (OMP/BLAS): $THREADS_PER_RANK"
echo "[hamada] case dir: ${LINSTAB2D_HAMADA_CASE_DIR}"
echo "[hamada] kappa: ${LINSTAB2D_HAMADA_KAPPA}"
echo "[hamada] zero_mu_t_derivs: ${LINSTAB2D_HAMADA_ZERO_MU_T_DERIVS}"
echo "[hamada] petsc_matrix_mode: ${LINSTAB2D_PETSC_MATRIX_MODE}"
echo "[hamada] root_assembly_mode: ${LINSTAB2D_HAMADA_ROOT_ASSEMBLY}"
echo "[hamada] petsc_root_scatter: ${LINSTAB2D_PETSC_ROOT_SCATTER}"

# -------------------------------------------------------------------
# 3) Clean generated outputs from previous run
# -------------------------------------------------------------------
echo "[hamada] cleaning generated Hamada artifacts"

rm -f \
  example_6_hamada.npz \
  example_6_hamada_preview.png \
  example_6_hamada_vs_matlab_spectrum.png \
  example_6_hamada_vs_matlab_mode7.png \
  artifacts/parity/hamada_operator_stage_parity.json \
  artifacts/hamada_spectrum_parity.json \
  artifacts/hamada_diagnostics_step1_4.json \
  artifacts/hamada_mapping_sweep.json

# -------------------------------------------------------------------
# 4) Run solver (PETSc/SLEPc) with live heartbeat
# -------------------------------------------------------------------
echo "[hamada] recomputing example_6_hamada"
if [[ "$MPI_PROCS" -gt 1 && -z "$MPI_LAUNCHER" ]]; then
  echo "[hamada] requested MPI_PROCS=${MPI_PROCS}, but no mpiexec/mpirun found."
  echo "[hamada] install MPI launcher (OpenMPI/Intel MPI) or set LINSTAB2D_MPI_PROCS=1."
  exit 1
fi
echo "[hamada] launching solver (heartbeat every ${HEARTBEAT_SEC}s)"
START_TS="$(date +%s)"

declare -a SOLVER_CMD=()
if [[ -n "$MPI_LAUNCHER" ]]; then
  SOLVER_CMD+=("$MPI_LAUNCHER" -n "$MPI_PROCS")
elif [[ "$MPI_PROCS" -ne 1 ]]; then
  echo "[hamada] internal error: MPI_PROCS=${MPI_PROCS} without MPI launcher."
  exit 1
fi
SOLVER_CMD+=(".venv/bin/python" -u -c "from example_6_hamada import compute; compute()")
if [[ "${#PETSC_VERBOSE_OPTS[@]}" -gt 0 ]]; then
  SOLVER_CMD+=("${PETSC_VERBOSE_OPTS[@]}")
fi

# Iterative ST defaults for the SLEPc path.
# Applied only when requested via LINSTAB2D_EIGEN_ST_SOLVER.
if [[ "$EIGEN_ST_SOLVER" == "slepc" || "$EIGEN_ST_SOLVER" == "iterative" || "$EIGEN_ST_SOLVER" == "krylov" ]]; then
  ST_OPTS_STRING="${LINSTAB2D_SLEPC_ST_OPTS:--st_ksp_type gmres -st_pc_type asm -st_pc_asm_overlap 2 -st_sub_ksp_type preonly -st_sub_pc_type lu -st_ksp_rtol 1e-8 -st_ksp_max_it 20000 -st_ksp_error_if_not_converged}"
  IFS=' ' read -r -a ST_OPTS <<< "$ST_OPTS_STRING"
  if [[ "${#ST_OPTS[@]}" -gt 0 ]]; then
    SOLVER_CMD+=("${ST_OPTS[@]}")
    echo "[hamada] iterative ST opts: ${ST_OPTS[*]}"
  fi
fi

"${SOLVER_CMD[@]}" &
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
  --matlab-spectrum example_6_hamada/MATLAB_exports/spectrum.mat \
  --artifact artifacts/hamada_spectrum_parity.json
PARITY_EXIT=$?
set -e
if [[ "$PARITY_EXIT" -ne 0 ]]; then
  echo "[hamada] parity check failed (exit=${PARITY_EXIT}) - continuing to generate figures"
fi

echo "[hamada] writing operator-stage parity artifact"
set +e
.venv/bin/python tools/parity/check_hamada_operator_stages.py \
  --case-dir "${LINSTAB2D_HAMADA_CASE_DIR}" \
  --exports-dir "${LINSTAB2D_HAMADA_CASE_DIR}/MATLAB_exports" \
  --re "${LINSTAB2D_HAMADA_RE:-395000.0}" \
  --ma "${LINSTAB2D_HAMADA_MA:-2.0}" \
  --pr "${LINSTAB2D_HAMADA_PR:-0.747}" \
  --kappa "${LINSTAB2D_HAMADA_KAPPA}" \
  --t0 "${LINSTAB2D_HAMADA_T0:-1540.0}" \
  --beta "${LINSTAB2D_HAMADA_BETA:-0.0}" \
  --penalty "${LINSTAB2D_HAMADA_BC_PENALTY:-1e8}" \
  --fd-order "${LINSTAB2D_HAMADA_FD_ORDER:-4}" \
  --stride-ksi "${LINSTAB2D_HAMADA_STRIDE_KSI:-1}" \
  --stride-eta "${LINSTAB2D_HAMADA_STRIDE_ETA:-1}" \
  --artifact artifacts/parity/hamada_operator_stage_parity.json
OP_PARITY_EXIT=$?
set -e
if [[ "$OP_PARITY_EXIT" -ne 0 ]]; then
  echo "[hamada] operator-stage parity check failed (exit=${OP_PARITY_EXIT}) - continuing to generate figures"
fi

# -------------------------------------------------------------------
# 6) Generate quick diagnostic figures for visual comparison
# -------------------------------------------------------------------
echo "[hamada] generating figures"
.venv/bin/python generate_hamada_figures.py \
  --python-npz example_6_hamada.npz \
  --matlab-spectrum example_6_hamada/MATLAB_exports/spectrum.mat \
  --matlab-mode example_6_hamada/MATLAB_exports/eigenvector_7.mat

# -------------------------------------------------------------------
# 7) End-of-run summary
# -------------------------------------------------------------------
echo "[hamada] done"
SCRIPT_END_TS="$(date +%s)"
SCRIPT_END_HUMAN="$(date +"%Y-%m-%d %H:%M:%S %Z")"
SCRIPT_ELAPSED="$((SCRIPT_END_TS - SCRIPT_START_TS))"
echo "[hamada] end: $SCRIPT_END_HUMAN"
echo "[hamada] total elapsed: ${SCRIPT_ELAPSED}s | cores=${CORES} | mpi=${MPI_PROCS} | threads_per_rank=${THREADS_PER_RANK}"
