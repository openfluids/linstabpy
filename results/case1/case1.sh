#!/usr/bin/env bash
set -euo pipefail

# Case 1 (SWBLI) full pipeline runner.
#
# Goal:
# - Always run one reproducible pipeline for the single Case~1 (SWBLI) reference case:
#   1) solve with PETSc/SLEPc (MUMPS backend),
#   2) compare spectrum to MATLAB reference,
#   3) generate summary figures.
#
# Design choices:
# - Single case directory by default: `results/case1/`
# - MPI rank count is user-configurable (`LINSTAB2D_MPI_PROCS`).
# - Threading per rank is configurable (`LINSTAB2D_THREADS_PER_RANK`).
# - PETSc matrix conversion mode is configurable (`LINSTAB2D_PETSC_MATRIX_MODE`).
# - Every run writes a timestamped log under `logs/`.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# -------------------------------------------------------------------
# 0) Logging/session metadata
# -------------------------------------------------------------------
LOG_DIR="${LINSTAB2D_LOG_DIR:-output/logs}"
mkdir -p "$LOG_DIR"
TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LINSTAB2D_LOG_FILE:-$LOG_DIR/case1_${TS}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[case1] log file: $LOG_FILE"
SCRIPT_START_TS="$(date +%s)"
SCRIPT_START_HUMAN="$(date +"%Y-%m-%d %H:%M:%S %Z")"
echo "[case1] start: $SCRIPT_START_HUMAN"

# Basic runtime prerequisite: repo-local virtual environment.
if [[ ! -x ".venv/bin/python" ]]; then
  echo "[case1] missing .venv/bin/python. Run install first."
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
echo "[case1] verbosity level: $VERBOSE"
echo "[case1] heartbeat interval: ${HEARTBEAT_SEC}s"
if [[ "${#PETSC_VERBOSE_OPTS[@]}" -gt 0 ]]; then
  echo "[case1] PETSc verbose opts: ${PETSC_VERBOSE_OPTS[*]}"
fi
echo "[case1] monitor legend: ram_mb=resident physical memory (RSS), cpu_pct=process CPU usage, mem_pct=percent of system RAM"
echo "[case1] state legend: R=running, S=sleeping/waiting, D=I/O wait, T=stopped, Z=zombie"

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
  echo "[case1] invalid LINSTAB2D_MPI_PROCS=${MPI_PROCS}; expected positive integer."
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
  echo "[case1] serial configuration detected (MPI_PROCS=1 and THREADS_PER_RANK=1)."
  echo "[case1] refusing to run in serial mode."
  echo "[case1] set LINSTAB2D_MPI_PROCS>1 and/or LINSTAB2D_THREADS_PER_RANK>1."
  exit 1
fi

# Safety cleanup for unattended runs: clear stale python jobs.
echo "[case1] killing existing python processes"
KILLED=0
for pid in $(ps -ax -o pid=,comm= | awk '$2 ~ /python/ {print $1}'); do
  if kill -9 "$pid" 2>/dev/null; then
    KILLED=$((KILLED + 1))
  fi
done
echo "[case1] killed python processes: $KILLED"

export OMP_NUM_THREADS="$THREADS_PER_RANK"
export OPENBLAS_NUM_THREADS="$THREADS_PER_RANK"
export MKL_NUM_THREADS="$THREADS_PER_RANK"
export NUMEXPR_NUM_THREADS="$THREADS_PER_RANK"
export VECLIB_MAXIMUM_THREADS="$THREADS_PER_RANK"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
LINSTAB2D_CASE1_CONFIG_BASE="${LINSTAB2D_CASE1_CONFIG:-parity_config.jsonc}"
export LINSTAB2D_CASE1_CONFIG="$LINSTAB2D_CASE1_CONFIG_BASE"
export LINSTAB2D_CASE1_CASE_DIR="${LINSTAB2D_CASE1_CASE_DIR:-"."}"
# MATLAB export uses baseFlow.kappa=1.4 in case_inputs_used.mat.
export LINSTAB2D_CASE1_KAPPA="${LINSTAB2D_CASE1_KAPPA:-1.4}"
# MATLAB export path uses uppercase dMUdT/d2MUdT2 fields (stored as zeros).
export LINSTAB2D_CASE1_ZERO_MU_T_DERIVS="${LINSTAB2D_CASE1_ZERO_MU_T_DERIVS:-1}"
export LINSTAB2D_CASE1_USE_MATLAB_SPONGE_AUX="${LINSTAB2D_CASE1_USE_MATLAB_SPONGE_AUX:-1}"
export LINSTAB2D_CASE1_CASE_INPUTS_MAT="${LINSTAB2D_CASE1_CASE_INPUTS_MAT:-MATLAB_exports_resolvent/case_inputs_used.mat}"
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
# Resolvent linear solve mode:
#   mumps      -> robust direct baseline
#   iterative  -> scalable PETSc Krylov route
export LINSTAB2D_RESOLVENT_TYPE="${LINSTAB2D_RESOLVENT_TYPE:-mumps}"
if [[ "${LINSTAB2D_RESOLVENT_TYPE}" == "iterative" || "${LINSTAB2D_RESOLVENT_TYPE}" == "krylov" || "${LINSTAB2D_RESOLVENT_TYPE}" == "petsc" ]]; then
  export LINSTAB2D_RESOLVENT_KSP_TYPE="${LINSTAB2D_RESOLVENT_KSP_TYPE:-fgmres}"
  export LINSTAB2D_RESOLVENT_PC_TYPE="${LINSTAB2D_RESOLVENT_PC_TYPE:-asm}"
  export LINSTAB2D_RESOLVENT_KSP_RTOL="${LINSTAB2D_RESOLVENT_KSP_RTOL:-1e-10}"
  export LINSTAB2D_RESOLVENT_KSP_ATOL="${LINSTAB2D_RESOLVENT_KSP_ATOL:-1e-14}"
  export LINSTAB2D_RESOLVENT_KSP_MAX_IT="${LINSTAB2D_RESOLVENT_KSP_MAX_IT:-20000}"
  export LINSTAB2D_RESOLVENT_SUB_KSP_TYPE="${LINSTAB2D_RESOLVENT_SUB_KSP_TYPE:-preonly}"
  export LINSTAB2D_RESOLVENT_SUB_PC_TYPE="${LINSTAB2D_RESOLVENT_SUB_PC_TYPE:-lu}"
fi
# Root-owned SciPy->PETSc transfer mode for MPI iterative eigensolve.
# When MPI>1, default to assembling the SciPy operator only on rank 0 and
# scattering row-block CSR payloads to owning ranks inside PETSc conversion.
if [[ "$MPI_PROCS" -gt 1 ]]; then
  if [[ "$EIGEN_ST_SOLVER" == "slepc" || "$EIGEN_ST_SOLVER" == "iterative" || "$EIGEN_ST_SOLVER" == "krylov" ]]; then
    export LINSTAB2D_CASE1_ROOT_ASSEMBLY="${LINSTAB2D_CASE1_ROOT_ASSEMBLY:-1}"
    export LINSTAB2D_PETSC_ROOT_SCATTER="${LINSTAB2D_PETSC_ROOT_SCATTER:-1}"
  else
    export LINSTAB2D_CASE1_ROOT_ASSEMBLY="${LINSTAB2D_CASE1_ROOT_ASSEMBLY:-0}"
    export LINSTAB2D_PETSC_ROOT_SCATTER="${LINSTAB2D_PETSC_ROOT_SCATTER:-0}"
  fi
else
  export LINSTAB2D_CASE1_ROOT_ASSEMBLY="${LINSTAB2D_CASE1_ROOT_ASSEMBLY:-0}"
  export LINSTAB2D_PETSC_ROOT_SCATTER="${LINSTAB2D_PETSC_ROOT_SCATTER:-0}"
fi
# Keep MATLAB convention by default: omega = eig(i*L0). Set to 1 only for
# diagnostics on the raw operator eig(L0).
export LINSTAB2D_CASE1_USE_RAW_EIGS="${LINSTAB2D_CASE1_USE_RAW_EIGS:-0}"

echo "[case1] using logical cores: $CORES"
echo "[case1] host: ${HOST_NAME}"
echo "[case1] platform: ${OS_NAME} ${ARCH_NAME}"
echo "[case1] cpu_model: ${CPU_MODEL}"
echo "[case1] sockets=${SOCKETS} cores_per_socket=${CORES_PER_SOCKET} threads_per_core=${THREADS_PER_CORE}"
echo "[case1] physical_cores=${PHYSICAL_CORES} logical_cores=${CORES}"
echo "[case1] total_memory_gib=${TOTAL_MEM_GIB}"
echo "[case1] mpi_launcher: ${MPI_LAUNCHER:-none}"
echo "[case1] mpi_version: ${MPI_VERSION}"
if [[ "$IS_LINUX" -eq 1 && "$ARCH_NAME" == "x86_64" && -n "$MPI_LAUNCHER" ]]; then
  echo "[case1] machine_test: PASS (linux/x86_64 with MPI launcher detected)"
elif [[ "$IS_LINUX" -eq 1 ]]; then
  echo "[case1] machine_test: WARN (linux detected but MPI launcher missing)"
else
  echo "[case1] machine_test: WARN (non-linux node; production scaling validation should run on Linux/Xeon)"
fi
echo "[case1] using MPI processes: $MPI_PROCS"
echo "[case1] using threaded cores per rank (OMP/BLAS): $THREADS_PER_RANK"
echo "[case1] case dir: ${LINSTAB2D_CASE1_CASE_DIR}"
echo "[case1] config: ${LINSTAB2D_CASE1_CONFIG}"
echo "[case1] kappa: ${LINSTAB2D_CASE1_KAPPA}"
echo "[case1] zero_mu_t_derivs: ${LINSTAB2D_CASE1_ZERO_MU_T_DERIVS}"
echo "[case1] use_matlab_sponge_aux: ${LINSTAB2D_CASE1_USE_MATLAB_SPONGE_AUX}"
echo "[case1] case_inputs_mat: ${LINSTAB2D_CASE1_CASE_INPUTS_MAT}"
echo "[case1] resolvent_linear_solver: ${LINSTAB2D_RESOLVENT_TYPE}"
echo "[case1] petsc_matrix_mode: ${LINSTAB2D_PETSC_MATRIX_MODE}"
echo "[case1] root_assembly_mode: ${LINSTAB2D_CASE1_ROOT_ASSEMBLY}"
echo "[case1] petsc_root_scatter: ${LINSTAB2D_PETSC_ROOT_SCATTER}"

# -------------------------------------------------------------------
# 3) Two-pass run plan and reference mapping
# -------------------------------------------------------------------
RUN_PLAN="${LINSTAB2D_CASE1_RUN_PLAN:-both}"
STABILITY_MATLAB_DIR="${LINSTAB2D_CASE1_MATLAB_STABILITY_DIR:-MATLAB_exports}"
RESOLVENT_MATLAB_DIR="${LINSTAB2D_CASE1_MATLAB_RESOLVENT_DIR:-MATLAB_exports_resolvent}"
STABILITY_SPECTRUM_REF="${STABILITY_MATLAB_DIR}/spectrum.mat"
STABILITY_MODE7_REF="${STABILITY_MATLAB_DIR}/eigenvector_7.mat"
RESOLVENT_SVD_REF="${RESOLVENT_MATLAB_DIR}/res_svd_final.mat"

echo "[case1] base config: ${LINSTAB2D_CASE1_CONFIG_BASE}"
echo "[case1] run plan: ${RUN_PLAN}"
echo "[case1] stability reference: ${STABILITY_SPECTRUM_REF}"
echo "[case1] stability mode reference: ${STABILITY_MODE7_REF}"
echo "[case1] resolvent reference: ${RESOLVENT_SVD_REF}"

for required_path in "$STABILITY_SPECTRUM_REF" "$STABILITY_MODE7_REF" "$RESOLVENT_SVD_REF"; do
  if [[ ! -f "$required_path" ]]; then
    echo "[case1] missing required reference file: $required_path"
    exit 1
  fi
done

mkdir -p output/parity output/runs output/figures output/tmp

if [[ "$MPI_PROCS" -gt 1 && -z "$MPI_LAUNCHER" ]]; then
  echo "[case1] requested MPI_PROCS=${MPI_PROCS}, but no mpiexec/mpirun found."
  echo "[case1] install MPI launcher (OpenMPI/Intel MPI) or set LINSTAB2D_MPI_PROCS=1."
  exit 1
fi

clean_generated_outputs() {
  rm -f \
    case1.npz \
    case1_preview.png \
    case1_vs_matlab_spectrum.png \
    case1_leading_modes.png \
    case1_vs_matlab_mode7.png \
    case1_mode7_cuts.png \
    case1_mode7_2d_compare.png \
    case1_mode7_error_maps.png \
    case1_mode7_parity_scatter.png \
    case1_dashboard_stability.png \
    case1_dashboard_resolvent.png \
    case1_dashboard_overview.png \
    case1_dashboard_master.png \
    case1_modes2d.png \
    case1_resolvent_gains.png \
    case1_resolvent_mode1_maps.png \
    case1_resolvent_mode1_maps_enhanced.png \
    case1_resolvent_mode1_maps_zoomed.png \
    case1_resolvent_mode1_error_maps.png \
    case1_resolvent_mode1_cuts.png \
    case1_resolvent_mode1_zoom_logmag.png \
    case1_resolvent_response_gallery.png \
    case1_resolvent_parity_errors.png
}

prepare_run_config() {
  local pass_name="$1"
  local run_stability="$2"
  local run_resolvent="$3"
  local use_sponge_aux="$4"
  local out_cfg="output/tmp/case1_${pass_name}_config.json"

  .venv/bin/python - "$LINSTAB2D_CASE1_CONFIG_BASE" "$out_cfg" "$run_stability" "$run_resolvent" "$use_sponge_aux" <<'PY'
import json
import sys
from pathlib import Path

from examples._helpers import load_jsonc

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
run_stability = bool(int(sys.argv[3]))
run_resolvent = bool(int(sys.argv[4]))
use_sponge_aux = bool(int(sys.argv[5]))

cfg = load_jsonc(src)
cfg.setdefault("analysis", {})
cfg.setdefault("physics", {})
cfg["analysis"]["run_stability"] = run_stability
cfg["analysis"]["run_resolvent"] = run_resolvent
cfg["physics"]["use_matlab_sponge_aux"] = use_sponge_aux
dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY

  printf "%s" "$out_cfg"
}

# Resolve a useful monitoring PID (prefer python worker over mpiexec wrappers).
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

LAST_PASS_ELAPSED=0

run_solver_pass() {
  local pass_name="$1"
  local pass_run_stability="$2"
  local pass_run_resolvent="$3"
  local pass_use_sponge_aux="$4"
  local pass_cfg_path
  local pass_start_ts
  local pass_elapsed
  local last_heartbeat_ts
  local now_ts
  local elapsed
  local solver_pid
  local monitor_pid
  local stat_line
  local fig_count

  echo "[case1] ------------------------------------------------------------"
  echo "[case1] pass=${pass_name} run_stability=${pass_run_stability} run_resolvent=${pass_run_resolvent} use_matlab_sponge_aux=${pass_use_sponge_aux}"

  clean_generated_outputs

  pass_cfg_path="$(prepare_run_config "$pass_name" "$pass_run_stability" "$pass_run_resolvent" "$pass_use_sponge_aux")"
  export LINSTAB2D_CASE1_CONFIG="$pass_cfg_path"
  export LINSTAB2D_CASE1_RUN_STABILITY="$pass_run_stability"
  export LINSTAB2D_CASE1_RUN_RESOLVENT="$pass_run_resolvent"
  export LINSTAB2D_CASE1_USE_MATLAB_SPONGE_AUX="$pass_use_sponge_aux"

  echo "[case1] pass config: ${pass_cfg_path}"
  echo "[case1] launching solver (heartbeat every ${HEARTBEAT_SEC}s)"
  pass_start_ts="$(date +%s)"

  declare -a solver_cmd=()
  if [[ -n "$MPI_LAUNCHER" ]]; then
    solver_cmd+=("$MPI_LAUNCHER" -n "$MPI_PROCS")
  elif [[ "$MPI_PROCS" -ne 1 ]]; then
    echo "[case1] internal error: MPI_PROCS=${MPI_PROCS} without MPI launcher."
    exit 1
  fi
  solver_cmd+=(".venv/bin/python" -u -c "from case1 import compute; compute()")
  if [[ "${#PETSC_VERBOSE_OPTS[@]}" -gt 0 ]]; then
    solver_cmd+=("${PETSC_VERBOSE_OPTS[@]}")
  fi
  if [[ "$EIGEN_ST_SOLVER" == "slepc" || "$EIGEN_ST_SOLVER" == "iterative" || "$EIGEN_ST_SOLVER" == "krylov" ]]; then
    ST_OPTS_STRING="${LINSTAB2D_SLEPC_ST_OPTS:--st_ksp_type gmres -st_pc_type asm -st_pc_asm_overlap 2 -st_sub_ksp_type preonly -st_sub_pc_type lu -st_ksp_rtol 1e-8 -st_ksp_max_it 20000 -st_ksp_error_if_not_converged}"
    IFS=' ' read -r -a ST_OPTS <<< "$ST_OPTS_STRING"
    if [[ "${#ST_OPTS[@]}" -gt 0 ]]; then
      solver_cmd+=("${ST_OPTS[@]}")
      echo "[case1] iterative ST opts: ${ST_OPTS[*]}"
    fi
  fi

  "${solver_cmd[@]}" &
  solver_pid=$!
  last_heartbeat_ts=0
  while kill -0 "$solver_pid" 2>/dev/null; do
    now_ts="$(date +%s)"
    if (( last_heartbeat_ts == 0 || now_ts - last_heartbeat_ts >= HEARTBEAT_SEC )); then
      elapsed="$((now_ts - pass_start_ts))"
      monitor_pid="$(resolve_monitor_pid "$solver_pid")"
      stat_line="$(ps -p "$monitor_pid" -o etime=,%cpu=,%mem=,rss=,state=,command= 2>/dev/null | xargs || true)"
      if [[ -n "$stat_line" ]]; then
        set -- $stat_line
        p_etime="${1:-?}"
        p_cpu="${2:-?}"
        p_mem="${3:-?}"
        p_rss_kb="${4:-0}"
        p_state="${5:-?}"
        shift 5
        p_cmd="$*"
        case "$p_rss_kb" in
          ''|*[!0-9]*)
            p_rss_mb="?"
            ;;
          *)
            p_rss_mb="$((p_rss_kb / 1024))"
            ;;
        esac
        echo "[case1] pass=${pass_name} t=${elapsed}s | pid=${monitor_pid} elapsed=${p_etime} cpu_pct=${p_cpu} mem_pct=${p_mem} ram_mb=${p_rss_mb} proc_state=${p_state} cmd=${p_cmd}"
      else
        echo "[case1] pass=${pass_name} t=${elapsed}s | solver pid=${solver_pid}"
      fi
      last_heartbeat_ts="$now_ts"
    fi
    sleep 1
  done
  wait "$solver_pid"
  pass_elapsed="$(( $(date +%s) - pass_start_ts ))"
  LAST_PASS_ELAPSED="$pass_elapsed"
  echo "[case1] pass=${pass_name} solver finished in ${pass_elapsed}s"

  cp case1.npz "output/runs/case1_${pass_name}.npz"
  echo "[case1] saved: output/runs/case1_${pass_name}.npz"

  if [[ "$pass_run_stability" == "1" ]]; then
    echo "[case1] pass=${pass_name} writing spectrum parity artifact"
    set +e
    .venv/bin/python check_spectrum.py \
      --python-npz case1.npz \
      --matlab-spectrum "$STABILITY_SPECTRUM_REF" \
      --artifact "output/parity/case1_spectrum_parity_${pass_name}.json"
    parity_exit=$?
    set -e
    if [[ "$parity_exit" -ne 0 ]]; then
      echo "[case1] pass=${pass_name} spectrum parity failed (exit=${parity_exit})"
    fi
  fi

  if [[ "$pass_run_resolvent" == "1" ]]; then
    echo "[case1] pass=${pass_name} writing operator-stage parity artifact"
    set +e
    .venv/bin/python check_operator_stages.py \
      --config "$pass_cfg_path" \
      --artifact "output/parity/case1_operator_stage_parity_${pass_name}.json"
    op_parity_exit=$?
    set -e
    if [[ "$op_parity_exit" -ne 0 ]]; then
      echo "[case1] pass=${pass_name} operator-stage parity failed (exit=${op_parity_exit})"
    fi

    echo "[case1] pass=${pass_name} writing resolvent parity artifact"
    set +e
    .venv/bin/python check_resolvent.py \
      --config "$pass_cfg_path" \
      --python-npz case1.npz \
      --matlab-resolvent "$RESOLVENT_SVD_REF" \
      --artifact "output/parity/case1_resolvent_parity_${pass_name}.json"
    res_parity_exit=$?
    set -e
    if [[ "$res_parity_exit" -ne 0 ]]; then
      echo "[case1] pass=${pass_name} resolvent parity failed (exit=${res_parity_exit})"
    fi
  fi

  echo "[case1] pass=${pass_name} generating figures"
  .venv/bin/python generate_figures.py \
    --python-npz case1.npz \
    --matlab-spectrum "$STABILITY_SPECTRUM_REF" \
    --matlab-mode "$STABILITY_MODE7_REF" \
    --matlab-resolvent "$RESOLVENT_SVD_REF" \
    --parity-config "$pass_cfg_path"

  mkdir -p "output/figures/${pass_name}"
  fig_count=0
  shopt -s nullglob
  for fig in case1*.png; do
    cp "$fig" "output/figures/${pass_name}/"
    fig_count=$((fig_count + 1))
  done
  shopt -u nullglob
  echo "[case1] pass=${pass_name} saved figures: ${fig_count} -> output/figures/${pass_name}/"
}

build_master_dashboard() {
  local stability_dir="output/figures/stability"
  local resolvent_dir="output/figures/resolvent"
  local master_png="output/figures/case1_dashboard_master.png"

  if [[ ! -d "$stability_dir" || ! -d "$resolvent_dir" ]]; then
    echo "[case1] master dashboard skipped: missing pass figure folders"
    return 0
  fi

  echo "[case1] building master dashboard (2x3)"
  .venv/bin/python - "$stability_dir" "$resolvent_dir" "$master_png" <<'PY'
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

stability = Path(sys.argv[1])
resolvent = Path(sys.argv[2])
out = Path(sys.argv[3])

tiles = [
    (stability / "case1_vs_matlab_spectrum.png", "Stability spectrum parity"),
    (stability / "case1_mode7_2d_compare.png", "Stability mode 7 parity"),
    (stability / "case1_mode7_error_maps.png", "Stability mode 7 errors"),
    (resolvent / "case1_resolvent_gains.png", "Resolvent gains parity"),
    (resolvent / "case1_resolvent_mode1_maps_zoomed.png", "Resolvent mode 1 zoom"),
    (resolvent / "case1_resolvent_parity_errors.png", "Resolvent parity errors"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (path, title) in zip(axes.ravel(), tiles):
    ax.set_title(title, fontsize=11)
    if path.exists():
        image = mpimg.imread(path)
        ax.imshow(image)
        ax.axis("off")
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"missing:\n{path.name}",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )

fig.suptitle("Case 1 dashboard (stability + resolvent)", fontsize=15)
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
PY

  cp "$master_png" "case1_dashboard_master.png"
  cp "$master_png" "$stability_dir/"
  cp "$master_png" "$resolvent_dir/"
  echo "[case1] generated: $master_png"
}

# -------------------------------------------------------------------
# 4) Execute selected run plan (stability, resolvent, or both)
# -------------------------------------------------------------------
STABILITY_ELAPSED=""
RESOLVENT_ELAPSED=""

case "$RUN_PLAN" in
  both)
    run_solver_pass "stability" "1" "0" "0"
    STABILITY_ELAPSED="$LAST_PASS_ELAPSED"
    run_solver_pass "resolvent" "0" "1" "1"
    RESOLVENT_ELAPSED="$LAST_PASS_ELAPSED"
    ;;
  stability)
    run_solver_pass "stability" "1" "0" "0"
    STABILITY_ELAPSED="$LAST_PASS_ELAPSED"
    ;;
  resolvent)
    run_solver_pass "resolvent" "0" "1" "1"
    RESOLVENT_ELAPSED="$LAST_PASS_ELAPSED"
    ;;
  *)
    echo "[case1] invalid LINSTAB2D_CASE1_RUN_PLAN=${RUN_PLAN} (expected: both|stability|resolvent)"
    exit 1
    ;;
esac

if [[ "$RUN_PLAN" == "both" ]]; then
  build_master_dashboard
fi

# -------------------------------------------------------------------
# 5) End-of-run summary
# -------------------------------------------------------------------
echo "[case1] done"
SCRIPT_END_TS="$(date +%s)"
SCRIPT_END_HUMAN="$(date +"%Y-%m-%d %H:%M:%S %Z")"
SCRIPT_ELAPSED="$((SCRIPT_END_TS - SCRIPT_START_TS))"
echo "[case1] end: $SCRIPT_END_HUMAN"
if [[ -n "$STABILITY_ELAPSED" ]]; then
  echo "[case1] pass_runtime stability=${STABILITY_ELAPSED}s"
fi
if [[ -n "$RESOLVENT_ELAPSED" ]]; then
  echo "[case1] pass_runtime resolvent=${RESOLVENT_ELAPSED}s"
fi
echo "[case1] total elapsed: ${SCRIPT_ELAPSED}s | cores=${CORES} | mpi=${MPI_PROCS} | threads_per_rank=${THREADS_PER_RANK}"
