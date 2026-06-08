#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  cat >&2 <<'EOF'
usage: run_case_once.sh <config-jsonc> <output-npz> [case-module]

Examples:
  LINSTABPY_MPI_PROCS=1 \
  examples/case_studies/case2_lui/run_case_once.sh \
    examples/case_studies/case2_lui/configs/parity_config.jsonc \
    examples/case_studies/case2_lui/output/runs/case2_smoke.npz

  LINSTABPY_MPI_PROCS=4 \
  LINSTABPY_MPI_PRESET=none \
  examples/case_studies/case2_lui/run_case_once.sh \
    examples/case_studies/case2_lui/configs/parity_config.jsonc \
    examples/case_studies/case2_lui/output/runs/case2_socket_single.npz

  # Otimização específica desta máquina:
  examples/case_studies/case2_lui/run_case_once.sh \
    examples/case_studies/case2_lui/configs/parity_config.jsonc \
    examples/case_studies/case2_lui/output/runs/case2_manual.npz

  LINSTABPY_MPI_PROCS=10 \
  LINSTABPY_MPI_PRESET=socket0 \
  LINSTABPY_STAGE_HEARTBEAT_SECONDS=30 \
  examples/case_studies/case2_lui/run_case_once.sh \
    examples/case_studies/case2_lui/configs/parity_config.jsonc \
    examples/case_studies/case2_lui/output/runs/case2_manual.npz
EOF
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
CASE_DIR="$ROOT/examples/case_studies/case2_lui"
PYTHON_EXE="$ROOT/.venv/bin/python"

CONFIG_JSONC="$(realpath "$1")"
OUTPUT_NPZ="$(realpath -m "$2")"
CASE_MODULE="${3:-case2}"

if [ ! -f "$CONFIG_JSONC" ]; then
  echo "config not found: $CONFIG_JSONC" >&2
  exit 2
fi

MPI_PROCS="${LINSTABPY_MPI_PROCS:-1}"
MPI_PRESET="${LINSTABPY_MPI_PRESET:-none}"
HEARTBEAT_SECONDS="${LINSTABPY_STAGE_HEARTBEAT_SECONDS:-30}"

mkdir -p "$(dirname "$OUTPUT_NPZ")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export LINSTABPY_CASE_MODULE="$CASE_MODULE"
export LINSTABPY_CASE2_CONFIG="$CONFIG_JSONC"
export LINSTABPY_CASE_CONFIG="$CONFIG_JSONC"
export LINSTABPY_CASE2_OUTPUT_NPZ="$OUTPUT_NPZ"
export LINSTABPY_CASE_OUTPUT_NPZ="$OUTPUT_NPZ"
export LINSTABPY_STAGE_HEARTBEAT_SECONDS="$HEARTBEAT_SECONDS"

echo "[case2-run] config=$CONFIG_JSONC"
echo "[case2-run] output=$OUTPUT_NPZ"
echo "[case2-run] case_module=$CASE_MODULE"
echo "[case2-run] mpi_procs=$MPI_PROCS"
echo "[case2-run] mpi_preset=$MPI_PRESET"

cd "$CASE_DIR"

if [ "$MPI_PROCS" -le 1 ]; then
  exec "$PYTHON_EXE" case2.py
fi

launcher=()

# Rankfile override: explicit path takes precedence over preset
if [ -n "${LINSTABPY_MPI_RANKFILE:-}" ]; then
  if [ ! -f "$LINSTABPY_MPI_RANKFILE" ]; then
    echo "LINSTABPY_MPI_RANKFILE not found: $LINSTABPY_MPI_RANKFILE" >&2
    exit 2
  fi
  launcher=(mpiexec --rankfile "$LINSTABPY_MPI_RANKFILE" --bind-to core)
else
  case "$MPI_PRESET" in
    socket0)
      launcher=(mpiexec --rankfile "$CASE_DIR/socket0.rankfile" --bind-to core)
      ;;
    socket1)
      launcher=(mpiexec --rankfile "$CASE_DIR/socket1.rankfile" --bind-to core)
      ;;
    socket01)
      launcher=(mpiexec --rankfile "$CASE_DIR/socket01.rankfile" --bind-to core)
      ;;
    c9)
      # c9: Intel MPI 2021.x on an sm_86 GPU node. Hardware-agnostic preset.
      # Under PBS the host/slot layout comes from $PBS_NODEFILE; otherwise plain mpiexec.
      # Explicit node/host/rankfile is injected at runtime via LINSTABPY_MPI_RANKFILE.
      # NO hostnames or absolute paths here (public repo boundary).
      if [ -n "${PBS_NODEFILE:-}" ]; then
        launcher=(mpiexec -machinefile "$PBS_NODEFILE")
      else
        launcher=(mpiexec)
      fi
      ;;
    none)
      launcher=(mpiexec)
      ;;
    custom)
      if [ -z "${LINSTABPY_MPI_LAUNCHER:-}" ]; then
        echo "set LINSTABPY_MPI_LAUNCHER when LINSTABPY_MPI_PRESET=custom" >&2
        exit 2
      fi
      read -r -a launcher <<< "${LINSTABPY_MPI_LAUNCHER}"
      ;;
    *)
      echo "unknown LINSTABPY_MPI_PRESET: $MPI_PRESET" >&2
      echo "valid values: socket0, socket1, socket01, c9, none, custom" >&2
      exit 2
      ;;
  esac
fi

# Dry-run hook: print the exact launcher that the real exec below runs.
# The entry script is always case2.py; the case module is selected at runtime via
# the LINSTABPY_CASE_MODULE env var (exported above), NOT via the command line, so
# print case2.py here and surface the module env separately to avoid implying a
# "$CASE_MODULE.py" file is launched.
if [ -n "${LINSTABPY_MPI_DRYRUN:-}" ]; then
  echo "LAUNCHER: ${launcher[*]} -n $MPI_PROCS $PYTHON_EXE case2.py (LINSTABPY_CASE_MODULE=$CASE_MODULE)"
  exit 0
fi

exec "${launcher[@]}" -n "$MPI_PROCS" "$PYTHON_EXE" case2.py
