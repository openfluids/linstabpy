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
    echo "valid values: socket0, socket1, socket01, none, custom" >&2
    exit 2
    ;;
esac

exec "${launcher[@]}" -n "$MPI_PROCS" "$PYTHON_EXE" case2.py
