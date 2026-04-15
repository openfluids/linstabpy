#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  cat >&2 <<'EOF'
usage: run.sh <config-jsonc>

Runs the shock_cleanup (Ducros sensor) experiment with the given config.
The experiment loads the Case 2 baseflow, applies the optional cleanup, and
executes the stability and/or resolvent analysis on both the raw and
cleaned baseflows for side-by-side comparison.

Examples:

  # Step 1: sensor only — fast, calibrate ducros_threshold visually
  bash results/case2/ducros/run.sh results/case2/ducros/01_sensor_only.jsonc

  # Step 2: sensor + light filter — compare spectra with vs. without cleanup
  bash results/case2/ducros/run.sh results/case2/ducros/02_sensor_filter.jsonc

  # Parallel (4 ranks, no rankfile — portable)
  LINSTAB2D_MPI_PROCS=4 LINSTAB2D_MPI_PRESET=none \
    bash results/case2/ducros/run.sh results/case2/ducros/02_sensor_filter.jsonc

  # Parallel (10 ranks pinned to socket 0 on this machine)
  LINSTAB2D_MPI_PROCS=10 LINSTAB2D_MPI_PRESET=socket0 \
    bash results/case2/ducros/run.sh results/case2/ducros/02_sensor_filter.jsonc

Env vars (all optional):
  LINSTAB2D_MPI_PROCS=<int>              rank count; default 1 (serial).
  LINSTAB2D_MPI_PRESET=<preset>          one of: none, socket0, socket1,
                                         socket01, custom. Default: none.
  LINSTAB2D_MPI_LAUNCHER="<cmd>"         required when preset=custom.
  LINSTAB2D_STAGE_HEARTBEAT_SECONDS=<s>  progress heartbeat; default 30.

Important path note:
  The legacy runner resolves <config.output_dir> relative to results/case2/legacy/.
  The JSONCs in results/case2/ducros/ already account for that and write into
  results/case2/ducros/output/<case>/.
EOF
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
CASE_DIR="$ROOT/results/case2"
PYTHON_EXE="$ROOT/.venv/bin/python"
RUNNER="$CASE_DIR/legacy/run_shock_cleanup_experiment.py"

CONFIG_JSONC="$(realpath "$1")"

if [ ! -f "$CONFIG_JSONC" ]; then
  echo "config not found: $CONFIG_JSONC" >&2
  exit 2
fi
if [ ! -f "$RUNNER" ]; then
  echo "runner not found: $RUNNER" >&2
  exit 2
fi
if [ ! -x "$PYTHON_EXE" ]; then
  echo "venv python not found: $PYTHON_EXE (run install.sh)" >&2
  exit 2
fi

MPI_PROCS="${LINSTAB2D_MPI_PROCS:-1}"
MPI_PRESET="${LINSTAB2D_MPI_PRESET:-none}"
HEARTBEAT_SECONDS="${LINSTAB2D_STAGE_HEARTBEAT_SECONDS:-30}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export LINSTAB2D_CASE2_SHOCK_EXPERIMENT_CONFIG="$CONFIG_JSONC"
export LINSTAB2D_STAGE_HEARTBEAT_SECONDS="$HEARTBEAT_SECONDS"

echo "[ducros-run] config=$CONFIG_JSONC"
echo "[ducros-run] mpi_procs=$MPI_PROCS"
echo "[ducros-run] mpi_preset=$MPI_PRESET"

cd "$CASE_DIR"

if [ "$MPI_PROCS" -le 1 ]; then
  exec "$PYTHON_EXE" "$RUNNER"
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
    if [ -z "${LINSTAB2D_MPI_LAUNCHER:-}" ]; then
      echo "set LINSTAB2D_MPI_LAUNCHER when LINSTAB2D_MPI_PRESET=custom" >&2
      exit 2
    fi
    read -r -a launcher <<< "${LINSTAB2D_MPI_LAUNCHER}"
    ;;
  *)
    echo "unknown LINSTAB2D_MPI_PRESET: $MPI_PRESET" >&2
    echo "valid values: socket0, socket1, socket01, none, custom" >&2
    exit 2
    ;;
esac

exec "${launcher[@]}" -n "$MPI_PROCS" "$PYTHON_EXE" "$RUNNER"
