#!/usr/bin/env bash
set -euo pipefail

# Ziming_case runner: LNSE resolvent/stability from precomputed .npy baseflow.
#
# Prerequisites:
#   - linstab2d installed (or repo with Ziming_case at <repo>/Ziming_case)
#   - Ziming_case/baseflow/ containing x.npy, y.npy, rho.npy, u.npy, v.npy, w.npy, T.npy, sponge.npy
#
# Usage:
#   From repo root:  ./Ziming_case/run_case.sh
#   Or from anywhere: /path/to/linstab2d/Ziming_case/run_case.sh
#
# Optional env:
#   LINSTAB2D_MPI_PROCS  - number of MPI ranks (default: all cores, or 1 if no MPI)
#   LINSTAB2D_PYTHON    - Python executable (default: .venv/bin/python if present, else python3)
#   LINSTAB2D_MPI_EXTRA - extra args for mpiexec/mpirun (e.g. "--mca pmix_gds ^shmem" if you see PMIX gds/shmem errors)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CASE_DIR_NAME="$(basename "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Case directory relative to repo root (so baseflow is at $CASE_DIR_REL/baseflow/)
CASE_DIR_REL="$CASE_DIR_NAME"

# Python
if [[ -n "${LINSTAB2D_PYTHON:-}" ]]; then
  PYTHON="$LINSTAB2D_PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Ensure PETSc/SLEPc shared libraries built by install.sh are visible even when
# we don't 'source .venv/bin/activate' (install.sh normally sets LD_LIBRARY_PATH
# in the venv activation script, but this runner bypasses that).
if [[ -z "${PETSC_DIR:-}" || -z "${SLEPC_DIR:-}" ]]; then
  if [[ -x "$PYTHON" ]]; then
    mapfile -t _PETSCSLEPC_DIRS < <("$PYTHON" - <<'PY'
import petsc, slepc
print(petsc.get_petsc_dir())
print(slepc.get_slepc_dir())
PY
    ) || _PETSCSLEPC_DIRS=()
    if [[ ${#_PETSCSLEPC_DIRS[@]} -ge 2 ]]; then
      PETSC_DIR="${PETSC_DIR:-${_PETSCSLEPC_DIRS[0]}}"
      SLEPC_DIR="${SLEPC_DIR:-${_PETSCSLEPC_DIRS[1]}}"
      export PETSC_DIR SLEPC_DIR
    fi
  fi
fi

if [[ -n "${PETSC_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${PETSC_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${SLEPC_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${SLEPC_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

# MPI
MPI_PROCS="${LINSTAB2D_MPI_PROCS:-1}"
MPI_LAUNCHER=""
if command -v mpiexec >/dev/null 2>&1; then
  MPI_LAUNCHER="mpiexec"
elif command -v mpirun >/dev/null 2>&1; then
  MPI_LAUNCHER="mpirun"
fi

# Use all cores if MPI available and LINSTAB2D_MPI_PROCS not set
if [[ -z "${LINSTAB2D_MPI_PROCS:-}" && -n "$MPI_LAUNCHER" ]]; then
  N="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  if [[ -z "$N" ]] && command -v sysctl >/dev/null 2>&1; then
    N="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
  fi
  if [[ -n "$N" && "$N" =~ ^[0-9]+$ ]]; then
    MPI_PROCS="$N"
  fi
fi

# Check baseflow exists
BASEFLOW_DIR="$ROOT_DIR/$CASE_DIR_REL/baseflow"
if [[ ! -d "$BASEFLOW_DIR" ]]; then
  echo "[Ziming_case] ERROR: baseflow directory not found: $BASEFLOW_DIR"
  exit 1
fi
for f in x y rho u v w T sponge; do
  if [[ ! -f "$BASEFLOW_DIR/${f}.npy" ]]; then
    echo "[Ziming_case] ERROR: missing $BASEFLOW_DIR/${f}.npy"
    exit 1
  fi
done

echo "[Ziming_case] ROOT_DIR=$ROOT_DIR"
echo "[Ziming_case] CASE_DIR=$CASE_DIR_REL"
echo "[Ziming_case] PYTHON=$PYTHON"
echo "[Ziming_case] MPI_LAUNCHER=${MPI_LAUNCHER:-none} MPI_PROCS=$MPI_PROCS"

CMD=()
if [[ -n "$MPI_LAUNCHER" && "$MPI_PROCS" -gt 1 ]]; then
  CMD+=("$MPI_LAUNCHER" -np "$MPI_PROCS")
  # Work around PMIX "gds/shmem" not found (Open MPI env mismatch): use hash gds only
  if [[ -z "${LINSTAB2D_MPI_EXTRA:-}" ]]; then
    CMD+=("--mca" "pmix_gds" "^shmem")
  else
    IFS=' ' read -r -a MPI_EXTRA_ARRAY <<< "${LINSTAB2D_MPI_EXTRA}"
    CMD+=("${MPI_EXTRA_ARRAY[@]}")
  fi
fi
CMD+=("$PYTHON" -u -m Ziming_case.run_case --case-dir "$CASE_DIR_REL")

echo "[Ziming_case] running: ${CMD[*]}"
exec "${CMD[@]}"
