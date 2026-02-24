#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LINSTAB2D_LOG_DIR:-artifacts/logs}"
mkdir -p "$LOG_DIR"
TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LINSTAB2D_LOG_FILE:-$LOG_DIR/example_6_hamada_${TS}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[hamada] log file: $LOG_FILE"
SCRIPT_START_TS="$(date +%s)"
SCRIPT_START_HUMAN="$(date +"%Y-%m-%d %H:%M:%S %Z")"
echo "[hamada] start: $SCRIPT_START_HUMAN"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[hamada] missing .venv/bin/python. Run install first."
  exit 1
fi

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

echo "[hamada] using logical cores: $CORES"
if [[ "$MPI_PROCS" -ne 1 ]]; then
  echo "[hamada] requested MPI processes: $MPI_PROCS"
  echo "[hamada] forcing MPI processes to 1 (current SciPy->PETSc path is single-rank CSR)."
  MPI_PROCS=1
fi
echo "[hamada] using MPI processes: $MPI_PROCS"
echo "[hamada] using threaded cores (OMP/BLAS): $CORES"
echo "[hamada] cleaning generated Hamada artifacts"

rm -f \
  linstab2d/examples/example_6_hamada.npz \
  linstab2d/examples/example_6_hamada_preview.png \
  linstab2d/examples/example_6_hamada_vs_matlab_spectrum.png \
  linstab2d/examples/example_6_hamada_vs_matlab_mode7.png \
  artifacts/hamada_spectrum_parity.json \
  artifacts/hamada_diagnostics_step1_4.json \
  artifacts/hamada_mapping_sweep.json

echo "[hamada] recomputing example_6_hamada"
if ! command -v mpiexec >/dev/null 2>&1; then
  echo "[hamada] mpiexec not found; install MPI to run PETSc/SLEPc in parallel."
  exit 1
fi
echo "[hamada] launching solver (heartbeat every ${HEARTBEAT_SEC}s)"
START_TS="$(date +%s)"
if [[ "${#PETSC_VERBOSE_OPTS[@]}" -gt 0 ]]; then
  mpiexec -n "$MPI_PROCS" .venv/bin/python -u -c "from linstab2d.examples.example_6_hamada import compute; compute()" "${PETSC_VERBOSE_OPTS[@]}" &
else
  mpiexec -n "$MPI_PROCS" .venv/bin/python -u -c "from linstab2d.examples.example_6_hamada import compute; compute()" &
fi
SOLVER_PID=$!
while kill -0 "$SOLVER_PID" 2>/dev/null; do
  NOW_TS="$(date +%s)"
  ELAPSED="$((NOW_TS - START_TS))"
  PY_LINE="$(ps -ax -o pid=,etime=,%cpu=,%mem=,state=,command= | rg 'linstab2d\\.examples\\.example_6_hamada' | head -n 1 || true)"
  if [[ -n "$PY_LINE" ]]; then
    echo "[hamada] running t=${ELAPSED}s | $PY_LINE"
  else
    echo "[hamada] running t=${ELAPSED}s | solver pid=${SOLVER_PID}"
  fi
  sleep "$HEARTBEAT_SEC"
done
wait "$SOLVER_PID"
echo "[hamada] solver finished in $(( $(date +%s) - START_TS ))s"

echo "[hamada] writing spectrum parity artifact"
.venv/bin/python tools/parity/check_hamada_spectrum.py \
  --python-npz linstab2d/examples/example_6_hamada.npz \
  --matlab-spectrum hamada_test/MATLAB_results/spectrum.mat \
  --artifact artifacts/hamada_spectrum_parity.json

echo "[hamada] generating figures"
.venv/bin/python - <<'PYFIG'
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

root = Path(".")
npz_path = root / "linstab2d/examples/example_6_hamada.npz"
mat_spec_path = root / "hamada_test/MATLAB_results/spectrum.mat"
mat_mode_path = root / "hamada_test/MATLAB_results/eigenvector_7.mat"

data = np.load(npz_path, allow_pickle = False)
omega_py = np.asarray(data["omega"], dtype=complex).reshape(-1)
omega_target = complex(np.asarray(data["omega_target"]).reshape(-1)[0])

omega_mat_raw = np.asarray(loadmat(mat_spec_path)["omega"])
if omega_mat_raw.ndim == 2 and omega_mat_raw.shape[0] == omega_mat_raw.shape[1]:
    omega_mat = np.diag(omega_mat_raw).astype(complex)
else:
    omega_mat = omega_mat_raw.reshape(-1).astype(complex)

n = int(min(100, omega_py.size, omega_mat.size))
py_sel = omega_py[np.argsort(np.abs(omega_py - omega_target))[:n]]
mat_sel = omega_mat[np.argsort(np.abs(omega_mat - omega_target))[:n]]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(mat_sel.real, mat_sel.imag, s=16, label="MATLAB", alpha=0.8)
ax.scatter(py_sel.real, py_sel.imag, s=16, label="Python", alpha=0.8)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlabel("Re(omega)")
ax.set_ylabel("Im(omega)")
ax.set_title(f"Hamada spectrum parity (n={n})")
ax.legend()
fig.tight_layout()
fig.savefig(root / "linstab2d/examples/example_6_hamada_vs_matlab_spectrum.png", dpi=220)
plt.close(fig)

if "X" in data and "Y" in data and "U" in data:
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    U = np.asarray(data["U"], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4))
    c = ax.pcolormesh(X, Y, U, shading="auto")
    fig.colorbar(c, ax=ax, label="U")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Hamada baseflow preview (U)")
    fig.tight_layout()
    fig.savefig(root / "linstab2d/examples/example_6_hamada_preview.png", dpi=220)
    plt.close(fig)

direct = np.asarray(data["direct_modes"], dtype=complex)
idx_u = np.asarray(data["idx_u_j"], dtype=int)
mode_mat = np.asarray(loadmat(mat_mode_path)["u"], dtype=complex).reshape(-1)

mode_idx_mat = min(6, omega_mat.size - 1)
omega_ref = omega_mat[mode_idx_mat]
mode_idx_py = int(np.argmin(np.abs(omega_py - omega_ref)))
u_py = direct[:, mode_idx_py][idx_u].reshape(-1)

num = np.vdot(u_py, mode_mat)
if num != 0:
    u_py = u_py * np.exp(-1j * np.angle(num))

u_mat_n = mode_mat / max(np.linalg.norm(mode_mat), 1e-30)
u_py_n = u_py / max(np.linalg.norm(u_py), 1e-30)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(np.real(u_mat_n), linewidth=0.9)
axes[0].set_ylabel("Re(u)")
axes[0].set_title("MATLAB mode 7 (normalized)")
axes[1].plot(np.real(u_py_n), linewidth=0.9)
axes[1].set_ylabel("Re(u)")
axes[1].set_title(f"Python nearest mode ({mode_idx_py + 1})")
axes[1].set_xlabel("grid dof index")
fig.tight_layout()
fig.savefig(root / "linstab2d/examples/example_6_hamada_vs_matlab_mode7.png", dpi=220)
plt.close(fig)
PYFIG

echo "[hamada] done"
SCRIPT_END_TS="$(date +%s)"
SCRIPT_END_HUMAN="$(date +"%Y-%m-%d %H:%M:%S %Z")"
SCRIPT_ELAPSED="$((SCRIPT_END_TS - SCRIPT_START_TS))"
echo "[hamada] end: $SCRIPT_END_HUMAN"
echo "[hamada] total elapsed: ${SCRIPT_ELAPSED}s | cores=${CORES} | mpi=${MPI_PROCS}"
