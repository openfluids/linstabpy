#!/usr/bin/env bash
# Collect and save machine/software configuration for performance comparison.
# Usage: ./save_machine_config.sh [output_file]
# Default output: machine_config_YYYYMMDD_HHMMSS.txt in current directory.

set -euo pipefail

TS="$(date +"%Y%m%d_%H%M%S")"
OUT="${1:-machine_config_${TS}.txt}"
ROOT_DIR="${LINSTAB2D_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"

exec 3>&1
exec > "$OUT"
exec 2>&1

echo "=============================================="
echo "  Machine configuration report"
echo "  Generated: $(date -Iseconds 2>/dev/null || date)"
echo "  Output file: $OUT"
echo "=============================================="
echo ""

# ---- Host / OS ----
echo "========== HOST & OS =========="
echo "hostname: $(hostname 2>/dev/null || echo 'n/a')"
echo "uname -a: $(uname -a 2>/dev/null || echo 'n/a')"
echo "OS: $(uname -s 2>/dev/null)"
echo "Arch: $(uname -m 2>/dev/null)"
echo "Kernel: $(uname -r 2>/dev/null)"
echo ""

# ---- CPU ----
echo "========== CPU =========="
if command -v lscpu >/dev/null 2>&1; then
  lscpu 2>/dev/null || true
else
  echo "lscpu not available"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    echo "Logical CPUs: $(sysctl -n hw.logicalcpu 2>/dev/null || echo 'n/a')"
    echo "Physical CPUs: $(sysctl -n hw.physicalcpu 2>/dev/null || echo 'n/a')"
  fi
  echo "Online processors (getconf): $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 'n/a')"
fi
echo ""

# ---- Memory ----
echo "========== MEMORY =========="
if command -v free >/dev/null 2>&1; then
  free -h
elif [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Physical memory:"
  sysctl hw.memsize 2>/dev/null | awk '{ printf "  Total: %.2f GiB\n", $2/1024/1024/1024 }'
fi
echo ""

# ---- GPU (optional) ----
echo "========== GPU =========="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi 2>/dev/null || echo "nvidia-smi failed"
else
  echo "nvidia-smi not found (no NVIDIA GPU or driver not in PATH)"
fi
echo ""

# ---- MATLAB ----
echo "========== MATLAB =========="
MATLAB_PATH=""
if command -v matlab >/dev/null 2>&1; then
  MATLAB_PATH="$(command -v matlab)"
  echo "matlab executable: $MATLAB_PATH"
  echo "matlab version (batch):"
  matlab -batch "disp(['Version: ' version]); disp(['Release: ' version('-release')]);" 2>/dev/null || echo "  (matlab -batch failed or not available)"
else
  echo "matlab not found in PATH"
  if [[ -x "/usr/local/MATLAB/R2017a/bin/matlab" ]]; then
    echo "Found: /usr/local/MATLAB/R2017a/bin/matlab"
    MATLAB_PATH="/usr/local/MATLAB/R2017a/bin/matlab"
  fi
  for d in /usr/local/MATLAB/*/bin/matlab; do
    if [[ -x "$d" ]]; then
      echo "Found: $d"
      MATLAB_PATH="$d"
    fi
  done 2>/dev/null || true
fi
echo ""

# ---- Python & venv ----
echo "========== PYTHON & VENV =========="
PYTHON_EXE=""
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_EXE="$ROOT_DIR/.venv/bin/python"
  echo "Using repo venv: $PYTHON_EXE"
  echo "Venv path: $ROOT_DIR/.venv"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_EXE="${VIRTUAL_ENV}/bin/python"
  echo "Using VIRTUAL_ENV: $VIRTUAL_ENV"
  echo "Python: $PYTHON_EXE"
else
  PYTHON_EXE="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  echo "System/default Python: ${PYTHON_EXE:-not found}"
fi

if [[ -n "$PYTHON_EXE" && -x "$PYTHON_EXE" ]]; then
  echo ""
  echo "Python version:"
  "$PYTHON_EXE" --version 2>/dev/null || true
  echo ""
  echo "Key packages (pip list):"
  "$PYTHON_EXE" -m pip list 2>/dev/null | grep -iE "linstab|numpy|scipy|petsc|mpi|slepc" || "$PYTHON_EXE" -m pip list 2>/dev/null || echo "  (pip list failed)"
  echo ""
  echo "Full pip freeze (for reproducibility):"
  "$PYTHON_EXE" -m pip freeze 2>/dev/null || echo "  (pip freeze failed)"
fi
echo ""

# ---- MPI ----
echo "========== MPI =========="
if command -v mpiexec >/dev/null 2>&1; then
  echo "mpiexec: $(command -v mpiexec)"
  mpiexec --version 2>&1 | head -5 || true
elif command -v mpirun >/dev/null 2>&1; then
  echo "mpirun: $(command -v mpirun)"
  mpirun --version 2>&1 | head -5 || true
else
  echo "mpiexec/mpirun not found in PATH"
fi
echo ""

# ---- PETSc (if available via Python) ----
echo "========== PETSc (from Python) =========="
if [[ -n "$PYTHON_EXE" && -x "$PYTHON_EXE" ]]; then
  "$PYTHON_EXE" -c "
try:
  from petsc4py import PETSc
  PETSc.initialize()
  print('petsc4py: OK')
  print('PETSc version:', PETSc.getVersion())
  PETSc.finalize()
except Exception as e:
  print('petsc4py:', e)
" 2>/dev/null || echo "  (petsc4py not available or failed)"
fi
echo ""

echo "=============================================="
echo "  End of report"
echo "=============================================="

exec 1>&3 3>&-
echo "Configuration written to: $OUT"
