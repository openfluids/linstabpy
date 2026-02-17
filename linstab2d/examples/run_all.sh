#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG="$SCRIPT_DIR/run_all.log"
PYTHON="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: $PYTHON not found. Create a venv first (see install.sh)."
    exit 1
fi

# ── Thread control ───────────────────────────────────────────
# MUMPS (direct solver): PyPI wheels ship without OpenMP.
#   Source builds may have it — controlled by OMP_NUM_THREADS.
# BLAS  (NumPy/SciPy matrix ops): multi-threaded by default.
#   macOS Accelerate → VECLIB_MAXIMUM_THREADS
#   OpenBLAS (Linux)  → OPENBLAS_NUM_THREADS
#   Intel MKL         → MKL_NUM_THREADS
#
# Override: NTHREADS=4 ./run_all.sh
NTHREADS="${NTHREADS:-1}"
export OMP_NUM_THREADS="$NTHREADS"
export VECLIB_MAXIMUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"

# Clean cached outputs
rm -f "$SCRIPT_DIR"/*.npz "$SCRIPT_DIR"/*.png "$SCRIPT_DIR"/*.log

# Duplicate all output to terminal + log file
exec > >(tee "$LOG") 2>&1

cd "$REPO_ROOT" || { echo "Cannot cd to $REPO_ROOT"; exit 1; }

EXAMPLES=(
    example_1_blasius
    example_2_jet
    example_3_streaky_bl
    example_4_cavity
    example_5_paraboloid
)

# ── Environment info ─────────────────────────────────────────
"$PYTHON" -c "
import sys, os, platform
import numpy, scipy, matplotlib
import petsc4py; petsc4py.init()
from petsc4py import PETSc
import slepc4py; slepc4py.init()
from slepc4py import SLEPc
from mpi4py import MPI

def _ver(t): return '.'.join(map(str, t))

# MUMPS version from bundled header
_mumps_ver = 'unknown'
try:
    import pathlib, re
    _hdr = pathlib.Path(petsc4py.__file__).resolve().parent
    _hdr = _hdr.parent / 'petsc' / 'include' / 'zmumps_c.h'
    _m = re.search(r'MUMPS_VERSION \"([\d.]+)\"', _hdr.read_text())
    if _m: _mumps_ver = _m.group(1)
except Exception:
    pass

# BLAS backend (Accelerate / OpenBLAS / MKL)
_blas = 'unknown'
try:
    _cfg = numpy.show_config(mode='dicts')
    _blas = _cfg.get('Build Dependencies', {}).get('blas', {}).get('name', 'unknown')
except Exception:
    pass

print('========================================')
print(' ENVIRONMENT')
print('========================================')
print(f'  Platform     {platform.system()} {platform.machine()}')
print(f'  Python       {sys.version.split()[0]}')
print(f'  NumPy        {numpy.__version__}')
print(f'  SciPy        {scipy.__version__}')
print(f'  Matplotlib   {matplotlib.__version__}')
print(f'  PETSc        {_ver(PETSc.Sys.getVersion())}  (scalar: {PETSc.ScalarType.__name__})')
print(f'  SLEPc        {_ver(SLEPc.Sys.getVersion())}')
print(f'  MUMPS        {_mumps_ver}')
import mpi4py
_mpi_lib = MPI.Get_library_version().split(chr(10))[0].split()
_mpi_lib = f'{_mpi_lib[0]} {_mpi_lib[-1]}' if _mpi_lib else 'unknown'
print(f'  mpi4py       {mpi4py.__version__}  (MPI {MPI.Get_version()[0]}.{MPI.Get_version()[1]}, {_mpi_lib})')
print(f'  MPI procs    {MPI.COMM_WORLD.Get_size()}')
print(f'  BLAS         {_blas}')
print(f'  Threads      {os.environ.get(\"OMP_NUM_THREADS\", \"unset\")}  (OMP_NUM_THREADS)')
print('========================================')
print()
"

total_t0=$(date +%s)
n_fail=0
declare -a results

for ex in "${EXAMPLES[@]}"; do
    echo "=== $ex ==="
    t0=$(date +%s)
    "$PYTHON" -m "linstab2d.examples.$ex"
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [[ $rc -eq 0 ]]; then
        results+=("PASS  $ex  (${dt}s)")
    else
        results+=("FAIL  $ex  (${dt}s, exit $rc)")
        (( n_fail++ ))
    fi
    echo "--- $ex: exit $rc, ${dt}s ---"
    echo ""
done

total_dt=$(( $(date +%s) - total_t0 ))

echo "=========================================="
echo " SUMMARY  (total ${total_dt}s)"
echo "=========================================="
for r in "${results[@]}"; do echo "  $r"; done

# Allow tee to drain (bash 3.2 doesn't wait for process substitutions)
sleep 0.1
exit "$n_fail"
