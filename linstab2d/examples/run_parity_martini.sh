#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
LOG="$SCRIPT_DIR/run_parity_martini.log"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: $PYTHON not found. Create a venv first (see install.sh)."
    exit 1
fi

NTHREADS="${NTHREADS:-1}"
export OMP_NUM_THREADS="$NTHREADS"
export VECLIB_MAXIMUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"

exec > >(tee "$LOG") 2>&1

cd "$REPO_ROOT" || { echo "Cannot cd to $REPO_ROOT"; exit 1; }

EXAMPLE_MODULES=(
    linstab2d.examples.example_0_curvilinear_validation
    linstab2d.examples.example_1_blasius
    linstab2d.examples.example_2_jet
    linstab2d.examples.example_4_cavity
    linstab2d.examples.example_5_paraboloid
)

EXAMPLE_OUTPUTS=(
    linstab2d/examples/example_0_curvilinear_validation.npz
    linstab2d/examples/example_0_curvilinear_validation.png
    linstab2d/examples/example_1_blasius.npz
    linstab2d/examples/example_1_blasius.png
    linstab2d/examples/example_2_jet.npz
    linstab2d/examples/example_2_jet.png
    linstab2d/examples/example_4_cavity.npz
    linstab2d/examples/example_4_cavity.png
    linstab2d/examples/example_4_cavity_direct.png
    linstab2d/examples/example_4_cavity_adjoint.png
    linstab2d/examples/example_5_paraboloid.npz
    linstab2d/examples/example_5_paraboloid.png
)

run_step() {
    local label="$1"
    shift
    echo "=== $label ==="
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "--- $label FAILED (exit $rc) ---"
        exit "$rc"
    fi
    echo "--- $label OK ---"
    echo ""
}

echo "========================================"
echo " MARTINI PARITY RUN"
echo "========================================"
echo "  Repo           $REPO_ROOT"
echo "  Python         $PYTHON"
echo "  Threads        $NTHREADS"
echo "  Log            $LOG"
echo "========================================"
echo ""

rm -f "${EXAMPLE_OUTPUTS[@]}"

for module in "${EXAMPLE_MODULES[@]}"; do
    run_step "$module" "$PYTHON" -m "$module"
done

run_step \
    "Extract Martini HDF5 bundles" \
    "$PYTHON" tools/parity/extract_martini_to_npz.py

run_step \
    "Compare Martini examples" \
    "$PYTHON" tools/parity/compare_martini_examples.py

echo "========================================"
echo " MARTINI PARITY COMPLETE"
echo "========================================"

sleep 0.1
exit 0
