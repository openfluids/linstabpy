#!/usr/bin/env bash
# run_all_examples.sh — recompute all linstab2d examples and run the parity check.
#
# USAGE
#   ./run_all_examples.sh           # use cached NPZs when present
#   ./run_all_examples.sh --fresh   # delete all NPZs first, force full recompute
#
# NOTE: Always use .venv/bin/python — do NOT replace with 'uv run'.
#   petsc4py/slepc4py require a one-time patched install performed by install.sh.
#   uv run attempts to rebuild petsc4py from source on every call and fails on macOS.
#   See CLAUDE.md for details.
#
# On Linux HPC nodes, replace .venv/bin/python with the appropriate interpreter
# (e.g. the venv created by install.sh on the target system).

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO/.venv/bin/python"
EXAMPLES_DIR="$REPO/linstab2d/examples"

# ── Argument handling ──────────────────────────────────────────────────────────
FRESH=0
for arg in "$@"; do
  case "$arg" in
    --fresh) FRESH=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: $PYTHON not found or not executable." >&2
  echo "Run install.sh first to set up the virtual environment." >&2
  exit 1
fi

# ── Optionally wipe cached NPZs ───────────────────────────────────────────────
if [[ $FRESH -eq 1 ]]; then
  echo "==> --fresh: deleting cached NPZ artifacts"
  rm -f \
    "$EXAMPLES_DIR/example_0_curvilinear_validation.npz" \
    "$EXAMPLES_DIR/example_1_blasius.npz" \
    "$EXAMPLES_DIR/example_2_jet.npz" \
    "$EXAMPLES_DIR/example_3_streaky_bl.npz" \
    "$EXAMPLES_DIR/example_4_cavity.npz" \
    "$EXAMPLES_DIR/example_5_paraboloid.npz"
fi

# ── Run examples ──────────────────────────────────────────────────────────────
run_example() {
  local module="$1"
  local label="$2"
  echo ""
  echo "══════════════════════════════════════════════════════"
  echo "  $label"
  echo "══════════════════════════════════════════════════════"
  "$PYTHON" -m "$module"
}

run_example linstab2d.examples.example_0_curvilinear_validation "Example 0: curvilinear derivative validation"
run_example linstab2d.examples.example_1_blasius                "Example 1: Blasius boundary layer resolvent"
run_example linstab2d.examples.example_2_jet                    "Example 2: turbulent jet resolvent"
run_example linstab2d.examples.example_3_streaky_bl             "Example 3: streaky boundary layer (temporal + spatial)"
run_example linstab2d.examples.example_4_cavity                 "Example 4: cavity direct and adjoint eigenvalues"
run_example linstab2d.examples.example_5_paraboloid             "Example 5: paraboloid resolvent"

# ── Parity comparison ─────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Parity comparison (Python vs MATLAB)"
echo "══════════════════════════════════════════════════════"
"$PYTHON" tools/parity/compare_martini_examples.py

echo ""
echo "Done."
