#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
LOG="$SCRIPT_DIR/run_parity.log"
STRICT_PARITY="${STRICT_PARITY:-0}"
STRICT_MAX_ASSEMBLY_S_PER_MNNZ="${STRICT_MAX_ASSEMBLY_S_PER_MNNZ:-5.0}"
STRICT_MAX_LINEAR_SOLVE_SECONDS="${STRICT_MAX_LINEAR_SOLVE_SECONDS:-0.2}"
STRICT_MAX_CASE_RUNTIME_SECONDS="${STRICT_MAX_CASE_RUNTIME_SECONDS:-1.5}"
STRICT_MAX_RSS_GROWTH_BYTES="${STRICT_MAX_RSS_GROWTH_BYTES:-120000000}"
STRICT_MAX_RSS_BYTES_PER_MNNZ="${STRICT_MAX_RSS_BYTES_PER_MNNZ:-2000000000}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: $PYTHON not found. Create a venv first (see install.sh)."
    exit 1
fi

exec > >(tee "$LOG") 2>&1

echo "========================================"
echo " PARITY RUN"
echo "========================================"
echo "  Repo           $REPO_ROOT"
echo "  Python         $PYTHON"
echo "  Strict mode    $STRICT_PARITY"
echo "  Log            $LOG"
echo "========================================"
echo ""

cd "$REPO_ROOT" || { echo "Cannot cd to $REPO_ROOT"; exit 1; }

n_fail=0

run_step() {
    local label="$1"
    shift
    echo "=== $label ==="
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "--- $label FAILED (exit $rc) ---"
        (( n_fail++ ))
    else
        echo "--- $label OK ---"
    fi
    echo ""
}

run_step "Examples" "$SCRIPT_DIR/run_all.sh"

if [[ "$STRICT_PARITY" == "1" ]]; then
    run_step "Parity matrices gate" "$PYTHON" tools/parity/check_parity_matrices.py
    run_step "Symbolic parity gate" "$PYTHON" tools/parity/test_symbolic_equation_parity.py
    run_step "Numeric operator parity gate" "$PYTHON" tools/parity/compare_operator_construction.py --strict
    run_step "Perf/memory parity gate" \
        "$PYTHON" tools/parity/perf_memory_operator_parity.py \
        --max-assembly-s-per-mnnz "$STRICT_MAX_ASSEMBLY_S_PER_MNNZ" \
        --max-linear-solve-seconds "$STRICT_MAX_LINEAR_SOLVE_SECONDS" \
        --max-case-runtime-seconds "$STRICT_MAX_CASE_RUNTIME_SECONDS" \
        --max-rss-growth-bytes "$STRICT_MAX_RSS_GROWTH_BYTES" \
        --max-rss-bytes-per-mnnz "$STRICT_MAX_RSS_BYTES_PER_MNNZ"
else
    run_step "Parity matrices report" "$PYTHON" tools/parity/check_parity_matrices.py --allow-incomplete
    run_step "Symbolic parity report" "$PYTHON" tools/parity/test_symbolic_equation_parity.py --allow-fail
    run_step "Numeric operator parity report" "$PYTHON" tools/parity/compare_operator_construction.py --allow-fail
    run_step "Perf/memory parity report" "$PYTHON" tools/parity/perf_memory_operator_parity.py --allow-failures
fi

echo "========================================"
echo " PARITY SUMMARY"
echo "========================================"
echo "  failures = $n_fail"
echo "========================================"

sleep 0.1
exit "$n_fail"
