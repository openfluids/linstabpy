#!/usr/bin/env bash
# ── linstabpy benchmark runner ──────────────────────────────────────────────
# Runs a small validation benchmark with controlled MPI rank counts,
# single-thread enforcement, and cold-cache timing. Writes structured results
# to JSON.
#
# Usage:
#   bash benchmark.sh [machine_label]
#
# Example 0 (curvilinear validation) is pure NumPy — runs single-process.
# Example 1 runs the real Blasius case so install-time profiling
# stays practical even on slower nodes. Override to the full case with:
#   LINSTABPY_BENCHMARK_EX1_MODULE=examples.example_1_blasius
#   LINSTABPY_BENCHMARK_EX1_LABEL=example_1_blasius
# ────────────────────────────────────────────────────────────────────────────
set -uo pipefail
trap 'printf "\n[benchmark] interrupted\n"; exit 130' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PY_BIN="${VENV_DIR}/bin/python"
MACHINE_LABEL="${1:-$(hostname 2>/dev/null || echo unknown-host)}"
BENCH_ID="$(date -u +"%Y%m%dT%H%M%SZ")_${MACHINE_LABEL}"
RESULTS_DIR="${SCRIPT_DIR}/benchmark_runs/${BENCH_ID}"
export LC_NUMERIC=C

has_cmd() { command -v "$1" >/dev/null 2>&1; }

normalize_float_for_output() {
    local raw="${1:-0}"
    raw="${raw//,/.}"
    case "${raw}" in
        .*) raw="0${raw}" ;;
        -.*) raw="-0${raw#-}" ;;
    esac
    if printf '%s\n' "${raw}" | grep -Eq '^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$'; then
        printf '%s' "${raw}"
    else
        printf '0'
    fi
}

COL_OK='\033[1;32m'
COL_WARN='\033[1;33m'
COL_INFO='\033[1;34m'
COL_ERR='\033[1;31m'
COL_RST='\033[0m'

# ── Preflight checks ───────────────────────────────────────────────────────
if [ ! -x "${PY_BIN}" ]; then
    printf "${COL_ERR}[benchmark] .venv not found. Run install.sh first.${COL_RST}\n"
    exit 1
fi

if [ ! -d "${SCRIPT_DIR}/examples" ]; then
    printf "${COL_ERR}[benchmark] examples/ directory not found.${COL_RST}\n"
    exit 1
fi

# ── Detect physical cores ──────────────────────────────────────────────────
detect_physical_cores() {
    local cores=0
    if has_cmd lscpu; then
        local sockets threads_per_core cores_per_socket
        sockets="$(lscpu | awk '/^Socket\(s\):/{print $2}')"
        cores_per_socket="$(lscpu | awk '/^Core\(s\) per socket:/{print $4}')"
        threads_per_core="$(lscpu | awk '/^Thread\(s\) per core:/{print $4}')"
        if [ -n "${sockets}" ] && [ -n "${cores_per_socket}" ]; then
            cores=$((sockets * cores_per_socket))
        fi
    fi
    if [ "${cores}" -eq 0 ] && [ -f /proc/cpuinfo ]; then
        cores="$(grep -c '^processor' /proc/cpuinfo)"
    fi
    if [ "${cores}" -eq 0 ]; then
        cores=1
    fi
    printf '%d' "${cores}"
}

PHYS_CORES="$(detect_physical_cores)"
HALF_CORES=$(( PHYS_CORES / 2 ))
[ "${HALF_CORES}" -lt 1 ] && HALF_CORES=1

mkdir -p "${RESULTS_DIR}"

printf "\n${COL_INFO}=== linstabpy benchmark ===${COL_RST}\n"
printf "  Machine:         %s\n" "${MACHINE_LABEL}"
printf "  Benchmark ID:    %s\n" "${BENCH_ID}"
printf "  Physical cores:  %d\n" "${PHYS_CORES}"
printf "  Half cores:      %d\n" "${HALF_CORES}"
printf "  Results dir:     %s\n" "${RESULTS_DIR}"
echo ""

EX1_MODULE="${LINSTABPY_BENCHMARK_EX1_MODULE:-examples.example_1_blasius}"
EX1_LABEL="${LINSTABPY_BENCHMARK_EX1_LABEL:-example_1_blasius}"
printf "  Example 1 case:  %s (%s)\n\n" "${EX1_LABEL}" "${EX1_MODULE}"

# ── Find mpiexec ────────────────────────────────────────────────────────────
MPIEXEC=""
for candidate in mpiexec.mpich mpiexec.hydra mpiexec.openmpi mpiexec; do
    has_cmd "${candidate}" || continue
    resolved="$(command -v "${candidate}")"
    if has_cmd timeout; then
        timeout -k 2 5 "${resolved}" -n 1 /bin/true >/dev/null 2>&1 && MPIEXEC="${resolved}" && break
    else
        "${resolved}" -n 1 /bin/true >/dev/null 2>&1 && MPIEXEC="${resolved}" && break
    fi
done

if [ -z "${MPIEXEC}" ]; then
    printf "${COL_ERR}[benchmark] No working mpiexec found. Cannot run parallel benchmarks.${COL_RST}\n"
    exit 1
fi
printf "  mpiexec:         %s\n\n" "${MPIEXEC}"
export LINSTABPY_MPIEXEC="${MPIEXEC}"
export LINSTABPY_PYTHON_BIN="${PY_BIN}"

# ── Thread pinning (single thread per rank) ─────────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

# ── JSON writer ─────────────────────────────────────────────────────────────
RESULTS_JSON="${RESULTS_DIR}/benchmark_results.json"
RESULTS_CSV="${RESULTS_DIR}/benchmark_summary.csv"

# Start the JSON array
printf '[\n' > "${RESULTS_JSON}"
# CSV header
printf 'example,variant,ranks,wall_seconds,peak_rss_kb,exit_code\n' > "${RESULTS_CSV}"

FIRST_ENTRY=1

append_result() {
    local example="$1" variant="$2" ranks="$3" wall="$4" peak_rss="$5" exit_code="$6"

    if [ "${FIRST_ENTRY}" -eq 0 ]; then
        printf ',\n' >> "${RESULTS_JSON}"
    fi
    FIRST_ENTRY=0

    cat >> "${RESULTS_JSON}" <<JSONEOF
  {
    "benchmark_id": "${BENCH_ID}",
    "machine": "${MACHINE_LABEL}",
    "example": "${example}",
    "variant": "${variant}",
    "ranks": ${ranks},
    "omp_threads": 1,
    "physical_cores": ${PHYS_CORES},
    "wall_seconds": ${wall},
    "peak_rss_kb": ${peak_rss},
    "exit_code": ${exit_code},
    "timestamp_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  }
JSONEOF

    printf '%s,%s,%d,%s,%d,%d\n' \
        "${example}" "${variant}" "${ranks}" "${wall}" "${peak_rss}" "${exit_code}" \
        >> "${RESULTS_CSV}"
}

# ── Run a single benchmark case ────────────────────────────────────────────
# Usage: run_case <example_module> <label> <variant_name> <n_ranks>
run_case() {
    local module="$1" label="$2" variant="$3" n_ranks="$4"
    local npz_glob="${SCRIPT_DIR}/examples/${label}*.npz"
    local log_file="${RESULTS_DIR}/${label}_${variant}.log"
    local time_file="${RESULTS_DIR}/${label}_${variant}.time"

    printf "${COL_INFO}[benchmark]${COL_RST} %s / %s (ranks=%d)\n" "${label}" "${variant}" "${n_ranks}"

    # Cold run: delete cached .npz files
    for f in ${npz_glob}; do
        [ -f "${f}" ] && rm -f "${f}" && printf "  cleared cache: %s\n" "$(basename "${f}")"
    done

    local -a cmd
    if [ "${n_ranks}" -eq 1 ]; then
        cmd=( "${PY_BIN}" -m "${module}" )
    else
        cmd=( bash "${SCRIPT_DIR}/run_all_examples.sh" --mpi --skip-parity -n "${n_ranks}" -m "${module}" )
    fi

    local wall_start wall_end wall_sec exit_code peak_rss
    wall_start="$(date +%s%N 2>/dev/null || date +%s)"

    if has_cmd /usr/bin/time; then
        /usr/bin/time -v -o "${time_file}" "${cmd[@]}" > "${log_file}" 2>&1
        exit_code=$?
    else
        "${cmd[@]}" > "${log_file}" 2>&1
        exit_code=$?
        printf "time_command_not_available\n" > "${time_file}"
    fi

    wall_end="$(date +%s%N 2>/dev/null || date +%s)"

    # Wall time in seconds (nanosecond precision if available)
    if [ ${#wall_start} -gt 10 ]; then
        wall_sec="$(echo "scale=2; (${wall_end} - ${wall_start}) / 1000000000" | bc 2>/dev/null || echo "0")"
    else
        wall_sec="$((wall_end - wall_start))"
    fi
    wall_sec="$(normalize_float_for_output "${wall_sec}")"

    # Peak RSS from /usr/bin/time -v output
    peak_rss=0
    if [ -f "${time_file}" ] && grep -q "Maximum resident" "${time_file}"; then
        peak_rss="$(grep 'Maximum resident' "${time_file}" | awk '{print $NF}')"
    fi

    if [ "${exit_code}" -eq 0 ]; then
        printf "  ${COL_OK}OK${COL_RST}  wall=%ss  peak_rss=%d KB\n" "${wall_sec}" "${peak_rss}"
    else
        printf "  ${COL_ERR}FAIL${COL_RST} (exit=%d)  wall=%ss  — see %s\n" "${exit_code}" "${wall_sec}" "${log_file}"
    fi

    append_result "${label}" "${variant}" "${n_ranks}" "${wall_sec}" "${peak_rss}" "${exit_code}"
}

# ── Execute benchmark suite ─────────────────────────────────────────────────

printf "${COL_INFO}--- Example 0: Curvilinear validation (single-process) ---${COL_RST}\n"
run_case "examples.example_0_curvilinear_validation" "example_0_curvilinear_validation" "single" 1
echo ""

printf "${COL_INFO}--- Example 1: Blasius PETSc (MPI parallel) ---${COL_RST}\n"
run_case "${EX1_MODULE}" "${EX1_LABEL}" "all_cores" "${PHYS_CORES}"
echo ""
run_case "${EX1_MODULE}" "${EX1_LABEL}" "half_cores" "${HALF_CORES}"
echo ""

# ── Close JSON array ────────────────────────────────────────────────────────
printf '\n]\n' >> "${RESULTS_JSON}"

# ── Summary ─────────────────────────────────────────────────────────────────
printf "${COL_INFO}=== Benchmark complete ===${COL_RST}\n"
printf "  Results:  %s\n" "${RESULTS_JSON}"
printf "  Summary:  %s\n" "${RESULTS_CSV}"
printf "  Logs:     %s/*.log\n" "${RESULTS_DIR}"
echo ""
printf "  CSV contents:\n"
cat "${RESULTS_CSV}"
echo ""
