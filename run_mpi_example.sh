#!/usr/bin/env bash
set -euo pipefail

# Run one example module under MPI with a conservative Linux runtime envelope:
# one thread per rank for OpenMP / BLAS, and a launcher that actually works.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BIN="${LINSTAB2D_PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODULE=""
RANKS="${LINSTAB2D_MPI_RANKS:-}"

has_cmd() { command -v "$1" >/dev/null 2>&1; }

probe_mpi_launch() {
    if has_cmd timeout; then
        timeout -k 2 8 "$@" >/dev/null 2>&1
    else
        "$@" >/dev/null 2>&1
    fi
}

intel_mpi_root() {
    if [ -n "${I_MPI_ROOT:-}" ] && [ -d "${I_MPI_ROOT}" ]; then
        printf '%s\n' "${I_MPI_ROOT%/}"
        return 0
    fi
    return 1
}

resolve_launcher_candidate() {
    local candidate="$1"
    if [ -x "${candidate}" ]; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    if has_cmd "${candidate}"; then
        command -v "${candidate}"
        return 0
    fi
    return 1
}

usage() {
    cat <<'EOF'
Usage:
  bash run_mpi_example.sh -m examples.example_1_blasius_smoke
  bash run_mpi_example.sh -n 10 -m examples.example_3_streaky_bl
  bash run_mpi_example.sh -n 1 -m examples.example_1_blasius_smoke -- -eps_monitor

Notes:
  - Uses .venv/bin/python by default.
  - Exports OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1
    unless the caller already set them.
  - If -n/--ranks is omitted, defaults to half of the physical cores.
EOF
}

detect_physical_cores() {
    local cores=0
    if has_cmd lscpu; then
        local sockets cores_per_socket
        sockets="$(lscpu | awk '/^Socket\(s\):/{print $2}')"
        cores_per_socket="$(lscpu | awk '/^Core\(s\) per socket:/{print $4}')"
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

find_mpiexec() {
    local candidate resolved root
    local candidates=()
    if root="$(intel_mpi_root)"; then
        candidates+=("${root}/bin/mpiexec" "${root}/bin/mpirun")
    fi
    candidates+=(mpiexec.mpich mpiexec.hydra mpiexec.openmpi mpirun.openmpi mpiexec mpirun)
    for candidate in "${candidates[@]}"; do
        resolved="$(resolve_launcher_candidate "${candidate}" || true)"
        [ -n "${resolved}" ] || continue
        if has_cmd timeout; then
            if timeout -k 2 5 "${resolved}" -n 1 /bin/true >/dev/null 2>&1; then
                printf '%s\n' "${resolved}"
                return 0
            fi
        else
            if "${resolved}" -n 1 /bin/true >/dev/null 2>&1; then
                printf '%s\n' "${resolved}"
                return 0
            fi
        fi
    done
    return 1
}

EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -n|--ranks)
            RANKS="${2:-}"
            shift 2
            ;;
        -m|--module)
            MODULE="${2:-}"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            if [ -z "${MODULE}" ]; then
                MODULE="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -z "${MODULE}" ]; then
    usage >&2
    exit 1
fi

if [ ! -x "${PY_BIN}" ]; then
    echo "[run_mpi_example] missing python runtime: ${PY_BIN}" >&2
    echo "[run_mpi_example] run 'bash install.sh' first or set LINSTAB2D_PYTHON_BIN" >&2
    exit 1
fi

MPIEXEC="${LINSTAB2D_MPIEXEC:-}"
if [ -z "${MPIEXEC}" ]; then
    if ! MPIEXEC="$(find_mpiexec)"; then
        echo "[run_mpi_example] no working mpiexec found" >&2
        exit 1
    fi
fi

if [ -z "${RANKS}" ]; then
    PHYS_CORES="$(detect_physical_cores)"
    RANKS=$(( PHYS_CORES / 2 ))
    if [ "${RANKS}" -lt 1 ]; then
        RANKS=1
    fi
fi

if ! [[ "${RANKS}" =~ ^[0-9]+$ ]] || [ "${RANKS}" -lt 1 ]; then
    echo "[run_mpi_example] invalid rank count: ${RANKS}" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

MPIEXEC_ENV=()
if [ "${RANKS}" -gt 1 ] && ! probe_mpi_launch "${MPIEXEC}" -n "${RANKS}" /bin/true; then
    case "$(basename "${MPIEXEC}")" in
        mpiexec.mpich|mpiexec.hydra|mpirun.mpich)
            if probe_mpi_launch env HYDRA_BOOTSTRAP=fork "${MPIEXEC}" -n "${RANKS}" /bin/true; then
                MPIEXEC_ENV=(HYDRA_BOOTSTRAP=fork)
            else
                echo "[run_mpi_example] ${MPIEXEC} failed the -n ${RANKS} local probe" >&2
                echo "[run_mpi_example] Tried HYDRA_BOOTSTRAP=fork as well; see local MPI launcher configuration." >&2
                exit 1
            fi
            ;;
        *)
            echo "[run_mpi_example] ${MPIEXEC} failed the -n ${RANKS} local probe" >&2
            exit 1
            ;;
    esac
fi

echo "[run_mpi_example] mpiexec=${MPIEXEC}"
echo "[run_mpi_example] ranks=${RANKS}"
echo "[run_mpi_example] python=${PY_BIN}"
echo "[run_mpi_example] module=${MODULE}"
echo "[run_mpi_example] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[run_mpi_example] MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "[run_mpi_example] OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
if [ "${#MPIEXEC_ENV[@]}" -gt 0 ]; then
    echo "[run_mpi_example] launcher_env=${MPIEXEC_ENV[*]}"
fi

if [ "${#MPIEXEC_ENV[@]}" -gt 0 ]; then
    exec env "${MPIEXEC_ENV[@]}" "${MPIEXEC}" -n "${RANKS}" "${PY_BIN}" -m "${MODULE}" "${EXTRA_ARGS[@]}"
fi

exec "${MPIEXEC}" -n "${RANKS}" "${PY_BIN}" -m "${MODULE}" "${EXTRA_ARGS[@]}"
