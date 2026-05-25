#!/usr/bin/env bash
# run_all_examples.sh — recompute linstabpy examples and run the parity check.
#
# USAGE
#   ./run_all_examples.sh                                # use cached NPZs when present
#   ./run_all_examples.sh --fresh                        # delete cached NPZs first
#   ./run_all_examples.sh --clean                        # delete cached NPZs and PNGs first
#   ./run_all_examples.sh --example 0 --skip-parity      # run one example only
#   ./run_all_examples.sh --mpi --example 1 -n 10        # run one example under MPI
#   ./run_all_examples.sh --mpi --example 1s -n 10       # run the Blasius smoke benchmark
#   ./run_all_examples.sh --continue-on-error            # keep going after failures
#
# NOTE: Always use .venv/bin/python — do NOT replace with 'uv run'.
#   petsc4py/slepc4py require a one-time patched install performed by install.sh.
#   uv run attempts to rebuild petsc4py from source on every call and fails on macOS.
#   See CLAUDE.md for details.
#
# On Linux HPC nodes, point LINSTABPY_PYTHON at the target interpreter created
# by install.sh instead of editing this script.

set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_PYTHON="$REPO/.venv/bin/python"
if [[ -n "${LINSTABPY_PYTHON:-}" ]]; then
  PYTHON="$LINSTABPY_PYTHON"
elif [[ -n "${LINSTAB2D_PYTHON_BIN:-}" ]]; then
  PYTHON="$LINSTAB2D_PYTHON_BIN"
elif [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON="$DEFAULT_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
else
  PYTHON="$DEFAULT_PYTHON"
fi

EXAMPLES_DIR="$REPO/examples"
PARITY_SCRIPT="$REPO/scripts/validation/martini2024/tools/compare_martini_examples.py"
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

EXAMPLE_SPECS=(
  "0|example_0_curvilinear_validation|examples.example_0_curvilinear_validation|Example 0: curvilinear derivative validation"
  "1|example_1_blasius|examples.example_1_blasius|Example 1: Blasius boundary layer resolvent"
  "2|example_2_jet|examples.example_2_jet|Example 2: turbulent jet resolvent"
  "3|example_3_streaky_bl|examples.example_3_streaky_bl|Example 3: streaky boundary layer (temporal + spatial)"
  "4|example_4_cavity|examples.example_4_cavity|Example 4: cavity direct and adjoint eigenvalues"
  "5|example_5_paraboloid|examples.example_5_paraboloid|Example 5: paraboloid resolvent"
)

EXTRA_EXAMPLE_SPECS=(
  "1s|example_1_blasius_smoke|examples.example_1_blasius_smoke|Example 1 smoke: reduced Blasius benchmark"
)

usage() {
  cat <<EOF
Usage: ./run_all_examples.sh [options]

Options:
  --fresh               Delete cached example NPZ artifacts before running
  --clean               Delete cached example NPZ and PNG artifacts before running
  --example <selector>  Run only one example. Repeatable. Select by index,
                        stem, or module name (for example: 0, example_1_blasius,
                        examples.example_1_blasius_smoke)
  -m, --module <name>    Alias for --example; accepts full module names
  --mpi                 Run selected examples through an MPI launcher
  -n, --ranks <count>   MPI rank count. Implies --mpi
  --skip-parity         Skip the Python vs MATLAB parity comparison
  --continue-on-error   Run remaining examples even if one fails
  --list                Print available examples and exit
  -h, --help            Show this help

Environment:
  LINSTABPY_PYTHON      Override the Python interpreter path
  LINSTABPY_MPIEXEC     Override MPI launcher path/name
  LINSTABPY_MPI_RANKS   Default MPI rank count

Extra PETSc/SLEPc arguments may be passed after -- when exactly one example
is selected, for example:
  ./run_all_examples.sh --mpi --example 1 -n 1 -- -eps_monitor
EOF
}

list_examples() {
  echo "Available examples:"
  local spec idx stem module label
  for spec in "${EXAMPLE_SPECS[@]}" "${EXTRA_EXAMPLE_SPECS[@]}"; do
    IFS='|' read -r idx stem module label <<<"$spec"
    printf '  %s  %-32s %-38s %s\n' "$idx" "$stem" "$module" "$label"
  done
}

format_seconds() {
  local total="$1"
  local minutes=$((total / 60))
  local seconds=$((total % 60))
  printf "%02dm%02ds" "$minutes" "$seconds"
}

contains_module() {
  local needle="$1"
  local item
  for item in "${SELECTED_MODULES[@]:-}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

select_example() {
  local selector="$1"
  local spec idx stem module label matched=0
  for spec in "${EXAMPLE_SPECS[@]}" "${EXTRA_EXAMPLE_SPECS[@]}"; do
    IFS='|' read -r idx stem module label <<<"$spec"
    if [[ "$selector" == "$idx" || "$selector" == "$stem" || "$selector" == "$module" ]]; then
      matched=1
      if ! contains_module "$module"; then
        SELECTED_MODULES+=("$module")
        SELECTED_LABELS+=("$label")
        SELECTED_STEMS+=("$stem")
      fi
      break
    fi
  done

  if [[ $matched -eq 0 ]]; then
    echo "ERROR: unknown example selector '$selector'." >&2
    echo "Run ./run_all_examples.sh --list to see valid selectors." >&2
    exit 1
  fi
}

require_readable_file() {
  local path="$1"
  local label="$2"
  if [[ ! -r "$path" ]]; then
    echo "ERROR: missing $label: $path" >&2
    exit 1
  fi
}

delete_example_artifacts() {
  local stem="$1"
  find "$EXAMPLES_DIR" -maxdepth 1 -type f \
    \( -name "${stem}.npz" -o -name "${stem}.png" -o -name "${stem}_*.png" \) \
    -print -delete
}

env_or_unset() {
  local name="$1"
  local value="${!name-}"
  if [[ -n "$value" ]]; then
    printf '%s=%s\n' "$name" "$value"
  else
    printf '%s=<unset>\n' "$name"
  fi
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

probe_mpi_launch() {
  if has_cmd timeout; then
    timeout -k 2 8 "$@" >/dev/null 2>&1
  else
    "$@" >/dev/null 2>&1
  fi
}

intel_mpi_root() {
  if [[ -n "${I_MPI_ROOT:-}" && -d "${I_MPI_ROOT}" ]]; then
    printf '%s\n' "${I_MPI_ROOT%/}"
    return 0
  fi
  return 1
}

resolve_launcher_candidate() {
  local candidate="$1"
  if [[ -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  if has_cmd "$candidate"; then
    command -v "$candidate"
    return 0
  fi
  return 1
}

detect_physical_cores() {
  local cores=0
  if has_cmd lscpu; then
    local sockets cores_per_socket
    sockets="$(lscpu | awk '/^Socket\(s\):/{print $2}')"
    cores_per_socket="$(lscpu | awk '/^Core\(s\) per socket:/{print $4}')"
    if [[ -n "$sockets" && -n "$cores_per_socket" ]]; then
      cores=$((sockets * cores_per_socket))
    fi
  fi
  if [[ "$cores" -eq 0 && -f /proc/cpuinfo ]]; then
    cores="$(awk '/^processor/{count++} END{print count + 0}' /proc/cpuinfo)"
  fi
  if [[ "$cores" -eq 0 ]] && has_cmd sysctl; then
    cores="$(sysctl -n hw.physicalcpu 2>/dev/null || printf '0')"
  fi
  if ! [[ "$cores" =~ ^[0-9]+$ ]] || [[ "$cores" -lt 1 ]]; then
    cores=1
  fi
  printf '%d' "$cores"
}

find_mpiexec() {
  local candidate resolved root
  local candidates=()
  if root="$(intel_mpi_root)"; then
    candidates+=("$root/bin/mpiexec" "$root/bin/mpirun")
  fi
  candidates+=(mpiexec.mpich mpiexec.hydra mpiexec.openmpi mpirun.openmpi mpiexec mpirun)
  for candidate in "${candidates[@]}"; do
    resolved="$(resolve_launcher_candidate "$candidate" || true)"
    [[ -n "$resolved" ]] || continue
    if probe_mpi_launch "$resolved" -n 1 /bin/true; then
      printf '%s\n' "$resolved"
      return 0
    fi
  done
  return 1
}

configure_mpi_runtime() {
  [[ "$MPI_MODE" -eq 1 ]] || return 0

  if [[ -z "$MPIEXEC" ]]; then
    if ! MPIEXEC="$(find_mpiexec)"; then
      echo "ERROR: no working MPI launcher found." >&2
      echo "Set LINSTABPY_MPIEXEC or install an mpiexec/mpirun implementation." >&2
      exit 1
    fi
  fi

  if [[ -z "$MPI_RANKS" ]]; then
    local physical_cores
    physical_cores="$(detect_physical_cores)"
    MPI_RANKS=$((physical_cores / 2))
    if [[ "$MPI_RANKS" -lt 1 ]]; then
      MPI_RANKS=1
    fi
  fi

  if ! [[ "$MPI_RANKS" =~ ^[0-9]+$ ]] || [[ "$MPI_RANKS" -lt 1 ]]; then
    echo "ERROR: invalid MPI rank count: $MPI_RANKS" >&2
    exit 1
  fi

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

  MPIEXEC_ENV=()
  if [[ "$MPI_RANKS" -gt 1 ]] && ! probe_mpi_launch "$MPIEXEC" -n "$MPI_RANKS" /bin/true; then
    case "$(basename "$MPIEXEC")" in
      mpiexec.mpich|mpiexec.hydra|mpirun.mpich)
        if probe_mpi_launch env HYDRA_BOOTSTRAP=fork "$MPIEXEC" -n "$MPI_RANKS" /bin/true; then
          MPIEXEC_ENV=(HYDRA_BOOTSTRAP=fork)
        else
          echo "ERROR: $MPIEXEC failed the -n $MPI_RANKS local probe." >&2
          echo "Tried HYDRA_BOOTSTRAP=fork as well; check local MPI launcher configuration." >&2
          exit 1
        fi
        ;;
      *)
        echo "ERROR: $MPIEXEC failed the -n $MPI_RANKS local probe." >&2
        exit 1
        ;;
    esac
  fi
}

print_lscpu_summary() {
  if ! command -v lscpu >/dev/null 2>&1; then
    echo "  lscpu: unavailable"
    return
  fi

  LC_ALL=C lscpu | awk -F: '
    BEGIN {
      keys["Model name"] = 1
      keys["CPU(s)"] = 1
      keys["On-line CPU(s) list"] = 1
      keys["Thread(s) per core"] = 1
      keys["Core(s) per socket"] = 1
      keys["Socket(s)"] = 1
      keys["NUMA node(s)"] = 1
    }
    $1 in keys {
      key = $1
      value = $2
      sub(/^[ \t]+/, "", value)
      printf "  %-24s %s\n", key ":", value
    }
  '
}

print_thread_environment() {
  echo "Thread/process environment:"
  local name
  for name in \
    OMP_NUM_THREADS \
    OMP_DYNAMIC \
    OMP_PROC_BIND \
    OMP_PLACES \
    MKL_NUM_THREADS \
    MKL_DYNAMIC \
    OPENBLAS_NUM_THREADS \
    BLIS_NUM_THREADS \
    NUMEXPR_NUM_THREADS \
    VECLIB_MAXIMUM_THREADS \
    PETSC_DIR \
    PETSC_ARCH; do
    printf '  %s\n' "$(env_or_unset "$name")"
  done
}

print_python_runtime_summary() {
  "$PYTHON" - <<'PY'
import importlib
import os
import platform
import sys

print("Python numeric runtime:")
print(f"  executable: {sys.executable}")
print(f"  version:    {platform.python_version()}")

def version_line(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"{module_name}: unavailable ({exc.__class__.__name__})"
    return f"{module_name}: {getattr(module, '__version__', 'unknown')}"

for name in ("numpy", "scipy", "petsc4py", "slepc4py"):
    print(f"  {version_line(name)}")

try:
    import numpy as np

    # Force BLAS to initialize before asking threadpoolctl for loaded pools.
    np.ones((1, 1)) @ np.ones((1, 1))
except Exception:
    np = None

if np is not None:
    print("  NumPy BLAS/LAPACK build:")
    try:
        numpy_config = np.__config__.show(mode="dicts")
    except TypeError:
        numpy_config = {}
    except Exception as exc:
        print(f"    unavailable ({exc.__class__.__name__})")
        numpy_config = {}

    build_deps = numpy_config.get("Build Dependencies", {}) if isinstance(numpy_config, dict) else {}
    for libname in ("blas", "lapack"):
        details = build_deps.get(libname)
        if not isinstance(details, dict):
            print(f"    {libname}: unavailable")
            continue
        name = details.get("name", "unknown")
        version = details.get("version", "unknown")
        found = details.get("found", "unknown")
        print(f"    {libname}: name={name}, version={version}, found={found}")
        openblas_config = details.get("openblas configuration")
        if openblas_config:
            print(f"      openblas configuration: {openblas_config}")

try:
    from threadpoolctl import threadpool_info
except Exception as exc:
    print(f"  threadpoolctl: unavailable ({exc.__class__.__name__})")
else:
    pools = threadpool_info()
    if not pools:
        print("  threadpools: none detected")
    else:
        print("  threadpools:")
        for pool in pools:
            user_api = pool.get("user_api", "unknown")
            internal_api = pool.get("internal_api", "unknown")
            num_threads = pool.get("num_threads", "unknown")
            prefix = pool.get("prefix", "unknown")
            version = pool.get("version", "unknown")
            filepath = os.path.basename(pool.get("filepath", "") or "unknown")
            print(
                "    "
                f"user_api={user_api}, internal_api={internal_api}, "
                f"threads={num_threads}, prefix={prefix}, version={version}, lib={filepath}"
            )
PY
}

print_system_settings() {
  echo ""
  echo "System settings:"
  echo "  host:                    $(hostname 2>/dev/null || echo unknown)"
  echo "  kernel:                  $(uname -srvmo 2>/dev/null || uname -a)"
  echo "  timestamp:               $(date -Iseconds)"
  if [[ $MPI_MODE -eq 1 ]]; then
    echo "  runner process model:    MPI; $MPI_RANKS Python ranks per example"
    echo "  MPI launcher:            $MPIEXEC"
    if [[ ${#MPIEXEC_ENV[@]} -gt 0 ]]; then
      echo "  MPI launcher env:        ${MPIEXEC_ENV[*]}"
    fi
  else
    echo "  runner process model:    serial; one Python process per example"
  fi
  echo "  selected example jobs:   ${#SELECTED_MODULES[@]}"
  if command -v nproc >/dev/null 2>&1; then
    echo "  available CPUs (nproc):  $(nproc)"
    if nproc --all >/dev/null 2>&1; then
      echo "  total CPUs (nproc all):  $(nproc --all)"
    fi
  else
    echo "  available CPUs (nproc):  unavailable"
  fi
  if command -v taskset >/dev/null 2>&1; then
    taskset -pc "$$" 2>/dev/null | sed 's/^/  CPU affinity:           /' || true
  fi
  echo "CPU topology:"
  print_lscpu_summary
  print_thread_environment
  print_python_runtime_summary
}

FRESH=0
CLEAN=0
SKIP_PARITY=0
CONTINUE_ON_ERROR=0
MPI_MODE=0
MPIEXEC="${LINSTABPY_MPIEXEC:-${LINSTAB2D_MPIEXEC:-}}"
MPI_RANKS="${LINSTABPY_MPI_RANKS:-${LINSTAB2D_MPI_RANKS:-}}"
MPIEXEC_ENV=()
EXTRA_ARGS=()
SELECTED_MODULES=()
SELECTED_LABELS=()
SELECTED_STEMS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)
      FRESH=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --skip-parity)
      SKIP_PARITY=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --example)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --example requires a selector." >&2
        exit 1
      fi
      select_example "$2"
      shift 2
      ;;
    -m|--module)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: $1 requires an example module or selector." >&2
        exit 1
      fi
      select_example "$2"
      shift 2
      ;;
    --mpi)
      MPI_MODE=1
      shift
      ;;
    -n|--ranks)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: $1 requires a positive rank count." >&2
        exit 1
      fi
      MPI_MODE=1
      MPI_RANKS="$2"
      shift 2
      ;;
    --list)
      list_examples
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: $PYTHON not found or not executable." >&2
  echo "Run install.sh first, activate the target virtual environment, or set LINSTABPY_PYTHON." >&2
  exit 1
fi

require_readable_file "$PARITY_SCRIPT" "parity driver"

if [[ ${#SELECTED_MODULES[@]} -eq 0 ]]; then
  for spec in "${EXAMPLE_SPECS[@]}"; do
    IFS='|' read -r _ stem module label <<<"$spec"
    SELECTED_MODULES+=("$module")
    SELECTED_LABELS+=("$label")
    SELECTED_STEMS+=("$stem")
  done
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 && ${#SELECTED_MODULES[@]} -ne 1 ]]; then
  echo "ERROR: arguments after -- require exactly one selected example." >&2
  exit 1
fi

configure_mpi_runtime

echo "Repo root: $REPO"
echo "Python:    $PYTHON"
echo "Examples:  ${#SELECTED_MODULES[@]} selected"
if [[ $MPI_MODE -eq 1 ]]; then
  echo "MPI:       $MPIEXEC -n $MPI_RANKS"
  if [[ ${#MPIEXEC_ENV[@]} -gt 0 ]]; then
    echo "MPI env:   ${MPIEXEC_ENV[*]}"
  fi
fi
if [[ $SKIP_PARITY -eq 1 ]]; then
  echo "Parity:    skipped"
elif [[ $MPI_MODE -eq 1 ]]; then
  echo "Parity:    skipped in MPI mode"
fi
print_system_settings

if [[ $FRESH -eq 1 ]]; then
  echo "==> --fresh: deleting cached NPZ artifacts"
  for stem in "${SELECTED_STEMS[@]}"; do
    rm -f "$EXAMPLES_DIR/${stem}.npz"
  done
fi

if [[ $CLEAN -eq 1 ]]; then
  echo "==> --clean: deleting cached NPZ and PNG artifacts"
  for stem in "${SELECTED_STEMS[@]}"; do
    delete_example_artifacts "$stem"
  done
fi

cd "$REPO"

run_example() {
  local module="$1"
  local label="$2"
  local start_time elapsed rc

  echo ""
  echo "══════════════════════════════════════════════════════"
  echo "  $label"
  echo "══════════════════════════════════════════════════════"

  start_time=$SECONDS
  if [[ $MPI_MODE -eq 1 ]]; then
    if [[ ${#MPIEXEC_ENV[@]} -gt 0 ]]; then
      if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        env "${MPIEXEC_ENV[@]}" "$MPIEXEC" -n "$MPI_RANKS" "$PYTHON" -m "$module" "${EXTRA_ARGS[@]}"
      else
        env "${MPIEXEC_ENV[@]}" "$MPIEXEC" -n "$MPI_RANKS" "$PYTHON" -m "$module"
      fi
    else
      if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        "$MPIEXEC" -n "$MPI_RANKS" "$PYTHON" -m "$module" "${EXTRA_ARGS[@]}"
      else
        "$MPIEXEC" -n "$MPI_RANKS" "$PYTHON" -m "$module"
      fi
    fi
    rc=$?
  else
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      "$PYTHON" -m "$module" "${EXTRA_ARGS[@]}"
    else
      "$PYTHON" -m "$module"
    fi
    rc=$?
  fi

  if [[ $rc -eq 0 ]]; then
    elapsed=$((SECONDS - start_time))
    echo "Status: PASS ($(format_seconds "$elapsed"))"
    PASSED_LABELS+=("$label")
    return 0
  fi

  elapsed=$((SECONDS - start_time))
  echo "Status: FAIL ($(format_seconds "$elapsed"), exit $rc)" >&2
  FAILED_LABELS+=("$label (exit $rc)")
  return "$rc"
}

PASSED_LABELS=()
FAILED_LABELS=()
RUNNER_STATUS=0

for i in "${!SELECTED_MODULES[@]}"; do
  if ! run_example "${SELECTED_MODULES[$i]}" "${SELECTED_LABELS[$i]}"; then
    RUNNER_STATUS=1
    if [[ $CONTINUE_ON_ERROR -eq 0 ]]; then
      break
    fi
  fi
done

if [[ $RUNNER_STATUS -eq 0 && $SKIP_PARITY -eq 0 && $MPI_MODE -eq 0 ]]; then
  echo ""
  echo "══════════════════════════════════════════════════════"
  echo "  Parity comparison (Python vs MATLAB)"
  echo "══════════════════════════════════════════════════════"
  if "$PYTHON" "$PARITY_SCRIPT"; then
    echo "Status: PASS"
  else
    rc=$?
    echo "Status: FAIL (exit $rc)" >&2
    RUNNER_STATUS=1
  fi
fi

echo ""
echo "Summary:"
echo "  Passed examples: ${#PASSED_LABELS[@]}/${#SELECTED_MODULES[@]}"
if [[ ${#FAILED_LABELS[@]} -gt 0 ]]; then
  printf '  Failed examples: %s\n' "${FAILED_LABELS[*]}"
else
  echo "  Failed examples: none"
fi

echo ""
if [[ $RUNNER_STATUS -eq 0 ]]; then
  echo "Done."
else
  echo "Completed with failures." >&2
fi

exit "$RUNNER_STATUS"
