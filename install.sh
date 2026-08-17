#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# linstabpy bootstrap installer
# =============================================================================
#
# Creates a Python virtual environment and installs all dependencies
# including PETSc/SLEPc with complex-scalar support and MUMPS direct solver.
#
# Usage:
#   bash install.sh                    # full install (no h5py by default)
#   bash install.sh --force            # remove .venv first, then reinstall everything
#   bash install.sh --force --latest   # reinstall with latest package releases
#   bash install.sh --stack gpu-strumpack --dry-run  # show isolated GPU-stack plan
#   LINSTABPY_BUILD_STACK=gpu-strumpack bash install.sh --dry-run
#   LINSTABPY_WITH_H5PY=1 bash install.sh   # also install h5py (needs HDF5 >= 1.10.7)
#   LINSTABPY_PYTHON_VERSION=3.14 bash install.sh --force --latest  # Python 3.14 + latest packages
#   LINSTABPY_MPI_STACK=intelmpi bash install.sh  # use an already-loaded Intel MPI stack
#   LINSTABPY_BLAS=mkl bash install.sh  # Linux BLAS: auto (default) | mkl | openblas | reference
#   LINSTABPY_AUTO_INSTALL_SYS_DEPS=1 bash install.sh  # allow package-manager installs
#   bash install.sh --check            # verify existing installation
#   bash install.sh --diagnose         # host preflight only; install nothing
#
# On Linux, missing system packages do not trigger sudo/package-manager calls by
# default. The script prints the suggested commands and exits so cluster/module
# environments can load the right toolchain first.
#
# If PETSc source configure fails with "Unable to run MPI program with /usr/bin/mpiexec":
#   LINSTABPY_PETSC_MPIEXEC=/usr/bin/mpiexec.mpich bash install.sh
#
# If mpi4py fails to import (e.g. ImportError: libmpi.so.40 while the system has MPICH):
#   The installer tries a wheel first, then rebuilds mpi4py from source automatically.
#   To force a source build: LINSTABPY_MPI4PY_NO_BINARY=1 bash install.sh
#
# If verification hangs after "mpi4py.MPI …" on MPICH: plain ``python`` can block in MPI_Init.
#   The script runs checks under ``mpiexec -n 1``. Override with LINSTABPY_VERIFY_MPIEXEC=/path/to/mpiexec.
#
# HYDRA_BOOTSTRAP=exec avoids some PETSc configure stalls but is not supported by all MPICH builds
# (error: unrecognized launcher: exec). This script applies it only while installing PETSc/SLEPc
# (not for mpiexec verification). Override with LINSTABPY_HYDRA_BOOTSTRAP=fork or
# LINSTABPY_HYDRA_BOOTSTRAP=  (empty, unset) if PETSc configure still misbehaves.
#
# The PETSc/SLEPc PyPI wheels embed absolute build-time paths in their config
# files and dylibs.  This script patches them automatically.
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_DIR="${ROOT_DIR}/.venv"
REQUESTED_STACK="${LINSTABPY_BUILD_STACK:-default}"
BUILD_STACK=""
VENV_DIR=""
VENV_DISPLAY=""
OS_NAME=$(uname -s)
PYTHON_VERSION="${LINSTABPY_PYTHON_VERSION:-3.13}"
INSTALL_MODE="install"
FORCE_REINSTALL=0
LATEST_PACKAGES=0
DRY_RUN=0
PETSC_VERSION="${LINSTABPY_PETSC_VERSION:-3.24.2}"
SLEPC_VERSION="${LINSTABPY_SLEPC_VERSION:-3.24.2}"
PETSC4PY_VERSION="${LINSTABPY_PETSC4PY_VERSION:-3.24.2}"
SLEPC4PY_VERSION="${LINSTABPY_SLEPC4PY_VERSION:-3.24.2}"

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$1"; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$1"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$1"; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }
mktemp_py() { mktemp "${TMPDIR:-/tmp}/linstabpy.XXXXXX"; }

usage() {
    sed -n '/^# Usage:$/,/^#[[:space:]]*$/p' "${BASH_SOURCE[0]}" | sed -e 's/^# //' -e 's/^#[[:space:]]*$//'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check)
            INSTALL_MODE="check"
            shift
            ;;
        --diagnose)
            INSTALL_MODE="diagnose"
            shift
            ;;
        --force)
            FORCE_REINSTALL=1
            shift
            ;;
        --latest)
            LATEST_PACKAGES=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --stack)
            if [ "$#" -lt 2 ]; then
                err "Missing value for --stack"
                usage
                exit 2
            fi
            REQUESTED_STACK="$2"
            shift 2
            ;;
        --stack=*)
            REQUESTED_STACK="${1#--stack=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown option: $1"
            usage
            exit 2
            ;;
    esac
done

resolve_build_stack() {
    case "${REQUESTED_STACK}" in
        ""|default|cpu)
            BUILD_STACK="default"
            VENV_DIR="${DEFAULT_VENV_DIR}"
            VENV_DISPLAY=".venv"
            ;;
        *[!A-Za-z0-9_.-]*|.*|-*|*-|*..*)
            err "Invalid build stack '${REQUESTED_STACK}'. Use letters, numbers, '.', '_', or '-', and do not start with '.' or '-'."
            exit 2
            ;;
        *)
            BUILD_STACK="${REQUESTED_STACK}"
            VENV_DIR="${ROOT_DIR}/.venv-${BUILD_STACK}"
            VENV_DISPLAY=".venv-${BUILD_STACK}"
            ;;
    esac
}

assert_stack_venv_is_safe() {
    if [ "${BUILD_STACK}" = "default" ]; then
        return 0
    fi
    if [ "${VENV_DIR}" = "${DEFAULT_VENV_DIR}" ]; then
        err "Refusing non-default stack '${BUILD_STACK}' because it resolves to the CPU .venv."
        exit 1
    fi
    # A pre-existing symlink (or otherwise aliased path) would let every later
    # write follow it into the CPU .venv; the string comparison above cannot
    # see that, so refuse aliased stack venvs before any use, not only before
    # removal.
    if [ -L "${VENV_DIR}" ]; then
        err "Refusing non-default stack '${BUILD_STACK}': ${VENV_DIR} is a symlink."
        exit 1
    fi
    if [ -d "${VENV_DIR}" ] && [ -d "${DEFAULT_VENV_DIR}" ]; then
        local resolved_stack resolved_default
        resolved_stack="$(cd "${VENV_DIR}" 2>/dev/null && pwd -P)" || resolved_stack=""
        resolved_default="$(cd "${DEFAULT_VENV_DIR}" 2>/dev/null && pwd -P)" || resolved_default=""
        if [ -n "${resolved_stack}" ] && [ "${resolved_stack}" = "${resolved_default}" ]; then
            err "Refusing non-default stack '${BUILD_STACK}': ${VENV_DIR} resolves to the CPU .venv."
            exit 1
        fi
    fi
}

remove_venv_dir() {
    local target="$1"
    if [ "${BUILD_STACK}" != "default" ] && [ "${target}" = "${DEFAULT_VENV_DIR}" ]; then
        err "Refusing to remove CPU .venv while building stack '${BUILD_STACK}'."
        exit 1
    fi
    rm -rf "${target}"
}

existing_venv_python_version() {
    if [ -d "${VENV_DIR}" ] && [ -x "${VENV_DIR}/bin/python" ]; then
        "${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true
    fi
}

venv_will_be_removed() {
    # Mirrors the real venv lifecycle below: check mode never removes; install
    # mode removes only an existing venv, either for --force or Python mismatch.
    local existing_py=""
    if [ "${INSTALL_MODE}" = "check" ] || [ "${INSTALL_MODE}" = "diagnose" ]; then
        return 1
    fi
    if [ ! -d "${VENV_DIR}" ]; then
        return 1
    fi
    if [ "${FORCE_REINSTALL}" = "1" ]; then
        return 0
    fi
    existing_py="$(existing_venv_python_version)"
    [ "${existing_py}" != "${PYTHON_VERSION}" ]
}

print_dry_run_plan() {
    assert_stack_venv_is_safe
    printf 'build_stack=%s\n' "${BUILD_STACK}"
    printf 'venv_dir=%s\n' "${VENV_DIR}"
    printf 'venv_display=%s\n' "${VENV_DISPLAY}"
    printf 'force=%s\n' "${FORCE_REINSTALL}"
    printf 'blas=%s\n' "${LINSTABPY_BLAS:-auto}"
    if [ "${BUILD_STACK}" = "default" ]; then
        printf 'petsc_configure_branch=cpu-mumps\n'
    else
        printf 'petsc_configure_branch=cuda-strumpack-design\n'
        printf 'protected_venv=%s\n' "${DEFAULT_VENV_DIR}"
    fi
    if venv_will_be_removed; then
        printf 'planned_remove=%s\n' "${VENV_DIR}"
    else
        printf 'planned_remove=none\n'
    fi
}

resolve_build_stack
assert_stack_venv_is_safe
if [ "${DRY_RUN}" = "1" ]; then
    print_dry_run_plan
    exit 0
fi

# Print a directory containing libmpi.so* / libmpich.so* (Open MPI, MPICH, Intel MPI)
# or return non-zero.
linstabpy_resolve_mpi_libdir() {
    local mpicc_bin=""
    local d=""
    local so=""
    local mpi_root=""
    # libmpi.so* covers Open MPI and Intel MPI; libmpich.so* covers MPICH on Debian/Ubuntu.
    local _has_mpi_so='ls "${1}"/libmpi.so* >/dev/null 2>&1 || ls "${1}"/libmpich.so* >/dev/null 2>&1'
    _dir_has_mpi_so() { ls "${1}"/libmpi.so* >/dev/null 2>&1 || ls "${1}"/libmpich.so* >/dev/null 2>&1; }
    if mpi_root="$(linux_intel_mpi_root 2>/dev/null || true)" && [ -n "${mpi_root}" ]; then
        for d in "${mpi_root}/lib/release" "${mpi_root}/lib"; do
            if [ -d "${d}" ] && _dir_has_mpi_so "${d}"; then
                printf '%s\n' "${d}"
                return 0
            fi
        done
    fi
    if mpicc_bin="$(linux_resolve_mpi_tool mpicc mpiicc mpiicx 2>/dev/null || true)" && [ -n "${mpicc_bin}" ]; then
        :
    elif has_cmd mpicc; then
        mpicc_bin="$(command -v mpicc)"
    elif [ -x /usr/bin/mpicc ]; then
        mpicc_bin="/usr/bin/mpicc"
    fi
    if [ -n "${mpicc_bin}" ]; then
        # Open MPI: --showme:libdir / --showme:link
        d="$("${mpicc_bin}" --showme:libdir 2>/dev/null || true)"
        if [ -z "${d}" ] || [ ! -d "${d}" ]; then
            d="$("${mpicc_bin}" --showme:link 2>/dev/null | tr ' ' '\n' | awk '/^-L/{print substr($0,3); exit}')"
        fi
        # MPICH / Intel MPI: -show (returns the underlying cc line with -L<dir>)
        if [ -z "${d}" ] || [ ! -d "${d}" ]; then
            d="$("${mpicc_bin}" -show 2>/dev/null | tr ' ' '\n' | awk '/^-L/{print substr($0,3); exit}')"
        fi
        # MPICH: -compile-info / -link-info
        if [ -z "${d}" ] || [ ! -d "${d}" ]; then
            d="$("${mpicc_bin}" -link-info 2>/dev/null | tr ' ' '\n' | awk '/^-L/{print substr($0,3); exit}')"
        fi
        d="$(printf '%s\n' "${d}" | sed -e 's|^"||' -e 's|"$||' -e "s|^'||" -e "s|'$||" -e 's|///*|/|g' -e 's|/$||')"
        if [ -n "${d}" ] && [ -d "${d}" ] && _dir_has_mpi_so "${d}"; then
            printf '%s\n' "${d}"
            return 0
        fi
    fi
    # Common install locations (Open MPI, MPICH, Intel oneAPI MPI).
    for _cand in \
        /usr/lib/x86_64-linux-gnu/openmpi/lib \
        /usr/lib/x86_64-linux-gnu \
        /usr/lib64/openmpi/lib \
        /usr/lib/openmpi/lib \
        /usr/lib/x86_64-linux-gnu/mpich \
        /usr/lib64/mpich/lib \
        /usr/lib/mpich/lib \
        /opt/intel/oneapi/mpi/latest/lib \
        /opt/intel/oneapi/mpi/latest/lib/release
    do
        if [ -d "${_cand}" ] && _dir_has_mpi_so "${_cand}"; then
            printf '%s\n' "${_cand}"
            return 0
        fi
    done
    # Intel oneAPI: glob versioned dirs as a last resort.
    for _cand in /opt/intel/oneapi/mpi/*/lib /opt/intel/oneapi/mpi/*/lib/release; do
        if [ -d "${_cand}" ] && _dir_has_mpi_so "${_cand}"; then
            printf '%s\n' "${_cand}"
            return 0
        fi
    done
    if has_cmd ldconfig; then
        so="$(ldconfig -p 2>/dev/null | awk '/lib(mpi|mpich)\.so\.[0-9]/{print $NF; exit}')"
        if [ -n "${so}" ] && [ -f "${so}" ]; then
            d="$(dirname "${so}")"
            d="$(printf '%s\n' "${d}" | sed -e 's|///*|/|g' -e 's|/$||')"
            printf '%s\n' "${d}"
            return 0
        fi
    fi
    return 1
}

# Run the verify Python snippet, filtering benign Hydra "cmd=finalize / Bad file
# descriptor" chatter that MPICH prints after the child process exits. Any other
# stderr output is forwarded untouched. Preserves the child's exit code.
#
# Usage: linstabpy_run_verify <py_bin> <heredoc_path>
linstabpy_run_verify() {
    if declare -p LINSTAB_VERIFY_MPIWRAP >/dev/null 2>&1 && [ "${#LINSTAB_VERIFY_MPIWRAP[@]}" -gt 0 ]; then
        linstabpy_run_verify_wrapped "$1" "$2" "${LINSTAB_VERIFY_MPIWRAP[@]}"
    else
        linstabpy_run_verify_wrapped "$1" "$2"
    fi
}

linstabpy_run_verify_wrapped() {
    local py_bin="$1"; shift
    local snippet_file="$1"; shift
    local _err _rc
    _err="$(mktemp)"
    set +e
    if [ "$#" -gt 0 ] && [[ "$1" == *=* ]]; then
        env "$@" "${py_bin}" "${snippet_file}" 2>"${_err}"
    else
        "$@" "${py_bin}" "${snippet_file}" 2>"${_err}"
    fi
    _rc=$?
    set -e
    grep -v -E '^\[cli_[0-9]+\]: write_line error|^system msg for write_line failure|^:cmd=finalize$|^:$' "${_err}" >&2 || true
    rm -f "${_err}"
    return "${_rc}"
}

ensure_reference_submodule() {
    [ -d "${ROOT_DIR}/.git" ] || return 0
    if ! has_cmd git; then
        warn "git not found; skipping submodule init (install git or sync baseflows without submodules)."
        return 0
    fi

    local submodule_path="src/linstabpy/data/baseflows"
    if ! git -C "${ROOT_DIR}" config --file .gitmodules --get-regexp '^submodule\..*\.path$' \
        | awk '{print $2}' | grep -Fxq "${submodule_path}"; then
        warn "Skipping baseflow submodule init; ${submodule_path} is not declared in .gitmodules."
        return 0
    fi

    if git -C "${ROOT_DIR}" submodule status -- "${submodule_path}" >/dev/null 2>&1; then
        ok "Reference baseflow submodule is available."
        return 0
    fi

    warn "Initializing MATLAB reference submodule..."
    git -C "${ROOT_DIR}" submodule update --init --recursive -- "${submodule_path}"
}

linux_wrap_venv_python() {
    [ "${OS_NAME}" = "Linux" ] || return 0

    local py_bin="${VENV_DIR}/bin/python"
    local py_real="${VENV_DIR}/bin/python-real"
    local real_target=""

    [ -e "${py_bin}" ] || return 0

    if [ -L "${py_bin}" ]; then
        real_target="$(readlink -f "${py_bin}")"
    elif [ -L "${py_real}" ]; then
        real_target="$(readlink -f "${py_real}")"
    else
        warn "Skipping Linux runtime wrapper; ${py_bin} is not a venv symlink."
        return 0
    fi
    if [ -z "${real_target}" ] || [ ! -x "${real_target}" ]; then
        warn "Could not resolve the base Python interpreter for Linux runtime wrapper."
        return 0
    fi

    rm -f "${py_real}"
    ln -s "${real_target}" "${py_real}"
    rm -f "${py_bin}"
    cat > "${py_bin}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# LINSTAB2D Linux runtime wrapper:
# PETSc wheels depend on bundled ScaLAPACK/MUMPS shared libraries that are not
# discoverable from a bare uv-created venv Python on this workstation.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_venv_dir="$(cd "${_script_dir}/.." && pwd)"
_site_packages="$(printf '%s\n' "${_venv_dir}"/lib/python*/site-packages | head -n 1)"
_petsc_lib="${_site_packages}/petsc/lib"
_slepc_lib="${_site_packages}/slepc/lib"
_ld_file="${_venv_dir}/linstabpy_ld_library_path"

# Prefer the install-time runtime manifest. This preserves an Intel MPI install
# even when /usr/bin/mpicc from Open MPI is also present on the host.
_ld_paths=()
if [ -f "${_ld_file}" ]; then
    while IFS= read -r _d; do
        [ -n "${_d}" ] && [ -d "${_d}" ] && _ld_paths+=("${_d}")
    done < "${_ld_file}"
fi

# Fallback for older installs without the manifest.
_mpicc=""
_mpi_ld=""
if [ "${#_ld_paths[@]}" -eq 0 ]; then
    if [ -n "${I_MPI_ROOT:-}" ] && [ -d "${I_MPI_ROOT}/lib/release" ] && ls "${I_MPI_ROOT}/lib/release"/libmpi.so* >/dev/null 2>&1; then
        _mpi_ld="${I_MPI_ROOT}/lib/release"
    elif command -v mpicc >/dev/null 2>&1; then
        _mpicc="$(command -v mpicc)"
    elif [ -x /usr/bin/mpicc ]; then
        _mpicc="/usr/bin/mpicc"
    fi
fi
if [ "${#_ld_paths[@]}" -eq 0 ] && [ -n "${_mpicc}" ]; then
    _d="$("${_mpicc}" --showme:libdir 2>/dev/null || true)"
    if [ -n "${_d}" ] && [ -d "${_d}" ] && ls "${_d}"/libmpi.so* >/dev/null 2>&1; then
        _mpi_ld="${_d}"
    fi
    if [ -z "${_mpi_ld}" ]; then
        _d="$("${_mpicc}" -show 2>/dev/null | tr ' ' '\n' | awk '/^-L/{print substr($0,3); exit}')"
        if [ -n "${_d}" ] && [ -d "${_d}" ] && ls "${_d}"/libmpi.so* >/dev/null 2>&1; then
            _mpi_ld="${_d}"
        fi
    fi
fi
if [ "${#_ld_paths[@]}" -eq 0 ] && [ -z "${_mpi_ld}" ]; then
    for _cand in \
        /usr/lib/x86_64-linux-gnu/openmpi/lib \
        /usr/lib/x86_64-linux-gnu \
        /usr/lib64/openmpi/lib \
        /usr/lib/openmpi/lib
    do
        if [ -d "${_cand}" ] && ls "${_cand}"/libmpi.so* >/dev/null 2>&1; then
            _mpi_ld="${_cand}"
            break
        fi
    done
fi
if [ "${#_ld_paths[@]}" -eq 0 ] && [ -z "${_mpi_ld}" ] && command -v ldconfig >/dev/null 2>&1; then
    _so="$(ldconfig -p 2>/dev/null | awk '/libmpi\.so\.[0-9]/{print $NF; exit}')"
    if [ -n "${_so}" ] && [ -f "${_so}" ]; then
        _mpi_ld="$(dirname "${_so}")"
    fi
fi

if [ "${#_ld_paths[@]}" -gt 0 ]; then
    _joined="$(IFS=:; printf '%s' "${_ld_paths[*]}")"
    export LD_LIBRARY_PATH="${_joined}:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="${_mpi_ld:+${_mpi_ld}:}${_petsc_lib}:${_slepc_lib}:${LD_LIBRARY_PATH:-}"
fi
exec -a "${_script_dir}/python" "${_script_dir}/python-real" "$@"
EOF
    chmod +x "${py_bin}"

    for link_name in python3 "python${PYTHON_VERSION}"; do
        ln -sfn python "${VENV_DIR}/bin/${link_name}"
    done

    ok "Installed Linux runtime wrapper for .venv/bin/python"
}

darwin_openblas_dylib() {
    [ "${OS_NAME}" = "Darwin" ] || return 1

    local prefix=""
    if has_cmd brew; then
        prefix="$(brew --prefix openblas 2>/dev/null || true)"
        if [ -n "${prefix}" ] && [ -r "${prefix}/lib/libopenblas.0.dylib" ]; then
            printf '%s\n' "${prefix}/lib/libopenblas.0.dylib"
            return 0
        fi
    fi

    for prefix in /opt/homebrew/opt/openblas /usr/local/opt/openblas; do
        if [ -r "${prefix}/lib/libopenblas.0.dylib" ]; then
            printf '%s\n' "${prefix}/lib/libopenblas.0.dylib"
            return 0
        fi
    done

    return 1
}

darwin_brew_prefix() {
    [ "${OS_NAME}" = "Darwin" ] || return 1

    local formula="$1"
    local prefix=""
    if has_cmd brew; then
        prefix="$(brew --prefix "${formula}" 2>/dev/null || true)"
        if [ -n "${prefix}" ] && [ -d "${prefix}" ]; then
            printf '%s\n' "${prefix}"
            return 0
        fi
    fi

    for prefix in "/opt/homebrew/opt/${formula}" "/usr/local/opt/${formula}"; do
        if [ -d "${prefix}" ]; then
            printf '%s\n' "${prefix}"
            return 0
        fi
    done

    return 1
}

darwin_append_petsc_linalg_options() {
    [ "${OS_NAME}" = "Darwin" ] || return 1

    local openblas_prefix=""
    local scalapack_prefix=""
    openblas_prefix="$(darwin_brew_prefix openblas || true)"
    scalapack_prefix="$(darwin_brew_prefix scalapack || true)"

    if [ -z "${openblas_prefix}" ] || [ -z "${scalapack_prefix}" ]; then
        warn "Homebrew OpenBLAS/ScaLAPACK not found; falling back to PETSc downloaded BLAS/LAPACK/ScaLAPACK."
        warn "For the robust macOS path, install them with: brew install openblas scalapack"
        return 1
    fi

    _PETSC_CONF_OPTS+=(
        "--with-blaslapack-lib=${openblas_prefix}/lib/libopenblas.dylib"
        "--with-blaslapack-include=${openblas_prefix}/include"
        "--with-scalapack-lib=${scalapack_prefix}/lib/libscalapack.dylib"
    )
    log "PETSc configure: using Homebrew OpenBLAS and ScaLAPACK."
    return 0
}

# Linux BLAS/LAPACK for PETSc, MUMPS and ScaLAPACK.
#
# PETSc's --download-fblaslapack is the reference Fortran BLAS: no blocking, no
# vector units. Measured 2026-08-17 on a Xeon Silver 4210R, 4 MPI ranks: MUMPS
# did 9.6e11 flops in 461 s, i.e. 0.5 GFlop/s per core, 5-20x below a tuned
# BLAS. Every MUMPS factor, every solve and every wall time built on it was
# that much slower than the hardware allows. So the reference BLAS is never
# the default here.
#
# LINSTABPY_BLAS selects the library:
#   auto       (default) mkl when MKLROOT points at a real install (after
#              `source /opt/intel/oneapi/setvars.sh`), otherwise openblas
#   mkl        Intel MKL from MKLROOT
#   openblas   OpenBLAS, downloaded and built by PETSc
#   reference  --download-fblaslapack, kept only for A/B checks
# ScaLAPACK is downloaded and built on top of whichever BLAS is chosen.
linux_append_petsc_linalg_options() {
    local choice="${LINSTABPY_BLAS:-auto}"
    if [ "${choice}" = "auto" ]; then
        if [ -n "${MKLROOT:-}" ] && [ -d "${MKLROOT}" ]; then
            choice="mkl"
        else
            choice="openblas"
        fi
    fi
    case "${choice}" in
        mkl)
            if [ -z "${MKLROOT:-}" ] || [ ! -d "${MKLROOT}" ]; then
                err "LINSTABPY_BLAS=mkl but MKLROOT is unset or not a directory. Source oneAPI setvars.sh first."
                exit 2
            fi
            _PETSC_CONF_OPTS+=(
                "--with-blaslapack-dir=${MKLROOT}"
                "--download-scalapack"
            )
            log "PETSc configure: BLAS/LAPACK = MKL at ${MKLROOT}; ScaLAPACK downloaded on top of it."
            ;;
        openblas)
            _PETSC_CONF_OPTS+=(
                "--download-openblas"
                "--download-scalapack"
            )
            log "PETSc configure: BLAS/LAPACK = OpenBLAS (downloaded by PETSc); ScaLAPACK downloaded."
            ;;
        reference)
            _PETSC_CONF_OPTS+=(
                "--download-fblaslapack"
                "--download-scalapack"
            )
            warn "PETSc configure: BLAS/LAPACK = reference fblaslapack. 5-20x slower than a tuned BLAS; A/B checks only."
            ;;
        *)
            err "LINSTABPY_BLAS must be auto, mkl, openblas or reference; got '${choice}'."
            exit 2
            ;;
    esac
    LINSTABPY_BLAS_CHOICE="${choice}"
}

darwin_wrap_venv_python() {
    [ "${OS_NAME}" = "Darwin" ] || return 0
    [ "${LINSTABPY_DARWIN_INSERT_OPENBLAS:-1}" = "1" ] || return 0

    local py_bin="${VENV_DIR}/bin/python"
    local py_real="${VENV_DIR}/bin/python-real"
    local openblas=""
    local real_target=""

    [ -e "${py_bin}" ] || return 0
    openblas="$(darwin_openblas_dylib || true)"
    if [ -z "${openblas}" ]; then
        warn "OpenBLAS dylib not found; skipping macOS runtime wrapper."
        return 0
    fi

    if [ -f "${py_bin}" ] && grep -q 'LINSTAB2D Darwin runtime wrapper' "${py_bin}" 2>/dev/null; then
        if [ -L "${py_real}" ]; then
            real_target="$(readlink "${py_real}")"
        elif [ -x "${py_real}" ]; then
            real_target="${py_real}"
        fi
    elif [ -L "${py_bin}" ]; then
        real_target="$(readlink "${py_bin}")"
        case "${real_target}" in
            /*) ;;
            *) real_target="$(cd "$(dirname "${py_bin}")" && cd "$(dirname "${real_target}")" && pwd)/$(basename "${real_target}")" ;;
        esac
    elif [ -L "${py_real}" ]; then
        real_target="$(readlink "${py_real}")"
    else
        warn "Skipping macOS runtime wrapper; ${py_bin} is not a venv symlink."
        return 0
    fi
    if [ -z "${real_target}" ] || [ ! -x "${real_target}" ]; then
        warn "Could not resolve the base Python interpreter for macOS runtime wrapper."
        return 0
    fi

    rm -f "${py_real}"
    ln -s "${real_target}" "${py_real}"
    rm -f "${py_bin}"
    cat > "${py_bin}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

# LINSTAB2D Darwin runtime wrapper:
# Python imports NumPy/Accelerate before PETSc/SLEPc. PETSc's downloaded
# ScaLAPACK can then bind LAPACK calls to Apple's Accelerate LAPACK, which
# crashes in iterative non-Hermitian EPS solves. Insert Homebrew OpenBLAS
# before Python starts so ScaLAPACK resolves LAPACK symbols consistently.
_script_dir="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
_openblas="${openblas}"
if [ -r "\${_openblas}" ]; then
    case ":\${DYLD_INSERT_LIBRARIES:-}:" in
        *:"\${_openblas}":*) ;;
        *) export DYLD_INSERT_LIBRARIES="\${_openblas}\${DYLD_INSERT_LIBRARIES:+:\${DYLD_INSERT_LIBRARIES}}" ;;
    esac
fi
exec -a "\${_script_dir}/python" "\${_script_dir}/python-real" "\$@"
EOF
    chmod +x "${py_bin}"

    for link_name in python3 "python${PYTHON_VERSION}"; do
        ln -sfn python "${VENV_DIR}/bin/${link_name}"
    done

    ok "Installed macOS runtime wrapper for .venv/bin/python"
}

linux_hdf5_available() {
    has_cmd h5cc && return 0
    has_cmd pkg-config && pkg-config --exists hdf5 2>/dev/null && return 0
    has_cmd pkg-config && pkg-config --exists hdf5-serial 2>/dev/null && return 0
    return 1
}

linux_intel_mpi_root() {
    if [ -n "${I_MPI_ROOT:-}" ] && [ -d "${I_MPI_ROOT}" ]; then
        printf '%s\n' "${I_MPI_ROOT%/}"
        return 0
    fi
    return 1
}

linux_intel_mpi_available() {
    linux_intel_mpi_root >/dev/null && return 0
    [ -x /opt/intel/oneapi/setvars.sh ] && return 0
    [ -x /opt/intel/oneapi/mpi/latest/bin/mpicc ] && return 0
    for _cand in /opt/intel/oneapi/mpi/*/bin/mpicc; do
        [ -x "${_cand}" ] && return 0
    done
    return 1
}

linux_cmd_looks_intel_mpi() {
    local cmd="$1"
    local resolved=""
    if [ -x "${cmd}" ]; then
        resolved="${cmd}"
    elif has_cmd "${cmd}"; then
        resolved="$(command -v "${cmd}")"
    else
        return 1
    fi

    case "${resolved}" in
        *intel*oneapi*mpi*|*oneapi*/mpi/*|*impi*)
            return 0
            ;;
    esac
    "${resolved}" -V 2>&1 | head -n5 | grep -qi 'Intel.*MPI\|Intel(R).*MPI' && return 0
    "${resolved}" --version 2>&1 | head -n5 | grep -qi 'Intel.*MPI\|Intel(R).*MPI' && return 0
    return 1
}

linux_mpi_stack_preference() {
    case "${LINSTABPY_MPI_STACK:-}" in
        intel|intelmpi|impi)
            printf 'intelmpi\n'
            return 0
            ;;
        openmpi|mpich)
            printf '%s\n' "${LINSTABPY_MPI_STACK}"
            return 0
            ;;
    esac
    if [ -n "${I_MPI_ROOT:-}" ] || has_cmd impi_info || linux_cmd_looks_intel_mpi mpiexec || linux_cmd_looks_intel_mpi mpirun; then
        printf 'intelmpi\n'
        return 0
    fi
    if has_cmd mpiexec.openmpi || has_cmd mpirun.openmpi || has_cmd ompi_info; then
        printf 'openmpi\n'
        return 0
    fi
    if has_cmd mpiexec; then
        local mpiexec_id
        mpiexec_id="$(mpiexec --version 2>&1 | head -n3 || true)"
        if printf '%s\n' "${mpiexec_id}" | grep -qi 'open.mpi\|OpenRTE\|ORTE'; then
            printf 'openmpi\n'
            return 0
        fi
        if printf '%s\n' "${mpiexec_id}" | grep -qi 'mpich\|hydra\|HYDRA'; then
            printf 'mpich\n'
            return 0
        fi
    fi
    if has_cmd mpiexec.mpich || has_cmd mpiexec.hydra || has_cmd mpirun.mpich; then
        printf 'mpich\n'
        return 0
    fi
    printf 'openmpi\n'
}

linux_resolve_mpi_tool() {
    local name root candidate resolved
    root="$(linux_intel_mpi_root || true)"
    if [ -n "${root}" ]; then
        for candidate in "$@"; do
            if [ -x "${root}/bin/${candidate}" ]; then
                printf '%s\n' "${root}/bin/${candidate}"
                return 0
            fi
        done
    fi
    for candidate in "$@"; do
        if has_cmd "${candidate}"; then
            resolved="$(command -v "${candidate}")"
            case "${resolved}" in
                *conda*|*miniconda*|*anaconda*)
                    ;;
                *)
                    printf '%s\n' "${resolved}"
                    return 0
                    ;;
            esac
        fi
    done
    for candidate in "$@"; do
        if [ -x "/usr/bin/${candidate}" ]; then
            printf '%s\n' "/usr/bin/${candidate}"
            return 0
        fi
    done
    for candidate in "$@"; do
        if has_cmd "${candidate}"; then
            command -v "${candidate}"
            return 0
        fi
    done
    return 1
}

linux_mpi_compile_available() {
    LINSTABPY_MPI_COMPILE_ERROR=""
    local mpicc_bin
    mpicc_bin="$(linux_resolve_mpi_tool mpicc mpiicc mpiicx || true)"
    [ -n "${mpicc_bin}" ] || {
        LINSTABPY_MPI_COMPILE_ERROR="mpicc not found"
        return 1
    }

    local tmpdir src bin err rc
    tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/linstabpy-mpi-compile.XXXXXX")" || {
        LINSTABPY_MPI_COMPILE_ERROR="could not create temporary compile directory"
        return 1
    }
    src="${tmpdir}/mpi_probe.c"
    bin="${tmpdir}/mpi_probe"
    err="${tmpdir}/mpicc.err"
    cat > "${src}" <<'EOF'
#include <mpi.h>
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
    return 0;
}
EOF
    if "${mpicc_bin}" "${src}" -o "${bin}" >"${err}" 2>&1; then
        rc=0
    else
        LINSTABPY_MPI_COMPILE_ERROR="$(sed -n '1,3p' "${err}" | tr '\n' ' ' | sed -e 's/[[:space:]][[:space:]]*/ /g' -e 's/^ //g' -e 's/ $//g')"
        [ -n "${LINSTABPY_MPI_COMPILE_ERROR}" ] || LINSTABPY_MPI_COMPILE_ERROR="${mpicc_bin} failed to compile a program including mpi.h"
        rc=1
    fi
    rm -rf "${tmpdir}"
    return "${rc}"
}

# Fill SYSDEP_* with the same suggested package-manager lines used by install
# and by --diagnose. Prints nothing. Returns 1 when no package manager is found.
linux_sys_dep_plan() {
    SYSDEP_PM=""
    SYSDEP_UPDATE_CMD=""
    SYSDEP_INSTALL_CMD=""
    SYSDEP_HDF5_PKG=""
    SYSDEP_MPI_STACK="$(linux_mpi_stack_preference)"
    if has_cmd apt-get; then
        SYSDEP_PM="apt"
        SYSDEP_UPDATE_CMD="apt-get update"
        if [ "${SYSDEP_MPI_STACK}" = "openmpi" ]; then
            SYSDEP_INSTALL_CMD="apt-get install -y build-essential cmake gfortran pkg-config curl ca-certificates openmpi-bin libopenmpi-dev libx11-dev"
        else
            SYSDEP_INSTALL_CMD="apt-get install -y build-essential cmake gfortran pkg-config curl ca-certificates mpich libmpich-dev libx11-dev"
        fi
        SYSDEP_HDF5_PKG="libhdf5-dev"
    elif has_cmd dnf; then
        SYSDEP_PM="dnf"
        if [ "${SYSDEP_MPI_STACK}" = "openmpi" ]; then
            SYSDEP_INSTALL_CMD="dnf install -y gcc gcc-c++ gcc-gfortran make cmake pkgconf-pkg-config curl ca-certificates openmpi openmpi-devel libX11-devel"
        else
            SYSDEP_INSTALL_CMD="dnf install -y gcc gcc-c++ gcc-gfortran make cmake pkgconf-pkg-config curl ca-certificates mpich mpich-devel libX11-devel"
        fi
        SYSDEP_HDF5_PKG="hdf5-devel"
    elif has_cmd yum; then
        SYSDEP_PM="yum"
        if [ "${SYSDEP_MPI_STACK}" = "openmpi" ]; then
            SYSDEP_INSTALL_CMD="yum install -y gcc gcc-c++ gcc-gfortran make cmake pkgconfig curl ca-certificates openmpi openmpi-devel libX11-devel"
        else
            SYSDEP_INSTALL_CMD="yum install -y gcc gcc-c++ gcc-gfortran make cmake pkgconfig curl ca-certificates mpich mpich-devel libX11-devel"
        fi
        SYSDEP_HDF5_PKG="hdf5-devel"
    elif has_cmd pacman; then
        SYSDEP_PM="pacman"
        SYSDEP_UPDATE_CMD="pacman -Sy --noconfirm"
        if [ "${SYSDEP_MPI_STACK}" = "openmpi" ]; then
            SYSDEP_INSTALL_CMD="pacman -S --noconfirm --needed base-devel cmake gcc-fortran pkgconf curl ca-certificates openmpi libx11"
        else
            SYSDEP_INSTALL_CMD="pacman -S --noconfirm --needed base-devel cmake gcc-fortran pkgconf curl ca-certificates mpich libx11"
        fi
        SYSDEP_HDF5_PKG="hdf5"
    elif has_cmd zypper; then
        SYSDEP_PM="zypper"
        if [ "${SYSDEP_MPI_STACK}" = "openmpi" ]; then
            SYSDEP_INSTALL_CMD="zypper --non-interactive install gcc gcc-c++ gcc-fortran make cmake pkg-config curl ca-certificates openmpi openmpi-devel libX11-devel"
        else
            SYSDEP_INSTALL_CMD="zypper --non-interactive install gcc gcc-c++ gcc-fortran make cmake pkg-config curl ca-certificates mpich mpich-devel libX11-devel"
        fi
        SYSDEP_HDF5_PKG="hdf5-devel"
    else
        return 1
    fi
    if [ "${LINSTABPY_WITH_H5PY:-0}" = "1" ] && [ -n "${SYSDEP_HDF5_PKG}" ]; then
        SYSDEP_INSTALL_CMD="${SYSDEP_INSTALL_CMD} ${SYSDEP_HDF5_PKG}"
    fi
    if [ "${SYSDEP_MPI_STACK}" = "intelmpi" ]; then
        case "${SYSDEP_PM}" in
            apt)
                SYSDEP_INSTALL_CMD="apt-get install -y build-essential cmake gfortran pkg-config curl ca-certificates libx11-dev"
                ;;
            dnf)
                SYSDEP_INSTALL_CMD="dnf install -y gcc gcc-c++ gcc-gfortran make cmake pkgconf-pkg-config curl ca-certificates libX11-devel"
                ;;
            yum)
                SYSDEP_INSTALL_CMD="yum install -y gcc gcc-c++ gcc-gfortran make cmake pkgconfig curl ca-certificates libX11-devel"
                ;;
            pacman)
                SYSDEP_INSTALL_CMD="pacman -S --noconfirm --needed base-devel cmake gcc-fortran pkgconf curl ca-certificates libx11"
                ;;
            zypper)
                SYSDEP_INSTALL_CMD="zypper --non-interactive install gcc gcc-c++ gcc-fortran make cmake pkg-config curl ca-certificates libX11-devel"
                ;;
        esac
        if [ "${LINSTABPY_WITH_H5PY:-0}" = "1" ] && [ -n "${SYSDEP_HDF5_PKG}" ]; then
            SYSDEP_INSTALL_CMD="${SYSDEP_INSTALL_CMD} ${SYSDEP_HDF5_PKG}"
        fi
    fi
    return 0
}

linux_install_system_deps() {
    [ "${OS_NAME}" = "Linux" ] || return 0
    [ "${LINSTABPY_SKIP_SYS_DEPS:-0}" = "1" ] && {
        warn "Skipping Linux system dependency installation (LINSTABPY_SKIP_SYS_DEPS=1)."
        return 0
    }

    local required=(curl make cmake gcc g++ gfortran pkg-config)
    local missing=()
    local mpi_compile_error=""
    for c in "${required[@]}"; do
        has_cmd "$c" || missing+=("$c")
    done
    linux_resolve_mpi_tool mpicc mpiicc mpiicx >/dev/null || missing+=("mpicc")
    linux_resolve_mpi_tool mpicxx mpiicpc mpiicpx >/dev/null || missing+=("mpicxx")
    linux_resolve_mpi_tool mpifort mpif90 mpiifort mpiifx >/dev/null || missing+=("mpifort")
    if linux_resolve_mpi_tool mpicc mpiicc mpiicx >/dev/null && ! linux_mpi_compile_available; then
        missing+=("mpi-dev")
        mpi_compile_error="${LINSTABPY_MPI_COMPILE_ERROR}"
    fi
    if [ "${LINSTABPY_WITH_H5PY:-0}" = "1" ] && ! linux_hdf5_available; then
        missing+=("hdf5-dev")
    fi

    if [ ${#missing[@]} -eq 0 ]; then
        ok "Linux build prerequisites already installed."
        return 0
    fi

    warn "Missing Linux build tools: ${missing[*]}"
    [ -n "${mpi_compile_error}" ] && warn "MPI compile probe failed: ${mpi_compile_error}"

    local pm=""
    local update_cmd=""
    local install_cmd=""
    local hdf5_pkg=""
    local mpi_stack=""
    if ! linux_sys_dep_plan; then
        err "Unsupported Linux distro/package manager. Install manually: curl make cmake gcc g++ gfortran pkg-config MPI dev tools (mpicc/mpicxx/mpifort) and X11 headers."
        warn "HDF5 dev tools are optional and needed only when LINSTABPY_WITH_H5PY=1."
        exit 1
    fi
    pm="${SYSDEP_PM}"
    update_cmd="${SYSDEP_UPDATE_CMD}"
    install_cmd="${SYSDEP_INSTALL_CMD}"
    hdf5_pkg="${SYSDEP_HDF5_PKG}"
    mpi_stack="${SYSDEP_MPI_STACK}"

    if [ "${LINSTABPY_AUTO_INSTALL_SYS_DEPS:-0}" != "1" ]; then
        err "Refusing to install system packages automatically."
        err "This is intentional so cluster/module environments do not trigger unexpected sudo/package-manager calls."
        warn "Run 'bash install.sh --diagnose' first, load the required modules or install the packages manually, then re-run install.sh."
        warn "If you really want the installer to call the package manager, re-run with LINSTABPY_AUTO_INSTALL_SYS_DEPS=1."
        if [ "${LINSTABPY_WITH_H5PY:-0}" != "1" ]; then
            warn "HDF5/h5cc is optional and is not required for the default install path."
        fi
        warn "Suggested commands:"
        [ -n "${update_cmd}" ] && printf '  %s\n' "${update_cmd}"
        printf '  %s\n' "${install_cmd}"
        if [ "${mpi_stack}" != "intelmpi" ] && linux_intel_mpi_available; then
            warn "Alternative: Intel MPI appears to be available but is not active."
            warn "  source /opt/intel/oneapi/setvars.sh --force"
            warn "  LINSTABPY_MPI_STACK=intelmpi bash install.sh --diagnose"
            warn "  LINSTABPY_MPI_STACK=intelmpi bash install.sh"
        fi
        if [ "${mpi_stack}" = "intelmpi" ]; then
            warn "Intel MPI is not installed by this script. Load oneAPI first, for example:"
            warn "  source /opt/intel/oneapi/setvars.sh --force"
            warn "or load your cluster Intel MPI module, then re-run diagnose/install."
        fi
        exit 1
    fi
    log "Installing Linux system dependencies via ${pm} (LINSTABPY_AUTO_INSTALL_SYS_DEPS=1)..."
    if [ "$(id -u)" -eq 0 ]; then
        [ -n "${update_cmd}" ] && bash -lc "${update_cmd}"
        bash -lc "${install_cmd}"
    elif has_cmd sudo; then
        [ -n "${update_cmd}" ] && sudo bash -lc "${update_cmd}"
        sudo bash -lc "${install_cmd}"
    else
        err "Need root privileges to install system dependencies (sudo not found). Re-run as root or install manually."
        exit 1
    fi

    missing=()
    for c in "${required[@]}"; do
        has_cmd "$c" || missing+=("$c")
    done
    linux_resolve_mpi_tool mpicc mpiicc mpiicx >/dev/null || missing+=("mpicc")
    linux_resolve_mpi_tool mpicxx mpiicpc mpiicpx >/dev/null || missing+=("mpicxx")
    linux_resolve_mpi_tool mpifort mpif90 mpiifort mpiifx >/dev/null || missing+=("mpifort")
    if linux_resolve_mpi_tool mpicc mpiicc mpiicx >/dev/null && ! linux_mpi_compile_available; then
        missing+=("mpi-dev")
    fi
    if [ "${LINSTABPY_WITH_H5PY:-0}" = "1" ] && ! linux_hdf5_available; then
        missing+=("hdf5-dev")
    fi
    if [ ${#missing[@]} -gt 0 ]; then
        err "System dependency install incomplete; still missing: ${missing[*]}"
        exit 1
    fi
    ok "Linux system dependencies installed."
}

configure_linux_petsc_toolchain() {
    [ "${OS_NAME}" = "Linux" ] || return 0

    # Force GNU toolchain so PETSc does not accidentally pick Conda compilers.
    export CC="${CC:-gcc}"
    export CXX="${CXX:-g++}"
    export FC="${FC:-gfortran}"
    export F77="${F77:-gfortran}"
    export F90="${F90:-gfortran}"

    # Intel MPI generic wrappers honor these and keep PETSc on GNU compilers.
    export I_MPI_CC="${I_MPI_CC:-gcc}"
    export I_MPI_CXX="${I_MPI_CXX:-g++}"
    export I_MPI_F90="${I_MPI_F90:-gfortran}"
    export I_MPI_FC="${I_MPI_FC:-gfortran}"

    # Prefer the active MPI wrappers from PATH/I_MPI_ROOT; do not hard-code
    # /usr/bin, because that masks a loaded Intel oneAPI MPI module.
    local mpicc_bin mpicxx_bin mpifort_bin mpi_root
    mpicc_bin="$(linux_resolve_mpi_tool mpicc mpiicc mpiicx || true)"
    mpicxx_bin="$(linux_resolve_mpi_tool mpicxx mpiicpc mpiicpx || true)"
    mpifort_bin="$(linux_resolve_mpi_tool mpifort mpif90 mpiifort mpiifx || true)"
    if [ -n "${mpicc_bin}" ] && [ -n "${mpicxx_bin}" ] && [ -n "${mpifort_bin}" ]; then
        export MPICC="${MPICC:-${mpicc_bin}}"
        export MPICXX="${MPICXX:-${mpicxx_bin}}"
        export MPIFORT="${MPIFORT:-${mpifort_bin}}"
        if mpi_root="$(linux_intel_mpi_root)"; then
            export MPI_DIR="${MPI_DIR:-${mpi_root}}"
        elif [[ "${mpicc_bin}" = /usr/bin/* ]]; then
            export MPI_DIR="${MPI_DIR:-/usr}"
        fi
    fi

    # Hint common MPI stacks to use GNU Fortran under the hood.
    export OMPI_FC="${OMPI_FC:-gfortran}"

    ok "Configured Linux PETSc toolchain (GNU Fortran)."
}

linstabpy_resolve_launcher_candidate() {
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

# Echo absolute path to an mpiexec that successfully runs -n 1 /bin/true, or return 1.
linstabpy_find_mpiexec_for_n1() {
    local candidate resolved mpi_root
    local candidates=()
    if mpi_root="$(linux_intel_mpi_root)"; then
        candidates+=("${mpi_root}/bin/mpiexec" "${mpi_root}/bin/mpirun")
    fi
    candidates+=(mpiexec.mpich mpiexec.hydra mpirun.mpich mpirun.openmpi mpiexec.openmpi mpiexec mpirun)
    for candidate in "${candidates[@]}"; do
        resolved="$(linstabpy_resolve_launcher_candidate "${candidate}" || true)"
        [ -n "${resolved}" ] || continue
        if has_cmd timeout; then
            if timeout 25 "$resolved" -n 1 /bin/true 2>/dev/null; then
                printf '%s\n' "${resolved}"
                return 0
            fi
        else
            if "$resolved" -n 1 /bin/true 2>/dev/null; then
                printf '%s\n' "${resolved}"
                return 0
            fi
        fi
    done
    return 1
}

linstabpy_probe_mpi_launch() {
    if has_cmd timeout; then
        timeout 25 "$@" >/dev/null 2>&1
    else
        "$@" >/dev/null 2>&1
    fi
}

configure_petsc_build_mpiexec() {
    # The PyPI ``petsc`` / ``slepc`` sdists run ``mpiexec -n 1`` during configure.
    # On Debian/Ubuntu, ``/usr/bin/mpiexec`` (alternatives default) often points at a
    # launcher that hangs or times out on single-node builds while ``mpiexec.mpich``
    # works. PETSc honors the MPIEXEC environment variable for this probe.
    [ "${OS_NAME}" = "Linux" ] || return 0
    if [[ -n "${LINSTABPY_PETSC_MPIEXEC:-}" ]]; then
        export MPIEXEC="${LINSTABPY_PETSC_MPIEXEC}"
        log "PETSc/SLEPc configure: using LINSTABPY_PETSC_MPIEXEC=${MPIEXEC}"
        return 0
    fi
    local resolved
    if ! resolved="$(linstabpy_find_mpiexec_for_n1)"; then
        err "No working mpiexec for PETSc configure. Try: mpiexec.mpich -n 1 /bin/true"
        err "Or set LINSTABPY_PETSC_MPIEXEC=/path/to/mpiexec and re-run install.sh"
        exit 1
    fi
    export MPIEXEC="${resolved}"
    log "PETSc/SLEPc configure: MPIEXEC=${MPIEXEC} (verified: -n 1 /bin/true)"
}

# On Linux MPICH/Hydra, ``import mpi4py.MPI`` in a plain python process can hang in MPI_Init.
# Populate LINSTAB_VERIFY_MPIWRAP as: empty or ( mpiexec -n 1 ).
linstabpy_prepare_verify_mpiwrap() {
    LINSTAB_VERIFY_MPIWRAP=()
    [ "${OS_NAME}" = "Linux" ] || return 0
    # Drop inherited HYDRA_BOOTSTRAP (e.g. conda / old shells): Hydra \"exec\" is unsupported on many distros.
    unset HYDRA_BOOTSTRAP 2>/dev/null || true
    local mx="${LINSTABPY_VERIFY_MPIEXEC:-${MPIEXEC:-}}"
    if [ -z "${mx}" ]; then
        mx="$(linstabpy_find_mpiexec_for_n1 || true)"
    fi
    if [ -n "${mx}" ]; then
        LINSTAB_VERIFY_MPIWRAP=( "${mx}" -n 1 )
        log "Verification will use: ${mx} -n 1 (avoids MPICH singleton hang on mpi4py import)"
    else
        warn "No mpiexec found for verification. If the verify step hangs at mpi4py, set LINSTABPY_VERIFY_MPIEXEC=/usr/bin/mpiexec.mpich"
    fi
}

linstabpy_prepare_multirank_verify_mpiwrap() {
    local ranks="$1"
    LINSTAB_VERIFY_MULTIWRAP=()
    LINSTAB_VERIFY_MULTIENV=()
    [ "${OS_NAME}" = "Linux" ] || return 1
    unset HYDRA_BOOTSTRAP 2>/dev/null || true
    local mx="${LINSTABPY_VERIFY_MPIEXEC:-${MPIEXEC:-}}"
    if [ -z "${mx}" ]; then
        mx="$(linstabpy_find_mpiexec_for_n1 || true)"
    fi
    [ -n "${mx}" ] || return 1

    if linstabpy_probe_mpi_launch "${mx}" -n "${ranks}" /bin/true; then
        LINSTAB_VERIFY_MULTIWRAP=( "${mx}" -n "${ranks}" )
        log "Multi-rank verification will use: ${mx} -n ${ranks}"
        return 0
    fi

    case "$(basename "${mx}")" in
        mpiexec.mpich|mpiexec.hydra|mpirun.mpich)
            if linstabpy_probe_mpi_launch env HYDRA_BOOTSTRAP=fork "${mx}" -n "${ranks}" /bin/true; then
                LINSTAB_VERIFY_MULTIENV=( HYDRA_BOOTSTRAP=fork )
                LINSTAB_VERIFY_MULTIWRAP=( "${mx}" -n "${ranks}" )
                log "Multi-rank verification will use: HYDRA_BOOTSTRAP=fork ${mx} -n ${ranks}"
                return 0
            fi
            ;;
    esac
    return 1
}

linstabpy_cpu_count() {
    if has_cmd getconf; then
        getconf _NPROCESSORS_ONLN 2>/dev/null && return 0
    fi
    if has_cmd nproc; then
        nproc 2>/dev/null && return 0
    fi
    printf '1\n'
}

linstabpy_tool_summary() {
    local tool="$1"
    local resolved=""
    if [ -x "${tool}" ]; then
        resolved="${tool}"
    elif has_cmd "${tool}"; then
        resolved="$(command -v "${tool}")"
    else
        printf 'not found\n'
        return 0
    fi

    local first_line=""
    first_line="$("${resolved}" --version 2>&1 | head -n1 || true)"
    if [ -z "${first_line}" ]; then
        first_line="$("${resolved}" -V 2>&1 | head -n1 || true)"
    fi
    if [ -n "${first_line}" ]; then
        printf '%s  [%s]\n' "${resolved}" "${first_line}"
    else
        printf '%s\n' "${resolved}"
    fi
}

linstabpy_mpi_family() {
    local tool="${1:-}"
    local text=""
    if [ -n "${tool}" ] && [ -x "${tool}" ]; then
        text="$("${tool}" --version 2>&1 | head -n5 || true)"
        text="${text}"$'\n'"$("${tool}" -V 2>&1 | head -n5 || true)"
    fi
    if printf '%s\n' "${tool} ${text}" | grep -qi 'Intel.*MPI\|Intel(R).*MPI\|oneapi'; then
        printf 'Intel MPI\n'
    elif printf '%s\n' "${tool} ${text}" | grep -qi 'Open MPI\|OpenRTE\|ORTE'; then
        printf 'Open MPI\n'
    elif printf '%s\n' "${tool} ${text}" | grep -qi 'MPICH\|HYDRA'; then
        printf 'MPICH\n'
    else
        printf 'unknown MPI family\n'
    fi
}

linstabpy_report_runtime_environment() {
    log "MPI runtime summary:"
    local mpicc_bin=""
    mpicc_bin="$(linux_resolve_mpi_tool mpicc mpiicc mpiicx 2>/dev/null || true)"
    [ -n "${mpicc_bin}" ] && log "  mpicc   = $(linstabpy_tool_summary "${mpicc_bin}")"
    local report_mpiexec="${MPIEXEC:-}"
    if [ -z "${report_mpiexec}" ] && declare -p LINSTAB_VERIFY_MPIWRAP >/dev/null 2>&1 && [ "${#LINSTAB_VERIFY_MPIWRAP[@]}" -gt 0 ]; then
        report_mpiexec="${LINSTAB_VERIFY_MPIWRAP[0]}"
    fi
    if [ -n "${report_mpiexec}" ]; then
        log "  mpiexec = $(linstabpy_tool_summary "${report_mpiexec}")"
        log "  family  = $(linstabpy_mpi_family "${report_mpiexec}")"
    else
        warn "  mpiexec = not selected yet"
    fi
    local _mpi_ld_report=""
    if _mpi_ld_report="$(linstabpy_resolve_mpi_libdir 2>/dev/null)"; then
        log "  libdir  = ${_mpi_ld_report}"
    fi

    log "Threading environment for MPI runs:"
    local var val
    for var in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS; do
        val="${!var:-unset}"
        if [ "${val}" = "1" ]; then
            ok "  ${var}=1"
        elif [ "${val}" = "unset" ]; then
            warn "  ${var} is unset; recommended: export ${var}=1"
        else
            warn "  ${var}=${val}; recommended: export ${var}=1 for MPI PETSc/SLEPc runs"
        fi
    done
    log "Recommended ~/.bashrc block for MPI PETSc/SLEPc runs:"
    printf '  export OMP_NUM_THREADS=1\n'
    printf '  export OPENBLAS_NUM_THREADS=1\n'
    printf '  export MKL_NUM_THREADS=1\n'
    printf '  export NUMEXPR_NUM_THREADS=1\n'
}

install_mpi4py_compat() {
    # PyPI wheels often link libmpi.so.40 (Open MPI 4) while Debian/Ubuntu defaults
    # are MPICH (libmpi.so.12). Building from source with system mpicc fixes imports.
    if [ "${OS_NAME}" != "Linux" ]; then
        log "Installing mpi4py..."
        ${PIP_INSTALL} "${MPI4PY_SPEC}"
        return 0
    fi
    if [[ "${LINSTABPY_MPI4PY_NO_BINARY:-0}" = "1" ]]; then
        log "Installing mpi4py from source (LINSTABPY_MPI4PY_NO_BINARY=1)..."
        ${PIP_INSTALL} --no-binary mpi4py "${MPI4PY_SPEC}"
        return 0
    fi
    log "Installing mpi4py (binary wheel; falls back to source build if it does not match system MPI)..."
    ${PIP_INSTALL} "${MPI4PY_SPEC}"
    # MPICH: plain python can hang in MPI_Init; prefer mpiexec -n 1 (MPIEXEC is set for PETSc).
    _mpi4_import_ok=0
    if [ -n "${MPIEXEC:-}" ] && "${MPIEXEC}" -n 1 "${PY_BIN}" -c "import mpi4py.MPI" 2>/dev/null; then
        _mpi4_import_ok=1
    elif [ -z "${MPIEXEC:-}" ] && "${PY_BIN}" -c "import mpi4py.MPI" 2>/dev/null; then
        _mpi4_import_ok=1
    fi
    if [ "${_mpi4_import_ok}" = "1" ]; then
        ok "mpi4py loads against the system MPI libraries."
        return 0
    fi
    warn "mpi4py wheel failed to import (MPI ABI mismatch). Rebuilding mpi4py from source with mpicc..."
    ${PIP_INSTALL} --force-reinstall --no-binary mpi4py "${MPI4PY_SPEC}"
}

configure_linux_hdf5_env() {
    [ "${OS_NAME}" = "Linux" ] || return 0

    # Help h5py locate HDF5 on distros where libhdf5.so lives in non-default dirs
    # (for example /usr/lib/x86_64-linux-gnu/hdf5/serial).
    if ! has_cmd h5cc && [ ! -x /usr/bin/h5cc ]; then
        warn "h5cc not found; skipping HDF5 environment setup."
        return 0
    fi

    # Avoid conflicting knobs for h5py detection across repeated runs.
    unset HDF5_DIR HDF5_LIBDIR HDF5_INCLUDEDIR HDF5_PKGCONFIG_NAME

    local h5cc_path h5prefix h5lib h5inc
    # Prefer system HDF5 over Conda HDF5 to avoid old ABI/version mismatches.
    if [ -x /usr/bin/h5cc ]; then
        h5cc_path="/usr/bin/h5cc"
    else
        h5cc_path="$(command -v h5cc)"
    fi
    h5prefix="$("${h5cc_path}" -showconfig 2>/dev/null | awk -F: '/Installation point:/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"

    # If h5cc still resolves to a Conda prefix, fall back to distro defaults.
    case "${h5prefix}" in
        *anaconda*|*miniconda*|*mambaforge*|*micromamba*)
            h5prefix=""
            ;;
    esac

    if [ -n "${h5prefix}" ] && [ -d "${h5prefix}" ]; then
        export HDF5_DIR="${h5prefix}"
    fi

    h5lib="$(dirname "$(ldconfig -p 2>/dev/null | awk '/libhdf5\.so(\.[0-9]+)*$/{print $NF; exit}')")"
    case "${h5lib}" in
        *anaconda*|*miniconda*|*mambaforge*|*micromamba*) h5lib="" ;;
    esac
    if [ -z "${h5lib}" ]; then
        for d in \
            /usr/lib/x86_64-linux-gnu/hdf5/serial \
            /usr/lib64/hdf5/serial \
            /usr/lib/hdf5/serial \
            /usr/lib/x86_64-linux-gnu \
            /usr/lib64 \
            /usr/lib \
            /usr/local/lib
        do
            if [ -d "$d" ] && ls "$d"/libhdf5.so* >/dev/null 2>&1; then
                h5lib="$d"
                break
            fi
        done
    fi

    h5inc=""
    for d in \
        /usr/include/hdf5/serial \
        /usr/include/hdf5/openmpi \
        /usr/include/hdf5/mpich \
        /usr/include \
        /usr/local/include
    do
        if [ -f "$d/hdf5.h" ]; then
            h5inc="$d"
            break
        fi
    done

    if [ -n "${h5lib}" ] && [ -d "${h5lib}" ]; then
        export LD_LIBRARY_PATH="${h5lib}:${LD_LIBRARY_PATH:-}"
        export LDFLAGS="-L${h5lib} ${LDFLAGS:-}"
    fi
    if [ -n "${h5inc}" ] && [ -d "${h5inc}" ]; then
        export CPPFLAGS="-I${h5inc} ${CPPFLAGS:-}"
    fi

    # Prefer pkg-config driven discovery on Linux distros (esp. Debian/Ubuntu
    # where the package name is usually "hdf5-serial").
    if has_cmd pkg-config; then
        if pkg-config --exists hdf5-serial; then
            unset HDF5_DIR HDF5_LIBDIR HDF5_INCLUDEDIR
            export HDF5_PKGCONFIG_NAME="hdf5-serial"
        elif pkg-config --exists hdf5; then
            unset HDF5_DIR HDF5_LIBDIR HDF5_INCLUDEDIR
            export HDF5_PKGCONFIG_NAME="hdf5"
        fi
    fi

    # h5py requires exactly one discovery mode:
    # - HDF5_DIR (prefix), or
    # - HDF5_LIBDIR + HDF5_INCLUDEDIR (pair), or
    # - HDF5_PKGCONFIG_NAME.
    # If no pkg-config mode was selected and HDF5_DIR is unavailable, fall back
    # to explicit lib/include pair.
    if [ -z "${HDF5_PKGCONFIG_NAME:-}" ] && [ -z "${HDF5_DIR:-}" ] && [ -n "${h5lib}" ] && [ -n "${h5inc}" ]; then
        export HDF5_LIBDIR="${h5lib}"
        export HDF5_INCLUDEDIR="${h5inc}"
    fi

    if [ -z "${HDF5_DIR:-}" ] && { [ -z "${HDF5_LIBDIR:-}" ] || [ -z "${HDF5_INCLUDEDIR:-}" ]; }; then
        warn "Could not infer complete HDF5 paths for h5py build."
        warn "If install fails, set HDF5_LIBDIR and HDF5_INCLUDEDIR explicitly."
    fi

    ok "Configured Linux HDF5 build environment."
}

run_host_diagnose() {
    # Diagnose must not abort on a missing tool. install.sh uses set -e.
    (
        set +e
        set -uo pipefail
        COL_OK='\033[1;32m'
        COL_MISS='\033[1;31m'
        COL_WARN='\033[1;33m'
        COL_INFO='\033[1;34m'
        COL_RST='\033[0m'
        FAIL_COUNT=0
        WARNINGS=()

        tool_version() {
            local c="$1"
            local v=""
            v="$("${c}" --version 2>/dev/null | head -n1)" \
                || v="$("${c}" -V 2>/dev/null | head -n1)" \
                || v="$("${c}" -v 2>/dev/null | head -n1)" \
                || v=""
            printf '%s' "${v}"
        }

        check_tool() {
            local name="$1"
            local optional="${2:-}"
            if has_cmd "${name}"; then
                local path ver
                path="$(command -v "${name}")"
                ver="$(tool_version "${name}")"
                printf "  %-14s ${COL_OK}OK${COL_RST}   %s" "${name}" "${path}"
                [ -n "${ver}" ] && printf "  (%s)" "${ver}"
                printf '\n'
            else
                if [ "${optional}" = "optional" ]; then
                    printf "  %-14s ${COL_WARN}MISS${COL_RST} (optional)\n" "${name}"
                else
                    printf "  %-14s ${COL_MISS}MISS${COL_RST}\n" "${name}"
                    FAIL_COUNT=$((FAIL_COUNT + 1))
                fi
            fi
        }

        check_any_mpi_tool() {
            local label="$1"
            shift
            local path ver
            path="$(linux_resolve_mpi_tool "$@" || true)"
            if [ -n "${path}" ]; then
                ver="$(tool_version "${path}")"
                printf "  %-14s ${COL_OK}OK${COL_RST}   %s" "${label}" "${path}"
                [ -n "${ver}" ] && printf "  (%s)" "${ver}"
                printf '\n'
                return 0
            fi
            printf "  %-14s ${COL_MISS}MISS${COL_RST}\n" "${label}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            return 1
        }

        echo ""
        printf "=== ${COL_INFO}linstabpy install diagnostic${COL_RST} ===\n"
        echo ""

        OS_NAME=$(uname -s)
        KERNEL=$(uname -r 2>/dev/null || echo "unknown")
        ARCH=$(uname -m 2>/dev/null || echo "unknown")
        DISTRO="unknown"
        if has_cmd lsb_release; then
            DISTRO="$(lsb_release -ds 2>/dev/null || echo "unknown")"
        elif [ -f /etc/os-release ]; then
            DISTRO="$(awk -F= '/^PRETTY_NAME=/{gsub(/"/, "", $2); print $2; exit}' /etc/os-release 2>/dev/null || echo "unknown")"
        fi
        printf "  OS:        %s\n" "${DISTRO}"
        printf "  Kernel:    %s\n" "${KERNEL}"
        printf "  Arch:      %s\n" "${ARCH}"

        PM=""
        if   has_cmd apt-get; then PM="apt"
        elif has_cmd dnf;     then PM="dnf"
        elif has_cmd yum;     then PM="yum"
        elif has_cmd pacman;  then PM="pacman"
        elif has_cmd zypper;  then PM="zypper"
        fi
        if [ -n "${PM}" ]; then
            printf "  Pkg mgr:   %s\n" "${PM}"
        else
            printf "  Pkg mgr:   ${COL_MISS}not detected${COL_RST}\n"
        fi
        echo ""

        printf "${COL_INFO}[tools]${COL_RST}\n"
        for t in curl make cmake gcc g++ gfortran pkg-config; do
            check_tool "${t}"
        done
        check_any_mpi_tool mpicc mpicc mpiicc mpiicx
        check_any_mpi_tool mpicxx mpicxx mpiicpc mpiicpx
        check_any_mpi_tool mpifort mpifort mpif90 mpiifort mpiifx
        check_tool mpiexec
        check_tool h5cc optional
        check_tool uv optional
        echo ""

        printf "${COL_INFO}[python]${COL_RST}\n"
        if has_cmd python3; then
            PY3_PATH="$(command -v python3)"
            PY3_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || echo "?")"
            printf "  %-14s ${COL_OK}OK${COL_RST}   %s  (%s)\n" "python3" "${PY3_PATH}" "${PY3_VER}"
        else
            printf "  %-14s ${COL_MISS}MISS${COL_RST}\n" "python3"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        if has_cmd uv; then
            UV_PATH="$(command -v uv)"
            UV_VER="$(uv --version 2>/dev/null | head -n1 || echo "?")"
            printf "  %-14s ${COL_OK}OK${COL_RST}   %s  (%s)\n" "uv" "${UV_PATH}" "${UV_VER}"
        else
            printf "  %-14s ${COL_WARN}MISS${COL_RST} (will be auto-installed by install.sh)\n" "uv"
        fi
        echo ""

        printf "${COL_INFO}[mpi]${COL_RST}\n"
        OPENMPI_FOUND=0
        MPICH_FOUND=0
        INTELMPI_FOUND=0
        PREFER_INTELMPI=0
        if [ -n "${LINSTABPY_MPI_STACK:-}" ]; then
            printf "  Requested:   %s (LINSTABPY_MPI_STACK)\n" "${LINSTABPY_MPI_STACK}"
            case "${LINSTABPY_MPI_STACK}" in
                intel|intelmpi|impi) INTELMPI_FOUND=1; PREFER_INTELMPI=1 ;;
                openmpi) OPENMPI_FOUND=1 ;;
                mpich) MPICH_FOUND=1 ;;
            esac
        fi
        if [ -n "${I_MPI_ROOT:-}" ]; then
            INTELMPI_FOUND=1
            printf "  Intel MPI:   ${COL_OK}FOUND${COL_RST}  I_MPI_ROOT=%s\n" "${I_MPI_ROOT}"
        elif has_cmd impi_info; then
            INTELMPI_FOUND=1
            printf "  Intel MPI:   ${COL_OK}FOUND${COL_RST}  impi_info -> %s\n" "$(command -v impi_info)"
        elif linux_intel_mpi_available; then
            printf "  Intel MPI:   ${COL_INFO}AVAILABLE${COL_RST} oneAPI found, but not active in this shell\n"
        fi
        for candidate in mpiexec mpirun; do
            if linux_cmd_looks_intel_mpi "${candidate}"; then
                INTELMPI_FOUND=1
                resolved="$(command -v "${candidate}")"
                printf "  Intel MPI:   ${COL_OK}FOUND${COL_RST}  %s -> %s\n" "${candidate}" "${resolved}"
            fi
        done
        for candidate in mpiexec.openmpi mpirun.openmpi; do
            if has_cmd "${candidate}"; then
                OPENMPI_FOUND=1
                resolved="$(command -v "${candidate}")"
                printf "  Open MPI:    ${COL_OK}FOUND${COL_RST}  %s -> %s\n" "${candidate}" "${resolved}"
            fi
        done
        for candidate in mpiexec.mpich mpiexec.hydra mpirun.mpich; do
            if has_cmd "${candidate}"; then
                linux_cmd_looks_intel_mpi "${candidate}" && continue
                MPICH_FOUND=1
                resolved="$(command -v "${candidate}")"
                printf "  MPICH:       ${COL_OK}FOUND${COL_RST}  %s -> %s\n" "${candidate}" "${resolved}"
            fi
        done
        if has_cmd mpiexec; then
            generic_mpiexec="$(command -v mpiexec)"
            printf "  mpiexec:     ${COL_OK}FOUND${COL_RST}  %s" "${generic_mpiexec}"
            mpiexec_id="$(mpiexec --version 2>&1 | head -n3 || true)"
            if echo "${mpiexec_id}" | grep -qi "Intel.*MPI\|Intel(R).*MPI"; then
                printf "  (Intel MPI)"
                INTELMPI_FOUND=1
            elif echo "${mpiexec_id}" | grep -qi "open.mpi\|OpenRTE\|ORTE"; then
                printf "  (Open MPI)"
                OPENMPI_FOUND=1
            elif echo "${mpiexec_id}" | grep -qi "mpich\|hydra\|HYDRA"; then
                printf "  (MPICH)"
                MPICH_FOUND=1
            fi
            printf '\n'
        fi
        if [ "${OPENMPI_FOUND}" -eq 0 ] && [ "${MPICH_FOUND}" -eq 0 ] && [ "${INTELMPI_FOUND}" -eq 0 ]; then
            printf "  ${COL_MISS}No MPI installation detected.${COL_RST}\n"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        MPI_STACK_COUNT=$((OPENMPI_FOUND + MPICH_FOUND + INTELMPI_FOUND))
        if [ "${MPI_STACK_COUNT}" -gt 1 ]; then
            if [ "${INTELMPI_FOUND}" -eq 1 ] && linux_cmd_looks_intel_mpi mpiexec && linux_cmd_looks_intel_mpi "$(linux_resolve_mpi_tool mpicc mpiicc mpiicx 2>/dev/null || true)"; then
                printf "\n  ${COL_INFO}NOTE: Additional MPI launchers are visible, but the active mpicc/mpiexec are Intel MPI.${COL_RST}\n"
                printf "  Keep Intel MPI first in PATH while installing and running this environment.\n"
            else
                printf "\n  ${COL_WARN}WARNING: Multiple MPI stacks are visible in this environment.${COL_RST}\n"
                printf "  This causes ABI conflicts (e.g. mpi4py wheel links to one,\n"
                printf "  but mpiexec invokes the other). Remove one before installing.\n"
                WARNINGS+=("dual-mpi")
            fi
        fi

        echo ""
        MPI_TRUE_PROBE="/bin/true"
        if [ "${OS_NAME}" = "Darwin" ] && [ -x /usr/bin/true ]; then
            MPI_TRUE_PROBE="/usr/bin/true"
        fi
        printf "  Probing mpiexec -n 1 %s (5s timeout each):\n" "${MPI_TRUE_PROBE}"
        WORKING_MPIEXEC=""
        MPI_PROBE_CANDIDATES=()
        if [ -n "${I_MPI_ROOT:-}" ] && [ -d "${I_MPI_ROOT}/bin" ]; then
            MPI_PROBE_CANDIDATES+=("${I_MPI_ROOT}/bin/mpiexec" "${I_MPI_ROOT}/bin/mpirun")
        fi
        MPI_PROBE_CANDIDATES+=(mpiexec.mpich mpiexec.hydra mpiexec.openmpi mpirun.openmpi mpiexec mpirun)
        for candidate in "${MPI_PROBE_CANDIDATES[@]}"; do
            resolved="$(linstabpy_resolve_launcher_candidate "${candidate}" || true)"
            [ -n "${resolved}" ] || continue
            if has_cmd timeout; then
                if timeout -k 2 5 "${resolved}" -n 1 "${MPI_TRUE_PROBE}" >/dev/null 2>&1; then
                    printf "    %-22s ${COL_OK}OK${COL_RST}\n" "${candidate}"
                    [ -z "${WORKING_MPIEXEC}" ] && WORKING_MPIEXEC="${resolved}"
                else
                    printf "    %-22s ${COL_MISS}FAIL/TIMEOUT${COL_RST}\n" "${candidate}"
                fi
            else
                printf "    %-22s ${COL_WARN}SKIP${COL_RST} (timeout command not available)\n" "${candidate}"
            fi
        done
        if [ -n "${WORKING_MPIEXEC}" ]; then
            printf "  Best candidate: %s\n" "${WORKING_MPIEXEC}"
        else
            if [ "${OPENMPI_FOUND}" -eq 1 ] || [ "${MPICH_FOUND}" -eq 1 ] || [ "${INTELMPI_FOUND}" -eq 1 ]; then
                printf "  ${COL_MISS}No mpiexec passed the -n 1 %s probe.${COL_RST}\n" "${MPI_TRUE_PROBE}"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        fi

        echo ""
        if has_cmd ldconfig; then
            libmpi_line="$(ldconfig -p 2>/dev/null | grep 'libmpi\.so\.' | head -n1 || true)"
            if [ -n "${libmpi_line}" ]; then
                printf "  libmpi.so:   %s\n" "${libmpi_line}"
            else
                printf "  libmpi.so:   ${COL_WARN}not found in ldconfig cache${COL_RST}\n"
            fi
        fi
        if linux_resolve_mpi_tool mpicc mpiicc mpiicx >/dev/null; then
            if linux_mpi_compile_available; then
                printf "  MPI compile: ${COL_OK}OK${COL_RST}   mpicc can compile and link mpi.h\n"
            else
                printf "  MPI compile: ${COL_MISS}FAIL${COL_RST} %s\n" "${LINSTABPY_MPI_COMPILE_ERROR}"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        else
            printf "  MPI compile: ${COL_WARN}SKIP${COL_RST} (mpicc not found)\n"
        fi
        echo ""

        printf "${COL_INFO}[hdf5]${COL_RST} (optional, needed for h5py)\n"
        if has_cmd h5cc; then
            h5ver="$(h5cc -showconfig 2>/dev/null | awk '/HDF5 Version:/{print $NF; exit}' || echo "?")"
            printf "  h5cc:        ${COL_OK}OK${COL_RST}   %s  (HDF5 %s)\n" "$(command -v h5cc)" "${h5ver}"
        elif has_cmd pkg-config && pkg-config --exists hdf5 2>/dev/null; then
            h5ver="$(pkg-config --modversion hdf5 2>/dev/null || echo "?")"
            printf "  hdf5:        ${COL_OK}OK${COL_RST}   via pkg-config  (%s)\n" "${h5ver}"
        elif has_cmd pkg-config && pkg-config --exists hdf5-serial 2>/dev/null; then
            h5ver="$(pkg-config --modversion hdf5-serial 2>/dev/null || echo "?")"
            printf "  hdf5-serial: ${COL_OK}OK${COL_RST}   via pkg-config  (%s)\n" "${h5ver}"
        else
            printf "  hdf5:        ${COL_WARN}MISS${COL_RST} (not required unless LINSTABPY_WITH_H5PY=1)\n"
        fi
        echo ""

        WARN_COUNT=${#WARNINGS[@]}
        if [ "${FAIL_COUNT}" -gt 0 ] || [ "${WARN_COUNT}" -gt 0 ]; then
            printf "=== ${COL_WARN}ACTION REQUIRED${COL_RST} ===\n"
            echo ""
            if [ "${WARN_COUNT}" -gt 0 ]; then
                for w in "${WARNINGS[@]}"; do
                    if [ "${w}" = "dual-mpi" ]; then
                        printf "  ${COL_WARN}Resolve MPI conflict (pick one):${COL_RST}\n"
                        printf "    To use Intel MPI: unload/hide Open MPI and MPICH, then source oneAPI or load the Intel MPI module\n"
                    fi
                done
                echo ""
            fi
            if [ "${FAIL_COUNT}" -gt 0 ]; then
                printf "  ${COL_WARN}Install missing tools:${COL_RST}\n"
                if linux_sys_dep_plan; then
                    [ -n "${SYSDEP_UPDATE_CMD}" ] && printf "    %s\n" "${SYSDEP_UPDATE_CMD}"
                    printf "    %s\n" "${SYSDEP_INSTALL_CMD}"
                else
                    printf "    Install manually: curl make cmake gcc g++ gfortran pkg-config, MPI dev tools, X11 headers\n"
                fi
                if [ "${LINSTABPY_WITH_H5PY:-0}" != "1" ]; then
                    printf "  ${COL_INFO}Note:${COL_RST} HDF5/h5cc is optional; install HDF5 only when LINSTABPY_WITH_H5PY=1.\n"
                fi
                if [ "${PREFER_INTELMPI}" -eq 0 ] && [ "${INTELMPI_FOUND}" -eq 0 ] && linux_intel_mpi_available; then
                    printf "  ${COL_INFO}Alternative:${COL_RST} Intel MPI is available but not active. To use it instead:\n"
                    printf "    source /opt/intel/oneapi/setvars.sh --force\n"
                    printf "    LINSTABPY_MPI_STACK=intelmpi bash install.sh --diagnose\n"
                    printf "    LINSTABPY_MPI_STACK=intelmpi bash install.sh\n"
                fi
                echo ""
            fi
            exit 1
        fi
        printf "=== ${COL_OK}ALL CHECKS PASSED${COL_RST} ===\n"
        printf "  Ready to run: bash install.sh\n"
        if [ -n "${WORKING_MPIEXEC}" ]; then
            printf "  Working mpiexec: %s\n" "${WORKING_MPIEXEC}"
            printf "  Hint: LINSTABPY_PETSC_MPIEXEC=%s bash install.sh\n" "${WORKING_MPIEXEC}"
        fi
        echo ""
        exit 0
    )
}

if [ "${INSTALL_MODE}" = "diagnose" ]; then
    run_host_diagnose
    exit $?
fi

# Ensure reference MATLAB data/code is present locally for local inspection.
ensure_reference_submodule

# ── Quick verification mode ──────────────────────────────────────────────────
if [[ "${INSTALL_MODE}" == "check" ]]; then
    if [ ! -d "${VENV_DIR}" ]; then
        err "No virtual environment found at ${VENV_DIR}"
        exit 1
    fi
    linstabpy_prepare_verify_mpiwrap
    linstabpy_report_runtime_environment
    _check_snippet="$(mktemp_py)"
    cat >"${_check_snippet}" <<'PY'
import sys, numpy as np
# mpi4py must bind MPI before petsc4py (same rule as linstabpy.utils.petsc_init).
import mpi4py.MPI  # noqa: F401
from petsc4py import PETSc
from slepc4py import SLEPc
scalar = "complex" if np.issubdtype(PETSc.ScalarType, np.complexfloating) else "real"
if scalar != "complex":
    print(f"FAIL: PETSc scalar type is {scalar}, expected complex", file=sys.stderr)
    sys.exit(1)
print(f"python={sys.version.split()[0]}")
print(f"petsc4py={'.'.join(map(str,PETSc.Sys.getVersion()))}, scalar={scalar}")
print(f"slepc4py={'.'.join(map(str,SLEPc.Sys.getVersion()))}")
mat = PETSc.Mat().createAIJ([2, 2], nnz=2, comm=PETSc.COMM_SELF)
mat.setUp()
mat[0, 0] = 1.0
mat[0, 1] = 2.0
mat[1, 1] = 3.0
mat.assemble()
eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
eps.setOperators(mat)
eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
eps.setDimensions(1)
eps.solve()
if eps.getConverged() < 1:
    print("FAIL: SLEPc NHEP check converged no eigenpairs", file=sys.stderr)
    sys.exit(1)
eps.destroy()
mat.destroy()
print("All checks passed.")
PY
    linstabpy_run_verify "${VENV_DIR}/bin/python" "${_check_snippet}"
    _check_rc=$?
    rm -f "${_check_snippet}"
    if [ "${_check_rc}" -ne 0 ]; then
        exit "${_check_rc}"
    fi

    _cpu_count="$(linstabpy_cpu_count)"
    if [ "${OS_NAME}" = "Linux" ] && [ "${_cpu_count}" -gt 1 ]; then
        if ! linstabpy_prepare_multirank_verify_mpiwrap 2; then
            err "Quick check failed: no working 2-rank mpiexec launcher was found."
            err "Try: HYDRA_BOOTSTRAP=fork ${MPIEXEC:-mpiexec} -n 2 /bin/true"
            exit 1
        fi
        _check_multirank_snippet="$(mktemp_py)"
        cat >"${_check_multirank_snippet}" <<'PY'
import mpi4py.MPI as MPI
from petsc4py import PETSc

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
petsc_size = PETSc.COMM_WORLD.getSize()

assert size == 2, f"expected mpi4py COMM_WORLD size 2, got {size}"
assert petsc_size == 2, f"expected PETSc COMM_WORLD size 2, got {petsc_size}"

vec = PETSc.Vec().createMPI(4, comm=PETSc.COMM_WORLD)
start, end = vec.getOwnershipRange()
for idx in range(start, end):
    vec.setValue(idx, rank + 1)
vec.assemble()
expected_vec_sum = MPI.COMM_WORLD.allreduce((end - start) * (rank + 1), op=MPI.SUM)
actual_vec_sum = vec.sum()
assert abs(actual_vec_sum - expected_vec_sum) < 1e-12, (
    f"PETSc distributed Vec sum mismatch: got {actual_vec_sum}, expected {expected_vec_sum}"
)
vec.destroy()

mat = PETSc.Mat().createAIJ([4, 4], nnz=1, comm=PETSc.COMM_WORLD)
mat.setUp()
row_start, row_end = mat.getOwnershipRange()
for idx in range(row_start, row_end):
    mat.setValue(idx, idx, 1.0)
mat.assemble()
b = PETSc.Vec().createMPI(4, comm=PETSc.COMM_WORLD)
b.set(1.0)
b.assemble()
x = b.duplicate()
ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
ksp.setOperators(mat)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.JACOBI)
ksp.solve(b, x)
assert ksp.getConvergedReason() > 0, f"PETSc KSP did not converge: {ksp.getConvergedReason()}"
assert abs(x.sum() - 4.0) < 1e-12, f"PETSc distributed KSP solution sum mismatch: {x.sum()}"
ksp.destroy()
x.destroy()
b.destroy()
mat.destroy()

if rank == 0:
    print(f"mpi4py_multirank=size={size}")
    print(f"petsc4py_multirank=size={petsc_size}")
    print(f"petsc4py_multirank_vec_sum={actual_vec_sum}")
    print("petsc4py_multirank_ksp=OK")
PY
        linstabpy_run_verify_wrapped \
            "${VENV_DIR}/bin/python" \
            "${_check_multirank_snippet}" \
            "${LINSTAB_VERIFY_MULTIENV[@]}" \
            "${LINSTAB_VERIFY_MULTIWRAP[@]}"
        _check_multirank_rc=$?
        rm -f "${_check_multirank_snippet}"
        if [ "${_check_multirank_rc}" -ne 0 ]; then
            err "Quick check failed during 2-rank MPI verification (exit ${_check_multirank_rc})."
            err "Try: HYDRA_BOOTSTRAP=fork ${MPIEXEC:-mpiexec} -n 2 ${VENV_DIR}/bin/python -c 'import mpi4py.MPI; from petsc4py import PETSc'"
            exit "${_check_multirank_rc}"
        fi
    fi

    exit 0
fi

log "Detected OS: ${OS_NAME}"
linux_install_system_deps

# ── Locate or install uv ─────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
UV_BIN="$(command -v uv)"
log "Using uv at: ${UV_BIN}"

# ── Create/validate venv with the requested Python version ───────────────────
if [ "${FORCE_REINSTALL}" = "1" ] && [ -d "${VENV_DIR}" ]; then
    warn "Removing existing virtual environment at ${VENV_DIR} (--force)"
    remove_venv_dir "${VENV_DIR}"
fi

if [ "${FORCE_REINSTALL}" = "1" ]; then
    # uv caches the COMPILED petsc/slepc sdists and reuses them even when the
    # configure options change (e.g. adding --with-openmp / --with-cuda). That is a
    # silent stale-build trap: the regression that built STRUMPACK without OpenMP
    # survived because a rebuild reused the cached, OpenMP-less PETSc. On --force,
    # evict those cache entries so PETSC_CONFIGURE_OPTIONS below actually take effect.
    warn "Evicting cached petsc/slepc builds so configure options take effect (--force)"
    # Clear the Python bindings too: a cached petsc4py/slepc4py built against a
    # differently-configured PETSc (e.g. with vs without MUMPS) fails to import with
    # "undefined symbol". Rebuilding all four together keeps the ABI consistent.
    "${UV_BIN}" cache clean petsc slepc petsc4py slepc4py >/dev/null 2>&1 || true
fi

if [ -d "${VENV_DIR}" ]; then
    EXISTING_PY="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [ "${EXISTING_PY}" != "${PYTHON_VERSION}" ]; then
        warn "Existing venv uses Python ${EXISTING_PY:-unknown}; recreating with Python ${PYTHON_VERSION}"
        remove_venv_dir "${VENV_DIR}"
        log "Creating virtual environment (Python ${PYTHON_VERSION}) at ${VENV_DIR}"
        "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
    else
        log "Using existing virtual environment at ${VENV_DIR}"
    fi
else
    log "Creating virtual environment (Python ${PYTHON_VERSION}) at ${VENV_DIR}"
    "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi

linux_wrap_venv_python
darwin_wrap_venv_python

PY_BIN="${VENV_DIR}/bin/python"
PIP_UPGRADE_ARGS=()
if [ "${LATEST_PACKAGES}" = "1" ]; then
    PIP_UPGRADE_ARGS=(--upgrade)
    CORE_DEPS=(numpy scipy matplotlib)
    MPI4PY_SPEC="mpi4py"
    PETSC_SPEC="petsc"
    SLEPC_SPEC="slepc"
    PETSC4PY_SPEC="petsc4py"
    SLEPC4PY_SPEC="slepc4py"
    BUILD_DEPS=(Cython setuptools)
    H5PY_SPEC="h5py"
    DEV_DEPS=(pytest pytest-cov sympy)
    log "Latest-package mode enabled: uv will request newest compatible releases."
else
    CORE_DEPS=("numpy>=1.23" "scipy>=1.10" "matplotlib>=3.7")
    MPI4PY_SPEC="mpi4py>=3.1"
    PETSC_SPEC="petsc==${PETSC_VERSION}"
    SLEPC_SPEC="slepc==${SLEPC_VERSION}"
    PETSC4PY_SPEC="petsc4py==${PETSC4PY_VERSION}"
    SLEPC4PY_SPEC="slepc4py==${SLEPC4PY_VERSION}"
    BUILD_DEPS=("Cython>=3.0" "setuptools<72")
    H5PY_SPEC="h5py>=3.16"
    DEV_DEPS=(pytest pytest-cov "sympy>=1.13")
fi
PIP_INSTALL="${UV_BIN} pip install --python ${PY_BIN} ${PIP_UPGRADE_ARGS[*]}"

# ── Step 1: Core dependencies ────────────────────────────────────────────────
log "Installing core dependencies..."
${PIP_INSTALL} "${CORE_DEPS[@]}"

# ── Step 2: MPI + PETSc + SLEPc C libraries ─────────────────────────────────
log "Installing PETSc (sdist) then mpi4py..."
unset PETSC_DIR PETSC_ARCH SLEPC_DIR 2>/dev/null || true
configure_linux_petsc_toolchain
configure_petsc_build_mpiexec
linstabpy_report_runtime_environment
# Optional Hydra bootstrap for PETSc/SLEPc configure only (not exported — breaks some mpiexec).
if [ "${LINSTABPY_HYDRA_BOOTSTRAP+x}" = x ]; then
    _PETSC_HYDRA_BOOTSTRAP="${LINSTABPY_HYDRA_BOOTSTRAP}"
else
    _PETSC_HYDRA_BOOTSTRAP=exec
fi
# Compiler optimisation for PETSc and everything it downloads (MUMPS, METIS,
# ScaLAPACK, OpenBLAS). The PyPI sdist default is "-g -O", i.e. -O1: measured
# 2026-08-17 in the c9 venv (CC_FLAGS/FC_FLAGS "-g -O"). -O3 is PETSc's own
# recommendation for production builds. LINSTABPY_PETSC_OPTFLAGS overrides it.
# Keep it ONE token: these options travel through PETSC_CONFIGURE_OPTIONS,
# which the petsc sdist installer may split on whitespace.
_PETSC_OPTFLAGS="${LINSTABPY_PETSC_OPTFLAGS:--O3}"
_PETSC_CONF_OPTS=(
    "--with-scalar-type=complex"
    "--download-cmake"
    "--COPTFLAGS=${_PETSC_OPTFLAGS}"
    "--CXXOPTFLAGS=${_PETSC_OPTFLAGS}"
    "--FOPTFLAGS=${_PETSC_OPTFLAGS}"
)
if [ "${BUILD_STACK}" = "default" ]; then
    _PETSC_CONF_OPTS+=(
        "--download-mumps"
        "--download-metis"
        "--download-parmetis"
    )
else
    # --with-openmp is MANDATORY for the GPU stack. STRUMPACK uses OpenMP both for
    # CPU-side multifrontal threading AND to schedule dense fronts onto the GPU.
    # Omitting it builds STRUMPACK serial ("running serially, no OpenMP support!"),
    # which makes the factorization single-threaded on the CPU and leaves the GPU
    # idle even at scale. (Regression root-caused & fixed 2026-06-25; the post-install
    # verify below now hard-fails if OpenMP/CUDA/STRUMPACK are missing.)
    # The GPU stack is an ALL-BACKENDS venv: it must carry MUMPS (and SuperLU_DIST)
    # alongside STRUMPACK so a single venv can run the CPU(MUMPS)-vs-GPU(STRUMPACK)
    # comparison in one process, and so the prebuilt petsc4py (which references MUMPS
    # symbols, e.g. MatMumpsGetRinfog) links. Dropping MUMPS here breaks petsc4py import.
    # SLATE (+ its blaspp/lapackpp deps) is MANDATORY for GPU offload of the LARGE
    # distributed dense fronts. Without it STRUMPACK warns "SLATE is required for full
    # GPU support" and falls back to ScaLAPACK on the CPU for the root front: at multi-
    # rank that blows host RAM (~112 GB) and SystemError-crashes, and at single rank it
    # leaves the GPU mostly idle (~3.7 GB) and runs for hours. (Root-caused 2026-06-26.)
    _PETSC_CONF_OPTS+=(
        "--with-openmp"
        "--with-cuda"
        "--download-strumpack"
        "--download-slate"
        "--download-blaspp"
        "--download-lapackpp"
        "--download-mumps"
        "--download-superlu_dist"
        "--download-metis"
        "--download-parmetis"
        "--download-ptscotch"
    )
fi
if ! darwin_append_petsc_linalg_options; then
    linux_append_petsc_linalg_options
fi
# PyPI petsc's configure ignores MPIEXEC for the probe and defaults to /usr/bin/mpiexec
# (broken under Ubuntu alternatives). --with-mpiexec forces the working launcher.
if [[ -n "${MPIEXEC:-}" ]]; then
    _PETSC_CONF_OPTS+=("--with-mpiexec=${MPIEXEC}")
    log "PETSc configure: --with-mpiexec=${MPIEXEC}"
fi
export PETSC_CONFIGURE_OPTIONS="${_PETSC_CONF_OPTS[*]}"
if [ -n "${_PETSC_HYDRA_BOOTSTRAP}" ]; then
    log "PETSc pip: HYDRA_BOOTSTRAP=${_PETSC_HYDRA_BOOTSTRAP} (PETSc/SLEPc configure only)"
    env HYDRA_BOOTSTRAP="${_PETSC_HYDRA_BOOTSTRAP}" ${PIP_INSTALL} "${PETSC_SPEC}"
else
    log "PETSc pip: HYDRA_BOOTSTRAP unset (LINSTABPY_HYDRA_BOOTSTRAP is empty — use on MPICH without \"exec\" bootstrap)"
    ${PIP_INSTALL} "${PETSC_SPEC}"
fi
# Avoid leaking HYDRA_BOOTSTRAP into later mpiexec / Python (many MPICH builds reject launcher \"exec\").
unset HYDRA_BOOTSTRAP 2>/dev/null || true
install_mpi4py_compat

# Locate PETSc installation directory and export for SLEPc build
PETSC_DIR=$("${PY_BIN}" -c "import petsc; print(petsc.get_petsc_dir())")
PETSC_ARCH=""  # PyPI wheels use un-arched layout
export PETSC_DIR PETSC_ARCH

log "PETSC_DIR = ${PETSC_DIR}"

# ── Step 3: Patch PETSc config files ─────────────────────────────────────────
# PyPI wheels embed the build-cache path.  We replace it with the actual path.
log "Patching PETSc configuration files..."

PETSC_CONF_DIR="${PETSC_DIR}/lib/petsc/conf"
if [ -f "${PETSC_CONF_DIR}/petscvariables" ]; then
    # Find the stale build-cache path baked into petscvariables
    STALE_PATH=$(grep -m1 'PETSC_DIR = ' "${PETSC_CONF_DIR}/petscvariables" | sed -E 's|.*= ||' || true)
    if [ -n "${STALE_PATH}" ] && [ "${STALE_PATH}" != "${PETSC_DIR}" ]; then
        log "  Replacing stale path: ${STALE_PATH}"
        log "  With actual path:     ${PETSC_DIR}"

        # Patch all text config files under lib/petsc/conf/
        while IFS= read -r -d '' f; do
            if file "$f" | grep -q text; then
                if grep -q "${STALE_PATH}" "$f" 2>/dev/null; then
                    sed -i.bak "s|${STALE_PATH}|${PETSC_DIR}|g" "$f"
                    rm -f "${f}.bak"
                fi
            fi
        done < <(find "${PETSC_CONF_DIR}" -type f -print0)

        # Also patch include/petscconf.h (PETSC_PYTHON_EXE)
        PETSCCONF_H="${PETSC_DIR}/include/petscconf.h"
        if [ -f "${PETSCCONF_H}" ]; then
            sed -i.bak "s|${STALE_PATH}|${PETSC_DIR}|g" "${PETSCCONF_H}"
            rm -f "${PETSCCONF_H}.bak"
        fi
    else
        ok "PETSc config paths already correct."
    fi

    # Ensure PETSc Python executables point to python3 (not a build-env python)
    sed -i.bak \
        -e 's|^PETSC_PYTHON_EXE = .*|PETSC_PYTHON_EXE = python3|' \
        -e 's|^PYTHON = .*|PYTHON = python3|' \
        -e 's|^PYTHON_EXE = .*|PYTHON_EXE = "python3"|' \
        "${PETSC_CONF_DIR}/petscvariables"
    rm -f "${PETSC_CONF_DIR}/petscvariables.bak"
    if [ -f "${PETSC_DIR}/include/petscconf.h" ]; then
        sed -i.bak 's|#define PETSC_PYTHON_EXE .*|#define PETSC_PYTHON_EXE "python3"|' "${PETSC_DIR}/include/petscconf.h"
        rm -f "${PETSC_DIR}/include/petscconf.h.bak"
    fi
else
    warn "PETSc config directory not found at ${PETSC_CONF_DIR}"
fi
PETSC_PKGCONFIG_DIR="${PETSC_DIR}/lib/pkgconfig"
PETSC_PC="${PETSC_PKGCONFIG_DIR}/PETSc.pc"
if [ -f "${PETSC_PC}" ]; then
    STALE_PETSC_PC_PATH=$(sed -n -E 's|^prefix=(.*)|\1|p' "${PETSC_PC}" | head -n1)
    if [ -n "${STALE_PETSC_PC_PATH}" ] && [ "${STALE_PETSC_PC_PATH}" != "${PETSC_DIR}" ]; then
        log "  Replacing stale PETSc pkg-config path: ${STALE_PETSC_PC_PATH}"
        while IFS= read -r -d '' f; do
            if file "$f" | grep -q text; then
                if grep -q "${STALE_PETSC_PC_PATH}" "$f" 2>/dev/null; then
                    sed -i.bak "s|${STALE_PETSC_PC_PATH}|${PETSC_DIR}|g" "$f"
                    rm -f "${f}.bak"
                fi
            fi
        done < <(find "${PETSC_PKGCONFIG_DIR}" -type f -print0)
    fi
    sed -i.bak "s|^prefix=.*|prefix=${PETSC_DIR}|" "${PETSC_PC}"
    rm -f "${PETSC_PC}.bak"
fi

# Install SLEPc after PETSc path patching so configure can link PETSc properly.
log "Installing SLEPc wheel..."
export SLEPC_CONFIGURE_OPTIONS="--with-clean"
if [ -n "${_PETSC_HYDRA_BOOTSTRAP}" ]; then
    env HYDRA_BOOTSTRAP="${_PETSC_HYDRA_BOOTSTRAP}" ${PIP_INSTALL} "${SLEPC_SPEC}"
else
    ${PIP_INSTALL} "${SLEPC_SPEC}"
fi
unset HYDRA_BOOTSTRAP 2>/dev/null || true
SLEPC_DIR=$("${PY_BIN}" -c "import slepc; print(slepc.get_slepc_dir())")
export SLEPC_DIR
log "SLEPC_DIR = ${SLEPC_DIR}"

# PyPI SLEPc installs can retain build-cache paths in make/pkg-config
# metadata. Patch them before building slepc4py or user examples.
log "Patching SLEPc configuration files..."
SLEPC_CONF_DIR="${SLEPC_DIR}/lib/slepc/conf"
if [ -f "${SLEPC_CONF_DIR}/slepc_variables" ]; then
    STALE_SLEPC_PATH=$(grep -m1 '^SLEPC_LIB_DIR[[:space:]]*=' "${SLEPC_CONF_DIR}/slepc_variables" | sed -E 's|.*=[[:space:]]*||; s|/lib$||' || true)
    if [ -n "${STALE_SLEPC_PATH}" ] && [ "${STALE_SLEPC_PATH}" != "${SLEPC_DIR}" ]; then
        log "  Replacing stale path: ${STALE_SLEPC_PATH}"
        log "  With actual path:     ${SLEPC_DIR}"

        while IFS= read -r -d '' f; do
            if file "$f" | grep -q text; then
                if grep -q "${STALE_SLEPC_PATH}" "$f" 2>/dev/null; then
                    sed -i.bak "s|${STALE_SLEPC_PATH}|${SLEPC_DIR}|g" "$f"
                    rm -f "${f}.bak"
                fi
            fi
        done < <(find "${SLEPC_DIR}" -type f -print0)
    else
        ok "SLEPc config paths already correct."
    fi

    if [ -f "${SLEPC_CONF_DIR}/slepcvariables" ]; then
        sed -i.bak "s|^SLEPC_INSTALLDIR = .*|SLEPC_INSTALLDIR = ${SLEPC_DIR}|" "${SLEPC_CONF_DIR}/slepcvariables"
        rm -f "${SLEPC_CONF_DIR}/slepcvariables.bak"
    fi
else
    warn "SLEPc config directory not found at ${SLEPC_CONF_DIR}"
fi
SLEPC_PC="${SLEPC_DIR}/lib/pkgconfig/SLEPc.pc"
if [ -f "${SLEPC_PC}" ]; then
    sed -i.bak "s|^prefix=.*|prefix=${SLEPC_DIR}|" "${SLEPC_PC}"
    rm -f "${SLEPC_PC}.bak"
fi

# ── Step 4: Fix dylib rpaths (macOS only) ────────────────────────────────────
if [ "${OS_NAME}" = "Darwin" ]; then
    log "Fixing dylib install names and rpaths (macOS)..."
    PETSC_LIB="${PETSC_DIR}/lib"

    # Fix PETSc and dependency dylibs
    for dylib in "${PETSC_LIB}"/*.dylib; do
        [ -f "$dylib" ] || continue
        basename_lib=$(basename "$dylib")

        # Fix the install name to use @rpath
        current_id=$(otool -D "$dylib" 2>/dev/null | tail -1)
        if [[ "${current_id}" != *"@rpath"* ]]; then
            install_name_tool -id "@rpath/${basename_lib}" "$dylib" 2>/dev/null || true
        fi

        # Fix references to other dylibs in this directory
        otool -L "$dylib" 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r dep; do
            dep_base=$(basename "$dep")
            if [ -f "${PETSC_LIB}/${dep_base}" ] && [[ "${dep}" != "/usr/"* ]] && [[ "${dep}" != "/System/"* ]]; then
                new_dep="@rpath/${dep_base}"
                [[ "${dep_base}" == libpetsc*.dylib ]] && new_dep="@rpath/libpetsc.dylib"
                if [[ "${dep}" != "${new_dep}" ]]; then
                    install_name_tool -change "$dep" "${new_dep}" "$dylib" 2>/dev/null || true
                fi
            fi
        done
    done

    # Add rpaths to petsc4py/slepc4py .so files (done after they're installed below)
fi

# ── Step 5: Install petsc4py and slepc4py ────────────────────────────────────
log "Installing build dependencies for petsc4py/slepc4py..."
${PIP_INSTALL} "${BUILD_DEPS[@]}"

log "Installing petsc4py..."
${PIP_INSTALL} --no-build-isolation "${PETSC4PY_SPEC}"

log "Installing slepc4py..."
export SLEPC_DIR
${PIP_INSTALL} --no-build-isolation "${SLEPC4PY_SPEC}"

# Make sure the linker can find ScaLAPACK and other PETSc/SLEPc libraries
export LD_LIBRARY_PATH="${PETSC_DIR}/lib:${SLEPC_DIR}/lib:${LD_LIBRARY_PATH:-}"

# ── Step 6: Fix rpaths for Python extension modules (macOS) ──────────────────
if [ "${OS_NAME}" = "Darwin" ]; then
    log "Adding rpaths to Python extension modules..."
    SITE_PACKAGES=$("${PY_BIN}" -c "import site; print(site.getsitepackages()[0])")
    SLEPC_LIB="${SLEPC_DIR}/lib"

    # Add rpaths so petsc4py/slepc4py .so files can find PETSc/SLEPc dylibs
    while IFS= read -r -d '' so_file; do
        # Add PETSc lib dir as rpath if not already present
        if ! otool -l "$so_file" 2>/dev/null | grep -q "${PETSC_LIB}"; then
            install_name_tool -add_rpath "${PETSC_LIB}" "$so_file" 2>/dev/null || true
        fi
        # Add SLEPc lib dir as rpath if not already present
        if ! otool -l "$so_file" 2>/dev/null | grep -q "${SLEPC_LIB}"; then
            install_name_tool -add_rpath "${SLEPC_LIB}" "$so_file" 2>/dev/null || true
        fi
        # Fix references to PETSc/SLEPc dylibs
        otool -L "$so_file" 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r dep; do
            dep_base=$(basename "$dep")
            new_dep=""
            if [ -f "${PETSC_LIB}/${dep_base}" ]; then
                new_dep="@rpath/${dep_base}"
                [[ "${dep_base}" == libpetsc*.dylib ]] && new_dep="@rpath/libpetsc.dylib"
            elif [ -f "${SLEPC_LIB}/${dep_base}" ]; then
                new_dep="@rpath/${dep_base}"
                [[ "${dep_base}" == libslepc*.dylib ]] && new_dep="@rpath/libslepc.dylib"
            fi
            if [ -n "${new_dep}" ] && [[ "${dep}" != "${new_dep}" ]] && [[ "${dep}" != "/usr/"* ]] && [[ "${dep}" != "/System/"* ]]; then
                install_name_tool -change "$dep" "${new_dep}" "$so_file" 2>/dev/null || true
            fi
        done
    done < <(find "${SITE_PACKAGES}" \( -path "*/petsc4py/*" -o -path "*/slepc4py/*" \) -name "*.so" -print0)

    log "Normalizing SLEPc dylib install names and dependencies..."
    for dylib in "${SLEPC_LIB}"/*.dylib; do
        [ -f "$dylib" ] || continue
        dylib_base=$(basename "$dylib")
        install_name_tool -id "@rpath/${dylib_base}" "$dylib" 2>/dev/null || true
        otool -l "$dylib" 2>/dev/null | grep -q "${PETSC_LIB}" || install_name_tool -add_rpath "${PETSC_LIB}" "$dylib" 2>/dev/null || true
        otool -l "$dylib" 2>/dev/null | grep -q "${SLEPC_LIB}" || install_name_tool -add_rpath "${SLEPC_LIB}" "$dylib" 2>/dev/null || true
        otool -L "$dylib" 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r dep; do
            dep_base=$(basename "$dep")
            new_dep=""
            if [ -f "${PETSC_LIB}/${dep_base}" ]; then
                new_dep="@rpath/${dep_base}"
                [[ "${dep_base}" == libpetsc*.dylib ]] && new_dep="@rpath/libpetsc.dylib"
            elif [ -f "${SLEPC_LIB}/${dep_base}" ]; then
                new_dep="@rpath/${dep_base}"
                [[ "${dep_base}" == libslepc*.dylib ]] && new_dep="@rpath/libslepc.dylib"
            fi
            if [ -n "${new_dep}" ] && [[ "${dep}" != "${new_dep}" ]] && [[ "${dep}" != "/usr/"* ]] && [[ "${dep}" != "/System/"* ]]; then
                install_name_tool -change "$dep" "${new_dep}" "$dylib" 2>/dev/null || true
            fi
        done
    done
fi

# ── Step 7: Install linstabpy in editable mode ──────────────────────────────
if [ -f "${ROOT_DIR}/pyproject.toml" ]; then
    log "Installing linstabpy (editable)..."
    ${PIP_INSTALL} --no-deps -e "${ROOT_DIR}"
else
    log "No pyproject.toml (minimal bundle); using .pth fallback."
    _sp_fallback="$(printf '%s\n' "${VENV_DIR}"/lib/python*/site-packages | head -n 1)"
    if [ -z "${_sp_fallback}" ] || [ ! -d "${_sp_fallback}" ]; then
        err "Could not locate site-packages for .pth fallback."
        exit 1
    fi
    printf '%s\n' "${ROOT_DIR}/src" > "${_sp_fallback}/linstabpy_src_fallback.pth"
    ok "Installed .pth fallback: ${ROOT_DIR}/src -> ${_sp_fallback}/linstabpy_src_fallback.pth"
fi

# h5py is optional (older distro HDF5, Song2023 LNSE, and many workflows do not need it).
if [ "${LINSTABPY_WITH_H5PY:-0}" = "1" ]; then
    log "Installing h5py (LINSTABPY_WITH_H5PY=1)..."
    configure_linux_hdf5_env
    if ${PIP_INSTALL} "${H5PY_SPEC}"; then
        ok "h5py installed."
    else
        warn "h5py installation failed; continuing without it."
        warn "Install a newer HDF5 or build h5py against HDF5 >= 1.10.7, then re-run with LINSTABPY_WITH_H5PY=1."
    fi
else
    log "Skipping h5py (set LINSTABPY_WITH_H5PY=1 for MATLAB v7.3 / HDF5 tooling and related tests)."
fi

# Record DSO search path for mpi4py + PETSc wheels. sitecustomize prepends this
# before any import (pytest / tools that bypass the bin/python shell wrapper).
if _mpi_ld_install="$(linstabpy_resolve_mpi_libdir 2>/dev/null)"; then
    ok "Resolved MPI libdir for venv: ${_mpi_ld_install}"
else
    warn "Could not resolve system MPI libdir; mpi4py may need LD_LIBRARY_PATH if imports fail."
    _mpi_ld_install=""
fi
: > "${VENV_DIR}/linstabpy_ld_library_path"
[ -n "${_mpi_ld_install}" ] && printf '%s\n' "${_mpi_ld_install}" >> "${VENV_DIR}/linstabpy_ld_library_path"
printf '%s\n' "${PETSC_DIR}/lib" >> "${VENV_DIR}/linstabpy_ld_library_path"
printf '%s\n' "${SLEPC_DIR}/lib" >> "${VENV_DIR}/linstabpy_ld_library_path"

# Workaround: Python 3.13 (uv standalone builds) skips .pth files during startup.
# Setuptools' __editable__*.pth files are never processed, breaking src-layout
# editable installs. sitecustomize also applies LD_LIBRARY_PATH for mpi4py PETSc.
for _sp in "${VENV_DIR}"/lib/python*/site-packages; do
    [ -d "$_sp" ] || continue
    _sc="${_sp}/sitecustomize.py"
    cat > "$_sc" << 'PYEOF'
"""linstabpy venv hooks (install.sh): LD_LIBRARY_PATH + editable .pth workaround."""
from __future__ import annotations

import os
import sys

_sp = os.path.dirname(os.path.abspath(__file__))
_vroot = os.path.abspath(os.path.join(_sp, os.pardir, os.pardir, os.pardir))
_ldfile = os.path.join(_vroot, "linstabpy_ld_library_path")
if os.path.isfile(_ldfile):
    _extra: list[str] = []
    with open(_ldfile, encoding="utf-8") as _fh:
        for _line in _fh:
            _d = _line.strip()
            if _d and os.path.isdir(_d):
                _extra.append(_d)
    if _extra:
        os.environ["LD_LIBRARY_PATH"] = ":".join(_extra) + ":" + os.environ.get(
            "LD_LIBRARY_PATH", ""
        )

for _f in os.listdir(_sp):
    if _f.endswith(".pth") and "linstabpy" in _f:
        _path = open(os.path.join(_sp, _f), encoding="utf-8").read().strip()
        if os.path.isdir(_path) and _path not in sys.path:
            sys.path.insert(0, _path)
        break
PYEOF
    log "Wrote sitecustomize (.pth + LD_LIBRARY_PATH for mpi4py/PETSc)"
done

# ── Step 8: Install dev dependencies ─────────────────────────────────────────
log "Installing dev dependencies (pytest, sympy)..."
${PIP_INSTALL} "${DEV_DEPS[@]}"

# Persist runtime library paths in the venv activation script.
_mpi_activate=""
_mpi_activate="$(linstabpy_resolve_mpi_libdir 2>/dev/null || true)"
if ! grep -q "LINSTABPY_RUNTIME_LD" "${VENV_DIR}/bin/activate" 2>/dev/null; then
    {
        echo ""
        echo "# LINSTABPY_RUNTIME_LD (added by install.sh; MPI + PETSc for mpi4py / PETSc wheels)"
        echo "export LD_LIBRARY_PATH=\"${_mpi_activate:+${_mpi_activate}:}${PETSC_DIR}/lib:\${LD_LIBRARY_PATH}\""
    } >> "${VENV_DIR}/bin/activate"
fi
# ── Verification ─────────────────────────────────────────────────────────────
echo
if [[ "${LINSTABPY_INSTALL_SKIP_VERIFY:-0}" = "1" ]]; then
    warn "Skipping post-install verification (LINSTABPY_INSTALL_SKIP_VERIFY=1)."
    warn "Run manually: ${PY_BIN} -c \"import mpi4py.MPI; from petsc4py import PETSc; print(PETSc.Sys.getVersion())\""
else
log "Verifying installation..."
linstabpy_prepare_verify_mpiwrap
_verify_snippet="$(mktemp_py)"
cat >"${_verify_snippet}" <<'PY'
import sys
import numpy as np
import scipy
import matplotlib

print(f"  python       = {sys.version.split()[0]}", flush=True)
print(f"  numpy        = {np.__version__}", flush=True)
print(f"  scipy        = {scipy.__version__}", flush=True)
print(f"  matplotlib   = {matplotlib.__version__}", flush=True)

print("  mpi4py.MPI (before petsc4py) …", flush=True)
import mpi4py.MPI  # noqa: F401 — required before petsc4py on many MPICH systems

print("  petsc4py …", flush=True)
from petsc4py import PETSc
v = PETSc.Sys.getVersion()
scalar = "complex" if np.issubdtype(PETSc.ScalarType, np.complexfloating) else "real"
print(f"  petsc4py     = {v[0]}.{v[1]}.{v[2]}  (scalar={scalar})", flush=True)
assert scalar == "complex", \
    "FATAL: PETSc scalar type is real, must be complex for stability analysis"

print("  slepc4py …", flush=True)
from slepc4py import SLEPc
v = SLEPc.Sys.getVersion()
print(f"  slepc4py     = {v[0]}.{v[1]}.{v[2]}", flush=True)

print("  SLEPc NHEP   …", flush=True)
mat = PETSc.Mat().createAIJ([2, 2], nnz=2, comm=PETSc.COMM_SELF)
mat.setUp()
mat[0, 0] = 1.0
mat[0, 1] = 2.0
mat[1, 1] = 3.0
mat.assemble()
eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
eps.setOperators(mat)
eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
eps.setDimensions(1)
eps.solve()
if eps.getConverged() < 1:
    print("FATAL: SLEPc NHEP check converged no eigenpairs", file=sys.stderr)
    sys.exit(1)
eps.destroy()
mat.destroy()

import mpi4py
print(f"  mpi4py       = {mpi4py.__version__}", flush=True)

print("  linstabpy …", flush=True)
import linstabpy
print(f"  linstabpy    = {linstabpy.__version__}", flush=True)

# GPU-stack hard gate: a build that drops OpenMP/CUDA/STRUMPACK makes STRUMPACK
# run serially with the GPU idle (the 2026-06-25 regression). Fail the install
# loudly rather than ship a silently CPU-only "GPU" venv.
import os as _os
if _os.environ.get("LINSTABPY_EXPECT_GPU") == "1":
    print("  GPU-stack build config (OpenMP / CUDA / STRUMPACK / SLATE) …", flush=True)
    import petsc4py as _p4
    _pdir = _p4.get_config().get("PETSC_DIR", "")
    _conf_h = _os.path.join(_pdir, "include", "petscconf.h")
    with open(_conf_h) as _fh:
        _conf = _fh.read()
    _missing = [name for name, tok in (
        ("OpenMP", "PETSC_HAVE_OPENMP"),
        ("CUDA", "PETSC_HAVE_CUDA"),
        ("STRUMPACK", "PETSC_HAVE_STRUMPACK"),
        ("MUMPS", "PETSC_HAVE_MUMPS"),
        # SLATE: GPU-capable ScaLAPACK replacement. Absent → STRUMPACK does the large
        # distributed dense fronts on CPU (root-front crash at -n>1, idle GPU at -n1).
        ("SLATE", "PETSC_HAVE_SLATE"),
    ) if f"#define {tok} 1" not in _conf]
    if _missing:
        print(
            "FATAL: GPU stack PETSc is missing " + ", ".join(_missing)
            + ". Without SLATE the large dense fronts fall back to CPU ScaLAPACK "
            "(root-front crash at multi-rank, idle GPU at single rank); without "
            "OpenMP STRUMPACK runs serially. "
            "Rebuild with: bash install.sh --stack gpu-strumpack --force",
            file=sys.stderr,
        )
        sys.exit(1)
    print("    OpenMP + CUDA + STRUMPACK + SLATE present ✓", flush=True)

print()
print("  All checks passed.", flush=True)
PY
# Tell the verify snippet to hard-gate OpenMP/CUDA/STRUMPACK for non-default
# (GPU) stacks, so a serial/OpenMP-less STRUMPACK build can never ship silently.
if [ "${BUILD_STACK}" != "default" ]; then
    export LINSTABPY_EXPECT_GPU=1
fi
linstabpy_run_verify "${PY_BIN}" "${_verify_snippet}"
_verify_rc=$?
rm -f "${_verify_snippet}"
if [ "${_verify_rc}" -ne 0 ]; then
    err "Verification failed (exit ${_verify_rc})."
    exit "${_verify_rc}"
fi

_cpu_count="$(linstabpy_cpu_count)"
if [ "${OS_NAME}" = "Linux" ] && [ "${_cpu_count}" -gt 1 ]; then
    if ! linstabpy_prepare_multirank_verify_mpiwrap 2; then
        err "Verification failed: no working 2-rank mpiexec launcher was found."
        err "Try: HYDRA_BOOTSTRAP=fork ${MPIEXEC:-mpiexec} -n 2 /bin/true"
        exit 1
    fi
    _verify_multirank_snippet="$(mktemp_py)"
    cat >"${_verify_multirank_snippet}" <<'PY'
import mpi4py.MPI as MPI
from petsc4py import PETSc

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
petsc_size = PETSc.COMM_WORLD.getSize()

assert size == 2, f"expected mpi4py COMM_WORLD size 2, got {size}"
assert petsc_size == 2, f"expected PETSc COMM_WORLD size 2, got {petsc_size}"

vec = PETSc.Vec().createMPI(4, comm=PETSc.COMM_WORLD)
start, end = vec.getOwnershipRange()
for idx in range(start, end):
    vec.setValue(idx, rank + 1)
vec.assemble()
expected_vec_sum = MPI.COMM_WORLD.allreduce((end - start) * (rank + 1), op=MPI.SUM)
actual_vec_sum = vec.sum()
assert abs(actual_vec_sum - expected_vec_sum) < 1e-12, (
    f"PETSc distributed Vec sum mismatch: got {actual_vec_sum}, expected {expected_vec_sum}"
)
vec.destroy()

mat = PETSc.Mat().createAIJ([4, 4], nnz=1, comm=PETSc.COMM_WORLD)
mat.setUp()
row_start, row_end = mat.getOwnershipRange()
for idx in range(row_start, row_end):
    mat.setValue(idx, idx, 1.0)
mat.assemble()
b = PETSc.Vec().createMPI(4, comm=PETSc.COMM_WORLD)
b.set(1.0)
b.assemble()
x = b.duplicate()
ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
ksp.setOperators(mat)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.JACOBI)
ksp.solve(b, x)
assert ksp.getConvergedReason() > 0, f"PETSc KSP did not converge: {ksp.getConvergedReason()}"
assert abs(x.sum() - 4.0) < 1e-12, f"PETSc distributed KSP solution sum mismatch: {x.sum()}"
ksp.destroy()
x.destroy()
b.destroy()
mat.destroy()

if rank == 0:
    print(f"  mpi4py multirank = OK  (size={size})", flush=True)
    print(f"  petsc4py multirank = OK  (size={petsc_size})", flush=True)
    print(f"  petsc4py distributed Vec = OK  (sum={actual_vec_sum})", flush=True)
    print("  petsc4py distributed KSP = OK", flush=True)
PY
    linstabpy_run_verify_wrapped \
        "${PY_BIN}" \
        "${_verify_multirank_snippet}" \
        "${LINSTAB_VERIFY_MULTIENV[@]}" \
        "${LINSTAB_VERIFY_MULTIWRAP[@]}"
    _verify_multirank_rc=$?
    rm -f "${_verify_multirank_snippet}"
    if [ "${_verify_multirank_rc}" -ne 0 ]; then
        err "Multi-rank verification failed (exit ${_verify_multirank_rc})."
        err "Try: HYDRA_BOOTSTRAP=fork ${MPIEXEC:-mpiexec} -n 2 ${PY_BIN} -c 'import mpi4py.MPI; from petsc4py import PETSc'"
        exit "${_verify_multirank_rc}"
    fi
fi
fi

echo
ok "Setup complete. Activate with: source ${VENV_DISPLAY}/bin/activate"
ok "Optional alias: echo \"alias linstabpy-activate='source ${VENV_DIR}/bin/activate'\" >> ~/.bashrc"
ok "Quick check:  bash install.sh --check"
ok "Run tests:    ${VENV_DISPLAY}/bin/python -m pytest tests/ -v"
