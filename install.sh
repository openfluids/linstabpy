#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# linstab2d bootstrap installer
# =============================================================================
#
# Creates a Python 3.13 virtual environment and installs all dependencies
# including PETSc/SLEPc with complex-scalar support and MUMPS direct solver.
#
# Usage:
#   bash install.sh                    # full install (no h5py by default)
#   LINSTAB2D_WITH_H5PY=1 bash install.sh   # also install h5py (needs HDF5 >= 1.10.7)
#   LINSTAB2D_AUTO_INSTALL_SYS_DEPS=1 bash install.sh  # allow package-manager installs
#   bash install.sh --check            # verify existing installation
#
# On Linux, missing system packages do not trigger sudo/package-manager calls by
# default. The script prints the suggested commands and exits so cluster/module
# environments can load the right toolchain first.
#
# If PETSc source configure fails with "Unable to run MPI program with /usr/bin/mpiexec":
#   LINSTAB2D_PETSC_MPIEXEC=/usr/bin/mpiexec.mpich bash install.sh
#
# If mpi4py fails to import (e.g. ImportError: libmpi.so.40 while the system has MPICH):
#   The installer tries a wheel first, then rebuilds mpi4py from source automatically.
#   To force a source build: LINSTAB2D_MPI4PY_NO_BINARY=1 bash install.sh
#
# If verification hangs after "mpi4py.MPI …" on MPICH: plain ``python`` can block in MPI_Init.
#   The script runs checks under ``mpiexec -n 1``. Override with LINSTAB2D_VERIFY_MPIEXEC=/path/to/mpiexec.
#
# HYDRA_BOOTSTRAP=exec avoids some PETSc configure stalls but is not supported by all MPICH builds
# (error: unrecognized launcher: exec). This script applies it only while installing PETSc/SLEPc
# (not for mpiexec verification). Override with LINSTAB2D_HYDRA_BOOTSTRAP=fork or
# LINSTAB2D_HYDRA_BOOTSTRAP=  (empty, unset) if PETSc configure still misbehaves.
#
# The PETSc/SLEPc PyPI wheels embed absolute build-time paths in their config
# files and dylibs.  This script patches them automatically.
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
OS_NAME=$(uname -s)
PYTHON_VERSION="${LINSTAB2D_PYTHON_VERSION:-3.13}"
PETSC_VERSION="3.24.2"
SLEPC_VERSION="3.24.2"
PETSC4PY_VERSION="3.24.2"
SLEPC4PY_VERSION="3.24.2"

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$1"; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$1"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$1"; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }

# Print a directory containing libmpi.so* / libmpich.so* (Open MPI, MPICH, Intel MPI)
# or return non-zero.
linstab2d_resolve_mpi_libdir() {
    local mpicc_bin=""
    local d=""
    local so=""
    # libmpi.so* covers Open MPI and Intel MPI; libmpich.so* covers MPICH on Debian/Ubuntu.
    local _has_mpi_so='ls "${1}"/libmpi.so* >/dev/null 2>&1 || ls "${1}"/libmpich.so* >/dev/null 2>&1'
    _dir_has_mpi_so() { ls "${1}"/libmpi.so* >/dev/null 2>&1 || ls "${1}"/libmpich.so* >/dev/null 2>&1; }
    if [ -x /usr/bin/mpicc ]; then
        mpicc_bin="/usr/bin/mpicc"
    elif has_cmd mpicc; then
        mpicc_bin="$(command -v mpicc)"
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
        d="$(printf '%s\n' "${d}" | sed -e 's|///*|/|g' -e 's|/$||')"
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
# Usage: linstab2d_run_verify <py_bin> <heredoc_path>
linstab2d_run_verify() {
    linstab2d_run_verify_wrapped "$1" "$2" "${LINSTAB_VERIFY_MPIWRAP[@]}"
}

linstab2d_run_verify_wrapped() {
    local py_bin="$1"; shift
    local snippet_file="$1"; shift
    local _err _rc
    _err="$(mktemp)"
    if [ "$#" -gt 0 ] && [[ "$1" == *=* ]]; then
        env "$@" "${py_bin}" "${snippet_file}" 2>"${_err}"
    else
        "$@" "${py_bin}" "${snippet_file}" 2>"${_err}"
    fi
    _rc=$?
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

    local submodule_path="src/linstab2d/data/baseflows"
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
    if [ -f "${py_bin}" ] && grep -q 'LINSTAB2D Linux runtime wrapper' "${py_bin}" 2>/dev/null; then
        return 0
    fi

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

# mpi4py links against the system MPI runtime (e.g. Open MPI libmpi.so.*).
# Prefer /usr/bin/mpicc, then PATH; verify with ldconfig if needed.
_mpicc=""
[ -x /usr/bin/mpicc ] && _mpicc="/usr/bin/mpicc"
[ -z "${_mpicc}" ] && command -v mpicc >/dev/null 2>&1 && _mpicc="$(command -v mpicc)"
_mpi_ld=""
if [ -n "${_mpicc}" ]; then
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
if [ -z "${_mpi_ld}" ]; then
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
if [ -z "${_mpi_ld}" ] && command -v ldconfig >/dev/null 2>&1; then
    _so="$(ldconfig -p 2>/dev/null | awk '/libmpi\.so\.[0-9]/{print $NF; exit}')"
    if [ -n "${_so}" ] && [ -f "${_so}" ]; then
        _mpi_ld="$(dirname "${_so}")"
    fi
fi

export LD_LIBRARY_PATH="${_mpi_ld:+${_mpi_ld}:}${_petsc_lib}:${_slepc_lib}:${LD_LIBRARY_PATH:-}"
exec -a "${_script_dir}/python" "${_script_dir}/python-real" "$@"
EOF
    chmod +x "${py_bin}"

    for link_name in python3 "python${PYTHON_VERSION}"; do
        ln -sfn python "${VENV_DIR}/bin/${link_name}"
    done

    ok "Installed Linux runtime wrapper for .venv/bin/python"
}

linux_install_system_deps() {
    [ "${OS_NAME}" = "Linux" ] || return 0
    [ "${LINSTAB2D_SKIP_SYS_DEPS:-0}" = "1" ] && {
        warn "Skipping Linux system dependency installation (LINSTAB2D_SKIP_SYS_DEPS=1)."
        return 0
    }

    local required=(curl make cmake gcc g++ gfortran pkg-config h5cc mpicc mpicxx mpifort)
    local missing=()
    for c in "${required[@]}"; do
        has_cmd "$c" || missing+=("$c")
    done

    if [ ${#missing[@]} -eq 0 ]; then
        ok "Linux build prerequisites already installed."
        return 0
    fi

    warn "Missing Linux build tools: ${missing[*]}"

    local pm=""
    local update_cmd=""
    local install_cmd=""
    if has_cmd apt-get; then
        pm="apt"
        update_cmd="apt-get update"
        install_cmd="apt-get install -y build-essential cmake gfortran pkg-config curl ca-certificates mpich libmpich-dev libx11-dev libhdf5-dev"
    elif has_cmd dnf; then
        pm="dnf"
        install_cmd="dnf install -y gcc gcc-c++ gcc-gfortran make cmake pkgconf-pkg-config curl ca-certificates mpich mpich-devel libX11-devel hdf5-devel"
    elif has_cmd yum; then
        pm="yum"
        install_cmd="yum install -y gcc gcc-c++ gcc-gfortran make cmake pkgconfig curl ca-certificates mpich mpich-devel libX11-devel hdf5-devel"
    elif has_cmd pacman; then
        pm="pacman"
        update_cmd="pacman -Sy --noconfirm"
        install_cmd="pacman -S --noconfirm --needed base-devel cmake gcc-fortran pkgconf curl ca-certificates mpich libx11 hdf5"
    elif has_cmd zypper; then
        pm="zypper"
        install_cmd="zypper --non-interactive install gcc gcc-c++ gcc-fortran make cmake pkg-config curl ca-certificates mpich mpich-devel libX11-devel hdf5-devel"
    else
        err "Unsupported Linux distro/package manager. Install manually: curl make cmake gcc g++ gfortran pkg-config HDF5 dev (h5cc/libhdf5) MPI dev tools (mpicc/mpicxx/mpifort) and X11 headers."
        exit 1
    fi

    if [ "${LINSTAB2D_AUTO_INSTALL_SYS_DEPS:-0}" != "1" ]; then
        err "Refusing to install system packages automatically."
        err "This is intentional so cluster/module environments do not trigger unexpected sudo/package-manager calls."
        warn "Run 'bash diagnose.sh' first, load the required modules or install the packages manually, then re-run install.sh."
        warn "If you really want the installer to call the package manager, re-run with LINSTAB2D_AUTO_INSTALL_SYS_DEPS=1."
        warn "Suggested commands:"
        [ -n "${update_cmd}" ] && printf '  %s\n' "${update_cmd}"
        printf '  %s\n' "${install_cmd}"
        exit 1
    fi
    log "Installing Linux system dependencies via ${pm} (LINSTAB2D_AUTO_INSTALL_SYS_DEPS=1)..."
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
    if [ ${#missing[@]} -gt 0 ]; then
        err "System dependency install incomplete; still missing: ${missing[*]}"
        exit 1
    fi
    ok "Linux system dependencies installed."
}

configure_linux_petsc_toolchain() {
    [ "${OS_NAME}" = "Linux" ] || return 0

    # Force GNU toolchain so PETSc does not accidentally pick Intel/Conda MPI wrappers.
    export CC="${CC:-gcc}"
    export CXX="${CXX:-g++}"
    export FC="${FC:-gfortran}"
    export F77="${F77:-gfortran}"
    export F90="${F90:-gfortran}"

    # Prefer system MPI wrappers when present.
    if [ -x /usr/bin/mpicc ] && [ -x /usr/bin/mpicxx ] && [ -x /usr/bin/mpifort ]; then
        export MPICC=/usr/bin/mpicc
        export MPICXX=/usr/bin/mpicxx
        export MPIFORT=/usr/bin/mpifort
        export MPI_DIR=/usr
    fi

    # Hint common MPI stacks to use GNU Fortran under the hood.
    export OMPI_FC="${OMPI_FC:-gfortran}"
    export I_MPI_F90="${I_MPI_F90:-gfortran}"

    ok "Configured Linux PETSc toolchain (GNU Fortran)."
}

# Echo absolute path to an mpiexec that successfully runs -n 1 /bin/true, or return 1.
linstab2d_find_mpiexec_for_n1() {
    local candidate resolved
    for candidate in mpiexec.mpich mpiexec.hydra mpirun.mpich mpirun.openmpi mpiexec.openmpi mpiexec; do
        has_cmd "$candidate" || continue
        resolved="$(command -v "$candidate")"
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

linstab2d_probe_mpi_launch() {
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
    if [[ -n "${LINSTAB2D_PETSC_MPIEXEC:-}" ]]; then
        export MPIEXEC="${LINSTAB2D_PETSC_MPIEXEC}"
        log "PETSc/SLEPc configure: using LINSTAB2D_PETSC_MPIEXEC=${MPIEXEC}"
        return 0
    fi
    local resolved
    if ! resolved="$(linstab2d_find_mpiexec_for_n1)"; then
        err "No working mpiexec for PETSc configure. Try: mpiexec.mpich -n 1 /bin/true"
        err "Or set LINSTAB2D_PETSC_MPIEXEC=/path/to/mpiexec and re-run install.sh"
        exit 1
    fi
    export MPIEXEC="${resolved}"
    log "PETSc/SLEPc configure: MPIEXEC=${MPIEXEC} (verified: -n 1 /bin/true)"
}

# On Linux MPICH/Hydra, ``import mpi4py.MPI`` in a plain python process can hang in MPI_Init.
# Populate LINSTAB_VERIFY_MPIWRAP as: empty or ( mpiexec -n 1 ).
linstab2d_prepare_verify_mpiwrap() {
    LINSTAB_VERIFY_MPIWRAP=()
    [ "${OS_NAME}" = "Linux" ] || return 0
    # Drop inherited HYDRA_BOOTSTRAP (e.g. conda / old shells): Hydra \"exec\" is unsupported on many distros.
    unset HYDRA_BOOTSTRAP 2>/dev/null || true
    local mx="${LINSTAB2D_VERIFY_MPIEXEC:-${MPIEXEC:-}}"
    if [ -z "${mx}" ]; then
        mx="$(linstab2d_find_mpiexec_for_n1 || true)"
    fi
    if [ -n "${mx}" ]; then
        LINSTAB_VERIFY_MPIWRAP=( "${mx}" -n 1 )
        log "Verification will use: ${mx} -n 1 (avoids MPICH singleton hang on mpi4py import)"
    else
        warn "No mpiexec found for verification. If the verify step hangs at mpi4py, set LINSTAB2D_VERIFY_MPIEXEC=/usr/bin/mpiexec.mpich"
    fi
}

linstab2d_prepare_multirank_verify_mpiwrap() {
    local ranks="$1"
    LINSTAB_VERIFY_MULTIWRAP=()
    LINSTAB_VERIFY_MULTIENV=()
    [ "${OS_NAME}" = "Linux" ] || return 1
    unset HYDRA_BOOTSTRAP 2>/dev/null || true
    local mx="${LINSTAB2D_VERIFY_MPIEXEC:-${MPIEXEC:-}}"
    if [ -z "${mx}" ]; then
        mx="$(linstab2d_find_mpiexec_for_n1 || true)"
    fi
    [ -n "${mx}" ] || return 1

    if linstab2d_probe_mpi_launch "${mx}" -n "${ranks}" /bin/true; then
        LINSTAB_VERIFY_MULTIWRAP=( "${mx}" -n "${ranks}" )
        log "Multi-rank verification will use: ${mx} -n ${ranks}"
        return 0
    fi

    case "$(basename "${mx}")" in
        mpiexec.mpich|mpiexec.hydra|mpirun.mpich)
            if linstab2d_probe_mpi_launch env HYDRA_BOOTSTRAP=fork "${mx}" -n "${ranks}" /bin/true; then
                LINSTAB_VERIFY_MULTIENV=( HYDRA_BOOTSTRAP=fork )
                LINSTAB_VERIFY_MULTIWRAP=( "${mx}" -n "${ranks}" )
                log "Multi-rank verification will use: HYDRA_BOOTSTRAP=fork ${mx} -n ${ranks}"
                return 0
            fi
            ;;
    esac
    return 1
}

linstab2d_cpu_count() {
    if has_cmd getconf; then
        getconf _NPROCESSORS_ONLN 2>/dev/null && return 0
    fi
    if has_cmd nproc; then
        nproc 2>/dev/null && return 0
    fi
    printf '1\n'
}

install_mpi4py_compat() {
    # PyPI wheels often link libmpi.so.40 (Open MPI 4) while Debian/Ubuntu defaults
    # are MPICH (libmpi.so.12). Building from source with system mpicc fixes imports.
    if [ "${OS_NAME}" != "Linux" ]; then
        log "Installing mpi4py..."
        ${PIP_INSTALL} "mpi4py>=3.1"
        return 0
    fi
    if [[ "${LINSTAB2D_MPI4PY_NO_BINARY:-0}" = "1" ]]; then
        log "Installing mpi4py from source (LINSTAB2D_MPI4PY_NO_BINARY=1)..."
        ${PIP_INSTALL} --no-binary mpi4py "mpi4py>=3.1"
        return 0
    fi
    log "Installing mpi4py (binary wheel; falls back to source build if it does not match system MPI)..."
    ${PIP_INSTALL} "mpi4py>=3.1"
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
    ${PIP_INSTALL} --force-reinstall --no-binary mpi4py "mpi4py>=3.1"
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

# Ensure reference MATLAB data/code is present locally for local inspection.
ensure_reference_submodule

# ── Quick verification mode ──────────────────────────────────────────────────
if [[ "${1:-}" == "--check" ]]; then
    if [ ! -d "${VENV_DIR}" ]; then
        err "No virtual environment found at ${VENV_DIR}"
        exit 1
    fi
    linstab2d_prepare_verify_mpiwrap
    _check_snippet="$(mktemp --suffix=.py)"
    cat >"${_check_snippet}" <<'PY'
import sys, numpy as np
# mpi4py must bind MPI before petsc4py (same rule as linstab2d.utils.petsc_init).
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
print("All checks passed.")
PY
    linstab2d_run_verify "${VENV_DIR}/bin/python" "${_check_snippet}"
    _check_rc=$?
    rm -f "${_check_snippet}"
    if [ "${_check_rc}" -ne 0 ]; then
        exit "${_check_rc}"
    fi

    _cpu_count="$(linstab2d_cpu_count)"
    if [ "${OS_NAME}" = "Linux" ] && [ "${_cpu_count}" -gt 1 ]; then
        if ! linstab2d_prepare_multirank_verify_mpiwrap 2; then
            err "Quick check failed: no working 2-rank mpiexec launcher was found."
            err "Try: HYDRA_BOOTSTRAP=fork ${MPIEXEC:-mpiexec} -n 2 /bin/true"
            exit 1
        fi
        _check_multirank_snippet="$(mktemp --suffix=.py)"
        cat >"${_check_multirank_snippet}" <<'PY'
import mpi4py.MPI as MPI
from petsc4py import PETSc

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
petsc_size = PETSc.COMM_WORLD.getSize()

assert size == 2, f"expected mpi4py COMM_WORLD size 2, got {size}"
assert petsc_size == 2, f"expected PETSc COMM_WORLD size 2, got {petsc_size}"

if rank == 0:
    print(f"mpi4py_multirank=size={size}")
    print(f"petsc4py_multirank=size={petsc_size}")
PY
        linstab2d_run_verify_wrapped \
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

# ── Create/validate venv with Python 3.13 ────────────────────────────────────
if [ -d "${VENV_DIR}" ]; then
    EXISTING_PY="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [ "${EXISTING_PY}" != "${PYTHON_VERSION}" ]; then
        warn "Existing venv uses Python ${EXISTING_PY:-unknown}; recreating with Python ${PYTHON_VERSION}"
        rm -rf "${VENV_DIR}"
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

PY_BIN="${VENV_DIR}/bin/python"
PIP_INSTALL="${UV_BIN} pip install --python ${PY_BIN}"

# ── Step 1: Core dependencies ────────────────────────────────────────────────
log "Installing core dependencies..."
${PIP_INSTALL} "numpy>=1.23" "scipy>=1.10" "matplotlib>=3.7"

# ── Step 2: MPI + PETSc + SLEPc C libraries ─────────────────────────────────
log "Installing PETSc (sdist) then mpi4py..."
unset PETSC_DIR PETSC_ARCH SLEPC_DIR 2>/dev/null || true
configure_linux_petsc_toolchain
configure_petsc_build_mpiexec
# Optional Hydra bootstrap for PETSc/SLEPc configure only (not exported — breaks some mpiexec).
if [ "${LINSTAB2D_HYDRA_BOOTSTRAP+x}" = x ]; then
    _PETSC_HYDRA_BOOTSTRAP="${LINSTAB2D_HYDRA_BOOTSTRAP}"
else
    _PETSC_HYDRA_BOOTSTRAP=exec
fi
_PETSC_CONF_OPTS=(
    "--with-scalar-type=complex"
    "--download-cmake"
    "--download-fblaslapack"
    "--download-mumps"
    "--download-scalapack"
    "--download-metis"
    "--download-parmetis"
)
# PyPI petsc's configure ignores MPIEXEC for the probe and defaults to /usr/bin/mpiexec
# (broken under Ubuntu alternatives). --with-mpiexec forces the working launcher.
if [[ -n "${MPIEXEC:-}" ]]; then
    _PETSC_CONF_OPTS+=("--with-mpiexec=${MPIEXEC}")
    log "PETSc configure: --with-mpiexec=${MPIEXEC}"
fi
export PETSC_CONFIGURE_OPTIONS="${_PETSC_CONF_OPTS[*]}"
if [ -n "${_PETSC_HYDRA_BOOTSTRAP}" ]; then
    log "PETSc pip: HYDRA_BOOTSTRAP=${_PETSC_HYDRA_BOOTSTRAP} (PETSc/SLEPc configure only)"
    env HYDRA_BOOTSTRAP="${_PETSC_HYDRA_BOOTSTRAP}" ${PIP_INSTALL} "petsc==${PETSC_VERSION}"
else
    log "PETSc pip: HYDRA_BOOTSTRAP unset (LINSTAB2D_HYDRA_BOOTSTRAP is empty — use on MPICH without \"exec\" bootstrap)"
    ${PIP_INSTALL} "petsc==${PETSC_VERSION}"
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

# Install SLEPc after PETSc path patching so configure can link PETSc properly.
log "Installing SLEPc wheel..."
export SLEPC_CONFIGURE_OPTIONS="--with-clean"
if [ -n "${_PETSC_HYDRA_BOOTSTRAP}" ]; then
    env HYDRA_BOOTSTRAP="${_PETSC_HYDRA_BOOTSTRAP}" ${PIP_INSTALL} "slepc==${SLEPC_VERSION}"
else
    ${PIP_INSTALL} "slepc==${SLEPC_VERSION}"
fi
unset HYDRA_BOOTSTRAP 2>/dev/null || true
SLEPC_DIR=$("${PY_BIN}" -c "import slepc; print(slepc.get_slepc_dir())")
export SLEPC_DIR
log "SLEPC_DIR = ${SLEPC_DIR}"

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
            if [ -f "${PETSC_LIB}/${dep_base}" ] && [[ "${dep}" != *"@rpath"* ]] && [[ "${dep}" != "/usr/"* ]] && [[ "${dep}" != "/System/"* ]]; then
                install_name_tool -change "$dep" "@rpath/${dep_base}" "$dylib" 2>/dev/null || true
            fi
        done
    done

    # Add rpaths to petsc4py/slepc4py .so files (done after they're installed below)
fi

# ── Step 5: Install petsc4py and slepc4py ────────────────────────────────────
log "Installing build dependencies for petsc4py/slepc4py..."
${PIP_INSTALL} "Cython>=3.0" "setuptools<72"

log "Installing petsc4py..."
${PIP_INSTALL} --no-build-isolation "petsc4py==${PETSC4PY_VERSION}"

log "Installing slepc4py..."
export SLEPC_DIR
${PIP_INSTALL} --no-build-isolation "slepc4py==${SLEPC4PY_VERSION}"

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
            if { [ -f "${PETSC_LIB}/${dep_base}" ] || [ -f "${SLEPC_LIB}/${dep_base}" ]; } && [[ "${dep}" != *"@rpath"* ]] && [[ "${dep}" != "/usr/"* ]] && [[ "${dep}" != "/System/"* ]]; then
                install_name_tool -change "$dep" "@rpath/${dep_base}" "$so_file" 2>/dev/null || true
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
            if { [ -f "${PETSC_LIB}/${dep_base}" ] || [ -f "${SLEPC_LIB}/${dep_base}" ]; } && [[ "${dep}" != *"@rpath"* ]] && [[ "${dep}" != "/usr/"* ]] && [[ "${dep}" != "/System/"* ]]; then
                install_name_tool -change "$dep" "@rpath/${dep_base}" "$dylib" 2>/dev/null || true
            fi
        done
    done
fi

# ── Step 7: Install linstab2d in editable mode ──────────────────────────────
if [ -f "${ROOT_DIR}/pyproject.toml" ]; then
    log "Installing linstab2d (editable)..."
    ${PIP_INSTALL} --no-deps -e "${ROOT_DIR}"
else
    log "No pyproject.toml (minimal bundle); using .pth fallback."
    _sp_fallback="$(printf '%s\n' "${VENV_DIR}"/lib/python*/site-packages | head -n 1)"
    if [ -z "${_sp_fallback}" ] || [ ! -d "${_sp_fallback}" ]; then
        err "Could not locate site-packages for .pth fallback."
        exit 1
    fi
    printf '%s\n' "${ROOT_DIR}/src" > "${_sp_fallback}/linstab2d_src_fallback.pth"
    ok "Installed .pth fallback: ${ROOT_DIR}/src -> ${_sp_fallback}/linstab2d_src_fallback.pth"
fi

# h5py is optional (older distro HDF5, Song2023 LNSE, and many workflows do not need it).
if [ "${LINSTAB2D_WITH_H5PY:-0}" = "1" ]; then
    log "Installing h5py (LINSTAB2D_WITH_H5PY=1)..."
    configure_linux_hdf5_env
    if ${PIP_INSTALL} "h5py>=3.16"; then
        ok "h5py installed."
    else
        warn "h5py installation failed; continuing without it."
        warn "Install a newer HDF5 or build h5py against HDF5 >= 1.10.7, then re-run with LINSTAB2D_WITH_H5PY=1."
    fi
else
    log "Skipping h5py (set LINSTAB2D_WITH_H5PY=1 for MATLAB v7.3 / HDF5 tooling and related tests)."
fi

# Record DSO search path for mpi4py + PETSc wheels. sitecustomize prepends this
# before any import (pytest / tools that bypass the bin/python shell wrapper).
if _mpi_ld_install="$(linstab2d_resolve_mpi_libdir 2>/dev/null)"; then
    ok "Resolved MPI libdir for venv: ${_mpi_ld_install}"
else
    warn "Could not resolve system MPI libdir; mpi4py may need LD_LIBRARY_PATH if imports fail."
    _mpi_ld_install=""
fi
: > "${VENV_DIR}/linstab2d_ld_library_path"
[ -n "${_mpi_ld_install}" ] && printf '%s\n' "${_mpi_ld_install}" >> "${VENV_DIR}/linstab2d_ld_library_path"
printf '%s\n' "${PETSC_DIR}/lib" >> "${VENV_DIR}/linstab2d_ld_library_path"
printf '%s\n' "${SLEPC_DIR}/lib" >> "${VENV_DIR}/linstab2d_ld_library_path"

# Workaround: Python 3.13 (uv standalone builds) skips .pth files during startup.
# Setuptools' __editable__*.pth files are never processed, breaking src-layout
# editable installs. sitecustomize also applies LD_LIBRARY_PATH for mpi4py PETSc.
for _sp in "${VENV_DIR}"/lib/python*/site-packages; do
    [ -d "$_sp" ] || continue
    _sc="${_sp}/sitecustomize.py"
    cat > "$_sc" << 'PYEOF'
"""linstab2d venv hooks (install.sh): LD_LIBRARY_PATH + editable .pth workaround."""
from __future__ import annotations

import os
import sys

_vroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ldfile = os.path.join(_vroot, "linstab2d_ld_library_path")
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

_sp = os.path.dirname(os.path.abspath(__file__))
for _f in os.listdir(_sp):
    if _f.endswith(".pth") and "linstab2d" in _f:
        _path = open(os.path.join(_sp, _f), encoding="utf-8").read().strip()
        if os.path.isdir(_path) and _path not in sys.path:
            sys.path.insert(0, _path)
        break
PYEOF
    log "Wrote sitecustomize (.pth + LD_LIBRARY_PATH for mpi4py/PETSc)"
done

# ── Step 8: Install dev dependencies ─────────────────────────────────────────
log "Installing dev dependencies (pytest, sympy)..."
${PIP_INSTALL} pytest pytest-cov "sympy>=1.13"

# Persist runtime library paths in the venv activation script.
_mpi_activate=""
_mpi_activate="$(linstab2d_resolve_mpi_libdir 2>/dev/null || true)"
if ! grep -q "LINSTAB2D_RUNTIME_LD" "${VENV_DIR}/bin/activate" 2>/dev/null; then
    {
        echo ""
        echo "# LINSTAB2D_RUNTIME_LD (added by install.sh; MPI + PETSc for mpi4py / PETSc wheels)"
        echo "export LD_LIBRARY_PATH=\"${_mpi_activate:+${_mpi_activate}:}${PETSC_DIR}/lib:\${LD_LIBRARY_PATH}\""
    } >> "${VENV_DIR}/bin/activate"
fi
# ── Verification ─────────────────────────────────────────────────────────────
echo
if [[ "${LINSTAB2D_INSTALL_SKIP_VERIFY:-0}" = "1" ]]; then
    warn "Skipping post-install verification (LINSTAB2D_INSTALL_SKIP_VERIFY=1)."
    warn "Run manually: ${PY_BIN} -c \"import mpi4py.MPI; from petsc4py import PETSc; print(PETSc.Sys.getVersion())\""
else
log "Verifying installation..."
linstab2d_prepare_verify_mpiwrap
_verify_snippet="$(mktemp --suffix=.py)"
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

import mpi4py
print(f"  mpi4py       = {mpi4py.__version__}", flush=True)

print("  linstab2d …", flush=True)
import linstab2d
print(f"  linstab2d    = {linstab2d.__version__}", flush=True)

print()
print("  All checks passed.", flush=True)
PY
linstab2d_run_verify "${PY_BIN}" "${_verify_snippet}"
_verify_rc=$?
rm -f "${_verify_snippet}"
if [ "${_verify_rc}" -ne 0 ]; then
    err "Verification failed (exit ${_verify_rc})."
    exit "${_verify_rc}"
fi

_cpu_count="$(linstab2d_cpu_count)"
if [ "${OS_NAME}" = "Linux" ] && [ "${_cpu_count}" -gt 1 ]; then
    if ! linstab2d_prepare_multirank_verify_mpiwrap 2; then
        err "Verification failed: no working 2-rank mpiexec launcher was found."
        err "Try: HYDRA_BOOTSTRAP=fork ${MPIEXEC:-mpiexec} -n 2 /bin/true"
        exit 1
    fi
    _verify_multirank_snippet="$(mktemp --suffix=.py)"
    cat >"${_verify_multirank_snippet}" <<'PY'
import mpi4py.MPI as MPI
from petsc4py import PETSc

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
petsc_size = PETSc.COMM_WORLD.getSize()

assert size == 2, f"expected mpi4py COMM_WORLD size 2, got {size}"
assert petsc_size == 2, f"expected PETSc COMM_WORLD size 2, got {petsc_size}"

if rank == 0:
    print(f"  mpi4py multirank = OK  (size={size})", flush=True)
    print(f"  petsc4py multirank = OK  (size={petsc_size})", flush=True)
PY
    linstab2d_run_verify_wrapped \
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
ok "Setup complete. Activate with: source .venv/bin/activate"
ok "Quick check:  bash install.sh --check"
ok "Run tests:    .venv/bin/python -m pytest tests/ -v"
