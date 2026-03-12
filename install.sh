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
#   bash install.sh          # full install
#   bash install.sh --check  # verify existing installation
#
# The PETSc/SLEPc PyPI wheels embed absolute build-time paths in their config
# files and dylibs.  This script patches them automatically.
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
OS_NAME=$(uname -s)
PYTHON_VERSION="3.13"
PETSC_VERSION="3.24.2"
SLEPC_VERSION="3.24.2"
PETSC4PY_VERSION="3.24.2"
SLEPC4PY_VERSION="3.24.2"

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$1"; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$1"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$1"; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }

linux_install_system_deps() {
    [ "${OS_NAME}" = "Linux" ] || return 0
    [ "${LINSTAB2D_SKIP_SYS_DEPS:-0}" = "1" ] && {
        warn "Skipping Linux system dependency installation (LINSTAB2D_SKIP_SYS_DEPS=1)."
        return 0
    }

    local required=(curl make gcc g++ gfortran pkg-config mpicc mpicxx mpifort)
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
        install_cmd="apt-get install -y build-essential gfortran pkg-config curl ca-certificates mpich libmpich-dev libx11-dev"
    elif has_cmd dnf; then
        pm="dnf"
        install_cmd="dnf install -y gcc gcc-c++ gcc-gfortran make pkgconf-pkg-config curl ca-certificates mpich mpich-devel libX11-devel"
    elif has_cmd yum; then
        pm="yum"
        install_cmd="yum install -y gcc gcc-c++ gcc-gfortran make pkgconfig curl ca-certificates mpich mpich-devel libX11-devel"
    elif has_cmd pacman; then
        pm="pacman"
        update_cmd="pacman -Sy --noconfirm"
        install_cmd="pacman -S --noconfirm --needed base-devel gcc-fortran pkgconf curl ca-certificates mpich libx11"
    elif has_cmd zypper; then
        pm="zypper"
        install_cmd="zypper --non-interactive install gcc gcc-c++ gcc-fortran make pkg-config curl ca-certificates mpich mpich-devel libX11-devel"
    else
        err "Unsupported Linux distro/package manager. Install manually: curl make gcc g++ gfortran pkg-config MPI dev tools (mpicc/mpicxx/mpifort) and X11 headers."
        exit 1
    fi

    log "Installing Linux system dependencies via ${pm}..."
    if [ "$(id -u)" -eq 0 ]; then
        [ -n "${update_cmd}" ] && bash -lc "${update_cmd}"
        bash -lc "${install_cmd}"
    elif has_cmd sudo; then
        [ -n "${update_cmd}" ] && sudo bash -lc "${update_cmd}"
        sudo bash -lc "${install_cmd}"
    else
        err "Need root privileges to install system dependencies (sudo not found). Re-run as root."
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

# Ensure reference MATLAB data/code is present locally for local inspection.
if [ -d "${ROOT_DIR}/.git" ]; then
    if ! git -C "${ROOT_DIR}" submodule status linstab2d/data/baseflows >/dev/null 2>&1; then
        warn "Initializing MATLAB reference submodule..."
        git -C "${ROOT_DIR}" submodule update --init --recursive
    fi
fi

# ── Quick verification mode ──────────────────────────────────────────────────
if [[ "${1:-}" == "--check" ]]; then
    if [ ! -d "${VENV_DIR}" ]; then
        err "No virtual environment found at ${VENV_DIR}"
        exit 1
    fi
    "${VENV_DIR}/bin/python" - <<'PY'
import sys, numpy as np
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
    exit $?
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

PY_BIN="${VENV_DIR}/bin/python"
PIP_INSTALL="${UV_BIN} pip install --python ${PY_BIN}"

# ── Step 1: Core dependencies ────────────────────────────────────────────────
log "Installing core dependencies..."
${PIP_INSTALL} "numpy>=1.23" "scipy>=1.10" "matplotlib>=3.7"

# ── Step 2: MPI + PETSc + SLEPc C libraries ─────────────────────────────────
log "Installing mpi4py and PETSc wheels..."
unset PETSC_DIR PETSC_ARCH SLEPC_DIR 2>/dev/null || true
export PETSC_CONFIGURE_OPTIONS="--with-scalar-type=complex --download-mumps --download-scalapack --download-metis --download-parmetis"
${PIP_INSTALL} mpi4py "petsc==${PETSC_VERSION}"

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
${PIP_INSTALL} "slepc==${SLEPC_VERSION}"
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
log "Installing linstab2d (editable)..."
${PIP_INSTALL} -e "${ROOT_DIR}"

# Workaround: Python 3.13 (uv standalone builds) skips .pth files during startup.
# Setuptools' __editable__*.pth files are never processed, breaking src-layout
# editable installs. We add a sitecustomize.py that reads the .pth content manually.
for _sp in "${VENV_DIR}"/lib/python*/site-packages; do
    [ -d "$_sp" ] || continue
    _sc="${_sp}/sitecustomize.py"
    if [ ! -f "$_sc" ] || ! grep -q 'linstab2d' "$_sc" 2>/dev/null; then
        cat > "$_sc" << 'PYEOF'
"""Add src/ to sys.path for editable install (Python 3.13 .pth workaround)."""
import os, sys
_sp = os.path.dirname(os.path.abspath(__file__))
for _f in os.listdir(_sp):
    if _f.endswith('.pth') and 'linstab2d' in _f:
        _path = open(os.path.join(_sp, _f)).read().strip()
        if os.path.isdir(_path) and _path not in sys.path:
            sys.path.insert(0, _path)
        break
PYEOF
        log "Created sitecustomize.py (.pth workaround for Python 3.13)"
    fi
done

# ── Step 8: Install dev dependencies ─────────────────────────────────────────
log "Installing dev dependencies (pytest, sympy)..."
${PIP_INSTALL} pytest pytest-cov "sympy>=1.13"

# Persist the setting in the virtual environment activation script
if ! grep -q "PETSC_DIR/lib" "${VENV_DIR}/bin/activate" 2>/dev/null; then
    echo "export LD_LIBRARY_PATH=\"${PETSC_DIR}/lib:\${LD_LIBRARY_PATH}\"" >> "${VENV_DIR}/bin/activate"
fi
# ── Verification ─────────────────────────────────────────────────────────────
echo
log "Verifying installation..."
"${PY_BIN}" - <<'PY'
import sys
import numpy as np
import scipy
import matplotlib

print(f"  python       = {sys.version.split()[0]}")
print(f"  numpy        = {np.__version__}")
print(f"  scipy        = {scipy.__version__}")
print(f"  matplotlib   = {matplotlib.__version__}")

from petsc4py import PETSc
v = PETSc.Sys.getVersion()
scalar = "complex" if np.issubdtype(PETSc.ScalarType, np.complexfloating) else "real"
print(f"  petsc4py     = {v[0]}.{v[1]}.{v[2]}  (scalar={scalar})")
assert scalar == "complex", \
    "FATAL: PETSc scalar type is real, must be complex for stability analysis"

from slepc4py import SLEPc
v = SLEPc.Sys.getVersion()
print(f"  slepc4py     = {v[0]}.{v[1]}.{v[2]}")

import mpi4py
print(f"  mpi4py       = {mpi4py.__version__}")

import linstab2d
print(f"  linstab2d    = {linstab2d.__version__}")

print()
print("  All checks passed.")
PY

echo
ok "Setup complete. Activate with: source .venv/bin/activate"
ok "Quick check:  bash install.sh --check"
ok "Run tests:    .venv/bin/python -m pytest tests/ -v"
