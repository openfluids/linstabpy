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

# Ensure reference MATLAB data/code is present locally for local inspection.
if [ -d "${ROOT_DIR}/.git" ]; then
    if ! git -C "${ROOT_DIR}" submodule status linstab2d/data/baseflows >/dev/null 2>&1; then
        warn "Initializing MATLAB reference submodule..."
        git -C "${ROOT_DIR}" submodule update --init --recursive
    fi
fi

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$1"; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$1"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$1"; }

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

# ── Locate or install uv ─────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
UV_BIN="$(command -v uv)"
log "Using uv at: ${UV_BIN}"

# ── Create venv with Python 3.13 ─────────────────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    log "Creating virtual environment (Python ${PYTHON_VERSION}) at ${VENV_DIR}"
    "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
else
    log "Using existing virtual environment at ${VENV_DIR}"
fi

PY_BIN="${VENV_DIR}/bin/python"
PIP_INSTALL="${UV_BIN} pip install --python ${PY_BIN}"

# ── Step 1: Core dependencies ────────────────────────────────────────────────
log "Installing core dependencies..."
${PIP_INSTALL} "numpy>=1.23" "scipy>=1.10" "matplotlib>=3.7"

# ── Step 2: MPI + PETSc + SLEPc C libraries ─────────────────────────────────
log "Installing mpi4py, PETSc, and SLEPc wheels..."
export PETSC_CONFIGURE_OPTIONS="--with-scalar-type=complex --download-mumps --download-scalapack --download-metis --download-parmetis"
${PIP_INSTALL} mpi4py petsc slepc

# Locate PETSc and SLEPc installation directories
PETSC_DIR=$("${PY_BIN}" -c "import petsc; print(petsc.get_petsc_dir())")
SLEPC_DIR=$("${PY_BIN}" -c "import slepc; print(slepc.get_slepc_dir())")
PETSC_ARCH=""  # PyPI wheels use un-arched layout

log "PETSC_DIR = ${PETSC_DIR}"
log "SLEPC_DIR = ${SLEPC_DIR}"

export PETSC_DIR PETSC_ARCH

# ── Step 3: Patch PETSc config files ─────────────────────────────────────────
# PyPI wheels embed the build-cache path.  We replace it with the actual path.
log "Patching PETSc configuration files..."

PETSC_CONF_DIR="${PETSC_DIR}/lib/petsc/conf"
if [ -f "${PETSC_CONF_DIR}/petscvariables" ]; then
    # Find the stale build-cache path baked into petscvariables
    STALE_PATH=$(grep -m1 'PETSC_DIR = ' "${PETSC_CONF_DIR}/petscvariables" | sed 's/PETSC_DIR = //' || true)
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

    # Ensure PETSC_PYTHON_EXE points to python3 (not a build-env python)
    if grep -q 'PETSC_PYTHON_EXE' "${PETSC_CONF_DIR}/petscvariables" 2>/dev/null; then
        sed -i.bak 's|PETSC_PYTHON_EXE = .*|PETSC_PYTHON_EXE = python3|' "${PETSC_CONF_DIR}/petscvariables"
        rm -f "${PETSC_CONF_DIR}/petscvariables.bak"
    fi
    if [ -f "${PETSC_DIR}/include/petscconf.h" ]; then
        sed -i.bak 's|#define PETSC_PYTHON_EXE .*|#define PETSC_PYTHON_EXE "python3"|' "${PETSC_DIR}/include/petscconf.h"
        rm -f "${PETSC_DIR}/include/petscconf.h.bak"
    fi
else
    warn "PETSc config directory not found at ${PETSC_CONF_DIR}"
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
${PIP_INSTALL} --no-build-isolation petsc4py

log "Installing slepc4py..."
export SLEPC_DIR
${PIP_INSTALL} --no-build-isolation slepc4py

# ── Step 6: Fix rpaths for Python extension modules (macOS) ──────────────────
if [ "${OS_NAME}" = "Darwin" ]; then
    log "Adding rpaths to Python extension modules..."
    SITE_PACKAGES=$("${PY_BIN}" -c "import site; print(site.getsitepackages()[0])")

    # Add rpaths so petsc4py/slepc4py .so files can find PETSc/SLEPc dylibs
    for so_file in "${SITE_PACKAGES}"/petsc4py/**/*.so "${SITE_PACKAGES}"/slepc4py/**/*.so; do
        [ -f "$so_file" ] || continue
        # Add PETSc lib dir as rpath if not already present
        if ! otool -l "$so_file" 2>/dev/null | grep -q "${PETSC_LIB}"; then
            install_name_tool -add_rpath "${PETSC_LIB}" "$so_file" 2>/dev/null || true
        fi
        # Fix references to PETSc/SLEPc dylibs
        otool -L "$so_file" 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r dep; do
            dep_base=$(basename "$dep")
            if [ -f "${PETSC_LIB}/${dep_base}" ] && [[ "${dep}" != *"@rpath"* ]] && [[ "${dep}" != "/usr/"* ]] && [[ "${dep}" != "/System/"* ]]; then
                install_name_tool -change "$dep" "@rpath/${dep_base}" "$so_file" 2>/dev/null || true
            fi
        done
    done
fi

# ── Step 7: Install linstab2d in editable mode ──────────────────────────────
log "Installing linstab2d (editable)..."
${PIP_INSTALL} -e "${ROOT_DIR}"

# ── Step 8: Install dev dependencies ─────────────────────────────────────────
log "Installing dev dependencies (pytest)..."
${PIP_INSTALL} pytest pytest-cov

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
