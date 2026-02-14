#!/usr/bin/env bash
set -euo pipefail

# LinStab2D setup bootstrap.
# Creates a project virtual environment and installs required Python packages.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
OS_NAME=$(uname -s)
REQUIRED_PACKAGES=(numpy scipy matplotlib scikit-umfpack)

log() {
    printf '[install] %s\n' "$1"
}

log "Detected OS: ${OS_NAME}"

if ! command -v uv >/dev/null 2>&1; then
    log "uv not found: installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ -x "${HOME}/.local/bin/uv" ]; then
        UV_BIN="${HOME}/.local/bin/uv"
    else
        UV_BIN="$(command -v uv || true)"
    fi
else
    UV_BIN="$(command -v uv)"
fi

if [ -z "${UV_BIN}" ]; then
    log "Unable to locate uv binary after install attempt."
    exit 1
fi

log "Using uv at: ${UV_BIN}"

if [ ! -d "${VENV_DIR}" ]; then
    log "Creating virtual environment at ${VENV_DIR}"
    "${UV_BIN}" venv "${VENV_DIR}"
else
    log "Using existing virtual environment at ${VENV_DIR}"
fi

log "Installing packages: ${REQUIRED_PACKAGES[*]}"
source "${VENV_DIR}/bin/activate"
"${UV_BIN}" pip install "${REQUIRED_PACKAGES[@]}"

echo
log "Version summary:"
python - <<'PY'
import importlib.metadata
import numpy as np
import scipy
import matplotlib

print(f"python={__import__('sys').version.split()[0]}")
print(f"numpy={np.__version__}")
print(f"scipy={scipy.__version__}")
print(f"matplotlib={matplotlib.__version__}")
try:
    print(f"scikit-umfpack={importlib.metadata.version('scikit-umfpack')}")
except Exception:
    print("scikit-umfpack=not_installed")
PY

echo
log "Setup complete. Activate with: source .venv/bin/activate"
