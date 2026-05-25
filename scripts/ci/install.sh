#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${LINSTAB2D_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ID="${LINSTAB2D_INSTALL_RUN_ID:-preflight}"
LOG_DIR="${LINSTAB2D_INSTALL_LOG_DIR:-${SCRIPT_DIR}/install_runs}"

err() { printf '[install_tester] ERROR: %s\n' "$1" >&2; }

require_repo_root() {
    if [ ! -f "${REPO_ROOT}/pyproject.toml" ] || [ ! -d "${REPO_ROOT}/src/linstabpy" ]; then
        err "LINSTAB2D_REPO_ROOT does not point at a LinStabPy checkout: ${REPO_ROOT}"
        err "Run from a full checkout or export LINSTAB2D_REPO_ROOT=/path/to/linstabpy."
        exit 1
    fi
    if [ ! -f "${REPO_ROOT}/install.sh" ]; then
        err "Missing canonical installer at ${REPO_ROOT}/install.sh."
        exit 1
    fi
}

write_preflight_snapshot() {
    local run_dir="${LOG_DIR}/${RUN_ID}"
    local snapshot="${run_dir}/machine_versions.txt"
    mkdir -p "${run_dir}"
    {
        printf 'timestamp_utc=%s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        printf 'repo_root=%s\n' "${REPO_ROOT}"
        printf 'script_dir=%s\n' "${SCRIPT_DIR}"
        printf 'uname=%s\n' "$(uname -a 2>/dev/null || true)"
        if command -v git >/dev/null 2>&1; then
            if git_toplevel="$(git -C "${REPO_ROOT}" rev-parse --show-toplevel 2>/dev/null)"; then
                printf 'git_toplevel=%s\n' "${git_toplevel}"
            fi
            if git_head="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null)"; then
                printf 'git_head=%s\n' "${git_head}"
            fi
            git -C "${REPO_ROOT}" status --short --branch 2>/dev/null | sed 's/^/git_status=/' || true
        fi
        for tool in bash zsh git uv python3 mpicc mpicxx mpifort mpiexec cmake gcc g++ gfortran h5cc; do
            if command -v "${tool}" >/dev/null 2>&1; then
                printf '%s_path=%s\n' "${tool}" "$(command -v "${tool}")"
                "${tool}" --version 2>/dev/null | head -n 1 | sed "s/^/${tool}_version=/" || true
            else
                printf '%s_path=\n' "${tool}"
            fi
        done
    } > "${snapshot}"
    printf '[install_tester] wrote %s\n' "${snapshot}"
}

require_repo_root

if [[ "${1:-}" == "--preflight" ]]; then
    write_preflight_snapshot
    exit 0
fi

exec bash "${REPO_ROOT}/install.sh" "$@"
