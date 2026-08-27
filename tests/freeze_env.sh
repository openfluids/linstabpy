#!/usr/bin/env bash
# Write a pip freeze file for the repository virtual environment.
# FREEZE_ENV_PYTHON overrides the interpreter, so a checkout without its
# own .venv (a test tree, a CI clone) can freeze the environment in use.
set -euo pipefail

version="${1:-dev}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python="${FREEZE_ENV_PYTHON:-${repo_root}/.venv/bin/python}"
output="constraints-${version}.txt"

if [[ ! -x "${python}" ]]; then
  echo "error: expected executable venv python at ${python}" >&2
  exit 1
fi

if "${python}" -m pip --version >/dev/null 2>&1; then
  "${python}" -m pip freeze > "${output}"
elif command -v uv >/dev/null 2>&1; then
  uv pip freeze --python "${python}" > "${output}"
else
  echo "error: neither ${python} -m pip nor uv is available for environment freeze" >&2
  exit 1
fi
printf 'wrote %s\n' "${output}"
