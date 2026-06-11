# SPDX-License-Identifier: GPL-3.0-only
"""Runtime provenance collection for persisted result artifacts."""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
from pathlib import Path

_PROVENANCE_KEYS = ("git_sha", "config_md5", "hostname", "compute_mode", "mpi_stack")
_VALID_COMPUTE_MODES = {"cpu", "gpu", "mixed"}


def _git_head_sha() -> str:
    """Return the current git HEAD SHA, or ``"unknown"`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _config_md5(config_path: str | Path | None) -> str:
    """Return the MD5 hex digest of raw config bytes, or ``""`` if absent."""
    if config_path is None:
        return ""
    path = Path(config_path)
    if not path.is_file():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - provenance only


def _compute_mode_from_env() -> str:
    mode = os.getenv("LINSTABPY_COMPUTE_MODE", "cpu").strip() or "cpu"
    if mode not in _VALID_COMPUTE_MODES:
        valid = ", ".join(sorted(_VALID_COMPUTE_MODES))
        raise ValueError(f"invalid LINSTABPY_COMPUTE_MODE={mode!r}; expected one of: {valid}")
    return mode


def collect_provenance(config_path: str | Path | None) -> dict[str, str]:
    """Collect the five provenance fields stored in runtime artifacts."""
    provenance = {
        "git_sha": _git_head_sha(),
        "config_md5": _config_md5(config_path),
        "hostname": socket.gethostname(),
        "compute_mode": _compute_mode_from_env(),
        "mpi_stack": os.getenv("LINSTABPY_MPI_STACK", "unknown").strip() or "unknown",
    }
    assert tuple(provenance) == _PROVENANCE_KEYS
    return provenance
