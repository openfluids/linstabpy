# SPDX-License-Identifier: GPL-3.0-only
"""Runtime provenance collection for persisted result artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import socket
import subprocess
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

_PROVENANCE_KEYS = (
    "git_sha",
    "config_md5",
    "hostname",
    "compute_mode",
    "mpi_stack",
    "python_version",
    "numpy_version",
    "scipy_version",
    "petsc4py_version",
    "slepc4py_version",
    "petsc_arch",
    "mumps_version",
    "mpi_impl",
)
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


def _probe_string(probe: Callable[[], str]) -> str:
    """Return a non-empty probe string, or ``"unavailable"`` on failure."""
    try:
        value = str(probe()).strip()
    except Exception:
        return "unavailable"
    return value or "unavailable"


def _package_version(distribution: str) -> str:
    return _probe_string(lambda: metadata.version(distribution))


def _petsc_arch() -> str:
    def probe() -> str:
        import petsc4py

        config = petsc4py.get_config()
        arch = str(config.get("PETSC_ARCH") or os.getenv("PETSC_ARCH") or "").strip()
        if arch:
            return arch
        petsc_dir = Path(str(config.get("PETSC_DIR", "")))
        config_info = petsc_dir / "include" / "petscconfiginfo.h"
        if config_info.is_file():
            match = re.search(r"PETSC_ARCH=([^\s\"']+)", config_info.read_text())
            if match:
                return match.group(1)
        return "unavailable"

    return _probe_string(probe)


def _mumps_version() -> str:
    def probe() -> str:
        import petsc4py
        from petsc4py import PETSc

        if not PETSc.Sys.hasExternalPackage("mumps"):
            return "unavailable"
        petsc_dir = Path(str(petsc4py.get_config().get("PETSC_DIR", "")))
        for header in (
            petsc_dir / "include" / "zmumps_c.h",
            petsc_dir / "include" / "dmumps_c.h",
            petsc_dir / "include" / "cmumps_c.h",
            petsc_dir / "include" / "smumps_c.h",
        ):
            if not header.is_file():
                continue
            match = re.search(r'#define\s+MUMPS_VERSION\s+"([^"]+)"', header.read_text())
            if match:
                return match.group(1)
        return "unavailable"

    return _probe_string(probe)


def _mpi_impl() -> str:
    def probe() -> str:
        from mpi4py import MPI

        return MPI.Get_library_version().splitlines()[0]

    return _probe_string(probe)


def collect_provenance(config_path: str | Path | None) -> dict[str, str]:
    """Collect provenance fields stored in runtime artifacts."""
    provenance = {
        "git_sha": _git_head_sha(),
        "config_md5": _config_md5(config_path),
        "hostname": socket.gethostname(),
        "compute_mode": _compute_mode_from_env(),
        "mpi_stack": os.getenv("LINSTABPY_MPI_STACK", "unknown").strip() or "unknown",
        "python_version": sys.version.split()[0],
        "numpy_version": _package_version("numpy"),
        "scipy_version": _package_version("scipy"),
        "petsc4py_version": _package_version("petsc4py"),
        "slepc4py_version": _package_version("slepc4py"),
        "petsc_arch": _petsc_arch(),
        "mumps_version": _mumps_version(),
        "mpi_impl": _mpi_impl(),
    }
    assert tuple(provenance) == _PROVENANCE_KEYS
    return provenance
