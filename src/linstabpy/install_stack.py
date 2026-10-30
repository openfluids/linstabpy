# SPDX-License-Identifier: GPL-3.0-only
"""Build the CPU solver stack (complex PETSc/SLEPc with MUMPS) into this environment.

``pip`` cannot supply it: PyPI has PETSc and SLEPc only as source archives.
This runs the same pip builds that ``install.sh`` runs, minus its host checks.
It needs a C compiler, a Fortran compiler, ``make`` and an MPI with compiler
wrappers (``mpicc``, ``mpifort``): the MPI wheels on PyPI have no Fortran
bindings, and MUMPS needs them. PETSc downloads and builds BLAS (OpenBLAS),
ScaLAPACK, METIS, ParMETIS and MUMPS. GPU stacks, cluster MPI modules and macOS
tuning stay with ``install.sh``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Sequence
from pathlib import Path

PETSC_VERSION = "3.24.2"

# One token per option: the PETSc source archive splits this variable on spaces.
CONFIGURE_OPTIONS = (
    "--with-scalar-type=complex",
    "--with-debugging=0",
    "--download-cmake",
    "--download-mumps",
    "--download-scalapack",
    "--download-metis",
    "--download-parmetis",
    "--download-openblas",
    "--COPTFLAGS=-O3",
    "--CXXOPTFLAGS=-O3",
    "--FOPTFLAGS=-O3",
)

# Cython 3.3 rejects the petsc4py 3.24 sources.
BUILD_DEPS = ("numpy>=1.23", "Cython>=3.0,<3.3", "setuptools<72", "wheel")

CHECK = (
    "import numpy as np\n"
    "from petsc4py import PETSc\n"
    "from slepc4py import SLEPc\n"
    "assert np.issubdtype(PETSc.ScalarType, np.complexfloating), PETSc.ScalarType\n"
    "assert PETSc.Sys.hasExternalPackage('mumps'), 'PETSc has no MUMPS'\n"
    "print('PETSc', PETSc.Sys.getVersion(), 'SLEPc', SLEPc.Sys.getVersion(), 'complex, MUMPS')\n"
)


def _missing_tools() -> list[str]:
    needed = {
        "C compiler": ("cc", "gcc"),
        "Fortran compiler": ("gfortran",),
        "make": ("make",),
        "MPI C wrapper (mpicc)": ("mpicc",),
        "MPI Fortran wrapper (mpifort)": ("mpifort", "mpif90"),
    }
    return [
        label for label, names in needed.items() if not any(shutil.which(name) for name in names)
    ]


def _pip(*args: str, env: dict[str, str], fresh: bool = False) -> list[str]:
    """Return a pip command for the running interpreter (uv when it is installed)."""
    if shutil.which("uv"):
        return [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            *(("--no-cache",) if fresh else ()),
            *args,
        ]
    return [sys.executable, "-m", "pip", "install", *(("--no-cache-dir",) if fresh else ()), *args]


def _rewrite(files, old: str, new: str) -> None:
    """Replace ``old`` by ``new`` in every text file of ``files``."""
    for path in files:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if old in text:
            path.write_text(text.replace(old, new))


def _stale_prefix(variables: Path, pattern: str) -> str | None:
    match = (
        re.search(pattern, variables.read_text(), flags=re.MULTILINE)
        if variables.is_file()
        else None
    )
    return match.group(1).strip() if match else None


def _relocate_petsc(prefix: str) -> None:
    """Point the PETSc config files at the installed prefix instead of the build directory."""
    root = Path(prefix)
    stale = _stale_prefix(root / "lib/petsc/conf/petscvariables", r"^PETSC_DIR = (.*)$")
    if stale and stale != prefix:
        files = [*(root / "lib/petsc/conf").rglob("*"), *(root / "lib/pkgconfig").rglob("*")]
        _rewrite([*files, root / "include/petscconf.h"], stale, prefix)
    variables = root / "lib/petsc/conf/petscvariables"
    if variables.is_file():
        text = re.sub(
            r"^(PETSC_PYTHON_EXE|PYTHON) = .*$",
            r"\1 = python3",
            variables.read_text(),
            flags=re.MULTILINE,
        )
        variables.write_text(text)


def _relocate_slepc(prefix: str) -> None:
    """Point the SLEPc config files at the installed prefix instead of the build directory."""
    root = Path(prefix)
    stale = _stale_prefix(
        root / "lib/slepc/conf/slepc_variables", r"^SLEPC_LIB_DIR\s*=\s*(.*?)/lib$"
    )
    if stale and stale != prefix:
        _rewrite(root.rglob("*"), stale, prefix)
    for path, pattern, repl in (
        (
            root / "lib/slepc/conf/slepcvariables",
            r"^SLEPC_INSTALLDIR = .*$",
            f"SLEPC_INSTALLDIR = {prefix}",
        ),
        (root / "lib/pkgconfig/SLEPc.pc", r"^prefix=.*$", f"prefix={prefix}"),
    ):
        if path.is_file():
            path.write_text(re.sub(pattern, repl, path.read_text(), flags=re.MULTILINE))


Step = tuple[str, "list[str] | Callable[[], None]", dict[str, str]]


def _steps(env: dict[str, str]) -> list[Step]:
    site = sysconfig.get_paths()["purelib"]
    petsc_dir = os.path.join(site, "petsc")
    slepc_dir = os.path.join(site, "slepc")
    mpicc = shutil.which("mpicc", path=env.get("PATH"))
    build = {"PETSC_CONFIGURE_OPTIONS": " ".join(CONFIGURE_OPTIONS)}
    with_petsc = {"PETSC_DIR": petsc_dir, "MPICC": mpicc or "mpicc"}
    isolated = ("--no-build-isolation",)
    return [
        ("build tools", _pip(*BUILD_DEPS, env=env), {}),
        (
            "PETSc (compiles OpenBLAS, ScaLAPACK, MUMPS: 20 to 40 min)",
            _pip(*isolated, f"petsc=={PETSC_VERSION}", env=env, fresh=True),
            build,
        ),
        ("fix the PETSc build paths", lambda: _relocate_petsc(petsc_dir), {}),
        (
            "mpi4py",
            _pip(*isolated, "--no-binary", "mpi4py", "mpi4py>=3.1", env=env, fresh=True),
            with_petsc,
        ),
        (
            "petsc4py",
            _pip(*isolated, f"petsc4py=={PETSC_VERSION}", env=env, fresh=True),
            with_petsc,
        ),
        (
            "SLEPc",
            _pip(*isolated, f"slepc=={PETSC_VERSION}", env=env, fresh=True),
            {**with_petsc, "SLEPC_CONFIGURE_OPTIONS": "--with-clean"},
        ),
        ("fix the SLEPc build paths", lambda: _relocate_slepc(slepc_dir), {}),
        (
            "slepc4py",
            _pip(*isolated, f"slepc4py=={PETSC_VERSION}", env=env, fresh=True),
            {**with_petsc, "SLEPC_DIR": slepc_dir},
        ),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="linstabpy install-stack",
        description=(
            "Build complex-scalar PETSc/SLEPc with MUMPS into the current Python "
            "environment. Needs a C compiler, gfortran and make."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="print the steps and stop")
    parser.add_argument(
        "--force", action="store_true", help="rebuild although petsc4py is installed"
    )
    args = parser.parse_args(argv)

    if importlib.util.find_spec("petsc4py") and not (args.force or args.dry_run):
        print("petsc4py is already installed in this environment; use --force to rebuild it.")
        return 0

    missing = _missing_tools()
    if missing:
        print(f"Missing on this machine: {', '.join(missing)}.", file=sys.stderr)
        print(
            "Debian/Ubuntu: sudo apt install build-essential gfortran libopenmpi-dev openmpi-bin",
            file=sys.stderr,
        )
        print("macOS: xcode-select --install && brew install gcc open-mpi", file=sys.stderr)
        print("A cluster: load its compiler and MPI modules, then run this again.", file=sys.stderr)
        return 1

    env = {k: v for k, v in os.environ.items() if k not in {"PETSC_DIR", "PETSC_ARCH", "SLEPC_DIR"}}
    for label, command, extra in _steps(env):
        print(f"[install-stack] {label}", flush=True)
        if args.dry_run:
            print(
                "  "
                + (" ".join(command) if isinstance(command, list) else "(rewrites config files)")
            )
            continue
        if callable(command):
            command()
        else:
            subprocess.run(command, check=True, env={**env, **extra})
    if args.dry_run:
        return 0
    subprocess.run([sys.executable, "-c", CHECK], check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
