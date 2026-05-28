# SPDX-License-Identifier: GPL-3.0-only
"""PETSc/SLEPc initialization helpers.

Ensures petsc4py/slepc4py parse process argv exactly once so command-line
options (e.g. ``-eps_monitor``, ``-st_ksp_*``) reach the solver stack.
The top-level package import stays PETSc-free; solver modules call these
helpers only when a PETSc-backed path is imported or executed.
"""

from __future__ import annotations

import sys

_PETSC_INIT_DONE = False
_SLEPC_INIT_DONE = False


def _already_initialized(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "already initialized" in msg or "already been initialized" in msg


def _sanitize_petsc_argv(argv: list[str]) -> list[str]:
    """Keep PETSc/SLEPc options, drop Python launcher and script argparse args.

    Rules:
    - Drop Python launcher flags (`-c`, `-m`) and their payload.
    - Drop GNU-style long options (`--foo`, `--foo=bar`) and optional value.
    - Keep single-dash options (`-eps_*`, `-st_*`, `-ksp_*`, ...) and their
      immediate value token when present.
    PETSc treats unknown single-dash tokens as possible option-database keys.
    The sanitizer therefore keeps PETSc-style flags while stripping common
    Python and application flags that would otherwise produce noisy "unused
    option" warnings or alter solver behavior by accident.
    """
    if not argv:
        return []

    argv0 = argv[0]
    if argv0.startswith("-"):
        argv0 = "python"
    out = [argv0]
    i = 1
    n = len(argv)
    while i < n:
        tok = argv[i]

        if tok == "-c":
            i += 1
            if i < n and not argv[i].startswith("-"):
                i += 1
            continue
        if tok == "-m":
            i += 2
            continue

        if tok.startswith("--"):
            if "=" in tok:
                i += 1
            else:
                i += 1
                if i < n and not argv[i].startswith("-"):
                    i += 1
            continue

        if tok.startswith("-"):
            out.append(tok)
            if i + 1 < n and not argv[i + 1].startswith("-"):
                out.append(argv[i + 1])
                i += 2
            else:
                i += 1
            continue

        i += 1

    return out


def init_petsc_from_argv(argv: list[str] | None = None) -> None:
    """Initialize petsc4py once, forwarding argv for options database parsing.

    mpi4py must bind to the launcher's MPI before ``petsc4py.init``. On some
    Linux MPICH / Hydra configurations, importing petsc4py first can leave
    every process believing ``COMM_WORLD`` has size 1 even under ``mpiexec``.
    Keeping this order here gives installer checks and solver code the same
    MPI ownership rule.
    """
    global _PETSC_INIT_DONE
    if _PETSC_INIT_DONE:
        return
    import mpi4py.MPI  # noqa: F401 - import before petsc4py (upstream requirement)
    import petsc4py

    args = _sanitize_petsc_argv(list(sys.argv if argv is None else argv))
    try:
        petsc4py.init(args)
    except Exception as exc:  # pragma: no cover - defensive guard
        if not _already_initialized(exc):
            raise
    _PETSC_INIT_DONE = True


def init_slepc_from_argv(argv: list[str] | None = None) -> None:
    """Initialize petsc4py + slepc4py once, forwarding argv for options parsing."""
    global _SLEPC_INIT_DONE
    init_petsc_from_argv(argv)
    if _SLEPC_INIT_DONE:
        return

    import slepc4py

    args = _sanitize_petsc_argv(list(sys.argv if argv is None else argv))
    try:
        slepc4py.init(args)
    except Exception as exc:  # pragma: no cover - defensive guard
        if not _already_initialized(exc):
            raise
    _SLEPC_INIT_DONE = True
