# SPDX-License-Identifier: GPL-3.0-only
"""Registry of known ``LINSTABPY_*`` environment variables.

Single choke-point for the typo guard: :func:`find_unknown_env_vars` lists
``LINSTABPY_*`` names present in the environment that nothing in the repo
reads, and the CLI warns (stderr-only, warn-never-fail) instead of silently
ignoring them.

Coverage is enforced by ``tests/test_env_registry.py``, which scans the
executable sources (``src/``, installer/runner scripts, ``examples/``) for
``LINSTABPY_*`` literals and requires each to be listed here. When adding a
new variable, add it here first, then use it.
"""

from __future__ import annotations

import difflib
import os
import sys
from collections.abc import Mapping, Sequence
from typing import TextIO

#: Every ``LINSTABPY_*`` name read anywhere in the repo: runtime (``src/``),
#: installer (``install.sh``), runners (``benchmark.sh``,
#: ``run_all_examples.sh``), shipped examples, and test gating/harness vars.
#: Deliberately excludes plausible-but-unread names (see
#: :data:`KNOWN_INEFFECTIVE_VARS`): those must keep warning.
KNOWN_LINSTABPY_VARS: frozenset[str] = frozenset(
    {
        # -- driver / case selection (driver/run.py, driver/cli.py) --
        "LINSTABPY_CASE_CONFIG",
        "LINSTABPY_CASE_OUTPUT_NPZ",
        "LINSTABPY_CASE_LOG_PREFIX",
        "LINSTABPY_CASE_ROOT_ASSEMBLY",
        "LINSTABPY_FORCE_RECOMPUTE",
        "LINSTABPY_STAGE_HEARTBEAT_SECONDS",
        # -- cost metrics (utils/cost_metrics.py) --
        "LINSTABPY_GPUS",
        "LINSTABPY_HARDWARE_TAG",
        "LINSTABPY_FLOQUET_SHIFT",
        "LINSTABPY_JOB_DEADLINE",
        # -- provenance (utils/provenance.py) --
        "LINSTABPY_MPI_STACK",
        "LINSTABPY_COMPUTE_MODE",
        "LINSTABPY_HOSTNAME",
        # -- factorisation policy (config/typed.py, linalg/backends.py) --
        "LINSTABPY_FACTOR_BACKEND",
        "LINSTABPY_FACTOR_PRECISION",
        "LINSTABPY_MUMPS_BLR",
        "LINSTABPY_MUMPS_BLR_TOL",
        "LINSTABPY_MUMPS_ORDERING",
        "LINSTABPY_AUTO_ROUTE_DOF_THRESHOLD",
        # -- eigen solver family (config/typed.py EigenCfg) --
        "LINSTABPY_EIGEN_ST_SOLVER",
        "LINSTABPY_EIGEN_PC_FACTOR_SOLVER_TYPE",
        "LINSTABPY_EIGEN_COND_WARN",
        "LINSTABPY_EIGEN_COND_WARN_MAX_N",
        "LINSTABPY_EIGEN_COND_WARN_THRESHOLD",
        "LINSTABPY_EIGEN_KSP_TYPE",
        "LINSTABPY_EIGEN_PC_TYPE",
        "LINSTABPY_EIGEN_KSP_PC_SIDE",
        "LINSTABPY_EIGEN_KSP_RTOL",
        "LINSTABPY_EIGEN_KSP_MAX_IT",
        "LINSTABPY_EIGEN_ASM_OVERLAP",
        "LINSTABPY_EIGEN_SUB_KSP_TYPE",
        "LINSTABPY_EIGEN_SUB_PC_TYPE",
        "LINSTABPY_EIGEN_CONV_TEST",
        "LINSTABPY_EIGEN_EPS_TOL",
        "LINSTABPY_EIGEN_EPS_MAX_IT",
        "LINSTABPY_EIGEN_RESIDUAL_ERROR_TYPE",
        "LINSTABPY_EIGEN_MONITOR",
        "LINSTABPY_EIGEN_MONITOR_EVERY",
        "LINSTABPY_EIGEN_TRUE_RESIDUAL",
        "LINSTABPY_EIGEN_EXTRACT_ALL_CONVERGED",
        "LINSTABPY_EIGEN_WEIGHTED_NORM_TOL",
        # -- spatial eigen (config/typed.py SpatialCfg) --
        "LINSTABPY_SPATIAL_ST_SOLVER",
        # -- assembly / metrics (config/typed.py MeshCfg) --
        "LINSTABPY_ASSEMBLY_BACKEND",
        "LINSTABPY_ASSEMBLY_METRICS",
        "LINSTABPY_ASSEMBLY_METRICS_OUTPUT",
        "LINSTABPY_ASSEMBLY_METRICS_DIR",
        "LINSTABPY_ASSEMBLY_METRICS_TAG",
        "LINSTABPY_ASSEMBLY_METRICS_LOG",
        # -- PETSc init-time options (config/typed.py MpiCfg) --
        "LINSTABPY_PETSC_MATRIX_MODE",
        "LINSTABPY_PETSC_ROOT_SCATTER",
        "LINSTABPY_PETSC_MUMPS_ICNTL_14",
        "LINSTABPY_PETSC_MUMPS_ICNTL_24",
        # Removed knob: still recognised so the reader raises its helpful
        # "gone, set LINSTABPY_MUMPS_ORDERING=metis" error instead of the
        # generic unknown-variable warning.
        "LINSTABPY_PETSC_SET_MUMPS_ICNTL_7",
        # -- resolvent EPS sub-family (config/typed.py ResolventEpsCfg) --
        "LINSTABPY_RESOLVENT_EPS_TOL",
        "LINSTABPY_RESOLVENT_EPS_MAX_IT",
        "LINSTABPY_RESOLVENT_EPS_CONV_TEST",
        "LINSTABPY_RESOLVENT_EPS_TRUE_RESIDUAL",
        "LINSTABPY_RESOLVENT_EPS_MONITOR",
        "LINSTABPY_RESOLVENT_EPS_MONITOR_EVERY",
        "LINSTABPY_RESOLVENT_FORCING_NORM",
        # -- linear-solve family, LINSTABPY_LINSOLVE prefix --
        # (config/typed.py LinSolveCfg via f"{prefix}_..." expansion)
        "LINSTABPY_LINSOLVE_TYPE",
        "LINSTABPY_LINSOLVE_PC_FACTOR_SOLVER_TYPE",
        "LINSTABPY_LINSOLVE_KSP_TYPE",
        "LINSTABPY_LINSOLVE_PC_TYPE",
        "LINSTABPY_LINSOLVE_KSP_PC_SIDE",
        "LINSTABPY_LINSOLVE_KSP_RTOL",
        "LINSTABPY_LINSOLVE_KSP_ATOL",
        "LINSTABPY_LINSOLVE_KSP_MAX_IT",
        "LINSTABPY_LINSOLVE_ASM_OVERLAP",
        "LINSTABPY_LINSOLVE_SUB_KSP_TYPE",
        "LINSTABPY_LINSOLVE_SUB_PC_TYPE",
        "LINSTABPY_LINSOLVE_PC_FACTOR_LEVELS",
        "LINSTABPY_LINSOLVE_PC_FACTOR_SHIFT_TYPE",
        "LINSTABPY_LINSOLVE_PC_FACTOR_SHIFT_AMOUNT",
        "LINSTABPY_LINSOLVE_KSP_MONITOR",
        # -- linear-solve family, LINSTABPY_RESOLVENT prefix --
        "LINSTABPY_RESOLVENT_TYPE",
        "LINSTABPY_RESOLVENT_PC_FACTOR_SOLVER_TYPE",
        "LINSTABPY_RESOLVENT_KSP_TYPE",
        "LINSTABPY_RESOLVENT_PC_TYPE",
        "LINSTABPY_RESOLVENT_KSP_PC_SIDE",
        "LINSTABPY_RESOLVENT_KSP_RTOL",
        "LINSTABPY_RESOLVENT_KSP_ATOL",
        "LINSTABPY_RESOLVENT_KSP_MAX_IT",
        "LINSTABPY_RESOLVENT_ASM_OVERLAP",
        "LINSTABPY_RESOLVENT_SUB_KSP_TYPE",
        "LINSTABPY_RESOLVENT_SUB_PC_TYPE",
        "LINSTABPY_RESOLVENT_PC_FACTOR_LEVELS",
        "LINSTABPY_RESOLVENT_PC_FACTOR_SHIFT_TYPE",
        "LINSTABPY_RESOLVENT_PC_FACTOR_SHIFT_AMOUNT",
        "LINSTABPY_RESOLVENT_KSP_MONITOR",
        # -- cuDSS factor backend (linalg/cudss_factor.py) --
        "LINSTABPY_CUDSS_MT_LIB",
        "LINSTABPY_CUDSS_THREADS",
        # -- installer (install.sh) --
        "LINSTABPY_AUTO_INSTALL_SYS_DEPS",
        "LINSTABPY_BLAS",
        "LINSTABPY_BLAS_CHOICE",
        "LINSTABPY_BUILD_STACK",
        "LINSTABPY_DARWIN_INSERT_OPENBLAS",
        "LINSTABPY_EXPECT_GPU",
        "LINSTABPY_HYDRA_BOOTSTRAP",
        "LINSTABPY_INSTALL_SKIP_VERIFY",
        "LINSTABPY_MPI4PY_NO_BINARY",
        "LINSTABPY_MPI_COMPILE_ERROR",
        "LINSTABPY_PETSC4PY_VERSION",
        "LINSTABPY_PETSC_MPIEXEC",
        "LINSTABPY_PETSC_OPTFLAGS",
        "LINSTABPY_PETSC_VERSION",
        "LINSTABPY_PYTHON_VERSION",
        "LINSTABPY_RUNTIME_LD",
        "LINSTABPY_SKIP_SYS_DEPS",
        "LINSTABPY_SLEPC4PY_VERSION",
        "LINSTABPY_SLEPC_VERSION",
        "LINSTABPY_VERIFY_MPIEXEC",
        "LINSTABPY_WITH_H5PY",
        # -- runners (benchmark.sh, run_all_examples.sh) --
        "LINSTABPY_MPIEXEC",
        "LINSTABPY_PYTHON_BIN",
        "LINSTABPY_MPI_RANKS",
        "LINSTABPY_PYTHON",
        "LINSTABPY_BENCHMARK_EX1_LABEL",
        "LINSTABPY_BENCHMARK_EX1_MODULE",
        # -- shipped examples (examples/) --
        "LINSTABPY_SONG2023_CONFIG",
        "LINSTABPY_EXAMPLE_STEM",
        "LINSTABPY_EXAMPLE_CONFIG",
        "LINSTABPY_BASEFLOW_DATA_DIR",
        # -- test gating / harness (tests/) --
        "LINSTABPY_MATLAB_PARITY",
        "LINSTABPY_MATLAB_FIXTURES_NPZ",
        "LINSTABPY_SONG32_CSV",
        "LINSTABPY_TEST_REFERENCE_SPECTRUM",
        "LINSTABPY_DUAL_PATH",
        "LINSTABPY_DUAL_PATH_REPORT",
    }
)

#: Plausible-looking names that are intentionally *not* read, kept out of
#: :data:`KNOWN_LINSTABPY_VARS` so the typo guard keeps firing on them.
#: ``tests/test_env_registry.py`` asserts they stay unknown.
KNOWN_INEFFECTIVE_VARS: frozenset[str] = frozenset(
    {
        # ncv is YAML-only by design (Config.load ignores these env names).
        "LINSTABPY_EIGEN_NCV",
        "LINSTABPY_RESOLVENT_EPS_NCV",
        "LINSTABPY_NCV",
        # Inexact-eigen block is YAML-only (InexactCfg.from_mapping takes no env).
        "LINSTABPY_EIGEN_INEXACT_KSP_TYPE",
        # Bare family prefixes select a YAML subsection; they are not env vars.
        "LINSTABPY_EIGEN",
        "LINSTABPY_RESOLVENT",
        "LINSTABPY_LINSOLVE",
        # Removed in 1.0.0: serial cuDSS always refactorizes in place.
        "LINSTABPY_FAST_REFACTOR",
    }
)

_INEFFECTIVE_HINTS: dict[str, str] = {
    "LINSTABPY_EIGEN_NCV": "ncv is YAML-only; env has no effect",
    "LINSTABPY_RESOLVENT_EPS_NCV": "ncv is YAML-only; env has no effect",
    "LINSTABPY_NCV": "ncv is YAML-only; env has no effect",
    "LINSTABPY_EIGEN_INEXACT_KSP_TYPE": "inexact block is YAML-only; env has no effect",
    "LINSTABPY_EIGEN": "family prefix, not an env var; use LINSTABPY_EIGEN_*",
    "LINSTABPY_RESOLVENT": "family prefix, not an env var; use LINSTABPY_RESOLVENT_*",
    "LINSTABPY_LINSOLVE": "family prefix, not an env var; use LINSTABPY_LINSOLVE_*",
    "LINSTABPY_FAST_REFACTOR": "removed in 1.0.0; serial cuDSS always refactorizes in place",
}


def find_unknown_env_vars(env: Mapping[str, str] | None = None) -> list[str]:
    """Return sorted ``LINSTABPY_*`` names in *env* that nothing reads."""
    if env is None:
        env = os.environ
    return sorted(n for n in env if n.startswith("LINSTABPY_") and n not in KNOWN_LINSTABPY_VARS)


def suggest_env_var(name: str) -> str | None:
    """Return the closest known variable name, or None if nothing is close."""
    matches = difflib.get_close_matches(name, KNOWN_LINSTABPY_VARS, n=1, cutoff=0.85)
    return matches[0] if matches else None


def _describe_unknown(name: str) -> str:
    hint = _INEFFECTIVE_HINTS.get(name)
    if hint is not None:
        return f"{name} ({hint})"
    suggestion = suggest_env_var(name)
    if suggestion is not None:
        return f"{name} (did you mean {suggestion}?)"
    return name


def format_unknown_env_warning(unknown: Sequence[str]) -> str:
    """Render the warn-never-fail message for unknown ``LINSTABPY_*`` names."""
    descriptions = ", ".join(_describe_unknown(n) for n in unknown)
    if len(unknown) == 1:
        return f"warning: unknown {descriptions} \u2014 has no effect and will be ignored."
    return (
        f"warning: unknown LINSTABPY_* variables {descriptions} "
        "\u2014 these have no effect and will be ignored."
    )


def warn_unknown_env_vars(
    unknown: Sequence[str],
    *,
    file: TextIO | None = None,
) -> list[str]:
    """Print the unknown-variable warning to stderr; never raises.

    Takes the precomputed list from :func:`find_unknown_env_vars` so callers
    under MPI can gate on rank 0. Returns the list unchanged.
    """
    unknowns = list(unknown)
    if unknowns:
        print(format_unknown_env_warning(unknowns), file=file if file is not None else sys.stderr)
    return unknowns
