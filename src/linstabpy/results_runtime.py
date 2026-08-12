# SPDX-License-Identifier: GPL-3.0-only
"""Deprecated compatibility shim for :mod:`linstabpy.driver` runtime entry points."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

_RUN_MODULE = "linstabpy.driver.run"
_CLI_MODULE = "linstabpy.driver.cli"
_IO_MODULE = "linstabpy.driver.io"

_RUN_EXPORTS = {
    "RUNTIME_NPZ_CONTRACT_KEYS",
    "EMPTY_FLOAT",
    "EMPTY_COMPLEX",
    "EMPTY_INT32",
    "EMPTY_INT64",
    "_STAGE_WALL",
    "_array_from_mapping",
    "_config_hash",
    "_git_head_sha",
    "_first_env",
    "_output_npz_path",
    "_log_prefix",
    "_log",
    "_peak_rss_bytes",
    "_stage_start",
    "_log_stage",
    "_default_stage_heartbeat_seconds",
    "_parse_alpha_filter",
    "_effective_alpha_filter",
    "_stage_heartbeat",
    "_restrict_operator_for_stability",
    "_expand_reduced_modes",
    "_build_distributed_vec_from_global_array",
    "_postcheck_stability_modes",
    "_resolve_omega_target",
    "_resolve_optional_float",
    "_cfg_first",
    "_cfg_get",
    "_RuntimeConfigSource",
    "_stage_load_runtime_config",
    "_LoadedCase",
    "_stage_load_case",
    "_stage_build_mesh",
    "_stage_compute_chu_norm",
    "_AssemblyStageResult",
    "_stage_assemble_operators",
    "_ResolventStageResult",
    "_empty_resolvent_result",
    "_stage_run_resolvent",
    "_StabilityStageResult",
    "_stage_run_stability",
    "_StabilityAcceptanceStageResult",
    "_stage_evaluate_stability_acceptance",
    "run",
    "compute",
    # Imported names that were historically reachable from results_runtime.
    "MPI",
    "PETSc",
    "Config",
    "Mesh",
    "load_hamada_case",
    "sutherland_air",
    "analyze_bc_recipe",
    "dirichlet_free_dofs",
    "build_lnse_system_result",
    "compute_chu_norm",
    "resolvent_analysis",
    "_is_petsc_matrix",
    "scipy_to_petsc",
    "vec_to_numpy",
    "_solve_near_target",
    "temporal_eigenvalues",
    "_postcheck_target_distance",
    "check_conjugate_pair_completeness",
    "evaluate_stability_acceptance",
    "load_jsonc",
    "load_config",
}

_EXPORTS = _RUN_EXPORTS | {"main", "save_results", "load_results", "_stage_write_npz"}

__all__ = sorted(_EXPORTS)


def _resolve(name: str) -> tuple[str, Any]:
    if name == "main":
        module_name = _CLI_MODULE
        attr_name = "main"
    elif name in {"save_results", "load_results", "_stage_write_npz"}:
        module_name = _IO_MODULE
        attr_name = name
    elif name == "compute":
        module_name = _RUN_MODULE
        attr_name = "run"
    else:
        module_name = _RUN_MODULE
        attr_name = name
    module = importlib.import_module(module_name)
    return module_name, getattr(module, attr_name)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr = _resolve(name)
        warnings.warn(
            f"linstabpy.results_runtime.{name} is deprecated; use {module_name}.{attr.__name__ if callable(attr) else name}",
            DeprecationWarning,
            stacklevel=2,
        )
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
