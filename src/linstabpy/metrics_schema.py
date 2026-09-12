# SPDX-License-Identifier: GPL-3.0-only
"""Cost and quality record written next to each run NPZ.

Rank 0 writes ``<stem>.metrics.json`` after ``save_results`` stores the NPZ.
The JSON is a flat dict. ``SCHEMA`` groups the same keys into provenance,
problem, cost, and quality so an error names the group.

Units
-----
- Wall time in seconds: ``cost_wall_*_s``, ``stability_eps_*_seconds``.
- Memory in bytes: ``cost_peak_rss_bytes``, ``cost_peak_gpu_mem_bytes``.
- No run date. The emitters do not write one.

Required keys must be present. Quality keys are optional: a resolvent run
may omit eigenvalue residuals, and a stability run may omit gains.
``validate_metrics`` reports every missing or mistyped key. It never fills
a default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

TypeSpec = type | tuple[type, ...]


@dataclass(frozen=True)
class FieldSpec:
    """Accepted types. ``required`` is True when the key must be present."""

    types: TypeSpec
    required: bool = True


_STRING = (str, np.str_)
_INT = (Integral, np.integer)
_REAL = (Real, np.floating, np.integer)
_REAL_NULLABLE = (Real, np.floating, np.integer, type(None))
_ARRAY = (list, tuple, np.ndarray)
_ARRAY_OR_SCALAR = (
    list,
    tuple,
    np.ndarray,
    Real,
    np.floating,
    np.integer,
    complex,
    np.complexfloating,
)

SCHEMA: dict[str, dict[str, FieldSpec]] = {
    "provenance": {
        "git_sha": FieldSpec(_STRING),
        "config_md5": FieldSpec(_STRING),
        "hostname": FieldSpec(_STRING),
        "compute_mode": FieldSpec(_STRING),
        "mpi_stack": FieldSpec(_STRING),
        "python_version": FieldSpec(_STRING),
        "numpy_version": FieldSpec(_STRING),
        "scipy_version": FieldSpec(_STRING),
        "petsc4py_version": FieldSpec(_STRING),
        "slepc4py_version": FieldSpec(_STRING),
        "petsc_arch": FieldSpec(_STRING),
        "mumps_version": FieldSpec(_STRING),
        "mpi_impl": FieldSpec(_STRING),
        "blas": FieldSpec(_STRING),
    },
    "problem": {
        "case": FieldSpec(_STRING),
        "fd_order": FieldSpec(_INT),
        # Common mesh stride, or -1 when the ksi and eta strides differ.
        "stride": FieldSpec(_INT),
        # Optional ksi and eta strides. Older JSON without them stays valid.
        "stride_ksi": FieldSpec(_INT, required=False),
        "stride_eta": FieldSpec(_INT, required=False),
        "nx": FieldSpec(_INT),
        "ny": FieldSpec(_INT),
        "ndof": FieldSpec(_INT),
        "nnz": FieldSpec(_INT),
        "beta": FieldSpec(_REAL),
        "lambda_over_L": FieldSpec(_REAL),
        "omega_target": FieldSpec(_ARRAY_OR_SCALAR),
        "stability_eps_tol": FieldSpec(_REAL),
        "stability_eps_ncv": FieldSpec(_INT),
        "provenance_factor_backend": FieldSpec(_STRING),
    },
    "cost": {
        "cost_wall_total_s": FieldSpec(_REAL),
        "cost_wall_assemble_s": FieldSpec(_REAL_NULLABLE),
        "cost_wall_factor_s": FieldSpec(_REAL_NULLABLE),
        "stability_eps_setup_seconds": FieldSpec(_REAL_NULLABLE, required=False),
        "stability_eps_solve_seconds": FieldSpec(_REAL_NULLABLE, required=False),
        "cost_wall_solve_s": FieldSpec(_REAL_NULLABLE),
        "cost_mpi_ranks": FieldSpec(_INT),
        "cost_gpus": FieldSpec(_INT),
        "n_threads": FieldSpec(_INT, required=False),
        "cost_peak_rss_bytes": FieldSpec(_INT),
        "cost_peak_gpu_mem_bytes": FieldSpec(_REAL_NULLABLE),
        "cost_petsc_options": FieldSpec(_STRING),
        "cost_hardware_tag": FieldSpec(_STRING, required=False),
        "cost_git_sha": FieldSpec(_STRING, required=False),
    },
    "quality": {
        # Optional. A stability run emits residual and acceptance keys.
        # A resolvent run emits gains. Neither run type must emit the other.
        "stability_eps_nconv": FieldSpec(_INT, required=False),
        "stability_accepted_count": FieldSpec(_INT, required=False),
        "omega": FieldSpec(_ARRAY_OR_SCALAR, required=False),
        "gains": FieldSpec(_ARRAY_OR_SCALAR, required=False),
        "stability_postcheck_residual_rel": FieldSpec(_ARRAY_OR_SCALAR, required=False),
        "stability_postcheck_target_distance": FieldSpec(_ARRAY_OR_SCALAR, required=False),
        "stability_all_reject_reason_code": FieldSpec(_ARRAY, required=False),
    },
}

METRICS_SIDECAR_KEYS = frozenset(key for fields in SCHEMA.values() for key in fields)

__all__ = ["FieldSpec", "SCHEMA", "METRICS_SIDECAR_KEYS", "validate_metrics"]


def validate_metrics(d: Mapping[str, Any]) -> list[str]:
    """Return every schema error for a per-run metrics payload.

    The validator never fills defaults.
    Required keys must be present.
    Optional keys that are present go through the same type checks.
    """
    errors: list[str] = []
    for section, fields in SCHEMA.items():
        for key, spec in fields.items():
            if key not in d:
                if spec.required:
                    errors.append(f"{section}.{key}: missing required key")
                continue
            value = d[key]
            if not isinstance(value, spec.types):
                expected = _type_name(spec.types)
                errors.append(f"{section}.{key}: expected {expected}, got {type(value).__name__}")
    return errors


def _type_name(types: TypeSpec) -> str:
    """Render a type or a union for a validator error message."""
    if isinstance(types, tuple):
        return " or ".join(t.__name__ for t in types)
    return types.__name__
