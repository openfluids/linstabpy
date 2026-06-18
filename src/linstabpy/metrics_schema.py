# SPDX-License-Identifier: GPL-3.0-only
"""Schema and validator for per-run metrics sidecars.

This module is the single source of truth for the uniform cost+quality metrics
payload.  CAMPAIGN §2 names are preserved where the production emitters already
use them.  Known emitter deltas are intentionally matched rather than renamed:

- cost keys are emitted with a ``cost_`` prefix and byte units
  (``cost_peak_rss_bytes``/``cost_peak_gpu_mem_bytes`` rather than KiB/MiB).
- wall-time keys are emitted as ``cost_wall_*_s``.
- SLEPc setup/solve timings and convergence metadata are emitted as
  ``stability_eps_*`` keys.
- stability quality maxima are emitted as arrays under
  ``stability_postcheck_residual_rel`` and
  ``stability_postcheck_target_distance``; the validator accepts these measured
  arrays instead of inventing aggregate sidecar defaults.
- factor backend provenance is emitted as ``provenance_factor_backend``.
- the provenance emitter does not currently emit a run date; this schema does
  not require a fabricated date.
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
    """Validation rule for one emitted metrics field."""

    types: TypeSpec
    required: bool = True


_STRING = (str, np.str_)
_BOOL = (bool, np.bool_)
_INT = (Integral, np.integer)
_REAL = (Real, np.floating, np.integer)
_REAL_NULLABLE = (Real, np.floating, np.integer, type(None))
_ARRAY = (list, tuple, np.ndarray)
_ARRAY_OR_SCALAR = (list, tuple, np.ndarray, Real, np.floating, np.integer, complex, np.complexfloating)

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
    },
    "problem": {
        "case": FieldSpec(_STRING),
        "fd_order": FieldSpec(_INT),
        "stride": FieldSpec(_INT),
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
    },
    "quality": {
        "stability_eps_nconv": FieldSpec(_INT, required=False),
        "stability_accepted_count": FieldSpec(_INT, required=False),
        "omega": FieldSpec(_ARRAY_OR_SCALAR, required=False),
        "gains": FieldSpec(_ARRAY_OR_SCALAR, required=False),
        "stability_postcheck_residual_rel": FieldSpec(_ARRAY_OR_SCALAR, required=False),
        "stability_postcheck_target_distance": FieldSpec(_ARRAY_OR_SCALAR, required=False),
        "stability_all_reject_reason_code": FieldSpec(_ARRAY, required=False),
    },
}

QUALITY_VARIANTS: dict[str, tuple[str, ...]] = {
    "stability": (
        "stability_eps_nconv",
        "stability_accepted_count",
        "omega",
        "stability_postcheck_residual_rel",
        "stability_postcheck_target_distance",
        "stability_all_reject_reason_code",
    ),
    "resolvent": ("gains",),
}

METRICS_SIDECAR_KEYS = frozenset(
    key for fields in SCHEMA.values() for key in fields
) | {"cost_hardware_tag", "cost_git_sha"}


def validate_metrics(d: Mapping[str, Any]) -> list[str]:
    """Return all schema errors for a per-run metrics payload.

    The validator never fills defaults.  Required section fields must be present;
    optional quality/run-type fields are validated when present but are not
    required for run types that do not produce them.
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
                errors.append(
                    f"{section}.{key}: expected {expected}, got {type(value).__name__}"
                )
    return errors


def _type_name(types: TypeSpec) -> str:
    if isinstance(types, tuple):
        return " or ".join(t.__name__ for t in types)
    return types.__name__
