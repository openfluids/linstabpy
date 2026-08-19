# SPDX-License-Identifier: GPL-3.0-only
"""Schema and validator for per-run metrics sidecars.

Single source of truth for the uniform cost+quality metrics payload.
CAMPAIGN §2 names stay where the production emitters already use them.
The schema matches known emitter deltas rather than renaming them.

- cost keys use a ``cost_`` prefix and byte units
  (``cost_peak_rss_bytes``/``cost_peak_gpu_mem_bytes`` rather than KiB/MiB).
- wall-time keys use ``cost_wall_*_s``.
- SLEPc setup/solve timings and convergence metadata use
  ``stability_eps_*`` keys.
- stability quality maxima sit as arrays under
  ``stability_postcheck_residual_rel`` and
  ``stability_postcheck_target_distance``.
  The validator accepts these measured arrays instead of inventing
  aggregate sidecar defaults.
- factor backend provenance uses ``provenance_factor_backend``.
- the provenance emitter does not currently emit a run date.
  This schema does not require a fabricated date.
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
        # `stride` is a single number by contract, but the mesh uses independent strides
        # independently in ksi and eta. The emitter sets `stride` to the common
        # value when they agree and -1 when they do not, and always emits both
        # axes here so an anisotropic run is never reported as if one axis
        # governed both. Optional: they are additions, so an older sidecar
        # without them is still valid.
        "stride": FieldSpec(_INT),
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

METRICS_SIDECAR_KEYS = frozenset(key for fields in SCHEMA.values() for key in fields) | {
    "cost_hardware_tag",
    "cost_git_sha",
}


def validate_metrics(d: Mapping[str, Any]) -> list[str]:
    """Return all schema errors for a per-run metrics payload.

    The validator never fills defaults.
    Required section fields must be present.
    Optional quality/run-type fields, when present, still go through the
    same checks.
    Run types that do not produce them need not include them.
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
    if isinstance(types, tuple):
        return " or ".join(t.__name__ for t in types)
    return types.__name__
