# SPDX-License-Identifier: GPL-3.0-only
"""Runtime result archive I/O helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger(__name__)


def _jsonify_metrics_value(value: Any) -> Any:
    """Convert NumPy/Python metrics values to JSON-safe sidecar values."""
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.number):
        item = value.item()
        if isinstance(item, float) and np.isnan(item):
            return None
        if isinstance(item, complex):
            return {"real": item.real, "imag": item.imag}
        return item
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _jsonify_metrics_value(value.item())
        return [_jsonify_metrics_value(item) for item in value.tolist()]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (list, tuple)):
        return [_jsonify_metrics_value(item) for item in value]
    return value


def _is_root_rank() -> bool:
    """Return True on rank 0, or when PETSc is not installed."""
    try:
        import mpi4py.MPI  # noqa: F401
        from petsc4py import PETSc

        return PETSc.COMM_WORLD.getRank() == 0
    except Exception:
        return True


def save_results(
    *,
    output_npz: Path,
    heartbeat_seconds: float,
    payload: dict[str, object],
    peak_rss_bytes: int,
    stage_wall: dict[str, float] | None = None,
    provenance: dict[str, str] | None = None,
) -> None:
    """Write the runtime NPZ and the metrics sidecar.

    ``peak_rss_bytes`` is the already-reduced peak RSS from the caller.
    The write path does no MPI collective.
    ``heartbeat_seconds`` exists for the driver call site. The write path
    does not read it.
    """
    del heartbeat_seconds
    if provenance is None:
        from linstabpy.utils.provenance import collect_provenance

        provenance = collect_provenance(None)

    from linstabpy.utils.cost_metrics import collect_cost_metrics

    cost = collect_cost_metrics(
        {} if stage_wall is None else stage_wall,
        git_sha=provenance["git_sha"],
        peak_rss_bytes=peak_rss_bytes,
    )
    payload = {**payload, **cost, **provenance}
    provenance_lines = "\n".join(f"  {key}: {value}" for key, value in provenance.items())
    _LOG.info("provenance block for %s:\n%s", output_npz, provenance_lines)
    np.savez_compressed(output_npz, **payload)
    if _is_root_rank():
        from linstabpy.metrics_schema import METRICS_SIDECAR_KEYS, validate_metrics

        sidecar = output_npz.with_suffix(".metrics.json")
        sidecar_payload = {
            key: value for key, value in payload.items() if key in METRICS_SIDECAR_KEYS
        }
        schema_errors = validate_metrics(sidecar_payload)
        if schema_errors:
            _LOG.warning(
                "metrics schema validation failed for %s:\n%s",
                sidecar,
                "\n".join(f"- {error}" for error in schema_errors),
            )
            sidecar_payload["schema_errors"] = schema_errors
        serializable_payload = {k: _jsonify_metrics_value(v) for k, v in sidecar_payload.items()}
        with open(sidecar, "w") as fh:
            json.dump(serializable_payload, fh, indent=2, sort_keys=True)


def load_results(path: str | Path):
    """Load a runtime ``.npz`` archive without pickle support."""
    return np.load(path, allow_pickle=False)
