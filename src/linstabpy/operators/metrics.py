# SPDX-License-Identifier: GPL-3.0-only
"""Assembly metrics, timers, and CSR derivative-cache helpers."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from linstabpy.config import Config
from linstabpy.utils.cost_metrics import peak_rss_bytes as _peak_rss_bytes

logger = logging.getLogger(__name__)

_CURRENT_ASSEMBLY_METRICS: ContextVar["_AssemblyMetricsCollector | None"] = ContextVar(
    "_CURRENT_ASSEMBLY_METRICS",
    default=None,
)
_COEFF_TENSOR_NAMES = ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonify_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify_metric_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_metric_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _detect_rank_size() -> tuple[int, int]:
    try:
        from petsc4py import PETSc

        comm = PETSc.COMM_WORLD
        return int(comm.getRank()), int(comm.getSize())
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("PETSc rank/size probe failed; falling back to (0, 1): %s", exc)
        return 0, 1


@dataclass(frozen=True)
class _AssemblyMetricsConfig:
    enabled: bool
    output: str | None = None
    directory: str | None = None
    tag: str | None = None
    log: bool = False


@dataclass
class _AssemblyMetricsCollector:
    config: _AssemblyMetricsConfig
    metadata: dict[str, Any]
    rank: int
    size: int
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    started_at_utc: str = field(default_factory=_now_utc)
    start_time: float = field(default_factory=time.perf_counter)

    def record_stage(self, name: str, payload: dict[str, Any]) -> None:
        self.stages[name] = _jsonify_metric_value(payload)

    def update_metadata(self, *, overwrite: bool = True, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if value is None:
                continue
            if overwrite or key not in self.metadata:
                self.metadata[key] = _jsonify_metric_value(value)

    def _output_path(self) -> Path | None:
        if self.config.output:
            template = self.config.output
            try:
                rendered = template.format(rank=self.rank, size=self.size)
            except KeyError as exc:  # pragma: no cover - config guard
                raise ValueError(
                    f"Invalid assembly_metrics_output placeholder {exc!s}; "
                    "supported placeholders are {rank} and {size}"
                ) from exc
            path = Path(rendered)
            if path.suffix == "":
                path = path.with_suffix(".json")
            if self.size > 1 and "{rank" not in template and "{size" not in template:
                path = path.with_name(f"{path.stem}_rank{self.rank:04d}{path.suffix}")
            return path
        if not self.config.directory:
            return None
        stem = "assembly_metrics"
        if self.config.tag:
            stem = f"{stem}_{self.config.tag}"
        filename = f"{stem}_rank{self.rank:04d}.json"
        return Path(self.config.directory) / filename

    def _log_summary(self, payload: dict[str, Any], output_path: Path | None) -> None:
        if not self.config.log:
            return
        logger.info(
            "Assembly metrics: backend=%s resolved=%s ndof=%s nnz=%s wall=%.3fs path=%s",
            payload.get("requested_backend", "?"),
            payload.get("resolved_backend", "?"),
            payload.get("ndof", "?"),
            payload.get("nnz", "?"),
            payload.get("wall_seconds", float("nan")),
            output_path if output_path is not None else "<log-only>",
        )
        for name, stage in payload.get("stages", {}).items():
            logger.info(
                "  %s: %.3fs peak_rss_delta=%.1f MiB%s",
                name,
                float(stage.get("seconds", 0.0)),
                float(stage.get("rss_peak_delta_bytes", 0)) / (1024.0 * 1024.0),
                " [skipped]" if stage.get("skipped") else "",
            )

    def finish(self) -> None:
        payload = {
            **self.metadata,
            "rank": self.rank,
            "size": self.size,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": _now_utc(),
            "wall_seconds": time.perf_counter() - self.start_time,
            "stages": self.stages,
        }
        output_path = self._output_path()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
        self._log_summary(payload, output_path)


def _resolve_assembly_metrics_config(solver_cfg: dict | None) -> _AssemblyMetricsConfig:
    cfg = Config.load(solver_cfg if solver_cfg is not None else {})
    output = cfg.mesh.assembly_metrics_output
    directory = cfg.mesh.assembly_metrics_dir
    tag = cfg.mesh.assembly_metrics_tag
    enabled = cfg.mesh.assembly_metrics or output is not None or directory is not None
    log_explicit = cfg.mesh.assembly_metrics_log_explicit
    log_enabled = log_explicit if log_explicit is not None else enabled
    return _AssemblyMetricsConfig(
        enabled=enabled,
        output=output,
        directory=directory,
        tag=tag,
        log=log_enabled,
    )


def _current_assembly_metrics() -> _AssemblyMetricsCollector | None:
    return _CURRENT_ASSEMBLY_METRICS.get()


def _note_assembly_metadata(*, overwrite: bool = True, **kwargs: Any) -> None:
    collector = _current_assembly_metrics()
    if collector is None:
        return
    collector.update_metadata(overwrite=overwrite, **kwargs)


@contextmanager
def _assembly_metrics_session(
    *,
    solver_cfg: dict | None,
    **metadata: Any,
) -> Iterator[_AssemblyMetricsCollector | None]:
    existing = _current_assembly_metrics()
    if existing is not None:
        existing.update_metadata(overwrite=False, **metadata)
        yield existing
        return

    config = _resolve_assembly_metrics_config(solver_cfg)
    if not config.enabled:
        yield None
        return

    rank, size = _detect_rank_size()
    collector = _AssemblyMetricsCollector(
        config=config,
        metadata={},
        rank=rank,
        size=size,
    )
    collector.update_metadata(
        rank=rank,
        size=size,
        status="running",
        **metadata,
    )
    token = _CURRENT_ASSEMBLY_METRICS.set(collector)
    try:
        yield collector
    except Exception as exc:
        collector.update_metadata(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    else:
        collector.update_metadata(status="ok")
    finally:
        _CURRENT_ASSEMBLY_METRICS.reset(token)
        collector.finish()


@contextmanager
def _assembly_stage(name: str) -> Iterator[dict[str, Any]]:
    stage: dict[str, Any] = {}
    collector = _current_assembly_metrics()
    rss0 = _peak_rss_bytes()
    t0 = time.perf_counter()
    try:
        yield stage
    except Exception as exc:
        stage["status"] = "error"
        stage["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if collector is not None:
            rss1 = _peak_rss_bytes()
            stage.setdefault("seconds", time.perf_counter() - t0)
            stage.setdefault("rss_peak_before_bytes", rss0)
            stage.setdefault("rss_peak_after_bytes", rss1)
            stage.setdefault("rss_peak_delta_bytes", max(0, rss1 - rss0))
            collector.record_stage(name, stage)


@dataclass(frozen=True)
class _CSRDerivativeCache:
    csr: sp.csr_matrix
    row_nnz: np.ndarray
    row_indices: np.ndarray


def _make_csr_derivative_cache(op: sp.spmatrix) -> _CSRDerivativeCache:
    """Convert a derivative operator into a canonical CSR cache."""
    if not sp.issparse(op):
        raise TypeError("Derivative operator must be a sparse matrix")
    csr = op if sp.isspmatrix_csr(op) else op.tocsr()
    if not getattr(csr, "has_canonical_format", False):
        csr = csr.copy()
        csr.sum_duplicates()
        csr.sort_indices()
    if not np.issubdtype(csr.dtype, np.complexfloating):
        csr = csr.astype(complex)
    row_nnz = np.diff(csr.indptr).astype(np.int64, copy=False)
    row_indices = np.repeat(np.arange(csr.shape[0], dtype=np.int64), row_nnz)
    return _CSRDerivativeCache(csr=csr, row_nnz=row_nnz, row_indices=row_indices)


def _as_csr_derivative_cache(op: _CSRDerivativeCache | sp.spmatrix) -> _CSRDerivativeCache:
    if isinstance(op, _CSRDerivativeCache):
        return op
    return _make_csr_derivative_cache(op)


def _cache_5_block_derivatives(
    op: list[sp.spmatrix] | tuple[sp.spmatrix, ...] | sp.spmatrix,
    ngp: int,
) -> list[_CSRDerivativeCache]:
    if isinstance(op, (list, tuple)):
        if len(op) != 5:
            raise ValueError("Expected five derivative blocks for axy mode")
        return [_make_csr_derivative_cache(mat) for mat in op]

    if not sp.issparse(op):
        raise TypeError("Derivative operator must be a sparse matrix or 5-block list/tuple")

    cache = _make_csr_derivative_cache(op)
    if cache.csr.shape != (ngp, ngp):
        raise ValueError(
            f"Expected derivative operator with shape {(ngp, ngp)}, got {cache.csr.shape}"
        )
    return [cache] * 5


AssemblyMetrics = _AssemblyMetricsCollector
