# SPDX-License-Identifier: GPL-3.0-only
"""RAM-calibrated operating-point selection for resolvent sweeps.

The memory model follows the §7c prior

``mem ≈ [nnz·c_val·prec + fill·prec + m·n·prec] / ranks``

but does not trust a formula-only fill estimate.  Each call first runs an
empirical sparse-LU analysis probe on the supplied nonzero pattern.  The probe
measures the actual LU fill count and bytes per factor nonzero on this Python / SciPy
build, and those measured quantities replace the symbolic ``fill`` prior in the
selection decision.  The returned :class:`OperatingPoint` contains only resource
knobs; user physics choices such as grid, finite-difference order, modes, or
frequency/wavenumber ranges are never read or changed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

BackendChoice = Literal["direct", "compressed-sweep"]
Precision = Literal["double", "single"]

_COMPLEX128_BYTES = np.dtype(np.complex128).itemsize
_COMPLEX64_BYTES = np.dtype(np.complex64).itemsize
_DEFAULT_OPERATOR_VALUE_MULTIPLIER = 2.0
_DEFAULT_MEMORY_FRACTION = 0.80
_DEFAULT_REORDER = "colamd"
_DEFAULT_RESTART = 40
_DEFAULT_COMPRESSION_LEVELS = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3)
_DEFAULT_COMPRESSION_MEMORY_RATIOS = (0.55, 0.35, 0.20, 0.08)
_SPLU_REORDER = {
    "natural": "NATURAL",
    "colamd": "COLAMD",
    "mmd_ata": "MMD_ATA",
    "mmd_at_plus_a": "MMD_AT_PLUS_A",
}


@dataclass(frozen=True)
class SymbolicProbeResult:
    """Measured sparse factor footprint from a real symbolic/numeric probe."""

    fill_count: int
    bytes_per_nonzero: float
    factor_bytes: int
    nnz: int
    reorder: str
    backend: str = "scipy-splu"


@dataclass(frozen=True)
class OperatingPoint:
    """Resource-only knobs chosen by :func:`ram_autoselect`."""

    backend: BackendChoice
    reorder: str
    compression_level: float | None
    restart: int
    ranks: int
    precision: Precision


def ram_autoselect(
    L0_pattern: sp.spmatrix | Any, cfg: Mapping[str, Any] | object | None
) -> OperatingPoint:
    """Choose a direct or compressed-sweep operating point from pattern and RAM.

    Parameters
    ----------
    L0_pattern:
        Square sparse operator pattern.  SciPy sparse matrices are supported
        directly; PETSc AIJ matrices are supported on one rank via
        ``getValuesCSR()``.
    cfg:
        Mapping/object with resource-selection options.  Tests and callers may
        inject ``available_ram_bytes`` (or ``ram_bytes``) to make the budget
        deterministic.  The optional ``symbolic_probe`` callable is used only as
        an injection seam; the production default is the SciPy sparse-LU probe.
    """
    pattern = _as_csc_pattern(L0_pattern)
    n = int(pattern.shape[0])
    nnz = int(pattern.nnz)
    budget = int(
        _available_ram_bytes(cfg)
        * _positive_float(
            _cfg_get(cfg, "memory_fraction", _DEFAULT_MEMORY_FRACTION), "memory_fraction"
        )
    )
    if budget <= 0:
        raise ValueError("usable RAM budget must be positive")

    probe_fn = _cfg_get(cfg, "symbolic_probe", None)
    if probe_fn is None:
        probe = _symbolic_analysis_probe(pattern, cfg)
    else:
        probe = probe_fn(pattern, cfg)
    if not isinstance(probe, SymbolicProbeResult):
        raise TypeError("symbolic_probe must return SymbolicProbeResult")
    if probe.fill_count <= 0 or probe.bytes_per_nonzero <= 0.0:
        raise ValueError("symbolic probe returned non-positive fill or bytes-per-nonzero")

    reorder = str(_cfg_get(cfg, "reorder", probe.reorder or _DEFAULT_REORDER)).strip().lower()
    base_ranks = _positive_int(_cfg_get(cfg, "ranks", 1), "ranks")
    direct = OperatingPoint(
        backend="direct",
        reorder=reorder,
        compression_level=None,
        restart=0,
        ranks=base_ranks,
        precision="double",
    )
    if (
        _footprint_bytes(
            n=n,
            nnz=nnz,
            probe=probe,
            precision=direct.precision,
            ranks=direct.ranks,
            restart=direct.restart,
            compression_ratio=1.0,
            cfg=cfg,
        )
        <= budget
    ):
        return direct

    rank_candidates = _rank_candidates(cfg, base_ranks)
    precision_candidates = _precision_candidates(cfg)
    compression_levels, compression_ratios = _compression_candidates(cfg)
    restart = _positive_int(_cfg_get(cfg, "restart", _DEFAULT_RESTART), "restart")

    best_point: OperatingPoint | None = None
    best_bytes = float("inf")
    for level, ratio in zip(compression_levels, compression_ratios, strict=True):
        for precision in precision_candidates:
            for ranks in rank_candidates:
                point = OperatingPoint(
                    backend="compressed-sweep",
                    reorder=reorder,
                    compression_level=float(level),
                    restart=restart,
                    ranks=int(ranks),
                    precision=precision,
                )
                estimate = _footprint_bytes(
                    n=n,
                    nnz=nnz,
                    probe=probe,
                    precision=point.precision,
                    ranks=point.ranks,
                    restart=point.restart,
                    compression_ratio=float(ratio),
                    cfg=cfg,
                )
                if estimate < best_bytes:
                    best_point = point
                    best_bytes = estimate
                if estimate <= budget:
                    return point

    raise MemoryError(
        "no ram_autoselect operating point fits the usable RAM budget "
        f"({budget} bytes); smallest estimate was {best_bytes:.0f} bytes for {best_point}"
    )


def available_ram_bytes(cfg: Mapping[str, Any] | object | None = None) -> int:
    """Return available RAM in bytes, honoring deterministic cfg injection first."""
    return _available_ram_bytes(cfg)


def _cfg_get(cfg: Mapping[str, Any] | object | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        if key in cfg:
            return cfg[key]
        nested = cfg.get("ram_autoselect")
        if isinstance(nested, Mapping) and key in nested:
            return nested[key]
        return default
    nested = getattr(cfg, "ram_autoselect", None)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(nested, Mapping) and key in nested:
        return nested[key]
    return getattr(nested, key, default) if nested is not None else default


def _available_ram_bytes(cfg: Mapping[str, Any] | object | None) -> int:
    injected = _cfg_first(
        cfg, ("available_ram_bytes", "ram_bytes", "available_memory_bytes", "memory_budget_bytes")
    )
    if injected is not None:
        return _positive_int(injected, "available_ram_bytes")

    proc_value = _proc_mem_available_bytes()
    if proc_value is not None:
        return proc_value

    pages = _sysconf_int("SC_AVPHYS_PAGES")
    page_size = _sysconf_int("SC_PAGE_SIZE")
    if pages is None:
        pages = _sysconf_int("SC_PHYS_PAGES")
    if pages is not None and page_size is not None:
        return int(pages * page_size)
    raise RuntimeError("could not determine available RAM; inject available_ram_bytes in cfg")


def _cfg_first(cfg: Mapping[str, Any] | object | None, keys: Sequence[str]) -> Any:
    for key in keys:
        value = _cfg_get(cfg, key, None)
        if value is not None:
            return value
    return None


def _proc_mem_available_bytes() -> int | None:
    path = Path("/proc/meminfo")
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def _sysconf_int(name: str) -> int | None:
    try:
        value = os.sysconf(name)
    except (AttributeError, OSError, ValueError):
        return None
    return int(value) if value and value > 0 else None


def _as_csc_pattern(matrix: sp.spmatrix | Any) -> sp.csc_matrix:
    if sp.issparse(matrix):
        pattern = sp.csc_matrix(matrix, copy=True)
    elif hasattr(matrix, "getValuesCSR") and hasattr(matrix, "getSize"):
        comm = matrix.getComm() if hasattr(matrix, "getComm") else None
        if comm is not None and hasattr(comm, "getSize") and int(comm.getSize()) != 1:
            raise NotImplementedError(
                "PETSc ram_autoselect probe supports serial AIJ matrices only"
            )
        rows, cols = matrix.getSize()
        indptr, indices, _values = matrix.getValuesCSR()
        pattern = sp.csr_matrix(
            (
                np.ones(len(indices), dtype=complex),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(int(rows), int(cols)),
        ).tocsc()
    else:
        raise TypeError("L0_pattern must be a SciPy sparse matrix or serial PETSc AIJ matrix")
    if pattern.shape[0] != pattern.shape[1]:
        raise ValueError("L0_pattern must be square")
    pattern.sum_duplicates()
    if pattern.nnz == 0:
        raise ValueError("L0_pattern must contain nonzeros")
    return pattern


def _symbolic_analysis_probe(
    pattern: sp.spmatrix, cfg: Mapping[str, Any] | object | None
) -> SymbolicProbeResult:
    pattern = _as_csc_pattern(pattern)
    reorder = (
        str(_cfg_get(cfg, "probe_reorder", _cfg_get(cfg, "reorder", _DEFAULT_REORDER)))
        .strip()
        .lower()
    )
    if reorder not in _SPLU_REORDER:
        reorder = _DEFAULT_REORDER
    probe_matrix = _well_conditioned_matrix_with_pattern(pattern)
    try:
        lu = spla.splu(probe_matrix, permc_spec=_SPLU_REORDER[reorder])
    except RuntimeError as exc:
        raise RuntimeError("symbolic sparse-LU probe failed for supplied pattern") from exc
    fill_count = int(lu.L.nnz + lu.U.nnz)
    factor_bytes = _sparse_nbytes(lu.L) + _sparse_nbytes(lu.U)
    bytes_per_nonzero = float(factor_bytes / fill_count)
    return SymbolicProbeResult(
        fill_count=fill_count,
        bytes_per_nonzero=bytes_per_nonzero,
        factor_bytes=int(factor_bytes),
        nnz=int(pattern.nnz),
        reorder=reorder,
    )


def _well_conditioned_matrix_with_pattern(pattern: sp.spmatrix) -> sp.csc_matrix:
    coo = sp.coo_matrix(pattern)
    n = int(coo.shape[0])
    rows = coo.row.astype(np.int64, copy=False)
    cols = coo.col.astype(np.int64, copy=False)
    data = np.empty(rows.size, dtype=complex)
    diag_present = np.zeros(n, dtype=bool)
    offdiag_abs = np.zeros(n, dtype=float)

    for k, (row, col) in enumerate(zip(rows, cols, strict=True)):
        if row == col:
            diag_present[row] = True
            data[k] = 1.0 + 0.0j
        else:
            scale = 1.0e-3 / (1.0 + abs(int(row) - int(col)))
            value = scale * (1.0 + 0.1j * ((int(row) + int(col)) % 5 + 1))
            data[k] = value
            offdiag_abs[row] += abs(value)

    if not bool(np.all(diag_present)):
        raise ValueError("L0_pattern must contain an explicit diagonal for the LU probe")
    for k, (row, col) in enumerate(zip(rows, cols, strict=True)):
        if row == col:
            data[k] = 2.0 + offdiag_abs[row] + 0.01j * ((int(row) % 7) + 1)

    return sp.coo_matrix((data, (rows, cols)), shape=coo.shape).tocsc()


def _sparse_nbytes(matrix: sp.spmatrix) -> int:
    compressed = matrix.tocsc()
    return int(compressed.data.nbytes + compressed.indices.nbytes + compressed.indptr.nbytes)


def _footprint_bytes(
    *,
    n: int,
    nnz: int,
    probe: SymbolicProbeResult,
    precision: Precision,
    ranks: int,
    restart: int,
    compression_ratio: float,
    cfg: Mapping[str, Any] | object | None,
) -> float:
    scalar_bytes = _precision_bytes(precision)
    value_scale = scalar_bytes / _COMPLEX128_BYTES
    operator_multiplier = _positive_float(
        _cfg_get(cfg, "operator_value_multiplier", _DEFAULT_OPERATOR_VALUE_MULTIPLIER),
        "operator_value_multiplier",
    )
    operator_bytes_prior = nnz * operator_multiplier * scalar_bytes
    measured_factor_bytes = (
        probe.fill_count * probe.bytes_per_nonzero * value_scale * compression_ratio
    )
    krylov_bytes = max(0, int(restart)) * n * scalar_bytes
    return float((operator_bytes_prior + measured_factor_bytes + krylov_bytes) / ranks)


def _precision_bytes(precision: Precision) -> int:
    if precision == "double":
        return _COMPLEX128_BYTES
    if precision == "single":
        return _COMPLEX64_BYTES
    raise ValueError(f"unsupported precision {precision!r}")


def _rank_candidates(cfg: Mapping[str, Any] | object | None, base_ranks: int) -> tuple[int, ...]:
    raw = _cfg_get(cfg, "rank_candidates", None)
    if raw is None:
        max_ranks = _positive_int(_cfg_get(cfg, "max_ranks", base_ranks), "max_ranks")
        candidates = [base_ranks]
        value = base_ranks
        while value < max_ranks:
            value = min(max_ranks, value * 2)
            candidates.append(value)
    else:
        candidates = [_positive_int(item, "rank_candidates") for item in raw]
    return tuple(sorted(set(candidates)))


def _precision_candidates(cfg: Mapping[str, Any] | object | None) -> tuple[Precision, ...]:
    raw = _cfg_get(cfg, "precision_candidates", ("double", "single"))
    values: list[Precision] = []
    for item in raw:
        text = str(item).strip().lower()
        if text not in {"double", "single"}:
            raise ValueError("precision_candidates entries must be 'double' or 'single'")
        values.append(text)  # type: ignore[arg-type]
    if not values:
        raise ValueError("precision_candidates must not be empty")
    return tuple(dict.fromkeys(values))


def _compression_candidates(
    cfg: Mapping[str, Any] | object | None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    levels = tuple(
        float(item) for item in _cfg_get(cfg, "compression_levels", _DEFAULT_COMPRESSION_LEVELS)
    )
    ratios = tuple(
        float(item)
        for item in _cfg_get(cfg, "compression_memory_ratios", _DEFAULT_COMPRESSION_MEMORY_RATIOS)
    )
    if len(levels) != len(ratios):
        raise ValueError(
            "compression_levels and compression_memory_ratios must have the same length"
        )
    if not levels:
        raise ValueError("compression_levels must not be empty")
    for level in levels:
        if not np.isfinite(level) or level <= 0.0:
            raise ValueError("compression_levels entries must be positive finite numbers")
    for ratio in ratios:
        if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
            raise ValueError("compression_memory_ratios entries must be finite numbers in [0, 1]")
    return levels, ratios


def _positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


__all__ = ["OperatingPoint", "SymbolicProbeResult", "available_ram_bytes", "ram_autoselect"]
