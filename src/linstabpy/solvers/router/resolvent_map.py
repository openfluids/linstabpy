# SPDX-License-Identifier: GPL-3.0-only
"""Turnkey beta/frequency resolvent gain-map orchestration.

This entry point deliberately orchestrates existing certified/direct pieces.  It
assembles one LNSE operator per beta, asks the band-marker for the intended
frequency-band labels, and computes every point through the certified direct
``resolvent_analysis`` path because the discounted and ROM regime engines do not
exist yet.
"""

from __future__ import annotations

import json
import logging
import os
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from linstabpy.solvers.router.band_marking import NEAR_POLE, SMOOTH, STABLE, BandMarks

LOGGER = logging.getLogger(__name__)

REASON_OK = 0
REASON_PENDING = -1
REASON_FAILED = 1

_CP3_COST_KEYS = frozenset(
    {
        "cost_wall_total_s",
        "cost_wall_assemble_s",
        "cost_wall_factor_s",
        "cost_wall_solve_s",
        "cost_peak_rss_bytes",
        "cost_peak_gpu_mem_bytes",
        "cost_mpi_ranks",
        "cost_gpus",
        "cost_petsc_options",
        "cost_hardware_tag",
        "cost_git_sha",
    }
)


@dataclass(frozen=True)
class RouterResult:
    """Result returned by :func:`resolvent_map`.

    Attributes
    ----------
    beta, f:
        One-dimensional validated sweep coordinates.  ``f`` is passed through
        to the existing resolvent frequency seams as ``omega``; no unit
        conversion is performed by the router.
    gain:
        Two-dimensional leading-gain array with shape ``(len(beta), len(f))``.
        Failed points are ``NaN`` and carry a nonzero ``reason_code``.
    regime_label:
        Band labels returned by ``mark_unstable_band`` with the same shape as
        ``gain``.  These are intended routing labels; every computed gain is
        currently produced by the certified/direct regime-(a) engine.
    cost_metrics:
        CP3 cost sidecar dictionary from
        :func:`linstabpy.utils.cost_metrics.collect_cost_metrics`.
    failure_records:
        Structured records for point failures.  A failed point does not abort
        the sweep and is retryable on a later resumed run.
    reason_code:
        Integer per-point status, ``0`` for success, ``1`` for failed, ``-1``
        for not-yet-computed/pending.
    point_records:
        Structured per-point success/skip/failure records emitted during the
        current call.
    """

    beta: np.ndarray
    f: np.ndarray
    gain: np.ndarray
    regime_label: np.ndarray
    cost_metrics: dict[str, object]
    failure_records: list[dict[str, object]] = field(default_factory=list)
    reason_code: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.int16))
    point_records: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class _AssemblyResult:
    L0: Any
    idx: dict[str, Any]
    bc_indices: np.ndarray
    wall_s: float


@dataclass(frozen=True)
class _DirectPointResult:
    gain: float
    iterations: int | None
    residual: float | None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _router_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = _as_mapping(cfg.get("router"))
    if direct:
        return direct
    solver = _as_mapping(cfg.get("solver"))
    return _as_mapping(solver.get("router"))


def _cfg_value(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in cfg:
        return cfg[key]
    router = _router_cfg(cfg)
    if key in router:
        return router[key]
    solver = _as_mapping(cfg.get("solver"))
    if key in solver:
        return solver[key]
    return default


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "", "none", "null"}:
        return False
    raise ValueError(f"{name} must be boolean-like, got {value!r}")


def _cfg_bool(cfg: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = _cfg_value(cfg, key, default)
    return _as_bool(value, name=key)


def _normalise_cfg(cfg: Mapping[str, Any] | str | Path | object | None) -> tuple[Mapping[str, Any], Any]:
    """Return a plain mapping for router keys and a Config-compatible object.

    ``Config`` objects are passed through to the resolver while router-specific
    keys are unavailable unless the caller supplied a mapping or JSONC path.
    """
    if cfg is None:
        return {}, None
    if isinstance(cfg, Mapping):
        return cfg, None
    if isinstance(cfg, (str, Path)):
        from linstabpy.utils.jsonc import load_jsonc

        loaded = load_jsonc(cfg)
        return loaded, None
    return {}, cfg


def _finite_1d_range(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one finite value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _heartbeat_seconds(cfg: Mapping[str, Any]) -> float:
    logging_cfg = _as_mapping(cfg.get("logging"))
    if "stage_heartbeat_seconds" in logging_cfg:
        return float(logging_cfg["stage_heartbeat_seconds"])
    value = _cfg_value(cfg, "stage_heartbeat_seconds", None)
    if value is not None:
        return float(value)
    for env_name in ("LINSTABPY_STAGE_HEARTBEAT_SECONDS", "LINSTAB2D_STAGE_HEARTBEAT_SECONDS"):
        raw = os.getenv(env_name)
        if raw is not None:
            return float(raw)
    return 30.0


@contextmanager
def _heartbeat(stage: str, cfg: Mapping[str, Any]):
    interval = _heartbeat_seconds(cfg)
    try:
        from linstabpy.driver.run import _stage_heartbeat
    except ImportError:
        with nullcontext():
            yield
        return
    with _stage_heartbeat(stage, interval_seconds=interval):
        yield


def _resolve_mesh(base_flow: Any, cfg: Mapping[str, Any]) -> Any:
    mesh = _cfg_value(cfg, "mesh", None)
    if mesh is not None:
        return mesh
    mesh = getattr(base_flow, "mesh", None)
    if mesh is not None:
        return mesh

    mesh_cfg = _as_mapping(cfg.get("mesh_config")) or _as_mapping(_cfg_value(cfg, "mesh_config", {}))
    if mesh_cfg:
        from linstabpy.mesh import Mesh

        ny, nx = np.asarray(base_flow.RHO).shape
        return Mesh.create(
            tuple(mesh_cfg.get("xrange", cfg.get("xrange", (0.0, 1.0)))),
            tuple(mesh_cfg.get("yrange", cfg.get("yrange", (0.0, 1.0)))),
            nx=int(mesh_cfg.get("nx", cfg.get("nx", nx))),
            ny=int(mesh_cfg.get("ny", cfg.get("ny", ny))),
            fd_order=int(mesh_cfg.get("fd_order", cfg.get("fd_order", 4))),
            y_symmetry=_as_bool(mesh_cfg.get("y_symmetry", cfg.get("y_symmetry", False)), name="y_symmetry"),
            x_periodicity=_as_bool(
                mesh_cfg.get("x_periodicity", cfg.get("x_periodicity", False)),
                name="x_periodicity",
            ),
            alpha_filter=mesh_cfg.get("alpha_filter", cfg.get("alpha_filter", "none")),
            numerics_operator_dissipation=mesh_cfg.get("numerics_operator_dissipation"),
        )

    raise ValueError(
        "resolvent_map requires a Mesh via cfg['mesh'], cfg['mesh_config'], or base_flow.mesh"
    )


def _solver_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    explicit = _cfg_value(cfg, "solver_cfg", None)
    if isinstance(explicit, Mapping):
        return explicit
    return cfg


def _assembly_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    bc_recipe = _cfg_value(cfg, "bc_recipe", None)
    if bc_recipe is None:
        bc = _cfg_value(cfg, "bc", None)
        if isinstance(bc, Mapping) and "recipe" in bc:
            bc_recipe = bc["recipe"]
        else:
            bc_recipe = bc
    return {
        "model": str(_cfg_value(cfg, "model", "2D")),
        "m": float(_cfg_value(cfg, "m", 0.0)),
        "floquet_exp": float(_cfg_value(cfg, "floquet_exp", 0.0)),
        "dissipation_epsilon": float(
            _cfg_value(cfg, "dissipation_epsilon", _cfg_value(cfg, "eps_resolv", 0.0))
        ),
        "bc_recipe": bc_recipe,
        "bc_penalty": float(_cfg_value(cfg, "bc_penalty", _cfg_value(cfg, "penalty", 1.0e4))),
        "backend": _cfg_value(cfg, "assembly_backend", None),
        "solver_cfg": _solver_cfg(cfg),
    }


def _assemble_beta_operator(mesh: Any, base_flow: Any, beta: float, cfg: Mapping[str, Any]) -> _AssemblyResult:
    from linstabpy.operators.assembly import build_lnse_system

    kwargs = _assembly_kwargs(cfg)
    start = time.perf_counter()
    L0, idx, bc_indices = build_lnse_system(mesh, base_flow, kx=float(beta), **kwargs)
    return _AssemblyResult(
        L0=L0,
        idx=idx,
        bc_indices=np.asarray(bc_indices, dtype=np.int64).reshape(-1),
        wall_s=time.perf_counter() - start,
    )


def _compute_w_diag(mesh: Any, base_flow: Any, cfg: Mapping[str, Any]) -> np.ndarray:
    explicit = _cfg_value(cfg, "W_diag", None)
    if explicit is not None:
        return np.asarray(explicit, dtype=float).reshape(-1)
    from linstabpy.operators.energy_norm import compute_chu_norm

    model = str(_cfg_value(cfg, "model", "2D"))
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO = base_flow.RHO.ravel(order="F")[p]
    T = base_flow.T.ravel(order="F")[p]
    if model.lower() == "axy":
        R = mesh.Y.ravel(order="F")[p]
        W_diag, _inv = compute_chu_norm(
            W_int, RHO, T, base_flow.kappa, base_flow.Ma, mesh.ngp, model=model, R=R
        )
    else:
        W_diag, _inv = compute_chu_norm(W_int, RHO, T, base_flow.kappa, base_flow.Ma, mesh.ngp)
    return np.asarray(W_diag, dtype=float).reshape(-1)


def _is_petsc_matrix(operator: Any) -> bool:
    return hasattr(operator, "getInfo") and hasattr(operator, "getSize")


def _maybe_petsc_diag(diagonal: np.ndarray, like_operator: Any, solver_cfg: Mapping[str, Any]) -> Any:
    mat = sp.diags(np.asarray(diagonal, dtype=float), format="csc")
    if _is_petsc_matrix(like_operator):
        from linstabpy.solvers.petsc_utils import scipy_to_petsc

        return scipy_to_petsc(mat, solver_cfg)
    return mat


def _forcing_response_mats(
    L0: Any,
    ndof: int,
    bc_indices: np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[Any, Any]:
    B_mat = _cfg_value(cfg, "B_mat", None)
    C_mat = _cfg_value(cfg, "C_mat", None)
    if B_mat is not None or C_mat is not None:
        return B_mat, C_mat
    if not _cfg_bool(cfg, "mask_bc_forcing", True):
        return None, None
    diag = np.ones(ndof, dtype=float)
    valid = bc_indices[(bc_indices >= 0) & (bc_indices < ndof)]
    diag[valid] = 0.0
    B_mat = _maybe_petsc_diag(diag, L0, _solver_cfg(cfg))
    C_mat = B_mat.copy() if hasattr(B_mat, "copy") else B_mat
    return B_mat, C_mat


def _mark_unstable_band(L0_beta: Any, f_range: np.ndarray, cfg: Mapping[str, Any]) -> BandMarks:
    from linstabpy.solvers.router.band_marking import mark_unstable_band

    return mark_unstable_band(L0_beta, f_range, cfg)


def _direct_engine_result(
    L0: Any,
    omega: float,
    W_diag: np.ndarray,
    n_modes: int,
    cfg: Mapping[str, Any],
    cfg_obj: Any,
    *,
    B_mat: Any = None,
    C_mat: Any = None,
    filter_fn: Any = None,
    filter_ct_fn: Any = None,
) -> _DirectPointResult:
    from linstabpy.resolvent.analysis import resolvent_analysis

    gains, _responses, _forcings, rho_fwd, rho_adj = resolvent_analysis(
        L0,
        float(omega),
        int(n_modes),
        W_diag,
        B_mat=B_mat,
        C_mat=C_mat,
        filter_fn=filter_fn,
        filter_ct_fn=filter_ct_fn,
        solver_cfg=_solver_cfg(cfg),
        cfg=cfg_obj,
        residuals=True,
    )
    residual = None
    if np.asarray(rho_fwd).size or np.asarray(rho_adj).size:
        residual = float(
            max(
                float(np.max(np.abs(rho_fwd))) if np.asarray(rho_fwd).size else 0.0,
                float(np.max(np.abs(rho_adj))) if np.asarray(rho_adj).size else 0.0,
            )
        )
    return _DirectPointResult(gain=float(np.asarray(gains, dtype=float).reshape(-1)[0]), iterations=None, residual=residual)


def _intended_missing_engine(label: object) -> tuple[str, str, str] | None:
    label_text = str(label)
    if label_text == NEAR_POLE:
        return "regime-b", "discounted_resolvent", "CP1"
    if label_text == SMOOTH:
        return "regime-c", "rational_rom", "S9 / r3uu.14.9"
    return None


def _check_missing_engine_gate(label: object, cfg: Mapping[str, Any], emitted_notes: set[str]) -> str:
    missing = _intended_missing_engine(label)
    if missing is None:
        return "regime-a"
    regime, engine, bead = missing
    enable_key = "enable_regime_b" if regime == "regime-b" else "enable_regime_c"
    if _cfg_bool(cfg, enable_key, False):
        raise NotImplementedError(
            f"{regime} requested for label {label!r}, but {engine} is not implemented yet "
            f"(missing bead {bead})"
        )
    if regime not in emitted_notes:
        LOGGER.info(
            "resolvent_map fallback: intended_regime=%s missing_engine=%s missing_bead=%s "
            "enabled=false action=regime-a-direct",
            regime,
            engine,
            bead,
        )
        emitted_notes.add(regime)
    return "regime-a"


def _n_modes(cfg: Mapping[str, Any]) -> int:
    return int(_cfg_value(cfg, "n_modes_resolvent", _cfg_value(cfg, "n_modes", 1)))


def _store_path(cfg: Mapping[str, Any]) -> Path | None:
    raw = _cfg_value(
        cfg,
        "output_npz",
        _cfg_value(cfg, "output_store", _cfg_value(cfg, "cache_npz", None)),
    )
    if raw in (None, ""):
        return None
    return Path(raw)


def _json_records(records: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([json.dumps(record, sort_keys=True) for record in records], dtype=str)


def _records_from_json_array(values: np.ndarray) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw in np.asarray(values, dtype=str).reshape(-1):
        try:
            decoded = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            records.append(decoded)
    return records


def _load_store(
    path: Path | None,
    beta: np.ndarray,
    f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    gain = np.full((beta.size, f.size), np.nan, dtype=float)
    labels = np.full((beta.size, f.size), "", dtype="U32")
    reason = np.full((beta.size, f.size), REASON_PENDING, dtype=np.int16)
    failures: list[dict[str, object]] = []
    if path is None or not path.exists():
        return gain, labels, reason, failures
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_beta = np.asarray(data["beta"], dtype=float).reshape(-1)
            cached_f = np.asarray(data["f"], dtype=float).reshape(-1)
            if cached_beta.shape != beta.shape or cached_f.shape != f.shape:
                raise ValueError("range shape mismatch")
            if not (np.allclose(cached_beta, beta, rtol=0.0, atol=0.0) and np.allclose(cached_f, f, rtol=0.0, atol=0.0)):
                raise ValueError("range value mismatch")
            gain = np.asarray(data["gain"], dtype=float).reshape(beta.size, f.size)
            labels = np.asarray(data["regime_label"], dtype=str).reshape(beta.size, f.size)
            if "reason_code" in data.files:
                reason = np.asarray(data["reason_code"], dtype=np.int16).reshape(beta.size, f.size)
            else:
                reason = np.where(np.isfinite(gain), REASON_OK, REASON_PENDING).astype(np.int16)
            if "failure_records_json" in data.files:
                failures = _records_from_json_array(data["failure_records_json"])
    except (OSError, KeyError, ValueError) as exc:
        LOGGER.info("resolvent_map cache ignored path=%s reason=%s", path, exc)
    return gain, labels, reason, failures


def _save_store(
    path: Path | None,
    *,
    beta: np.ndarray,
    f: np.ndarray,
    gain: np.ndarray,
    regime_label: np.ndarray,
    reason_code: np.ndarray,
    failure_records: list[dict[str, object]],
    point_records: list[dict[str, object]],
    cost_metrics: Mapping[str, object] | None = None,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "beta": beta,
        "f": f,
        "gain": gain,
        "regime_label": np.asarray(regime_label, dtype=str),
        "reason_code": reason_code,
        "failure_records_json": _json_records(failure_records),
        "point_records_json": _json_records(point_records),
    }
    if cost_metrics:
        payload.update(cost_metrics)
    np.savez_compressed(path, **payload)


def _collect_cost_metrics(stage_wall: Mapping[str, float]) -> dict[str, object]:
    from linstabpy.utils.cost_metrics import collect_cost_metrics
    from linstabpy.utils.provenance import _git_head_sha

    metrics = collect_cost_metrics(
        dict(stage_wall),
        git_sha=_git_head_sha(),
        peak_rss_bytes=_peak_rss_bytes(),
    )
    missing = _CP3_COST_KEYS - set(metrics)
    if missing:
        raise RuntimeError(f"collect_cost_metrics did not return CP3 keys: {sorted(missing)}")
    return metrics


def _point_index(beta_i: int, f_i: int, nf: int) -> int:
    return beta_i * nf + f_i + 1


def _point_record(
    *,
    status: str,
    point_index: int,
    beta_i: int,
    f_i: int,
    beta: float,
    omega: float,
    total: int,
    regime_label: object,
    engine: str,
    leading_gain: float | None,
    iterations: int | None,
    residual: float | None,
    wall_s: float,
    peak_rss_bytes: int,
    reason_code: int,
    exception: BaseException | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "status": status,
        "index": int(point_index),
        "beta_index": beta_i,
        "f_index": f_i,
        "beta": float(beta),
        "omega": float(omega),
        "total": int(total),
        "regime_label": str(regime_label),
        "engine": engine,
        "leading_gain": None if leading_gain is None else float(leading_gain),
        "iterations": None if iterations is None else int(iterations),
        "residual": None if residual is None else float(residual),
        "wall_s": float(wall_s),
        "peak_rss_bytes": int(peak_rss_bytes),
        "reason_code": int(reason_code),
    }
    if exception is not None:
        record["exception_class"] = exception.__class__.__name__
        record["exception"] = str(exception)
    return record


def resolvent_map(
    base_flow: Any,
    beta_range: Sequence[float] | np.ndarray,
    f_range: Sequence[float] | np.ndarray,
    cfg: Mapping[str, Any] | str | Path | object | None = None,
) -> RouterResult:
    """Compute a leading-gain map over beta and forcing-frequency ranges.

    Parameters
    ----------
    base_flow:
        Base-flow object consumed by the existing LNSE assembly seam.  The mesh
        must be supplied as ``cfg['mesh']``, ``cfg['mesh_config']``, or
        ``base_flow.mesh``.
    beta_range, f_range:
        Non-empty finite one-dimensional ranges.  Values in ``f_range`` are the
        angular frequencies forwarded to ``resolvent_analysis``.
    cfg:
        Mapping, JSONC path, typed ``Config`` object, or ``None``.  Router keys
        may be placed at top level or under ``router`` / ``solver.router``.

    Returns
    -------
    RouterResult
        Gain map, intended band labels, CP3 cost metrics, and point records.
    """
    cfg_map, cfg_obj = _normalise_cfg(cfg)
    beta = _finite_1d_range(beta_range, "beta_range")
    f = _finite_1d_range(f_range, "f_range")
    mesh = _resolve_mesh(base_flow, cfg_map)
    W_diag = _compute_w_diag(mesh, base_flow, cfg_map)
    store = _store_path(cfg_map)

    gain, regime_label, reason_code, failure_records = _load_store(store, beta, f)
    point_records: list[dict[str, object]] = []
    stage_wall: dict[str, float] = {}
    total_points = int(beta.size * f.size)
    n_modes = _n_modes(cfg_map)
    emitted_fallback_notes: set[str] = set()

    with _heartbeat("resolvent_map", cfg_map):
        for beta_i, beta_value in enumerate(beta):
            beta_clean = np.isfinite(gain[beta_i]) & (reason_code[beta_i] == REASON_OK)
            if bool(np.all(beta_clean)):
                LOGGER.info(
                    "resolvent_map beta skipped beta_index=%d beta=%.12g cached_points=%d/%d",
                    beta_i,
                    float(beta_value),
                    int(np.count_nonzero(beta_clean)),
                    int(f.size),
                )
                for f_i, omega in enumerate(f):
                    record = _point_record(
                        status="skipped",
                        point_index=_point_index(beta_i, f_i, f.size),
                        beta_i=beta_i,
                        f_i=f_i,
                        beta=float(beta_value),
                        omega=float(omega),
                        total=total_points,
                        regime_label=regime_label[beta_i, f_i],
                        engine="regime-a",
                        leading_gain=float(gain[beta_i, f_i]),
                        iterations=None,
                        residual=None,
                        wall_s=0.0,
                        peak_rss_bytes=_peak_rss_bytes(),
                        reason_code=int(reason_code[beta_i, f_i]),
                    )
                    point_records.append(record)
                continue

            LOGGER.info("resolvent_map assembling beta_index=%d beta=%.12g", beta_i, float(beta_value))
            assembly = _assemble_beta_operator(mesh, base_flow, float(beta_value), cfg_map)
            stage_wall["assemble_operator_bc"] = stage_wall.get("assemble_operator_bc", 0.0) + assembly.wall_s
            band_start = time.perf_counter()
            marks = _mark_unstable_band(assembly.L0, f, cfg_map)
            stage_wall["stability"] = stage_wall.get("stability", 0.0) + time.perf_counter() - band_start
            labels = np.asarray(marks.labels, dtype=str).reshape(-1)
            if labels.size != f.size:
                raise ValueError(
                    f"mark_unstable_band returned {labels.size} labels for {f.size} frequencies"
                )

            ndof = W_diag.size
            B_mat, C_mat = _forcing_response_mats(assembly.L0, ndof, assembly.bc_indices, cfg_map)
            use_mesh_filters = _cfg_bool(cfg_map, "use_mesh_filters", True)
            filter_fn = mesh.filters.filter if use_mesh_filters and getattr(mesh, "filters", None) is not None else None
            filter_ct_fn = (
                mesh.filters.filter_ct if use_mesh_filters and getattr(mesh, "filters", None) is not None else None
            )

            for f_i, omega in enumerate(f):
                if np.isfinite(gain[beta_i, f_i]) and int(reason_code[beta_i, f_i]) == REASON_OK:
                    LOGGER.info(
                        "resolvent_map point skipped index=%d total=%d beta=%.12g omega=%.12g "
                        "leading_gain=%.12g reason_code=%d",
                        _point_index(beta_i, f_i, f.size),
                        total_points,
                        float(beta_value),
                        float(omega),
                        float(gain[beta_i, f_i]),
                        int(reason_code[beta_i, f_i]),
                    )
                    record = _point_record(
                        status="skipped",
                        point_index=_point_index(beta_i, f_i, f.size),
                        beta_i=beta_i,
                        f_i=f_i,
                        beta=float(beta_value),
                        omega=float(omega),
                        total=total_points,
                        regime_label=regime_label[beta_i, f_i],
                        engine="regime-a",
                        leading_gain=float(gain[beta_i, f_i]),
                        iterations=None,
                        residual=None,
                        wall_s=0.0,
                        peak_rss_bytes=_peak_rss_bytes(),
                        reason_code=int(reason_code[beta_i, f_i]),
                    )
                    point_records.append(record)
                    continue

                regime_label[beta_i, f_i] = labels[f_i]
                engine = _check_missing_engine_gate(labels[f_i], cfg_map, emitted_fallback_notes)
                point_start = time.perf_counter()
                LOGGER.info(
                    "resolvent_map point start index=%d total=%d beta_index=%d f_index=%d "
                    "beta=%.12g omega=%.12g regime_label=%s engine=%s",
                    _point_index(beta_i, f_i, f.size),
                    total_points,
                    beta_i,
                    f_i,
                    float(beta_value),
                    float(omega),
                    str(labels[f_i]),
                    engine,
                )
                try:
                    point = _direct_engine_result(
                        assembly.L0,
                        float(omega),
                        W_diag,
                        n_modes,
                        cfg_map,
                        cfg_obj,
                        B_mat=B_mat,
                        C_mat=C_mat,
                        filter_fn=filter_fn,
                        filter_ct_fn=filter_ct_fn,
                    )
                except NotImplementedError:
                    raise
                except Exception as exc:  # noqa: BLE001 - per-point tolerance is an acceptance criterion.
                    wall_s = time.perf_counter() - point_start
                    stage_wall["resolvent"] = stage_wall.get("resolvent", 0.0) + wall_s
                    gain[beta_i, f_i] = np.nan
                    reason_code[beta_i, f_i] = REASON_FAILED
                    record = _point_record(
                        status="failed",
                        point_index=_point_index(beta_i, f_i, f.size),
                        beta_i=beta_i,
                        f_i=f_i,
                        beta=float(beta_value),
                        omega=float(omega),
                        total=total_points,
                        regime_label=labels[f_i],
                        engine=engine,
                        leading_gain=None,
                        iterations=None,
                        residual=None,
                        wall_s=wall_s,
                        peak_rss_bytes=_peak_rss_bytes(),
                        reason_code=REASON_FAILED,
                        exception=exc,
                    )
                    failure_records.append(record.copy())
                    point_records.append(record)
                    LOGGER.exception(
                        "resolvent_map point failed index=%d total=%d beta=%.12g omega=%.12g "
                        "exception_class=%s iters=%s residual=%s wall_s=%.6g peak_rss_bytes=%d "
                        "reason_code=%d",
                        _point_index(beta_i, f_i, f.size),
                        total_points,
                        float(beta_value),
                        float(omega),
                        exc.__class__.__name__,
                        None,
                        None,
                        wall_s,
                        int(record["peak_rss_bytes"]),
                        REASON_FAILED,
                    )
                    _save_store(
                        store,
                        beta=beta,
                        f=f,
                        gain=gain,
                        regime_label=regime_label,
                        reason_code=reason_code,
                        failure_records=failure_records,
                        point_records=point_records,
                    )
                    continue

                wall_s = time.perf_counter() - point_start
                stage_wall["resolvent"] = stage_wall.get("resolvent", 0.0) + wall_s
                gain[beta_i, f_i] = point.gain
                reason_code[beta_i, f_i] = REASON_OK
                record = _point_record(
                    status="ok",
                    point_index=_point_index(beta_i, f_i, f.size),
                    beta_i=beta_i,
                    f_i=f_i,
                    beta=float(beta_value),
                    omega=float(omega),
                    total=total_points,
                    regime_label=labels[f_i],
                    engine=engine,
                    leading_gain=point.gain,
                    iterations=point.iterations,
                    residual=point.residual,
                    wall_s=wall_s,
                    peak_rss_bytes=_peak_rss_bytes(),
                    reason_code=REASON_OK,
                )
                point_records.append(record)
                LOGGER.info(
                    "resolvent_map point complete index=%d total=%d beta=%.12g omega=%.12g "
                    "leading_gain=%.12g iters=%s residual=%s wall_s=%.6g peak_rss_bytes=%d",
                    _point_index(beta_i, f_i, f.size),
                    total_points,
                    float(beta_value),
                    float(omega),
                    point.gain,
                    point.iterations,
                    None if point.residual is None else f"{point.residual:.6e}",
                    wall_s,
                    int(record["peak_rss_bytes"]),
                )
                _save_store(
                    store,
                    beta=beta,
                    f=f,
                    gain=gain,
                    regime_label=regime_label,
                    reason_code=reason_code,
                    failure_records=failure_records,
                    point_records=point_records,
                )

    cost_metrics = _collect_cost_metrics(stage_wall)
    _save_store(
        store,
        beta=beta,
        f=f,
        gain=gain,
        regime_label=regime_label,
        reason_code=reason_code,
        failure_records=failure_records,
        point_records=point_records,
        cost_metrics=cost_metrics,
    )
    return RouterResult(
        beta=beta,
        f=f,
        gain=gain,
        regime_label=np.asarray(regime_label, dtype=str),
        cost_metrics=cost_metrics,
        failure_records=failure_records,
        reason_code=reason_code,
        point_records=point_records,
    )


__all__ = ["RouterResult", "resolvent_map"]
