# SPDX-License-Identifier: GPL-3.0-only
"""Turnkey beta/frequency resolvent gain-map orchestration.

This entry point deliberately orchestrates existing certified/direct pieces.  It
assembles one LNSE operator per beta, asks the band-marker for the intended
frequency-band labels, and computes every point through the certified direct
``resolvent_analysis`` path because the discounted and ROM regime engines do not
exist yet.

SBP sawtooth dissipation (required for clean gain maps)
-------------------------------------------------------
Without operator-level SBP dissipation the assembled operator carries
grid-scale (odd/even "sawtooth") modes.  These show up in a gain map as
spurious near-singular resonances at high spanwise wavenumber -- bright
horizontal stripes (constant ``beta``, all low frequencies) whose gain can be
many decades above the physical OM-1 / LU-S responses.  To suppress them this
path needs BOTH halves of the dissipation, set independently:

* ``numerics.operator_dissipation = {"enabled": true, ...}`` -- builds the SBP
  dissipation operators ``Q_diss_x/Q_diss_y`` onto the mesh.
* top-level ``dissipation_epsilon`` (alias ``eps_resolv``) > 0 -- the strength
  by which ``build_lnse_system`` subtracts ``Q_diss`` from ``L0``.

GOTCHA: this router reads the strength from the *top-level* ``dissipation_epsilon``,
NOT from ``operator_dissipation.epsilon`` (that nested field is consumed by the
non-router case2 assembly path).  Enabling ``operator_dissipation`` alone, or
setting ``dissipation_epsilon`` without enabling ``operator_dissipation``, is a
silent no-op -- the operator is unchanged.  ``epsilon`` must clear the sawtooth
poles yet stay below the physical-mode scale (for the Song SWBLI fd4 stride-4
case the sweet spot is ``~0.10``; ``1.0`` over-damps the real OM-1/LU-S ridges
by ~100x).
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
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from linstabpy.solvers.router.anchors import choose_anchors_adaptive
from linstabpy.solvers.router.band_marking import NEAR_POLE, SMOOTH, STABLE, BandMarks
from linstabpy.solvers.router.stable_sweep import sweep_stable_regime

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


def _analysis_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(cfg.get("analysis"))


def _analysis_bool(cfg: Mapping[str, Any], key: str, default: bool = False) -> bool:
    analysis = _analysis_cfg(cfg)
    value = analysis.get(key, _cfg_value(cfg, key, default))
    return _as_bool(value, name=f"analysis.{key}")


def _stable_sweep_cfg(cfg: Mapping[str, Any], W_diag: np.ndarray, cfg_obj: Any) -> dict[str, Any]:
    analysis = _analysis_cfg(cfg)
    sweep_cfg: dict[str, Any] = {}
    sweep_cfg.update(cfg)
    sweep_cfg.update(analysis)
    W = np.asarray(W_diag, dtype=complex).reshape(-1)
    sweep_cfg["W"] = W
    sweep_cfg["invW"] = 1.0 / W
    sweep_cfg["solver_cfg"] = _solver_cfg(cfg)
    if cfg_obj is not None:
        sweep_cfg["factorize_cfg"] = cfg_obj
    return sweep_cfg


def _normalise_cfg(
    cfg: Mapping[str, Any] | str | Path | object | None,
) -> tuple[Mapping[str, Any], Any]:
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

    mesh_cfg = _as_mapping(cfg.get("mesh_config")) or _as_mapping(
        _cfg_value(cfg, "mesh_config", {})
    )
    if mesh_cfg:
        from linstabpy.mesh import Mesh

        ny, nx = np.asarray(base_flow.RHO).shape
        return Mesh.create(
            tuple(mesh_cfg.get("xrange", cfg.get("xrange", (0.0, 1.0)))),
            tuple(mesh_cfg.get("yrange", cfg.get("yrange", (0.0, 1.0)))),
            nx=int(mesh_cfg.get("nx", cfg.get("nx", nx))),
            ny=int(mesh_cfg.get("ny", cfg.get("ny", ny))),
            fd_order=int(mesh_cfg.get("fd_order", cfg.get("fd_order", 4))),
            y_symmetry=_as_bool(
                mesh_cfg.get("y_symmetry", cfg.get("y_symmetry", False)), name="y_symmetry"
            ),
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


def _assemble_beta_operator(
    mesh: Any, base_flow: Any, beta: float, cfg: Mapping[str, Any]
) -> _AssemblyResult:
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


def _maybe_petsc_diag(
    diagonal: np.ndarray, like_operator: Any, solver_cfg: Mapping[str, Any]
) -> Any:
    mat = sp.diags(np.asarray(diagonal, dtype=float), format="csc")
    if _is_petsc_matrix(like_operator):
        from linstabpy.solvers.petsc_utils import scipy_to_petsc

        return scipy_to_petsc(mat, solver_cfg)
    return mat


_REGION_VARS = ("rho", "u", "v", "w", "T")


def _region_bounds(
    values: np.ndarray, raw_bounds: Any, *, axis: str, name: str
) -> tuple[float, float, bool]:
    if raw_bounds is None:
        return float(np.min(values)), float(np.max(values)), False
    bounds = np.asarray(raw_bounds, dtype=float).reshape(-1)
    if bounds.size != 2:
        raise ValueError(f"{name} {axis} bounds must contain exactly two finite values")
    if not np.all(np.isfinite(bounds)):
        raise ValueError(f"{name} {axis} bounds must contain exactly two finite values")
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower > upper:
        raise ValueError(f"{name} {axis} bounds must be increasing, got [{lower}, {upper}]")
    return lower, upper, True


def _region_vars(region: Mapping[str, Any], *, name: str) -> tuple[str, ...]:
    raw_vars = region.get("vars", _REGION_VARS)
    if raw_vars is None:
        raw_vars = _REGION_VARS
    if isinstance(raw_vars, str):
        requested = (raw_vars,)
    else:
        try:
            requested = tuple(raw_vars)
        except TypeError as exc:
            raise ValueError(f"{name} vars must be an iterable of variable names") from exc
    if not requested:
        raise ValueError(f"{name} vars must select at least one variable")
    unknown = [str(var) for var in requested if str(var) not in _REGION_VARS]
    if unknown:
        raise ValueError(
            f"{name} unknown variable name(s): {unknown}; valid variables are {list(_REGION_VARS)}"
        )
    return tuple(str(var) for var in requested)


def _build_region_mask(
    mesh: Any,
    idx: Mapping[str, Any],
    ndof: int,
    region_spec: Mapping[str, Any],
    like_operator: Any,
    solver_cfg: Mapping[str, Any],
    *,
    name: str = "region",
    bc_indices: np.ndarray | None = None,
) -> Any:
    """Build a sharp 0/1 diagonal forcing/response mask from a user region.

    The mask follows the LNSE state layout exactly: active physical points are
    taken from ``mesh.X/Y.ravel(order="F")[mesh.used_ind]`` and variable blocks
    from ``idx['rho_j']``, ``idx['u_j']``, ``idx['v_j']``, ``idx['w_j']``, and
    ``idx['T_j']``.  A region mask changes the resolvent question to a
    windowed/component-restricted gain; it is not an approximation to the global
    unmasked gain.
    """
    if not isinstance(region_spec, Mapping):
        raise ValueError(f"{name} must be a mapping with optional x, y, and vars keys")

    used_ind = np.asarray(mesh.used_ind)
    X_act = np.asarray(mesh.X, dtype=float).ravel(order="F")[used_ind]
    Y_act = np.asarray(mesh.Y, dtype=float).ravel(order="F")[used_ind]
    if X_act.size == 0 or Y_act.size == 0:
        raise ValueError(f"{name} cannot build region mask: zero active points in mesh")
    if X_act.size != Y_act.size:
        raise ValueError(f"{name} cannot build region mask: X/Y active point count mismatch")
    if not (np.all(np.isfinite(X_act)) and np.all(np.isfinite(Y_act))):
        raise ValueError(f"{name} cannot build region mask: active coordinates must be finite")

    x0, x1, x_explicit = _region_bounds(X_act, region_spec.get("x"), axis="x", name=name)
    y0, y1, y_explicit = _region_bounds(Y_act, region_spec.get("y"), axis="y", name=name)
    variables = _region_vars(region_spec, name=name)

    domain_x0, domain_x1 = float(np.min(X_act)), float(np.max(X_act))
    domain_y0, domain_y1 = float(np.min(Y_act)), float(np.max(Y_act))
    outside_x = x_explicit and (x1 < domain_x0 or x0 > domain_x1)
    outside_y = y_explicit and (y1 < domain_y0 or y0 > domain_y1)
    if outside_x or outside_y:
        raise ValueError(
            f"{name} region is outside domain: requested x=[{x0}, {x1}] y=[{y0}, {y1}], "
            f"domain x=[{domain_x0}, {domain_x1}] y=[{domain_y0}, {domain_y1}]"
        )

    point_mask = (X_act >= x0) & (X_act <= x1) & (Y_act >= y0) & (Y_act <= y1)
    point_indices = np.nonzero(point_mask)[0]
    if point_indices.size == 0:
        raise ValueError(
            f"{name} region selects zero active points: requested x=[{x0}, {x1}] y=[{y0}, {y1}], "
            f"domain x=[{domain_x0}, {domain_x1}] y=[{domain_y0}, {domain_y1}]"
        )

    diag = np.zeros(int(ndof), dtype=float)
    selected_x = X_act[point_indices]
    selected_y = Y_act[point_indices]
    bbox = (
        float(np.min(selected_x)),
        float(np.max(selected_x)),
        float(np.min(selected_y)),
        float(np.max(selected_y)),
    )
    n_total = int(X_act.size)
    for var in variables:
        idx_key = f"{var}_j"
        if idx_key not in idx:
            raise ValueError(f"{name} cannot build region mask: missing index block {idx_key!r}")
        var_idx = np.asarray(idx[idx_key], dtype=np.int64).reshape(-1)
        if var_idx.size != n_total:
            raise ValueError(
                f"{name} cannot build region mask: index block {idx_key!r} has "
                f"{var_idx.size} entries for {n_total} active points"
            )
        dofs = var_idx[point_indices]
        if np.any((dofs < 0) | (dofs >= int(ndof))):
            raise ValueError(
                f"{name} cannot build region mask: index block {idx_key!r} exceeds ndof={ndof}"
            )
        diag[dofs] = 1.0
        LOGGER.info(
            "resolvent_map region_mask name=%s var=%s n_active=%d n_total=%d "
            "bbox_x=[%.12g,%.12g] bbox_y=[%.12g,%.12g]",
            name,
            var,
            int(point_indices.size),
            n_total,
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
        )

    if bc_indices is not None:
        # A boundary-condition row is a constraint, not an equation, so forcing it is
        # meaningless: the penalty makes the response violate the very condition it
        # encodes. The region and the BC mask therefore COMPOSE -- the region says
        # where the user wants to force, the BC mask says where forcing has meaning.
        # Without this a region mask silently switches off `mask_bc_forcing`, which
        # defaults to true precisely to keep forcing off those rows.
        bc = np.asarray(bc_indices, dtype=np.int64).reshape(-1)
        bc = bc[(bc >= 0) & (bc < int(ndof))]
        removed = int(np.count_nonzero(diag[bc]))
        diag[bc] = 0.0
        LOGGER.info(
            "resolvent_map region_mask name=%s composed with bc mask: "
            "%d of %d selected dofs dropped as boundary-condition rows",
            name,
            removed,
            removed + int(np.count_nonzero(diag)),
        )
        if not np.any(diag):
            raise ValueError(
                f"{name} selects only boundary-condition rows: after composing with the "
                "BC mask nothing is left to force. Widen the region, or set "
                "mask_bc_forcing=false to force constrained rows deliberately."
            )
    return _maybe_petsc_diag(diag, like_operator, solver_cfg)


def _bc_mask(L0: Any, ndof: int, bc_indices: np.ndarray, solver_cfg: Mapping[str, Any]) -> Any:
    diag = np.ones(ndof, dtype=float)
    bc = np.asarray(bc_indices, dtype=np.int64).reshape(-1)
    valid = bc[(bc >= 0) & (bc < ndof)]
    diag[valid] = 0.0
    return _maybe_petsc_diag(diag, L0, solver_cfg)


def _forcing_response_mats(
    L0: Any,
    ndof: int,
    bc_indices: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    mesh: Any | None = None,
    idx: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    solver_cfg = _solver_cfg(cfg)
    B_mat = _cfg_value(cfg, "B_mat", None)
    C_mat = _cfg_value(cfg, "C_mat", None)
    input_region = _cfg_value(cfg, "input_region", None)
    output_region = _cfg_value(cfg, "output_region", None)
    if (input_region is not None or output_region is not None) and (mesh is None or idx is None):
        raise ValueError(
            "input_region/output_region masks require mesh and idx from build_lnse_system"
        )
    compose_bc = _cfg_bool(cfg, "mask_bc_forcing", True)
    region_bc = bc_indices if compose_bc else None
    if B_mat is None and input_region is not None:
        B_mat = _build_region_mask(
            mesh, idx, ndof, input_region, L0, solver_cfg, name="input_region",
            bc_indices=region_bc,
        )
    if C_mat is None and output_region is not None:
        C_mat = _build_region_mask(
            mesh, idx, ndof, output_region, L0, solver_cfg, name="output_region",
            bc_indices=region_bc,
        )
    if _cfg_bool(cfg, "mask_bc_forcing", True):
        default_mask = None
        if B_mat is None:
            default_mask = _bc_mask(L0, ndof, bc_indices, solver_cfg)
            B_mat = default_mask
        if C_mat is None:
            if default_mask is None:
                default_mask = _bc_mask(L0, ndof, bc_indices, solver_cfg)
            C_mat = default_mask.copy() if hasattr(default_mask, "copy") else default_mask
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
    gamma: float = 0.0,
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
        gamma=gamma,
    )
    residual = None
    if np.asarray(rho_fwd).size or np.asarray(rho_adj).size:
        residual = float(
            max(
                float(np.max(np.abs(rho_fwd))) if np.asarray(rho_fwd).size else 0.0,
                float(np.max(np.abs(rho_adj))) if np.asarray(rho_adj).size else 0.0,
            )
        )
    return _DirectPointResult(
        gain=float(np.asarray(gains, dtype=float).reshape(-1)[0]),
        iterations=None,
        residual=residual,
    )


def _regime_b_discount_gamma(cfg: Mapping[str, Any]) -> float:
    """Select discount gamma for regime-b NEAR_POLE points.

    Policy:
    - Use cfg["regime_b_discount_gamma"] if present (explicit override)
    - Else: derive from pole proximity to keep the shifted solve well-posed
      Default: 0.25 (validated in test_resolvent_discount.py to tame 1e5 blowup to ~4.0)
    """
    explicit = _cfg_value(cfg, "regime_b_discount_gamma", None)
    if explicit is not None:
        gamma = float(explicit)
        if not (gamma > 0.0):
            raise ValueError(f"regime_b_discount_gamma must be positive, got {gamma}")
        return gamma
    # Default: use 0.25 (validated in CP1/r3uu.3.12 to handle 1e5 poles gracefully)
    return 0.25


def _intended_missing_engine(label: object) -> tuple[str, str, str] | None:
    label_text = str(label)
    if label_text == NEAR_POLE:
        return "regime-b", "discounted_resolvent", "CP1"
    if label_text == SMOOTH:
        return "regime-c", "rational_rom", "S9 / r3uu.14.9"
    return None


def _check_missing_engine_gate(
    label: object, cfg: Mapping[str, Any], emitted_notes: set[str]
) -> tuple[str, float]:
    """Determine which regime to use and any discount gamma.

    Returns (regime, gamma) tuple where:
    - regime is "regime-a", "regime-b", or "regime-c"
    - gamma is 0.0 for regimes a/c, or >0 for regime-b
    """
    missing = _intended_missing_engine(label)
    if missing is None:
        return "regime-a", 0.0
    regime, engine, bead = missing

    # For regime-b (NEAR_POLE), attempt discounted resolvent
    if regime == "regime-b":
        gamma = _regime_b_discount_gamma(cfg)
        return regime, gamma

    # For regime-c (SMOOTH), still raise NotImplementedError
    enable_key = "enable_regime_c"
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
    return "regime-a", 0.0


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


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if np.iscomplexobj(arr):
            return {"real": arr.real.tolist(), "imag": arr.imag.tolist()}
        return arr.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return value


def _json_records(records: list[dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [json.dumps(_json_safe(record), sort_keys=True) for record in records], dtype=str
    )


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
            if not (
                np.allclose(cached_beta, beta, rtol=0.0, atol=0.0)
                and np.allclose(cached_f, f, rtol=0.0, atol=0.0)
            ):
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
    use_amortized_stable_sweep = _analysis_bool(cfg_map, "amortized_stable_sweep", False)

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

            LOGGER.info(
                "resolvent_map assembling beta_index=%d beta=%.12g", beta_i, float(beta_value)
            )
            assembly = _assemble_beta_operator(mesh, base_flow, float(beta_value), cfg_map)
            stage_wall["assemble_operator_bc"] = (
                stage_wall.get("assemble_operator_bc", 0.0) + assembly.wall_s
            )
            band_start = time.perf_counter()
            if _analysis_bool(cfg_map, "skip_band_marking", False):
                # Opt-in: skip the band-marking EVP and label every frequency
                # regime-a (stable interior). The band-marking eigenproblem uses a
                # real shift sigma=-median(omega) that sits near the spectrum; its
                # shift-invert does not scale on GPU at high resolution (the
                # well-conditioned imaginary-shift resolvent solve does). For a base
                # flow / frequency range known to be away from global-mode poles
                # this avoids that full-size EVP. WARNING: a frequency that is in
                # fact near a pole is then handled by the certified resolvent
                # directly (no regime-b discounting), so only enable it when the
                # stable-regime assumption holds.
                omega_arr = np.asarray(f, dtype=float).reshape(-1)
                marks = BandMarks(
                    omegas=omega_arr,
                    labels=np.full(omega_arr.shape, STABLE, dtype=object),
                    distances=np.full(omega_arr.shape, np.inf),
                    danger_radius=0.0,
                )
                LOGGER.info(
                    "resolvent_map: band-marking SKIPPED (skip_band_marking=true) "
                    "-> all %d frequencies labelled regime-a",
                    omega_arr.size,
                )
            else:
                marks = _mark_unstable_band(assembly.L0, f, cfg_map)
            stage_wall["stability"] = (
                stage_wall.get("stability", 0.0) + time.perf_counter() - band_start
            )
            labels = np.asarray(marks.labels, dtype=str).reshape(-1)
            if labels.size != f.size:
                raise ValueError(
                    f"mark_unstable_band returned {labels.size} labels for {f.size} frequencies"
                )

            ndof = W_diag.size
            B_mat, C_mat = _forcing_response_mats(
                assembly.L0,
                ndof,
                assembly.bc_indices,
                cfg_map,
                mesh=mesh,
                idx=assembly.idx,
            )
            use_mesh_filters = _cfg_bool(cfg_map, "use_mesh_filters", True)
            filter_fn = (
                mesh.filters.filter
                if use_mesh_filters and getattr(mesh, "filters", None) is not None
                else None
            )
            filter_ct_fn = (
                mesh.filters.filter_ct
                if use_mesh_filters and getattr(mesh, "filters", None) is not None
                else None
            )
            regime_label[beta_i, :] = labels
            sweep_completed: set[int] = set()

            stable_indices = [
                int(f_i)
                for f_i, label in enumerate(labels)
                if str(label) == STABLE
                and not (
                    np.isfinite(gain[beta_i, f_i]) and int(reason_code[beta_i, f_i]) == REASON_OK
                )
            ]
            can_use_sweep = (
                use_amortized_stable_sweep
                and stable_indices
                and B_mat is None
                and C_mat is None
                and filter_fn is None
                and filter_ct_fn is None
            )
            if can_use_sweep:
                sweep_start = time.perf_counter()
                stable_omegas = np.asarray(f[stable_indices], dtype=complex)
                sweep_cfg = _stable_sweep_cfg(cfg_map, W_diag, cfg_obj)
                try:
                    anchors, anchor_records = choose_anchors_adaptive(
                        stable_omegas, marks, sweep_cfg
                    )
                    sweep = sweep_stable_regime(assembly.L0, stable_omegas, anchors, sweep_cfg)
                except Exception as exc:  # noqa: BLE001 - fallback to certified direct path per point.
                    wall_s = time.perf_counter() - sweep_start
                    stage_wall["resolvent"] = stage_wall.get("resolvent", 0.0) + wall_s
                    LOGGER.exception(
                        "resolvent_map stable sweep failed beta_index=%d beta=%.12g points=%d "
                        "exception_class=%s action=direct-fallback",
                        beta_i,
                        float(beta_value),
                        len(stable_indices),
                        exc.__class__.__name__,
                    )
                else:
                    wall_s = time.perf_counter() - sweep_start
                    stage_wall["resolvent"] = stage_wall.get("resolvent", 0.0) + wall_s
                    sweep_omegas = np.asarray(sweep.omega, dtype=complex).reshape(-1)
                    sweep_gains = np.asarray(sweep.gain, dtype=float).reshape(-1)
                    sweep_iterations = np.asarray(sweep.iterations).reshape(-1)
                    sweep_residuals = np.asarray(sweep.residuals, dtype=float).reshape(-1)
                    if (
                        sweep_omegas.size != len(stable_indices)
                        or sweep_gains.size != len(stable_indices)
                        or sweep_iterations.size != len(stable_indices)
                        or sweep_residuals.size != len(stable_indices)
                    ):
                        LOGGER.error(
                            "resolvent_map stable sweep returned mismatched result lengths "
                            "beta_index=%d expected=%d omega=%d gain=%d iterations=%d residuals=%d "
                            "action=direct-fallback",
                            beta_i,
                            len(stable_indices),
                            sweep_omegas.size,
                            sweep_gains.size,
                            sweep_iterations.size,
                            sweep_residuals.size,
                        )
                    else:
                        response_modes = np.asarray(sweep.responses)
                        forcing_modes = np.asarray(sweep.forcings)
                        reports = tuple(sweep.certification_reports)
                        for local_i, f_i in enumerate(stable_indices):
                            omega = f[f_i]
                            if not (
                                np.isclose(sweep_omegas[local_i], complex(omega))
                                and np.isfinite(sweep_gains[local_i])
                                and np.isfinite(sweep_residuals[local_i])
                            ):
                                LOGGER.error(
                                    "resolvent_map stable sweep point invalid index=%d beta=%.12g "
                                    "omega=%.12g action=direct-fallback",
                                    _point_index(beta_i, f_i, f.size),
                                    float(beta_value),
                                    float(omega),
                                )
                                continue
                            gain[beta_i, f_i] = float(sweep_gains[local_i])
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
                                engine="stable-sweep",
                                leading_gain=float(sweep_gains[local_i]),
                                iterations=int(sweep_iterations[local_i]),
                                residual=float(sweep_residuals[local_i]),
                                wall_s=wall_s / max(len(stable_indices), 1),
                                peak_rss_bytes=_peak_rss_bytes(),
                                reason_code=REASON_OK,
                            )
                            if response_modes.ndim >= 2 and response_modes.shape[1] > local_i:
                                record["response"] = response_modes[:, local_i].copy()
                            if forcing_modes.ndim >= 2 and forcing_modes.shape[1] > local_i:
                                record["forcing"] = forcing_modes[:, local_i].copy()
                            if len(reports) > local_i:
                                record["certification_report"] = reports[local_i]
                            record["sweep_metrics"] = dict(sweep.metrics)
                            record["anchor_omegas"] = np.asarray(anchors, dtype=complex).copy()
                            record["anchor_records"] = anchor_records
                            point_records.append(record)
                            sweep_completed.add(f_i)
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

            for f_i, omega in enumerate(f):
                if f_i in sweep_completed:
                    continue
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
                engine, gamma = _check_missing_engine_gate(
                    labels[f_i], cfg_map, emitted_fallback_notes
                )
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
                        gamma=gamma,
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
                # Finiteness gate for regime-b points
                if engine == "regime-b":
                    if not np.isfinite(point.gain):
                        raise ValueError(
                            f"regime-b (NEAR_POLE) point yielded non-finite gain {point.gain:.6e} "
                            f"at beta={beta_value:.6e}, omega={omega:.6e}; "
                            f"consider reducing regime_b_discount_gamma"
                        )
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
                if gamma > 0.0:
                    record["resolvent_discount_gamma"] = float(gamma)
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
