# SPDX-License-Identifier: GPL-3.0-only
"""Turnkey beta/frequency resolvent gain-map orchestration.

This entry point assembles one LNSE operator per beta.
It asks the band-marker for the intended frequency-band labels, then
walks each frequency line.
The first frequency on a line is a real factor. Later frequencies apply
``K(omega)^{-1}`` with that factor as a preconditioner. A residual gate
falls back to a new factor. The leading gain still comes from SLEPc.

Whether a gain map needs SBP sawtooth dissipation depends on the MESH.
The odd-even ("sawtooth") mode needs the two mesh directions to exchange
grid-scale content.
That happens only on a mesh that is curvilinear in BOTH directions, where
the metric cross-terms ``Ji12``/``Ji21`` are O(1).
On a rectilinear tensor mesh (``x = x(xi)``, ``y = y(eta)``, stretched or
not) the cross-terms vanish identically and the mode has nowhere to live.
There, dissipation measured 0.008-1.4% on the gain for +15% nnz and about
triple the wall time.
Measure the mesh with ``driver.preflight.metric_cross_coupling(X, Y)``.
Above ~1e-8, enable dissipation and tune ``epsilon`` at that mesh.
At or below it, record the coupling and run undamped on purpose, as the
shipped Savarino case declares (``enabled: false  # rectilinear mesh``).
On a mesh that mixes, the undamped operator carries grid-scale modes that
appear as spurious near-singular resonances in the map.
To suppress them, set ``numerics.operator_dissipation`` as for every other
path: ``enabled``, ``epsilon`` and ``apply_to_resolvent``. The block builds
the SBP dissipation operators ``Q_diss_x/Q_diss_y`` on the mesh, and
``config.defaults.stage_dissipation_epsilon`` turns it into the strength
subtracted from ``L0``. The old top-level ``dissipation_epsilon`` and
``eps_resolv`` keys are refused. A caller-supplied mesh without ``Q_diss``
and a nonzero strength is refused too, because the assembly would skip the
damping without a word.
``epsilon`` must clear the sawtooth poles yet stay below the
physical-mode scale, and a value tuned on one mesh does not transfer.
For the Song SWBLI fd4 stride-4 case on the curved-wall LES mesh the
tuned value was ``~0.10``.
``1.0`` over-damps the real OM-1/LU-S ridges by ~100x.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from linstabpy.config.defaults import REMOVED_DISSIPATION_KEYS, stage_dissipation_epsilon
from linstabpy.solvers.router.band_marking import NEAR_POLE, SMOOTH, STABLE, BandMarks
from linstabpy.solvers.router.map_record import (
    atomic_savez,
    differing_fields,
    digest,
    object_digests,
)
from linstabpy.solvers.router.map_walk import LineWalker, WalkDeferred
from linstabpy.utils._solver_cfg import as_bool as _as_bool
from linstabpy.utils.cost_metrics import peak_rss_bytes as _peak_rss_bytes
from linstabpy.utils.petsc_core import _is_petsc_matrix

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
        One-dimensional validated sweep coordinates.  ``f`` goes through
        to the existing resolvent frequency seams as ``omega``. The router
        performs no unit conversion.
    gain:
        Two-dimensional leading-gain array with shape ``(len(beta), len(f))``.
        Failed points are ``NaN`` and carry a nonzero ``reason_code``.
    regime_label:
        Band labels returned by ``mark_unstable_band`` with the same shape as
        ``gain``.  These are routing labels; every computed gain currently
        comes from the certified/direct regime-(a) engine.
    cost_metrics:
        CP3 cost sidecar dictionary from
        :func:`linstabpy.utils.cost_metrics.collect_cost_metrics`.
    failure_records:
        Structured records for point failures.  A failed point does not abort
        the sweep and is retryable on a later resumed run.
    reason_code:
        Integer per-point status, ``0`` for success, ``1`` for failed, ``-1``
        for not-yet-computed/pending.
    discount_gamma:
        Discount rate used for each point, same shape as ``gain``. ``0`` for
        an ordinary point. A NEAR_POLE point is solved with ``gamma > 0``
        (``regime_b_discount_gamma``, default 0.25), so its gain belongs to
        the discounted resolvent and is not the gain at the real frequency.
        A store written before this array existed gives NaN for its NEAR_POLE
        points, because the discount used is unknown.
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
    discount_gamma: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))


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
    source: str = "direct"


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


def _cfg_bool(cfg: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = _cfg_value(cfg, key, default)
    return _as_bool(value, name=key)


def _analysis_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(cfg.get("analysis"))


def _analysis_bool(cfg: Mapping[str, Any], key: str, default: bool = False) -> bool:
    analysis = _analysis_cfg(cfg)
    value = analysis.get(key, _cfg_value(cfg, key, default))
    return _as_bool(value, name=f"analysis.{key}")


def _normalise_cfg(
    cfg: Mapping[str, Any] | str | Path | object | None,
) -> tuple[Mapping[str, Any], Any]:
    """Return a plain mapping for router keys and a Config-compatible object.

    Hand ``Config`` objects to the resolver.
    Router-specific keys stay unavailable unless the caller supplied a
    mapping or YAML path.
    """
    if cfg is None:
        return {}, None
    if isinstance(cfg, Mapping):
        return cfg, None
    if isinstance(cfg, (str, Path)):
        from linstabpy.utils.yaml_io import load_config

        loaded = load_config(cfg)
        return loaded, None
    return {}, cfg


def _finite_1d_range(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one finite value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


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
            numerics_operator_dissipation=mesh_cfg.get(
                "numerics_operator_dissipation", _operator_dissipation(cfg) or None
            ),
        )

    raise ValueError(
        "resolvent_map requires a Mesh via cfg['mesh'], cfg['mesh_config'], or base_flow.mesh"
    )


def _solver_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    explicit = _cfg_value(cfg, "solver_cfg", None)
    if isinstance(explicit, Mapping):
        return explicit
    return cfg


def _operator_dissipation(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the enabled ``numerics.operator_dissipation`` block, refusing the old keys."""
    for key in REMOVED_DISSIPATION_KEYS:
        if _cfg_value(cfg, key, None) is not None:
            raise ValueError(
                f"{key!r} is no longer read. Set the strength in "
                "numerics.operator_dissipation (enabled, epsilon, apply_to_resolvent), "
                "the block every other path reads."
            )
    numerics = _as_mapping(_cfg_value(cfg, "numerics", None))
    od = numerics.get("operator_dissipation")
    return dict(od) if isinstance(od, Mapping) and od.get("enabled", False) else {}


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
        "matlab_parity_dmudt": _cfg_bool(cfg, "matlab_parity_dmudt", False),
        "dissipation_epsilon": stage_dissipation_epsilon(_operator_dissipation(cfg), "resolvent"),
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


def _select_region_points(
    mesh: Any, region_spec: Mapping[str, Any], *, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Resolve the x/y box of a region spec into active-point indices.

    Return ``(point_indices, X_act, Y_act, n_total)``.
    ``point_indices`` indexes into the active-point arrays ``X_act``/``Y_act``.
    The same convention applies to ``idx['{var}_j']`` blocks and base-flow
    arrays restricted to ``mesh.used_ind``.
    Shared by the diagonal primitive-variable mask
    (:func:`_build_region_mask`) and the conservative-momentum forcing matrix
    (:func:`_build_conservative_momentum_region`).
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
    return point_indices, X_act, Y_act, int(X_act.size)


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

    The mask follows the LNSE state layout exactly.
    Active physical points come from
    ``mesh.X/Y.ravel(order="F")[mesh.used_ind]``.
    Variable blocks come from ``idx['rho_j']``, ``idx['u_j']``,
    ``idx['v_j']``, ``idx['w_j']``, and ``idx['T_j']``.
    A region mask changes the resolvent question to a
    windowed/component-restricted gain.
    It is not an approximation to the global unmasked gain.
    """
    point_indices, X_act, Y_act, n_total = _select_region_points(mesh, region_spec, name=name)
    variables = _region_vars(region_spec, name=name)

    diag = np.zeros(int(ndof), dtype=float)
    selected_x = X_act[point_indices]
    selected_y = Y_act[point_indices]
    bbox = (
        float(np.min(selected_x)),
        float(np.max(selected_x)),
        float(np.min(selected_y)),
        float(np.max(selected_y)),
    )
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


_CONSERVATIVE_MOMENTUM_SPACE = "conservative_momentum"


def _build_conservative_momentum_region(
    mesh: Any,
    idx: Mapping[str, Any],
    ndof: int,
    region_spec: Mapping[str, Any],
    base_flow: Any,
    like_operator: Any,
    solver_cfg: Mapping[str, Any],
    *,
    name: str = "input_region",
    bc_indices: np.ndarray | None = None,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Build the whitened conservative-momentum forcing matrix for a region.

    The optimisation variable is the CONSERVATIVE momentum forcing
    ``f_c = (f_rou, f_rov, f_row)`` with a mass-weighted L2 norm.
    That replaces a primitive-variable diagonal mask.
    See ``linstabpy.resolvent.conservative_forcing`` for the physics.
    The whitening trick lets this reuse the existing square ``B_mat``
    resolvent machinery unchanged (``forcing_norm="l2"``).
    """
    from linstabpy.resolvent.conservative_forcing import (
        build_conservative_momentum_matrix,
        validate_conservative_momentum_vars,
    )

    if not isinstance(region_spec, Mapping):
        raise ValueError(f"{name} must be a mapping with optional x, y, and vars keys")
    if base_flow is None:
        raise ValueError(
            f"{name} space='{_CONSERVATIVE_MOMENTUM_SPACE}' requires the base flow "
            "(rho_bar, u_bar, v_bar, w_bar, kappa, Ma) but none was supplied"
        )

    # The whitening below is only correct when the gain machinery applies the
    # IDENTITY forcing norm to the coefficient vector. The config default is
    # "chu", which would silently re-weight the whitened coefficients and
    # produce a sigma that does not match the intended inner product, with no error --
    # this forcing space exists to remove. Fail loudly instead.
    from linstabpy.config import Config

    resolved_norm = Config.load(dict(solver_cfg) if solver_cfg else {}).resolvent.eps.forcing_norm
    if resolved_norm != "l2":
        raise ValueError(
            f"{name} space='{_CONSERVATIVE_MOMENTUM_SPACE}' requires forcing_norm='l2' "
            f"(the mass weighting is baked into the whitened B matrix), but the "
            f"config resolves forcing_norm={resolved_norm!r}. Set "
            "resolvent.eigen.forcing_norm to 'l2'."
        )

    momentum_vars = validate_conservative_momentum_vars(region_spec.get("vars"), name=name)
    point_indices, _X_act, _Y_act, _n_total = _select_region_points(mesh, region_spec, name=name)

    used_ind = np.asarray(mesh.used_ind)
    p = used_ind
    rho_bar_active = np.asarray(base_flow.RHO, dtype=float).ravel(order="F")[p]
    u_bar_active = np.asarray(base_flow.U, dtype=float).ravel(order="F")[p]
    v_bar_active = np.asarray(base_flow.V, dtype=float).ravel(order="F")[p]
    w_bar_active = np.asarray(base_flow.W, dtype=float).ravel(order="F")[p]
    w_int_active = np.asarray(mesh.W, dtype=float).ravel(order="F")[p]

    excluded_dofs = None
    if bc_indices is not None:
        bc = np.asarray(bc_indices, dtype=np.int64).reshape(-1)
        excluded_dofs = bc[(bc >= 0) & (bc < int(ndof))]

    B_tilde, momentum_dewhiten, info = build_conservative_momentum_matrix(
        idx,
        ndof,
        point_indices,
        rho_bar_active,
        u_bar_active,
        v_bar_active,
        w_bar_active,
        w_int_active,
        base_flow.kappa,
        base_flow.Ma,
        momentum_vars=momentum_vars,
        excluded_dofs=excluded_dofs,
    )
    LOGGER.info(
        "resolvent_map region_mask name=%s space=%s vars=%s n_region=%d n_momentum_dofs=%d "
        "forcing_norm=mass-weighted-L2 (whitened, run as forcing_norm='l2')",
        name,
        _CONSERVATIVE_MOMENTUM_SPACE,
        momentum_vars,
        info["n_region"],
        int(info["momentum_dofs"].size),
    )

    if _is_petsc_matrix(like_operator):
        from linstabpy.solvers.petsc_utils import scipy_to_petsc

        return scipy_to_petsc(B_tilde, solver_cfg), momentum_dewhiten, info
    return B_tilde, momentum_dewhiten, info


def _forcing_response_mats(
    L0: Any,
    ndof: int,
    bc_indices: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    mesh: Any | None = None,
    idx: Mapping[str, Any] | None = None,
    base_flow: Any | None = None,
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
    input_space = None
    if isinstance(input_region, Mapping):
        input_space = input_region.get("space")
    if input_space is not None and input_space != _CONSERVATIVE_MOMENTUM_SPACE:
        raise ValueError(
            f"input_region space={input_space!r} is not recognised; the only "
            f"supported value is '{_CONSERVATIVE_MOMENTUM_SPACE}' (omit the key "
            "for the primitive-variable diagonal mask)"
        )
    if isinstance(output_region, Mapping) and output_region.get("space") is not None:
        raise ValueError(
            "output_region does not support 'space': the response norm is the Chu "
            "energy on primitive variables; 'space' is a forcing-side "
            "(input_region) concept only"
        )
    if B_mat is None and input_region is not None and input_space == _CONSERVATIVE_MOMENTUM_SPACE:
        B_mat, _momentum_dewhiten, _info = _build_conservative_momentum_region(
            mesh,
            idx,
            ndof,
            input_region,
            base_flow,
            L0,
            solver_cfg,
            name="input_region",
            bc_indices=region_bc,
        )
    elif B_mat is None and input_region is not None:
        B_mat = _build_region_mask(
            mesh,
            idx,
            ndof,
            input_region,
            L0,
            solver_cfg,
            name="input_region",
            bc_indices=region_bc,
        )
    if C_mat is None and output_region is not None:
        C_mat = _build_region_mask(
            mesh,
            idx,
            ndof,
            output_region,
            L0,
            solver_cfg,
            name="output_region",
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

    # Read the sign from the same solver config as the resolvent solves.
    parity = _resolvent_sign_convention(cfg, None) == "matlab_parity"
    return mark_unstable_band(L0_beta, f_range, cfg, matlab_parity=parity)


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

    - Use cfg["regime_b_discount_gamma"] if present (explicit override).
    - Else: derive from pole proximity to keep the shifted solve well-posed.
      Default: 0.25 (validated in test_resolvent_discount.py to tame 1e5 blowup to ~4.0).
    """
    explicit = _cfg_value(cfg, "regime_b_discount_gamma", None)
    if explicit is not None:
        gamma = float(explicit)
        if not (gamma > 0.0):
            raise ValueError(f"regime_b_discount_gamma must be positive, got {gamma}")
        return gamma
    # Default: use 0.25 (validated to handle 1e5 poles gracefully)
    return 0.25


def _intended_missing_engine(label: object) -> tuple[str, str] | None:
    label_text = str(label)
    if label_text == NEAR_POLE:
        return "regime-b", "discounted_resolvent"
    if label_text == SMOOTH:
        return "regime-c", "rational_rom"
    return None


def _check_missing_engine_gate(
    label: object, cfg: Mapping[str, Any], emitted_notes: set[str]
) -> tuple[str, float]:
    """Determine which regime to use and any discount gamma.

    Return (regime, gamma).

    - regime is "regime-a", "regime-b", or "regime-c".
    - gamma is 0.0 for regimes a/c, or >0 for regime-b.
    """
    missing = _intended_missing_engine(label)
    if missing is None:
        return "regime-a", 0.0
    regime, engine = missing

    # For regime-b (NEAR_POLE), attempt discounted resolvent
    if regime == "regime-b":
        gamma = _regime_b_discount_gamma(cfg)
        return regime, gamma

    # For regime-c (SMOOTH), still raise NotImplementedError
    enable_key = "enable_regime_c"
    if _cfg_bool(cfg, enable_key, False):
        raise NotImplementedError(
            f"{regime} requested for label {label!r}, but {engine} is not implemented yet"
        )
    if regime not in emitted_notes:
        LOGGER.info(
            "resolvent_map fallback: intended_regime=%s missing_engine=%s "
            "enabled=false action=regime-a-direct",
            regime,
            engine,
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


class StoreSignConventionError(RuntimeError):
    """The resume store was written with another resolvent sign convention."""


def _resolvent_sign_convention(cfg_map: Mapping[str, Any], cfg_obj: Any) -> str:
    from linstabpy.config import Config

    cfg = cfg_obj if cfg_obj is not None else Config.load(_solver_cfg(cfg_map))
    return cfg.resolvent.sign_convention


class StoreFingerprintError(RuntimeError):
    """The resume store was written for another problem than this run."""


# Keys that name files or set logging. They do not change a gain.
_FINGERPRINT_SKIP = frozenset(
    {"output_npz", "output_store", "cache_npz", "logging", "stage_heartbeat_seconds", "mesh"}
)


def _fingerprint(
    cfg_map: Mapping[str, Any],
    cfg_obj: Any,
    mesh: Any,
    base_flow: Any,
    W_diag: np.ndarray,
) -> dict[str, str]:
    """Return one digest per input that changes a gain.

    The inputs are every config key, the typed config object, the mesh arrays
    and operators, the base-flow arrays and the forcing norm ``W_diag``.
    A resumed run compares these with the ones stored with the map.
    """
    fields = {f"cfg.{k}": digest(v) for k, v in cfg_map.items() if k not in _FINGERPRINT_SKIP}
    if cfg_obj is not None:
        fields["cfg_object"] = digest(cfg_obj)
    fields.update(object_digests("mesh", mesh))
    fields.update(object_digests("base_flow", base_flow))
    fields["W_diag"] = digest(W_diag)
    return fields


def _load_store(
    path: Path | None,
    beta: np.ndarray,
    f: np.ndarray,
    sign_convention: str,
    fingerprint: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    gain = np.full((beta.size, f.size), np.nan, dtype=float)
    discount = np.zeros((beta.size, f.size), dtype=float)
    labels = np.full((beta.size, f.size), "", dtype="U32")
    reason = np.full((beta.size, f.size), REASON_PENDING, dtype=np.int16)
    failures: list[dict[str, object]] = []
    if path is None or not path.exists():
        return gain, labels, reason, discount, failures
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
            # A store written before this key existed used the LinStab2D sign.
            stored = (
                str(data["resolvent_sign_convention"])
                if "resolvent_sign_convention" in data.files
                else "matlab_parity"
            )
            if stored != sign_convention:
                raise StoreSignConventionError(
                    f"resolvent_map: cannot resume {path}. It was written with "
                    f"resolvent sign convention {stored!r}, this run uses {sign_convention!r}. "
                    "One map never mixes conventions. To finish the old map, set "
                    f"solver.resolvent_sign_convention: {stored}. To use the current "
                    "convention, write to a new output_npz path. To convert an old map "
                    "without recomputing, see docs/resolvent_sign_convention.md."
                )
            if "fingerprint_json" in data.files:
                stored_fp = json.loads(str(data["fingerprint_json"]))
                changed = differing_fields(stored_fp, fingerprint)
                if changed:
                    raise StoreFingerprintError(
                        f"resolvent_map: cannot resume {path}. It was written for another "
                        f"problem. These inputs differ: {', '.join(changed)}. Write to a new "
                        "output_npz path, or restore the inputs of the stored map."
                    )
            else:
                LOGGER.warning(
                    "resolvent_map: resuming %s, which has no input fingerprint. Whether it "
                    "matches this problem cannot be verified.",
                    path,
                )
            gain = np.asarray(data["gain"], dtype=float).reshape(beta.size, f.size)
            labels = np.asarray(data["regime_label"], dtype=str).reshape(beta.size, f.size)
            if "discount_gamma" in data.files:
                discount = np.asarray(data["discount_gamma"], dtype=float)
                discount = discount.reshape(beta.size, f.size)
            else:
                discount = np.where(labels == NEAR_POLE, np.nan, 0.0)
            if "reason_code" in data.files:
                reason = np.asarray(data["reason_code"], dtype=np.int16).reshape(beta.size, f.size)
            else:
                reason = np.where(np.isfinite(gain), REASON_OK, REASON_PENDING).astype(np.int16)
            if "failure_records_json" in data.files:
                failures = _records_from_json_array(data["failure_records_json"])
    except (OSError, KeyError, ValueError) as exc:
        LOGGER.info("resolvent_map cache ignored path=%s reason=%s", path, exc)
    return gain, labels, reason, discount, failures


def _save_store(
    path: Path | None,
    *,
    beta: np.ndarray,
    f: np.ndarray,
    gain: np.ndarray,
    regime_label: np.ndarray,
    reason_code: np.ndarray,
    discount_gamma: np.ndarray,
    failure_records: list[dict[str, object]],
    point_records: list[dict[str, object]],
    sign_convention: str,
    fingerprint: Mapping[str, str],
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
        "discount_gamma": discount_gamma,
        "failure_records_json": _json_records(failure_records),
        "point_records_json": _json_records(point_records),
        "resolvent_sign_convention": np.asarray(sign_convention, dtype=str),
        "fingerprint_json": np.asarray(json.dumps(dict(fingerprint), sort_keys=True), dtype=str),
    }
    if cost_metrics:
        payload.update(cost_metrics)
    atomic_savez(path, **payload)


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
        Mapping, YAML path, typed ``Config`` object, or ``None``.  Router keys
        may sit at top level or under ``router`` / ``solver.router``.

    Returns
    -------
    RouterResult
        Gain map, intended band labels, CP3 cost metrics, and point records.

    Notes
    -----
    With ``output_npz`` set, the run resumes from that file. The file stores a
    fingerprint of every input that changes a gain: config keys, mesh, base
    flow and forcing norm. A resume with other inputs raises
    ``StoreFingerprintError`` that names the differing fields, and leaves the
    file as it is. A file without a fingerprint resumes with a warning.
    """
    cfg_map, cfg_obj = _normalise_cfg(cfg)
    beta = _finite_1d_range(beta_range, "beta_range")
    f = _finite_1d_range(f_range, "f_range")
    mesh = _resolve_mesh(base_flow, cfg_map)
    W_diag = _compute_w_diag(mesh, base_flow, cfg_map)
    store = _store_path(cfg_map)

    sign_convention = _resolvent_sign_convention(cfg_map, cfg_obj)
    fingerprint = _fingerprint(cfg_map, cfg_obj, mesh, base_flow, W_diag)
    gain, regime_label, reason_code, discount_gamma, failure_records = _load_store(
        store, beta, f, sign_convention, fingerprint
    )
    point_records: list[dict[str, object]] = []
    stage_wall: dict[str, float] = {}
    total_points = int(beta.size * f.size)
    n_modes = _n_modes(cfg_map)
    emitted_fallback_notes: set[str] = set()
    use_amortized_stable_sweep = _analysis_bool(cfg_map, "amortized_stable_sweep", False)
    if use_amortized_stable_sweep:
        LOGGER.warning(
            "amortized_stable_sweep is the 1-D frequency-line tool, not the map; "
            "resolvent_map walks each beta line with SLEPc"
        )

    walker = None
    stop_line = False
    with _heartbeat("resolvent_map", cfg_map):
        for beta_i, beta_value in enumerate(beta):
            if stop_line:
                break
            if walker is not None:
                walker.close()
                walker = None
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
                # fact near a pole then goes to the certified resolvent
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
                base_flow=base_flow,
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
            if int(f.size) > 1:
                from linstabpy.config import Config

                solver_cfg = dict(_solver_cfg(cfg_map))
                walk_cfg = cfg_obj if cfg_obj is not None else Config.load(solver_cfg)
                walker = LineWalker(solver_cfg=solver_cfg, cfg_obj=walk_cfg)

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
                    use_walk = walker is not None and engine == "regime-a" and float(gamma) == 0.0
                    if use_walk:
                        walked = walker.compute(
                            assembly.L0,
                            float(omega),
                            W_diag,
                            n_modes,
                            B_mat=B_mat,
                            C_mat=C_mat,
                            filter_fn=filter_fn,
                            filter_ct_fn=filter_ct_fn,
                            gamma=0.0,
                        )
                        point = _DirectPointResult(
                            gain=walked.gain,
                            iterations=walked.iterations,
                            residual=walked.residual,
                            source=walked.source,
                        )
                    else:
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
                except WalkDeferred:
                    LOGGER.info(
                        "resolvent_map stop: remaining wall is shorter than a factor "
                        "beta=%.12g omega=%.12g",
                        float(beta_value),
                        float(omega),
                    )
                    stop_line = True
                    break
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
                        discount_gamma=discount_gamma,
                        failure_records=failure_records,
                        point_records=point_records,
                        sign_convention=sign_convention,
                        fingerprint=fingerprint,
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
                discount_gamma[beta_i, f_i] = gamma
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
                if point.source:
                    record["walk_source"] = point.source
                point_records.append(record)
                LOGGER.info(
                    "resolvent_map point complete index=%d total=%d beta=%.12g omega=%.12g "
                    "leading_gain=%.12g discount_gamma=%.6g iters=%s residual=%s walk_source=%s "
                    "wall_s=%.6g peak_rss_bytes=%d",
                    _point_index(beta_i, f_i, f.size),
                    total_points,
                    float(beta_value),
                    float(omega),
                    point.gain,
                    float(gamma),
                    point.iterations,
                    None if point.residual is None else f"{point.residual:.6e}",
                    point.source,
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
                    discount_gamma=discount_gamma,
                    failure_records=failure_records,
                    point_records=point_records,
                    sign_convention=sign_convention,
                    fingerprint=fingerprint,
                )

        if walker is not None:
            walker.close()
            walker = None

    cost_metrics = _collect_cost_metrics(stage_wall)
    _save_store(
        store,
        beta=beta,
        f=f,
        gain=gain,
        regime_label=regime_label,
        reason_code=reason_code,
        discount_gamma=discount_gamma,
        failure_records=failure_records,
        point_records=point_records,
        sign_convention=sign_convention,
        fingerprint=fingerprint,
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
        discount_gamma=discount_gamma,
    )


__all__ = ["RouterResult", "resolvent_map"]
