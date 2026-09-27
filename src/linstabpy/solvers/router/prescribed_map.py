# SPDX-License-Identifier: GPL-3.0-only
"""Map driver for prescribed-forcing resolvent: known f_hat, single solve per point.

Reads a table of (beta, omega, npz_path) rows from the config. Loads each
prescribed forcing vector and solves the linear resolvent once per point.
Reuses assembly and BC masking from the classical resolvent path.
No SVD, no band-marking, no LineWalker.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from linstabpy.config.defaults import stage_dissipation_epsilon
from linstabpy.resolvent.prescribed import prescribed_forcing_response
from linstabpy.resolvent.prescribed_io import load_prescribed_forcing
from linstabpy.solvers.router.map_record import append_row
from linstabpy.utils.cost_metrics import peak_rss_bytes as _peak_rss_bytes

LOGGER = logging.getLogger(__name__)

# CSV record fields and provenance columns.
_RECORD_FIELDS = (
    "row",
    "beta",
    "omega",
    "gain",
    "rho_fwd",
    "gamma",
    "forcing_norm",
    "status",
    "wall_s",
    "peak_rss_bytes",
)
_PROVENANCE_FIELDS = ("git_sha", "backend", "ranks", "fd_order", "epsilon")


@dataclass(frozen=True)
class PrescribedMapResult:
    """Minimal result from prescribed_map for API compatibility."""

    point_records: list[dict[str, object]]
    cost_metrics: dict[str, object]


def prescribed_map(
    base_flow: object,
    cfg: object = None,
) -> PrescribedMapResult:
    """Compute prescribed-forcing resolvent gains from a table of forcing files.

    Parameters
    ----------
    base_flow
        Base-flow object consumed by the existing LNSE assembly seam. The mesh
        must be supplied as ``cfg['mesh']``, ``cfg['mesh_config']``, or
        ``base_flow.mesh``.
    cfg
        Mapping, YAML path, typed ``Config`` object, or ``None``.
        Router keys may sit at top level or under ``router`` / ``solver.router``.
        Must include a ``prescribed_forcing`` key pointing to a table of
        (beta, omega, forcing_path) row dicts.

    Returns
    -------
    PrescribedMapResult
        Point records (one row per solved point) and cost metrics.

    Raises
    ------
    ConfigValidationError
        If both ``prescribed_forcing`` and classical SVD/band-marking options
        are set (they answer different questions and cannot coexist).
    ValueError
        If the ``prescribed_forcing`` table is empty or a row has no
        ``forcing_path`` (or ``path``) key.
    """
    from linstabpy.driver.preflight import config_advisories, format_advisories
    from linstabpy.solvers.router.resolvent_map import (
        _analysis_bool,
        _assemble_beta_operator,
        _cfg_bool,
        _collect_cost_metrics,
        _compute_w_diag,
        _forcing_response_mats,
        _normalise_cfg,
        _resolve_mesh,
        _solver_cfg,
    )

    cfg_map, cfg_obj = _normalise_cfg(cfg)

    # Validate config exclusivity (prescribed vs SVD).
    _validate_prescribed_exclusivity(cfg_map)

    # Load prescribed-forcing table.
    forcing_table = cfg_map.get("prescribed_forcing", [])
    if not forcing_table:
        message = "prescribed_forcing config is empty or missing"
        raise ValueError(message)

    mesh = _resolve_mesh(base_flow, cfg_map)
    W_diag = _compute_w_diag(mesh, base_flow, cfg_map)

    solver_cfg = dict(_solver_cfg(cfg_map))
    gamma = float(cfg_map.get("resolvent_discount_gamma", 0.0))
    config_gamma = (
        float(cfg_map["resolvent_discount_gamma"])
        if "resolvent_discount_gamma" in cfg_map
        else None
    )
    forcing_norm = cfg_map.get("forcing_norm", "chu")
    allow_zero_bin = _analysis_bool(cfg_map, "prescribed_allow_zero_bin", default=False)
    assembled_betas: dict[float, dict[str, object]] = {}
    point_records: list[dict[str, object]] = []
    stage_wall: dict[str, float] = {}

    advisories = config_advisories(cfg_map)
    if advisories:
        LOGGER.warning("%s", format_advisories(advisories, title="Config advisories"))

    # Extract filter functions from mesh.
    use_mesh_filters = _cfg_bool(cfg_map, "use_mesh_filters", default=True)
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

    for row_index, row in enumerate(forcing_table):
        beta = float(row.get("beta"))
        omega = float(row.get("omega"))
        # Accept both "forcing_path" and "path" keys; raise on both missing.
        if "forcing_path" in row:
            forcing_path = str(row["forcing_path"])
        elif "path" in row:
            forcing_path = str(row["path"])
        else:
            message = f"prescribed_forcing row {row_index}: missing 'forcing_path' (or 'path') key"
            raise ValueError(message)

        if beta not in assembled_betas:
            LOGGER.info("prescribed_map assembling operator row=%d beta=%.12g", row_index, beta)
            assembly = _assemble_beta_operator(mesh, base_flow, beta, cfg_map)
            stage_wall["assemble_operator_bc"] = (
                stage_wall.get("assemble_operator_bc", 0.0) + assembly.wall_s
            )
            assembled_betas[beta] = {
                "L0": assembly.L0,
                "bc_indices": assembly.bc_indices,
            }

        assembly_data = assembled_betas[beta]
        L0 = assembly_data["L0"]
        bc_indices = assembly_data["bc_indices"]
        ndof = int(L0.shape[0]) if hasattr(L0, "shape") else L0.getSize()[0]

        try:
            forcing = load_prescribed_forcing(
                forcing_path,
                expected_ndof=ndof,
                config_gamma=config_gamma,
                allow_zero_bin=allow_zero_bin,
            )
            _require_row_matches_forcing(
                row_index=row_index,
                forcing_path=forcing_path,
                beta_row=beta,
                omega_row=omega,
                beta_npz=forcing.beta,
                omega_npz=forcing.omega,
            )
        except ValueError as e:
            record = _prescribed_point_record(
                row_index=row_index,
                beta=beta,
                omega=omega,
                status="failed",
                wall_s=0.0,
                peak_rss_bytes=_peak_rss_bytes(),
                exception=e,
            )
            point_records.append(record)
            LOGGER.exception(
                "prescribed_map load failed row=%d beta=%.12g omega=%.12g",
                row_index,
                beta,
                omega,
            )
            continue

        # Use the loaded gamma if present, else the config gamma.
        solve_gamma = forcing.gamma if forcing.gamma is not None else gamma

        # Assemble BC masks and filters reusing the classical path.
        B_mat, C_mat = _forcing_response_mats(
            L0, ndof, bc_indices, cfg_map, mesh=mesh, idx=None, base_flow=base_flow
        )

        try:
            t0 = time.perf_counter()
            _q_hat, gain, info = prescribed_forcing_response(
                L0,
                omega=omega,
                W_diag=W_diag,
                f_hat=forcing.f,
                gamma=solve_gamma,
                B_mat=B_mat,
                C_mat=C_mat,
                filter_fn=filter_fn,
                filter_ct_fn=filter_ct_fn,
                forcing_norm=forcing_norm,
                solver_cfg=solver_cfg,
                cfg=cfg_obj,
            )
            wall_s = time.perf_counter() - t0
            status = "ok"
            exception = None
        except Exception as e:
            wall_s = time.perf_counter() - t0
            status = "failed"
            exception = e
            gain = None
            info = {}
            LOGGER.exception(
                "prescribed_map solve failed row=%d beta=%.12g omega=%.12g",
                row_index,
                beta,
                omega,
            )

        record = _prescribed_point_record(
            row_index=row_index,
            beta=beta,
            omega=omega,
            gain=gain,
            rho_fwd=info.get("rho_fwd"),
            gamma=info.get("gamma", solve_gamma),
            forcing_norm=forcing_norm,
            status=status,
            wall_s=wall_s,
            peak_rss_bytes=_peak_rss_bytes(),
            exception=exception,
        )
        stage_wall["prescribed_solve"] = stage_wall.get("prescribed_solve", 0.0) + wall_s
        point_records.append(record)
        LOGGER.info(
            "point %s row=%d beta=%.12g omega=%.12g gain=%s rho_fwd=%s wall_s=%.6g",
            status,
            row_index,
            beta,
            omega,
            gain if gain is not None else "none",
            info.get("rho_fwd"),
            wall_s,
        )

    # Write the CSV file with provenance columns.
    csv_path_raw = cfg_map.get("prescribed_output_csv")
    if csv_path_raw:
        csv_path = Path(csv_path_raw).resolve()
        _write_prescribed_csv(csv_path, point_records, cfg_map)

    cost_metrics = _collect_cost_metrics(stage_wall)
    cost_metrics["prescribed_map_rows"] = len(point_records)
    cost_metrics["prescribed_map_unique_betas"] = len(assembled_betas)

    LOGGER.info(
        "prescribed_map complete rows=%d unique_betas=%d",
        len(point_records),
        len(assembled_betas),
    )
    return PrescribedMapResult(point_records=point_records, cost_metrics=cost_metrics)


def _validate_prescribed_exclusivity(cfg_map: dict[str, object]) -> None:
    """Raise if both prescribed_forcing and classical SVD options are set.

    Raises
    ------
    ConfigValidationError
        If the config contains incompatible option pairs.
    """
    from linstabpy.config.schema import ConfigValidationError

    is_prescribed = "prescribed_forcing" in cfg_map and cfg_map["prescribed_forcing"]
    if not is_prescribed:
        return

    # Check for classical resolvent options that would be silent no-ops or confusing.
    # They may sit at top level or under ``analysis``, which is where the
    # classical path reads run_resolvent from (_analysis_bool).
    analysis = cfg_map.get("analysis")
    analysis = analysis if isinstance(analysis, dict) else {}
    classical_markers = [
        ("beta_range", "classical beta_range"),
        ("f_range", "classical f_range"),
        ("run_resolvent", "classical run_resolvent"),
    ]
    for key, label in classical_markers:
        if key in cfg_map or key in analysis:
            message = (
                f"Config has both prescribed_forcing and {label}. "
                "These answer different questions (known f_hat vs SVD optimal forcing). "
                "Pick one: remove prescribed_forcing to run the classical SVD map, "
                f"or remove {key} to run the prescribed path."
            )
            raise ConfigValidationError(message)


def _require_row_matches_forcing(
    *,
    row_index: int,
    forcing_path: str,
    beta_row: float,
    omega_row: float,
    beta_npz: float,
    omega_npz: float,
) -> None:
    """Raise when the forcing npz (beta, omega) differ from the table row.

    The npz carries its own (beta, omega). A difference means the row
    names the file of another point. The solve would then fill this map
    cell from that other point's forcing.

    Raises
    ------
    ValueError
        If either value differs by more than 1e-9 relative.
    """
    for name, row_value, npz_value in (
        ("beta", beta_row, beta_npz),
        ("omega", omega_row, omega_npz),
    ):
        if abs(npz_value - row_value) > 1.0e-9 * max(1.0, abs(row_value)):
            message = (
                f"prescribed_forcing row {row_index}: table {name}={row_value:.12g} "
                f"but {forcing_path} carries {name}={npz_value:.12g}"
            )
            raise ValueError(message)


def _prescribed_point_record(
    *,
    row_index: int,
    beta: float,
    omega: float,
    gain: float | None = None,
    rho_fwd: float | None = None,
    gamma: float = 0.0,
    forcing_norm: str = "chu",
    status: str = "ok",
    wall_s: float = 0.0,
    peak_rss_bytes: int = 0,
    exception: BaseException | None = None,
) -> dict[str, object]:
    """Assemble one prescribed-forcing point record for CSV output.

    Parameters
    ----------
    row_index : int
        Row number in the input table (for user reference).
    beta : float
        Spanwise wavenumber.
    omega : float
        Frequency.
    gain : float, optional
        Computed gain, or None if solve failed.
    rho_fwd : float, optional
        Inner-solve residual norm.
    gamma : float, optional
        Discount used in solve.
    forcing_norm : str, optional
        Norm used for forcing ("chu" or "l2").
    status : str, optional
        "ok" or "failed".
    wall_s : float, optional
        Wall-clock time for the solve.
    peak_rss_bytes : int, optional
        Peak RSS memory during solve.
    exception : BaseException, optional
        Exception if solve failed.

    Returns
    -------
    dict
        Record suitable for CSV row or JSON storage.
    """
    record: dict[str, object] = {
        "row": int(row_index),
        "beta": float(beta),
        "omega": float(omega),
        "gain": None if gain is None else float(gain),
        "rho_fwd": None if rho_fwd is None else float(rho_fwd),
        "gamma": float(gamma),
        "forcing_norm": str(forcing_norm),
        "status": str(status),
        "wall_s": float(wall_s),
        "peak_rss_bytes": int(peak_rss_bytes),
    }
    if exception is not None:
        record["exception_class"] = exception.__class__.__name__
        record["exception"] = str(exception)
    return record


def _write_prescribed_csv(
    csv_path: Path,
    point_records: list[dict[str, object]],
    cfg_map: dict[str, object],
) -> None:
    """Write prescribed-forcing results to a CSV file with provenance."""
    from linstabpy.utils.provenance import _git_head_sha

    if not point_records:
        LOGGER.info("No point records to write to CSV")
        return

    # Build the full header from record fields and provenance fields.
    fields = list(_RECORD_FIELDS) + list(_PROVENANCE_FIELDS)

    # Provenance data to append to every row.
    git_sha = _git_head_sha()
    backend = "unknown"
    solver_cfg = cfg_map.get("solver", {})
    if isinstance(solver_cfg, dict):
        backend_cfg = solver_cfg.get("linear_solver_backend", "unknown")
        backend = (
            str(backend_cfg.get("name", "unknown"))
            if isinstance(backend_cfg, dict)
            else str(backend_cfg)
        )
    ranks = 1  # Prescribed path is serial today; column exists for schema consistency.
    fd_order = int(cfg_map.get("fd_order", 4))
    numerics = cfg_map.get("numerics", {})
    od = numerics.get("operator_dissipation") if isinstance(numerics, dict) else None
    epsilon = stage_dissipation_epsilon(od, "resolvent")

    for record in point_records:
        row_values = [record.get(f) for f in _RECORD_FIELDS]
        row_values.extend([git_sha, backend, ranks, fd_order, epsilon])
        append_row(csv_path, fields, row_values)
