#!/usr/bin/env python3
"""Capture compact Phase 0 characterization artifacts before seam refactors."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "architecture_refactor"
    / "architecture_refactor_phase0"
    / "phase0_characterization.json"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _stage(fn: Callable[[], Any]) -> tuple[Any, dict[str, float | int]]:
    rss0 = _peak_rss_bytes()
    t0 = time.perf_counter()
    value = fn()
    t1 = time.perf_counter()
    rss1 = _peak_rss_bytes()
    return value, {
        "seconds": float(t1 - t0),
        "rss_peak_before_bytes": int(rss0),
        "rss_peak_after_bytes": int(rss1),
        "rss_peak_delta_bytes": int(max(0, rss1 - rss0)),
    }


def _array_digest(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


def _index_payload(indices: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(indices, dtype=np.int64)
    return {
        "count": int(arr.size),
        "sha256": _array_digest(arr),
        "indices": arr.astype(int).tolist(),
    }


def _number_payload(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": _array_digest(arr),
        "nbytes": int(arr.nbytes),
        "finite": bool(np.all(np.isfinite(arr))),
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "l2_norm": float(np.linalg.norm(arr.ravel())) if arr.size else 0.0,
        "nnz": int(np.count_nonzero(arr)),
    }


def _complex_vector_payload(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.complex128)
    return {
        "count": int(arr.size),
        "sha256": _array_digest(arr),
        "real_min": float(np.min(arr.real)) if arr.size else 0.0,
        "real_max": float(np.max(arr.real)) if arr.size else 0.0,
        "imag_min": float(np.min(arr.imag)) if arr.size else 0.0,
        "imag_max": float(np.max(arr.imag)) if arr.size else 0.0,
        "first_values": [[float(z.real), float(z.imag)] for z in arr[:10]],
    }


def _float_vector_payload(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "count": int(arr.size),
        "sha256": _array_digest(arr),
        "finite_count": int(finite.size),
        "finite_max": float(np.max(finite)) if finite.size else None,
        "finite_median": float(np.median(finite)) if finite.size else None,
        "first_values": [float(x) for x in arr[:10]],
    }


def _row_payload(rows: list[tuple[int, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    h = hashlib.sha256()
    max_abs = 0.0
    nnz = 0
    for row_id, cols, vals in rows:
        cols_arr = np.asarray(cols, dtype=np.int64)
        vals_arr = np.asarray(vals, dtype=np.complex128)
        h.update(np.asarray([row_id], dtype=np.int64).view(np.uint8))
        h.update(cols_arr.view(np.uint8))
        h.update(vals_arr.view(np.uint8))
        nnz += int(cols_arr.size)
        if vals_arr.size:
            max_abs = max(max_abs, float(np.max(np.abs(vals_arr))))
    return {
        "row_count": int(len(rows)),
        "nnz": nnz,
        "sha256": h.hexdigest(),
        "max_abs": max_abs,
    }


def _recipe_hash(recipe: list[dict[str, Any]]) -> str:
    text = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _uniform_baseflow(mesh, *, re: float = 1.0e4, ma: float = 0.3):
    from linstabpy.baseflow import BaseFlow, sutherland_air

    ny, nx = mesh.X.shape
    ones = np.ones((ny, nx))
    zeros = np.zeros((ny, nx))
    baseflow = BaseFlow(
        Re=re,
        Ma=ma,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
        RHO=ones.copy(),
        U=ones.copy(),
        V=zeros.copy(),
        W=zeros.copy(),
        T=ones.copy(),
    )
    sutherland_air(baseflow)
    return baseflow


def _mesh(label: str, *, model: str):
    from linstabpy.mesh import Mesh

    if model == "axy":
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.05, 1.05),
            nx=18,
            ny=14,
            fd_order=4,
            y_symmetry=True,
        )
    elif label == "sheared":
        base = Mesh.create((0.0, 1.0), (0.0, 1.0), 20, 20, fd_order=2)
        xi = base.X.copy()
        eta = base.Y.copy()
        mesh = base.deform(X_new=xi + 0.3 * eta, Y_new=eta.copy())
    else:
        mesh = Mesh.create(xrange=(0.0, 2.0), yrange=(0.0, 1.0), nx=24, ny=16, fd_order=4)
    return mesh


def _bc_characterization() -> list[dict[str, Any]]:
    from linstabpy.boundary_conditions import analyze_bc_recipe
    from linstabpy.boundary_conditions.neumann import (
        _build_neumann_rows,
        _collect_neumann_indices,
    )
    from linstabpy.operators import build_lnse_operator

    cases = [
        (
            "case1_overlap",
            "cartesian",
            "2D",
            [
                {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
                {"type": "dirichlet", "borders": "b", "variables": "ruvw"},
                {"type": "neumann", "borders": "b", "variables": "T"},
                {"type": "neumann", "borders": "r", "variables": "ruvwT"},
            ],
        ),
        (
            "case1_exclude_corners",
            "cartesian",
            "2D",
            [
                {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
                {"type": "dirichlet", "borders": "b", "variables": "ruvw"},
                {"type": "neumann", "borders": "b", "variables": "T", "exclude_corners": True},
                {
                    "type": "neumann",
                    "borders": "r",
                    "variables": "ruvwT",
                    "exclude_corners": True,
                },
            ],
        ),
        (
            "sheared_curvilinear_neumann",
            "sheared",
            "2D",
            [
                {"type": "dirichlet", "borders": "b", "variables": "uvw"},
                {
                    "type": "neumann",
                    "borders": "r",
                    "variables": "uT",
                    "curvilinear_normals": True,
                },
            ],
        ),
        (
            "same_border_later_neumann_owns_overlap",
            "cartesian",
            "2D",
            [
                {"type": "dirichlet", "borders": "b", "variables": "T"},
                {"type": "neumann", "borders": "b", "variables": "T"},
            ],
        ),
        (
            "pure_neumann_lr_exclude_corners",
            "cartesian",
            "2D",
            [
                {"type": "neumann", "borders": "lr", "variables": "u", "exclude_corners": True},
            ],
        ),
        (
            "axisymmetric_dirichlet_ltr",
            "axisymmetric",
            "axy",
            [
                {"type": "dirichlet", "borders": "ltr", "variables": "uvwT"},
            ],
        ),
    ]
    reports: list[dict[str, Any]] = []
    for label, mesh_label, model, recipe in cases:
        mesh = _mesh(mesh_label, model=model)
        baseflow = _uniform_baseflow(mesh)
        _, idx = build_lnse_operator(
            mesh,
            baseflow,
            kx=0.0,
            model=model,
            m=1.0 if model == "axy" else 0.0,
        )
        analysis = analyze_bc_recipe(mesh, idx, recipe)
        payload = {
            "label": label,
            "mesh": mesh_label,
            "model": model,
            "ngp": int(mesh.ngp),
            "recipe_sha256": _recipe_hash(recipe),
            "recipe": recipe,
            "sets": {
                name: _index_payload(np.asarray(analysis[name], dtype=np.int64))
                for name in (
                    "dirichlet_indices",
                    "neumann_indices",
                    "effective_dirichlet_indices",
                    "effective_neumann_indices",
                    "column_zeroed_dirichlet_indices",
                    "overlap_indices",
                    "total_bc_indices",
                )
            },
            "rule_reports": [],
        }
        if label == "sheared_curvilinear_neumann":
            index_set, index_mesh, border_tags = _collect_neumann_indices(
                idx,
                "r",
                "uT",
                mesh.ngp,
                exclude_corners=False,
            )
            axis_rows = _build_neumann_rows(
                mesh,
                index_set,
                index_mesh,
                border_tags,
                mesh.ngp,
                penalty=1.0,
                curvilinear_normals=False,
            )
            curv_rows = _build_neumann_rows(
                mesh,
                index_set,
                index_mesh,
                border_tags,
                mesh.ngp,
                penalty=1.0,
                curvilinear_normals=True,
            )
            axis_payload = _row_payload(axis_rows)
            curv_payload = _row_payload(curv_rows)
            payload["curvilinear_normal_rows"] = {
                "axis_aligned": axis_payload,
                "curvilinear": curv_payload,
                "same_fingerprint": axis_payload["sha256"] == curv_payload["sha256"],
            }
        for rule in analysis["rule_reports"]:
            payload["rule_reports"].append(
                {
                    "rule_id": int(rule["rule_id"]),
                    "type": str(rule["type"]),
                    "borders": str(rule["borders"]),
                    "variables": str(rule["variables"]),
                    "exclude_corners": bool(rule["exclude_corners"]),
                    "count": int(rule["count"]),
                    "indices": _index_payload(np.asarray(rule["indices"], dtype=np.int64)),
                    "overlap_previous": _index_payload(
                        np.asarray(rule["overlap_previous"], dtype=np.int64)
                    ),
                    "cross_type_overlap": _index_payload(
                        np.asarray(rule["cross_type_overlap"], dtype=np.int64)
                    ),
                }
            )
        reports.append(payload)
    return reports


def _coefficient_characterization() -> list[dict[str, Any]]:
    from linstabpy.operators.assembly import _assemble_from_prepared, _prepare_coefficients

    reports: list[dict[str, Any]] = []
    for model, mesh_label, kwargs in (
        ("2D", "cartesian", {"kx": 0.2, "m": 0.0}),
        ("axy", "axisymmetric", {"kx": 0.0, "m": 1.0}),
    ):
        mesh = _mesh(mesh_label, model=model)
        baseflow = _uniform_baseflow(mesh)
        prep, prepare_stage = _stage(
            lambda mesh=mesh, baseflow=baseflow, model=model, kwargs=kwargs: _prepare_coefficients(
                mesh,
                baseflow,
                kx=float(kwargs["kx"]),
                model=model,
                m=float(kwargs["m"]),
            )
        )
        operator, assemble_stage = _stage(lambda prep=prep: _assemble_from_prepared(prep))
        coeffs = prep["coeffs"]
        coeff_names = ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B")
        reports.append(
            {
                "label": f"{model}_uniform",
                "model": model,
                "mesh": mesh_label,
                "ngp": int(mesh.ngp),
                "ndof": int(operator.shape[0]),
                "operator_nnz": int(operator.nnz),
                "parameters": {"kx": float(kwargs["kx"]), "m": float(kwargs["m"])},
                "stages": {
                    "prepare_coefficients": prepare_stage,
                    "assemble_from_prepared": assemble_stage,
                },
                "coefficients": {
                    name: _number_payload(np.asarray(getattr(coeffs, name))) for name in coeff_names
                },
            }
        )
    return reports


def _coefficient_snapshot_arrays() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from linstabpy.operators.assembly import _prepare_coefficients

    coeff_names = ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B")
    metadata: dict[str, Any] = {"cases": []}
    arrays: dict[str, np.ndarray] = {}
    for model, mesh_label, kwargs in (
        ("2D", "cartesian", {"kx": 0.2, "m": 0.0}),
        ("axy", "axisymmetric", {"kx": 0.0, "m": 1.0}),
    ):
        mesh = _mesh(mesh_label, model=model)
        baseflow = _uniform_baseflow(mesh)
        prep = _prepare_coefficients(
            mesh,
            baseflow,
            kx=float(kwargs["kx"]),
            model=model,
            m=float(kwargs["m"]),
        )
        coeffs = prep["coeffs"]
        label = f"{model}_uniform"
        case_meta = {
            "label": label,
            "model": model,
            "mesh": mesh_label,
            "ngp": int(mesh.ngp),
            "parameters": {"kx": float(kwargs["kx"]), "m": float(kwargs["m"])},
            "arrays": {},
        }
        for name in coeff_names:
            key = f"{label}/{name}"
            arr = np.asarray(getattr(coeffs, name))
            arrays[key] = arr
            case_meta["arrays"][name] = {
                "npz_key": key,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": _array_digest(arr),
            }
        metadata["cases"].append(case_meta)
    return metadata, arrays


def _write_npz(path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "__metadata_json__": np.array(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        **arrays,
    }
    np.savez_compressed(path, **payload)


def _energy_weights(mesh, baseflow, *, model: str) -> np.ndarray:
    from linstabpy.operators import compute_chu_norm

    p = mesh.used_ind
    w_int = mesh.W.ravel(order="F")[p]
    rho = baseflow.RHO.ravel(order="F")[p]
    temp = baseflow.T.ravel(order="F")[p]
    if model.lower() == "axy":
        radius = mesh.Y.ravel(order="F")[p]
        weights, _ = compute_chu_norm(
            w_int,
            rho,
            temp,
            baseflow.kappa,
            baseflow.Ma,
            mesh.ngp,
            model=model,
            R=radius,
        )
    else:
        weights, _ = compute_chu_norm(
            w_int,
            rho,
            temp,
            baseflow.kappa,
            baseflow.Ma,
            mesh.ngp,
        )
    return weights


def _high_mode_case(
    *,
    label: str,
    with_boundary_conditions: bool,
    n_modes: int,
    ncv: int,
    target: complex,
) -> dict[str, Any]:
    from linstabpy.operators import build_lnse_operator, build_lnse_system
    from linstabpy.stability import eigenvalues as temporal_eigenvalues

    mesh = _mesh("cartesian", model="2D")
    baseflow = _uniform_baseflow(mesh)
    bc_recipe = [
        {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
        {"type": "dirichlet", "borders": "b", "variables": "ruvw"},
        {"type": "neumann", "borders": "b", "variables": "T"},
        {"type": "neumann", "borders": "r", "variables": "ruvwT"},
    ]
    if with_boundary_conditions:
        operator, idx, bc_indices = build_lnse_system(
            mesh,
            baseflow,
            kx=0.2,
            model="2D",
            bc_recipe=bc_recipe,
            bc_penalty=1.0e4,
            backend="scipy",
        )
    else:
        operator, idx = build_lnse_operator(mesh, baseflow, kx=0.2, model="2D")
        bc_indices = np.array([], dtype=np.int64)

    w_diag = _energy_weights(mesh, baseflow, model="2D")
    solver_cfg = {
        "eigen": {
            "eps_tol": 1.0e-8,
            "eps_max_it": 500,
            "cond_warn": False,
        }
    }
    (omega, _, info), solve_stage = _stage(
        lambda: temporal_eigenvalues(
            operator,
            n_modes=n_modes,
            target=target,
            W_diag=w_diag,
            ncv=ncv,
            solver_cfg=solver_cfg,
            return_info=True,
        )
    )
    return {
        "label": label,
        "model": "2D",
        "mesh": "cartesian",
        "ngp": int(mesh.ngp),
        "ndof": int(operator.shape[0]),
        "operator_nnz": int(operator.nnz),
        "with_boundary_conditions": bool(with_boundary_conditions),
        "bc_indices": _index_payload(np.asarray(bc_indices, dtype=np.int64)),
        "target": [float(target.real), float(target.imag)],
        "n_modes_requested": int(n_modes),
        "ncv_requested": int(ncv),
        "solve_stage": solve_stage,
        "eigenvalues": _complex_vector_payload(omega),
        "solver_info": {
            "method": str(info.get("method")),
            "nev_requested": int(info.get("nev_requested", n_modes)),
            "ncv_used": int(info.get("ncv_used", ncv)),
            "nconv": int(info.get("nconv", 0)),
            "eps_iterations": int(info.get("eps_iterations", 0)),
            "eps_reason": str(info.get("eps_reason")),
            "ksp_reason": str(info.get("ksp_reason")),
            "eps_tol": float(info.get("eps_tol", np.nan)),
            "eps_setup_seconds": float(info.get("eps_setup_seconds", np.nan)),
            "eps_solve_seconds": float(info.get("eps_solve_seconds", np.nan)),
            "residual_relative": _float_vector_payload(
                np.asarray(info.get("residual_relative", []), dtype=float)
            ),
            "residual_backward": _float_vector_payload(
                np.asarray(info.get("residual_backward", []), dtype=float)
            ),
            "leading_weighted_norm_error": float(info.get("leading_weighted_norm_error", np.nan)),
        },
    }


def _high_mode_baseline(*, n_modes: int, ncv: int) -> list[dict[str, Any]]:
    return [
        _high_mode_case(
            label="interior_target_100mode",
            with_boundary_conditions=False,
            n_modes=n_modes,
            ncv=ncv,
            target=0.8 - 1.0j,
        ),
        _high_mode_case(
            label="boundary_influenced_100mode",
            with_boundary_conditions=True,
            n_modes=n_modes,
            ncv=ncv,
            target=0.5 - 1.0j,
        ),
    ]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--high-mode-output",
        type=Path,
        default=None,
        help="Write the high-mode baseline as a separate artifact.",
    )
    parser.add_argument("--include-high-mode", action="store_true")
    parser.add_argument("--high-mode-count", type=int, default=100)
    parser.add_argument("--high-mode-ncv", type=int, default=140)
    parser.add_argument(
        "--coefficient-snapshot-output",
        type=Path,
        default=None,
        help="Write selected raw coefficient tensors as a compressed NPZ baseline.",
    )
    args = parser.parse_args(argv)

    report = {
        "generated_at_utc": _now(),
        "git_sha": _git_sha(),
        "status": "pass",
        "bc_characterization": _bc_characterization(),
        "coefficient_characterization": _coefficient_characterization(),
    }
    if args.include_high_mode:
        report["high_mode_baseline"] = _high_mode_baseline(
            n_modes=args.high_mode_count,
            ncv=args.high_mode_ncv,
        )
    _write_json(args.output, report)
    if args.high_mode_output is not None:
        high_mode_report = {
            "generated_at_utc": _now(),
            "git_sha": _git_sha(),
            "status": "pass",
            "high_mode_baseline": _high_mode_baseline(
                n_modes=args.high_mode_count,
                ncv=args.high_mode_ncv,
            ),
        }
        _write_json(args.high_mode_output, high_mode_report)
    if args.coefficient_snapshot_output is not None:
        snapshot_metadata, snapshot_arrays = _coefficient_snapshot_arrays()
        snapshot_metadata.update(
            {
                "generated_at_utc": _now(),
                "git_sha": _git_sha(),
                "status": "pass",
                "artifact_kind": "phase0_coefficient_tensor_snapshot",
            }
        )
        _write_npz(args.coefficient_snapshot_output, snapshot_metadata, snapshot_arrays)
    print(f"Phase 0 characterization: status=pass output={args.output}")
    if args.high_mode_output is not None:
        print(f"Phase 0 high-mode baseline: status=pass output={args.high_mode_output}")
    if args.coefficient_snapshot_output is not None:
        print(
            f"Phase 0 coefficient snapshots: status=pass output={args.coefficient_snapshot_output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
