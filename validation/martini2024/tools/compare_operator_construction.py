#!/usr/bin/env python3
"""Numeric parity checks for operator construction and PETSc matvec parity."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _complex_to_parts(value: Any) -> dict[str, float]:
    z = complex(value)
    return {"real": float(np.real(z)), "imag": float(np.imag(z))}


def _resolve_threshold(raw_value: float | None, fallback: float, name: str) -> float:
    if raw_value is None:
        return float(fallback)
    value = float(raw_value)
    if value < 0:
        raise ValueError(f"Threshold {name} must be non-negative, got {raw_value}")
    return value


def _build_thresholds(tol: float, args: argparse.Namespace) -> dict[str, Any]:
    if not args.strict:
        return {
            "strict": False,
            "rebuild": {"abs": float(tol), "rel": float(tol)},
            "matvec": {
                "L0": float(tol),
                "Lc": float(tol),
                "R": float(tol),
                "Lw": float(tol),
            },
        }

    matvec_default = _resolve_threshold(args.matvec_threshold, tol, "matvec-threshold")
    return {
        "strict": bool(args.strict),
        "rebuild": {
            "abs": _resolve_threshold(args.rebuild_abs_threshold, tol, "rebuild-abs-threshold"),
            "rel": _resolve_threshold(args.rebuild_rel_threshold, tol, "rebuild-rel-threshold"),
        },
        "matvec": {
            "L0": _resolve_threshold(args.l0_matvec_threshold, matvec_default, "l0-matvec-threshold"),
            "Lc": _resolve_threshold(args.lc_matvec_threshold, matvec_default, "lc-matvec-threshold"),
            "R": _resolve_threshold(args.r_matvec_threshold, matvec_default, "r-matvec-threshold"),
            "Lw": _resolve_threshold(args.lw_matvec_threshold, matvec_default, "lw-matvec-threshold"),
        },
    }


def _delta_unresolved_terms(
    delta,
    reference,
    abs_threshold: float,
    rel_threshold: float,
    max_terms: int,
) -> dict[str, Any]:
    if delta.nnz == 0:
        return {
            "unresolved_count": 0,
            "max_abs_residual": 0.0,
            "max_rel_residual": 0.0,
            "exceeded_abs_threshold_count": 0,
            "exceeded_rel_threshold_count": 0,
            "term_count": 0,
            "term_limit": max_terms,
            "terms": [],
        }

    coo = delta.tocoo(copy=False)
    ref = reference.tocoo(copy=False)
    ref_scale = float(np.max(np.abs(ref.data))) if ref.nnz else 1e-30
    ref_scale = max(ref_scale, 1e-30)

    delta_abs = np.abs(coo.data)
    delta_rel = delta_abs / ref_scale

    exceeds_abs = delta_abs > abs_threshold
    exceeds_rel = delta_abs / max(ref_scale, 1e-30) > rel_threshold
    exceeds = np.logical_or(exceeds_abs, exceeds_rel)
    unresolved_indices = np.flatnonzero(exceeds)
    term_count = int(coo.data.size)

    terms: list[dict[str, Any]] = []
    if unresolved_indices.size and max_terms > 0:
        order = unresolved_indices[np.argsort(delta_abs[unresolved_indices])[::-1]]
        order = order[:max_terms]
        ref_map: dict[tuple[int, int], complex] = {
            (int(r), int(c)): complex(v) for r, c, v in zip(ref.row, ref.col, ref.data)
        }
        for i in order:
            row = int(coo.row[i])
            col = int(coo.col[i])
            value_delta = complex(coo.data[i])
            value_reference = ref_map.get((row, col), 0.0)
            terms.append(
                {
                    "row": row,
                    "col": col,
                    "abs_residual": float(delta_abs[i]),
                    "rel_residual": float(delta_rel[i]),
                    "value_delta": _complex_to_parts(value_delta),
                    "value_reference": _complex_to_parts(value_reference),
                    "exceeds_abs_threshold": bool(exceeds_abs[i]),
                    "exceeds_rel_threshold": bool(exceeds_rel[i]),
                }
            )

    return {
        "unresolved_count": int(unresolved_indices.size),
        "max_abs_residual": float(np.max(delta_abs)),
        "max_rel_residual": float(np.max(delta_rel)),
        "exceeded_abs_threshold_count": int(np.count_nonzero(exceeds_abs)),
        "exceeded_rel_threshold_count": int(np.count_nonzero(exceeds_rel)),
        "term_count": term_count,
        "term_limit": max_terms,
        "terms": terms,
    }


def _smooth_field(X: np.ndarray, Y: np.ndarray, seed: int, scale: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=6)
    return (
        1.0
        + scale * a[0] * np.sin(0.7 * X)
        + scale * a[1] * np.cos(0.9 * Y)
        + scale * a[2] * np.sin(0.4 * X + 0.6 * Y)
        + scale * a[3] * np.cos(0.5 * X - 0.2 * Y)
        + scale * a[4] * np.sin(0.3 * X) * np.cos(0.8 * Y)
        + scale * a[5] * np.cos(0.2 * X) * np.sin(0.7 * Y)
    )


def _case_setup(model: str, seed: int, Mesh, BaseFlow, sutherland_air):
    if model == "2D":
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=28,
            ny=20,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        U = _smooth_field(X, Y, seed + 1, scale=0.10)
        V = 0.08 * (_smooth_field(X, Y, seed + 2, scale=0.30) - 1.0)
        W = 0.08 * (_smooth_field(X, Y, seed + 3, scale=0.25) - 1.0)
        RHO = np.maximum(0.2, _smooth_field(X, Y, seed + 4, scale=0.10))
        T = np.maximum(0.2, _smooth_field(X, Y, seed + 5, scale=0.07))
        bf = BaseFlow(
            Re=1e4,
            Ma=0.3,
            Pr=0.7,
            kappa=1.4,
            T_0=288.15,
            RHO=RHO,
            U=U,
            V=V,
            W=W,
            T=T,
        )
        sutherland_air(bf)
        return mesh, bf, {"model": "2D", "kx": 0.6, "m": 0.0}

    if model == "axy":
        mesh = Mesh.create(
            xrange=(0.0, 3.0),
            yrange=(0.0, 1.5),
            nx=24,
            ny=18,
            fd_order=4,
            y_symmetry=True,
        )
        X, Y = mesh.X, mesh.Y
        U = 0.8 + 0.15 * np.exp(-2.2 * Y**2) + 0.03 * np.sin(0.6 * X)  # axial
        V = 0.02 * np.sin(0.4 * X) * np.exp(-1.5 * Y)                    # radial
        W = 0.01 * np.cos(0.5 * X) * np.exp(-1.2 * Y)                    # azimuthal
        RHO = np.maximum(0.2, 1.0 + 0.04 * np.cos(0.5 * X) - 0.03 * Y)
        T = np.maximum(0.2, 1.0 + 0.03 * np.sin(0.8 * X) + 0.02 * Y)
        bf = BaseFlow(
            Re=1e4,
            Ma=0.3,
            Pr=0.7,
            kappa=1.4,
            T_0=288.15,
            RHO=RHO,
            U=U,
            V=V,
            W=W,
            T=T,
        )
        sutherland_air(bf)
        return mesh, bf, {"model": "axy", "kx": 0.0, "m": 1.0}

    raise ValueError(f"Unsupported model '{model}'")


def _petsc_matvec_residual(A_sp, vec, scipy_to_petsc, vec_to_numpy, PETSc) -> float:
    A = scipy_to_petsc(A_sp)
    x = PETSc.Vec().createWithArray(np.ascontiguousarray(vec, dtype=complex))
    y = A.createVecLeft()
    A.mult(x, y)
    y_petsc = vec_to_numpy(y)
    y_ref = A_sp @ vec
    return float(np.linalg.norm(y_ref - y_petsc) / max(np.linalg.norm(y_ref), 1e-30))


def run_numeric_parity(
    seed: int,
    tol: float,
    thresholds: dict[str, Any],
    max_unresolved_terms: int,
) -> dict[str, Any]:
    from petsc4py import PETSc
    from slepc4py import SLEPc

    from linstab2d.baseflow import BaseFlow, sutherland_air
    from linstab2d.mesh import Mesh
    from linstab2d.operators import build_lnse_operator, build_spatial_operator
    from linstab2d.operators.assembly import _assemble_from_prepared, _prepare_coefficients
    from linstab2d.utils.petsc_core import scipy_to_petsc, vec_to_numpy

    rng = np.random.default_rng(seed)
    case_reports = []

    rebuild_thresholds = thresholds["rebuild"]
    matvec_thresholds = thresholds["matvec"]

    for model in ("2D", "axy"):
        mesh, bf, opts = _case_setup(model, seed, Mesh, BaseFlow, sutherland_air)
        kx = float(opts["kx"])
        m = float(opts["m"])

        L0, _ = build_lnse_operator(mesh, bf, kx=kx, model=model, m=m)
        prep = _prepare_coefficients(mesh, bf, kx=kx, model=model, m=m)
        L0_rebuilt = _assemble_from_prepared(prep, floquet_exp=0.0)

        delta = L0 - L0_rebuilt
        rebuild_abs = float(np.max(np.abs(delta.data))) if delta.nnz else 0.0
        rebuild_rel = float(np.linalg.norm(delta.data) / max(np.linalg.norm(L0.data), 1e-30)) if delta.nnz else 0.0
        rebuild_terms = _delta_unresolved_terms(
            delta,
            L0,
            abs_threshold=rebuild_thresholds["abs"],
            rel_threshold=rebuild_thresholds["rel"],
            max_terms=max_unresolved_terms,
        )

        v = rng.standard_normal(L0.shape[1]) + 1j * rng.standard_normal(L0.shape[1])
        l0_matvec_resid = _petsc_matvec_residual(L0, v, scipy_to_petsc, vec_to_numpy, PETSc)
        Lc, R, Lw, _ = build_spatial_operator(mesh, bf, model=model, kx=kx, m=m, delta=1.0)
        v_sp = rng.standard_normal(Lc.shape[1]) + 1j * rng.standard_normal(Lc.shape[1])
        lc_matvec_resid = _petsc_matvec_residual(Lc, v_sp, scipy_to_petsc, vec_to_numpy, PETSc)
        r_matvec_resid = _petsc_matvec_residual(R, v_sp, scipy_to_petsc, vec_to_numpy, PETSc)
        lw_matvec_resid = _petsc_matvec_residual(Lw, v_sp, scipy_to_petsc, vec_to_numpy, PETSc)

        block_checks = {
            "rebuild": {
                "pass": rebuild_abs <= rebuild_thresholds["abs"] and rebuild_rel <= rebuild_thresholds["rel"],
                "abs_residual": rebuild_abs,
                "abs_threshold": rebuild_thresholds["abs"],
                "rel_residual": rebuild_rel,
                "rel_threshold": rebuild_thresholds["rel"],
                "unresolved_deltas": rebuild_terms,
            },
            "L0_matvec": {
                "pass": l0_matvec_resid <= matvec_thresholds["L0"],
                "residual": l0_matvec_resid,
                "threshold": matvec_thresholds["L0"],
            },
            "Lc_matvec": {
                "pass": lc_matvec_resid <= matvec_thresholds["Lc"],
                "residual": lc_matvec_resid,
                "threshold": matvec_thresholds["Lc"],
            },
            "R_matvec": {
                "pass": r_matvec_resid <= matvec_thresholds["R"],
                "residual": r_matvec_resid,
                "threshold": matvec_thresholds["R"],
            },
            "Lw_matvec": {
                "pass": lw_matvec_resid <= matvec_thresholds["Lw"],
                "residual": lw_matvec_resid,
                "threshold": matvec_thresholds["Lw"],
            },
        }
        failed_blocks = [name for name, item in block_checks.items() if not item["pass"]]

        case_pass = len(failed_blocks) == 0
        case_reports.append(
            {
                "model": model,
                "ngp": int(mesh.ngp),
                "L0_shape": [int(L0.shape[0]), int(L0.shape[1])],
                "L0_nnz": int(L0.nnz),
                "spatial_shape": [int(Lc.shape[0]), int(Lc.shape[1])],
                "Lc_nnz": int(Lc.nnz),
                "R_nnz": int(R.nnz),
                "Lw_nnz": int(Lw.nnz),
                "rebuild_abs_residual": rebuild_abs,
                "rebuild_rel_residual": rebuild_rel,
                "petsc_matvec_rel_residuals": {
                    "L0": l0_matvec_resid,
                    "Lc": lc_matvec_resid,
                    "R": r_matvec_resid,
                    "Lw": lw_matvec_resid,
                },
                "thresholds": {
                    "rebuild_abs": rebuild_thresholds["abs"],
                    "rebuild_rel": rebuild_thresholds["rel"],
                    "matvec": {
                        "L0": matvec_thresholds["L0"],
                        "Lc": matvec_thresholds["Lc"],
                        "R": matvec_thresholds["R"],
                        "Lw": matvec_thresholds["Lw"],
                    },
                },
                "block_checks": block_checks,
                "failed_blocks": failed_blocks,
                "pass": case_pass,
            }
        )

    overall_pass = all(c["pass"] for c in case_reports)
    return {
        "generated_at_utc": _now_utc(),
        "seed": seed,
        "tolerance": tol,
        "strict_mode": bool(thresholds["strict"]),
        "backend": {
            "petsc_version": list(PETSc.Sys.getVersion()),
            "slepc_version": list(SLEPc.Sys.getVersion()),
            "petsc_scalar_type": str(PETSc.ScalarType.__name__),
        },
        "cases": case_reports,
        "pass": overall_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tol", type=float, default=1e-11)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/martini2024/output/operator_numeric_parity.json"),
    )
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict per-block/per-metric threshold checks and triage output.",
    )
    parser.add_argument(
        "--matvec-threshold",
        type=float,
        default=None,
        help="Default threshold for all PETSc matvec checks.",
    )
    parser.add_argument(
        "--rebuild-abs-threshold",
        type=float,
        default=None,
        help="Absolute threshold for rebuild residual.",
    )
    parser.add_argument(
        "--rebuild-rel-threshold",
        type=float,
        default=None,
        help="Relative threshold for rebuild residual.",
    )
    parser.add_argument(
        "--l0-matvec-threshold",
        type=float,
        default=None,
        help="Threshold for L0 PETSc matvec residual.",
    )
    parser.add_argument(
        "--lc-matvec-threshold",
        type=float,
        default=None,
        help="Threshold for Lc PETSc matvec residual.",
    )
    parser.add_argument(
        "--r-matvec-threshold",
        type=float,
        default=None,
        help="Threshold for R PETSc matvec residual.",
    )
    parser.add_argument(
        "--lw-matvec-threshold",
        type=float,
        default=None,
        help="Threshold for Lw PETSc matvec residual.",
    )
    parser.add_argument(
        "--max-unresolved-terms",
        type=int,
        default=20,
        help="Maximum number of unresolved delta terms reported per model.",
    )
    args = parser.parse_args()

    try:
        thresholds = _build_thresholds(args.tol, args)
        if args.max_unresolved_terms < 0:
            raise ValueError("--max-unresolved-terms must be >= 0")
        report = run_numeric_parity(
            seed=args.seed,
            tol=args.tol,
            thresholds=thresholds,
            max_unresolved_terms=args.max_unresolved_terms,
        )
    except Exception as exc:  # pragma: no cover - environment/runtime dependent
        report = {
            "generated_at_utc": _now_utc(),
            "seed": args.seed,
            "tolerance": args.tol,
            "pass": False,
            "status": "blocked",
            "error": str(exc),
        }

    _write_json(args.output, report)
    print(
        "numeric operator parity:",
        f"pass={report.get('pass', False)}",
        f"strict={report.get('strict_mode', False)}",
        f"output={args.output}",
    )
    if not report.get("pass", False) and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
