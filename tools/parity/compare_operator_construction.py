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


def run_numeric_parity(seed: int, tol: float) -> dict[str, Any]:
    from petsc4py import PETSc
    from slepc4py import SLEPc

    from linstab2d.baseflow import BaseFlow, sutherland_air
    from linstab2d.mesh import Mesh
    from linstab2d.operators import build_lnse_operator, build_spatial_operator
    from linstab2d.operators.assembly import _assemble_from_prepared, _prepare_coefficients
    from linstab2d.utils.petsc_core import scipy_to_petsc, vec_to_numpy

    rng = np.random.default_rng(seed)
    case_reports = []

    for model in ("2D", "axy"):
        mesh, bf, opts = _case_setup(model, seed, Mesh, BaseFlow, sutherland_air)
        kx = float(opts["kx"])
        m = float(opts["m"])

        L0, _ = build_lnse_operator(mesh, bf, kx=kx, model=model, m=m)
        prep = _prepare_coefficients(mesh, bf, kx=kx, model=model, m=m)
        L0_rebuilt = _assemble_from_prepared(prep, floquet_exp=0.0)

        delta = L0 - L0_rebuilt
        rebuild_abs = float(np.max(np.abs(delta.data))) if delta.nnz else 0.0
        rebuild_rel = float(
            np.linalg.norm(delta.data) / max(np.linalg.norm(L0.data), 1e-30)
        ) if delta.nnz else 0.0

        v = rng.standard_normal(L0.shape[1]) + 1j * rng.standard_normal(L0.shape[1])
        l0_matvec_resid = _petsc_matvec_residual(L0, v, scipy_to_petsc, vec_to_numpy, PETSc)

        Lc, R, Lw, _ = build_spatial_operator(mesh, bf, model=model, kx=kx, m=m, delta=1.0)
        v_sp = rng.standard_normal(Lc.shape[1]) + 1j * rng.standard_normal(Lc.shape[1])
        lc_matvec_resid = _petsc_matvec_residual(Lc, v_sp, scipy_to_petsc, vec_to_numpy, PETSc)
        r_matvec_resid = _petsc_matvec_residual(R, v_sp, scipy_to_petsc, vec_to_numpy, PETSc)
        lw_matvec_resid = _petsc_matvec_residual(Lw, v_sp, scipy_to_petsc, vec_to_numpy, PETSc)

        case_pass = (
            rebuild_abs <= tol
            and rebuild_rel <= tol
            and l0_matvec_resid <= tol
            and lc_matvec_resid <= tol
            and r_matvec_resid <= tol
            and lw_matvec_resid <= tol
        )
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
                "pass": case_pass,
            }
        )

    overall_pass = all(c["pass"] for c in case_reports)
    return {
        "generated_at_utc": _now_utc(),
        "seed": seed,
        "tolerance": tol,
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
        default=Path("artifacts/operator_numeric_parity.json"),
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    try:
        report = run_numeric_parity(seed=args.seed, tol=args.tol)
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
        f"output={args.output}",
    )
    if not report.get("pass", False) and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
