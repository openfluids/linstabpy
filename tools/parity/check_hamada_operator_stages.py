#!/usr/bin/env python3
"""Matrix-stage parity for Hamada operator assembly vs MATLAB exports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp

from linstab2d.baseflow import load_hamada_case, sutherland_air
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.boundary_conditions.neumann import apply_neumann
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_matlab_sparse(path: Path, key: str) -> sp.csc_matrix:
    with h5py.File(path, "r") as handle:
        group = handle[key]
        data = np.array(group["data"])
        indices = np.array(group["ir"], dtype=np.int64)
        indptr = np.array(group["jc"], dtype=np.int64)
    n = indptr.size - 1
    return sp.csc_matrix((data, indices, indptr), shape=(n, n))


def _top_delta_terms(delta: sp.coo_matrix, k: int) -> list[dict[str, Any]]:
    if delta.nnz == 0 or k <= 0:
        return []
    order = np.argsort(np.abs(delta.data))[::-1][:k]
    out: list[dict[str, Any]] = []
    for i in order:
        z = complex(delta.data[i])
        out.append(
            {
                "row": int(delta.row[i]),
                "col": int(delta.col[i]),
                "abs_error": float(abs(z)),
                "value_real": float(z.real),
                "value_imag": float(z.imag),
            }
        )
    return out


def _stage_report(
    name: str,
    py_mat: sp.csc_matrix,
    ref_mat: sp.csc_matrix,
    *,
    tol_abs: float,
    tol_rel: float,
    enforce_abs: bool,
    top_k: int,
) -> dict[str, Any]:
    same_shape = py_mat.shape == ref_mat.shape
    same_indptr = same_shape and np.array_equal(py_mat.indptr, ref_mat.indptr)
    same_indices = same_shape and np.array_equal(py_mat.indices, ref_mat.indices)
    structure_match = bool(same_shape and same_indptr and same_indices)

    if structure_match:
        diff_data = py_mat.data - ref_mat.data
        max_abs = float(np.max(np.abs(diff_data))) if diff_data.size else 0.0
        ref_scale = max(float(np.max(np.abs(ref_mat.data))) if ref_mat.nnz else 0.0, 1e-30)
        max_rel = max_abs / ref_scale
        frob_rel = float(np.linalg.norm(diff_data) / max(np.linalg.norm(ref_mat.data), 1e-30))
        delta = (py_mat - ref_mat).tocoo()
        excess_nnz = 0
        missing_nnz = 0
    else:
        py_bin = py_mat.copy().astype(float)
        ref_bin = ref_mat.copy().astype(float)
        py_bin.data[:] = 1.0
        ref_bin.data[:] = 1.0
        pattern_delta = (py_bin - ref_bin).tocoo()
        excess_nnz = int(np.count_nonzero(pattern_delta.data > 0.0))
        missing_nnz = int(np.count_nonzero(pattern_delta.data < 0.0))

        delta = (py_mat - ref_mat).tocoo()
        abs_delta = np.abs(delta.data)
        max_abs = float(np.max(abs_delta)) if abs_delta.size else 0.0
        ref_scale = max(float(np.max(np.abs(ref_mat.data))) if ref_mat.nnz else 0.0, 1e-30)
        max_rel = max_abs / ref_scale
        frob_rel = float(np.linalg.norm(delta.data) / max(np.linalg.norm(ref_mat.data), 1e-30))

    pass_value = max_rel <= tol_rel and (max_abs <= tol_abs if enforce_abs else True)
    stage_pass = bool(structure_match and pass_value)

    return {
        "name": name,
        "shape_python": [int(py_mat.shape[0]), int(py_mat.shape[1])],
        "shape_matlab": [int(ref_mat.shape[0]), int(ref_mat.shape[1])],
        "nnz_python": int(py_mat.nnz),
        "nnz_matlab": int(ref_mat.nnz),
        "structure": {
            "shape_match": same_shape,
            "indptr_match": same_indptr,
            "indices_match": same_indices,
            "match": structure_match,
            "excess_nnz": excess_nnz,
            "missing_nnz": missing_nnz,
        },
        "value": {
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "frob_rel_error": frob_rel,
            "tol_abs": float(tol_abs),
            "tol_rel": float(tol_rel),
            "enforce_abs": bool(enforce_abs),
            "top_terms": _top_delta_terms(delta, top_k),
        },
        "pass": stage_pass,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    exports_dir = args.exports_dir
    matlab_stages = {
        "pre": _load_matlab_sparse(exports_dir / "L0_preBC.mat", "L0_preBC"),
        "after_lt": _load_matlab_sparse(exports_dir / "L0_after_dirichlet_lt.mat", "L0_after_dirichlet_lt"),
        "after_b": _load_matlab_sparse(exports_dir / "L0_after_dirichlet_b.mat", "L0_after_dirichlet_b"),
        "after_nb": _load_matlab_sparse(exports_dir / "L0_after_neumann_bT.mat", "L0_after_neumann_bT"),
        "post": _load_matlab_sparse(exports_dir / "L0_postBC.mat", "L0_postBC"),
    }

    X, Y, baseflow, sponge = load_hamada_case(
        args.case_dir,
        Re=args.re,
        Ma=args.ma,
        Pr=args.pr,
        kappa=args.kappa,
        T_0=args.t0,
        stride_ksi=args.stride_ksi,
        stride_eta=args.stride_eta,
        transpose_input=True,
        zero_spanwise=False,
    )
    sutherland_air(baseflow)
    if args.zero_mu_t_derivs:
        baseflow.dmudT[:] = 0.0
        baseflow.d2mudT2[:] = 0.0

    mesh = Mesh.create((0.0, 1.0), (0.0, 0.5), nx=X.shape[1], ny=X.shape[0], fd_order=args.fd_order)
    mesh = mesh.deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge

    l0_pre, idx = build_lnse_operator(mesh, baseflow, kx=args.beta, model="2D")
    l0_lt, _ = apply_dirichlet(l0_pre, idx, "lt", "ruvwT", mesh.ngp, penalty=args.penalty)
    l0_b, _ = apply_dirichlet(l0_lt, idx, "b", "ruvw", mesh.ngp, penalty=args.penalty)
    l0_nb, _ = apply_neumann(l0_b, mesh, idx, "b", "T", penalty=args.penalty)
    l0_post, _ = apply_neumann(l0_nb, mesh, idx, "r", "ruvwT", penalty=args.penalty)

    python_stages = {
        "pre": l0_pre.tocsc(),
        "after_lt": l0_lt.tocsc(),
        "after_b": l0_b.tocsc(),
        "after_nb": l0_nb.tocsc(),
        "post": l0_post.tocsc(),
    }
    abs_stage_names = {"pre", "after_lt", "after_b"}

    reports = []
    for name in ("pre", "after_lt", "after_b", "after_nb", "post"):
        reports.append(
            _stage_report(
                name=name,
                py_mat=python_stages[name],
                ref_mat=matlab_stages[name],
                tol_abs=args.tol_abs,
                tol_rel=args.tol_rel,
                enforce_abs=(name in abs_stage_names),
                top_k=args.top_k,
            )
        )

    overall_pass = all(stage["pass"] for stage in reports)
    worst_abs_stage = max(reports, key=lambda r: r["value"]["max_abs_error"])
    worst_rel_stage = max(reports, key=lambda r: r["value"]["max_rel_error"])

    return {
        "generated_at_utc": _now_utc(),
        "case_dir": str(args.case_dir),
        "exports_dir": str(exports_dir),
        "config": {
            "re": float(args.re),
            "ma": float(args.ma),
            "pr": float(args.pr),
            "kappa": float(args.kappa),
            "t0": float(args.t0),
            "beta": float(args.beta),
            "penalty": float(args.penalty),
            "fd_order": int(args.fd_order),
            "stride_ksi": int(args.stride_ksi),
            "stride_eta": int(args.stride_eta),
            "zero_mu_t_derivs": bool(args.zero_mu_t_derivs),
            "tol_abs": float(args.tol_abs),
            "tol_rel": float(args.tol_rel),
            "top_k": int(args.top_k),
        },
        "stages": reports,
        "worst_stage_by_abs": {
            "name": worst_abs_stage["name"],
            "max_abs_error": worst_abs_stage["value"]["max_abs_error"],
        },
        "worst_stage_by_rel": {
            "name": worst_rel_stage["name"],
            "max_rel_error": worst_rel_stage["value"]["max_rel_error"],
        },
        "pass": overall_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=Path("example_6_hamada"))
    parser.add_argument("--exports-dir", type=Path, default=Path("example_6_hamada/MATLAB_exports"))
    parser.add_argument("--re", type=float, default=395000.0)
    parser.add_argument("--ma", type=float, default=2.0)
    parser.add_argument("--pr", type=float, default=0.747)
    parser.add_argument("--kappa", type=float, default=1.4)
    parser.add_argument("--t0", type=float, default=1540.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--penalty", type=float, default=1e8)
    parser.add_argument("--fd-order", type=int, default=4)
    parser.add_argument("--stride-ksi", type=int, default=1)
    parser.add_argument("--stride-eta", type=int, default=1)
    parser.add_argument("--zero-mu-t-derivs", action="store_true", default=True)
    parser.add_argument("--tol-abs", type=float, default=1e-3)
    parser.add_argument("--tol-rel", type=float, default=1e-8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/parity/hamada_operator_stage_parity.json"),
    )
    args = parser.parse_args()

    report = run(args)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    with args.artifact.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        "Hamada operator stage parity:",
        f"pass={report['pass']}",
        f"worst_abs={report['worst_stage_by_abs']['name']}:{report['worst_stage_by_abs']['max_abs_error']:.3e}",
        f"worst_rel={report['worst_stage_by_rel']['name']}:{report['worst_stage_by_rel']['max_rel_error']:.3e}",
        f"artifact={args.artifact}",
    )
    if (not report["pass"]) and (not args.allow_fail):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
