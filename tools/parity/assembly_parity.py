#!/usr/bin/env python3
"""Compare LNSE assembly artifacts or live backend pairs.

This tool supports two workflows:

1. ``artifacts``: compare two ``.npz`` sparse-matrix artifacts.
2. ``pair``: build a deterministic synthetic case twice (for example
   ``scipy`` vs ``petsc``), compare the assembled operators, and optionally
   save the resulting matrix artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _complex_parts(value: complex) -> dict[str, float]:
    z = complex(value)
    return {"real": float(np.real(z)), "imag": float(np.imag(z))}


def _save_sparse_artifact(
    path: Path,
    matrix: sp.spmatrix,
    *,
    key: str = "L0",
    metadata: dict[str, Any] | None = None,
) -> None:
    csr = sp.csr_matrix(matrix, dtype=complex)
    payload: dict[str, Any] = {
        f"{key}_data": csr.data,
        f"{key}_indices": csr.indices.astype(np.int64),
        f"{key}_indptr": csr.indptr.astype(np.int64),
        f"{key}_shape": np.asarray(csr.shape, dtype=np.int64),
        f"{key}_format": np.asarray("csr"),
    }
    if metadata:
        for name, value in metadata.items():
            if isinstance(value, (list, tuple)):
                payload[name] = np.asarray(value)
            elif isinstance(value, (int, float, complex, np.generic, str, bool)):
                payload[name] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _load_sparse_artifact(path: Path, *, key: str = "L0") -> tuple[sp.csr_matrix, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        required = [f"{key}_data", f"{key}_indices", f"{key}_indptr", f"{key}_shape"]
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"Artifact {path} missing sparse keys for '{key}': {', '.join(missing)}")

        shape = tuple(int(x) for x in np.asarray(data[f"{key}_shape"], dtype=np.int64).reshape(-1))
        if len(shape) != 2:
            raise ValueError(f"Artifact {path} has invalid shape payload for '{key}': {shape}")

        matrix = sp.csr_matrix(
            (
                np.asarray(data[f"{key}_data"], dtype=complex),
                np.asarray(data[f"{key}_indices"], dtype=np.int64),
                np.asarray(data[f"{key}_indptr"], dtype=np.int64),
            ),
            shape=shape,
            dtype=complex,
        )
        metadata = {
            name: np.asarray(data[name]).tolist()
            for name in data.files
            if not name.startswith(f"{key}_")
        }
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix, metadata


def _block_nnz_report(matrix: sp.spmatrix) -> dict[str, Any]:
    csr = sp.csr_matrix(matrix, dtype=complex)
    nrows, ncols = csr.shape
    if nrows != ncols or nrows % 5 != 0:
        return {
            "available": False,
            "shape": [int(nrows), int(ncols)],
            "reason": "matrix shape is not square 5x5-block compatible",
        }

    ngp = nrows // 5
    per_block = np.zeros((5, 5), dtype=np.int64)
    for i in range(5):
        r0 = i * ngp
        r1 = (i + 1) * ngp
        for j in range(5):
            c0 = j * ngp
            c1 = (j + 1) * ngp
            per_block[i, j] = int(csr[r0:r1, c0:c1].nnz)

    return {
        "available": True,
        "ngp": int(ngp),
        "total_nnz": int(csr.nnz),
        "per_block": per_block.tolist(),
    }


def _compare_sparse_matrices(
    before: sp.spmatrix,
    after: sp.spmatrix,
    *,
    abs_tol: float,
    rel_tol: float,
    max_terms: int,
) -> dict[str, Any]:
    from linstab2d.utils.petsc_core import compare_sparse_matrices

    report = compare_sparse_matrices(
        before,
        after,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        check_structure=True,
        max_terms=max_terms,
    )
    mismatches = []
    for item in report.get("largest_mismatches", []):
        mismatches.append(
            {
                "row": int(item["row"]),
                "col": int(item["col"]),
                "reference": _complex_parts(item["reference"]),
                "candidate": _complex_parts(item["candidate"]),
                "abs_residual": float(item["abs_residual"]),
                "rel_residual": float(item["rel_residual"]),
            }
        )
    report["largest_mismatches"] = mismatches
    return report


def _compare_block_nnz(before: sp.spmatrix, after: sp.spmatrix) -> dict[str, Any]:
    left = _block_nnz_report(before)
    right = _block_nnz_report(after)
    if not left["available"] or not right["available"]:
        return {
            "available": False,
            "before": left,
            "after": right,
            "match": False,
        }

    before_arr = np.asarray(left["per_block"], dtype=np.int64)
    after_arr = np.asarray(right["per_block"], dtype=np.int64)
    delta = after_arr - before_arr
    return {
        "available": True,
        "before": left,
        "after": right,
        "delta_per_block": delta.tolist(),
        "match": bool(np.array_equal(before_arr, after_arr)),
        "max_abs_delta": int(np.max(np.abs(delta))) if delta.size else 0,
    }


def _compute_temporal_eigs(
    matrix: sp.spmatrix,
    *,
    n_modes: int,
    target: complex,
    solver_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from linstab2d.solvers.eigenvalue import temporal_eigenvalues

    eigvals, _, info = temporal_eigenvalues(
        sp.csc_matrix(matrix, dtype=complex),
        n_modes=n_modes,
        target=target,
        solver_cfg=solver_cfg,
        return_info=True,
    )
    residuals = info.get("eigenpair_residuals")
    return {
        "status": "ok",
        "eigenvalues": [_complex_parts(val) for val in eigvals],
        "max_residual": None if residuals is None else float(np.max(np.asarray(residuals, dtype=float))),
        "info": {
            key: value
            for key, value in info.items()
            if key in {"eps_iterations", "nconv", "eps_reason", "ksp_reason", "operator_is_petsc"}
        },
    }


def _compare_eigenvalues(
    before: sp.spmatrix,
    after: sp.spmatrix,
    *,
    n_modes: int,
    target: complex,
    abs_tol: float,
    rel_tol: float,
    max_ndof: int,
    skip: bool,
) -> dict[str, Any]:
    ndof = int(before.shape[0])
    if skip:
        return {"status": "skipped", "reason": "requested by --skip-eigs"}
    if before.shape != after.shape:
        return {"status": "skipped", "reason": "shape mismatch"}
    if ndof > max_ndof:
        return {
            "status": "skipped",
            "reason": f"ndof={ndof} exceeds max_eig_ndof={max_ndof}",
        }

    left = _compute_temporal_eigs(before, n_modes=n_modes, target=target)
    right = _compute_temporal_eigs(after, n_modes=n_modes, target=target)
    left_vals = np.array([complex(v["real"], v["imag"]) for v in left["eigenvalues"]], dtype=complex)
    right_vals = np.array([complex(v["real"], v["imag"]) for v in right["eigenvalues"]], dtype=complex)
    delta = right_vals - left_vals
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    max_rel = float(max_abs / max(np.max(np.abs(left_vals)), 1e-30)) if delta.size else 0.0
    passed = bool(max_abs <= abs_tol and max_rel <= rel_tol)
    return {
        "status": "ok",
        "pass": passed,
        "before": left,
        "after": right,
        "max_abs_residual": max_abs,
        "max_rel_residual": max_rel,
        "abs_tol": float(abs_tol),
        "rel_tol": float(rel_tol),
    }


def _compare_matrices(
    before: sp.spmatrix,
    after: sp.spmatrix,
    *,
    sparse_abs_tol: float,
    sparse_rel_tol: float,
    eig_abs_tol: float,
    eig_rel_tol: float,
    max_terms: int,
    eig_modes: int,
    eig_target: complex,
    max_eig_ndof: int,
    skip_eigs: bool,
    before_meta: dict[str, Any] | None = None,
    after_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sparse_report = _compare_sparse_matrices(
        before,
        after,
        abs_tol=sparse_abs_tol,
        rel_tol=sparse_rel_tol,
        max_terms=max_terms,
    )
    block_report = _compare_block_nnz(before, after)
    eigen_report = _compare_eigenvalues(
        before,
        after,
        n_modes=eig_modes,
        target=eig_target,
        abs_tol=eig_abs_tol,
        rel_tol=eig_rel_tol,
        max_ndof=max_eig_ndof,
        skip=skip_eigs,
    )

    pass_sparse = bool(sparse_report["pass"] and sparse_report["structure_match"])
    pass_blocks = bool(block_report.get("match", False)) if block_report.get("available", False) else True
    pass_eigs = bool(eigen_report.get("pass", False)) if eigen_report.get("status") == "ok" else True

    return {
        "generated_at_utc": _now_utc(),
        "before_metadata": before_meta or {},
        "after_metadata": after_meta or {},
        "shape": [int(before.shape[0]), int(before.shape[1])],
        "sparse": sparse_report,
        "block_nnz": block_report,
        "eigen_sanity": eigen_report,
        "pass": bool(pass_sparse and pass_blocks and pass_eigs),
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


def _case_setup(model: str, seed: int):
    from linstab2d.baseflow import BaseFlow, sutherland_air
    from linstab2d.mesh import Mesh

    if model == "2D":
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=18,
            ny=12,
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
            nx=16,
            ny=12,
            fd_order=4,
            y_symmetry=True,
        )
        X, Y = mesh.X, mesh.Y
        U = 0.8 + 0.15 * np.exp(-2.2 * Y**2) + 0.03 * np.sin(0.6 * X)
        V = 0.02 * np.sin(0.4 * X) * np.exp(-1.5 * Y)
        W = 0.01 * np.cos(0.5 * X) * np.exp(-1.2 * Y)
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

    raise ValueError(f"Unsupported model {model!r}; expected '2D' or 'axy'")


def _build_backend_matrix(
    *,
    backend: str,
    model: str,
    seed: int,
    petsc_matrix_mode: str,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    from linstab2d.operators import build_lnse_system
    from linstab2d.utils.petsc_core import petsc_to_scipy

    mesh, bf, opts = _case_setup(model, seed)
    solver_cfg = {"petsc_matrix_mode": petsc_matrix_mode}
    L0, _, _ = build_lnse_system(
        mesh,
        bf,
        kx=float(opts["kx"]),
        model=str(opts["model"]),
        m=float(opts["m"]),
        backend=backend,
        solver_cfg=solver_cfg,
    )
    if hasattr(L0, "getValuesCSR"):
        if int(L0.getComm().getSize()) != 1:
            raise NotImplementedError(
                "assembly_parity.py live backend mode currently supports only sequential PETSc matrices"
            )
        try:
            matrix = petsc_to_scipy(L0, format="csr")
        finally:
            L0.destroy()
    else:
        matrix = sp.csr_matrix(L0, dtype=complex)

    metadata = {
        "backend": backend,
        "model": model,
        "seed": int(seed),
        "kx": float(opts["kx"]),
        "m": float(opts["m"]),
        "ngp": int(mesh.ngp),
        "petsc_matrix_mode": petsc_matrix_mode,
    }
    return matrix, metadata


def _common_compare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sparse-abs-tol", type=float, default=1e-12)
    parser.add_argument("--sparse-rel-tol", type=float, default=1e-12)
    parser.add_argument("--eig-abs-tol", type=float, default=1e-8)
    parser.add_argument("--eig-rel-tol", type=float, default=1e-8)
    parser.add_argument("--max-terms", type=int, default=20)
    parser.add_argument("--eig-modes", type=int, default=4)
    parser.add_argument("--eig-target", type=complex, default=-0.05j)
    parser.add_argument("--max-eig-ndof", type=int, default=1500)
    parser.add_argument("--skip-eigs", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("tools/parity/output/assembly_parity.json"))
    parser.add_argument("--allow-fail", action="store_true")


def _run_artifact_mode(args: argparse.Namespace) -> dict[str, Any]:
    before, before_meta = _load_sparse_artifact(args.before, key=args.key_before)
    after, after_meta = _load_sparse_artifact(args.after, key=args.key_after)
    report = _compare_matrices(
        before,
        after,
        sparse_abs_tol=args.sparse_abs_tol,
        sparse_rel_tol=args.sparse_rel_tol,
        eig_abs_tol=args.eig_abs_tol,
        eig_rel_tol=args.eig_rel_tol,
        max_terms=args.max_terms,
        eig_modes=args.eig_modes,
        eig_target=args.eig_target,
        max_eig_ndof=args.max_eig_ndof,
        skip_eigs=args.skip_eigs,
        before_meta={**before_meta, "artifact_path": str(args.before.resolve())},
        after_meta={**after_meta, "artifact_path": str(args.after.resolve())},
    )
    report["mode"] = "artifacts"
    report["before_key"] = args.key_before
    report["after_key"] = args.key_after
    return report


def _run_pair_mode(args: argparse.Namespace) -> dict[str, Any]:
    before, before_meta = _build_backend_matrix(
        backend=args.backend_before,
        model=args.model,
        seed=args.seed,
        petsc_matrix_mode=args.petsc_matrix_mode,
    )
    after, after_meta = _build_backend_matrix(
        backend=args.backend_after,
        model=args.model,
        seed=args.seed,
        petsc_matrix_mode=args.petsc_matrix_mode,
    )

    if args.artifact_before is not None:
        _save_sparse_artifact(args.artifact_before, before, metadata=before_meta)
    if args.artifact_after is not None:
        _save_sparse_artifact(args.artifact_after, after, metadata=after_meta)

    report = _compare_matrices(
        before,
        after,
        sparse_abs_tol=args.sparse_abs_tol,
        sparse_rel_tol=args.sparse_rel_tol,
        eig_abs_tol=args.eig_abs_tol,
        eig_rel_tol=args.eig_rel_tol,
        max_terms=args.max_terms,
        eig_modes=args.eig_modes,
        eig_target=args.eig_target,
        max_eig_ndof=args.max_eig_ndof,
        skip_eigs=args.skip_eigs,
        before_meta=before_meta,
        after_meta=after_meta,
    )
    report["mode"] = "pair"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    artifacts = sub.add_parser("artifacts", help="Compare two sparse-matrix NPZ artifacts")
    artifacts.add_argument("before", type=Path)
    artifacts.add_argument("after", type=Path)
    artifacts.add_argument("--key-before", default="L0")
    artifacts.add_argument("--key-after", default="L0")
    _common_compare_arguments(artifacts)

    pair = sub.add_parser("pair", help="Build and compare deterministic backend pairs")
    pair.add_argument("--backend-before", default="scipy", choices=("scipy", "petsc", "auto"))
    pair.add_argument("--backend-after", default="petsc", choices=("scipy", "petsc", "auto"))
    pair.add_argument("--model", default="2D", choices=("2D", "axy"))
    pair.add_argument("--seed", type=int, default=1234)
    pair.add_argument("--petsc-matrix-mode", default="local", choices=("local", "serial"))
    pair.add_argument("--artifact-before", type=Path, default=None)
    pair.add_argument("--artifact-after", type=Path, default=None)
    _common_compare_arguments(pair)

    args = parser.parse_args(argv)
    if args.max_terms < 0:
        raise ValueError("--max-terms must be >= 0")
    if args.eig_modes <= 0:
        raise ValueError("--eig-modes must be > 0")

    if args.mode == "artifacts":
        report = _run_artifact_mode(args)
    else:
        report = _run_pair_mode(args)

    _write_json(args.output, report)
    print(
        "assembly parity:",
        f"mode={args.mode}",
        f"pass={report['pass']}",
        f"output={args.output}",
    )
    if not report["pass"] and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
