#!/usr/bin/env python3
"""Compare MATLAB and Python operator stages for the Song (2023) case.

This mirrors the Hamada stage-parity tooling, but for Song (2023):

- MATLAB stages are exported by `main_export_stages.m` into
  `matlab_code/SolverMartini/MATLAB_exports_Ziming/`:

    L0_preBC.mat
    L0_after_dirichlet_lt.mat
    L0_after_dirichlet_b.mat
    L0_after_neumann_bT.mat
    L0_postBC.mat

- Python stages are rebuilt using:

    build_lnse_operator(mesh, baseflow, kx=beta, model="2D")
    apply_dirichlet / apply_neumann with the same recipe as MATLAB.

Run from repo root, for example:

    python -m validation.song_2023.compare_stages \\
        --case-dir validation/song_2023 \\
        --matlab-exports validation/song_2023/matlab_code/SolverMartini/MATLAB_exports_Ziming
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import scipy.io
import scipy.sparse as sp

from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.boundary_conditions.neumann import apply_neumann

from .baseflow_loader import load_my_case


def _load_matlab_sparse(path: Path, key: str) -> sp.csc_matrix:
    """Load a sparse matrix from a MATLAB .mat file and return as CSC.

    The exporter saves variables like `L0_preBC` using -v7 (non-HDF5).
    scipy.io.loadmat will return either a dense ndarray or a scipy.sparse
    matrix for MATLAB sparse; we normalise to csc_matrix with complex dtype.
    """
    data = scipy.io.loadmat(path)
    if key not in data:
        raise KeyError(f"Key {key!r} not found in MAT-file {path}")
    mat = data[key]
    if sp.issparse(mat):
        return mat.asfptype().tocsc()
    if not isinstance(mat, np.ndarray):
        raise TypeError(f"Unexpected type for {key} in {path}: {type(mat)}")
    return sp.csc_matrix(mat.astype(complex))


def _parity_metrics(py_mat: sp.csc_matrix, ref_mat: sp.csc_matrix) -> Tuple[float, float, float]:
    """Return (max_abs, max_rel, frob_rel) for py_mat - ref_mat."""
    delta = (py_mat - ref_mat).tocoo()
    if delta.nnz == 0:
        return 0.0, 0.0, 0.0

    abs_delta = np.abs(delta.data)
    max_abs = float(abs_delta.max())
    max_ref = max(float(np.max(np.abs(ref_mat.data))) if ref_mat.nnz else 0.0, 1e-30)
    max_rel = max_abs / max_ref
    frob_rel = float(np.linalg.norm(delta.data) / max(np.linalg.norm(ref_mat.data), 1e-30))
    return max_abs, max_rel, frob_rel


def _structure_metrics(py_mat: sp.csc_matrix, ref_mat: sp.csc_matrix) -> dict[str, Any]:
    """Check basic structural parity between two CSC matrices."""
    same_shape = py_mat.shape == ref_mat.shape
    same_indptr = same_shape and np.array_equal(py_mat.indptr, ref_mat.indptr)
    same_indices = same_shape and np.array_equal(py_mat.indices, ref_mat.indices)

    # Pattern-only comparison when structure differs: where do nnz patterns disagree?
    if not (same_shape and same_indptr and same_indices):
        py_bin = py_mat.copy().astype(complex)
        ref_bin = ref_mat.copy().astype(complex)
        py_bin.data[:] = 1.0 + 0.0j
        ref_bin.data[:] = 1.0 + 0.0j
        pattern_delta = (py_bin - ref_bin).tocoo()
        excess_nnz = int(np.count_nonzero(pattern_delta.data > 0.0))
        missing_nnz = int(np.count_nonzero(pattern_delta.data < 0.0))
    else:
        excess_nnz = 0
        missing_nnz = 0

    return {
        "shape_match": bool(same_shape),
        "indptr_match": bool(same_indptr),
        "indices_match": bool(same_indices),
        "pattern_match": bool(same_shape and same_indptr and same_indices),
        "excess_nnz": excess_nnz,
        "missing_nnz": missing_nnz,
    }


def compare_stages(
    case_dir: Path,
    matlab_exports_dir: Path,
    *,
    Re: float = 1.0e5,
    Ma: float = 2.15,
    Pr: float = 0.72,
    kappa: float = 1.4,
    T_0: float = 153.33,
    beta: float = 19.2,
    fd_order: int = 2,
    stride_ksi: int = 1,
    stride_eta: int = 1,
    penalty: float = 1e8,
) -> None:
    """Compare each Song (2023) operator stage between MATLAB and Python."""
    # Load MATLAB stages
    stages_matlab = {
        "pre": _load_matlab_sparse(matlab_exports_dir / "L0_preBC.mat", "L0_preBC"),
        "after_lt": _load_matlab_sparse(
            matlab_exports_dir / "L0_after_dirichlet_lt.mat", "L0_after_dirichlet_lt"
        ),
        "after_b": _load_matlab_sparse(
            matlab_exports_dir / "L0_after_dirichlet_b.mat", "L0_after_dirichlet_b"
        ),
        "after_nb": _load_matlab_sparse(
            matlab_exports_dir / "L0_after_neumann_bT.mat", "L0_after_neumann_bT"
        ),
        "post": _load_matlab_sparse(matlab_exports_dir / "L0_postBC.mat", "L0_postBC"),
    }

    # Build Python operator stages using the same recipe as MATLAB main_export_stages.m
    X, Y, baseflow, sponge = load_my_case(
        case_dir,
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        T_0=T_0,
        stride_ksi=stride_ksi,
        stride_eta=stride_eta,
    )

    ny, nx = X.shape
    mesh = Mesh.create(
        (0.0, 2.0),
        (0.0, 1.0),
        nx=nx,
        ny=ny,
        fd_order=fd_order,
    )
    mesh = mesh.deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge

    # Stage: pre-BC
    L0_pre, idx = build_lnse_operator(mesh, baseflow, kx=beta, model="2D")
    L0_pre = L0_pre.tocsc()

    # Stage: after Dirichlet on left+top (borders='lt', vars='ruvwT')
    L0_lt, _ = apply_dirichlet(
        L0_pre,
        idx,
        borders="lt",
        variables="ruvwT",
        ngp=mesh.ngp,
        penalty=penalty,
    )
    L0_lt = L0_lt.tocsc()

    # Stage: after Dirichlet on bottom (borders='b', vars='ruvw')
    L0_b, _ = apply_dirichlet(
        L0_lt,
        idx,
        borders="b",
        variables="ruvw",
        ngp=mesh.ngp,
        penalty=penalty,
    )
    L0_b = L0_b.tocsc()

    # Stage: after Neumann on bottom for T only (borders='b', vars='T')
    L0_nb, _ = apply_neumann(
        L0_b,
        mesh,
        idx,
        borders="b",
        variables="T",
        penalty=penalty,
    )
    L0_nb = L0_nb.tocsc()

    # Stage: final after Neumann on right for r,u,v,w,T (borders='r', vars='ruvwT')
    L0_post, _ = apply_neumann(
        L0_nb,
        mesh,
        idx,
        borders="r",
        variables="ruvwT",
        penalty=penalty,
    )
    L0_post = L0_post.tocsc()

    stages_python = {
        "pre": L0_pre,
        "after_lt": L0_lt,
        "after_b": L0_b,
        "after_nb": L0_nb,
        "post": L0_post,
    }

    print("=== song_2023 operator stage parity ===")
    for name in ("pre", "after_lt", "after_b", "after_nb", "post"):
        py_mat = stages_python[name]
        ref_mat = stages_matlab[name].tocsc()

        print(f"\n[Stage: {name}]")
        print(f"  Python: shape={py_mat.shape}, nnz={py_mat.nnz}")
        print(f"  MATLAB: shape={ref_mat.shape}, nnz={ref_mat.nnz}")

        struct = _structure_metrics(py_mat, ref_mat)
        print("  -- Structural --")
        print(f"    shape_match   : {struct['shape_match']}")
        print(f"    indptr_match  : {struct['indptr_match']}")
        print(f"    indices_match : {struct['indices_match']}")
        print(f"    pattern_match : {struct['pattern_match']}")
        print(f"    excess_nnz    : {struct['excess_nnz']} (Python only)")
        print(f"    missing_nnz   : {struct['missing_nnz']} (MATLAB only)")

        max_abs, max_rel, frob_rel = _parity_metrics(py_mat, ref_mat)
        print("  -- Values (py - matlab) --")
        print(f"    max_abs_error : {max_abs:.3e}")
        print(f"    max_rel_error : {max_rel:.3e}")
        print(f"    frob_rel_error: {frob_rel:.3e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MATLAB and Python operator stages for song_2023."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("validation/song_2023"),
        help="Directory containing song_2023 inputs (baseflow .npy fields).",
    )
    parser.add_argument(
        "--matlab-exports",
        type=Path,
        default=Path("validation/song_2023/matlab_code/SolverMartini/MATLAB_exports_Ziming"),
        help="Directory with MATLAB-exported operator stages from main_export_stages.m.",
    )
    parser.add_argument("--Re", type=float, default=1.0e5)
    parser.add_argument("--Ma", type=float, default=2.15)
    parser.add_argument("--Pr", type=float, default=0.72)
    parser.add_argument("--kappa", type=float, default=1.4)
    parser.add_argument("--T0", type=float, default=153.33)
    parser.add_argument("--beta", type=float, default=19.2)
    parser.add_argument("--fd-order", type=int, default=2)
    parser.add_argument("--stride-ksi", type=int, default=1)
    parser.add_argument("--stride-eta", type=int, default=1)
    parser.add_argument(
        "--penalty",
        type=float,
        default=1e8,
        help="Penalty parameter used for Dirichlet/Neumann BCs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compare_stages(
        case_dir=args.case_dir,
        matlab_exports_dir=args.matlab_exports,
        Re=args.Re,
        Ma=args.Ma,
        Pr=args.Pr,
        kappa=args.kappa,
        T_0=args.T0,
        beta=args.beta,
        fd_order=args.fd_order,
        stride_ksi=args.stride_ksi,
        stride_eta=args.stride_eta,
        penalty=args.penalty,
    )


if __name__ == "__main__":
    main()

