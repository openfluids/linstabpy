#!/usr/bin/env python3
"""Compare MATLAB and Python L0 operators for the Song (2023) case.

Usage (from repo root):

    python -m validation.song_2023.scripts.compare_L0 \
        --case-dir validation/song_2023 \
        --matlab-L0 validation/song_2023/reference/matlab/matlab_reference/L0.mat

This script:
  1. Loads the MATLAB `L0` matrix saved by the MATLAB reference code
     (after BCs and after division by -1i).
  2. Rebuilds the corresponding Python operator for the same baseflow and BC
     recipe, then applies the same temporal scaling (i * L0).
  3. Reports structural and value-wise metrics so you can see how close they are.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io
import scipy.sparse as sp

from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator

from ..baseflow_loader import load_my_case
from ..bc_setup import my_bc_setup


def _load_matlab_L0(path: Path) -> sp.csc_matrix:
    """Load MATLAB `L0` from a `.mat` file and return as CSC matrix.

    The MATLAB reference code saves the post-BC temporal operator via:

        L0 = L0/-1i;
        save('L0.mat', 'L0')

    This is normally a v7 MAT-file, which `scipy.io.loadmat` can read directly.
    """
    data = scipy.io.loadmat(path)
    if "L0" not in data:
        raise KeyError(f"'L0' not found in MAT-file {path}")

    L0_mat = data["L0"]
    # Ensure complex dtype so we never silently drop imaginary parts.
    if sp.issparse(L0_mat):
        return L0_mat.asfptype().tocsc()

    # Dense numeric array (real or complex)
    if not isinstance(L0_mat, np.ndarray):
        raise TypeError(f"Unexpected type for L0 in MAT-file: {type(L0_mat)}")
    return sp.csc_matrix(L0_mat.astype(complex))


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


def _structure_metrics(py_mat: sp.csc_matrix, ref_mat: sp.csc_matrix) -> dict:
    """Check basic structural parity between two CSC matrices."""
    same_shape = py_mat.shape == ref_mat.shape
    same_indptr = same_shape and np.array_equal(py_mat.indptr, ref_mat.indptr)
    same_indices = same_shape and np.array_equal(py_mat.indices, ref_mat.indices)

    # Pattern-only comparison when structure differs: where do nnz patterns disagree?
    if not (same_shape and same_indptr and same_indices):
        # Work in complex to avoid discarding imaginary parts; we only care
        # about the sparsity pattern, so set all nonzeros to 1+0j.
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


def compare_L0(
    case_dir: Path,
    matlab_L0_path: Path,
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
    """Build Python L0 for song_2023 and compare to MATLAB L0.mat."""
    matlab_L0 = _load_matlab_L0(matlab_L0_path)

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

    # Evolution operator dq/dt = L0_evol q
    L0_evol, idx = build_lnse_operator(mesh, baseflow, kx=beta, model="2D")
    L0_evol_bc, _ = my_bc_setup(mesh, idx, L0_evol, penalty)

    # MATLAB saves the temporal operator: L0_matlab = (1/(-1i)) * L0_evol_bc.
    # Since 1/(-1i) = i, this equals i * L0_evol_bc, which is exactly what
    # Python uses inside `temporal_eigenvalues`. So compare:
    L0_py_temporal = (1j * L0_evol_bc).tocsc()

    matlab_L0 = matlab_L0.tocsc()

    print("=== song_2023 L0 parity check ===")
    print(f"Python temporal L0 shape: {L0_py_temporal.shape}, nnz={L0_py_temporal.nnz}")
    print(f"MATLAB L0 shape        : {matlab_L0.shape}, nnz={matlab_L0.nnz}")

    struct = _structure_metrics(L0_py_temporal, matlab_L0)
    print("\n-- Structural comparison --")
    print(f"  shape_match   : {struct['shape_match']}")
    print(f"  indptr_match  : {struct['indptr_match']}")
    print(f"  indices_match : {struct['indices_match']}")
    print(f"  pattern_match : {struct['pattern_match']}")
    print(f"  excess_nnz    : {struct['excess_nnz']} (present in Python only)")
    print(f"  missing_nnz   : {struct['missing_nnz']} (present in MATLAB only)")

    max_abs, max_rel, frob_rel = _parity_metrics(L0_py_temporal, matlab_L0)
    print("\n-- Value comparison (L0_py_temporal - L0_matlab) --")
    print(f"  max_abs_error : {max_abs:.3e}")
    print(f"  max_rel_error : {max_rel:.3e}")
    print(f"  frob_rel_error: {frob_rel:.3e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MATLAB L0.mat and Python-built L0 for song_2023."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("validation/song_2023"),
        help="Directory containing song_2023 inputs (X/Y/baseflow/sponge).",
    )
    parser.add_argument(
        "--matlab-L0",
        type=Path,
        default=Path("validation/song_2023/reference/matlab/matlab_reference/L0.mat"),
        help="Path to MATLAB L0.mat saved by the MATLAB reference code.",
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
    compare_L0(
        case_dir=args.case_dir,
        matlab_L0_path=args.matlab_L0,
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
