#!/usr/bin/env python3
"""Low-level MATLAB/Python internal parity for Song (2023) GetLinProblem.

This script compares:

- Derivative operators Dr, D2r, Dz, D2z, DR, D2R, DZ, D2Z, D2RZ
  exported by `GetLinProblem_export.m` (derivatives.mat)
  against the Python `Mesh` derivatives for the same grid.

- (Optionally) the pre-BC operator `L0_preBC_only` exported by
  `main_preBC_export.m` against the Python `build_lnse_operator` result.

Run from repo root, for example:

    python -m validation.song_2023.scripts.compare_internal \\
        --case-dir validation/song_2023 \\
        --matlab-internal validation/song_2023/reference/matlab/matlab_code/SolverMartini/MATLAB_exports_Ziming_Internal
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

from ..baseflow_loader import load_my_case


def _load_matlab_sparse_from_mat(data: dict, key: str) -> sp.csc_matrix:
    """Extract a sparse matrix from a loaded .mat dict."""
    if key not in data:
        raise KeyError(f"Key {key!r} not found in MATLAB file")
    mat = data[key]
    if sp.issparse(mat):
        return mat.asfptype().tocsc()
    if isinstance(mat, np.ndarray):
        return sp.csc_matrix(mat.astype(complex))
    raise TypeError(f"Unexpected type for {key}: {type(mat)}")


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
    same_shape = py_mat.shape == ref_mat.shape
    same_indptr = same_shape and np.array_equal(py_mat.indptr, ref_mat.indptr)
    same_indices = same_shape and np.array_equal(py_mat.indices, ref_mat.indices)

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


def compare_derivatives(
    case_dir: Path,
    matlab_internal_dir: Path,
    *,
    fd_order: int = 2,
    stride_ksi: int = 1,
    stride_eta: int = 1,
) -> None:
    """Compare MATLAB and Python derivative operators."""
    deriv_path = matlab_internal_dir / "derivatives.mat"
    mat_data = scipy.io.loadmat(deriv_path)

    # Load Python mesh for the same grid
    X, Y, baseflow, sponge = load_my_case(
        case_dir,
        Re=1.0,  # dummy, not used for derivatives
        Ma=1.0,
        Pr=1.0,
        kappa=1.4,
        T_0=300.0,
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

    # MATLAB stores scalar operators Dr,D2r,Dz,D2z and 5-block DR,D2R,DZ,D2Z,D2RZ
    Dr_mat = _load_matlab_sparse_from_mat(mat_data, "Dr")
    D2r_mat = _load_matlab_sparse_from_mat(mat_data, "D2r")
    Dz_mat = _load_matlab_sparse_from_mat(mat_data, "Dz")
    D2z_mat = _load_matlab_sparse_from_mat(mat_data, "D2z")
    DR_mat = _load_matlab_sparse_from_mat(mat_data, "DR")
    D2R_mat = _load_matlab_sparse_from_mat(mat_data, "D2R")
    DZ_mat = _load_matlab_sparse_from_mat(mat_data, "DZ")
    D2Z_mat = _load_matlab_sparse_from_mat(mat_data, "D2Z")
    D2RZ_mat = _load_matlab_sparse_from_mat(mat_data, "D2RZ")

    print("=== Derivative operator parity (scalar) ===")
    for name, py_op, ref_op in [
        ("Dr", mesh.Dy, Dr_mat),
        ("D2r", mesh.D2y, D2r_mat),
        ("Dz", mesh.Dx, Dz_mat),
        ("D2z", mesh.D2x, D2z_mat),
    ]:
        py_csc = py_op.tocsc()
        ref_csc = ref_op.tocsc()
        print(f"\n[{name}]")
        print(f"  Python: shape={py_csc.shape}, nnz={py_csc.nnz}")
        print(f"  MATLAB: shape={ref_csc.shape}, nnz={ref_csc.nnz}")
        struct = _structure_metrics(py_csc, ref_csc)
        print("  -- Structural --")
        print(f"    shape_match   : {struct['shape_match']}")
        print(f"    indptr_match  : {struct['indptr_match']}")
        print(f"    indices_match : {struct['indices_match']}")
        print(f"    pattern_match : {struct['pattern_match']}")
        print(f"    excess_nnz    : {struct['excess_nnz']} (Python only)")
        print(f"    missing_nnz   : {struct['missing_nnz']} (MATLAB only)")
        max_abs, max_rel, frob_rel = _parity_metrics(py_csc, ref_csc)
        print("  -- Values (py - matlab) --")
        print(f"    max_abs_error : {max_abs:.3e}")
        print(f"    max_rel_error : {max_rel:.3e}")
        print(f"    frob_rel_error: {frob_rel:.3e}")

    # 5-block operators: DR,D2R,DZ,D2Z,D2RZ
    ngp = mesh.ngp
    def _blk(name: str, op: sp.spmatrix) -> None:
        ref = op.tocsc()
        # Python: build 5×5 block as in assembly._assemble_from_prepared
        if name in {"DR", "D2R"}:
            base = mesh.Dy if name == "DR" else mesh.D2y
        elif name in {"DZ", "D2Z"}:
            base = mesh.Dx if name == "DZ" else mesh.D2x
        elif name == "D2RZ":
            base = mesh.Dyx
        else:
            raise ValueError(name)
        base_csc = base.tocsc()
        blocks = sp.block_diag([base_csc] * 5, format="csc")

        print(f"\n[{name} (5-block)]")
        print(f"  Python: shape={blocks.shape}, nnz={blocks.nnz}")
        print(f"  MATLAB: shape={ref.shape}, nnz={ref.nnz}")
        struct = _structure_metrics(blocks, ref)
        print("  -- Structural --")
        print(f"    shape_match   : {struct['shape_match']}")
        print(f"    indptr_match  : {struct['indptr_match']}")
        print(f"    indices_match : {struct['indices_match']}")
        print(f"    pattern_match : {struct['pattern_match']}")
        print(f"    excess_nnz    : {struct['excess_nnz']} (Python only)")
        print(f"    missing_nnz   : {struct['missing_nnz']} (MATLAB only)")
        max_abs, max_rel, frob_rel = _parity_metrics(blocks, ref)
        print("  -- Values (py - matlab) --")
        print(f"    max_abs_error : {max_abs:.3e}")
        print(f"    max_rel_error : {max_rel:.3e}")
        print(f"    frob_rel_error: {frob_rel:.3e}")

    print("\n=== Derivative operator parity (5-block) ===")
    _blk("DR", DR_mat)
    _blk("D2R", D2R_mat)
    _blk("DZ", DZ_mat)
    _blk("D2Z", D2Z_mat)
    _blk("D2RZ", D2RZ_mat)


def compare_L0_preBC(
    case_dir: Path,
    matlab_internal_dir: Path,
    *,
    Re: float,
    Ma: float,
    Pr: float,
    kappa: float,
    T_0: float,
    beta: float,
    fd_order: int,
    stride_ksi: int,
    stride_eta: int,
) -> None:
    """Compare pre-BC L0 from MATLAB (L0_preBC_only) and Python."""
    mat_path = matlab_internal_dir / "L0_preBC_only.mat"
    data = scipy.io.loadmat(mat_path)
    if "L0_preBC" not in data:
        raise KeyError(f"'L0_preBC' not found in {mat_path}")
    L0_mat = _load_matlab_sparse_from_mat(data, "L0_preBC")

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

    L0_py, idx = build_lnse_operator(mesh, baseflow, kx=beta, model="2D")
    L0_py = L0_py.tocsc()

    print("\n=== Pre-BC L0 parity (raw GetLinProblem vs build_lnse_operator) ===")
    print(f"Python: shape={L0_py.shape}, nnz={L0_py.nnz}")
    print(f"MATLAB: shape={L0_mat.shape}, nnz={L0_mat.nnz}")

    struct = _structure_metrics(L0_py, L0_mat)
    print("  -- Structural --")
    print(f"    shape_match   : {struct['shape_match']}")
    print(f"    indptr_match  : {struct['indptr_match']}")
    print(f"    indices_match : {struct['indices_match']}")
    print(f"    pattern_match : {struct['pattern_match']}")
    print(f"    excess_nnz    : {struct['excess_nnz']} (Python only)")
    print(f"    missing_nnz   : {struct['missing_nnz']} (MATLAB only)")

    max_abs, max_rel, frob_rel = _parity_metrics(L0_py, L0_mat)
    print("  -- Values (py - matlab) --")
    print(f"    max_abs_error : {max_abs:.3e}")
    print(f"    max_rel_error : {max_rel:.3e}")
    print(f"    frob_rel_error: {frob_rel:.3e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MATLAB internal GetLinProblem exports with Python internals for Song (2023)."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("validation/song_2023"),
        help="Directory containing song_2023 inputs (baseflow .npy fields).",
    )
    parser.add_argument(
        "--matlab-internal",
        type=Path,
        default=Path(
            "validation/song_2023/reference/matlab/matlab_code/SolverMartini/MATLAB_exports_Ziming_Internal"
        ),
        help="Directory produced by GetLinProblem_export/main_preBC_export.m.",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compare_derivatives(
        case_dir=args.case_dir,
        matlab_internal_dir=args.matlab_internal,
        fd_order=args.fd_order,
        stride_ksi=args.stride_ksi,
        stride_eta=args.stride_eta,
    )
    compare_L0_preBC(
        case_dir=args.case_dir,
        matlab_internal_dir=args.matlab_internal,
        Re=args.Re,
        Ma=args.Ma,
        Pr=args.Pr,
        kappa=args.kappa,
        T_0=args.T0,
        beta=args.beta,
        fd_order=args.fd_order,
        stride_ksi=args.stride_ksi,
        stride_eta=args.stride_eta,
    )


if __name__ == "__main__":
    main()
