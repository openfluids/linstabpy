"""Regression parity test for Case 1 operator assembly vs MATLAB exports.

This test intentionally checks matrix construction only (no eigensolve) so it
stays fast while guarding the core MATLAB/Python operator equivalence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

h5py = pytest.importorskip("h5py")

from linstab2d.baseflow import load_hamada_case, sutherland_air
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.boundary_conditions.neumann import apply_neumann
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator
from tests._paths import case_fixture


def _load_matlab_sparse(path: Path, key: str) -> sp.csc_matrix:
    with h5py.File(path, "r") as handle:
        group = handle[key]
        data = np.array(group["data"])
        indices = np.array(group["ir"], dtype=np.int64)
        indptr = np.array(group["jc"], dtype=np.int64)
    n = indptr.size - 1
    return sp.csc_matrix((data, indices, indptr), shape=(n, n))


def _parity_metrics(py_mat: sp.csc_matrix, ref_mat: sp.csc_matrix) -> tuple[float, float, float]:
    delta = (py_mat - ref_mat).tocoo()
    if delta.nnz == 0:
        return 0.0, 0.0, 0.0

    abs_delta = np.abs(delta.data)
    max_abs = float(abs_delta.max())
    max_ref = max(float(np.max(np.abs(ref_mat.data))) if ref_mat.nnz else 0.0, 1e-30)
    max_rel = max_abs / max_ref
    frob_rel = float(np.linalg.norm(delta.data) / max(np.linalg.norm(ref_mat.data), 1e-30))
    return max_abs, max_rel, frob_rel


def test_case1_operator_stages_match_matlab_exports() -> None:
    case_dir = case_fixture("case1")
    baseflow_dir = case_dir / "data" / "baseflow"
    exports_dir = case_dir / "reference" / "matlab" / "MATLAB_exports"

    if not exports_dir.exists():
        pytest.skip(f"Missing MATLAB exports directory: {exports_dir}")

    matlab_stages = {
        "pre": _load_matlab_sparse(exports_dir / "L0_preBC.mat", "L0_preBC"),
        "after_lt": _load_matlab_sparse(exports_dir / "L0_after_dirichlet_lt.mat", "L0_after_dirichlet_lt"),
        "after_b": _load_matlab_sparse(exports_dir / "L0_after_dirichlet_b.mat", "L0_after_dirichlet_b"),
        "after_nb": _load_matlab_sparse(exports_dir / "L0_after_neumann_bT.mat", "L0_after_neumann_bT"),
        "post": _load_matlab_sparse(exports_dir / "L0_postBC.mat", "L0_postBC"),
    }

    X, Y, baseflow, sponge = load_hamada_case(
        baseflow_dir,
        Re=395000.0,
        Ma=2.0,
        Pr=0.747,
        kappa=1.4,
        T_0=1540.0,
        stride_ksi=1,
        stride_eta=1,
        transpose_input=True,
        zero_spanwise=False,
    )
    sutherland_air(baseflow)

    # MATLAB reference export path consumes uppercase dMUdT/d2MUdT2 fields.
    # In this dataset those are zero; we mirror that for strict parity mode.
    baseflow.dmudT[:] = 0.0
    baseflow.d2mudT2[:] = 0.0

    mesh = Mesh.create((0.0, 1.0), (0.0, 0.5), nx=X.shape[1], ny=X.shape[0], fd_order=4)
    mesh = mesh.deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge

    L0_pre, idx = build_lnse_operator(mesh, baseflow, kx=0.0, model="2D")
    L0_lt, _ = apply_dirichlet(L0_pre, idx, "lt", "ruvwT", mesh.ngp, penalty=1e8)
    L0_b, _ = apply_dirichlet(L0_lt, idx, "b", "ruvw", mesh.ngp, penalty=1e8)
    L0_nb, _ = apply_neumann(L0_b, mesh, idx, "b", "T", penalty=1e8)
    L0_post, _ = apply_neumann(L0_nb, mesh, idx, "r", "ruvwT", penalty=1e8)

    python_stages = {
        "pre": L0_pre.tocsc(),
        "after_lt": L0_lt.tocsc(),
        "after_b": L0_b.tocsc(),
        "after_nb": L0_nb.tocsc(),
        "post": L0_post.tocsc(),
    }

    for name, py_mat in python_stages.items():
        ref_mat = matlab_stages[name].tocsc()

        assert py_mat.shape == ref_mat.shape, f"{name}: shape mismatch"
        assert np.array_equal(py_mat.indptr, ref_mat.indptr), f"{name}: CSC indptr mismatch"
        assert np.array_equal(py_mat.indices, ref_mat.indices), f"{name}: CSC indices mismatch"

        max_abs, max_rel, frob_rel = _parity_metrics(py_mat, ref_mat)

        # Neumann-enforced rows involve large penalty scales; for robustness we gate
        # those by relative matrix errors. Earlier stages should also be absolutely
        # close in this reference case.
        assert max_rel < 1e-8, f"{name}: max_rel={max_rel:.3e}"
        assert frob_rel < 1e-8, f"{name}: frob_rel={frob_rel:.3e}"
        if name in {"pre", "after_lt", "after_b"}:
            assert max_abs < 1e-3, f"{name}: max_abs={max_abs:.3e}"
