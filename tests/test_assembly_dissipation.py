"""Phase 1 Step 4 gate tests for dissipation wiring in the assembler.

T1 (flag off): assembler with epsilon=0.0 is bitwise identical to baseline.
T1+ (positive correctness): with epsilon=0.1, the difference is exactly
    epsilon * I_5 (x) Q_diss_2D before BCs are applied.

Comparison notes
----------------
build_lnse_operator does NOT apply BCs.  Dirichlet rows are therefore not
involved in these tests; the comparison is on the raw pre-BC operator.
This is option (b) from the task description: the gate is on the post-BC-free
(pre-BC) operator, and Dirichlet rows are identical by construction anyway.

All heavy imports (Mesh, BaseFlow, build_lnse_operator) are done lazily inside
each test method to avoid module-level petsc4py/mpi4py import failures on
machines without PETSc (e.g. macOS developer machines).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest

_NX = 20
_NY = 15
_FD_ORDER = 4


def _make_mesh(enabled: bool = False):
    from linstab2d.mesh import Mesh
    od = {"enabled": enabled, "order_diss": _FD_ORDER} if enabled else None
    return Mesh.create(
        xrange=(0.0, 1.0),
        yrange=(0.0, 1.0),
        nx=_NX,
        ny=_NY,
        fd_order=_FD_ORDER,
        numerics_operator_dissipation=od,
    )


def _make_baseflow(mesh):
    from linstab2d.baseflow import BaseFlow, sutherland_air
    ny, nx = mesh.X.shape
    ones = np.ones((ny, nx))
    zeros = np.zeros((ny, nx))
    bf = BaseFlow(
        Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15,
        RHO=ones.copy(), U=ones.copy(), V=zeros.copy(),
        W=zeros.copy(), T=ones.copy(),
    )
    sutherland_air(bf)
    return bf


class TestDissipationT1FlagOff:
    """T1 gate: epsilon=0 path is bitwise identical to baseline (no Q_diss on mesh)."""

    def test_no_kwarg_identical_to_explicit_zero(self) -> None:
        """build_lnse_operator with no dissipation_epsilon kwarg equals epsilon=0.0 explicitly."""
        from linstab2d.operators import build_lnse_operator
        mesh = _make_mesh(enabled=False)
        bf = _make_baseflow(mesh)
        L_no_kwarg, _ = build_lnse_operator(mesh, bf)
        L_zero, _ = build_lnse_operator(mesh, bf, dissipation_epsilon=0.0)
        diff = (L_no_kwarg - L_zero).tocsr()
        max_diff = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
        assert diff.nnz == 0 or max_diff < 1e-14, (
            f"T1 default vs explicit-zero: max diff = {max_diff:.2e}"
        )

    def test_diss_mesh_epsilon_zero_identical_to_nodiss_mesh(self) -> None:
        """Mesh WITH Q_diss populated but epsilon=0.0 yields same operator as mesh without Q_diss.

        This is the load-bearing T1 gate: when epsilon=0.0 (default), the
        if-guard in build_lnse_operator is False, so no dissipation is applied
        and the operator is bitwise identical to the baseline path.
        Both meshes have identical grid parameters so the LNSE operators must match.
        """
        from linstab2d.operators import build_lnse_operator
        mesh_nodiss = _make_mesh(enabled=False)
        mesh_diss = _make_mesh(enabled=True)
        bf_nodiss = _make_baseflow(mesh_nodiss)
        bf_diss = _make_baseflow(mesh_diss)

        L_baseline, _ = build_lnse_operator(mesh_nodiss, bf_nodiss)
        L_off, _ = build_lnse_operator(mesh_diss, bf_diss, dissipation_epsilon=0.0)

        diff = (L_baseline - L_off).tocsr()
        if diff.nnz > 0:
            max_diff = float(np.max(np.abs(diff.data)))
            assert max_diff < 1e-14, (
                f"T1 FAILED: max entrywise diff = {max_diff:.2e} with epsilon=0.0 "
                "(dissipation path polluted the default assembly)"
            )


class TestDissipationPositiveCorrectness:
    """Positive correctness: L_on = L_baseline - epsilon * I_5 (x) Q_diss_2D."""

    def test_epsilon_subtracts_correct_block(self) -> None:
        """With epsilon=0.1, L_on - L_baseline == -0.1 * Q_diss_block entrywise (tol=1e-12).

        Verifies the kron order: Q_diss_x (nx,nx) is the outer factor,
        Q_diss_y (ny,ny) is the inner factor — matching the Dx_g = kron(D1x, I_ny)
        convention used in _build_operators_kron.
        """
        from linstab2d.operators import build_lnse_operator
        epsilon = 0.1
        mesh = _make_mesh(enabled=True)
        bf = _make_baseflow(mesh)

        L_baseline, _ = build_lnse_operator(mesh, bf, dissipation_epsilon=0.0)
        L_on, _ = build_lnse_operator(mesh, bf, dissipation_epsilon=epsilon)

        # Build Q_diss_block using the same kron convention as the assembler.
        ny, nx = mesh.X.shape
        I_ny = sp.eye(ny, format="csr")
        I_nx = sp.eye(nx, format="csr")
        Q_2D = (
            sp.kron(mesh.Q_diss_x, I_ny, format="csr")
            + sp.kron(I_nx, mesh.Q_diss_y, format="csr")
        )
        Q_block = sp.block_diag([Q_2D] * 5, format="csr")

        L_expected = L_baseline - epsilon * Q_block
        diff = (L_on - L_expected).tocsr()
        if diff.nnz > 0:
            max_diff = float(np.max(np.abs(diff.data)))
            assert max_diff < 1e-12, (
                f"Correctness FAILED: max entrywise diff = {max_diff:.2e} "
                "(sign error, wrong kron order, or wrong block structure)"
            )

    def test_epsilon_nonzero_changes_operator(self) -> None:
        """Sanity: epsilon=0.1 with Q_diss populated must change the operator."""
        from linstab2d.operators import build_lnse_operator
        mesh = _make_mesh(enabled=True)
        bf = _make_baseflow(mesh)

        L_baseline, _ = build_lnse_operator(mesh, bf, dissipation_epsilon=0.0)
        L_on, _ = build_lnse_operator(mesh, bf, dissipation_epsilon=0.1)

        diff = (L_baseline - L_on).tocsr()
        assert diff.nnz > 0, (
            "epsilon=0.1 did not change the operator — Q_diss was not subtracted"
        )

    def test_epsilon_positive_no_dissipation_matrices_unchanged(self) -> None:
        """If mesh has no Q_diss (flag off), epsilon > 0 must not change the operator."""
        from linstab2d.operators import build_lnse_operator
        mesh_nodiss = _make_mesh(enabled=False)
        bf = _make_baseflow(mesh_nodiss)

        L_baseline, _ = build_lnse_operator(mesh_nodiss, bf, dissipation_epsilon=0.0)
        L_with_eps, _ = build_lnse_operator(mesh_nodiss, bf, dissipation_epsilon=0.1)

        diff = (L_baseline - L_with_eps).tocsr()
        max_diff = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
        assert diff.nnz == 0 or max_diff < 1e-14, (
            f"epsilon > 0 with no Q_diss unexpectedly changed the operator: "
            f"max diff = {max_diff:.2e}"
        )


def test_petsc_backend_rejects_dissipation():
    """build_lnse_operator_petsc must raise NotImplementedError when dissipation_epsilon > 0."""
    pytest.importorskip("petsc4py", reason="petsc4py not available")
    from linstab2d.operators.assembly import build_lnse_operator_petsc
    from linstab2d.mesh import Mesh
    from linstab2d.baseflow import BaseFlow, sutherland_air
    import pytest as _pytest

    mesh = Mesh.create(
        xrange=(0.0, 1.0),
        yrange=(0.0, 1.0),
        nx=10,
        ny=8,
        fd_order=4,
        numerics_operator_dissipation={"enabled": True, "order_diss": 4},
    )
    ny, nx = mesh.X.shape
    ones = np.ones((ny, nx))
    zeros = np.zeros((ny, nx))
    bf = BaseFlow(
        Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15,
        RHO=ones.copy(), U=ones.copy(), V=zeros.copy(),
        W=zeros.copy(), T=ones.copy(),
    )
    sutherland_air(bf)

    with _pytest.raises(NotImplementedError, match="dissipation_epsilon"):
        build_lnse_operator_petsc(mesh, bf, dissipation_epsilon=1.0)
