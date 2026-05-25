# SPDX-License-Identifier: GPL-3.0-only
"""Tests for resolvent and eigenvalue solvers.

All tests in this file are marked @pytest.mark.slow.
"""

import json
import shutil
import textwrap

from tests.conftest import build_synthetic_jet
from tests.mpi_helpers import run_mpi_python, parse_mpi_reports, assert_split_ownership

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

from linstabpy.mesh import Mesh
from linstabpy.baseflow import BaseFlow, sutherland_air, load_blasius
from linstabpy.operators import (
    build_lnse_operator,
    build_lnse_operator_petsc,
    build_lnse_system,
    compute_chu_norm,
    build_spatial_operator,
    build_spatial_operator_petsc,
)
from linstabpy.boundary_conditions import (
    apply_bc_recipe,
    apply_bc_recipe_petsc,
    apply_dirichlet,
    apply_dirichlet_petsc,
    apply_dirichlet_spatial,
    apply_dirichlet_spatial_petsc,
)
from linstabpy.solvers import (
    resolvent_analysis,
    temporal_eigenvalues,
    adjoint_eigenvalues,
    spatial_eigenvalues,
)
from linstabpy.solvers.resolvent import compute_resolvent_svd_residuals
from linstabpy.solvers.eigenvalue import (
    _resolve_ncv as _resolve_temporal_ncv,
    _sort_nearest as _sort_temporal_nearest,
)
from linstabpy.solvers.spatial_eig import _resolve_ncv as _resolve_spatial_ncv
from linstabpy.solvers.linsolve import make_linear_solver
from linstabpy.solvers.petsc_utils import scipy_to_petsc


def _assert_temporal_spectrum_allclose(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    actual = np.asarray(actual, dtype=complex).reshape(-1)
    expected = np.asarray(expected, dtype=complex).reshape(-1)
    assert actual.shape == expected.shape

    direct_error = np.abs(actual[:, None] - expected[None, :])
    mirror_error = np.abs(actual[:, None] + np.conjugate(expected[None, :]))
    cost = np.minimum(direct_error, mirror_error)
    rows, cols = linear_sum_assignment(cost)
    errors = cost[rows, cols]
    tolerance = atol + rtol * np.abs(expected[cols])

    assert np.all(errors <= tolerance), (
        "Temporal spectra differ beyond tolerance, allowing real-baseflow "
        f"mirror partners. actual={actual}, expected={expected}, "
        f"errors={errors}, tolerance={tolerance}"
    )


def _build_linsolve_mpi_script() -> str:
    return textwrap.dedent(
        """
        import json
        import numpy as np
        from linstabpy.mesh import Mesh
        from linstabpy.baseflow import BaseFlow, sutherland_air
        from linstabpy.operators import build_lnse_system
        from linstabpy.solvers.linsolve import make_linear_solver
        import linstabpy.utils.petsc_core as petsc_core

        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=10,
            ny=8,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=8e3,
            Ma=0.3,
            Pr=0.7,
            kappa=1.4,
            T_0=290.0,
            RHO=1.0 + 0.02 * np.cos(np.pi * X) * np.cos(0.75 * np.pi * Y),
            U=0.9 + 0.09 * np.sin(np.pi * X) * np.exp(-0.5 * Y),
            V=0.02 * np.cos(0.5 * np.pi * X) * np.sin(np.pi * Y),
            W=0.01 * np.sin(np.pi * X) * np.cos(0.5 * np.pi * Y),
            T=1.0 + 0.02 * np.cos(0.8 * np.pi * X) * np.exp(-0.3 * Y),
        )
        sutherland_air(bf)

        recipe = [{"type": "dirichlet", "borders": "bt", "variables": "uvwT"}]

        def _unexpected_fallback(*args, **kwargs):
            raise AssertionError("SciPy-to-PETSc fallback should not be used for supported distributed 2D direct assembly")

        petsc_core.scipy_to_petsc = _unexpected_fallback

        L0_ref, _, _ = build_lnse_system(
            mesh,
            bf,
            kx=0.2,
            model="2D",
            bc_recipe=recipe,
            solver_cfg={"petsc_matrix_mode": "local"},
        )
        L0_sys, _, _ = build_lnse_system(
            mesh,
            bf,
            kx=0.2,
            model="2D",
            bc_recipe=recipe,
            backend="petsc",
            solver_cfg={"petsc_matrix_mode": "distributed"},
        )

        shift = 5.0 + 20.0j
        L_sigma_ref = L0_ref.copy()
        L_sigma_ref.setdiag(L_sigma_ref.diagonal() - shift)
        L_sigma_sys = L0_sys.copy()
        L_sigma_sys.shift(-shift)
        L_sigma_sys.assemble()

        try:
            solver_ref = make_linear_solver(L_sigma_ref)
            solver_sys = make_linear_solver(L_sigma_sys, solver_cfg={"petsc_matrix_mode": "distributed"})

            rng = np.random.default_rng(17)
            rhs = rng.standard_normal(L0_ref.shape[0]) + 1j * rng.standard_normal(L0_ref.shape[0])
            expected = solver_ref.solve(rhs)
            expected_adj = solver_ref.solve_transpose(rhs)

            b = L_sigma_sys.createVecLeft()
            r0, r1 = b.getOwnershipRange()
            if r1 > r0:
                rows = np.arange(r0, r1, dtype=np.int32)
                b.setValues(rows, rhs[r0:r1])
            b.assemblyBegin()
            b.assemblyEnd()

            x = solver_sys.solve_vec(b)
            x_adj = solver_sys.solve_transpose_vec(b)
            try:
                np.testing.assert_allclose(np.array(x.array_r, copy=True), expected[r0:r1], atol=1e-8, rtol=1e-8)
                np.testing.assert_allclose(np.array(x_adj.array_r, copy=True), expected_adj[r0:r1], atol=1e-8, rtol=1e-8)
                print(
                    json.dumps(
                        {
                            "rank": L_sigma_sys.getComm().getRank(),
                            "ownership_range": [int(r0), int(r1)],
                            "nrows": int(L_sigma_sys.getSize()[0]),
                            "local_rhs": int(r1 - r0),
                        }
                    )
                )
            finally:
                x.destroy()
                x_adj.destroy()
                b.destroy()
        finally:
            L_sigma_sys.destroy()
            L0_sys.destroy()
        """
    )


@pytest.fixture
def blasius_problem():
    """Build a small Blasius problem for testing solvers."""
    mesh = Mesh.create(xrange=(200, 300), yrange=(0, 30), nx=15, ny=10, fd_order=4)
    s = 3.0
    Y_new = 30.0 * np.sinh(s * mesh.Y / 30.0) / np.sinh(s)
    mesh = mesh.deform(Y_new=Y_new)

    bf = load_blasius(mesh.X, mesh.Y, Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15)
    sutherland_air(bf)

    L0, idx = build_lnse_operator(mesh, bf, kx=0.0)
    ngp = mesh.ngp
    L0, bc_indices = apply_dirichlet(L0, idx, "bt", "uvwT", ngp)

    p = mesh.used_ind
    RHO = bf.RHO.ravel(order="F")[p]
    T = bf.T.ravel(order="F")[p]
    W_int = mesh.W.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO, T, bf.kappa, bf.Ma, ngp)

    return mesh, bf, L0, idx, ngp, W_diag, invW_diag


def _build_blasius_problem_petsc(
    mesh: Mesh,
    bf: BaseFlow,
    ngp: int,
):
    """Build the direct-PETSc Blasius operator used by solver tests."""
    L0_petsc, idx = build_lnse_operator_petsc(mesh, bf, kx=0.0)
    L0_petsc, bc_indices = apply_dirichlet_petsc(L0_petsc, idx, "bt", "uvwT", ngp)
    return L0_petsc, idx, bc_indices


def _build_blasius_problem_recipe_petsc(
    mesh: Mesh,
    bf: BaseFlow,
):
    """Build the direct-PETSc Blasius operator via the BC recipe workflow."""
    recipe = [{"type": "dirichlet", "borders": "bt", "variables": "uvwT"}]
    L0_petsc, idx = build_lnse_operator_petsc(mesh, bf, kx=0.0)
    L0_petsc, bc_indices = apply_bc_recipe_petsc(L0_petsc, mesh, idx, recipe)
    return L0_petsc, idx, bc_indices


def _build_blasius_problem_distributed_fallback(
    mesh: Mesh,
    bf: BaseFlow,
    *,
    backend: str = "petsc",
):
    """Build a PETSc matrix through the distributed fallback workflow path."""
    recipe = [{"type": "dirichlet", "borders": "bt", "variables": "uvwT"}]
    solver_cfg = {"petsc_matrix_mode": "distributed"}
    L0_petsc, idx, bc_indices = build_lnse_system(
        mesh,
        bf,
        kx=0.0,
        bc_recipe=recipe,
        backend=backend,
        solver_cfg=solver_cfg,
    )
    return L0_petsc, idx, bc_indices, solver_cfg


def _shift_petsc_operator_diagonal(L0_petsc, shift: complex):

    shifted = L0_petsc.copy()
    diag = shifted.getDiagonal()
    try:
        diag_arr = diag.getArray()
        diag_arr -= complex(shift)
        shifted.setDiagonal(diag)
    finally:
        diag.destroy()
    shifted.assemble()
    return shifted


def test_temporal_resolve_ncv_respects_requested_subspace() -> None:
    assert _resolve_temporal_ncv(ndof=1000, n_modes=50, default_ncv=103, requested_ncv=300) == 300
    assert _resolve_temporal_ncv(ndof=1000, n_modes=50, default_ncv=103, requested_ncv=40) == 51
    assert _resolve_temporal_ncv(ndof=120, n_modes=50, default_ncv=103, requested_ncv=300) == 119


def test_spatial_resolve_ncv_respects_requested_subspace() -> None:
    assert _resolve_spatial_ncv(ndof=1000, n_modes=50, default_ncv=200, requested_ncv=300) == 300
    assert _resolve_spatial_ncv(ndof=1000, n_modes=50, default_ncv=200, requested_ncv=10) == 51
    assert _resolve_spatial_ncv(ndof=120, n_modes=50, default_ncv=200, requested_ncv=300) == 119


def test_adjoint_temporal_scaling() -> None:
    """temporal=True returns adjoint eigenvalues in the temporal convention."""
    diag = np.array([1.0, 2.0, 3.0], dtype=float)
    L0 = sp.diags(diag, format="csc", dtype=complex)
    W_diag = np.ones(diag.size)
    invW_diag = np.ones(diag.size)
    target = 2.2j

    direct_omegas, _ = temporal_eigenvalues(L0, n_modes=2, target=target)
    adjoint_omegas, _ = adjoint_eigenvalues(
        L0,
        n_modes=2,
        target=target,
        W_diag=W_diag,
        invW_diag=invW_diag,
        temporal=True,
    )

    expected = np.conj(direct_omegas)
    rows, cols = linear_sum_assignment(
        np.abs(adjoint_omegas[:, None] - expected[None, :])
    )
    np.testing.assert_allclose(adjoint_omegas[rows], expected[cols], atol=1e-12)


def test_resolvent_svd_residual_norm() -> None:
    """SVD residuals use the forcing W-norm in the denominator."""
    diag = -np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    L0 = sp.diags(diag, format="csc", dtype=complex)
    W_diag = np.array([1.0, 2.0, 0.5, 1.5, 3.0])
    solver_cfg = {"resolvent": {"eigen": {"eps_tol": 1e-12}}}

    gains, responses, forcings = resolvent_analysis(
        L0,
        omega=0.25,
        n_modes=2,
        W_diag=W_diag,
        solver_cfg=solver_cfg,
    )
    rho_fwd, rho_adj = compute_resolvent_svd_residuals(
        L0,
        omega=0.25,
        B_mat=None,
        C_mat=None,
        gains=gains,
        forcings=forcings,
        responses=responses,
        W_diag=W_diag,
        solver_cfg=solver_cfg,
    )

    assert np.all(np.isfinite(rho_fwd))
    assert np.all(np.isfinite(rho_adj))
    assert np.all(rho_fwd < 1e-10), f"rho_fwd unexpectedly large: {rho_fwd}"
    assert np.all(rho_adj < 1e-6), f"rho_adj unexpectedly large: {rho_adj}"


# ---------------------------------------------------------------------------
# Resolvent analysis
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestResolventAnalysis:
    """Tests for the resolvent_analysis solver."""

    def test_gain_monotonicity(self, blasius_problem):
        """Singular values must be sorted in decreasing order."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        gains, _, _ = resolvent_analysis(L0, omega=60.0, n_modes=3, W_diag=W_diag)
        assert gains[0] >= gains[1] >= gains[2] > 0

    def test_forcing_orthonormality(self, blasius_problem):
        """Forcing modes must be W-orthonormal."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        # atol=1e-8 requires the resolvent EPS run at 1e-12; the module default
        # 1e-8 only delivers orthonormality to ~1e-7.
        solver_cfg = {"resolvent": {"eigen": {"eps_tol": 1e-12}}}
        gains, _, forcings = resolvent_analysis(
            L0, omega=60.0, n_modes=3, W_diag=W_diag, solver_cfg=solver_cfg,
        )
        n = forcings.shape[1]
        gram = np.empty((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                gram[i, j] = forcings[:, i].conj() @ (W_diag * forcings[:, j])
        np.testing.assert_allclose(gram, np.eye(n), atol=1e-8)

    def test_response_orthonormality(self, blasius_problem):
        """Response modes must be W-orthonormal."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        solver_cfg = {"resolvent": {"eigen": {"eps_tol": 1e-12}}}
        gains, responses, _ = resolvent_analysis(
            L0, omega=60.0, n_modes=3, W_diag=W_diag, solver_cfg=solver_cfg,
        )
        n = responses.shape[1]
        gram = np.empty((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                gram[i, j] = responses[:, i].conj() @ (W_diag * responses[:, j])
        np.testing.assert_allclose(gram, np.eye(n), atol=1e-8)

    def test_gains_real_positive(self, blasius_problem):
        """Gains must be real and strictly positive."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        gains, _, _ = resolvent_analysis(L0, omega=60.0, n_modes=3, W_diag=W_diag)
        assert np.all(np.isreal(gains))
        assert np.all(gains > 0)

    def test_full_resolvent_default(self, blasius_problem):
        """B_mat=None, C_mat=None gives the full (unrestricted) resolvent."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        gains, responses, forcings = resolvent_analysis(
            L0, omega=60.0, n_modes=3, W_diag=W_diag
        )
        assert gains.shape == (3,)
        assert responses.shape[1] == 3
        assert forcings.shape[1] == 3

    def test_direct_petsc_resolvent_matches_scipy(self, blasius_problem):
        """Direct-PETSc resolvent input should reproduce the SciPy gains and modes."""
        mesh, bf, L0, _, ngp, W_diag, _ = blasius_problem
        L0_petsc, _, bc_indices = _build_blasius_problem_petsc(mesh, bf, ngp)

        B_diag = np.ones(L0.shape[0])
        B_diag[bc_indices] = 0.0
        B_mat_sp = sp.diags(B_diag).tocsc()
        C_mat_sp = B_mat_sp.copy()
        B_mat = scipy_to_petsc(B_mat_sp)
        C_mat = scipy_to_petsc(C_mat_sp)
        try:
            gains_sp, responses_sp, forcings_sp = resolvent_analysis(
                L0,
                omega=60.0,
                n_modes=2,
                W_diag=W_diag,
                B_mat=B_mat_sp,
                C_mat=C_mat_sp,
            )
            gains_petsc, responses_petsc, forcings_petsc = resolvent_analysis(
                L0_petsc,
                omega=60.0,
                n_modes=2,
                W_diag=W_diag,
                B_mat=B_mat,
                C_mat=C_mat,
            )

            np.testing.assert_allclose(gains_petsc, gains_sp, atol=1e-8, rtol=1e-8)
            for j in range(gains_sp.size):
                phase_response = np.vdot(responses_sp[:, j], responses_petsc[:, j])
                if phase_response != 0:
                    responses_petsc[:, j] *= np.exp(-1j * np.angle(phase_response))
                phase_forcing = np.vdot(forcings_sp[:, j], forcings_petsc[:, j])
                if phase_forcing != 0:
                    forcings_petsc[:, j] *= np.exp(-1j * np.angle(phase_forcing))
            np.testing.assert_allclose(responses_petsc, responses_sp, atol=1e-7, rtol=1e-7)
            np.testing.assert_allclose(forcings_petsc, forcings_sp, atol=1e-7, rtol=1e-7)
        finally:
            L0_petsc.destroy()
            B_mat.destroy()
            C_mat.destroy()

    def test_response_shape(self, blasius_problem):
        """Response array must be (ndof, n_modes)."""
        _, _, L0, _, ngp, W_diag, _ = blasius_problem
        gains, responses, forcings = resolvent_analysis(
            L0, omega=60.0, n_modes=3, W_diag=W_diag
        )
        ndof = L0.shape[0]
        assert responses.shape == (ndof, 3)

    def test_resolvent_svd_residual_norm(self, blasius_problem):
        """W-norm denominator: residuals must be finite and small for converged modes.

        compute_resolvent_svd_residuals uses ||v_i||_W in the denominator.
        For W-normalised forcing modes ||v_i||_W == 1, so rho_fwd and rho_adj
        must be well-conditioned.  rho_fwd is near eps by construction;
        rho_adj tests the adjoint half and must be below a loose tolerance.
        """
        _, _, L0, _, ngp, W_diag, invW_diag = blasius_problem
        solver_cfg = {"resolvent": {"eigen": {"eps_tol": 1e-12}}}
        gains, responses, forcings = resolvent_analysis(
            L0, omega=60.0, n_modes=3, W_diag=W_diag, solver_cfg=solver_cfg,
        )
        rho_fwd, rho_adj = compute_resolvent_svd_residuals(
            L0, omega=60.0, B_mat=None, C_mat=None,
            gains=gains, forcings=forcings, responses=responses,
            W_diag=W_diag, solver_cfg=solver_cfg,
        )
        assert np.all(np.isfinite(rho_fwd)), "rho_fwd contains non-finite values"
        assert np.all(np.isfinite(rho_adj)), "rho_adj contains non-finite values"
        # Forward residual: near machine eps by construction
        assert np.all(rho_fwd < 1e-10), f"rho_fwd unexpectedly large: {rho_fwd}"
        # Adjoint residual: converges to EPS tolerance for correct solutions
        assert np.all(rho_adj < 1e-6), f"rho_adj unexpectedly large: {rho_adj}"


# ---------------------------------------------------------------------------
# Temporal eigenvalues
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTemporalEigenvalues:
    """Tests for the temporal_eigenvalues solver."""

    def test_returns_n_modes(self, blasius_problem):
        """Must return exactly n_modes eigenvalues."""
        _, _, L0, _, _, _, _ = blasius_problem
        omega, modes = temporal_eigenvalues(L0, n_modes=5, target=-0.1j)
        assert omega.shape == (5,)
        assert modes.shape[1] == 5

    def test_eigenpair_residual(self, blasius_problem):
        """Eigenvalue residual: ||1j*L0 @ v - omega*v|| / ||v|| must be small."""
        _, _, L0, _, _, _, _ = blasius_problem
        omega, modes = temporal_eigenvalues(L0, n_modes=5, target=-0.1j)
        iL0 = 1j * L0
        for i in range(min(3, len(omega))):
            v = modes[:, i]
            residual = iL0 @ v - omega[i] * v
            rel_err = np.linalg.norm(residual) / np.linalg.norm(v)
            assert rel_err < 1e-6, f"Mode {i}: relative residual {rel_err:.2e}"

    def test_eigenvalues_near_target(self, blasius_problem):
        """Eigenvalues should be sorted by distance to the target."""
        _, _, L0, _, _, _, _ = blasius_problem
        target = -0.1j
        omega, _ = temporal_eigenvalues(L0, n_modes=5, target=target)
        dists = np.abs(omega - target)
        np.testing.assert_array_less(dists[:-1], dists[1:] + 1e-12)

    def test_direct_petsc_matches_scipy_eigenvalues(self, blasius_problem):
        """Direct-PETSc input should reproduce the SciPy temporal spectrum."""
        mesh, bf, L0, _, ngp, _, _ = blasius_problem
        L0_petsc, _, _ = _build_blasius_problem_petsc(mesh, bf, ngp)

        target = -0.1j
        omega_sp, _ = temporal_eigenvalues(L0, n_modes=3, target=target)
        omega_petsc, _ = temporal_eigenvalues(L0_petsc, n_modes=3, target=target)

        omega_sp, _ = _sort_temporal_nearest(omega_sp, np.eye(omega_sp.size, dtype=complex), target)
        omega_petsc, _ = _sort_temporal_nearest(
            omega_petsc, np.eye(omega_petsc.size, dtype=complex), target
        )
        _assert_temporal_spectrum_allclose(omega_petsc, omega_sp, atol=1e-8, rtol=1e-8)

    def test_n_modes_zero_raises(self, blasius_problem):
        """n_modes <= 0 must raise ValueError."""
        _, _, L0, _, _, _, _ = blasius_problem
        with pytest.raises(ValueError):
            temporal_eigenvalues(L0, n_modes=0, target=0.0)


@pytest.mark.slow
class TestLinearSolverLifecycle:
    """Tests for LinearSolver resource management."""

    def test_linear_solver_destroy_releases_petsc_objects(self):
        """LinearSolver.destroy() should release KSP and Mat without error."""
        n = 50
        diag = np.arange(1, n + 1, dtype=complex) + 0.1j
        A_sp = sp.diags(diag, format="csc")
        solver = make_linear_solver(A_sp)
        assert solver._ksp is not None
        assert solver._mat is not None
        solver.destroy()
        assert solver._ksp is None
        assert solver._mat is None
        # Double destroy should be safe (no-op)
        solver.destroy()


class TestDirectPetscSolverInputs:
    """Tests for feeding direct-PETSc operators into solver entry points."""

    def test_linear_solver_matches_scipy_solution(self, blasius_problem):
        """Direct-PETSc linear solves should match the SciPy-based path."""
        mesh, bf, L0, _, ngp, _, _ = blasius_problem
        L0_petsc, _, _ = _build_blasius_problem_petsc(mesh, bf, ngp)

        shift = 5.0 + 20.0j
        L_sigma = L0.copy()
        L_sigma.setdiag(L_sigma.diagonal() - shift)
        L_sigma_petsc = _shift_petsc_operator_diagonal(L0_petsc, shift)

        rng = np.random.default_rng(7)
        rhs = rng.standard_normal(L0.shape[0]) + 1j * rng.standard_normal(L0.shape[0])

        try:
            solver_sp = make_linear_solver(L_sigma)
            solver_petsc = make_linear_solver(L_sigma_petsc)

            x_sp = solver_sp.solve(rhs)
            x_petsc = solver_petsc.solve(rhs)

            np.testing.assert_allclose(x_petsc, x_sp, atol=1e-8, rtol=1e-8)
        finally:
            L0_petsc.destroy()
            L_sigma_petsc.destroy()

    def test_recipe_applied_direct_petsc_matches_scipy_eigenvalues(self, blasius_problem):
        """Recipe-driven PETSc assembly should match the SciPy temporal spectrum."""
        mesh, bf, L0, idx, _, _, _ = blasius_problem
        recipe = [{"type": "dirichlet", "borders": "bt", "variables": "uvwT"}]

        L0_ref, _ = apply_bc_recipe(build_lnse_operator(mesh, bf, kx=0.0)[0], mesh, idx, recipe)
        L0_petsc, _, _ = _build_blasius_problem_recipe_petsc(mesh, bf)
        try:
            target = -0.1j
            omega_sp, _ = temporal_eigenvalues(L0_ref, n_modes=3, target=target)
            omega_petsc, _ = temporal_eigenvalues(L0_petsc, n_modes=3, target=target)

            omega_sp, _ = _sort_temporal_nearest(
                omega_sp, np.eye(omega_sp.size, dtype=complex), target
            )
            omega_petsc, _ = _sort_temporal_nearest(
                omega_petsc, np.eye(omega_petsc.size, dtype=complex), target
            )
            _assert_temporal_spectrum_allclose(omega_petsc, omega_sp, atol=1e-8, rtol=1e-8)
        finally:
            L0_petsc.destroy()
@pytest.mark.slow
class TestDistributedPetscFallbackSolverInputs:
    """Tests for solver entry points fed by the distributed PETSc fallback workflow."""

    def test_temporal_eigenvalues_match_scipy(self, blasius_problem):
        """Distributed-fallback PETSc matrices should match SciPy temporal eigenvalues."""
        mesh, bf, L0, _, _, _, _ = blasius_problem
        L0_petsc, _, _, solver_cfg = _build_blasius_problem_distributed_fallback(mesh, bf)
        try:
            target = -0.1j
            omega_sp, _ = temporal_eigenvalues(L0, n_modes=3, target=target)
            omega_petsc, _ = temporal_eigenvalues(
                L0_petsc,
                n_modes=3,
                target=target,
                solver_cfg=solver_cfg,
            )

            omega_sp, _ = _sort_temporal_nearest(
                omega_sp, np.eye(omega_sp.size, dtype=complex), target
            )
            omega_petsc, _ = _sort_temporal_nearest(
                omega_petsc, np.eye(omega_petsc.size, dtype=complex), target
            )
            _assert_temporal_spectrum_allclose(omega_petsc, omega_sp, atol=1e-8, rtol=1e-8)
        finally:
            L0_petsc.destroy()

    def test_linear_solver_matches_scipy_solution(self, blasius_problem):
        """Distributed-fallback PETSc matrices should match SciPy linear solves."""
        mesh, bf, L0, _, _, _, _ = blasius_problem
        L0_petsc, _, _, solver_cfg = _build_blasius_problem_distributed_fallback(mesh, bf)
        shift = 5.0 + 20.0j
        L_sigma = L0.copy()
        L_sigma.setdiag(L_sigma.diagonal() - shift)
        L_sigma_petsc = L0_petsc.copy()
        L_sigma_petsc.shift(-shift)
        L_sigma_petsc.assemble()

        rng = np.random.default_rng(17)
        rhs = rng.standard_normal(L0.shape[0]) + 1j * rng.standard_normal(L0.shape[0])

        try:
            solver_sp = make_linear_solver(L_sigma)
            solver_petsc = make_linear_solver(L_sigma_petsc, solver_cfg=solver_cfg)

            x_sp = solver_sp.solve(rhs)
            x_petsc = solver_petsc.solve(rhs)
            np.testing.assert_allclose(x_petsc, x_sp, atol=1e-8, rtol=1e-8)

            b = L_sigma_petsc.createVecLeft()
            r0, r1 = b.getOwnershipRange()
            if r1 > r0:
                rows = np.arange(r0, r1, dtype=np.int32)
                b.setValues(rows, rhs[r0:r1])
            b.assemblyBegin()
            b.assemblyEnd()

            x_vec = solver_petsc.solve_vec(b)
            x_adj_vec = solver_petsc.solve_transpose_vec(b)
            try:
                np.testing.assert_allclose(np.array(x_vec.array_r, copy=True), x_sp[r0:r1], atol=1e-8, rtol=1e-8)
                np.testing.assert_allclose(
                    np.array(x_adj_vec.array_r, copy=True),
                    solver_sp.solve_transpose(rhs)[r0:r1],
                    atol=1e-8,
                    rtol=1e-8,
                )
            finally:
                x_vec.destroy()
                x_adj_vec.destroy()
                b.destroy()
        finally:
            L0_petsc.destroy()
            L_sigma_petsc.destroy()

    def test_resolvent_matches_scipy(self, blasius_problem):
        """Distributed-fallback PETSc matrices should match SciPy resolvent outputs."""
        mesh, bf, L0, _, _, W_diag, _ = blasius_problem
        L0_petsc, _, bc_indices, solver_cfg = _build_blasius_problem_distributed_fallback(mesh, bf)

        B_diag = np.ones(L0.shape[0])
        B_diag[bc_indices] = 0.0
        B_mat_sp = sp.diags(B_diag).tocsc()
        C_mat_sp = B_mat_sp.copy()
        B_mat = scipy_to_petsc(B_mat_sp, solver_cfg)
        C_mat = scipy_to_petsc(C_mat_sp, solver_cfg)
        try:
            gains_sp, responses_sp, forcings_sp = resolvent_analysis(
                L0,
                omega=60.0,
                n_modes=2,
                W_diag=W_diag,
                B_mat=B_mat_sp,
                C_mat=C_mat_sp,
            )
            gains_petsc, responses_petsc, forcings_petsc = resolvent_analysis(
                L0_petsc,
                omega=60.0,
                n_modes=2,
                W_diag=W_diag,
                B_mat=B_mat,
                C_mat=C_mat,
                solver_cfg=solver_cfg,
            )

            np.testing.assert_allclose(gains_petsc, gains_sp, atol=1e-8, rtol=1e-8)
            for j in range(gains_sp.size):
                phase_response = np.vdot(responses_sp[:, j], responses_petsc[:, j])
                if phase_response != 0:
                    responses_petsc[:, j] *= np.exp(-1j * np.angle(phase_response))
                phase_forcing = np.vdot(forcings_sp[:, j], forcings_petsc[:, j])
                if phase_forcing != 0:
                    forcings_petsc[:, j] *= np.exp(-1j * np.angle(phase_forcing))
            np.testing.assert_allclose(responses_petsc, responses_sp, atol=1e-7, rtol=1e-7)
            np.testing.assert_allclose(forcings_petsc, forcings_sp, atol=1e-7, rtol=1e-7)
        finally:
            L0_petsc.destroy()
            B_mat.destroy()
            C_mat.destroy()

    @pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec not available")
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_linear_solver_petsc_vec_real_mpi_matches_scipy_solution(self, n_ranks):
        """Real MPI PETSc Vec solves should match the SciPy reference slices on each rank."""
        proc = run_mpi_python(n_ranks, _build_linsolve_mpi_script())
        if proc.returncode != 0:
            pytest.fail(
                f"{n_ranks}-rank linear-solver MPI harness failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        reports = parse_mpi_reports(proc.stdout)
        assert_split_ownership(reports, n_ranks)
        assert all(report["local_rhs"] > 0 for report in reports)



@pytest.mark.slow
class TestAdjointEigenvalues:
    """Tests for the adjoint_eigenvalues solver."""

    def test_conjugate_pairing(self, blasius_problem):
        """Adjoint eigenvalues pair with temporal eigenvalues.

        temporal_eigenvalues returns omega_d = eig(i*L0) = i*eig(L0).
        adjoint_eigenvalues returns omega_a = eig(W^{-1} L0^H W) = conj(eig(L0)).
        Therefore: omega_a = i * conj(omega_d).
        """
        _, _, L0, _, _, W_diag, invW_diag = blasius_problem
        omega_d, _ = temporal_eigenvalues(L0, n_modes=3, target=-0.1j)
        expected_adj = 1j * np.conj(omega_d)
        # Request extra modes to cover the spectral neighborhood
        omega_a, _ = adjoint_eigenvalues(
            L0, n_modes=6, target=expected_adj[0], W_diag=W_diag, invW_diag=invW_diag
        )
        # Check that at least one temporal/adjoint pair matches
        matched = 0
        for ea in expected_adj:
            dists = np.abs(omega_a - ea)
            if np.min(dists) < 1e-4:
                matched += 1
        assert matched >= 1, (
            f"No adjoint eigenvalue paired with any temporal eigenvalue.\n"
            f"  expected_adj = {expected_adj}\n"
            f"  omega_a = {omega_a}"
        )

    def test_adjoint_eigenvector_residual(self, blasius_problem):
        """Adjoint eigenvector must satisfy L_adj @ v = omega_a * v.

        adjoint_eigenvalues builds L_adj = W^{-1} L0^H W (the raw adjoint
        of L0, NOT the temporal adjoint of i*L0) and returns its eigenvalues.
        """
        _, _, L0, _, _, W_diag, invW_diag = blasius_problem
        omega_a, modes_a = adjoint_eigenvalues(
            L0, n_modes=3, target=-0.1j, W_diag=W_diag, invW_diag=invW_diag
        )
        L0H = L0.conj().T
        for i in range(len(omega_a)):
            v = modes_a[:, i]
            Ladj_v = invW_diag * (L0H @ (W_diag * v))
            residual = Ladj_v - omega_a[i] * v
            rel_err = np.linalg.norm(residual) / np.linalg.norm(v)
            assert rel_err < 1e-6, f"Adjoint mode {i}: relative residual {rel_err:.2e}"

    def test_w_normalization(self, blasius_problem):
        """Adjoint modes should be W-normalized: v^H @ (W * v) ≈ 1."""
        _, _, L0, _, _, W_diag, invW_diag = blasius_problem
        _, modes_a = adjoint_eigenvalues(
            L0, n_modes=3, target=-0.1j, W_diag=W_diag, invW_diag=invW_diag
        )
        for i in range(modes_a.shape[1]):
            v = modes_a[:, i]
            norm_val = v.conj() @ (W_diag * v)
            np.testing.assert_allclose(np.abs(norm_val), 1.0, atol=1e-8)

    def test_petsc_input_raises_not_implemented(self, blasius_problem):
        """Adjoint solver should fail early on PETSc matrices until that path exists."""
        mesh, bf, _, _, ngp, W_diag, invW_diag = blasius_problem
        L0_petsc, _, _ = _build_blasius_problem_petsc(mesh, bf, ngp)
        try:
            with pytest.raises(NotImplementedError, match="does not yet support PETSc matrices"):
                adjoint_eigenvalues(
                    L0_petsc,
                    n_modes=3,
                    target=-0.1j,
                    W_diag=W_diag,
                    invW_diag=invW_diag,
                )
        finally:
            L0_petsc.destroy()

    def test_adjoint_temporal_scaling(self, blasius_problem):
        """temporal=True flag should return conj(direct_omegas).

        When adjoint_eigenvalues is called with temporal=True and the same
        target as temporal_eigenvalues, the returned eigenvalues should be
        np.conj(direct_omegas) (up to spectral-neighborhood ordering).
        """
        _, _, L0, _, _, W_diag, invW_diag = blasius_problem
        omega_d, _ = temporal_eigenvalues(L0, n_modes=3, target=-0.1j)
        # Search the adjoint in temporal convention around the first direct omega
        omega_a, _ = adjoint_eigenvalues(
            L0,
            n_modes=6,
            target=omega_d[0],
            W_diag=W_diag,
            invW_diag=invW_diag,
            temporal=True,
        )
        # Each direct omega should have a conjugate partner in the adjoint set
        matched = 0
        for od in omega_d:
            expected = np.conj(od)
            dists = np.abs(omega_a - expected)
            if np.min(dists) < 1e-4:
                matched += 1
        assert matched >= 1, (
            f"temporal=True: no adjoint omega matched conj(direct_omega).\n"
            f"  direct omegas  = {omega_d}\n"
            f"  expected conj  = {np.conj(omega_d)}\n"
            f"  adjoint omegas = {omega_a}"
        )


# ---------------------------------------------------------------------------
# Spatial eigenvalues
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSpatialEigenvalues:
    """Tests for the spatial_eigenvalues solver."""

    @pytest.fixture
    def spatial_problem(self):
        """Build a small Blasius spatial operator."""
        mesh = Mesh.create(xrange=(200, 300), yrange=(0, 30), nx=15, ny=10, fd_order=4)
        s = 3.0
        Y_new = 30.0 * np.sinh(s * mesh.Y / 30.0) / np.sinh(s)
        mesh = mesh.deform(Y_new=Y_new)

        bf = load_blasius(mesh.X, mesh.Y, Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15)
        sutherland_air(bf)

        Lc, R_mat, Lw, idx = build_spatial_operator(mesh, bf, model="2D")
        ngp = mesh.ngp
        Lc, R_mat, bc_indices = apply_dirichlet_spatial(Lc, R_mat, idx, "bt", "uvwT", ngp)
        return mesh, bf, Lc, Lw, R_mat, ngp

    def test_returns_n_modes(self, spatial_problem):
        """Must return exactly n_modes eigenvalues."""
        _, _, Lc, Lw, R_mat, _ = spatial_problem
        alpha, modes = spatial_eigenvalues(
            Lc, Lw, R_mat, omega=60.0, n_modes=3, target=0.1 + 0.01j
        )
        assert alpha.shape == (3,)
        assert modes.shape[1] == 3

    def test_direct_petsc_matches_scipy_spatial_eigenvalues(self, spatial_problem):
        """Direct-PETSc spatial matrices should reproduce the SciPy spectrum."""
        mesh, bf, Lc, Lw, R_mat, ngp = spatial_problem
        Lc_petsc, R_petsc, Lw_petsc, idx_petsc = build_spatial_operator_petsc(mesh, bf, model="2D")
        try:
            Lc_petsc, R_petsc, _ = apply_dirichlet_spatial_petsc(
                Lc_petsc,
                R_petsc,
                idx_petsc,
                "bt",
                "uvwT",
                ngp,
            )
            alpha_sp, _ = spatial_eigenvalues(Lc, Lw, R_mat, omega=60.0, n_modes=3, target=0.1 + 0.01j)
            alpha_petsc, _ = spatial_eigenvalues(
                Lc_petsc,
                Lw_petsc,
                R_petsc,
                omega=60.0,
                n_modes=3,
                target=0.1 + 0.01j,
            )
            np.testing.assert_allclose(
                np.sort_complex(alpha_petsc),
                np.sort_complex(alpha_sp),
                atol=1e-8,
                rtol=1e-8,
            )
        finally:
            Lc_petsc.destroy()
            R_petsc.destroy()
            Lw_petsc.destroy()

    def test_eigenvalues_finite_and_modes_nonzero(self, spatial_problem):
        """Spatial eigenvalues are finite; eigenvectors are nonzero.

        On this small 15x10 grid, R is rank-deficient after penalty BCs
        (rank 1495 of 1500), so the generalized EVP is ill-conditioned.
        We verify structural properties rather than the eigenpair residual.
        """
        _, _, Lc, Lw, R_mat, _ = spatial_problem
        alpha, modes = spatial_eigenvalues(
            Lc, Lw, R_mat, omega=60.0, n_modes=3, target=0.1 + 0.01j
        )
        assert np.all(np.isfinite(alpha)), f"Non-finite eigenvalues: {alpha}"
        for i in range(modes.shape[1]):
            assert np.linalg.norm(modes[:, i]) > 0, f"Zero eigenvector at mode {i}"

    def test_n_modes_zero_raises(self, spatial_problem):
        """n_modes <= 0 must raise ValueError."""
        _, _, Lc, Lw, R_mat, _ = spatial_problem
        with pytest.raises(ValueError):
            spatial_eigenvalues(Lc, Lw, R_mat, omega=60.0, n_modes=0, target=0.0)


# ---------------------------------------------------------------------------
# Cylindrical (axy) model tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCylindricalTemporalEigenvalues:
    """Temporal eigenvalue tests for the cylindrical (axy) model."""

    def test_axy_eigenpair_residual_m0(self):
        """Eigenvalues of axy model at m=0 satisfy eigenpair equation."""
        L0 = build_synthetic_jet(m=0)["L0"]
        omega, modes = temporal_eigenvalues(L0, n_modes=5, target=-0.1j)
        iL0 = 1j * L0
        for i in range(len(omega)):
            v = modes[:, i]
            residual = iL0 @ v - omega[i] * v
            rel_err = np.linalg.norm(residual) / np.linalg.norm(v)
            assert rel_err < 1e-6, (
                f"Mode {i}: eigenpair residual {rel_err:.2e} (expected < 1e-6)"
            )

    def test_direct_petsc_matches_scipy_axy_eigenvalues_m1(self):
        """Direct-PETSc axy input should reproduce the SciPy temporal spectrum."""
        mesh = Mesh.create(
            xrange=(0.0, 10.0),
            yrange=(0.0, 3.0),
            nx=15,
            ny=10,
            fd_order=4,
            y_symmetry=True,
        )
        R = mesh.Y
        bf = BaseFlow(
            Re=1e4,
            Ma=0.9,
            Pr=0.7,
            kappa=1.4,
            T_0=288.15,
            RHO=np.ones_like(R),
            U=0.9 / np.cosh(R) ** 2,
            V=np.zeros_like(R),
            W=np.zeros_like(R),
            T=np.ones_like(R),
        )
        sutherland_air(bf)

        recipe = [{"type": "dirichlet", "borders": "ltr", "variables": "uvwT"}]
        L0_sp, idx = build_lnse_operator(mesh, bf, model="axy", m=1)
        L0_sp, _ = apply_bc_recipe(L0_sp, mesh, idx, recipe)

        L0_petsc, idx_petsc = build_lnse_operator_petsc(mesh, bf, model="axy", m=1)
        try:
            L0_petsc, _ = apply_bc_recipe_petsc(L0_petsc, mesh, idx_petsc, recipe)
            target = -0.1j
            omega_sp, _ = temporal_eigenvalues(L0_sp, n_modes=3, target=target)
            omega_petsc, _ = temporal_eigenvalues(L0_petsc, n_modes=3, target=target)

            omega_sp, _ = _sort_temporal_nearest(
                omega_sp, np.eye(omega_sp.size, dtype=complex), target
            )
            omega_petsc, _ = _sort_temporal_nearest(
                omega_petsc, np.eye(omega_petsc.size, dtype=complex), target
            )
            np.testing.assert_allclose(omega_petsc, omega_sp, atol=1e-8, rtol=1e-8)
        finally:
            L0_petsc.destroy()


@pytest.mark.slow
class TestCylindricalResolventAnalysis:
    """Resolvent parity tests for the cylindrical (axy) model."""

    def test_direct_petsc_matches_scipy_axy_resolvent(self):
        """Direct-PETSc axy workflow should reproduce SciPy resolvent gains and modes."""
        mesh = Mesh.create(
            xrange=(0.0, 10.0),
            yrange=(0.0, 3.0),
            nx=15,
            ny=10,
            fd_order=4,
            y_symmetry=True,
        )
        R = mesh.Y
        bf = BaseFlow(
            Re=1e4,
            Ma=0.9,
            Pr=0.7,
            kappa=1.4,
            T_0=288.15,
            RHO=np.ones_like(R),
            U=0.9 / np.cosh(R) ** 2,
            V=np.zeros_like(R),
            W=np.zeros_like(R),
            T=np.ones_like(R),
        )
        sutherland_air(bf)

        recipe = [{"type": "dirichlet", "borders": "ltr", "variables": "uvwT"}]
        L0_sp, _, bc_indices_sp = build_lnse_system(
            mesh,
            bf,
            model="axy",
            m=0,
            bc_recipe=recipe,
            backend="scipy",
        )
        L0_petsc, _, bc_indices_petsc = build_lnse_system(
            mesh,
            bf,
            model="axy",
            m=0,
            bc_recipe=recipe,
            backend="petsc",
        )

        B_mat = None
        C_mat = None
        try:
            np.testing.assert_array_equal(bc_indices_petsc, bc_indices_sp)

            used_ind = mesh.used_ind
            W_int = mesh.W.ravel(order="F")[used_ind]
            rho = bf.RHO.ravel(order="F")[used_ind]
            T = bf.T.ravel(order="F")[used_ind]
            R_used = mesh.Y.ravel(order="F")[used_ind]
            W_diag, _ = compute_chu_norm(
                W_int,
                rho,
                T,
                bf.kappa,
                bf.Ma,
                mesh.ngp,
                model="axy",
                R=R_used,
            )

            B_diag = np.ones(L0_sp.shape[0])
            B_diag[bc_indices_sp] = 0.0
            B_mat_sp = sp.diags(B_diag).tocsc()
            C_mat_sp = B_mat_sp.copy()
            B_mat = scipy_to_petsc(B_mat_sp)
            C_mat = scipy_to_petsc(C_mat_sp)

            gains_sp, responses_sp, forcings_sp = resolvent_analysis(
                L0_sp,
                omega=2.0,
                n_modes=2,
                W_diag=W_diag,
                B_mat=B_mat_sp,
                C_mat=C_mat_sp,
            )
            gains_petsc, responses_petsc, forcings_petsc = resolvent_analysis(
                L0_petsc,
                omega=2.0,
                n_modes=2,
                W_diag=W_diag,
                B_mat=B_mat,
                C_mat=C_mat,
            )

            np.testing.assert_allclose(gains_petsc, gains_sp, atol=1e-8, rtol=1e-8)
            for j in range(gains_sp.size):
                phase_response = np.vdot(responses_sp[:, j], responses_petsc[:, j])
                if phase_response != 0:
                    responses_petsc[:, j] *= np.exp(-1j * np.angle(phase_response))
                phase_forcing = np.vdot(forcings_sp[:, j], forcings_petsc[:, j])
                if phase_forcing != 0:
                    forcings_petsc[:, j] *= np.exp(-1j * np.angle(phase_forcing))
            np.testing.assert_allclose(responses_petsc, responses_sp, atol=1e-7, rtol=1e-7)
            np.testing.assert_allclose(forcings_petsc, forcings_sp, atol=1e-7, rtol=1e-7)
        finally:
            L0_petsc.destroy()
            if B_mat is not None:
                B_mat.destroy()
            if C_mat is not None:
                C_mat.destroy()


@pytest.mark.slow
class TestCylindricalSpatialEigenvalues:
    """Spatial eigenvalue tests for the cylindrical (axy) model."""

    @pytest.fixture
    def spatial_axy_problem(self):
        """Build a small cylindrical spatial operator."""
        problem = build_synthetic_jet()
        mesh, bf = problem["mesh"], problem["bf"]
        Lc, R_mat, Lw, idx = build_spatial_operator(mesh, bf, model="axy", m=0)
        ngp = mesh.ngp
        Lc, R_mat, bc_indices = apply_dirichlet_spatial(
            Lc, R_mat, idx, "ltr", "uvwT", ngp,
        )
        return Lc, Lw, R_mat, ngp

    def test_spatial_axy_eigenvalues_finite(self, spatial_axy_problem):
        """Spatial axy eigenvalues are finite and eigenvectors are nonzero."""
        Lc, Lw, R_mat, _ = spatial_axy_problem
        alpha, modes = spatial_eigenvalues(
            Lc, Lw, R_mat, omega=2.0, n_modes=3, target=0.1 + 0.01j,
        )
        assert np.all(np.isfinite(alpha)), f"Non-finite eigenvalues: {alpha}"
        for i in range(modes.shape[1]):
            assert np.linalg.norm(modes[:, i]) > 0, f"Zero eigenvector at mode {i}"
