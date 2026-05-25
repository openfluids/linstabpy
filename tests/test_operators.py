# SPDX-License-Identifier: GPL-3.0-only
"""Tests for LNSE coefficients, assembly, energy norm, and spatial formulation."""

import json
import shutil
import textwrap

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.baseflow import BaseFlow, load_blasius, sutherland_air
from linstabpy.boundary_conditions import (
    apply_bc_recipe,
    apply_bc_recipe_petsc,
    apply_dirichlet,
    apply_dirichlet_petsc,
    apply_dirichlet_spatial,
    apply_dirichlet_spatial_petsc,
    apply_neumann,
    apply_neumann_petsc,
)
from linstabpy.mesh import Mesh
from linstabpy.operators import (
    build_lnse_operator,
    build_lnse_operator_petsc,
    build_lnse_system,
    build_spatial_operator,
    build_spatial_operator_petsc,
    compute_chu_norm,
)
from linstabpy.operators.assembly import (
    _apply_inv_B,
    _build_block_csr,
    _create_axy_diff_matrices,
    _make_csr_derivative_cache,
    _prepare_coefficients,
)
from linstabpy.operators.coefficients import (
    _pack_coefficients_in_physical_order,
    compute_cartesian_coefficients,
    compute_cylindrical_coefficients,
)
from linstabpy.utils.petsc_core import (
    compare_petsc_against_scipy,
    petsc_matvec_residual,
    scipy_to_petsc,
)
from tests.mpi_helpers import assert_split_ownership, parse_mpi_reports, run_mpi_python

slow = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_scalars(ngp, *, kappa=1.4, Ma=0.3):
    """Return uniform-flow scalars as flat arrays of length *ngp*."""
    U = np.ones(ngp)
    V = np.zeros(ngp)
    W = np.zeros(ngp)
    RHO = np.ones(ngp)
    T = np.ones(ngp)
    MU = np.ones(ngp)
    P = RHO * T / (kappa * Ma**2)
    return U, V, W, RHO, T, MU, P


def _zero_derivatives(ngp):
    """Return a dict of all derivative arrays set to zero."""
    names = [
        "dUdx",
        "dVdx",
        "dWdx",
        "dRHOdx",
        "dTdx",
        "dPdx",
        "dUdy",
        "dVdy",
        "dWdy",
        "dRHOdy",
        "dTdy",
        "dPdy",
        "d2Udx2",
        "d2Vdx2",
        "d2Wdx2",
        "d2Udy2",
        "d2Vdy2",
        "d2Wdy2",
        "d2Tdx2",
        "d2Tdy2",
        "d2Udxdy",
        "d2Vdxdy",
        "d2Wdxdy",
        "dmudT",
        "d2mudT2",
    ]
    return {n: np.zeros(ngp) for n in names}


def _call_cartesian(
    ngp,
    *,
    kx=0.0,
    Re=1e5,
    Ma=0.3,
    Pr=0.72,
    kappa=1.4,
    U=None,
    V=None,
    W=None,
    RHO=None,
    T=None,
    MU=None,
    P=None,
    derivs=None,
):
    """Convenience wrapper around compute_cartesian_coefficients."""
    if U is None:
        U, V, W, RHO, T, MU, P = _uniform_scalars(ngp, kappa=kappa, Ma=Ma)
    if derivs is None:
        derivs = _zero_derivatives(ngp)
    return compute_cartesian_coefficients(
        U,
        V,
        W,
        RHO,
        T,
        MU,
        P,
        kx,
        Re,
        Ma,
        Pr,
        kappa,
        derivs["dUdx"],
        derivs["dVdx"],
        derivs["dWdx"],
        derivs["dRHOdx"],
        derivs["dTdx"],
        derivs["dPdx"],
        derivs["dUdy"],
        derivs["dVdy"],
        derivs["dWdy"],
        derivs["dRHOdy"],
        derivs["dTdy"],
        derivs["dPdy"],
        derivs["d2Udx2"],
        derivs["d2Vdx2"],
        derivs["d2Wdx2"],
        derivs["d2Udy2"],
        derivs["d2Vdy2"],
        derivs["d2Wdy2"],
        derivs["d2Tdx2"],
        derivs["d2Tdy2"],
        derivs["d2Udxdy"],
        derivs["d2Vdxdy"],
        derivs["d2Wdxdy"],
        derivs["dmudT"],
        derivs["d2mudT2"],
    )


def _build_lnse_system_mpi_script(*, bc_recipe: list[dict] | None) -> str:
    recipe_json = json.dumps(bc_recipe)
    return textwrap.dedent(
        f"""
        import json
        import numpy as np
        import scipy.sparse as sp
        from linstabpy.mesh import Mesh
        from linstabpy.baseflow import BaseFlow, sutherland_air
        from linstabpy.operators import build_lnse_system
        from linstabpy.utils.petsc_core import petsc_owned_rows
        import linstabpy.utils.petsc_core as petsc_core

        recipe = json.loads({recipe_json!r})
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

        def _unexpected_fallback(*args, **kwargs):
            raise AssertionError("SciPy-to-PETSc fallback should not be used for supported distributed 2D direct assembly")

        petsc_core.scipy_to_petsc = _unexpected_fallback

        L0_ref, idx_ref, bc_ref = build_lnse_system(
            mesh,
            bf,
            kx=0.2,
            model="2D",
            bc_recipe=recipe,
            solver_cfg={{"petsc_matrix_mode": "local"}},
        )
        L0_sys, idx_sys, bc_sys = build_lnse_system(
            mesh,
            bf,
            kx=0.2,
            model="2D",
            bc_recipe=recipe,
            backend="petsc",
            solver_cfg={{"petsc_matrix_mode": "distributed"}},
        )

        try:
            for key, value in idx_ref.items():
                np.testing.assert_array_equal(idx_sys[key], value)
            np.testing.assert_array_equal(bc_sys, bc_ref)

            owned = petsc_owned_rows(L0_sys)
            r0, r1 = owned["ownership_range"]
            expected = L0_ref.tocsr()[r0:r1].copy()
            expected.sum_duplicates()
            expected.eliminate_zeros()
            expected.sort_indices()

            candidate = sp.csr_matrix(
                (owned["data"], owned["indices"], owned["indptr"]),
                shape=(r1 - r0, L0_sys.getSize()[1]),
                dtype=complex,
            )
            candidate.sum_duplicates()
            candidate.eliminate_zeros()
            candidate.sort_indices()

            np.testing.assert_array_equal(owned["row_ids"], np.arange(r0, r1, dtype=np.int64))
            np.testing.assert_array_equal(candidate.indptr, expected.indptr)
            np.testing.assert_array_equal(candidate.indices, expected.indices)
            np.testing.assert_allclose(candidate.data, expected.data, atol=1e-10, rtol=1e-10)

            print(
                json.dumps(
                    {{
                        "rank": L0_sys.getComm().getRank(),
                        "ownership_range": [int(r0), int(r1)],
                        "nrows": int(L0_sys.getSize()[0]),
                        "nnz": int(owned["data"].size),
                    }}
                )
            )
        finally:
            L0_sys.destroy()
        """
    )


# =========================================================================
# TestCartesianCoefficients
# =========================================================================


class TestCartesianCoefficients:
    """Tests for compute_cartesian_coefficients with uniform base flow."""

    NGP = 50

    def test_coefficient_shapes(self):
        """All seven coefficient arrays have shape (5, 5, ngp)."""
        coeffs = _call_cartesian(self.NGP)
        for name in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B"):
            arr = getattr(coeffs, name)
            assert arr.shape == (5, 5, self.NGP), f"{name} shape mismatch"

    def test_B_diagonal(self):
        """B[k,k,:] = 1j*RHO for k=1..4, B[0,0,:] = 1j; off-diag = 0."""
        coeffs = _call_cartesian(self.NGP)
        B = coeffs.B
        # Diagonal entries
        np.testing.assert_allclose(B[0, 0, :], 1j * np.ones(self.NGP))
        for k in range(1, 5):
            np.testing.assert_allclose(B[k, k, :], 1j * np.ones(self.NGP))
        # Off-diagonal entries
        for i in range(5):
            for j in range(5):
                if i != j:
                    np.testing.assert_allclose(
                        B[i, j, :],
                        np.zeros(self.NGP),
                        atol=1e-15,
                        err_msg=f"B[{i},{j},:] should be zero",
                    )

    def test_continuity_row_inviscid(self):
        """Second-derivative matrices vanish for the continuity row (i=0)."""
        coeffs = _call_cartesian(self.NGP)
        for name in ("Axx", "Ayy", "Axy"):
            arr = getattr(coeffs, name)
            np.testing.assert_allclose(
                arr[0, :, :],
                0.0,
                atol=1e-15,
                err_msg=f"{name}[0,:,:] should be zero (continuity is inviscid)",
            )

    def test_viscous_diagonal_positive(self):
        """Diagonal viscous entries (Axx, Ayy) for momentum/energy rows are positive.

        The coefficient matrices store the diffusion coefficient mu/(Re*rho) > 0;
        dissipation arises from the negative eigenvalues of the Laplacian.
        """
        coeffs = _call_cartesian(self.NGP)
        for name in ("Axx", "Ayy"):
            arr = getattr(coeffs, name)
            for k in range(1, 5):
                real_part = arr[k, k, :].real
                assert np.all(real_part >= -1e-15), (
                    f"{name}[{k},{k},:] real part should be >= 0 (diffusion coeff), "
                    f"min = {real_part.min()}"
                )

    def test_pack_coefficients_in_physical_order_reports_missing_keys(self):
        """Missing local coefficient keys raise a contextual KeyError."""
        with pytest.raises(KeyError, match="Missing coefficient locals"):
            _pack_coefficients_in_physical_order(
                {"c_A0_11": np.ones(2, dtype=complex)},
                ngp=2,
                c2p_idx=[0, 1, 2, 3, 4],
                c2p_sgn=np.ones(5),
            )

    def test_zero_baseflow_A0(self):
        """With U=V=W=0, A0 should simplify (no advection contributions)."""
        ngp = self.NGP
        U = np.zeros(ngp)
        V = np.zeros(ngp)
        W = np.zeros(ngp)
        RHO = np.ones(ngp)
        T = np.ones(ngp)
        MU = np.ones(ngp)
        kappa, Ma = 1.4, 0.3
        P = RHO * T / (kappa * Ma**2)
        derivs = _zero_derivatives(ngp)

        coeffs = _call_cartesian(
            ngp,
            U=U,
            V=V,
            W=W,
            RHO=RHO,
            T=T,
            MU=MU,
            P=P,
            derivs=derivs,
            kappa=kappa,
            Ma=Ma,
        )
        # With zero velocity and zero derivatives, the advection part of A0
        # should contribute nothing; check a few representative entries.
        # Continuity row: A0[0,1,:] (links to u) should be zero when RHO*U=0
        # and all velocity derivatives are zero.
        np.testing.assert_allclose(coeffs.Ax[0, 0, :], 0.0, atol=1e-14)

    def test_inviscid_limit(self):
        """When MU → 0, Axx/Ayy/Axy momentum rows → 0."""
        ngp = self.NGP
        U, V, W, RHO, T, _, P = _uniform_scalars(ngp)
        MU = np.full(ngp, 1e-30)  # effectively zero viscosity
        derivs = _zero_derivatives(ngp)

        coeffs = _call_cartesian(
            ngp,
            U=U,
            V=V,
            W=W,
            RHO=RHO,
            T=T,
            MU=MU,
            P=P,
            derivs=derivs,
            Re=1e5,
        )
        for name in ("Axx", "Ayy", "Axy"):
            arr = getattr(coeffs, name)
            for k in range(1, 5):
                np.testing.assert_allclose(
                    arr[k, :, :],
                    0.0,
                    atol=1e-20,
                    err_msg=f"Inviscid limit: {name}[{k},:,:] should vanish",
                )


# =========================================================================
# TestCylindricalCoefficients
# =========================================================================


class TestCylindricalCoefficients:
    """Tests for compute_cylindrical_coefficients."""

    NGP = 50

    def _call_cylindrical(self, ngp, *, R=None, m=0, **kwargs):
        """Convenience wrapper for cylindrical coefficients.

        The actual signature is:
            compute_cylindrical_coefficients(
                U, V, W, R, RHO, T, MU, m, Re, Ma, Pr, kappa,
                dUdr, dVdr, dWdr, dRHOdr, dTdr,
                dUdz, dVdz, dWdz, dRHOdz, dTdz,
                d2Udr2, d2Vdr2, d2Wdr2, d2Tdr2,
                d2Udz2, d2Vdz2, d2Wdz2, d2Tdz2,
                d2Udrz, d2Vdrz, d2Wdrz,
                dmudT, d2mudT2,
            )
        """
        Re = kwargs.get("Re", 1e5)
        Ma = kwargs.get("Ma", 0.3)
        Pr = kwargs.get("Pr", 0.72)
        kappa = kwargs.get("kappa", 1.4)
        U, V, W, RHO, T, MU, _ = _uniform_scalars(ngp, kappa=kappa, Ma=Ma)
        z = np.zeros(ngp)
        if R is None:
            R = np.ones(ngp) * 1e6  # large radius
        return compute_cylindrical_coefficients(
            U,
            V,
            W,
            R,
            RHO,
            T,
            MU,
            m,
            Re,
            Ma,
            Pr,
            kappa,
            z,
            z,
            z,
            z,
            z,  # dUdr..dTdr
            z,
            z,
            z,
            z,
            z,  # dUdz..dTdz
            z,
            z,
            z,
            z,  # d2Udr2..d2Tdr2
            z,
            z,
            z,
            z,  # d2Udz2..d2Tdz2
            z,
            z,
            z,  # d2Udrz..d2Wdrz
            z,
            z,  # dmudT, d2mudT2
        )

    def test_cylindrical_shapes(self):
        """Cylindrical coefficient arrays have shape (5, 5, ngp)."""
        coeffs = self._call_cylindrical(self.NGP)
        expected = (5, 5, self.NGP)
        for attr in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B"):
            assert getattr(coeffs, attr).shape == expected, f"{attr} shape mismatch"

    def test_large_R_approaches_cartesian(self):
        """At R >> 1, m=0: cylindrical viscous operators approach Cartesian ones.

        The B matrix differs structurally between coordinate systems (e.g., the
        cylindrical energy equation carries a cv prefactor), so only the
        second-derivative viscous operators are compared.
        """
        ngp = self.NGP
        R_large = np.ones(ngp) * 1e8
        cyl = self._call_cylindrical(ngp, R=R_large, m=0)
        cart = _call_cartesian(ngp)
        # Axx, Ayy should be close — 1/R corrections vanish (tolerance
        # is 1e-3 because some cylindrical entries carry O(1/R) metric terms)
        np.testing.assert_allclose(cyl.Axx, cart.Axx, atol=1e-3)
        np.testing.assert_allclose(cyl.Ayy, cart.Ayy, atol=1e-3)

    def test_axis_regularization_uses_mesh_spacing_floor(self):
        """R=0 axis singularity is regularized with local mesh spacing, not 1e-14.

        With the 1e-14 floor, 1/R² near the axis could reach ~10²⁸.  The fix
        clips R to the smallest positive inter-node gap, so 1/R stays O(1/dr).
        We verify:
        1. No NaN or Inf in any coefficient for a grid that includes R=0.
        2. The effective floor is >= the minimum inter-node spacing (>> 1e-14).
        """
        ngp = 10
        # Grid with the axis (R=0) as the first point
        R_axis = np.linspace(0.0, 1.0, ngp)
        z = np.zeros(ngp)
        Ma = 0.3
        kappa = 1.4
        U, V, W, RHO, T, MU, _ = _uniform_scalars(ngp, kappa=kappa, Ma=Ma)
        coeffs = compute_cylindrical_coefficients(
            U,
            V,
            W,
            R_axis,
            RHO,
            T,
            MU,
            m=0,
            Re=1e5,
            Ma=Ma,
            Pr=0.72,
            kappa=kappa,
            dUdr=z,
            dVdr=z,
            dWdr=z,
            dRHOdr=z,
            dTdr=z,
            dUdz=z,
            dVdz=z,
            dWdz=z,
            dRHOdz=z,
            dTdz=z,
            d2Udr2=z,
            d2Vdr2=z,
            d2Wdr2=z,
            d2Tdr2=z,
            d2Udz2=z,
            d2Vdz2=z,
            d2Wdz2=z,
            d2Tdz2=z,
            d2Udrz=z,
            d2Vdrz=z,
            d2Wdrz=z,
            dmudT=z,
            d2mudT2=z,
        )
        for attr in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B"):
            arr = getattr(coeffs, attr)
            assert np.all(np.isfinite(arr)), (
                f"{attr} contains non-finite values when R includes axis (R=0)"
            )
        # The smallest axis-normal value should be bounded by 1/dr, not 1e-14
        # i.e. max(|1/R_regularised|) <= 1/dr_min + small tolerance
        dr_min = float(np.min(np.diff(R_axis[R_axis > 0]))) if np.any(R_axis > 0) else 1.0
        max_abs_A0 = float(np.max(np.abs(coeffs.A0)))
        # Sanity: the largest coefficient must not be astronomically large
        threshold = 1.0 / dr_min * 1e4  # generous factor for combined coefficients
        assert max_abs_A0 < threshold, (
            f"max|A0| = {max_abs_A0:.3e} exceeds 1/dr threshold {threshold:.3e}; "
            "axis regularization may be using 1e-14 floor instead of mesh spacing"
        )


# =========================================================================
# TestOperatorAssembly
# =========================================================================


class TestOperatorAssembly:
    """Tests for build_lnse_operator using small_mesh_2d + uniform_baseflow."""

    def test_continuity_block_no_viscous(self, small_mesh_2d, uniform_baseflow):
        """First ngp rows (continuity) should have no second-derivative viscous entries.

        This is a structural check: the continuity equation is inviscid, so the
        operator block corresponding to the first variable (density) should not
        contain viscous diffusion patterns.  We verify the continuity row block
        is sparser than a momentum row block.
        """
        mesh = small_mesh_2d
        ngp = mesh.ngp
        L0, _ = build_lnse_operator(mesh, uniform_baseflow, kx=0.0)
        L0_csr = L0.tocsr()
        # Count non-zeros in continuity rows vs. first momentum rows
        cont_nnz = L0_csr[:ngp, :].nnz
        mom_nnz = L0_csr[ngp : 2 * ngp, :].nnz
        # Continuity block should be sparser (no viscous stencil)
        assert cont_nnz < mom_nnz, (
            f"Continuity nnz ({cont_nnz}) should be < momentum nnz ({mom_nnz})"
        )

    def test_variable_dof_ranges(self, small_mesh_2d, uniform_baseflow):
        """Variable DOF ranges: rho_j = [0, ngp), u_j = [ngp, 2*ngp), etc."""
        mesh = small_mesh_2d
        ngp = mesh.ngp
        _, idx = build_lnse_operator(mesh, uniform_baseflow, kx=0.0)
        np.testing.assert_array_equal(idx["rho_j"], np.arange(0, ngp))
        np.testing.assert_array_equal(idx["u_j"], np.arange(ngp, 2 * ngp))
        np.testing.assert_array_equal(idx["v_j"], np.arange(2 * ngp, 3 * ngp))
        np.testing.assert_array_equal(idx["w_j"], np.arange(3 * ngp, 4 * ngp))
        np.testing.assert_array_equal(idx["T_j"], np.arange(4 * ngp, 5 * ngp))

    def test_invalid_model_raises(self, small_mesh_2d, uniform_baseflow):
        """Invalid model string raises ValueError."""
        with pytest.raises(ValueError):
            build_lnse_operator(small_mesh_2d, uniform_baseflow, model="invalid")

    def test_build_block_csr_matches_explicit_row_scaled_sum(self, small_mesh_2d):
        """Optimized block assembly matches the explicit sparse-sum formula."""
        mesh = small_mesh_2d
        ngp = mesh.ngp
        rng = np.random.default_rng(123)

        a0 = rng.standard_normal(ngp) + 1j * rng.standard_normal(ngp)
        ax = rng.standard_normal(ngp) + 1j * rng.standard_normal(ngp)
        ay = rng.standard_normal(ngp) + 1j * rng.standard_normal(ngp)
        axx = rng.standard_normal(ngp) + 1j * rng.standard_normal(ngp)
        ayy = rng.standard_normal(ngp) + 1j * rng.standard_normal(ngp)
        axy = rng.standard_normal(ngp) + 1j * rng.standard_normal(ngp)

        candidate = _build_block_csr(
            ngp,
            a0,
            ax,
            ay,
            axx,
            ayy,
            axy,
            mesh.Dy,
            mesh.D2y,
            mesh.Dx,
            mesh.D2x,
            mesh.Dyx,
        )

        expected = sp.csr_matrix((ngp, ngp), dtype=complex)
        expected = expected + sp.diags(a0, format="csr")
        expected = expected + sp.diags(ay, format="csr") @ sp.csr_matrix(mesh.Dy, dtype=complex)
        expected = expected + sp.diags(ax, format="csr") @ sp.csr_matrix(mesh.Dx, dtype=complex)
        expected = expected + sp.diags(ayy, format="csr") @ sp.csr_matrix(mesh.D2y, dtype=complex)
        expected = expected + sp.diags(axx, format="csr") @ sp.csr_matrix(mesh.D2x, dtype=complex)
        expected = expected + sp.diags(axy, format="csr") @ sp.csr_matrix(mesh.Dyx, dtype=complex)
        expected.sum_duplicates()
        expected.eliminate_zeros()
        expected.sort_indices()

        np.testing.assert_array_equal(candidate.indptr, expected.indptr)
        np.testing.assert_array_equal(candidate.indices, expected.indices)
        np.testing.assert_allclose(candidate.data, expected.data, atol=1e-12, rtol=1e-12)

    def test_make_csr_derivative_cache_canonicalizes_duplicate_rows(self):
        """CSR derivative cache canonicalizes duplicate entries before row expansion."""
        op = sp.csr_matrix(
            (
                np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float),
                np.array([1, 0, 1, 1, 0], dtype=np.int32),
                np.array([0, 3, 5], dtype=np.int32),
            ),
            shape=(2, 2),
        )
        cache = _make_csr_derivative_cache(op)

        assert cache.csr.has_canonical_format
        np.testing.assert_array_equal(cache.row_nnz, np.array([2, 2], dtype=np.int64))
        np.testing.assert_array_equal(cache.row_indices, np.array([0, 0, 1, 1], dtype=np.int64))

        zeros = np.zeros(2, dtype=complex)
        ay = np.array([2.0, 3.0], dtype=complex)
        candidate = _build_block_csr(
            2,
            zeros,
            zeros,
            ay,
            zeros,
            zeros,
            zeros,
            cache,
            sp.csr_matrix((2, 2), dtype=complex),
            sp.csr_matrix((2, 2), dtype=complex),
            sp.csr_matrix((2, 2), dtype=complex),
            sp.csr_matrix((2, 2), dtype=complex),
        )
        expected = sp.diags(ay, format="csr") @ cache.csr
        expected.sum_duplicates()
        expected.eliminate_zeros()
        expected.sort_indices()

        np.testing.assert_array_equal(candidate.indptr, expected.indptr)
        np.testing.assert_array_equal(candidate.indices, expected.indices)
        np.testing.assert_allclose(candidate.data, expected.data, atol=1e-12, rtol=1e-12)

    def test_cartesian_prep_uses_dxy_for_mixed_derivative(self, small_mesh_2d, uniform_baseflow):
        """MATLAB 2D path uses Dzr = Dz*Dr = Dx @ Dy = mesh.Dxy in the mixed-derivative block."""
        mesh = small_mesh_2d.deform(
            X_new=small_mesh_2d.X + 0.05 * small_mesh_2d.X * small_mesh_2d.Y,
            Y_new=small_mesh_2d.Y + 0.1 * small_mesh_2d.Y**2,
        )
        assert (mesh.Dxy != mesh.Dyx).nnz > 0, "test mesh must distinguish Dxy and Dyx"

        prepared = _prepare_coefficients(mesh, uniform_baseflow, kx=0.0, model="2D", m=0.0)
        assert (prepared["D2RZ"] != mesh.Dxy).nnz == 0

    def test_build_lnse_system_emits_assembly_metrics_sidecar(
        self,
        small_mesh_2d,
        uniform_baseflow,
        tmp_path,
    ):
        """Assembly metrics sidecar records stage timings and core shape stats."""
        output = tmp_path / "assembly_metrics.json"
        L0, _, bc_indices = build_lnse_system(
            small_mesh_2d,
            uniform_baseflow,
            kx=0.0,
            model="2D",
            solver_cfg={
                "assembly_metrics": True,
                "assembly_metrics_log": False,
                "assembly_metrics_output": str(output),
            },
        )

        assert bc_indices.size == 0
        assert output.exists()

        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "ok"
        assert report["requested_backend"] == "scipy"
        assert report["resolved_backend"] == "scipy"
        assert report["ngp"] == small_mesh_2d.ngp
        assert report["ndof"] == 5 * small_mesh_2d.ngp
        assert report["nnz"] == L0.nnz
        assert report["coeff_tensor_bytes_total"] > 0
        assert report["matrix_type"] == "scipy"

        stages = report["stages"]
        for name in (
            "prepare_coefficients",
            "assemble_raw_blocks",
            "apply_invB",
            "apply_sponge",
            "apply_bc",
            "scipy_to_petsc",
            "petsc_compaction",
        ):
            assert name in stages

        assert stages["prepare_coefficients"]["ngp"] == small_mesh_2d.ngp
        assert stages["assemble_raw_blocks"]["nnz"] > 0
        assert stages["apply_bc"]["skipped"] is True
        assert stages["scipy_to_petsc"]["skipped"] is True
        assert stages["petsc_compaction"]["skipped"] is True
        assert {
            "seconds",
            "rss_peak_before_bytes",
            "rss_peak_after_bytes",
            "rss_peak_delta_bytes",
        } <= set(stages["assemble_raw_blocks"])

    def test_build_lnse_system_petsc_metrics_records_local_full_csr_mode(
        self,
        small_mesh_2d,
        uniform_baseflow,
        tmp_path,
    ):
        """Local PETSc assembly reports the CSR-based raw assembly mode."""
        output = tmp_path / "assembly_metrics_petsc.json"
        L0, _, _ = build_lnse_system(
            small_mesh_2d,
            uniform_baseflow,
            kx=0.0,
            model="2D",
            backend="petsc",
            solver_cfg={
                "petsc_matrix_mode": "local",
                "assembly_metrics": True,
                "assembly_metrics_log": False,
                "assembly_metrics_output": str(output),
            },
        )
        try:
            report = json.loads(output.read_text(encoding="utf-8"))
            assert report["status"] == "ok"
            assert report["requested_backend"] == "petsc"
            assert report["resolved_backend"] == "petsc"
            assert report["matrix_type"] == "petsc"
            assert report["raw_block_assembly_mode"] == "petsc_local_full_csr"
            assert report["nnz"] == int(L0.getInfo()["nz_used"])
        finally:
            L0.destroy()

    def test_build_lnse_system_petsc_local_sponge_update_allows_missing_diagonal_slots(
        self,
        small_mesh_2d,
        uniform_baseflow,
    ):
        """Local PETSc Song-style sponge updates should not abort on diagonal insertion."""
        small_mesh_2d.sponge = np.full_like(small_mesh_2d.X, 1.0e-3)
        L0, _, _ = build_lnse_system(
            small_mesh_2d,
            uniform_baseflow,
            kx=0.0,
            model="2D",
            backend="petsc",
            solver_cfg={"petsc_matrix_mode": "local"},
        )
        try:
            diag = L0.getDiagonal()
            try:
                assert diag.getSize() == 5 * small_mesh_2d.ngp
                assert np.isfinite(np.asarray(diag.getArray(), dtype=complex)).all()
            finally:
                diag.destroy()
        finally:
            L0.destroy()

    def test_scipy_to_petsc_pre_bc_l0_has_sparse_parity(self):
        """Pre-BC L0 survives SciPy->PETSc conversion with entrywise parity."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=14,
            ny=10,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=1e4,
            Ma=0.3,
            Pr=0.7,
            kappa=1.4,
            T_0=288.15,
            RHO=1.0 + 0.03 * np.cos(np.pi * X) * np.cos(0.5 * np.pi * Y),
            U=1.0 + 0.08 * np.sin(np.pi * X) * np.exp(-0.8 * Y),
            V=0.04 * np.cos(0.5 * np.pi * X) * np.sin(np.pi * Y),
            W=0.03 * np.sin(1.5 * np.pi * X) * np.cos(0.5 * np.pi * Y),
            T=1.0 + 0.05 * np.cos(np.pi * X) * np.exp(-0.4 * Y),
        )
        sutherland_air(bf)

        L0, _ = build_lnse_operator(mesh, bf, kx=0.35, model="2D")
        L0_petsc = scipy_to_petsc(L0)
        try:
            report = compare_petsc_against_scipy(L0, L0_petsc, check_structure=True)
            assert report["pass"], report
            assert report["nnz_delta"] == 0
        finally:
            L0_petsc.destroy()

    def test_scipy_to_petsc_pre_bc_l0_has_matvec_parity(self):
        """Pre-BC L0 has matching PETSc and SciPy operator action."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=12,
            ny=9,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=8e3,
            Ma=0.4,
            Pr=0.72,
            kappa=1.4,
            T_0=300.0,
            RHO=1.0 + 0.02 * np.cos(0.75 * np.pi * X) * np.cos(np.pi * Y),
            U=0.9 + 0.12 * np.sin(np.pi * X) * np.exp(-Y),
            V=0.03 * np.cos(np.pi * X) * np.sin(np.pi * Y),
            W=0.02 * np.sin(0.5 * np.pi * X) * np.cos(np.pi * Y),
            T=1.0 + 0.04 * np.sin(np.pi * X) * np.exp(-0.5 * Y),
        )
        sutherland_air(bf)

        L0, _ = build_lnse_operator(mesh, bf, kx=0.2, model="2D")
        L0_petsc = scipy_to_petsc(L0)
        try:
            report = petsc_matvec_residual(L0, L0_petsc, seed=11, trials=3)
            assert report["pass"], report
            assert report["max_abs_residual"] < 1e-11
            assert report["max_rel_residual"] < 1e-11
        finally:
            L0_petsc.destroy()

    def test_direct_petsc_pre_bc_l0_has_sparse_parity(self):
        """Direct PETSc pre-BC L0 matches the SciPy reference entrywise."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=14,
            ny=10,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=1e4,
            Ma=0.3,
            Pr=0.7,
            kappa=1.4,
            T_0=288.15,
            RHO=1.0 + 0.03 * np.cos(np.pi * X) * np.cos(0.5 * np.pi * Y),
            U=1.0 + 0.08 * np.sin(np.pi * X) * np.exp(-0.8 * Y),
            V=0.04 * np.cos(0.5 * np.pi * X) * np.sin(np.pi * Y),
            W=0.03 * np.sin(1.5 * np.pi * X) * np.cos(0.5 * np.pi * Y),
            T=1.0 + 0.05 * np.cos(np.pi * X) * np.exp(-0.4 * Y),
        )
        sutherland_air(bf)

        L0, idx = build_lnse_operator(mesh, bf, kx=0.35, model="2D")
        L0_petsc, idx_petsc = build_lnse_operator_petsc(mesh, bf, kx=0.35, model="2D")
        try:
            for key, value in idx.items():
                assert key in idx_petsc
                np.testing.assert_array_equal(idx_petsc[key], value)

            report = compare_petsc_against_scipy(L0, L0_petsc, check_structure=True)
            assert report["pass"], report
            assert report["nnz_delta"] == 0
        finally:
            L0_petsc.destroy()

    def test_direct_petsc_pre_bc_l0_has_matvec_parity(self):
        """Direct PETSc pre-BC L0 matches the SciPy operator action."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=12,
            ny=9,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=8e3,
            Ma=0.4,
            Pr=0.72,
            kappa=1.4,
            T_0=300.0,
            RHO=1.0 + 0.02 * np.cos(0.75 * np.pi * X) * np.cos(np.pi * Y),
            U=0.9 + 0.12 * np.sin(np.pi * X) * np.exp(-Y),
            V=0.03 * np.cos(np.pi * X) * np.sin(np.pi * Y),
            W=0.02 * np.sin(0.5 * np.pi * X) * np.cos(np.pi * Y),
            T=1.0 + 0.04 * np.sin(np.pi * X) * np.exp(-0.5 * Y),
        )
        sutherland_air(bf)

        L0, _ = build_lnse_operator(mesh, bf, kx=0.2, model="2D")
        L0_petsc, _ = build_lnse_operator_petsc(mesh, bf, kx=0.2, model="2D")
        try:
            report = petsc_matvec_residual(L0, L0_petsc, seed=11, trials=3)
            assert report["pass"], report
            assert report["max_abs_residual"] < 1e-10
            assert report["max_rel_residual"] < 1e-10
        finally:
            L0_petsc.destroy()

    def test_direct_petsc_post_dirichlet_l0_has_sparse_parity(self):
        """Direct PETSc Dirichlet enforcement matches the SciPy BC path."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=14,
            ny=10,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=1e4,
            Ma=0.3,
            Pr=0.7,
            kappa=1.4,
            T_0=288.15,
            RHO=1.0 + 0.03 * np.cos(np.pi * X) * np.cos(0.5 * np.pi * Y),
            U=1.0 + 0.08 * np.sin(np.pi * X) * np.exp(-0.8 * Y),
            V=0.04 * np.cos(0.5 * np.pi * X) * np.sin(np.pi * Y),
            W=0.03 * np.sin(1.5 * np.pi * X) * np.cos(0.5 * np.pi * Y),
            T=1.0 + 0.05 * np.cos(np.pi * X) * np.exp(-0.4 * Y),
        )
        sutherland_air(bf)

        borders = "lt"
        variables = "uvwT"
        penalty = 1.0e4

        L0, idx = build_lnse_operator(mesh, bf, kx=0.35, model="2D")
        L0_bc, bc_indices = apply_dirichlet(L0, idx, borders, variables, mesh.ngp, penalty=penalty)

        L0_petsc, idx_petsc = build_lnse_operator_petsc(mesh, bf, kx=0.35, model="2D")
        try:
            L0_petsc_bc, bc_indices_petsc = apply_dirichlet_petsc(
                L0_petsc,
                idx_petsc,
                borders,
                variables,
                mesh.ngp,
                penalty=penalty,
            )
            np.testing.assert_array_equal(bc_indices_petsc, bc_indices)

            report = compare_petsc_against_scipy(L0_bc, L0_petsc_bc, check_structure=False)
            assert report["pass"], report
            assert report["max_abs_residual"] < 1e-10
        finally:
            L0_petsc.destroy()

    def test_direct_petsc_post_dirichlet_l0_has_matvec_parity(self):
        """Direct PETSc Dirichlet enforcement preserves operator action parity."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=12,
            ny=9,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=8e3,
            Ma=0.4,
            Pr=0.72,
            kappa=1.4,
            T_0=300.0,
            RHO=1.0 + 0.02 * np.cos(0.75 * np.pi * X) * np.cos(np.pi * Y),
            U=0.9 + 0.12 * np.sin(np.pi * X) * np.exp(-Y),
            V=0.03 * np.cos(np.pi * X) * np.sin(np.pi * Y),
            W=0.02 * np.sin(0.5 * np.pi * X) * np.cos(np.pi * Y),
            T=1.0 + 0.04 * np.sin(np.pi * X) * np.exp(-0.5 * Y),
        )
        sutherland_air(bf)

        borders = "lrbt"
        variables = "uvwT"

        L0, idx = build_lnse_operator(mesh, bf, kx=0.2, model="2D")
        L0_bc, _ = apply_dirichlet(L0, idx, borders, variables, mesh.ngp)

        L0_petsc, idx_petsc = build_lnse_operator_petsc(mesh, bf, kx=0.2, model="2D")
        try:
            L0_petsc_bc, _ = apply_dirichlet_petsc(
                L0_petsc, idx_petsc, borders, variables, mesh.ngp
            )
            report = petsc_matvec_residual(L0_bc, L0_petsc_bc, seed=17, trials=3)
            assert report["pass"], report
            assert report["max_abs_residual"] < 1e-10
            assert report["max_rel_residual"] < 1e-10
        finally:
            L0_petsc.destroy()

    def test_direct_petsc_post_neumann_l0_has_sparse_parity(self):
        """Direct PETSc Neumann enforcement matches the SciPy BC path."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=14,
            ny=10,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=1e4,
            Ma=0.3,
            Pr=0.7,
            kappa=1.4,
            T_0=288.15,
            RHO=1.0 + 0.03 * np.cos(np.pi * X) * np.cos(0.5 * np.pi * Y),
            U=1.0 + 0.08 * np.sin(np.pi * X) * np.exp(-0.8 * Y),
            V=0.04 * np.cos(0.5 * np.pi * X) * np.sin(np.pi * Y),
            W=0.03 * np.sin(1.5 * np.pi * X) * np.cos(0.5 * np.pi * Y),
            T=1.0 + 0.05 * np.cos(np.pi * X) * np.exp(-0.4 * Y),
        )
        sutherland_air(bf)

        borders = "bt"
        variables = "T"
        penalty = 1.0e4

        L0, idx = build_lnse_operator(mesh, bf, kx=0.35, model="2D")
        L0_bc, bc_indices = apply_neumann(L0, mesh, idx, borders, variables, penalty=penalty)

        L0_petsc, idx_petsc = build_lnse_operator_petsc(mesh, bf, kx=0.35, model="2D")
        try:
            L0_petsc_bc, bc_indices_petsc = apply_neumann_petsc(
                L0_petsc,
                mesh,
                idx_petsc,
                borders,
                variables,
                penalty=penalty,
            )
            np.testing.assert_array_equal(bc_indices_petsc, bc_indices)

            report = compare_petsc_against_scipy(L0_bc, L0_petsc_bc, check_structure=False)
            assert report["pass"], report
            assert report["max_abs_residual"] < 1e-10
        finally:
            L0_petsc.destroy()

    def test_direct_petsc_post_neumann_l0_has_matvec_parity(self):
        """Direct PETSc Neumann enforcement preserves operator action parity."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=12,
            ny=9,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=8e3,
            Ma=0.4,
            Pr=0.72,
            kappa=1.4,
            T_0=300.0,
            RHO=1.0 + 0.02 * np.cos(0.75 * np.pi * X) * np.cos(np.pi * Y),
            U=0.9 + 0.12 * np.sin(np.pi * X) * np.exp(-Y),
            V=0.03 * np.cos(np.pi * X) * np.sin(np.pi * Y),
            W=0.02 * np.sin(0.5 * np.pi * X) * np.cos(np.pi * Y),
            T=1.0 + 0.04 * np.sin(np.pi * X) * np.exp(-0.5 * Y),
        )
        sutherland_air(bf)

        borders = "lr"
        variables = "u"

        L0, idx = build_lnse_operator(mesh, bf, kx=0.2, model="2D")
        L0_bc, _ = apply_neumann(L0, mesh, idx, borders, variables)

        L0_petsc, idx_petsc = build_lnse_operator_petsc(mesh, bf, kx=0.2, model="2D")
        try:
            L0_petsc_bc, _ = apply_neumann_petsc(L0_petsc, mesh, idx_petsc, borders, variables)
            report = petsc_matvec_residual(L0_bc, L0_petsc_bc, seed=23, trials=3)
            assert report["pass"], report
            assert report["max_abs_residual"] < 1e-10
            assert report["max_rel_residual"] < 1e-10
        finally:
            L0_petsc.destroy()

    def test_neumann_rejects_ambiguous_corner_rules(self, small_mesh_2d, uniform_baseflow):
        """Overlapping Neumann borders on the same variable should fail at corners."""
        L0, idx = build_lnse_operator(small_mesh_2d, uniform_baseflow, kx=0.0, model="2D")
        with pytest.raises(ValueError, match="Ambiguous Neumann BC"):
            apply_neumann(L0, small_mesh_2d, idx, borders="lb", variables="u")

        L0_petsc, idx_petsc = build_lnse_operator_petsc(
            small_mesh_2d,
            uniform_baseflow,
            kx=0.0,
            model="2D",
        )
        try:
            with pytest.raises(ValueError, match="Ambiguous Neumann BC"):
                apply_neumann_petsc(L0_petsc, small_mesh_2d, idx_petsc, borders="lb", variables="u")
        finally:
            L0_petsc.destroy()

    def test_bc_recipe_petsc_matches_scipy_with_mixed_conditions(self):
        """Recipe-based PETSc BC application matches the SciPy mixed-BC path."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=13,
            ny=9,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=9e3,
            Ma=0.35,
            Pr=0.71,
            kappa=1.4,
            T_0=295.0,
            RHO=1.0 + 0.025 * np.cos(np.pi * X) * np.cos(0.75 * np.pi * Y),
            U=0.95 + 0.10 * np.sin(np.pi * X) * np.exp(-0.7 * Y),
            V=0.02 * np.cos(0.5 * np.pi * X) * np.sin(np.pi * Y),
            W=0.015 * np.sin(1.2 * np.pi * X) * np.cos(0.5 * np.pi * Y),
            T=1.0 + 0.03 * np.cos(0.8 * np.pi * X) * np.exp(-0.5 * Y),
        )
        sutherland_air(bf)

        recipe = [
            {"type": "dirichlet", "borders": "lt", "variables": "uvw"},
            {"type": "neumann", "borders": "b", "variables": "T"},
        ]
        penalty = 1.0e4

        L0, idx = build_lnse_operator(mesh, bf, kx=0.25, model="2D")
        L0_bc, bc_indices = apply_bc_recipe(L0, mesh, idx, recipe, penalty=penalty)

        L0_petsc, idx_petsc = build_lnse_operator_petsc(mesh, bf, kx=0.25, model="2D")
        try:
            L0_petsc_bc, bc_indices_petsc = apply_bc_recipe_petsc(
                L0_petsc,
                mesh,
                idx_petsc,
                recipe,
                penalty=penalty,
            )
            np.testing.assert_array_equal(bc_indices_petsc, bc_indices)

            sparse_report = compare_petsc_against_scipy(L0_bc, L0_petsc_bc, check_structure=False)
            assert sparse_report["pass"], sparse_report
            assert sparse_report["max_abs_residual"] < 1e-10

            matvec_report = petsc_matvec_residual(L0_bc, L0_petsc_bc, seed=29, trials=3)
            assert matvec_report["pass"], matvec_report
            assert matvec_report["max_abs_residual"] < 1e-10
            assert matvec_report["max_rel_residual"] < 1e-10
        finally:
            L0_petsc.destroy()

    def test_build_lnse_system_scipy_matches_reference_pipeline(self):
        """Workflow helper should reproduce the existing SciPy assembly+BC path."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=13,
            ny=9,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=9e3,
            Ma=0.35,
            Pr=0.71,
            kappa=1.4,
            T_0=295.0,
            RHO=1.0 + 0.025 * np.cos(np.pi * X) * np.cos(0.75 * np.pi * Y),
            U=0.95 + 0.10 * np.sin(np.pi * X) * np.exp(-0.7 * Y),
            V=0.02 * np.cos(0.5 * np.pi * X) * np.sin(np.pi * Y),
            W=0.015 * np.sin(1.2 * np.pi * X) * np.cos(0.5 * np.pi * Y),
            T=1.0 + 0.03 * np.cos(0.8 * np.pi * X) * np.exp(-0.5 * Y),
        )
        sutherland_air(bf)

        recipe = [
            {"type": "dirichlet", "borders": "lt", "variables": "uvw"},
            {"type": "neumann", "borders": "b", "variables": "T"},
        ]
        penalty = 1.0e4

        L0_ref, idx_ref = build_lnse_operator(mesh, bf, kx=0.25, model="2D")
        L0_ref, bc_indices_ref = apply_bc_recipe(L0_ref, mesh, idx_ref, recipe, penalty=penalty)

        L0_sys, idx_sys, bc_indices_sys = build_lnse_system(
            mesh,
            bf,
            kx=0.25,
            model="2D",
            bc_recipe=recipe,
            bc_penalty=penalty,
            backend="scipy",
        )

        for key, value in idx_ref.items():
            assert key in idx_sys
            np.testing.assert_array_equal(idx_sys[key], value)
        np.testing.assert_array_equal(bc_indices_sys, bc_indices_ref)

        delta = (L0_sys - L0_ref).tocsr()
        assert delta.nnz == 0 or np.max(np.abs(delta.data)) < 1e-12

    def test_build_lnse_system_petsc_matches_recipe_path(self):
        """Workflow helper should reproduce the PETSc direct-assembly+recipe path."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=13,
            ny=9,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=9e3,
            Ma=0.35,
            Pr=0.71,
            kappa=1.4,
            T_0=295.0,
            RHO=1.0 + 0.025 * np.cos(np.pi * X) * np.cos(0.75 * np.pi * Y),
            U=0.95 + 0.10 * np.sin(np.pi * X) * np.exp(-0.7 * Y),
            V=0.02 * np.cos(0.5 * np.pi * X) * np.sin(np.pi * Y),
            W=0.015 * np.sin(1.2 * np.pi * X) * np.cos(0.5 * np.pi * Y),
            T=1.0 + 0.03 * np.cos(0.8 * np.pi * X) * np.exp(-0.5 * Y),
        )
        sutherland_air(bf)

        recipe = [
            {"type": "dirichlet", "borders": "lt", "variables": "uvw"},
            {"type": "neumann", "borders": "b", "variables": "T"},
        ]
        penalty = 1.0e4

        L0_ref, idx_ref = build_lnse_operator(mesh, bf, kx=0.25, model="2D")
        L0_ref, bc_indices_ref = apply_bc_recipe(L0_ref, mesh, idx_ref, recipe, penalty=penalty)

        L0_sys, idx_sys, bc_indices_sys = build_lnse_system(
            mesh,
            bf,
            kx=0.25,
            model="2D",
            bc_recipe=recipe,
            bc_penalty=penalty,
            backend="petsc",
        )
        try:
            for key, value in idx_ref.items():
                assert key in idx_sys
                np.testing.assert_array_equal(idx_sys[key], value)
            np.testing.assert_array_equal(bc_indices_sys, bc_indices_ref)

            sparse_report = compare_petsc_against_scipy(L0_ref, L0_sys, check_structure=False)
            assert sparse_report["pass"], sparse_report
            assert sparse_report["max_abs_residual"] < 1e-10

            matvec_report = petsc_matvec_residual(L0_ref, L0_sys, seed=31, trials=3)
            assert matvec_report["pass"], matvec_report
            assert matvec_report["max_abs_residual"] < 1e-10
            assert matvec_report["max_rel_residual"] < 1e-10
        finally:
            L0_sys.destroy()

    def test_build_lnse_system_auto_selects_petsc_for_local_2d(self):
        """Auto backend should select PETSc for the validated local 2D path."""
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

        L0_sys, _, _ = build_lnse_system(
            mesh,
            bf,
            kx=0.2,
            model="2D",
            backend="auto",
            solver_cfg={"petsc_matrix_mode": "local"},
        )
        try:
            assert hasattr(L0_sys, "getValuesCSR")
        finally:
            L0_sys.destroy()

    def test_build_lnse_system_auto_uses_direct_distributed_petsc_for_supported_2d(
        self, monkeypatch
    ):
        """Auto backend should use direct distributed PETSc assembly for supported 2D cases."""
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

        L0_ref, idx_ref, bc_ref = build_lnse_system(
            mesh,
            bf,
            kx=0.2,
            model="2D",
            bc_recipe=recipe,
            solver_cfg={"petsc_matrix_mode": "local"},
        )

        def _unexpected_fallback(*args, **kwargs):
            raise AssertionError(
                "SciPy-to-PETSc fallback should not be used for supported distributed 2D direct assembly"
            )

        monkeypatch.setattr(petsc_core, "scipy_to_petsc", _unexpected_fallback)

        L0_sys, idx_sys, bc_sys = build_lnse_system(
            mesh,
            bf,
            kx=0.2,
            model="2D",
            bc_recipe=recipe,
            backend="auto",
            solver_cfg={"petsc_matrix_mode": "distributed"},
        )
        try:
            for key, value in idx_ref.items():
                assert key in idx_sys
                np.testing.assert_array_equal(idx_sys[key], value)
            np.testing.assert_array_equal(bc_sys, bc_ref)

            report = compare_petsc_against_scipy(L0_ref, L0_sys, check_structure=False)
            assert report["pass"], report
            assert report["max_abs_residual"] < 1e-10

            matvec_report = petsc_matvec_residual(L0_ref, L0_sys, seed=41, trials=3)
            assert matvec_report["pass"], matvec_report
            assert matvec_report["max_abs_residual"] < 1e-10
            assert matvec_report["max_rel_residual"] < 1e-10
        finally:
            L0_sys.destroy()

    def test_build_lnse_system_petsc_uses_direct_distributed_petsc_for_supported_2d(
        self, monkeypatch
    ):
        """Explicit PETSc backend should use direct distributed PETSc assembly for supported 2D cases."""
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
        recipe = [{"type": "neumann", "borders": "lr", "variables": "u"}]

        L0_ref, idx_ref, bc_ref = build_lnse_system(
            mesh,
            bf,
            kx=0.2,
            model="2D",
            bc_recipe=recipe,
            solver_cfg={"petsc_matrix_mode": "local"},
        )

        def _unexpected_fallback(*args, **kwargs):
            raise AssertionError(
                "SciPy-to-PETSc fallback should not be used for supported distributed 2D direct assembly"
            )

        monkeypatch.setattr(petsc_core, "scipy_to_petsc", _unexpected_fallback)

        L0_sys, idx_sys, bc_sys = build_lnse_system(
            mesh,
            bf,
            kx=0.2,
            model="2D",
            bc_recipe=recipe,
            backend="petsc",
            solver_cfg={"petsc_matrix_mode": "distributed"},
        )
        try:
            for key, value in idx_ref.items():
                assert key in idx_sys
                np.testing.assert_array_equal(idx_sys[key], value)
            np.testing.assert_array_equal(bc_sys, bc_ref)

            report = compare_petsc_against_scipy(L0_ref, L0_sys, check_structure=False)
            assert report["pass"], report
            assert report["max_abs_residual"] < 1e-10

            matvec_report = petsc_matvec_residual(L0_ref, L0_sys, seed=43, trials=3)
            assert matvec_report["pass"], matvec_report
            assert matvec_report["max_abs_residual"] < 1e-10
            assert matvec_report["max_rel_residual"] < 1e-10
        finally:
            L0_sys.destroy()

    @pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec not available")
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_build_lnse_system_petsc_real_mpi_assembly_matches_owned_scipy_rows(self, n_ranks):
        """Real MPI direct PETSc assembly should match the owned SciPy reference rows."""
        proc = run_mpi_python(n_ranks, _build_lnse_system_mpi_script(bc_recipe=None))
        if proc.returncode != 0:
            pytest.fail(
                f"{n_ranks}-rank assembly MPI harness failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        reports = parse_mpi_reports(proc.stdout)
        assert_split_ownership(reports, n_ranks)
        assert all(report["nnz"] > 0 for report in reports)

    @pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec not available")
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_build_lnse_system_petsc_real_mpi_dirichlet_matches_owned_scipy_rows(self, n_ranks):
        """Real MPI Dirichlet BC application should match the owned SciPy reference rows."""
        recipe = [{"type": "dirichlet", "borders": "bt", "variables": "uvwT"}]
        proc = run_mpi_python(n_ranks, _build_lnse_system_mpi_script(bc_recipe=recipe))
        if proc.returncode != 0:
            pytest.fail(
                f"{n_ranks}-rank Dirichlet MPI harness failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        reports = parse_mpi_reports(proc.stdout)
        assert_split_ownership(reports, n_ranks)
        assert all(report["nnz"] > 0 for report in reports)

    @pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec not available")
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_build_lnse_system_petsc_real_mpi_neumann_matches_owned_scipy_rows(self, n_ranks):
        """Real MPI Neumann BC application should match the owned SciPy reference rows."""
        recipe = [{"type": "neumann", "borders": "lr", "variables": "u"}]
        proc = run_mpi_python(n_ranks, _build_lnse_system_mpi_script(bc_recipe=recipe))
        if proc.returncode != 0:
            pytest.fail(
                f"{n_ranks}-rank Neumann MPI harness failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        reports = parse_mpi_reports(proc.stdout)
        assert_split_ownership(reports, n_ranks)
        assert all(report["nnz"] > 0 for report in reports)

    def test_build_lnse_system_petsc_matches_axy_recipe_path(self):
        """Explicit PETSc backend should reproduce the SciPy axy workflow path."""
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

        L0_ref, idx_ref = build_lnse_operator(mesh, bf, model="axy", m=1)
        L0_ref, bc_indices_ref = apply_bc_recipe(L0_ref, mesh, idx_ref, recipe)

        L0_sys, idx_sys, bc_indices_sys = build_lnse_system(
            mesh,
            bf,
            model="axy",
            m=1,
            bc_recipe=recipe,
            backend="petsc",
        )
        try:
            for key, value in idx_ref.items():
                assert key in idx_sys
                np.testing.assert_array_equal(idx_sys[key], value)
            np.testing.assert_array_equal(bc_indices_sys, bc_indices_ref)

            sparse_report = compare_petsc_against_scipy(L0_ref, L0_sys, check_structure=False)
            assert sparse_report["pass"], sparse_report
            assert sparse_report["max_abs_residual"] < 1e-10

            matvec_report = petsc_matvec_residual(L0_ref, L0_sys, seed=37, trials=3)
            assert matvec_report["pass"], matvec_report
            assert matvec_report["max_abs_residual"] < 1e-10
            assert matvec_report["max_rel_residual"] < 1e-10
        finally:
            L0_sys.destroy()

    def test_build_lnse_system_auto_selects_petsc_for_local_axy(self):
        """Auto backend should select PETSc for the validated local axy path."""
        mesh = Mesh.create(
            xrange=(0.0, 10.0),
            yrange=(0.0, 3.0),
            nx=12,
            ny=9,
            fd_order=4,
            y_symmetry=True,
        )
        R = mesh.Y
        bf = BaseFlow(
            Re=8e3,
            Ma=0.8,
            Pr=0.7,
            kappa=1.4,
            T_0=288.15,
            RHO=np.ones_like(R),
            U=0.85 / np.cosh(0.8 * R) ** 2,
            V=np.zeros_like(R),
            W=np.zeros_like(R),
            T=np.ones_like(R),
        )
        sutherland_air(bf)

        L0_sys, _, _ = build_lnse_system(
            mesh,
            bf,
            model="axy",
            m=0,
            backend="auto",
            solver_cfg={"petsc_matrix_mode": "local"},
        )
        try:
            assert hasattr(L0_sys, "getValuesCSR")
        finally:
            L0_sys.destroy()

    def test_build_lnse_system_default_backend_is_scipy(self):
        """Unset backend should preserve the existing SciPy-first workflow default."""
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

        L0_sys, _, _ = build_lnse_system(
            mesh,
            bf,
            kx=0.2,
            model="2D",
            solver_cfg={"petsc_matrix_mode": "local"},
        )
        assert sp.issparse(L0_sys)

    def test_build_lnse_system_config_auto_matches_explicit_auto_for_distributed_axy(self):
        """Config-driven auto should match explicit auto fallback behavior for distributed axy."""
        mesh = Mesh.create(
            xrange=(0.0, 10.0),
            yrange=(0.0, 3.0),
            nx=12,
            ny=9,
            fd_order=4,
            y_symmetry=True,
        )
        R = mesh.Y
        bf = BaseFlow(
            Re=8e3,
            Ma=0.8,
            Pr=0.7,
            kappa=1.4,
            T_0=288.15,
            RHO=np.ones_like(R),
            U=0.85 / np.cosh(0.8 * R) ** 2,
            V=np.zeros_like(R),
            W=np.zeros_like(R),
            T=np.ones_like(R),
        )
        sutherland_air(bf)
        recipe = [{"type": "dirichlet", "borders": "ltr", "variables": "uvwT"}]
        solver_cfg_explicit = {"petsc_matrix_mode": "distributed"}
        solver_cfg_config = {
            "assembly_backend": "auto",
            "petsc_matrix_mode": "distributed",
        }

        L0_explicit, idx_explicit, bc_explicit = build_lnse_system(
            mesh,
            bf,
            model="axy",
            m=0,
            bc_recipe=recipe,
            backend="auto",
            solver_cfg=solver_cfg_explicit,
        )
        L0_config, idx_config, bc_config = build_lnse_system(
            mesh,
            bf,
            model="axy",
            m=0,
            bc_recipe=recipe,
            solver_cfg=solver_cfg_config,
        )
        L0_ref, idx_ref, bc_ref = build_lnse_system(
            mesh,
            bf,
            model="axy",
            m=0,
            bc_recipe=recipe,
            backend="scipy",
        )
        try:
            for key, value in idx_ref.items():
                assert key in idx_explicit
                assert key in idx_config
                np.testing.assert_array_equal(idx_explicit[key], value)
                np.testing.assert_array_equal(idx_config[key], value)
            np.testing.assert_array_equal(bc_explicit, bc_ref)
            np.testing.assert_array_equal(bc_config, bc_ref)

            explicit_report = compare_petsc_against_scipy(
                L0_ref, L0_explicit, check_structure=False
            )
            config_report = compare_petsc_against_scipy(L0_ref, L0_config, check_structure=False)
            assert explicit_report["pass"], explicit_report
            assert config_report["pass"], config_report
            assert explicit_report["max_abs_residual"] < 1e-10
            assert config_report["max_abs_residual"] < 1e-10
        finally:
            L0_explicit.destroy()
            L0_config.destroy()


# =========================================================================
# TestChuEnergyNorm
# =========================================================================


class TestChuEnergyNorm:
    """Tests for compute_chu_norm."""

    NGP = 50
    KAPPA = 1.4
    MA = 0.3

    def _make_norm(self, *, model="2D", R=None):
        """Compute Chu norm with RHO=T=1, W_int=ones."""
        ngp = self.NGP
        W_int = np.ones(ngp)
        RHO = np.ones(ngp)
        T = np.ones(ngp)
        return compute_chu_norm(
            W_int,
            RHO,
            T,
            self.KAPPA,
            self.MA,
            ngp,
            model=model,
            R=R,
        )

    def test_shapes(self):
        """W_diag and invW_diag have shape (5*ngp,)."""
        W_diag, invW_diag = self._make_norm()
        assert W_diag.shape == (5 * self.NGP,)
        assert invW_diag.shape == (5 * self.NGP,)

    def test_positive(self):
        """W_diag > 0 everywhere."""
        W_diag, _ = self._make_norm()
        assert np.all(W_diag > 0)

    def test_inverse_relation(self):
        """W_diag * invW_diag ≈ 1."""
        W_diag, invW_diag = self._make_norm()
        np.testing.assert_allclose(W_diag * invW_diag, 1.0, rtol=1e-12)

    def test_rho_block(self):
        """rho block: W[0:ngp] = 0.5 * T / (RHO * kappa * Ma^2) * W_int."""
        W_diag, _ = self._make_norm()
        ngp = self.NGP
        expected = 0.5 * 1.0 / (1.0 * self.KAPPA * self.MA**2) * np.ones(ngp)
        np.testing.assert_allclose(W_diag[:ngp], expected, rtol=1e-12)

    def test_velocity_blocks(self):
        """Velocity blocks: W[ngp:4*ngp] = 0.5 * RHO * W_int."""
        W_diag, _ = self._make_norm()
        ngp = self.NGP
        expected = 0.5 * np.ones(3 * ngp)
        np.testing.assert_allclose(W_diag[ngp : 4 * ngp], expected, rtol=1e-12)

    def test_T_block(self):
        """T block: W[4*ngp:5*ngp] = 0.5 * RHO / (kappa*(kappa-1)*T*Ma^2) * W_int."""
        W_diag, _ = self._make_norm()
        ngp = self.NGP
        expected = 0.5 * 1.0 / (self.KAPPA * (self.KAPPA - 1.0) * 1.0 * self.MA**2) * np.ones(ngp)
        np.testing.assert_allclose(W_diag[4 * ngp : 5 * ngp], expected, rtol=1e-12)

    def test_axy_model_R_scaling(self):
        """model='axy': R is multiplied into W_int."""
        ngp = self.NGP
        R = np.full(ngp, 3.0)
        W_diag_cart, _ = self._make_norm(model="2D")
        W_diag_cyl, _ = self._make_norm(model="axy", R=R)
        # Each block should be scaled by R=3 relative to the 2D case.
        # Check the rho block as representative.
        np.testing.assert_allclose(
            W_diag_cyl[:ngp],
            3.0 * W_diag_cart[:ngp],
            rtol=1e-12,
        )


# =========================================================================
# TestSpatialOperator
# =========================================================================


class TestSpatialOperator:
    """Parity checks for spatial SciPy/PETSc operator assembly."""

    def test_direct_petsc_spatial_operator_has_sparse_parity(self):
        """Direct PETSc spatial matrices should match the SciPy reference entrywise."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=12,
            ny=9,
            fd_order=4,
        )
        X, Y = mesh.X, mesh.Y
        bf = BaseFlow(
            Re=8e3,
            Ma=0.35,
            Pr=0.71,
            kappa=1.4,
            T_0=295.0,
            RHO=1.0 + 0.02 * np.cos(np.pi * X) * np.cos(0.7 * np.pi * Y),
            U=0.95 + 0.07 * np.sin(np.pi * X) * np.exp(-0.6 * Y),
            V=0.015 * np.cos(0.5 * np.pi * X) * np.sin(np.pi * Y),
            W=0.012 * np.sin(1.1 * np.pi * X) * np.cos(0.4 * np.pi * Y),
            T=1.0 + 0.03 * np.cos(0.8 * np.pi * X) * np.exp(-0.5 * Y),
        )
        sutherland_air(bf)

        Lc_sp, R_sp, Lw_sp, idx_sp = build_spatial_operator(mesh, bf, model="2D", delta=0.4)
        Lc_petsc, R_petsc, Lw_petsc, idx_petsc = build_spatial_operator_petsc(
            mesh,
            bf,
            model="2D",
            delta=0.4,
        )
        try:
            for key, value in idx_sp.items():
                assert key in idx_petsc
                np.testing.assert_array_equal(idx_petsc[key], value)

            for ref, candidate in ((Lc_sp, Lc_petsc), (R_sp, R_petsc), (Lw_sp, Lw_petsc)):
                report = compare_petsc_against_scipy(ref, candidate, check_structure=False)
                assert report["pass"], report
                assert report["max_abs_residual"] < 1e-10
        finally:
            Lc_petsc.destroy()
            R_petsc.destroy()
            Lw_petsc.destroy()

    def test_direct_petsc_spatial_operator_with_dirichlet_has_sparse_parity(self):
        """Spatial Dirichlet enforcement should match between SciPy and PETSc paths."""
        mesh = Mesh.create(
            xrange=(200.0, 300.0),
            yrange=(0.0, 30.0),
            nx=15,
            ny=10,
            fd_order=4,
        )
        s = 3.0
        Y_new = 30.0 * np.sinh(s * mesh.Y / 30.0) / np.sinh(s)
        mesh = mesh.deform(Y_new=Y_new)

        bf = load_blasius(mesh.X, mesh.Y, Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15)
        sutherland_air(bf)

        Lc_sp, R_sp, _, idx = build_spatial_operator(mesh, bf, model="2D")
        Lc_sp, R_sp, bc_indices = apply_dirichlet_spatial(Lc_sp, R_sp, idx, "bt", "uvwT", mesh.ngp)

        Lc_petsc, R_petsc, Lw_petsc, idx_petsc = build_spatial_operator_petsc(mesh, bf, model="2D")
        try:
            Lc_petsc, R_petsc, bc_indices_petsc = apply_dirichlet_spatial_petsc(
                Lc_petsc,
                R_petsc,
                idx_petsc,
                "bt",
                "uvwT",
                mesh.ngp,
            )
            np.testing.assert_array_equal(bc_indices_petsc, bc_indices)

            lc_report = compare_petsc_against_scipy(Lc_sp, Lc_petsc, check_structure=False)
            r_report = compare_petsc_against_scipy(R_sp, R_petsc, check_structure=False)
            assert lc_report["pass"], lc_report
            assert r_report["pass"], r_report
            assert lc_report["max_abs_residual"] < 1e-10
            assert r_report["max_abs_residual"] < 1e-10
        finally:
            Lc_petsc.destroy()
            R_petsc.destroy()
            Lw_petsc.destroy()


# =========================================================================
# TestAxyParity
# =========================================================================


class TestAxyParity:
    """Tests for parity logic in _create_axy_diff_matrices."""

    @pytest.fixture
    def symmetric_mesh(self):
        """20x30 mesh with y_symmetry on [0,2]x[0,1]."""
        return Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=20,
            ny=30,
            fd_order=4,
            y_symmetry=True,
        )

    def test_parity_m0(self, symmetric_mesh):
        """m=0: rho,u,T symmetric; v,w antisymmetric."""
        DR, D2R, DZ, D2Z, D2RZ = _create_axy_diff_matrices(symmetric_mesh, m=0)
        # DR is a tuple of 5 blocks; symmetric vars use Dy_symm
        # Check that symmetric blocks (rho=0, u=1, T=4) use the
        # symmetric derivative, and antisymmetric blocks (v=2, w=3) differ.
        assert symmetric_mesh.Dy_symm is not None
        Dy_sym = symmetric_mesh.Dy_symm.tocsc()
        Dy_asym = symmetric_mesh.Dy_asymm.tocsc()
        # rho (idx 0): symmetric
        np.testing.assert_array_equal(DR[0].toarray(), Dy_sym.toarray())
        # u (idx 1): symmetric
        np.testing.assert_array_equal(DR[1].toarray(), Dy_sym.toarray())
        # v (idx 2): antisymmetric
        np.testing.assert_array_equal(DR[2].toarray(), Dy_asym.toarray())
        # w (idx 3): antisymmetric
        np.testing.assert_array_equal(DR[3].toarray(), Dy_asym.toarray())
        # T (idx 4): symmetric
        np.testing.assert_array_equal(DR[4].toarray(), Dy_sym.toarray())

    def test_parity_m1(self, symmetric_mesh):
        """m=1: rho,u,T antisymmetric; v,w symmetric."""
        DR, D2R, DZ, D2Z, D2RZ = _create_axy_diff_matrices(symmetric_mesh, m=1)
        Dy_sym = symmetric_mesh.Dy_symm.tocsc()
        Dy_asym = symmetric_mesh.Dy_asymm.tocsc()
        # rho (idx 0): antisymmetric
        np.testing.assert_array_equal(DR[0].toarray(), Dy_asym.toarray())
        # u (idx 1): antisymmetric
        np.testing.assert_array_equal(DR[1].toarray(), Dy_asym.toarray())
        # v (idx 2): symmetric
        np.testing.assert_array_equal(DR[2].toarray(), Dy_sym.toarray())
        # w (idx 3): symmetric
        np.testing.assert_array_equal(DR[3].toarray(), Dy_sym.toarray())
        # T (idx 4): antisymmetric
        np.testing.assert_array_equal(DR[4].toarray(), Dy_asym.toarray())

    def test_parity_m2_all_asymmetric(self, symmetric_mesh):
        """m>=2: all variables use antisymmetric derivatives."""
        DR, D2R, DZ, D2Z, D2RZ = _create_axy_diff_matrices(symmetric_mesh, m=2)
        Dy_asym = symmetric_mesh.Dy_asymm.tocsc()
        for k in range(5):
            np.testing.assert_array_equal(
                DR[k].toarray(),
                Dy_asym.toarray(),
                err_msg=f"DR[{k}] should be antisymmetric for m>=2",
            )

    def test_parity_m5_all_asymmetric(self, symmetric_mesh):
        """m=5 also gives all antisymmetric (same as m>=2)."""
        DR5, _, _, _, _ = _create_axy_diff_matrices(symmetric_mesh, m=5)
        DR2, _, _, _, _ = _create_axy_diff_matrices(symmetric_mesh, m=2)
        for k in range(5):
            np.testing.assert_array_equal(
                DR5[k].toarray(),
                DR2[k].toarray(),
                err_msg=f"DR[{k}] should be identical for m=2 and m=5",
            )

    def test_no_symmetry_mesh_uses_standard_Dy(self):
        """Without y_symmetry, all variables use the standard Dy."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=20,
            ny=30,
            fd_order=4,
            y_symmetry=False,
        )
        DR, D2R, DZ, D2Z, D2RZ = _create_axy_diff_matrices(mesh, m=0)
        Dy = mesh.Dy.tocsc()
        for k in range(5):
            np.testing.assert_array_equal(
                DR[k].toarray(),
                Dy.toarray(),
                err_msg=f"DR[{k}] should be standard Dy without y_symmetry",
            )

    def test_DZ_blocks_always_same(self, symmetric_mesh):
        """Axial derivatives (DZ) are always Dx, regardless of parity."""
        _, _, DZ, D2Z, _ = _create_axy_diff_matrices(symmetric_mesh, m=0)
        Dx = symmetric_mesh.Dx.tocsc()
        for k in range(5):
            np.testing.assert_array_equal(DZ[k].toarray(), Dx.toarray())

    def test_invalid_m_raises(self, symmetric_mesh):
        """Negative or non-integer m raises ValueError."""
        with pytest.raises(ValueError, match="non-negative integer"):
            _create_axy_diff_matrices(symmetric_mesh, m=-1)
        with pytest.raises(ValueError, match="non-negative integer"):
            _create_axy_diff_matrices(symmetric_mesh, m=0.5)


# =========================================================================
# TestInvBBlockInverse
# =========================================================================


class TestInvBBlockInverse:
    """Test the analytical 5x5 block inverse of B in cylindrical coordinates."""

    @staticmethod
    def _build_B_matrix(ngp, RHO, U, V, W, T, kappa, Ma):
        """Construct B/1j as a dense (5*ngp, 5*ngp) block matrix.

        B/1j in physical ordering [rho, u, v, w, T]:
          [  1     0     0     0     0    ]
          [  U     RHO   0     0     0    ]
          [  V     0     RHO   0     0    ]
          [  W     0     0     RHO   0    ]
          [  KE    U*RHO V*RHO W*RHO cv*RHO]

        where KE = 0.5*(U^2+V^2+W^2) + cv*T and cv = 1/(kappa*(kappa-1)*Ma^2).
        """
        cv = 1.0 / (kappa * (kappa - 1.0) * Ma**2)
        KE = 0.5 * (U**2 + V**2 + W**2) + cv * T

        def D(v):
            return sp.diags(v, format="csc")

        Z = sp.csc_matrix((ngp, ngp))
        blocks = [
            [D(np.ones(ngp)), Z, Z, Z, Z],
            [D(U), D(RHO), Z, Z, Z],
            [D(V), Z, D(RHO), Z, Z],
            [D(W), Z, Z, D(RHO), Z],
            [D(KE), D(U * RHO), D(V * RHO), D(W * RHO), D(cv * RHO)],
        ]
        return sp.bmat(blocks, format="csc")

    def test_inv_B_times_B_is_identity(self):
        """inv(B/1j) @ (B/1j) = I for random flow fields."""
        ngp = 10
        rng = np.random.default_rng(42)
        RHO = 1.0 + rng.random(ngp)
        U = rng.standard_normal(ngp)
        V = rng.standard_normal(ngp)
        W = rng.standard_normal(ngp)
        T = 0.5 + rng.random(ngp)
        kappa, Ma = 1.4, 0.5

        B = self._build_B_matrix(ngp, RHO, U, V, W, T, kappa, Ma)
        # _apply_inv_B computes inv(B/1j) @ L0; pass L0=I to get inv(B/1j)
        I_5n = sp.eye(5 * ngp, format="csc")
        inv_B = _apply_inv_B(I_5n, "axy", ngp, RHO, U, V, W, T, kappa, Ma)

        product = (inv_B @ B).toarray()
        expected = np.eye(5 * ngp)
        np.testing.assert_allclose(
            product,
            expected,
            atol=1e-12,
            err_msg="inv(B/1j) @ (B/1j) should equal I",
        )

    def test_cartesian_inv_B_diagonal(self):
        """Cartesian inv(B) is diagonal: diag(1, 1/RHO, 1/RHO, 1/RHO, 1/RHO)."""
        ngp = 10
        rng = np.random.default_rng(99)
        RHO = 1.0 + rng.random(ngp)
        U = rng.standard_normal(ngp)
        V = rng.standard_normal(ngp)
        W = rng.standard_normal(ngp)
        T = 0.5 + rng.random(ngp)
        kappa, Ma = 1.4, 0.3

        I_5n = sp.eye(5 * ngp, format="csc")
        inv_B = _apply_inv_B(I_5n, "2d", ngp, RHO, U, V, W, T, kappa, Ma)

        # Should be diagonal
        dense = inv_B.toarray()
        expected_diag = np.concatenate(
            [
                np.ones(ngp),
                1.0 / RHO,
                1.0 / RHO,
                1.0 / RHO,
                1.0 / RHO,
            ]
        )
        np.testing.assert_allclose(
            np.diag(dense),
            expected_diag,
            atol=1e-14,
        )
        # Off-diagonal should be zero
        off_diag = dense - np.diag(np.diag(dense))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-14)

    def test_inv_B_uniform_flow(self):
        """With uniform RHO=1, U=V=W=0, inv(B/1j) has a simple structure."""
        ngp = 5
        RHO = np.ones(ngp)
        U = np.zeros(ngp)
        V = np.zeros(ngp)
        W = np.zeros(ngp)
        T = np.ones(ngp)
        kappa, Ma = 1.4, 0.3

        I_5n = sp.eye(5 * ngp, format="csc")
        inv_B = _apply_inv_B(I_5n, "axy", ngp, RHO, U, V, W, T, kappa, Ma)
        dense = inv_B.toarray()

        # With zero velocity, off-diagonal of momentum rows vanish (U/RHO=0),
        # and the energy row simplifies: e40 = -T/RHO = -1.
        cv = 1.0 / (kappa * (kappa - 1.0) * Ma**2)
        # Expected diagonal
        expected_diag = np.concatenate(
            [
                np.ones(ngp),  # rho
                np.ones(ngp),  # u: 1/RHO
                np.ones(ngp),  # v: 1/RHO
                np.ones(ngp),  # w: 1/RHO
                np.full(ngp, 1.0 / cv),  # T: 1/(RHO*cv)
            ]
        )
        np.testing.assert_allclose(np.diag(dense), expected_diag, atol=1e-14)

        # inv(B)[4,0] = KE_half/(RHO*cv) - T/RHO.
        # With U=V=W=0: = 0 - 1 = -1
        for k in range(ngp):
            row_T = 4 * ngp + k
            col_rho = k
            np.testing.assert_allclose(
                dense[row_T, col_rho],
                -1.0,
                atol=1e-14,
                err_msg=f"inv(B)[4,0] at gp {k}: expected -1 (U=V=W=0, T=RHO=1)",
            )
