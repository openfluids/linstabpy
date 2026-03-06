"""Comprehensive tests for the linstab2d.mesh module.

Covers SBP stencils, 1-D derivative operators, integration, mesh creation,
deformation, masking, pole stencils, and periodic operators.

Notes on the 2-D operator layout
---------------------------------
The 2-D operator assembly in ``_build_operators`` iterates over unique
coordinate values to build 1-D SBP operators along each grid line
(matching MATLAB ``AddDW2mesh.m``).  Data is flattened in column-major
(Fortran) order to match MATLAB indexing.  The operators satisfy the
standard convention:

    - ``Dx @ x_flat = 1``  (Dx differentiates along x)
    - ``Dx @ y_flat = 0``  (Dx does not differentiate along y)
    - ``Dy @ y_flat = 1``  (Dy differentiates along y)
    - ``Dy @ x_flat = 0``  (Dy does not differentiate along x)

This works for both square (``nx == ny``) and rectangular (``nx != ny``)
grids.

Pole-condition operators (``build_sbp_pole_matrices``) only work for
**order 2**; higher orders raise a broadcasting error in the Mohseni
pole-stencil code.  Tests are parametrised only over order 2.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from linstab2d.mesh import (
    Mesh,
    build_sbp_matrix,
    build_sbp_pole_matrices,
    min_points,
    sbp_stencils,
)
from linstab2d.mesh.mesh import _build_periodic_1d_operators


# ===================================================================
# Helper: square-mesh factory (nx == ny)
# ===================================================================

def _square_mesh(n: int = 30, fd_order: int = 4, L: float = 1.0) -> Mesh:
    """Create a square mesh on [0, L] x [0, L] with n x n points."""
    return Mesh.create(
        xrange=(0.0, L),
        yrange=(0.0, L),
        nx=n,
        ny=n,
        fd_order=fd_order,
    )


def _deep_interior(n: int, cw: int) -> np.ndarray:
    """Return flat indices (column-major, nx==ny==n) of deep-interior points,
    skipping *cw* rows/cols from every boundary."""
    idx = []
    for i in range(cw, n - cw):
        for j in range(cw, n - cw):
            idx.append(j + i * n)
    return np.array(idx)


# ===================================================================
# TestSBPStencils
# ===================================================================
class TestSBPStencils:
    """Verify SBP stencil properties for all supported orders."""

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_required_keys(self, order):
        s = sbp_stencils(order)
        for key in ("a_i1", "a_i2", "a_b1", "a_b2"):
            assert key in s, f"Missing key '{key}' for order {order}"

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_interior_d1_antisymmetric(self, order):
        a = sbp_stencils(order)["a_i1"]
        n = len(a)
        for k in range(n):
            assert a[k] == pytest.approx(-a[n - 1 - k], abs=1e-15), (
                f"a_i1 not antisymmetric at k={k}, order={order}"
            )

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_interior_d2_symmetric(self, order):
        a = sbp_stencils(order)["a_i2"]
        n = len(a)
        for k in range(n):
            assert a[k] == pytest.approx(a[n - 1 - k], abs=1e-15), (
                f"a_i2 not symmetric at k={k}, order={order}"
            )

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_interior_stencils_sum_to_zero(self, order):
        s = sbp_stencils(order)
        assert sum(s["a_i1"]) == pytest.approx(0.0, abs=1e-15)
        assert sum(s["a_i2"]) == pytest.approx(0.0, abs=1e-15)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            sbp_stencils(3)
        with pytest.raises(ValueError):
            sbp_stencils(0)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_min_points_matches_closure_width(self, order):
        s = sbp_stencils(order)
        assert min_points(order) == s["a_b1"].shape[1]


# ===================================================================
# TestSBPProperty
# ===================================================================
class TestSBPProperty:
    """Verify fundamental SBP properties: constant preservation, row-sum
    condition, and (for order 2) the full SBP identity Q + Q^T = B."""

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    @pytest.mark.parametrize("n", [20, 40, 60, 80])
    def test_constant_preservation(self, order, n):
        """D1 @ 1 = 0 (constant is in the null space)."""
        if n < min_points(order):
            pytest.skip(f"n={n} too small for order {order}")
        dz = 1.0 / (n - 1)
        D1, _ = build_sbp_matrix(n, dz, order)
        result = D1 @ np.ones(n)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    @pytest.mark.parametrize("n", [20, 40, 60, 80])
    def test_d2_constant_preservation(self, order, n):
        """D2 @ 1 = 0."""
        if n < min_points(order):
            pytest.skip(f"n={n} too small for order {order}")
        dz = 1.0 / (n - 1)
        _, D2 = build_sbp_matrix(n, dz, order)
        result = D2 @ np.ones(n)
        # D2 accumulates slightly more rounding error than D1 for higher orders
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    @pytest.mark.parametrize("n", [20, 40, 60, 80])
    def test_row_sums_zero(self, order, n):
        """Each row of D1 sums to zero (row-sum condition)."""
        if n < min_points(order):
            pytest.skip(f"n={n} too small for order {order}")
        dz = 1.0 / (n - 1)
        D1, _ = build_sbp_matrix(n, dz, order)
        row_sums = np.array(D1.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_sbp_identity_order2(self):
        """For order 2 the SBP norm is trapezoidal and Q + Q^T = B."""
        n = 40
        dz = 1.0 / (n - 1)
        D1, _ = build_sbp_matrix(n, dz, 2)

        h = np.ones(n) * dz
        h[0] = dz * 0.5
        h[-1] = dz * 0.5
        H = sparse.diags(h, format="csr")

        Q = H @ D1
        S = (Q + Q.T).toarray()

        B = np.zeros(n)
        B[0] = -1.0
        B[-1] = 1.0

        off_diag = S - np.diag(np.diag(S))
        assert np.max(np.abs(off_diag)) < 1e-13
        np.testing.assert_allclose(np.diag(S), B, atol=1e-13)


# ===================================================================
# TestDerivativeAccuracy
# ===================================================================
class TestDerivativeAccuracy:
    """Verify polynomial exactness and convergence rates of 1-D SBP operators."""

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d1_polynomial_exactness(self, order):
        n = max(80, 2 * min_points(order))
        dz = 1.0 / (n - 1)
        D1, _ = build_sbp_matrix(n, dz, order)
        x = np.linspace(0, 1, n)
        cw = min_points(order)

        for p in range(order + 1):
            f = x**p
            df_exact = p * x ** max(p - 1, 0) if p > 0 else np.zeros(n)
            df_num = D1 @ f
            interior = slice(cw, n - cw)
            np.testing.assert_allclose(
                df_num[interior], df_exact[interior], atol=1e-10,
                err_msg=f"D1 not exact on x^{p}, order={order}",
            )

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d2_polynomial_exactness(self, order):
        n = max(80, 2 * min_points(order))
        dz = 1.0 / (n - 1)
        _, D2 = build_sbp_matrix(n, dz, order)
        x = np.linspace(0, 1, n)
        cw = min_points(order)

        for p in range(order + 1):
            f = x**p
            if p >= 2:
                d2f_exact = p * (p - 1) * x ** (p - 2)
            else:
                d2f_exact = np.zeros(n)
            d2f_num = D2 @ f
            interior = slice(cw, n - cw)
            np.testing.assert_allclose(
                d2f_num[interior], d2f_exact[interior], atol=1e-10,
                err_msg=f"D2 not exact on x^{p}, order={order}",
            )

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d1_interior_convergence_rate(self, order):
        """Interior-only D1 convergence should match the interior order."""
        # Use grids large enough to have meaningful interior,
        # but small enough to avoid noise floor.
        ns = [50, 60, 80, 100]
        ns = [max(n, 4 * min_points(order)) for n in ns]
        ns = sorted(set(ns))
        if len(ns) < 3:
            pytest.skip("Not enough distinct grid sizes after clamping")

        cw = min_points(order)
        errors = []
        for n in ns:
            dz = 1.0 / (n - 1)
            D1, _ = build_sbp_matrix(n, dz, order)
            x = np.linspace(0, 1, n)
            # Use higher frequency for high orders to stay above noise floor
            freq = max(2 * np.pi, order * np.pi)
            f = np.sin(freq * x)
            df_exact = freq * np.cos(freq * x)
            df_num = D1 @ f
            err = np.abs(df_num - df_exact)
            errors.append(np.max(err[cw : n - cw]))

        log_h = np.log(1.0 / (np.array(ns, dtype=float) - 1))
        log_e = np.log(np.array(errors))
        slope = np.polyfit(log_h, log_e, 1)[0]
        assert slope >= order - 0.5, (
            f"D1 interior convergence slope {slope:.2f} < {order - 0.5}"
        )

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d2_interior_convergence_rate(self, order):
        """Interior-only D2 convergence should match the interior order."""
        ns = [50, 60, 80, 100]
        ns = [max(n, 4 * min_points(order)) for n in ns]
        ns = sorted(set(ns))
        if len(ns) < 3:
            pytest.skip("Not enough distinct grid sizes after clamping")

        cw = min_points(order)
        errors = []
        for n in ns:
            dz = 1.0 / (n - 1)
            _, D2 = build_sbp_matrix(n, dz, order)
            x = np.linspace(0, 1, n)
            freq = max(2 * np.pi, order * np.pi)
            f = np.sin(freq * x)
            d2f_exact = -(freq ** 2) * np.sin(freq * x)
            d2f_num = D2 @ f
            err = np.abs(d2f_num - d2f_exact)
            errors.append(np.max(err[cw : n - cw]))

        log_h = np.log(1.0 / (np.array(ns, dtype=float) - 1))
        log_e = np.log(np.array(errors))
        slope = np.polyfit(log_h, log_e, 1)[0]
        assert slope >= order - 0.5, (
            f"D2 interior convergence slope {slope:.2f} < {order - 0.5}"
        )


# ===================================================================
# TestIntegrationAccuracy
# ===================================================================
class TestIntegrationAccuracy:
    """Verify SBP quadrature weights integrate correctly."""

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_1d_trapezoidal_exact_for_linear(self, order):
        """Trapezoidal rule integrates constants and linears exactly."""
        n = max(40, 2 * min_points(order))
        dz = 1.0 / (n - 1)
        x = np.linspace(0, 1, n)

        h = np.ones(n) * dz
        h[0] = dz * 0.5
        h[-1] = dz * 0.5

        np.testing.assert_allclose(h @ np.ones(n), 1.0, atol=1e-12)
        np.testing.assert_allclose(h @ x, 0.5, atol=1e-12)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_1d_higher_order_convergence(self, order):
        """Trapezoidal rule converges at O(h^2) for x^2."""
        ns = [20, 40, 80, 160]
        errors = []
        for n in ns:
            dz = 1.0 / (n - 1)
            x = np.linspace(0, 1, n)
            h = np.ones(n) * dz
            h[0] = dz * 0.5
            h[-1] = dz * 0.5
            errors.append(abs(h @ x**2 - 1.0 / 3.0))

        log_h = np.log(1.0 / (np.array(ns, dtype=float) - 1))
        log_e = np.log(np.array(errors))
        slope = np.polyfit(log_h, log_e, 1)[0]
        assert slope >= 1.5, f"1D integration slope {slope:.2f} < 1.5"

    def test_2d_integration_constant_square(self):
        """W integrates 1 over [0,1]^2 on a square mesh; exact area = 1.0."""
        mesh = _square_mesh(n=30, fd_order=4, L=1.0)
        area = np.sum(mesh.W)
        np.testing.assert_allclose(area, 1.0, atol=1e-12)

    def test_2d_integration_constant_rectangular_domain(self):
        """W integrates 1 over [0,2]x[0,1] on a square mesh (nx=ny=30)."""
        mesh = Mesh.create(
            xrange=(0.0, 2.0), yrange=(0.0, 1.0), nx=30, ny=30, fd_order=4,
        )
        area = np.sum(mesh.W)
        np.testing.assert_allclose(area, 2.0, atol=1e-12)

    def test_2d_integration_xy_square(self):
        """On a square mesh, W integrates x*y over [0,1]^2; exact = 0.25."""
        mesh = _square_mesh(n=30, fd_order=4, L=1.0)
        f = mesh.X * mesh.Y
        integral = np.sum(mesh.W * f)
        np.testing.assert_allclose(integral, 0.25, atol=1e-6)

    def test_2d_integration_flat_vector(self):
        """W flattened (Fortran-order) dot with 1 = area on square mesh."""
        mesh = _square_mesh(n=30, fd_order=4, L=1.0)
        W_flat = mesh.W.ravel(order="F")
        f_flat = np.ones(mesh.X.size)
        integral = W_flat[mesh.used_ind] @ f_flat[mesh.used_ind]
        np.testing.assert_allclose(integral, 1.0, atol=1e-12)


# ===================================================================
# TestMeshCreation
# ===================================================================
class TestMeshCreation:
    """Verify basic mesh creation and attribute shapes."""

    def test_weights_positive(self, small_mesh_2d):
        mesh = small_mesh_2d
        W_flat = mesh.W.ravel(order="F")
        assert np.all(W_flat[mesh.used_ind] > 0)

    def test_boundary_indices_correct_coordinates(self, small_mesh_2d):
        """Boundary indices map to correct physical coordinates (F-order)."""
        mesh = small_mesh_2d  # nx=20, ny=30 (rectangular!)
        p = mesh.used_ind
        x = mesh.X.ravel(order="F")[p]
        y = mesh.Y.ravel(order="F")[p]

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # Bottom: all y = y_min
        bi = mesh.idx["bi"]
        assert len(bi) == 20  # nx points
        np.testing.assert_allclose(y[bi], y_min, atol=1e-14)

        # Top: all y = y_max
        ti = mesh.idx["ti"]
        assert len(ti) == 20
        np.testing.assert_allclose(y[ti], y_max, atol=1e-14)

        # Left: all x = x_min
        li = mesh.idx["li"]
        assert len(li) == 30  # ny points
        np.testing.assert_allclose(x[li], x_min, atol=1e-14)

        # Right: all x = x_max
        ri = mesh.idx["ri"]
        assert len(ri) == 30
        np.testing.assert_allclose(x[ri], x_max, atol=1e-14)

        # No overlap between opposite boundaries
        assert len(np.intersect1d(bi, ti)) == 0
        assert len(np.intersect1d(li, ri)) == 0

    def test_periodic_boundary_indices_empty(self, small_mesh_periodic):
        mesh = small_mesh_periodic
        assert len(mesh.idx["li"]) == 0
        assert len(mesh.idx["ri"]) == 0

    def test_constant_preservation_dx(self, small_mesh_2d):
        """Dx @ 1 = 0."""
        mesh = small_mesh_2d
        result = mesh.Dx @ np.ones(mesh.ngp)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_constant_preservation_dy(self, small_mesh_2d):
        """Dy @ 1 = 0."""
        mesh = small_mesh_2d
        result = mesh.Dy @ np.ones(mesh.ngp)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)



# ===================================================================
# Test2DOperatorAccuracy (square mesh, nx == ny)
# ===================================================================
class Test2DOperatorAccuracy:
    """Verify 2-D derivative accuracy on square meshes (nx == ny).

    Convention (matching MATLAB AddDW2mesh):
        Dx = d/dx   (differentiates along columns / streamwise)
        Dy = d/dy   (differentiates along rows / wall-normal)
    """

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_dx_differentiates_x(self, order):
        """Dx @ x = 1 at deep-interior points on [0,1]^2."""
        n = max(30, 2 * min_points(order))
        mesh = _square_mesh(n=n, fd_order=order)
        X_flat = mesh.X.ravel(order="F")
        result = mesh.Dx @ X_flat
        cw = min_points(order)
        interior = _deep_interior(n, cw)
        np.testing.assert_allclose(result[interior], 1.0, atol=1e-10)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_dy_differentiates_y(self, order):
        """Dy @ y = 1 at deep-interior points on [0,1]^2."""
        n = max(30, 2 * min_points(order))
        mesh = _square_mesh(n=n, fd_order=order)
        Y_flat = mesh.Y.ravel(order="F")
        result = mesh.Dy @ Y_flat
        cw = min_points(order)
        interior = _deep_interior(n, cw)
        np.testing.assert_allclose(result[interior], 1.0, atol=1e-10)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_dx_zeros_on_y(self, order):
        """Dx @ y = 0 (Dx does not differentiate along y-direction)."""
        n = max(30, 2 * min_points(order))
        mesh = _square_mesh(n=n, fd_order=order)
        Y_flat = mesh.Y.ravel(order="F")
        result = mesh.Dx @ Y_flat
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_dy_zeros_on_x(self, order):
        """Dy @ x = 0 (Dy does not differentiate along x-direction)."""
        n = max(30, 2 * min_points(order))
        mesh = _square_mesh(n=n, fd_order=order)
        X_flat = mesh.X.ravel(order="F")
        result = mesh.Dy @ X_flat
        np.testing.assert_allclose(result, 0.0, atol=1e-12)


# ===================================================================
# TestSchwarzTheorem
# ===================================================================
class TestSchwarzTheorem:
    """Verify that mixed partials commute (Dxy ~ Dyx) on a square mesh."""

    def test_smooth_function(self):
        """Dxy ~ Dyx on sin(pi*x)*cos(pi*y)."""
        mesh = _square_mesh(n=30, fd_order=4)
        X_flat = mesh.X.ravel(order="F")
        Y_flat = mesh.Y.ravel(order="F")
        f = np.sin(np.pi * X_flat) * np.cos(np.pi * Y_flat)
        dxy = mesh.Dxy @ f
        dyx = mesh.Dyx @ f
        np.testing.assert_allclose(dxy, dyx, atol=1e-3)

    def test_polynomial_exact_order4(self):
        """Dxy = Dyx exactly for f = x^2 * y^2 when order >= 4."""
        mesh = _square_mesh(n=30, fd_order=4)
        X_flat = mesh.X.ravel(order="F")
        Y_flat = mesh.Y.ravel(order="F")
        f = X_flat**2 * Y_flat**2
        dxy = mesh.Dxy @ f
        dyx = mesh.Dyx @ f
        np.testing.assert_allclose(dxy, dyx, atol=1e-10)


# ===================================================================
# TestMeshDeformation
# ===================================================================
class TestMeshDeformation:
    """Verify mesh deformation structural properties.

    Tests focus on structural properties (shapes, sparsity) and derivative
    accuracy after deformation.
    """

    def test_deformation_updates_coordinates(self, small_mesh_2d):
        """Deformed mesh stores the new coordinates."""
        mesh = small_mesh_2d
        s = 2.0
        y_max = mesh.Y.max()
        Y_new = y_max * np.sinh(s * mesh.Y / y_max) / np.sinh(s)
        mesh_def = mesh.deform(Y_new=Y_new)
        np.testing.assert_allclose(mesh_def.Y, Y_new)
        np.testing.assert_allclose(mesh_def.X, mesh.X)

    def test_shape_mismatch_raises(self, small_mesh_2d):
        """Deform with wrong shape raises ValueError."""
        mesh = small_mesh_2d
        wrong_shape = np.ones((5, 5))
        with pytest.raises(ValueError):
            mesh.deform(Y_new=wrong_shape)



# ===================================================================
# TestMeshMasking
# ===================================================================
class TestMeshMasking:
    """Verify mesh masking reduces DOFs and adjusts operators.

    The ``Mesh.mask`` method expects a 2-D boolean array shaped like
    ``X`` and ``Y`` (ny, nx).
    """

    def test_masking_reduces_ngp(self, small_mesh_2d):
        mesh = small_mesh_2d
        mask = np.ones(mesh.X.shape, dtype=bool)
        mask[0, :5] = False
        masked = mesh.mask(mask)
        assert masked.ngp == mesh.ngp - 5

    def test_operators_shrink(self, small_mesh_2d):
        mesh = small_mesh_2d
        mask = np.ones(mesh.X.shape, dtype=bool)
        mask[0, :5] = False
        masked = mesh.mask(mask)
        new_ngp = masked.ngp
        for name in ("Dx", "D2x", "Dy", "D2y"):
            op = getattr(masked, name)
            assert op.shape == (new_ngp, new_ngp), f"{name} not resized"

    def test_boundary_indices_valid(self, small_mesh_2d):
        mesh = small_mesh_2d
        mask = np.ones(mesh.X.shape, dtype=bool)
        mask[0, :5] = False
        masked = mesh.mask(mask)
        new_ngp = masked.ngp
        for key in ("bi", "ti", "li", "ri"):
            if key in masked.idx and len(masked.idx[key]) > 0:
                assert np.all(masked.idx[key] >= 0)
                assert np.all(masked.idx[key] < new_ngp), (
                    f"idx['{key}'] out of range"
                )

    def test_mask_all_raises(self, small_mesh_2d):
        mesh = small_mesh_2d
        mask = np.zeros(mesh.X.shape, dtype=bool)
        with pytest.raises(ValueError):
            mesh.mask(mask)

    def test_mask_shape_mismatch(self, small_mesh_2d):
        """Mask with wrong shape raises ValueError."""
        mesh = small_mesh_2d
        wrong_mask = np.ones((5, 5), dtype=bool)
        with pytest.raises(ValueError):
            mesh.mask(wrong_mask)

    def test_used_ind_reduces(self, small_mesh_2d):
        """used_ind shrinks by the number of masked-out points."""
        mesh = small_mesh_2d
        mask = np.ones(mesh.X.shape, dtype=bool)
        mask[0, :5] = False
        masked = mesh.mask(mask)
        assert masked.used_ind.size == masked.ngp
        assert masked.used_ind.size < mesh.used_ind.size


# ===================================================================
# TestPoleStencils (order 2 only)
# ===================================================================
class TestPoleStencils:
    """Verify pole-compatible SBP operators for cylindrical coordinates.

    Only order 2 is supported (higher orders raise a broadcasting error
    in the Mohseni pole-stencil code).
    """

    def test_pole_matrices_have_correct_shape(self):
        n = 40
        dz = 1.0 / (n - 1)
        D1_sym, D1_asymm, D2_sym, D2_asymm = build_sbp_pole_matrices(n, dz, 2)
        for mat in (D1_sym, D1_asymm, D2_sym, D2_asymm):
            assert mat.shape == (n, n)

    def test_pole_d1_sym_row0_modified(self):
        """D1_sym row 0 should match the Mohseni symmetric pole closure.

        For the order-2 interior stencil ``[-1/2, 0, 1/2]``, the symmetric
        pole folding turns the first stored row into ``[-1/2, 1/2] / dz``.
        This is the closure used by the MATLAB reference operators.
        """
        n = 40
        dz = 1.0 / (n - 1)
        D1_sym, _, _, _ = build_sbp_pole_matrices(n, dz, 2)
        row0 = D1_sym[0].toarray().ravel()
        np.testing.assert_allclose(row0[:2], np.array([-0.5, 0.5]) / dz, atol=1e-13)
        np.testing.assert_allclose(row0[2:], 0.0, atol=1e-13)

    def test_pole_d1_sym_symmetric_function_convergence(self):
        """D1_sym on f(r) = r^2 converges to 0 at r=0 with refinement.

        The derivative of r^2 at r=0 should be 0 for a symmetric function.
        Due to order-2 truncation, the value is O(h) at the pole.
        """
        errors = []
        for n in [40, 80, 160]:
            dz = 1.0 / (n - 1)
            D1_sym, _, _, _ = build_sbp_pole_matrices(n, dz, 2)
            r = np.linspace(0, 1, n)
            f = r**2
            df = D1_sym @ f
            errors.append(abs(df[0]))
        # Should converge: each refinement halves the error (order 1)
        assert errors[-1] < errors[0], "D1_sym @ r^2 at pole should converge"
        ratio = errors[0] / errors[-1]
        assert ratio > 3, f"Convergence ratio too low: {ratio:.1f}"

    def test_antisymmetric_function_nonzero_at_pole(self):
        """D1_asymm on f(r) = r gives nonzero df/dr at r=0."""
        n = 40
        dz = 1.0 / (n - 1)
        _, D1_asymm, _, _ = build_sbp_pole_matrices(n, dz, 2)
        r = np.linspace(0, 1, n)
        f = r
        df = D1_asymm @ f
        assert abs(df[0]) > 0.1, "D1_asymm @ r should be nonzero at r=0"

    def test_sym_asymm_differ_in_closure(self):
        """D1_sym and D1_asymm differ within the pole closure region."""
        n = 40
        dz = 1.0 / (n - 1)
        D1_sym, D1_asymm, _, _ = build_sbp_pole_matrices(n, dz, 2)
        w = (len(sbp_stencils(2)["a_i1"]) - 1) // 2
        differs = False
        for row in range(w):
            row_sym = D1_sym[row].toarray().ravel()
            row_asym = D1_asymm[row].toarray().ravel()
            if not np.allclose(row_sym, row_asym, atol=1e-15):
                differs = True
                break
        assert differs, "D1_sym and D1_asymm identical in all closure rows"

    def test_pole_d1_sym_interior_matches_standard(self):
        """D1_sym matches standard D1 at interior points (away from pole)."""
        n = 40
        dz = 1.0 / (n - 1)
        D1, _ = build_sbp_matrix(n, dz, 2)
        D1_sym, _, _, _ = build_sbp_pole_matrices(n, dz, 2)
        # Interior stencil width for order 2: w = 1, closure rows: r = 1
        # So rows 1 through n-2 should be identical
        D1_dense = D1.toarray()
        D1s_dense = D1_sym.toarray()
        np.testing.assert_allclose(
            D1s_dense[1:], D1_dense[1:], atol=1e-13,
            err_msg="D1_sym interior rows differ from standard D1",
        )


# ===================================================================
# TestPeriodicOperators
# ===================================================================
class TestPeriodicOperators:
    """Verify periodic 1-D derivative operators.

    Tests the ``_build_periodic_1d_operators`` building block directly,
    since the 2-D periodic assembly shares the nx/ny indexing constraint.
    """

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_periodic_1d_rows_sum_to_zero(self, order):
        """Each row of the periodic D1 sums to zero."""
        n = max(64, 4 * min_points(order))
        dx = 2.0 * np.pi / n
        D1, _ = _build_periodic_1d_operators(n, dx, order)
        row_sums = np.array(D1.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-13)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_periodic_1d_d1_sin(self, order):
        """D1 @ sin(x) ~ cos(x) on [0, 2*pi) with periodic wrapping."""
        n = max(64, 4 * min_points(order))
        dx = 2.0 * np.pi / n
        D1, _ = _build_periodic_1d_operators(n, dx, order)
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        f = np.sin(x)
        df_exact = np.cos(x)
        df_num = D1 @ f
        np.testing.assert_allclose(df_num, df_exact, atol=5e-3)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_periodic_1d_d2_sin(self, order):
        """D2 @ sin(x) ~ -sin(x) on [0, 2*pi) with periodic wrapping."""
        n = max(64, 4 * min_points(order))
        dx = 2.0 * np.pi / n
        _, D2 = _build_periodic_1d_operators(n, dx, order)
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        f = np.sin(x)
        d2f_exact = -np.sin(x)
        d2f_num = D2 @ f
        np.testing.assert_allclose(d2f_num, d2f_exact, atol=5e-3)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_periodic_1d_circulant(self, order):
        """Periodic D1 should be circulant (shift-invariant)."""
        n = max(64, 4 * min_points(order))
        dx = 2.0 * np.pi / n
        D1, _ = _build_periodic_1d_operators(n, dx, order)
        D1_dense = D1.toarray()
        row0 = D1_dense[0]
        for i in range(1, n):
            np.testing.assert_allclose(
                D1_dense[i], np.roll(row0, i), atol=1e-13,
                err_msg=f"Row {i} not a circular shift of row 0",
            )

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_periodic_1d_constant_preservation(self, order):
        """Periodic D1 @ 1 = 0."""
        n = max(64, 4 * min_points(order))
        dx = 2.0 * np.pi / n
        D1, D2 = _build_periodic_1d_operators(n, dx, order)
        ones = np.ones(n)
        np.testing.assert_allclose(D1 @ ones, 0.0, atol=1e-13)
        np.testing.assert_allclose(D2 @ ones, 0.0, atol=1e-12)

    def test_periodic_mesh_structural(self, small_mesh_periodic):
        """Periodic mesh has correct structure: constant preservation,
        li/ri empty, correct shape."""
        mesh = small_mesh_periodic
        ny, nx = mesh.X.shape
        assert nx == 32
        assert ny == 20
        assert mesh.ngp == nx * ny
        assert len(mesh.idx["li"]) == 0
        assert len(mesh.idx["ri"]) == 0
        assert mesh.x_periodicity is True

        ones = np.ones(mesh.ngp)
        np.testing.assert_allclose(mesh.Dx @ ones, 0.0, atol=1e-12)
        np.testing.assert_allclose(mesh.Dy @ ones, 0.0, atol=1e-12)


# ===================================================================
# TestMaskThenDeform
# ===================================================================
class TestMaskThenDeform:
    """Verify that mask boundary indices survive deformation.

    Regression test: ``deform()`` formerly passed an all-True mask to
    ``_build_indices``, silently dropping internal mask boundary points
    (cavity walls) from ``idx["mi"]``.
    """

    @staticmethod
    def _cavity_mesh():
        """Create a 20x20 mesh with a 4x4 hole (cavity mask)."""
        mesh = Mesh.create(
            xrange=(0.0, 1.0), yrange=(0.0, 1.0), nx=20, ny=20, fd_order=4,
        )
        keep = np.ones(mesh.X.shape, dtype=bool)
        keep[8:12, 8:12] = False
        return mesh.mask(keep)

    def test_mi_present_after_mask(self):
        """After mask(), mi should contain internal boundary points."""
        masked = self._cavity_mesh()
        assert len(masked.idx["mi"]) > 0, "mi empty after mask()"

    def test_mi_survives_deform_y(self):
        """After mask() + deform(Y), mi should still be non-empty."""
        masked = self._cavity_mesh()
        mi_before = masked.idx["mi"].copy()
        Y_new = masked.Y * 1.1  # mild uniform stretch
        deformed = masked.deform(Y_new=Y_new)
        assert len(deformed.idx["mi"]) > 0, "mi lost after deform(Y)"
        assert len(deformed.idx["mi"]) == len(mi_before), (
            f"mi count changed: {len(mi_before)} -> {len(deformed.idx['mi'])}"
        )

    def test_mi_survives_deform_x(self):
        """After mask() + deform(X), mi should still be non-empty."""
        masked = self._cavity_mesh()
        mi_before = masked.idx["mi"].copy()
        X_new = masked.X * 1.1
        deformed = masked.deform(X_new=X_new)
        assert len(deformed.idx["mi"]) > 0, "mi lost after deform(X)"
        assert len(deformed.idx["mi"]) == len(mi_before)

    def test_mi_survives_deform_chain(self):
        """After mask() + deform(Y) + deform(X), mi should persist."""
        masked = self._cavity_mesh()
        mi_before = masked.idx["mi"].copy()
        Y_new = masked.Y * 1.1
        X_new = masked.X * 1.05
        deformed = masked.deform(Y_new=Y_new).deform(X_new=X_new)
        assert len(deformed.idx["mi"]) > 0, "mi lost after chained deform"
        assert len(deformed.idx["mi"]) == len(mi_before)

    def test_mi_points_at_cavity_boundary(self):
        """mi points should be adjacent to the masked-out cavity hole."""
        masked = self._cavity_mesh()
        p = masked.used_ind
        x = masked.X.ravel(order="F")[p]
        y = masked.Y.ravel(order="F")[p]
        mi = masked.idx["mi"]
        # Cavity hole is at rows/cols 8..11 on a [0,1] grid with 20 pts
        # -> x in [x[8], x[11]], y in [y[8], y[11]]
        x_mi = x[mi]
        y_mi = y[mi]
        # Some mi points must be near the cavity region
        cavity_adjacent = (
            (x_mi >= 0.3) & (x_mi <= 0.65)
            & (y_mi >= 0.3) & (y_mi <= 0.65)
        )
        assert np.any(cavity_adjacent), "no mi points near cavity boundary"


def test_create_mesh_y_symmetry_matches_matlab_half_cell_shift():
    mesh = Mesh.create(
        xrange=(-1.0, 0.0),
        yrange=(0.0, 1.0),
        nx=100,
        ny=100,
        fd_order=4,
        y_symmetry=True,
        x_periodicity=False,
        alpha_filter=0.0,
    )

    dy = 1.0 / (100 - 0.5)
    expected_y0 = dy / 2.0
    expected_y1 = expected_y0 + dy

    assert mesh.Y[0, 0] == pytest.approx(expected_y0)
    assert mesh.Y[1, 0] == pytest.approx(expected_y1)


def test_create_mesh_y_symmetry_uses_pole_stencils_after_half_cell_shift():
    mesh = Mesh.create(
        xrange=(0.0, 1.0),
        yrange=(0.0, 1.0),
        nx=8,
        ny=20,
        fd_order=4,
        y_symmetry=True,
        x_periodicity=False,
        alpha_filter=0.0,
    )

    dy = mesh.Y[1, 0] - mesh.Y[0, 0]
    D1_sym, D1_asym, D2_sym, D2_asym = build_sbp_pole_matrices(mesh.Y.shape[0], dy, 4)
    ny = mesh.Y.shape[0]

    np.testing.assert_allclose(mesh.Dy_symm[:ny, :ny].toarray(), D1_sym.toarray())
    np.testing.assert_allclose(mesh.Dy_asymm[:ny, :ny].toarray(), D1_asym.toarray())
    np.testing.assert_allclose(mesh.D2y_symm[:ny, :ny].toarray(), D2_sym.toarray())
    np.testing.assert_allclose(mesh.D2y_asymm[:ny, :ny].toarray(), D2_asym.toarray())


def test_deform_separable_xy_reuses_asymm_y_only_family():
    base = Mesh.create(
        xrange=(0.0, 1.0),
        yrange=(0.0, 1.0),
        nx=12,
        ny=20,
        fd_order=4,
        y_symmetry=True,
        x_periodicity=False,
        alpha_filter=0.0,
    )

    X_new = np.broadcast_to(base.X[0:1, :] ** 1.2, base.X.shape)
    Y_new = np.broadcast_to(base.Y[:, 0:1] ** 1.1, base.Y.shape)

    full = base.deform(X_new=X_new, Y_new=Y_new)
    seq = base.deform(X_new=X_new).deform(Y_new=Y_new)

    np.testing.assert_allclose(full.Dy_asymm.toarray(), seq.Dy_asymm.toarray())
    np.testing.assert_allclose(full.D2y_asymm.toarray(), seq.D2y_asymm.toarray())
    np.testing.assert_allclose(full.Dxy_asymm.toarray(), seq.Dxy_asymm.toarray())
    np.testing.assert_allclose(full.Dyx_asymm.toarray(), seq.Dyx_asymm.toarray())
