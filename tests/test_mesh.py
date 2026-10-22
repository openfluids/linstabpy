# SPDX-License-Identifier: GPL-3.0-only
"""Check the linstabpy.mesh module.

Covers SBP stencils, 1-D derivative operators, integration, mesh creation,
deformation, masking, pole stencils, and periodic operators.

Notes on the 2-D operator layout.
The 2-D operator assembly in ``_build_operators`` iterates over unique
coordinate values.
It builds 1-D SBP operators along each grid line
(matching MATLAB ``AddDW2mesh.m``).
The assembler flattens data in column-major (Fortran) order to match MATLAB
indexing.
The operators satisfy the standard convention.

    - ``Dx @ x_flat = 1``.  (Dx differentiates along x)
    - ``Dx @ y_flat = 0``.  (Dx does not differentiate along y)
    - ``Dy @ y_flat = 1``.  (Dy differentiates along y)
    - ``Dy @ x_flat = 0``.  (Dy does not differentiate along x)

This works for both square (``nx == ny``) and rectangular (``nx != ny``)
grids.

Pole-condition operators (``build_sbp_pole_matrices``) only work for
**order 2**; higher orders raise a broadcasting error in the Mohseni
pole-stencil code.
These checks cover order 2 only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from linstabpy.mesh import (
    Mesh,
    build_sbp_matrix,
    build_sbp_pole_matrices,
    min_points,
    sbp_stencils,
)
from linstabpy.mesh.fdweights import fdweights
from linstabpy.mesh.mesh import (
    _axm_diag,
    _build_periodic_1d_operators,
    _line_breaks,
    _validate_create_inputs,
    _validate_range_pair,
)

# ===================================================================
# Square-mesh factory (nx == ny)
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
    def test_min_points_keeps_the_closures_apart(self, order):
        r, m = sbp_stencils(order)["a_b1"].shape
        assert min_points(order) == max(m, 2 * r)

    @pytest.mark.parametrize("order", [4, 6, 8])
    def test_sbp_identity_holds_from_min_points(self, order):
        """Q + Q^T = B, with Q = H D1, at the smallest allowed n; below it the builder refuses."""
        from linstabpy.mesh.sbp import _build_sbp_norm_diag

        n = min_points(order)
        D1, _ = build_sbp_matrix(n, 1.0, order)
        Q = np.diag(_build_sbp_norm_diag(n, 1.0, order)) @ D1.toarray()
        B = np.zeros((n, n))
        B[0, 0], B[-1, -1] = -1.0, 1.0
        np.testing.assert_allclose(Q + Q.T, B, atol=1e-12)
        with pytest.raises(ValueError, match="min_points"):
            build_sbp_matrix(n - 1, 1.0, order)


# ===================================================================
# TestSBPProperty
# ===================================================================
class TestSBPProperty:
    """Check constant preservation, the row-sum condition, and the SBP identity.

    For order 2 the identity is Q + Q^T = B.
    """

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
        cw = sbp_stencils(order)["a_b1"].shape[1]  # closure width

        for p in range(order + 1):
            f = x**p
            df_exact = p * x ** max(p - 1, 0) if p > 0 else np.zeros(n)
            df_num = D1 @ f
            interior = slice(cw, n - cw)
            np.testing.assert_allclose(
                df_num[interior],
                df_exact[interior],
                atol=1e-10,
                err_msg=f"D1 not exact on x^{p}, order={order}",
            )

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d2_polynomial_exactness(self, order):
        n = max(80, 2 * min_points(order))
        dz = 1.0 / (n - 1)
        _, D2 = build_sbp_matrix(n, dz, order)
        x = np.linspace(0, 1, n)
        cw = sbp_stencils(order)["a_b1"].shape[1]  # closure width

        for p in range(order + 1):
            f = x**p
            if p >= 2:
                d2f_exact = p * (p - 1) * x ** (p - 2)
            else:
                d2f_exact = np.zeros(n)
            d2f_num = D2 @ f
            interior = slice(cw, n - cw)
            np.testing.assert_allclose(
                d2f_num[interior],
                d2f_exact[interior],
                atol=1e-10,
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

        cw = sbp_stencils(order)["a_b1"].shape[1]  # closure width
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
        assert slope >= order - 0.5, f"D1 interior convergence slope {slope:.2f} < {order - 0.5}"

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_d2_interior_convergence_rate(self, order):
        """Interior-only D2 convergence should match the interior order."""
        ns = [50, 60, 80, 100]
        ns = [max(n, 4 * min_points(order)) for n in ns]
        ns = sorted(set(ns))
        if len(ns) < 3:
            pytest.skip("Not enough distinct grid sizes after clamping")

        cw = sbp_stencils(order)["a_b1"].shape[1]  # closure width
        errors = []
        for n in ns:
            dz = 1.0 / (n - 1)
            _, D2 = build_sbp_matrix(n, dz, order)
            x = np.linspace(0, 1, n)
            freq = max(2 * np.pi, order * np.pi)
            f = np.sin(freq * x)
            d2f_exact = -(freq**2) * np.sin(freq * x)
            d2f_num = D2 @ f
            err = np.abs(d2f_num - d2f_exact)
            errors.append(np.max(err[cw : n - cw]))

        log_h = np.log(1.0 / (np.array(ns, dtype=float) - 1))
        log_e = np.log(np.array(errors))
        slope = np.polyfit(log_h, log_e, 1)[0]
        assert slope >= order - 0.5, f"D2 interior convergence slope {slope:.2f} < {order - 0.5}"


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
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            nx=30,
            ny=30,
            fd_order=4,
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
        cw = sbp_stencils(order)["a_b1"].shape[1]  # closure width
        interior = _deep_interior(n, cw)
        np.testing.assert_allclose(result[interior], 1.0, atol=1e-10)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_dy_differentiates_y(self, order):
        """Dy @ y = 1 at deep-interior points on [0,1]^2."""
        n = max(30, 2 * min_points(order))
        mesh = _square_mesh(n=n, fd_order=order)
        Y_flat = mesh.Y.ravel(order="F")
        result = mesh.Dy @ Y_flat
        cw = sbp_stencils(order)["a_b1"].shape[1]  # closure width
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
# TestMeshStateViews
# ===================================================================
class TestMeshStateViews:
    """Verify the Mesh state views expose existing storage without migration."""

    def test_geometry_view_references_existing_coordinate_state(self, small_mesh_2d):
        mesh = small_mesh_2d
        geometry = mesh.geometry
        assert geometry.X is mesh.X
        assert geometry.Y is mesh.Y
        assert geometry.used_ind is mesh.used_ind
        assert geometry.idx is mesh.idx
        assert geometry.ngp == mesh.ngp
        assert geometry.y_symmetry == mesh.y_symmetry
        assert geometry.x_periodicity == mesh.x_periodicity

    def test_operator_view_references_existing_operator_state(self, small_mesh_2d):
        mesh = small_mesh_2d
        operators = mesh.operators
        assert operators.Dx is mesh.Dx
        assert operators.D2x is mesh.D2x
        assert operators.Dy is mesh.Dy
        assert operators.D2y is mesh.D2y
        assert operators.Dxy is mesh.Dxy
        assert operators.Dyx is mesh.Dyx
        assert operators.W is mesh.W
        assert operators.fd_order == mesh.fd_order

    def test_runtime_view_references_existing_optional_state(self):
        mesh = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=16,
            ny=18,
            fd_order=4,
            alpha_filter=0.4,
            numerics_operator_dissipation={"enabled": True},
        )
        runtime = mesh.runtime_state
        operators = mesh.operators
        assert runtime.filters is mesh.filters
        assert runtime.alpha_filter == mesh.alpha_filter
        assert runtime.sponge is mesh.sponge
        assert runtime.J11 is mesh.J11
        assert runtime.J22 is mesh.J22
        assert operators.Q_diss_x is mesh.Q_diss_x
        assert operators.Q_diss_y is mesh.Q_diss_y

    def test_deform_updates_geometry_and_operators_but_preserves_filters(self):
        mesh = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=16,
            ny=18,
            fd_order=4,
            alpha_filter=0.4,
        )
        Y_new = mesh.Y * 1.1
        deformed = mesh.deform(Y_new=Y_new)
        assert deformed.geometry.X is deformed.X
        assert deformed.geometry.Y is deformed.Y
        assert deformed.operators.Dy is deformed.Dy
        assert deformed.operators.Dy is not mesh.Dy
        assert deformed.runtime_state.filters is mesh.runtime_state.filters
        np.testing.assert_allclose(deformed.geometry.Y, Y_new)

    def test_mask_updates_geometry_and_rebuilds_runtime_filters(self):
        mesh = Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=16,
            ny=18,
            fd_order=4,
            alpha_filter=0.4,
        )
        mask = np.ones(mesh.X.shape, dtype=bool)
        mask[0, :4] = False
        masked = mesh.mask(mask)
        assert masked.geometry.used_ind is masked.used_ind
        assert masked.geometry.used_ind.size == masked.ngp
        assert masked.geometry.used_ind.size < mesh.geometry.used_ind.size
        assert masked.runtime_state.filters is masked.filters
        assert masked.runtime_state.filters is not mesh.runtime_state.filters


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
        """Deform with a mismatched array shape raises ValueError."""
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
                assert np.all(masked.idx[key] < new_ngp), f"idx['{key}'] out of range"

    def test_mask_all_raises(self, small_mesh_2d):
        mesh = small_mesh_2d
        mask = np.zeros(mesh.X.shape, dtype=bool)
        with pytest.raises(ValueError):
            mesh.mask(mask)

    def test_mask_shape_mismatch(self, small_mesh_2d):
        """Mask with a mismatched array shape raises ValueError."""
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

    Only order 2 works.
    Higher orders raise a broadcasting error in the Mohseni pole-stencil code.
    """

    def test_pole_matrices_have_correct_shape(self):
        n = 40
        dz = 1.0 / (n - 1)
        D1_sym, D1_asymm, D2_sym, D2_asymm = build_sbp_pole_matrices(n, dz, 2)
        for mat in (D1_sym, D1_asymm, D2_sym, D2_asymm):
            assert mat.shape == (n, n)

    def test_pole_d1_sym_row0_modified(self):
        """D1_sym row 0 should match the Mohseni symmetric pole closure.

        The order-2 interior stencil is ``[-1/2, 0, 1/2]``.
        Symmetric pole folding turns the first stored row into ``[-1/2, 1/2] / dz``.
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
            D1s_dense[1:],
            D1_dense[1:],
            atol=1e-13,
            err_msg="D1_sym interior rows differ from standard D1",
        )

    def test_pole_order4_keeps_first_interior_rows_as_interior(self):
        """Order-4 pole matrices keep rows w..r-1 on the interior stencil."""
        n = 40
        dz = 1.0 / (n - 0.5)
        stencils = sbp_stencils(4)
        a_i1 = stencils["a_i1"]
        a_i2 = stencils["a_i2"]
        w = (len(a_i1) - 1) // 2

        D1_sym, D1_asymm, D2_sym, D2_asymm = build_sbp_pole_matrices(n, dz, 4)
        for row in [w, w + 1]:
            cols = slice(row - w, row + w + 1)
            np.testing.assert_allclose(D1_sym[row, cols].toarray().ravel(), a_i1 / dz, atol=1e-13)
            np.testing.assert_allclose(D1_asymm[row, cols].toarray().ravel(), a_i1 / dz, atol=1e-13)
            np.testing.assert_allclose(
                D2_sym[row, cols].toarray().ravel(), a_i2 / dz**2, atol=1e-11
            )
            np.testing.assert_allclose(
                D2_asymm[row, cols].toarray().ravel(), a_i2 / dz**2, atol=1e-11
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
                D1_dense[i],
                np.roll(row0, i),
                atol=1e-13,
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
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=20,
            ny=20,
            fd_order=4,
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
        cavity_adjacent = (x_mi >= 0.3) & (x_mi <= 0.65) & (y_mi >= 0.3) & (y_mi <= 0.65)
        assert np.any(cavity_adjacent), "no mi points near cavity boundary"


# MATLAB-export parity check; the reference bundle is kept out of the repository
# and not distributed. Skips cleanly when the bundle is absent.
def test_cavity_mask_boundary_matches_exported_matlab_bundle() -> None:
    from tests._paths import martini_fixture

    bundle_path = martini_fixture("example_4_Cavity_results_base.npz")
    if not bundle_path.exists():
        pytest.skip(f"Missing MATLAB parity bundle: {bundle_path}")
    matlab = np.load(bundle_path, allow_pickle=False)

    x_cavity_start = 0.34029
    L_cavity = 1.9995
    mesh = Mesh.create(
        xrange=(x_cavity_start - L_cavity, x_cavity_start + 2 * L_cavity),
        yrange=(-1.0, 1.0),
        nx=241,
        ny=101,
        fd_order=4,
        y_symmetry=True,
        x_periodicity=False,
        alpha_filter=0.0,
    )
    remove = (mesh.Y < 0.0) & ((mesh.X < 0.34) | (mesh.X > 2.34))
    masked = mesh.mask(~remove)

    Y_new = masked.Y.copy()
    Y_new = (Y_new**3 + 0.5 * Y_new) / 1.5
    p = Y_new > 0.0
    Y_new[p] = Y_new[p] + 2.0 * (Y_new[p] / np.max(Y_new[p])) ** 3
    masked = masked.deform(Y_new=Y_new)

    X_new = masked.X.copy()
    p = X_new > 2.8
    X_new[p] = X_new[p] + 1.0 * ((X_new[p] - 2.8) / (np.max(X_new[p]) - 2.8)) ** 3
    masked = masked.deform(X_new=X_new)

    np.testing.assert_array_equal(
        masked.idx["mi"] + 1,
        matlab["results_Cavity_base__mesh__idx__mi"].reshape(-1).astype(int),
    )


# MATLAB-export parity check; the reference bundle is kept out of the repository
# and not distributed. Skips cleanly when the bundle is absent.
def test_cavity_sponge_matches_exported_matlab_bundle() -> None:
    from tests._paths import martini_fixture

    bundle_path = martini_fixture("example_4_Cavity_results_base.npz")
    if not bundle_path.exists():
        pytest.skip(f"Missing MATLAB parity bundle: {bundle_path}")
    matlab = np.load(bundle_path, allow_pickle=False)

    x_cavity_start = 0.34029
    L_cavity = 1.9995
    mesh = Mesh.create(
        xrange=(x_cavity_start - L_cavity, x_cavity_start + 2 * L_cavity),
        yrange=(-1.0, 1.0),
        nx=241,
        ny=101,
        fd_order=4,
        y_symmetry=True,
        x_periodicity=False,
        alpha_filter=0.0,
    )
    remove = (mesh.Y < 0.0) & ((mesh.X < 0.34) | (mesh.X > 2.34))
    mesh = mesh.mask(~remove)

    Y_new = mesh.Y.copy()
    Y_new = (Y_new**3 + 0.5 * Y_new) / 1.5
    p = Y_new > 0.0
    Y_new[p] = Y_new[p] + 2.0 * (Y_new[p] / np.max(Y_new[p])) ** 3
    mesh = mesh.deform(Y_new=Y_new)

    X_new = mesh.X.copy()
    p = X_new > 2.8
    X_new[p] = X_new[p] + 1.0 * ((X_new[p] - 2.8) / (np.max(X_new[p]) - 2.8)) ** 3
    mesh = mesh.deform(X_new=X_new)

    xs_trans = 0.5
    xs_offset = 0.5
    ys_trans = 0.5
    ys_offset = 1.0
    sponge_amp = 5.0
    sponge = np.full(mesh.X.shape, np.nan, dtype=float)
    used = mesh.used_ind
    x = mesh.X.ravel(order="F")[used]
    y = mesh.Y.ravel(order="F")[used]
    term_x = np.tanh(-(x - np.min(mesh.X) - xs_offset) / xs_trans) + 1
    term_y = np.tanh((x - np.max(mesh.X) + xs_offset) / xs_trans) + 1
    term_t = np.tanh((y - np.max(mesh.Y) + ys_offset) / ys_trans) + 1
    sponge_flat = sponge.ravel(order="F")
    sponge_flat[used] = sponge_amp / 2 * np.maximum(term_t, term_x + term_y)
    sponge = sponge_flat.reshape(mesh.X.shape, order="F")

    matlab_sponge = matlab["results_Cavity_base__mesh__sponge"]
    matlab_used = matlab["results_Cavity_base__mesh__usedInd"].reshape(-1).astype(int) - 1
    np.testing.assert_array_equal(used, matlab_used)
    np.testing.assert_allclose(
        sponge.T.ravel(order="F")[used],
        matlab_sponge.ravel(order="F")[used],
        atol=1e-12,
        rtol=1e-12,
    )


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


def test_deform_separable_xy_reuses_symmetry_y_only_family():
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

    np.testing.assert_allclose(full.Dy_symm.toarray(), seq.Dy_symm.toarray())
    np.testing.assert_allclose(full.D2y_symm.toarray(), seq.D2y_symm.toarray())
    np.testing.assert_allclose(full.Dxy_symm.toarray(), seq.Dxy_symm.toarray())
    np.testing.assert_allclose(full.Dyx_symm.toarray(), seq.Dyx_symm.toarray())
    np.testing.assert_allclose(full.Dy_asymm.toarray(), seq.Dy_asymm.toarray())
    np.testing.assert_allclose(full.D2y_asymm.toarray(), seq.D2y_asymm.toarray())
    np.testing.assert_allclose(full.Dxy_asymm.toarray(), seq.Dxy_asymm.toarray())
    np.testing.assert_allclose(full.Dyx_asymm.toarray(), seq.Dyx_asymm.toarray())


def test_deform_full_2d_family_metric_terms_use_full_operators():
    """Family dJ/Q metric terms must come from the full eta operators."""
    base = Mesh.create(
        xrange=(-0.5, 0.5),
        yrange=(0.0, 1.0),
        nx=10,
        ny=16,
        fd_order=4,
        y_symmetry=True,
        x_periodicity=False,
        alpha_filter=0.0,
    )
    X_new = base.X + 0.05 * base.Y**2
    Y_new = base.Y * (1.0 + 0.05 * base.X)

    deformed = base.deform(X_new=X_new, Y_new=Y_new)

    p = base.used_ind
    x = X_new.ravel(order="F")[p]
    y = Y_new.ravel(order="F")[p]
    Dx, D2x, Dy, D2y = base.Dx, base.D2x, base.Dy, base.D2y

    J11, J12, J21, J22 = Dx @ x, Dx @ y, Dy @ x, Dy @ y
    det = J11 * J22 - J12 * J21
    a, b, c, d = J22 / det, -J12 / det, -J21 / det, J11 / det

    dJ_xi = (D2x @ x, D2x @ y, Dx @ (Dy @ x), Dx @ (Dy @ y))
    dJ_eta = (Dy @ (Dx @ x), Dy @ (Dx @ y), D2y @ x, D2y @ y)

    def dji_row2(dj11, dj12, dj21, dj22):
        p11 = dj11 * a + dj12 * c
        p12 = dj11 * b + dj12 * d
        p21 = dj21 * a + dj22 * c
        p22 = dj21 * b + dj22 * d
        return -(c * p11 + d * p21), -(c * p12 + d * p22)

    dji21_xi, dji22_xi = dji_row2(*dJ_xi)
    dji21_eta, dji22_eta = dji_row2(*dJ_eta)
    q22_x = c * dji21_xi + d * dji21_eta
    q22_y = c * dji22_xi + d * dji22_eta

    def dg(vec):
        return sparse.diags(vec, offsets=0, format="csr")

    for fam in ("symm", "asymm"):
        expected = (
            dg(c * c) @ D2x
            + dg(c * d) @ getattr(base, f"Dxy_{fam}")
            + dg(d * c) @ getattr(base, f"Dyx_{fam}")
            + dg(d * d) @ getattr(base, f"D2y_{fam}")
            + dg(q22_x) @ Dx
            + dg(q22_y) @ getattr(base, f"Dy_{fam}")
        ).toarray()
        actual = getattr(deformed, f"D2y_{fam}").toarray()
        scale = np.abs(expected).max()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10 * scale)


def test_axm_diag_matches_explicit_sparse_diagonal_scaling():
    """Row scaling must remain identical to the old explicit diagonal product."""
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, -2.0],
                [0.0, 3.0, 4.0],
                [5.0, 0.0, 6.0],
            ]
        )
    )
    scale = np.array([2.0, -1.5, 0.25])

    explicit = sparse.diags(scale, offsets=0, shape=matrix.shape, format="csr") @ matrix
    scaled = _axm_diag(scale, matrix)

    assert sparse.isspmatrix_csr(scaled)
    np.testing.assert_allclose(scaled.toarray(), explicit.toarray())


def test_axm_diag_preserves_complex_scaling():
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1.0 + 2.0j, 0.0, -2.0j],
                [0.0, 3.0 - 1.0j, 4.0],
                [5.0, 0.0, 6.0 + 0.5j],
            ]
        )
    )
    scale = np.array([2.0 - 1.0j, -1.5, 0.25 + 0.75j])

    explicit = sparse.diags(scale, offsets=0, shape=matrix.shape, format="csr") @ matrix
    scaled = _axm_diag(scale, matrix)

    np.testing.assert_allclose(scaled.toarray(), explicit.toarray())


def test_axm_diag_rejects_wrong_row_count():
    matrix = sparse.eye(3, format="csr")
    with pytest.raises(ValueError, match="row count"):
        _axm_diag(np.ones(2), matrix)


def test_axm_diag_accepts_a_column_scale_vector():
    matrix = sparse.eye(3, format="csr")
    scaled = _axm_diag(np.array([[2.0], [3.0], [4.0]]), matrix)
    np.testing.assert_allclose(scaled.diagonal(), np.array([2.0, 3.0, 4.0]))


def test_mesh_create_rejects_bad_ranges_and_sizes():
    with pytest.raises(TypeError, match="pair"):
        _validate_range_pair("xrange", 1.0)
    with pytest.raises(ValueError, match="pair"):
        _validate_range_pair("xrange", (0.0, 1.0, 2.0))
    with pytest.raises(TypeError, match="finite numbers"):
        _validate_range_pair("xrange", (True, False))
    with pytest.raises(ValueError, match="finite"):
        _validate_range_pair("xrange", (0.0, np.inf))
    with pytest.raises(ValueError, match="distinct"):
        _validate_range_pair("xrange", (1.0, 1.0))
    with pytest.raises(TypeError, match="integer"):
        _validate_create_inputs((0.0, 1.0), (0.0, 1.0), 8.5, 8, 4)
    with pytest.raises(ValueError, match="fd_order"):
        _validate_create_inputs((0.0, 1.0), (0.0, 1.0), 20, 20, 3)
    with pytest.raises(ValueError, match="nx must be"):
        Mesh.create(xrange=(0.0, 1.0), yrange=(0.0, 1.0), nx=4, ny=20, fd_order=4)
    with pytest.raises(ValueError, match="ny must be"):
        Mesh.create(xrange=(0.0, 1.0), yrange=(0.0, 1.0), nx=20, ny=4, fd_order=4)


def test_deform_without_new_coordinates_returns_the_same_mesh():
    mesh = Mesh.create(xrange=(0.0, 1.0), yrange=(0.0, 1.0), nx=12, ny=12, fd_order=2)
    assert mesh.deform() is mesh


def test_y_symmetry_without_a_pole_reuses_the_plain_y_operators():
    mesh = Mesh.create(
        xrange=(0.0, 1.0),
        yrange=(1.0, 2.0),
        nx=12,
        ny=12,
        fd_order=2,
        y_symmetry=True,
    )
    assert mesh.Dy_symm is not None
    np.testing.assert_allclose(mesh.Dy_symm.toarray(), mesh.Dy.toarray())
    np.testing.assert_allclose(mesh.Dy_asymm.toarray(), mesh.Dy.toarray())


def test_mask_skips_short_line_segments():
    mesh = Mesh.create(xrange=(0.0, 1.0), yrange=(0.0, 1.0), nx=10, ny=10, fd_order=2)
    keep = np.ones_like(mesh.X, dtype=bool)
    keep[0, 2:] = False
    masked = mesh.mask(keep)
    assert masked.ngp == int(keep.sum())
    assert masked.Dx.shape == (masked.ngp, masked.ngp)


def test_line_breaks_and_periodic_operators_reject_degenerate_input():
    np.testing.assert_array_equal(_line_breaks(np.array([])), np.array([], dtype=int))
    np.testing.assert_array_equal(_line_breaks(np.zeros(4)), np.array([], dtype=int))
    with pytest.raises(ValueError, match="at least 2"):
        _build_periodic_1d_operators(1, 0.1, 4)
    d1, d2 = _build_periodic_1d_operators(6, 0.1, 8)
    assert d1.shape == (6, 6)
    assert d2.shape == (6, 6)


def test_fdweights_empty_nodes_return_an_empty_stencil():
    weights = fdweights(0.0, [], 1)
    assert weights.shape == (0,)
    assert weights.dtype == np.float64


@pytest.mark.parametrize(
    ("i0", "i1", "j0", "j1"), [(10, 11, 0, 25), (10, 12, 0, 25), (10, 13, 0, 25), (10, 11, 12, 13)]
)
def test_mask_splits_lines_at_a_hole(i0, i1, j0, j1):
    """A derivative line must not span a hole, even one point wide."""
    mesh = Mesh.create(xrange=(0.0, 1.0), yrange=(0.0, 1.0), nx=25, ny=25, fd_order=4)
    keep = np.ones(mesh.X.shape, dtype=bool)
    keep[j0:j1, i0:i1] = False
    masked = mesh.mask(keep)
    flat = np.flatnonzero(keep.ravel(order="F"))
    x_kept = mesh.X.ravel(order="F")[flat]
    y_kept = mesh.Y.ravel(order="F")[flat]
    np.testing.assert_allclose(masked.Dx @ x_kept, 1.0, atol=1e-10)
    np.testing.assert_allclose(masked.Dy @ y_kept, 1.0, atol=1e-10)


def test_line_breaks_ignore_smooth_stretching():
    deltas = 1.2 ** np.arange(20)  # 20 % growth per cell: no hole
    assert _line_breaks(deltas).size == 0
    holed = deltas.copy()
    holed[7] *= 2.0  # one removed point
    np.testing.assert_array_equal(_line_breaks(holed), [7])
