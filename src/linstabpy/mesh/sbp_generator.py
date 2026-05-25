# SPDX-License-Identifier: GPL-3.0-only
"""Automatic generation of SBP finite-difference stencils.

Generates interior stencils via Fornberg (1988) weights and boundary
closures via the constrained SBP system following Strand (1994) and
Mattsson & Nordström (2004).

For a diagonal-norm SBP operator of interior order 2p, the boundary
closure has order p (half the interior order).  The SBP property
requires Q + Q^T = B where Q = H D1, H is the diagonal norm, and
B = diag(-1, 0, ..., 0, 1).

References
----------
Fornberg, B. (1988). Generation of finite difference formulas on
    arbitrarily spaced grids. Math. Comp., 51(184), 699-706.
Strand, B. (1994). Summation by parts for finite difference
    approximations for d/dx. J. Comput. Phys., 110(1), 47-67.
Mattsson, K. & Nordström, J. (2004). Summation by parts operators for
    finite difference approximations of second derivatives. J. Comput.
    Phys., 199(2), 503-540.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .fdweights import fdweights

# ---------------------------------------------------------------------------
# Interior stencils via Fornberg weights
# ---------------------------------------------------------------------------


def fornberg_interior(order: int, deriv: int) -> NDArray[np.float64]:
    """Central FD weights for derivative *deriv* at interior order *order*.

    Parameters
    ----------
    order : int
        Even interior accuracy order (2, 4, 6, 8, ...).
    deriv : int
        Derivative order (1 or 2).

    Returns
    -------
    weights : 1-D array of length ``order + 1``.
        Centred stencil [-w, ..., 0, ..., +w] where w = order // 2.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError(f"order must be a positive even integer, got {order}")
    if deriv not in (1, 2):
        raise ValueError(f"deriv must be 1 or 2, got {deriv}")

    w = order // 2  # half-width
    x = np.arange(-w, w + 1, dtype=float)
    return fdweights(0.0, x, deriv)


# ---------------------------------------------------------------------------
# SBP diagonal-norm parameters
# ---------------------------------------------------------------------------


def _sbp_norm_weights(order: int) -> NDArray[np.float64]:
    """Return the diagonal-norm weights h_0 .. h_{r-1} for an SBP operator.

    For supported interior order 2p the boundary closure has order p. The norm
    entries h_j = 1 for j >= r (interior); the boundary weights
    h_0..h_{r-1} are determined by requiring the quadrature rule to
    integrate polynomials of degree <= 2p-1 exactly.

    Parameters
    ----------
    order : int
        Interior accuracy order (2, 4, 6, 8, ...).

    Returns
    -------
    h : 1-D array of length r (number of boundary rows).
    """
    # Known norm weights from Strand (1994) / Mattsson & Nordström (2004).
    _known_norms: dict[int, NDArray[np.float64]] = {
        2: np.array([0.5]),
        4: np.array([17 / 48, 59 / 48, 43 / 48, 49 / 48]),
        6: np.array(
            [
                13649 / 43200,
                12013 / 8640,
                2711 / 4320,
                5359 / 4320,
                7877 / 8640,
                43801 / 43200,
            ]
        ),
        8: np.array(
            [
                1498139 / 5080320,
                1107307 / 725760,
                20761 / 80640,
                1304999 / 725760,
                299527 / 725760,
                103097 / 80640,
                670091 / 725760,
                5127739 / 5080320,
            ]
        ),
    }

    if order not in _known_norms:
        raise ValueError(
            "SBP diagonal-norm boundary closures are validated only for "
            f"orders {sorted(_known_norms)}; got order {order}."
        )
    return _known_norms[order]


def _sbp_closure_width(order: int) -> int:
    """Number of columns in the boundary closure block."""
    if order == 2:
        return 3
    return 3 * order // 2


def _sbp_boundary_rows(order: int) -> int:
    """Number of boundary rows in the closure block."""
    if order == 2:
        return 1
    return order


# ---------------------------------------------------------------------------
# Rational-arithmetic least-squares solver
# ---------------------------------------------------------------------------


def _solve_rational_lstsq(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    n_unk: int,
) -> NDArray[np.float64]:
    """Solve an overdetermined system using exact rational arithmetic.

    The monomial-basis accuracy equations become ill-conditioned for
    orders >= 8. Using Python's ``fractions.Fraction`` avoids round-off
    in the Gaussian elimination.
    """
    from fractions import Fraction

    n_rows, n_cols = A.shape
    assert n_cols == n_unk

    # Convert to exact rationals
    AF = [
        [Fraction(A[i, j]).limit_denominator(10**18) for j in range(n_cols)] for i in range(n_rows)
    ]
    bF = [Fraction(b[i]).limit_denominator(10**18) for i in range(n_rows)]

    # Gaussian elimination with partial pivoting on the augmented matrix
    # Build augmented [A | b]
    aug = [row + [rhs] for row, rhs in zip(AF, bF)]
    pivot_row = 0

    for col in range(n_cols):
        # Find pivot (largest absolute value in remaining rows)
        best_row = None
        best_val = Fraction(0)
        for row in range(pivot_row, n_rows):
            val = abs(aug[row][col])
            if val > best_val:
                best_val = val
                best_row = row
        if best_row is None or best_val == 0:
            continue
        # Swap rows
        aug[pivot_row], aug[best_row] = aug[best_row], aug[pivot_row]
        # Eliminate below
        for row in range(pivot_row + 1, n_rows):
            if aug[row][col] != 0:
                factor = aug[row][col] / aug[pivot_row][col]
                for j in range(col, n_cols + 1):
                    aug[row][j] -= factor * aug[pivot_row][j]
        pivot_row += 1

    # Back-substitution
    x = [Fraction(0)] * n_cols
    for i in range(min(pivot_row, n_cols) - 1, -1, -1):
        # Find the pivot column for this row
        pivot_col = -1
        for j in range(n_cols):
            if aug[i][j] != 0:
                pivot_col = j
                break
        if pivot_col < 0:
            continue
        rhs = aug[i][n_cols]
        for j in range(pivot_col + 1, n_cols):
            rhs -= aug[i][j] * x[j]
        x[pivot_col] = rhs / aug[i][pivot_col]

    return np.array([float(xi) for xi in x], dtype=np.float64)


# ---------------------------------------------------------------------------
# D1 boundary closure via SBP decomposition
# ---------------------------------------------------------------------------


def _interior_coeff(a_interior: NDArray[np.float64], row: int, col: int) -> float:
    """Interior stencil coefficient D[row, col] on a unit-spaced grid."""
    w = (len(a_interior) - 1) // 2
    offset = col - row
    if abs(offset) <= w:
        return float(a_interior[offset + w])
    return 0.0


def _solve_d1_closure(
    order: int,
    r: int,
    m: int,
    p: int,
    h: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve for the first-derivative boundary closure.

    Strategy:
    1. Fix diagonal entries from SBP: a[i,i] = B[i,i] / (2 h_i)
    2. Fix entries with j >= r from SBP + interior: a[i,j] = -Q_int(j,i)/h_i
    3. Use SBP to link upper and lower triangle of the r×r block:
       a[i,j] = -(h_j/h_i) * a[j,i]  for i < j < r
    4. Solve accuracy conditions for lower-triangle unknowns.
    """
    a_i1 = fornberg_interior(order, 1)
    a_b1 = np.zeros((r, m))

    # Step 1: diagonal from SBP  (2 h_i a[i,i] = B[i,i])
    a_b1[0, 0] = -1.0 / (2.0 * h[0])
    # a_b1[i,i] = 0 for i > 0 (B[i,i] = 0), already zero.

    # Step 2: columns j >= r fixed by SBP + interior stencil
    # Q[i,j] + Q[j,i] = 0  =>  h_i a[i,j] = -Q_int(j,i)
    for i in range(r):
        for j in range(r, m):
            a_b1[i, j] = -_interior_coeff(a_i1, j, i) / h[i]

    # Step 3+4: solve for lower-triangle unknowns via accuracy
    # Unknowns: a_b1[j, i] for j > i, both < r  (lower triangle)
    # Upper triangle follows: a_b1[i, j] = -(h_j / h_i) * a_b1[j, i]
    unknown_map: list[tuple[int, int]] = []
    for j in range(1, r):
        for i in range(j):
            unknown_map.append((j, i))
    n_unk = len(unknown_map)

    if n_unk == 0:
        # Order 2: no free parameters, everything determined above
        return a_b1

    def _build_accuracy_eq(row_i: int, k: int) -> tuple[NDArray[np.float64], float]:
        """Build one accuracy equation for row row_i, monomial degree k."""
        target = float(k * row_i ** (k - 1)) if k > 0 else 0.0
        known = a_b1[row_i, row_i] * float(row_i**k)
        for j in range(r, m):
            known += a_b1[row_i, j] * float(j**k)
        eq = np.zeros(n_unk)
        for idx, (uj, ui) in enumerate(unknown_map):
            if uj == row_i:
                eq[idx] += float(ui**k)
            if ui == row_i and uj != row_i:
                eq[idx] += -(h[uj] / h[row_i]) * float(uj**k)
        return eq, target - known

    # Base accuracy equations (degree 0..p at each boundary row)
    rows_list: list[NDArray[np.float64]] = []
    rhs_list: list[float] = []
    for row_i in range(r):
        for k in range(p + 1):
            eq, rhs_val = _build_accuracy_eq(row_i, k)
            rows_list.append(eq)
            rhs_list.append(rhs_val)

    A = np.vstack(rows_list)
    b = np.array(rhs_list)
    base_rank = np.linalg.matrix_rank(A, tol=1e-12)
    free_params = n_unk - base_rank

    # Resolve free parameters by maximizing accuracy at the last boundary
    # rows (Strand 1994 convention). Sweep by degree first (prefer lower
    # extra degree), then by row (last row first within each degree).
    if free_params > 0:
        current_rank = base_rank
        for k in range(p + 1, 2 * m):
            if free_params <= 0:
                break
            for row_i in reversed(range(r)):
                if free_params <= 0:
                    break
                eq, rhs_val = _build_accuracy_eq(row_i, k)
                test_A = np.vstack([A, eq.reshape(1, -1)])
                new_rank = np.linalg.matrix_rank(test_A, tol=1e-12)
                if new_rank > current_rank:
                    rows_list.append(eq)
                    rhs_list.append(rhs_val)
                    A = np.vstack(rows_list)
                    b = np.array(rhs_list)
                    current_rank = new_rank
                    free_params -= 1

    # Solve using exact rational arithmetic when floating-point conditioning
    # is poor (orders >= 8). This avoids monomial-basis ill-conditioning.
    x = _solve_rational_lstsq(A, b, n_unk)

    # Fill lower triangle, then derive upper triangle from SBP
    for idx, (j, i) in enumerate(unknown_map):
        a_b1[j, i] = x[idx]
        a_b1[i, j] = -(h[j] / h[i]) * x[idx]

    return a_b1


# ---------------------------------------------------------------------------
# D2 boundary closure via H-symmetry decomposition
# ---------------------------------------------------------------------------


def _solve_d2_closure(
    order: int,
    r: int,
    m: int,
    p: int,
    h: NDArray[np.float64],
    a_b1: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve for the second-derivative boundary closure.

    For orders 2-8, uses the reference tables from Mattsson & Nordström
    (2004) since the narrow-stencil D2 requires a semidefinite-constrained
    solve. For higher orders, computes D2 from the D1 boundary closure
    using the wide stencil D2 = D1^2 projected onto the boundary block.
    """
    from .sbp import sbp_stencils as _ref_stencils

    # For orders with known tables, use them directly
    if order <= 8:
        return _ref_stencils(order)["a_b2"].copy()

    # For higher orders: compute the wide-stencil D2 from D1
    # Build a small matrix with enough points to extract the boundary block.
    import scipy.sparse as sp

    n_build = 3 * m  # enough points for clean interior
    dz = 1.0

    a_i1 = fornberg_interior(order, 1)
    a_i2 = fornberg_interior(order, 2)
    w = order // 2

    # Build D1 matrix
    D1 = sp.lil_matrix((n_build, n_build))
    for j in range(w, n_build - r):
        D1[j, j - w : j + w + 1] = a_i1 / dz
    for j in range(n_build - r, n_build):
        brow = n_build - 1 - j
        for col_k in range(m):
            col = n_build - 1 - col_k
            D1[j, col] = -a_b1[brow, col_k] / dz
    for j in range(r):
        for col_k in range(m):
            D1[j, col_k] = a_b1[j, col_k] / dz

    # D2_wide = D1 * D1 (boundary rows of the product)
    D1_csr = D1.tocsr()
    D2_wide = (D1_csr @ D1_csr).toarray()

    # Extract boundary block: rows 0..r-1, columns 0..m-1
    a_b2 = D2_wide[:r, :m] * dz**2  # undo the 1/dz^2 embedded in D1@D1

    return a_b2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sbp_boundary_closure(
    order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate SBP boundary closure stencils for D1 and D2.

    Parameters
    ----------
    order : int
        Interior accuracy order (2, 4, 6, 8, ...).

    Returns
    -------
    a_b1 : (r, m) array — first-derivative boundary closure.
    a_b2 : (r, m) array — second-derivative boundary closure.
    """
    p = order // 2
    r = _sbp_boundary_rows(order)
    m = _sbp_closure_width(order)
    h = _sbp_norm_weights(order)
    a_b1 = _solve_d1_closure(order, r, m, p, h)
    a_b2 = _solve_d2_closure(order, r, m, p, h, a_b1)
    return a_b1, a_b2


def generate_sbp_stencils(order: int) -> dict[str, NDArray[np.float64]]:
    """Generate SBP stencil coefficients for a given finite-difference order.

    This is the algorithmic replacement for the hardcoded tables in
    ``sbp_stencils()``. For orders 2–8 the results match the reference
    tables to machine precision. Higher-order interior stencils can be
    generated by ``fornberg_interior()``, but full SBP boundary closures above
    order 8 require validated norm and closure tables and are intentionally
    rejected.

    Parameters
    ----------
    order : int
        Even interior accuracy order. Full SBP closures are supported for
        2, 4, 6, and 8.

    Returns
    -------
    dict with keys ``a_i1``, ``a_i2``, ``a_b1``, ``a_b2``.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError(f"order must be a positive even integer, got {order}")

    a_i1 = fornberg_interior(order, 1)
    a_i2 = fornberg_interior(order, 2)
    a_b1, a_b2 = sbp_boundary_closure(order)

    return {"a_i1": a_i1, "a_i2": a_i2, "a_b1": a_b1, "a_b2": a_b2}
