# SPDX-License-Identifier: GPL-3.0-only
"""Summation-by-parts (SBP) finite difference stencils and matrix construction.

Implements standard SBP operators of orders 2, 4, 6, and 8 following
the stencil coefficients from Strand (1994) and Mattsson & Nordstrom (2004).
Pole-condition matrices (Mohseni & Colonius) are included for axisymmetric grids.

References
----------
Strand, B. (1994). Summation by parts for finite difference approximations
    for d/dx. J. Comput. Phys., 110(1), 47-67.
Mattsson, K., & Nordstrom, J. (2004). Summation by parts operators for
    finite difference approximations of second derivatives. J. Comput. Phys.,
    199(2), 503-540.
"""

from __future__ import annotations

import functools

import numpy as np
import scipy.sparse as sp


def sbp_stencils(order: int) -> dict[str, np.ndarray]:
    """Return SBP stencil coefficients for a given finite-difference order.

    For orders 2–8, returns the hardcoded reference tables from Strand (1994)
    and Mattsson & Nordström (2004). Higher-order SBP boundary closures are
    intentionally rejected until validated norm and closure tables are added.

    Parameters
    ----------
    order : int
        Even finite-difference order. Full SBP closures are supported for
        2, 4, 6, and 8.

    Returns
    -------
    dict with keys:
        ``a_i1`` : 1-D array — interior first-derivative stencil.
        ``a_i2`` : 1-D array — interior second-derivative stencil.
        ``a_b1`` : 2-D array (r, m) — boundary closure, first derivative.
        ``a_b2`` : 2-D array (r, m) — boundary closure, second derivative.
        Here *r* is the number of boundary rows and *m* is the closure width.

    Raises
    ------
    ValueError
        If *order* is not a positive even integer.
    """
    if order == 2:
        a_i1 = np.array([-1 / 2, 0, 1 / 2])
        a_i2 = np.array([1, -2, 1])
        a_b1 = np.array([[-1, 1, 0]])
        a_b2 = np.array([[1, -2, 1]])

    elif order == 4:
        a_i1 = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
        a_i2 = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])

        a_b1 = np.array([
            [-24 / 17, 59 / 34, -4 / 17, -3 / 34, 0, 0],
            [-1 / 2, 0, 1 / 2, 0, 0, 0],
            [4 / 43, -59 / 86, 0, 59 / 86, -4 / 43, 0],
            [3 / 98, 0, -59 / 98, 0, 32 / 49, -4 / 49],
        ])
        a_b2 = np.array([
            [2, -5, 4, -1, 0, 0],
            [1, -2, 1, 0, 0, 0],
            [-4 / 43, 59 / 43, -110 / 43, 59 / 43, -4 / 43, 0],
            [-1 / 49, 0, 59 / 49, -118 / 49, 64 / 49, -4 / 49],
        ])

    elif order == 6:
        a_i1 = np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60])
        a_i2 = np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90])

        a_b1 = np.array([
            [-21600 / 13649, 104009 / 54596, 30443 / 81894, -33311 / 27298, 16863 / 27298, -15025 / 163788, 0, 0, 0],
            [-104009 / 240260, 0, -311 / 72078, 20229 / 24026, -24337 / 48052, 36661 / 360390, 0, 0, 0],
            [-30443 / 162660, 311 / 32532, 0, -11155 / 16266, 41287 / 32532, -21999 / 54220, 0, 0, 0],
            [33311 / 107180, -20229 / 21436, 485 / 1398, 0, 4147 / 21436, 25427 / 321540, 72 / 5359, 0, 0],
            [-16863 / 78770, 24337 / 31508, -41287 / 47262, -4147 / 15754, 0, 342523 / 472620, -1296 / 7877, 144 / 7877, 0],
            [15025 / 525612, -36661 / 262806, 21999 / 87602, -25427 / 262806, -342523 / 525612, 0, 32400 / 43801, -6480 / 43801, 720 / 43801],
        ])
        a_b2 = np.array([
            [114170 / 40947, -438107 / 54596, 336409 / 40947, -276997 / 81894, 3747 / 13649, 21035 / 163788, 0, 0, 0],
            [6173 / 5860, -2066 / 879, 3283 / 1758, -303 / 293, 2111 / 3516, -601 / 4395, 0, 0, 0],
            [-52391 / 81330, 134603 / 32532, -21982 / 2711, 112915 / 16266, -46969 / 16266, 30409 / 54220, 0, 0, 0],
            [68603 / 321540, -12423 / 10718, 112915 / 32154, -75934 / 16077, 53369 / 21436, -54899 / 160770, 48 / 5359, 0, 0],
            [-7053 / 39385, 86551 / 94524, -46969 / 23631, 53369 / 15754, -87904 / 23631, 820271 / 472620, -1296 / 7877, 96 / 7877, 0],
            [21035 / 525612, -24641 / 131403, 30409 / 87602, -54899 / 131403, 820271 / 525612, -117600 / 43801, 64800 / 43801, -6480 / 43801, 480 / 43801],
        ])

    elif order == 8:
        a_i1 = np.array([
            1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280,
        ])
        a_i2 = np.array([
            -1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560,
        ])

        a_b1 = np.array([
            [-2540160 / 1498139, 5544277 / 5992556, 198794991 / 29962780, -256916579 / 17977668, 20708767 / 1498139, -41004357 / 5992556, 27390659 / 17977668, -2323531 / 29962780, 0, 0, 0, 0],
            [-5544277 / 31004596, 0, -85002381 / 22146140, 49607267 / 4429228, -165990199 / 13287684, 7655859 / 1107307, -7568311 / 4429228, 48319961 / 465068940, 0, 0, 0, 0],
            [-66264997 / 8719620, 9444709 / 415220, 0, -20335981 / 249132, 32320879 / 249132, -35518713 / 415220, 2502774 / 103805, -3177073 / 1743924, 0, 0, 0, 0],
            [256916579 / 109619916, -49607267 / 5219996, 61007943 / 5219996, 0, -68748371 / 5219996, 65088123 / 5219996, -66558305 / 15659988, 3870214 / 9134993, 0, 0, 0, 0],
            [-20708767 / 2096689, 165990199 / 3594324, -96962637 / 1198108, 68748371 / 1198108, 0, -27294549 / 1198108, 14054993 / 1198108, -42678199 / 25160268, -2592 / 299527, 0, 0, 0],
            [13668119 / 8660148, -850651 / 103097, 35518713 / 2061940, -21696041 / 1237164, 9098183 / 1237164, 0, -231661 / 412388, 7120007 / 43300740, 3072 / 103097, -288 / 103097, 0, 0],
            [-27390659 / 56287644, 7568311 / 2680364, -22524966 / 3350455, 66558305 / 8041092, -14054993 / 2680364, 2084949 / 2680364, 0, 70710683 / 93812740, -145152 / 670091, 27648 / 670091, -2592 / 670091, 0],
            [2323531 / 102554780, -48319961 / 307664340, 9531219 / 20510956, -3870214 / 5127739, 2246221 / 3238572, -21360021 / 102554780, -70710683 / 102554780, 0, 4064256 / 5127739, -1016064 / 5127739, 193536 / 5127739, -18144 / 5127739],
        ])

        a_b2 = np.array([
            [4870382994799 / 1358976868290, -893640087518 / 75498714905, 926594825119 / 60398971924, -1315109406200 / 135897686829, 39126983272 / 15099742981, 12344491342 / 75498714905, -451560522577 / 2717953736580, 0, 0, 0, 0, 0],
            [333806012194 / 390619153855, -154646272029 / 111605472530, 1168338040 / 33481641759, 82699112501 / 133926567036, -171562838 / 11160547253, -28244698346 / 167408208795, 11904122576 / 167408208795, -2598164715 / 312495323084, 0, 0, 0, 0],
            [7838984095 / 52731029988, 1168338040 / 5649753213, -88747895 / 144865467, 423587231 / 627750357, -43205598281 / 22599012852, 4876378562 / 1883251071, -5124426509 / 3766502142, 10496900965 / 39548272491, 0, 0, 0, 0],
            [-94978241528 / 828644350023, 82699112501 / 157837019052, 1270761693 / 13153084921, -167389605005 / 118377764289, 48242560214 / 39459254763, -31673996013 / 52612339684, 43556319241 / 118377764289, -44430275135 / 552429566682, 0, 0, 0, 0],
            [1455067816 / 21132528431, -171562838 / 3018932633, -43205598281 / 36227191596, 48242560214 / 9056797899, -52276055645 / 6037865266, 57521587238 / 9056797899, -80321706377 / 36227191596, 8078087158 / 21132528431, -1296 / 299527, 0, 0, 0],
            [10881504334 / 327321118845, -28244698346 / 140280479505, 4876378562 / 9352031967, -10557998671 / 12469375956, 57521587238 / 28056095901, -278531401019 / 93520319670, 73790130002 / 46760159835, -137529995233 / 785570685228, 2048 / 103097, -144 / 103097, 0, 0],
            [-135555328849 / 8509847458140, 11904122576 / 101307707835, -5124426509 / 13507694378, 43556319241 / 60784624701, -80321706377 / 81046166268, 73790130002 / 33769235945, -950494905688 / 303923123505, 239073018673 / 141830790969, -145152 / 670091, 18432 / 670091, -1296 / 670091, 0],
            [0, -2598164715 / 206729925524, 10496900965 / 155047444143, -44430275135 / 310094888286, 425162482 / 2720130599, -137529995233 / 620189776572, 239073018673 / 155047444143, -144648000000 / 51682481381, 8128512 / 5127739, -1016064 / 5127739, 129024 / 5127739, -9072 / 5127739],
        ])

    else:
        from .sbp_generator import generate_sbp_stencils
        return generate_sbp_stencils(order)

    return {"a_i1": a_i1, "a_i2": a_i2, "a_b1": a_b1, "a_b2": a_b2}


def min_points(order: int) -> int:
    """Minimum number of grid points required for an SBP operator of *order*.

    This equals the closure width *m* (number of columns in ``a_b1``).
    """
    s = sbp_stencils(order)
    return s["a_b1"].shape[1]


@functools.lru_cache(maxsize=64)
def build_sbp_matrix(
    n: int, dz: float, order: int
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Construct 1-D SBP differentiation matrices D1 and D2.

    Builds the standard (non-pole) first- and second-derivative matrices
    for *n* grid points with uniform spacing *dz* using SBP closure of
    the given *order*.

    Parameters
    ----------
    n : int
        Number of grid points.
    dz : float
        Uniform grid spacing.
    order : int
        Finite-difference order (2, 4, 6, or 8).

    Returns
    -------
    D1 : scipy.sparse.csr_matrix, shape (n, n)
        First-derivative matrix.
    D2 : scipy.sparse.csr_matrix, shape (n, n)
        Second-derivative matrix.
    """
    stencils = sbp_stencils(order)
    a_i1 = stencils["a_i1"]
    a_i2 = stencils["a_i2"]
    a_b1 = stencils["a_b1"]
    a_b2 = stencils["a_b2"]

    r, m = a_b1.shape  # r = boundary rows, m = closure width
    w = (len(a_i1) - 1) // 2  # interior half-width

    # Use LIL format for efficient row-by-row construction
    D1 = sp.lil_matrix((n, n))
    D2 = sp.lil_matrix((n, n))

    # --- Interior points (MATLAB: jz = 1+w : Nz-r, 1-indexed) -----------
    # Python 0-indexed: j = w, w+1, ..., n-r-1
    for j in range(w, n - r):
        D1[j, j - w : j + w + 1] = a_i1 / dz
        D2[j, j - w : j + w + 1] = a_i2 / dz**2

    # --- Plus boundary (top rows, MATLAB: jz = Nz-r+1 : Nz) -------------
    # Python 0-indexed: j = n-r, n-r+1, ..., n-1
    # MATLAB: D1(jz, Nz+1-jm) = -a_b1(1+Nz-jz, jm) / dz
    # In Python: brow = n-1-j, col = n-1-col_k
    for j in range(n - r, n):
        brow = n - 1 - j  # boundary row index (0-indexed from end)
        for col_k in range(m):
            col = n - 1 - col_k  # reversed column index
            D1[j, col] = -a_b1[brow, col_k] / dz
            D2[j, col] = a_b2[brow, col_k] / dz**2

    # --- Minus boundary (bottom rows, MATLAB: jz = 1 : r) ---------------
    # Python 0-indexed: j = 0, 1, ..., r-1
    for j in range(r):
        for col_k in range(m):
            D1[j, col_k] = a_b1[j, col_k] / dz
            D2[j, col_k] = a_b2[j, col_k] / dz**2

    return D1.tocsr(), D2.tocsr()


def build_sbp_pole_matrices(
    n: int, dz: float, order: int
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Construct 1-D SBP matrices with Mohseni & Colonius pole closure.

    Returns ``D1s``/``D1a`` and ``D2s``/``D2a`` for first and second
    derivatives with symmetric (``s``) and antisymmetric (``a``) pole
    conditions.

    This is a direct implementation of ``Dmats_SBP.m`` from the MATLAB
    reference. It is only used for derivatives along the symmetry axis
    in axisymmetric meshes.
    """
    # --- Build base SBP stencils with pole-closure variants ----------------
    D1, D2, D1_sym, D1_asymm, D2_sym, D2_asymm = _build_sbp_with_pole(n, dz, order)

    return D1_sym.tocsr(), D1_asymm.tocsr(), D2_sym.tocsr(), D2_asymm.tocsr()


def _build_sbp_with_pole(n: int, dz: float, order: int):
    """Internal helper returning 4 derivative operators with pole conditions."""
    stencils = sbp_stencils(order)
    a_i1 = stencils["a_i1"]
    a_i2 = stencils["a_i2"]
    a_b1 = stencils["a_b1"]
    a_b2 = stencils["a_b2"]

    if a_i1.ndim != 1 or a_i2.ndim != 1:
        raise ValueError("sbp_stencils produced invalid interior stencil shape")

    r, m = a_b1.shape
    w = (len(a_i1) - 1) // 2

    # Base 1-D operators (no pole closure).
    D1 = sp.lil_matrix((n, n))
    D2 = sp.lil_matrix((n, n))

    # --- Interior points (MATLAB: j = 1+w : n-r )
    for j in range(w, n - r):
        D1[j, j - w : j + w + 1] = a_i1 / dz
        D2[j, j - w : j + w + 1] = a_i2 / dz**2

    # --- Plus boundary (top rows)
    for j in range(n - r, n):
        brow = n - 1 - j
        for col_k in range(m):
            col = n - 1 - col_k
            D1[j, col] = -a_b1[brow, col_k] / dz
            D2[j, col] = a_b2[brow, col_k] / dz**2

    # --- Minus boundary (bottom rows)
    for j in range(r):
        for col_k in range(m):
            D1[j, col_k] = a_b1[j, col_k] / dz
            D2[j, col_k] = a_b2[j, col_k] / dz**2

    # Pole-modified stencils
    D1p = _mohseni_pole_stencil(a_i1, +1)
    D1m = _mohseni_pole_stencil(a_i1, -1)
    D2p = _mohseni_pole_stencil(a_i2, +1)
    D2m = _mohseni_pole_stencil(a_i2, -1)

    # MATLAB builds the pole-closure operators from scratch using
    # interior rows plus the plus-boundary closure only. Importantly, the
    # rows ``w .. r-1`` remain interior rows, not the standard minus-boundary
    # closure used by ``D1``/``D2``.
    D1_sym = sp.lil_matrix((n, n))
    D1_asymm = sp.lil_matrix((n, n))
    D2_sym = sp.lil_matrix((n, n))
    D2_asymm = sp.lil_matrix((n, n))

    for j in range(w, n - r):
        D1_sym[j, j - w : j + w + 1] = a_i1 / dz
        D1_asymm[j, j - w : j + w + 1] = a_i1 / dz
        D2_sym[j, j - w : j + w + 1] = a_i2 / dz**2
        D2_asymm[j, j - w : j + w + 1] = a_i2 / dz**2

    for j in range(n - r, n):
        brow = n - 1 - j
        for col_k in range(m):
            col = n - 1 - col_k
            D1_sym[j, col] = -a_b1[brow, col_k] / dz
            D1_asymm[j, col] = -a_b1[brow, col_k] / dz
            D2_sym[j, col] = a_b2[brow, col_k] / dz**2
            D2_asymm[j, col] = a_b2[brow, col_k] / dz**2

    for j in range(w):
        src_sym = D1p[j, w - j : 2 * w + 1]
        src_asymm = D1m[j, w - j : 2 * w + 1]
        n = min(src_sym.size, w + j + 1)
        D1_sym[j, :n] = src_sym[:n] / dz
        D1_asymm[j, :n] = src_asymm[:n] / dz

        src2_sym = D2p[j, w - j : 2 * w + 1]
        src2_asymm = D2m[j, w - j : 2 * w + 1]
        n2 = min(src2_sym.size, w + j + 1)
        D2_sym[j, :n2] = src2_sym[:n2] / dz**2
        D2_asymm[j, :n2] = src2_asymm[:n2] / dz**2

    return (
        D1.tocsr(),
        D2.tocsr(),
        D1_sym.tocsr(),
        D1_asymm.tocsr(),
        D2_sym.tocsr(),
        D2_asymm.tocsr(),
    )


def _mohseni_pole_stencil(a_interior: np.ndarray, h: int) -> np.ndarray:
    """Return the pole-closure stencil block used by Mohseni & Colonius."""
    if h not in (+1, -1):
        raise ValueError(f"Unsupported pole parity h={h}; expected +1 (symmetric) or -1 (antisymmetric).")

    w = (len(a_interior) - 1) // 2
    out = np.zeros((w, 2 * w + 1))

    for j in range(w):
        # Unmodified coefficients near the boundary (outgoing part)
        out[j, 2 * w - 2 * j :] = a_interior[2 * w - 2 * j :]

        # Reflected/mirrored coefficients.
        affected = slice(w - j, 2 * w - 2 * j)
        coeff = a_interior[w - j : 2 * w - 2 * j]
        reflected = a_interior[: w - j][::-1]
        out[j, affected] = coeff + h * reflected

    return out


# Single source of truth for MSN A_{2p}(n, r, s) defaults.
# Layers above (mesh.py, results_runtime.py) MUST reference these names rather
# than re-declaring the literals. See `phase1_dissipation_math.md` §4 for the
# meaning of the three parameters.
DISSIPATION_DEFAULTS: dict[str, float] = {
    "transition_fraction": 1.0 / 20.0,
    "endpoint_exponent": 2.0,
    "interior_value": 1.0,
}


def _build_sbp_norm_diag(n: int, h: float, order_interior: int) -> np.ndarray:
    """Return the H diagonal for the full-norm SBP operator."""
    from linstab2d.mesh.sbp_generator import _sbp_norm_weights

    bw = _sbp_norm_weights(order_interior)
    r = len(bw)
    H_diag = np.ones(n) * h
    H_diag[:r] = bw * h
    H_diag[n - r:] = bw[::-1] * h
    return H_diag


def _vandermonde_centered_stencil(half_w: int, derivative_order: int) -> np.ndarray:
    """Return undivided (2*half_w+1)-point centered FD stencil for d^k/dx^k.

    Solves the Vandermonde system V c = e_k where V[i,j] = (j-half_w)^i / i!
    and e_k has 1 in position k (derivative_order). Returns coefficients c
    such that sum_j c[j] * f(x + (j-half_w)*h) ≈ d^k f / dx^k * h^k
    (undivided; divide by h^k to get the divided-form approximation).
    """
    from math import factorial

    n = 2 * half_w + 1
    V = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            V[i, k] = (k - half_w) ** i / factorial(i)
    e = np.zeros(n)
    e[derivative_order] = 1.0
    return np.linalg.solve(V, e)


def _build_dp_matrix(n: int, h: float, p: int) -> sp.csr_matrix:
    """Build the divided p-th derivative matrix D_p.

    For p=3 and p=4 the boundary closure aliases the same one-sided stencil
    onto the first and last four rows: 4-point for d^3/dx^3 and 5-point for
    d^4/dx^4. This satisfies MSN 2004 Eq. (5) consistency at every boundary
    row by polynomial coincidence. The construction is therefore valid for
    SPSD-via-sandwich purposes (D_p^T B_p D_p is SPSD whenever B_p >= 0), but
    it is NOT an SBP-energy-compatible boundary closure in the strict sense —
    see MSN 2004 reference [10] for properly row-shifted explicit stencils.
    Acceptable here because B_p ~ h^2 near the wall suppresses the
    dissipation contribution there.
    """
    if p == 2:
        _, D2 = build_sbp_matrix(n, h, order=4)
        return D2

    if p == 3:
        half_w = 4
        a_i3_undiv = np.array([
            -7 / 240,
            3 / 10,
            -169 / 120,
            61 / 30,
            0,
            -61 / 30,
            169 / 120,
            -3 / 10,
            7 / 240,
        ])
        a_i3 = a_i3_undiv / h**3
        D3 = sp.lil_matrix((n, n))

        for j in range(half_w, n - half_w):
            for k, ck in enumerate(a_i3):
                D3[j, j - half_w + k] = ck

        fwd = np.array([-1.0, 3.0, -3.0, 1.0]) / h**3
        bwd = -fwd[::-1]
        for j in range(half_w):
            D3[j, 0:4] = fwd
        for j in range(n - half_w, n):
            D3[j, n - 4:n] = bwd

        return D3.tocsr()

    if p == 4:
        half_w = 4
        a_i4_undiv = _vandermonde_centered_stencil(half_w, 4)
        a_i4 = a_i4_undiv / h**4
        D4 = sp.lil_matrix((n, n))
        for j in range(half_w, n - half_w):
            for k, ck in enumerate(a_i4):
                D4[j, j - half_w + k] = ck
        fwd = np.array([1.0, -4.0, 6.0, -4.0, 1.0]) / h**4
        bwd = fwd[::-1].copy()  # even-order: symmetric, no sign flip
        for j in range(half_w):
            D4[j, 0:5] = fwd
        for j in range(n - half_w, n):
            D4[j, n - 5:n] = bwd
        return D4.tocsr()

    raise ValueError(
        f"Unsupported p={p}; only p=2 (fd4), p=3 (fd6), p=4 (fd8) implemented."
    )


def _build_bp_diag(
    n: int,
    h: float,
    transition_fraction: float,
    endpoint_exponent: float,
    interior_value: float,
) -> np.ndarray:
    """Build the MSN A_{2p} diagonal boundary-to-interior profile."""
    k = max(1, round(n * transition_fraction))
    B = np.full(n, interior_value)
    b0 = h**endpoint_exponent

    for j in range(k):
        t = j / (k - 1) if k > 1 else 1.0
        ramp = 3 * t**2 - 2 * t**3
        B[j] = b0 + (interior_value - b0) * ramp
        B[n - 1 - j] = B[j]

    return B


def build_sbp_dissipation_matrices(
    n: int,
    h: float,
    order_interior: int,
    order_diss: int | None = None,
    *,
    transition_fraction: float = DISSIPATION_DEFAULTS["transition_fraction"],
    endpoint_exponent: float = DISSIPATION_DEFAULTS["endpoint_exponent"],
    interior_value: float = DISSIPATION_DEFAULTS["interior_value"],
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return ``(Q_diss, H_norm)`` for 1-D MSN artificial dissipation.

    Constructs the Mattsson-Svard-Nordstrom (2004) dissipation operator

        Q_diss = h^{2p+1} H^{-1} D_p^T B_p D_p                      (divided form)
               = H_tilde^{-1} D_tilde_p^T B_p D_tilde_p             (undivided form)

    where ``p = order_interior // 2``, ``H = h * H_tilde`` is the full-norm
    SBP mass matrix, ``D_p = h^{-p} D_tilde_p`` is the divided p-th derivative
    matrix, and ``B_p`` is the diagonal
    ``A_{2p}(transition_fraction, endpoint_exponent, interior_value)`` profile
    from MSN Def. 4.3.

    The h^{2p+1} prefactor is mandatory for dimensional consistency: with
    H_tilde^{-1} = h * H^{-1}, D_tilde_p = h^p D_p, the substitution gives
    h^{1 + 2p} = h^{2p+1}. (Earlier internal drafts used h^{2p-1}, off by
    h^{-2}; the validation gate caught it.)

    The caller subtracts ``epsilon * Q_diss`` from the operator:

        L_reg = L - epsilon * Q_diss

    ``H * Q_diss = h^{2p+1} D_p^T B_p D_p`` is symmetric positive
    semidefinite whenever ``B_p >= 0``.

    Parameters
    ----------
    n : int
        Number of grid points.
    h : float
        Uniform grid spacing.
    order_interior : int
        Interior accuracy order of the underlying SBP first-derivative
        operator. Currently 4, 6, and 8 are implemented.
    order_diss : int or None
        Order of the dissipation operator (2*p_diss). When ``None`` (the
        default), defaults to ``order_interior`` so the dissipation order
        matches the underlying scheme. Phase 1 currently enforces equality;
        Phase 3 (when MSN 2004 allows decoupling, e.g. 4th-order dissipation
        on a 6th-order scheme) will lift this restriction. Passing a value
        that differs from ``order_interior`` raises ``NotImplementedError``.
    transition_fraction : float
        Fraction of the grid over which ``B_p`` ramps from boundary value to
        interior value. Default is ``1/20``.
    endpoint_exponent : float
        Exponent ``r`` such that the boundary value is ``h^r``.
    interior_value : float
        Interior value of ``B_p``.

    Returns
    -------
    Q_diss : scipy.sparse.csr_matrix, shape (n, n)
        Positive dissipation operator in the SBP norm.
    H_norm : scipy.sparse.csr_matrix, shape (n, n)
        Full-norm SBP diagonal mass matrix.

    References
    ----------
    K. Mattsson, M. Svard, J. Nordstrom. Stable and Accurate Artificial
    Dissipation. J. Sci. Comput. 21 (2004) 57-79.
    """
    if order_interior not in (4, 6, 8):
        raise ValueError(f"order_interior must be 4, 6, or 8, got {order_interior}")

    if order_diss is None:
        order_diss = order_interior
    elif order_diss != order_interior:
        raise NotImplementedError(
            f"Phase 1 enforces order_diss == order_interior; got "
            f"order_diss={order_diss}, order_interior={order_interior}. "
            f"Decoupled orders are a Phase 3 item."
        )

    p = order_interior // 2

    H_diag = _build_sbp_norm_diag(n, h, order_interior)
    H_norm = sp.diags(H_diag, format="csr")
    H_inv = sp.diags(1.0 / H_diag, format="csr")

    B_diag = _build_bp_diag(
        n,
        h,
        transition_fraction,
        endpoint_exponent,
        interior_value,
    )
    Bp = sp.diags(B_diag, format="csr")

    Dp = _build_dp_matrix(n, h, p)
    sandwich = Dp.T @ Bp @ Dp
    sandwich = 0.5 * (sandwich + sandwich.T)
    Q_diss = (h ** (2 * p + 1)) * H_inv @ sandwich

    return Q_diss.tocsr(), H_norm
