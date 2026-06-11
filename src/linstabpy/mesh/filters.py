# SPDX-License-Identifier: GPL-3.0-only
"""Spatial filter builders and mesh filter containers.

This module translates the MATLAB ``GetFilter.m``/``CreateFilter.m`` helpers
for generating implicit 10th-order filters on structured 1D lines and building
the separable 2D operator pair used by the mesh object.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# PETSc/SLEPc are only used inside ``_build_filter_solver`` (the MUMPS LU on
# the implicit 10th-order filter matrix). The init + import are deferred to
# that function so ``import linstabpy.mesh.filters`` is pure-numpy at module
# load — keeps the lazy contract from ``src/linstabpy/__init__.py`` honest
# for pure-math test jobs that build ``Mesh`` instances without PETSc.


@dataclass(frozen=True)
class MeshFilters:
    """Container for direct and adjoint-compliant filter operators."""

    filter: Callable[[np.ndarray], np.ndarray]
    filter_ct: Callable[[np.ndarray], np.ndarray]


def filter_O10_diffMat(
    nx: int,
    alpha_f: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build the 10th-order implicit filter matrices for one 1D segment.

    Returns
    -------
    M : scipy.sparse.csr_matrix, shape (nx, nx)
        Tri-diagonal implicit matrix.
    F : scipy.sparse.csr_matrix, shape (nx, nx)
        Full 10th-order filter matrix.
    """
    if nx < 2:
        raise ValueError("nx must be at least 2 for the 10th-order filter")

    beta_f = 1.0
    gamma_f = alpha_f

    f_a = 0.5 * (1.0 - 2.0 * alpha_f) / 512.0
    f_b = 0.5 * (-5.0 + 10.0 * alpha_f) / 256.0
    f_c = 0.5 * (45.0 - 90.0 * alpha_f) / 512.0
    f_d = 0.5 * (-15.0 + 30.0 * alpha_f) / 64.0
    f_e = 0.5 * (105.0 + 302.0 * alpha_f) / 256.0
    f_f = (193.0 + 126.0 * alpha_f) / 256.0

    f_dw5_b = (-1.0 + 2.0 * alpha_f) / 1024.0
    f_dw5_c = (5.0 - 10.0 * alpha_f) / 512.0
    f_dw5_d = (-45.0 + 90.0 * alpha_f) / 1024.0
    f_dw5_e = (15.0 + 98.0 * alpha_f) / 128.0
    f_dw5_f = (407.0 + 210.0 * alpha_f) / 512.0
    f_dw5_g = (63.0 + 130.0 * alpha_f) / 256.0
    f_dw5_h = (-105.0 + 210.0 * alpha_f) / 512.0
    f_dw5_i = (15.0 - 30.0 * alpha_f) / 128.0
    f_dw5_j = (-45.0 + 90.0 * alpha_f) / 1024.0
    f_dw5_k = (5.0 - 10.0 * alpha_f) / 512.0
    f_dw5_l = (-1.0 + 2.0 * alpha_f) / 1024.0

    f_dw4_c = (1.0 - 2.0 * alpha_f) / 1024.0
    f_dw4_d = (-5.0 + 10.0 * alpha_f) / 512.0
    f_dw4_e = (45.0 + 934.0 * alpha_f) / 1024.0
    f_dw4_f = (113.0 + 30.0 * alpha_f) / 128.0
    f_dw4_g = (105.0 + 302.0 * alpha_f) / 512.0
    f_dw4_h = (-63.0 + 126.0 * alpha_f) / 256.0
    f_dw4_i = (105.0 - 210.0 * alpha_f) / 512.0
    f_dw4_j = (-15.0 + 30.0 * alpha_f) / 128.0
    f_dw4_k = (45.0 - 90.0 * alpha_f) / 1024.0
    f_dw4_l = (-5.0 + 10.0 * alpha_f) / 512.0
    f_dw4_m = (1.0 - 2.0 * alpha_f) / 1024.0

    f_dw3_d = (-1.0 + 2.0 * alpha_f) / 1024.0
    f_dw3_e = (5.0 + 502.0 * alpha_f) / 512.0
    f_dw3_f = (979.0 + 90.0 * alpha_f) / 1024.0
    f_dw3_g = (15.0 + 98.0 * alpha_f) / 128.0
    f_dw3_h = (-105.0 + 210.0 * alpha_f) / 512.0
    f_dw3_i = (63.0 - 126.0 * alpha_f) / 256.0
    f_dw3_j = (-105.0 + 210.0 * alpha_f) / 512.0
    f_dw3_k = (15.0 - 30.0 * alpha_f) / 128.0
    f_dw3_l = (-45.0 + 90.0 * alpha_f) / 1024.0
    f_dw3_m = (5.0 - 10.0 * alpha_f) / 512.0
    f_dw3_n = (-1.0 + 2.0 * alpha_f) / 1024.0

    f_dw2_e = (1.0 + 1022.0 * alpha_f) / 1024.0
    f_dw2_f = (507.0 + 10.0 * alpha_f) / 512.0
    f_dw2_g = (45.0 + 934.0 * alpha_f) / 1024.0
    f_dw2_h = (-15.0 + 30.0 * alpha_f) / 128.0
    f_dw2_i = (105.0 - 210.0 * alpha_f) / 512.0
    f_dw2_j = (-63.0 + 126.0 * alpha_f) / 256.0
    f_dw2_k = (105.0 - 210.0 * alpha_f) / 512.0
    f_dw2_l = (-15.0 + 30.0 * alpha_f) / 128.0
    f_dw2_m = (45.0 - 90.0 * alpha_f) / 1024.0
    f_dw2_n = (-5.0 + 10.0 * alpha_f) / 512.0
    f_dw2_o = (1.0 - 2.0 * alpha_f) / 1024.0

    f_dw1_f = (1023.0 + 1.0 * alpha_f) / 1024.0
    f_dw1_g = (5.0 + 507.0 * alpha_f) / 512.0
    f_dw1_h = (-45.0 + 45.0 * alpha_f) / 1024.0
    f_dw1_i = (15.0 - 15.0 * alpha_f) / 128.0
    f_dw1_j = (-105.0 + 105.0 * alpha_f) / 512.0
    f_dw1_k = (63.0 - 63.0 * alpha_f) / 256.0
    f_dw1_l = (-105.0 + 105.0 * alpha_f) / 512.0
    f_dw1_m = (15.0 - 15.0 * alpha_f) / 128.0
    f_dw1_n = (-45.0 + 45.0 * alpha_f) / 1024.0
    f_dw1_o = (5.0 - 5.0 * alpha_f) / 512.0
    f_dw1_p = (-1.0 + 1.0 * alpha_f) / 1024.0

    row_coeffs = np.array([f_a, f_b, f_c, f_d, f_e, f_f, f_e, f_d, f_c, f_b, f_a])
    offsets = np.arange(-5, 6)
    M = sp.diags(
        [np.full(nx, gamma_f), np.full(nx, beta_f), np.full(nx, gamma_f)],
        offsets=(-1, 0, 1),
        shape=(nx, nx),
        format="lil",
    )
    F = sp.diags(
        [np.full(nx, c) for c in row_coeffs],
        offsets=offsets,
        shape=(nx, nx),
        format="lil",
    )

    # MATLAB boundary closures.
    F[0, 0:11] = np.array(
        [
            f_dw1_f,
            f_dw1_g,
            f_dw1_h,
            f_dw1_i,
            f_dw1_j,
            f_dw1_k,
            f_dw1_l,
            f_dw1_m,
            f_dw1_n,
            f_dw1_o,
            f_dw1_p,
        ],
    )
    F[1, 0:11] = np.array(
        [
            f_dw2_e,
            f_dw2_f,
            f_dw2_g,
            f_dw2_h,
            f_dw2_i,
            f_dw2_j,
            f_dw2_k,
            f_dw2_l,
            f_dw2_m,
            f_dw2_n,
            f_dw2_o,
        ],
    )
    F[2, 0:11] = np.array(
        [
            f_dw3_d,
            f_dw3_e,
            f_dw3_f,
            f_dw3_g,
            f_dw3_h,
            f_dw3_i,
            f_dw3_j,
            f_dw3_k,
            f_dw3_l,
            f_dw3_m,
            f_dw3_n,
        ],
    )
    F[3, 0:11] = np.array(
        [
            f_dw4_c,
            f_dw4_d,
            f_dw4_e,
            f_dw4_f,
            f_dw4_g,
            f_dw4_h,
            f_dw4_i,
            f_dw4_j,
            f_dw4_k,
            f_dw4_l,
            f_dw4_m,
        ],
    )
    F[4, 0:11] = np.array(
        [
            f_dw5_b,
            f_dw5_c,
            f_dw5_d,
            f_dw5_e,
            f_dw5_f,
            f_dw5_g,
            f_dw5_h,
            f_dw5_i,
            f_dw5_j,
            f_dw5_k,
            f_dw5_l,
        ],
    )

    F[nx - 1, -11:] = np.array(
        [
            f_dw1_f,
            f_dw1_g,
            f_dw1_h,
            f_dw1_i,
            f_dw1_j,
            f_dw1_k,
            f_dw1_l,
            f_dw1_m,
            f_dw1_n,
            f_dw1_o,
            f_dw1_p,
        ][::-1],
    )
    F[nx - 2, -11:] = np.array(
        [
            f_dw2_e,
            f_dw2_f,
            f_dw2_g,
            f_dw2_h,
            f_dw2_i,
            f_dw2_j,
            f_dw2_k,
            f_dw2_l,
            f_dw2_m,
            f_dw2_n,
            f_dw2_o,
        ][::-1],
    )
    F[nx - 3, -11:] = np.array(
        [
            f_dw3_d,
            f_dw3_e,
            f_dw3_f,
            f_dw3_g,
            f_dw3_h,
            f_dw3_i,
            f_dw3_j,
            f_dw3_k,
            f_dw3_l,
            f_dw3_m,
            f_dw3_n,
        ][::-1],
    )
    F[nx - 4, -11:] = np.array(
        [
            f_dw4_c,
            f_dw4_d,
            f_dw4_e,
            f_dw4_f,
            f_dw4_g,
            f_dw4_h,
            f_dw4_i,
            f_dw4_j,
            f_dw4_k,
            f_dw4_l,
            f_dw4_m,
        ][::-1],
    )
    F[nx - 5, -11:] = np.array(
        [
            f_dw5_b,
            f_dw5_c,
            f_dw5_d,
            f_dw5_e,
            f_dw5_f,
            f_dw5_g,
            f_dw5_h,
            f_dw5_i,
            f_dw5_j,
            f_dw5_k,
            f_dw5_l,
        ][::-1],
    )

    return M.tocsr(), F.tocsr()


def build_mesh_filters(
    X: np.ndarray,
    Y: np.ndarray,
    used_ind: np.ndarray,
    alpha_filter: float | str = 0.0,
) -> MeshFilters:
    """Create direct and adjoint-compatible filter functions for mesh data.

    Parameters
    ----------
    X, Y : ndarray
        Full mesh coordinates.
    used_ind : ndarray
        Flattened used-point indices (column-major indexing order).
    alpha_filter : float | str
        Filtering strength (``0``/``"none"`` disables filtering).
    """
    if _is_no_filter(alpha_filter):
        return MeshFilters(
            filter=lambda v: np.array(v, copy=True),
            filter_ct=lambda v: np.array(v, copy=True),
        )

    alpha_val = float(alpha_filter)
    if alpha_val < 0:
        raise ValueError("alpha_filter must be non-negative")

    active = np.asarray(used_ind, dtype=int)
    x_vec = np.asarray(X, dtype=float).ravel(order="F")[active]
    y_vec = np.asarray(Y, dtype=float).ravel(order="F")[active]
    n = active.size

    M_x, F_x = _build_2d_filter_pair(x_vec, y_vec, alpha_val)
    M_z, F_z = _build_2d_filter_pair(y_vec, x_vec, alpha_val)

    solve_Mx, solve_Mx_T = _make_filter_ksp(M_x)
    solve_Mz, solve_Mz_T = _make_filter_ksp(M_z)

    Fx_Fz = F_x @ F_z
    Fz_Fx_T = F_z.conj().T @ F_x.conj().T

    def _to_columns(v: np.ndarray) -> tuple[np.ndarray, bool]:
        arr = np.asarray(v)
        if arr.ndim == 1:
            if arr.size == n:
                return arr.reshape((n, 1), order="F"), True
            if arr.size % n == 0 and arr.size != n:
                return arr.reshape((n, -1), order="F"), True
            raise ValueError("Input length must be n or n*k for integer k")
        if arr.ndim == 2:
            if arr.shape[0] == n:
                return arr, False
            if arr.shape[1] == n:
                return arr.T, True
        raise ValueError("Input must be a vector of length n*k or an array with n rows")

    def _from_columns(v2d: np.ndarray, scalar_input: bool) -> np.ndarray:
        return v2d.ravel(order="F") if scalar_input else v2d

    def filter_fn(v: np.ndarray) -> np.ndarray:
        arr, scalar_input = _to_columns(v)
        filtered = solve_Mz(solve_Mx(Fx_Fz @ arr))
        if not np.iscomplexobj(v):
            filtered = filtered.real
        return _from_columns(filtered, scalar_input)

    def filter_ct_fn(v: np.ndarray) -> np.ndarray:
        arr, scalar_input = _to_columns(v)
        adj = Fz_Fx_T @ solve_Mx_T(solve_Mz_T(arr))
        if not np.iscomplexobj(v):
            adj = adj.real
        return _from_columns(adj, scalar_input)

    return MeshFilters(filter=filter_fn, filter_ct=filter_ct_fn)


def _make_filter_ksp(
    M_sp: sp.spmatrix,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Factor a filter matrix using PETSc KSP with MUMPS LU.

    Parameters
    ----------
    M_sp : sparse matrix
        Tridiagonal implicit filter matrix (real-valued).

    Returns
    -------
    solve : callable
        Forward solve ``M x = rhs``, handles 2D arrays column-wise.
    solve_transpose : callable
        Transpose solve ``M^T x = rhs``, handles 2D arrays column-wise.
    """
    # Lazy PETSc init: keep ``import linstabpy.mesh.filters`` pure-numpy at
    # module load. PETSc is only needed when a filter solver is actually
    # constructed (e.g. inside Mesh.create with filters enabled).
    from ..utils.petsc_init import init_petsc_from_argv

    init_petsc_from_argv()
    from petsc4py import PETSc

    from ..utils.petsc_core import require_complex

    require_complex()
    # Filter application currently works on gathered full-state vectors
    # inside Python callbacks (e.g. resolvent MatShell multiplies). The
    # filter factorization therefore has to be a local sequential PETSc
    # matrix on every rank; a distributed AIJ here would expect only the
    # owned vector slice and fails on the gathered full vector length.
    csr = M_sp.tocsr().astype(complex)
    A = PETSc.Mat().createAIJ(
        size=csr.shape,
        csr=(csr.indptr, csr.indices, csr.data),
        comm=PETSc.COMM_SELF,
    )
    A.assemble()
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    from ..linalg.backends import apply_factor_backend

    apply_factor_backend(pc, "mumps")
    ksp.setUp()

    def _solve_columns(rhs: np.ndarray, transpose: bool = False) -> np.ndarray:
        solve_fn = ksp.solveTranspose if transpose else ksp.solve
        out = np.empty_like(rhs, dtype=complex)
        for j in range(rhs.shape[1]):
            b = PETSc.Vec().createWithArray(
                np.ascontiguousarray(rhs[:, j], dtype=complex),
                comm=PETSc.COMM_SELF,
            )
            x = b.duplicate()
            solve_fn(b, x)
            out[:, j] = np.array(x.array_r, copy=True)
            b.destroy()
            x.destroy()
        return out

    def solve(rhs: np.ndarray) -> np.ndarray:
        return _solve_columns(rhs, transpose=False)

    def solve_transpose(rhs: np.ndarray) -> np.ndarray:
        return _solve_columns(rhs, transpose=True)

    return solve, solve_transpose


def _build_2d_filter_pair(
    primary: np.ndarray,
    secondary: np.ndarray,
    alpha_filter: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build 1-D filter blocks and assemble them into a global line-separable pair."""
    primary = np.asarray(primary, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    if primary.size != secondary.size:
        raise ValueError("primary and secondary arrays must be same length")

    n = primary.size
    if n == 0:
        return sp.identity(0, format="csr"), sp.identity(0, format="csr")

    M = sp.identity(n, format="lil")
    F = sp.identity(n, format="lil")

    sort_idx = np.argsort(secondary)
    sec_sorted = secondary[sort_idx]

    if sec_sorted.size > 1:
        mag = np.max(np.abs(sec_sorted))
        tol = 1e-12 * max(1.0, mag)
        breaks = np.where(np.abs(np.diff(sec_sorted)) > tol)[0] + 1
    else:
        breaks = np.array([], dtype=int)

    bounds = np.concatenate(([0], breaks, [n]))
    p_sorted = sort_idx

    for i in range(len(bounds) - 1):
        start = bounds[i]
        end = bounds[i + 1]
        seg = p_sorted[start:end]
        seg_n = seg.size
        if seg_n <= 1:
            continue

        # Keep identity on short segments (n <= 10 points).
        if seg_n <= 10:
            continue

        line_order = np.argsort(primary[seg])
        idx = seg[line_order]
        M_line, F_line = filter_O10_diffMat(seg_n, alpha_filter)
        M[np.ix_(idx, idx)] = M_line
        F[np.ix_(idx, idx)] = F_line

    return M.tocsr(), F.tocsr()


def _is_no_filter(alpha_filter: float | str) -> bool:
    """Whether the requested filter is disabled.

    Only the string ``'none'`` (or empty string) disables filtering.
    Numeric ``0.0`` is a *valid* filter strength (maximum-strength explicit
    10th-order filter, M = I) — this matches the MATLAB convention where
    ``alpha_filter = 0`` produces real filter matrices and the no-filter
    sentinel is the string ``'none'``.
    """
    if isinstance(alpha_filter, str):
        return alpha_filter.lower() == "none" or alpha_filter == ""
    return False
