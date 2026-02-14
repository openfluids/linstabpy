"""Structured mesh with SBP finite-difference operators and integration weights.

Translates the MATLAB routines ``CreateMesh.m``, ``AddDW2mesh.m``, and
``DeformMesh.m`` from LinStab2D.  Phase 1 covers the non-periodic,
non-symmetry case with optional Y-only mesh deformation (sufficient for
Blasius-type problems).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .sbp import build_sbp_matrix, build_sbp_pole_matrices, min_points
from .filters import MeshFilters, build_mesh_filters


@dataclass
class Mesh:
    """Two-dimensional structured mesh with differentiation operators.

    Do not instantiate directly — use the :meth:`create` classmethod.

    Attributes
    ----------
    X : np.ndarray, shape (ny, nx)
        Physical x-coordinates on the 2-D grid.
    Y : np.ndarray, shape (ny, nx)
        Physical y-coordinates on the 2-D grid.
    ngp : int
        Total number of grid points (``nx * ny``).
    used_ind : np.ndarray
        Indices of used grid points in the column-major flattened arrays.
    y_symmetry : bool
        Whether y-symmetry operators are present (Phase 2).
    x_periodicity : bool
        Whether x-periodicity is enforced (Phase 2).
    fd_order : int
        Finite-difference order of the SBP operators.
    Dx : sp.csr_matrix
        First x-derivative operator.
    D2x : sp.csr_matrix
        Second x-derivative operator.
    Dy : sp.csr_matrix
        First y-derivative operator.
    D2y : sp.csr_matrix
        Second y-derivative operator.
    Dy_symm : sp.csr_matrix | None
        Symmetric y-derivative operator for axisymmetric axis handling.
    D2y_symm : sp.csr_matrix | None
        Symmetric second y-derivative operator for axisymmetric axis handling.
    Dy_asymm : sp.csr_matrix | None
        Antisymmetric y-derivative operator for axisymmetric axis handling.
    D2y_asymm : sp.csr_matrix | None
        Antisymmetric second y-derivative operator for axisymmetric axis handling.
    Dxy_symm : sp.csr_matrix | None
        Mixed derivative ``Dx @ Dy_symm``.
    Dyx_symm : sp.csr_matrix | None
        Mixed derivative ``Dy_symm @ Dx``.
    Dxy_asymm : sp.csr_matrix | None
        Mixed derivative ``Dx @ Dy_asymm``.
    Dyx_asymm : sp.csr_matrix | None
        Mixed derivative ``Dy_asymm @ Dx``.
    Dxy : sp.csr_matrix
        Mixed derivative ``Dx @ Dy``.
    Dyx : sp.csr_matrix
        Mixed derivative ``Dy @ Dx``.
    W : np.ndarray
        Integration weights, shape ``(ny, nx)``.
    W_symm : np.ndarray | None
        Symmetric axisymmetric integration weights.
    W_asymm : np.ndarray | None
        Antisymmetric axisymmetric integration weights.
    idx : dict
        Boundary index sets: ``bi`` (bottom), ``ti`` (top), ``li`` (left),
        ``ri`` (right).  Indices refer to the column-major flattened grid.
    sponge : np.ndarray | None
        Sponge-layer damping profile (not used in Phase 1).
    alpha_filter : float | str
        Spatial filter strength used to build ``filters``.
    filters : MeshFilters | None
        Direct and adjoint filter operators.
    """

    X: np.ndarray
    Y: np.ndarray
    ngp: int
    used_ind: np.ndarray
    y_symmetry: bool
    x_periodicity: bool
    fd_order: int
    Dx: sp.csr_matrix
    D2x: sp.csr_matrix
    Dy: sp.csr_matrix
    D2y: sp.csr_matrix
    Dy_symm: sp.csr_matrix | None
    D2y_symm: sp.csr_matrix | None
    Dy_asymm: sp.csr_matrix | None
    D2y_asymm: sp.csr_matrix | None
    Dxy_symm: sp.csr_matrix | None
    Dyx_symm: sp.csr_matrix | None
    Dxy_asymm: sp.csr_matrix | None
    Dyx_asymm: sp.csr_matrix | None
    Dxy: sp.csr_matrix
    Dyx: sp.csr_matrix
    W: np.ndarray
    W_symm: np.ndarray | None
    W_asymm: np.ndarray | None
    idx: dict
    sponge: np.ndarray | None = None
    alpha_filter: float | str = 0.0
    filters: MeshFilters | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        xrange: tuple[float, float],
        yrange: tuple[float, float],
        nx: int,
        ny: int,
        fd_order: int,
        y_symmetry: bool = False,
        x_periodicity: bool = False,
        alpha_filter: float | str = 0.0,
    ) -> Mesh:
        """Build a uniform rectangular mesh with SBP operators.

        Translates ``CreateMesh.m`` + ``AddDW2mesh.m`` for the standard
        (non-periodic, non-symmetry) case.

        Parameters
        ----------
        xrange : (float, float)
            Domain extent in x, ``(x_min, x_max)``.
        yrange : (float, float)
            Domain extent in y, ``(y_min, y_max)``.
        nx, ny : int
            Number of grid points in x and y.
        fd_order : int
            SBP finite-difference order (2, 4, 6, or 8).
        y_symmetry : bool
            Whether y-symmetry operators are built.
        x_periodicity : bool
            Whether the x-direction is treated as periodic.
        alpha_filter : float | str
            Spatial filtering coefficient (`0` or `'none'` disables filtering).

        Returns
        -------
        Mesh
        """
        # --- 1. Uniform grid ------------------------------------------
        if x_periodicity:
            if nx < 3:
                raise ValueError("x_periodicity requires nx >= 3")
            # Keep the right endpoint open so periodic neighbors do not duplicate.
            x = np.linspace(xrange[0], xrange[1], nx + 1)[:-1]
        else:
            x = np.linspace(xrange[0], xrange[1], nx)
        y = np.linspace(yrange[0], yrange[1], ny)
        X, Y = np.meshgrid(x, y)  # shape (ny, nx)

        ny_full, nx_full = X.shape
        ngp = nx_full * ny_full
        used_ind = np.arange(ngp)  # all points used (column-major)

        dx = x[1] - x[0] if nx > 1 else 1.0
        dy = y[1] - y[0] if ny > 1 else 1.0

        # Flatten in column-major (Fortran) order to match MATLAB indexing
        x_flat = X.ravel(order="F")
        y_flat = Y.ravel(order="F")

        # --- 2. Build global derivative matrices ----------------------
        (
            Dx_g,
            D2x_g,
            Dy_g,
            D2y_g,
            Dy_symm_g,
            D2y_symm_g,
            Dy_asymm_g,
            D2y_asymm_g,
            Dxy_g,
            Dyx_g,
            Dxy_symm_g,
            Dyx_symm_g,
            Dxy_asymm_g,
            Dyx_asymm_g,
            W_g,
            W_symm_g,
            W_asymm_g,
        ) = _build_operators(
            x_flat,
            y_flat,
            X,
            Y,
            dx,
            dy,
            nx_full,
            ny_full,
            ngp,
            used_ind,
            fd_order,
            x_periodicity=x_periodicity,
            y_symmetry=y_symmetry,
        )

        filters = build_mesh_filters(X, Y, used_ind, alpha_filter=alpha_filter)

        idx = _build_indices(
            X,
            Y,
            used_ind,
            x_periodicity=x_periodicity,
            y_symmetry=y_symmetry,
            mask=np.ones_like(X, dtype=bool),
        )

        return cls(
            X=X,
            Y=Y,
            ngp=used_ind.size,
            used_ind=used_ind,
            y_symmetry=y_symmetry,
            x_periodicity=x_periodicity,
            fd_order=fd_order,
            Dx=Dx_g,
            D2x=D2x_g,
            Dy=Dy_g,
            D2y=D2y_g,
            Dy_symm=Dy_symm_g,
            D2y_symm=D2y_symm_g,
            Dy_asymm=Dy_asymm_g,
            D2y_asymm=D2y_asymm_g,
            Dxy=Dxy_g,
            Dyx=Dyx_g,
            Dxy_symm=Dxy_symm_g,
            Dyx_symm=Dyx_symm_g,
            Dxy_asymm=Dxy_asymm_g,
            Dyx_asymm=Dyx_asymm_g,
            W=W_g,
            W_symm=W_symm_g,
            W_asymm=W_asymm_g,
            idx=idx,
            alpha_filter=alpha_filter,
            filters=filters,
        )

    # ------------------------------------------------------------------
    # Mesh deformation
    # ------------------------------------------------------------------

    def deform(
        self,
        Y_new: np.ndarray | None = None,
        X_new: np.ndarray | None = None,
    ) -> Mesh:
        """Return a new mesh deformed to the given physical coordinates.

        Phase 2 supports full 2-D deformation via the full Jacobian update.

        Parameters
        ----------
        Y_new : np.ndarray, shape (ny, nx), optional
            New physical y-coordinates.
        X_new : np.ndarray, shape (ny, nx), optional
            New physical x-coordinates.

        Returns
        -------
        Mesh
            A new ``Mesh`` with updated coordinates, operators, and weights.
        """
        if Y_new is None:
            Y_new = self.Y
        if X_new is None:
            X_new = self.X
        if Y_new.shape != self.Y.shape or X_new.shape != self.X.shape:
            raise ValueError("Y_new and X_new must have the same shape as X, Y")

        # Flat arrays in the active index space
        p = self.used_ind
        x = X_new.ravel(order="F")[p]
        y = Y_new.ravel(order="F")[p]

        Dx = self.Dx
        D2x = self.D2x
        Dy = self.Dy
        D2y = self.D2y

        # Jacobian of the mapping from computational to physical space
        J11 = Dx @ x
        J12 = Dy @ x
        J21 = Dx @ y
        J22 = Dy @ y
        detJ = J11 * J22 - J12 * J21
        if np.any(np.abs(detJ) < 1e-15):
            raise ValueError("Degenerate mapping: Jacobian determinant is near zero")

        inv_det = 1.0 / detJ
        Ji11 = J22 * inv_det
        Ji12 = -J12 * inv_det
        Ji21 = -J21 * inv_det
        Ji22 = J11 * inv_det

        # Optional derivative blocks for y-symmetric operators
        Dy_symm = self.Dy_symm
        D2y_symm = self.D2y_symm
        Dy_asymm = self.Dy_asymm
        D2y_asymm = self.D2y_asymm

        Dx_new = _axm_diag(Ji11, Dx) + _axm_diag(Ji12, Dy)
        Dy_new = _axm_diag(Ji21, Dx) + _axm_diag(Ji22, Dy)

        Dxy = Dx @ Dy
        Dyx = Dy @ Dx
        D2x_new, Dxy_new, Dyx_new, D2y_new = _transform_second_derivatives(
            Ji11,
            Ji12,
            Ji21,
            Ji22,
            D2x,
            D2y,
            Dxy,
            Dyx,
            Dx,
            Dy,
            x,
            y,
        )

        Dxy_symm_new = None
        Dyx_symm_new = None
        Dxy_asymm_new = None
        Dyx_asymm_new = None
        D2y_symm_new = None
        D2y_asymm_new = None
        if self.y_symmetry and Dy_symm is not None and D2y_symm is not None:
            Dy_symm_new = _axm_diag(Ji21, Dx) + _axm_diag(Ji22, Dy_symm)
            Dy_asymm_new = _axm_diag(Ji21, Dx) + _axm_diag(Ji22, Dy_asymm)

            _, Dxy_symm_new, Dyx_symm_new, D2y_symm_new = _transform_second_derivatives(
                Ji11,
                Ji12,
                Ji21,
                Ji22,
                D2x,
                D2y_symm,
                self.Dxy_symm,
                self.Dyx_symm,
                Dx,
                Dy_symm,
                x,
                y,
            )
            _, Dxy_asymm_new, Dyx_asymm_new, D2y_asymm_new = _transform_second_derivatives(
                Ji11,
                Ji12,
                Ji21,
                Ji22,
                D2x,
                D2y_asymm,
                self.Dxy_asymm,
                self.Dyx_asymm,
                Dx,
                Dy_asymm,
                x,
                y,
            )

        # Weights multiply by |detJ| for mapped coordinates.
        W_new = self.W.copy()
        w_flat = W_new.ravel(order="F")
        w_flat[p] *= np.abs(detJ)
        W_new = w_flat.reshape(W_new.shape, order="F")

        W_symm_new = None
        W_asymm_new = None
        if self.W_symm is not None:
            W_symm_new = self.W_symm.copy()
            w_flat = W_symm_new.ravel(order="F")
            w_flat[p] *= np.abs(detJ)
            W_symm_new = w_flat.reshape(W_symm_new.shape, order="F")

        if self.W_asymm is not None:
            W_asymm_new = self.W_asymm.copy()
            w_flat = W_asymm_new.ravel(order="F")
            w_flat[p] *= np.abs(detJ)
            W_asymm_new = w_flat.reshape(W_asymm_new.shape, order="F")

        idx_new = _build_indices(
            X_new,
            Y_new,
            self.used_ind,
            x_periodicity=self.x_periodicity,
            y_symmetry=self.y_symmetry,
            mask=np.ones_like(self.X, dtype=bool),
        )
        filters_new = build_mesh_filters(X_new, Y_new, self.used_ind, self.alpha_filter)

        return Mesh(
            X=X_new.copy(),
            Y=Y_new.copy(),
            ngp=self.used_ind.size,
            used_ind=self.used_ind.copy(),
            y_symmetry=self.y_symmetry,
            x_periodicity=self.x_periodicity,
            fd_order=self.fd_order,
            Dx=Dx_new,
            D2x=D2x_new,
            Dy=Dy_new,
            D2y=D2y_new,
            Dy_symm=Dy_symm_new if self.y_symmetry else None,
            D2y_symm=D2y_symm_new if self.y_symmetry else None,
            Dy_asymm=Dy_asymm_new if self.y_symmetry else None,
            D2y_asymm=D2y_asymm_new if self.y_symmetry else None,
            Dxy=Dxy_new,
            Dyx=Dyx_new,
            Dxy_symm=Dxy_symm_new if self.y_symmetry else None,
            Dyx_symm=Dyx_symm_new if self.y_symmetry else None,
            Dxy_asymm=Dxy_asymm_new if self.y_symmetry else None,
            Dyx_asymm=Dyx_asymm_new if self.y_symmetry else None,
            W=W_new,
            W_symm=W_symm_new,
            W_asymm=W_asymm_new,
            idx=idx_new,
            alpha_filter=self.alpha_filter,
            filters=filters_new,
        )

    def mask(self, mask_array: np.ndarray) -> Mesh:
        """Return a new mesh masked by a boolean keep-mask.

        Parameters
        ----------
        mask_array : ndarray, shape like ``X`` and ``Y``
            ``True`` entries are kept; ``False`` entries are masked out.
        """
        if mask_array.shape != self.X.shape:
            raise ValueError("mask_array must match X and Y shape")

        used_mask = np.asarray(mask_array, dtype=bool)
        used_ind = np.flatnonzero(used_mask.ravel(order="F"))
        if used_ind.size == 0:
            raise ValueError("mask_array masks out all points")

        x_flat = self.X.ravel(order="F")
        y_flat = self.Y.ravel(order="F")
        dx = x_flat[1] - x_flat[0] if x_flat.size > 1 else 1.0
        dy = y_flat[1] - y_flat[0] if y_flat.size > 1 else 1.0

        (
            Dx_g,
            D2x_g,
            Dy_g,
            D2y_g,
            Dy_symm_g,
            D2y_symm_g,
            Dy_asymm_g,
            D2y_asymm_g,
            Dxy_g,
            Dyx_g,
            Dxy_symm_g,
            Dyx_symm_g,
            Dxy_asymm_g,
            Dyx_asymm_g,
            W_g,
            W_symm_g,
            W_asymm_g,
        ) = _build_operators(
            x_flat,
            y_flat,
            self.X,
            self.Y,
            dx,
            dy,
            self.X.shape[1],
            self.X.shape[0],
            used_ind.size,
            used_ind,
            self.fd_order,
            x_periodicity=self.x_periodicity,
            y_symmetry=self.y_symmetry,
            active_mask=used_mask,
        )

        idx = _build_indices(
            self.X,
            self.Y,
            used_ind,
            x_periodicity=self.x_periodicity,
            y_symmetry=self.y_symmetry,
            mask=used_mask,
        )
        filters = build_mesh_filters(self.X, self.Y, used_ind, self.alpha_filter)

        # Remap symmetric axis weights if present and available
        W_symm_masked = None
        W_asymm_masked = None
        if W_symm_g is not None:
            W_symm_masked = W_symm_g
        if W_asymm_g is not None:
            W_asymm_masked = W_asymm_g

        return Mesh(
            X=self.X.copy(),
            Y=self.Y.copy(),
            ngp=used_ind.size,
            used_ind=used_ind,
            y_symmetry=self.y_symmetry,
            x_periodicity=self.x_periodicity,
            fd_order=self.fd_order,
            Dx=Dx_g,
            D2x=D2x_g,
            Dy=Dy_g,
            D2y=D2y_g,
            Dy_symm=Dy_symm_g,
            D2y_symm=D2y_symm_g,
            Dy_asymm=Dy_asymm_g,
            D2y_asymm=D2y_asymm_g,
            Dxy=Dxy_g,
            Dyx=Dyx_g,
            Dxy_symm=Dxy_symm_g,
            Dyx_symm=Dyx_symm_g,
            Dxy_asymm=Dxy_asymm_g,
            Dyx_asymm=Dyx_asymm_g,
            W=W_g,
            W_symm=W_symm_masked,
            W_asymm=W_asymm_masked,
            idx=idx,
            alpha_filter=self.alpha_filter,
            filters=filters,
        )


# ======================================================================
# Private helpers
# ======================================================================


def _build_operators(
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    ngp: int,
    used_ind: np.ndarray,
    fd_order: int,
    x_periodicity: bool = False,
    y_symmetry: bool = False,
    active_mask: np.ndarray | None = None,
) -> tuple[
    sp.csr_matrix,
    sp.csr_matrix,
    sp.csr_matrix,
    sp.csr_matrix,
    sp.csr_matrix | None,
    sp.csr_matrix | None,
    sp.csr_matrix | None,
    sp.csr_matrix | None,
    sp.csr_matrix,
    sp.csr_matrix,
    sp.csr_matrix | None,
    sp.csr_matrix | None,
    sp.csr_matrix | None,
    sp.csr_matrix | None,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Build derivative matrices and quadrature weights.

    Supports:
    - optional x-periodic stencils
    - optional y-symmetry support (pole-condition closures)
    - optional masked domains (for ``Mesh.mask``)
    """
    full_n = X.size
    if active_mask is None:
        active_mask_1d = np.zeros(full_n, dtype=bool)
        active_mask_1d[used_ind] = True
    else:
        if np.asarray(active_mask).shape != X.shape:
            raise ValueError("active_mask must match X and Y shape")
        active_mask_1d = np.asarray(active_mask, dtype=bool).ravel(order="F")

    if active_mask_1d.sum() != ngp:
        raise ValueError("used_ind must match active_mask size")

    if used_ind.size != ngp:
        raise ValueError("used_ind must be a length-`ngp` index array")
    if np.any(used_ind < 0) or np.any(used_ind >= full_n):
        raise ValueError("used_ind entries must index flattened mesh points")
    if np.unique(used_ind).size != used_ind.size:
        raise ValueError("used_ind must not contain duplicates")

    global_to_local = np.full(full_n, -1, dtype=int)
    global_to_local[used_ind] = np.arange(ngp)

    w_full = np.zeros(full_n)
    w_full[used_ind] = 1.0

    Dx = sp.lil_matrix((ngp, ngp))
    D2x = sp.lil_matrix((ngp, ngp))
    Dy = sp.lil_matrix((ngp, ngp))
    D2y = sp.lil_matrix((ngp, ngp))
    if y_symmetry:
        Dy_symm = sp.lil_matrix((ngp, ngp))
        D2y_symm = sp.lil_matrix((ngp, ngp))
        Dy_asymm = sp.lil_matrix((ngp, ngp))
        D2y_asymm = sp.lil_matrix((ngp, ngp))
    else:
        Dy_symm = None
        D2y_symm = None
        Dy_asymm = None
        D2y_asymm = None

    periodic_period = None
    if x_periodicity and np.any(active_mask_1d):
        periodic_period = _build_periodic_1d_operators(nx, dx, fd_order)

    # --- X derivative matrices (loop over horizontal lines) -----------
    for j in range(ny):
        row_start = j * nx
        row_slice = np.arange(row_start, row_start + nx)
        active_row = row_slice[active_mask_1d[row_slice]]

        if active_row.size == 0:
            continue

        if x_periodicity and active_row.size == nx:
            row_local = global_to_local[row_slice]
            row_local = row_local[row_local >= 0]
            if row_local.size == 0:
                continue

            D1_1d, D2_1d = periodic_period
            Dx[np.ix_(row_local, row_local)] = D1_1d
            D2x[np.ix_(row_local, row_local)] = D2_1d

            w_full[row_slice] *= dx
            continue

        # Non-periodic: split by gaps and build each segment independently.
        order = np.argsort(x_flat[active_row])
        dom = active_row[order]
        x_line = x_flat[dom]
        dx_line = np.diff(x_line) / dx
        breaks = np.where(dx_line > 1.5)[0]
        dom_bounds = np.concatenate(([0], breaks + 1, [dom.size]))

        for i_seg in range(len(dom_bounds) - 1):
            seg = dom[dom_bounds[i_seg] : dom_bounds[i_seg + 1]]
            n_seg = seg.size
            if n_seg < 2:
                continue

            fd_ord = fd_order
            if n_seg < min_points(fd_ord):
                fd_ord = 2

            D1_1d, D2_1d = build_sbp_matrix(n_seg, dx, fd_ord)

            seg_local = global_to_local[seg]
            seg_local = seg_local[seg_local >= 0]
            if seg_local.size < 2:
                continue
            idx = np.ix_(seg_local, seg_local)
            Dx[idx] = D1_1d
            D2x[idx] = D2_1d

            w_full[seg] *= dx
            w_full[seg[0]] *= 0.5
            w_full[seg[-1]] *= 0.5

    # --- Y derivative matrices (loop over vertical lines) -------------
    for i in range(nx):
        col = np.arange(i, full_n, nx)
        active_col = col[active_mask_1d[col]]
        if active_col.size == 0:
            continue

        order = np.argsort(y_flat[active_col])
        dom = active_col[order]
        y_line = y_flat[dom]
        dy_line = np.diff(y_line) / dy
        breaks = np.where(dy_line > 1.5)[0]
        dom_bounds = np.concatenate(([0], breaks + 1, [dom.size]))

        for i_seg in range(len(dom_bounds) - 1):
            seg = dom[dom_bounds[i_seg] : dom_bounds[i_seg + 1]]
            n_seg = seg.size
            if n_seg < 2:
                continue

            fd_ord = fd_order
            if n_seg < min_points(fd_ord):
                fd_ord = 2

            D1_base, D2_base = build_sbp_matrix(n_seg, dy, fd_ord)
            D1_sym = None
            D1_asym = None
            D2_sym = None
            D2_asym = None

            if y_symmetry and np.isclose(y_line[dom_bounds[i_seg]], 0.0, atol=1e-12):
                D1_sym, D1_asym, D2_sym, D2_asym = build_sbp_pole_matrices(
                    n_seg,
                    dy,
                    fd_ord,
                )

            seg_local = global_to_local[seg]
            seg_local = seg_local[seg_local >= 0]
            if seg_local.size < 2:
                continue

            idx = np.ix_(seg_local, seg_local)
            Dy[idx] = D1_base
            D2y[idx] = D2_base
            if y_symmetry:
                Dy_symm[idx] = D1_sym if D1_sym is not None else D1_base
                D2y_symm[idx] = D2_sym if D2_sym is not None else D2_base
                Dy_asymm[idx] = D1_asym if D1_asym is not None else D1_base
                D2y_asymm[idx] = D2_asym if D2_asym is not None else D2_base

            w_full[seg] *= dy
            w_full[seg[0]] *= 0.5
            w_full[seg[-1]] *= 0.5

    Dx_g = Dx.tocsr()
    D2x_g = D2x.tocsr()
    Dy_g = Dy.tocsr()
    D2y_g = D2y.tocsr()
    Dxy_g = Dx_g @ Dy_g
    Dyx_g = Dy_g @ Dx_g

    Dy_symm_g = None
    D2y_symm_g = None
    Dy_asymm_g = None
    D2y_asymm_g = None
    Dxy_symm_g = None
    Dyx_symm_g = None
    Dxy_asymm_g = None
    Dyx_asymm_g = None
    if y_symmetry:
        Dy_symm_g = Dy_symm.tocsr() if Dy_symm is not None else None
        D2y_symm_g = D2y_symm.tocsr() if D2y_symm is not None else None
        Dy_asymm_g = Dy_asymm.tocsr() if Dy_asymm is not None else None
        D2y_asymm_g = D2y_asymm.tocsr() if D2y_asymm is not None else None
        Dxy_symm_g = Dx_g @ Dy_symm_g
        Dyx_symm_g = Dy_symm_g @ Dx_g
        Dxy_asymm_g = Dx_g @ Dy_asymm_g
        Dyx_asymm_g = Dy_asymm_g @ Dx_g

    W_g = w_full.reshape((ny, nx), order="F")

    W_symm_g = None
    W_asymm_g = None
    if y_symmetry:
        W_symm_g = W_g.copy()
        W_asymm_g = W_g.copy()
        axis_nodes = np.isclose(Y, 0.0, atol=1e-12)
        W_asymm_g[axis_nodes] *= 0.75

    return (
        Dx_g,
        D2x_g,
        Dy_g,
        D2y_g,
        Dy_symm_g,
        D2y_symm_g,
        Dy_asymm_g,
        D2y_asymm_g,
        Dxy_g,
        Dyx_g,
        Dxy_symm_g,
        Dyx_symm_g,
        Dxy_asymm_g,
        Dyx_asymm_g,
        W_g,
        W_symm_g,
        W_asymm_g,
    )


def _build_indices(
    X: np.ndarray,
    Y: np.ndarray,
    used_ind: np.ndarray,
    x_periodicity: bool,
    y_symmetry: bool,
    mask: np.ndarray,
) -> dict:
    """Build reduced index sets for boundaries and mask edges."""
    del y_symmetry
    if np.asarray(mask).shape != X.shape:
        raise ValueError("mask must match X and Y shape")

    mask_2d = np.asarray(mask, dtype=bool)
    full_n = X.size
    global_to_local = np.full(full_n, -1, dtype=int)
    global_to_local[used_ind] = np.arange(used_ind.size)

    def _to_local(ind_mask: np.ndarray) -> np.ndarray:
        local = global_to_local[np.flatnonzero(ind_mask)]
        local = local[local >= 0]
        return np.asarray(local, dtype=int)

    bi = _to_local(mask_2d[0, :])
    ti = _to_local(mask_2d[-1, :])

    if x_periodicity:
        li = np.array([], dtype=int)
        ri = np.array([], dtype=int)
    else:
        li = _to_local(mask_2d[:, 0])
        ri = _to_local(mask_2d[:, -1])

    neigh = np.zeros_like(mask_2d, dtype=int)
    neigh[1:, :] += mask_2d[:-1, :].astype(int)
    neigh[:-1, :] += mask_2d[1:, :].astype(int)

    if x_periodicity:
        neigh[:, 0] += mask_2d[:, -1].astype(int)
        neigh[:, -1] += mask_2d[:, 0].astype(int)
        neigh[:, 1:] += mask_2d[:, :-1].astype(int)
        neigh[:, :-1] += mask_2d[:, 1:].astype(int)
    else:
        neigh[:, 1:] += mask_2d[:, :-1].astype(int)
        neigh[:, :-1] += mask_2d[:, 1:].astype(int)

    mi_mask = mask_2d & (neigh < 4)
    mi = _to_local(mi_mask)

    return {"bi": bi, "ti": ti, "li": li, "ri": ri, "mi": mi}


def _build_periodic_1d_operators(
    n: int,
    h: float,
    fd_order: int,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return periodic 1-D SBP-like first/second derivative matrices."""
    if n < 2:
        raise ValueError("periodic 1-D operators require at least 2 points")

    fd_ord = fd_order
    if n < min_points(fd_ord):
        fd_ord = 2

    ghost = fd_order
    D1_ext, D2_ext = build_sbp_matrix(n + 2 * ghost, h, fd_ord)

    ext_indices = np.arange(-ghost, n + ghost) % n
    B = sp.csr_matrix(
        (
            np.ones_like(ext_indices, dtype=float),
            (np.arange(n + 2 * ghost), ext_indices),
        ),
        shape=(n + 2 * ghost, n),
    )
    R = sp.lil_matrix((n, n + 2 * ghost))
    R[np.arange(n), np.arange(ghost, ghost + n)] = 1.0
    R = R.tocsr()

    D1 = (R @ D1_ext @ B).tocsr()
    D2 = (R @ D2_ext @ B).tocsr()
    return D1, D2


def _axm_diag(a: np.ndarray, m: sp.spmatrix) -> sp.csr_matrix:
    """Apply a diagonal vector on the left of a sparse matrix."""
    return sp.diags(a, offsets=0, shape=(m.shape[0], m.shape[1]), format="csr") @ m


def _transform_second_derivatives(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    D2x: sp.spmatrix,
    D2y: sp.spmatrix,
    Dxy: sp.spmatrix,
    Dyx: sp.spmatrix,
    Dx: sp.spmatrix,
    Dy: sp.spmatrix,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Build transformed second derivatives for a full Jacobian map."""
    # Derivatives of J = [[dx/dξ, dy/dξ], [dx/dη, dy/dη]] in computational space.
    D2x_x = D2x @ x
    D2x_y = D2x @ y
    Dxy_x = Dx @ (Dy @ x)
    Dxy_y = Dx @ (Dy @ y)

    Dyx_x = Dy @ x
    Dyx_y = Dy @ y
    D2y_x = D2y @ x
    D2y_y = D2y @ y

    dJ11_xi = D2x_x
    dJ12_xi = D2x_y
    dJ21_xi = Dxy_x
    dJ22_xi = Dxy_y

    dJ11_eta = Dyx_x
    dJ12_eta = Dyx_y
    dJ21_eta = D2y_x
    dJ22_eta = D2y_y

    # dJi = -Ji * dJ/dσ * Ji
    p11_xi = (dJ11_xi * a + dJ12_xi * c)
    p12_xi = (dJ11_xi * b + dJ12_xi * d)
    p21_xi = (dJ21_xi * a + dJ22_xi * c)
    p22_xi = (dJ21_xi * b + dJ22_xi * d)
    p11_eta = (dJ11_eta * a + dJ12_eta * c)
    p12_eta = (dJ11_eta * b + dJ12_eta * d)
    p21_eta = (dJ21_eta * a + dJ22_eta * c)
    p22_eta = (dJ21_eta * b + dJ22_eta * d)

    dJi11_xi = -(a * p11_xi + b * p21_xi)
    dJi12_xi = -(a * p12_xi + b * p22_xi)
    dJi21_xi = -(c * p11_xi + d * p21_xi)
    dJi22_xi = -(c * p12_xi + d * p22_xi)

    dJi11_eta = -(a * p11_eta + b * p21_eta)
    dJi12_eta = -(a * p12_eta + b * p22_eta)
    dJi21_eta = -(c * p11_eta + d * p21_eta)
    dJi22_eta = -(c * p12_eta + d * p22_eta)

    # Q tensors contracted against Dx/Dy.
    q11_x = a * dJi11_xi + b * dJi11_eta
    q11_y = a * dJi12_xi + b * dJi12_eta
    q12_x = c * dJi11_xi + d * dJi11_eta
    q12_y = c * dJi12_xi + d * dJi12_eta
    q21_x = a * dJi21_xi + b * dJi21_eta
    q21_y = a * dJi22_xi + b * dJi22_eta
    q22_x = c * dJi21_xi + d * dJi21_eta
    q22_y = c * dJi22_xi + d * dJi22_eta

    D2x_new = _transform_second_block(
        a * a,
        a * b,
        b * a,
        b * b,
        q11_x,
        q11_y,
        D2x,
        D2y,
        Dxy,
        Dyx,
        Dx,
        Dy,
    )

    Dxy_new = _transform_second_block(
        a * c,
        c * b,
        d * a,
        d * b,
        q12_x,
        q12_y,
        D2x,
        D2y,
        Dxy,
        Dyx,
        Dx,
        Dy,
    )

    Dyx_new = _transform_second_block(
        c * a,
        c * d,
        d * a,
        d * d,
        q21_x,
        q21_y,
        D2x,
        D2y,
        Dxy,
        Dyx,
        Dx,
        Dy,
    )

    D2y_new = _transform_second_block(
        c * c,
        c * d,
        d * c,
        d * d,
        q22_x,
        q22_y,
        D2x,
        D2y,
        Dxy,
        Dyx,
        Dx,
        Dy,
    )

    return D2x_new, Dxy_new, Dyx_new, D2y_new


def _transform_second_block(
    c11: np.ndarray,
    c12: np.ndarray,
    c21: np.ndarray,
    c22: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    D2x: sp.spmatrix,
    D2y: sp.spmatrix,
    Dxy: sp.spmatrix,
    Dyx: sp.spmatrix,
    Dx: sp.spmatrix,
    Dy: sp.spmatrix,
) -> sp.csr_matrix:
    return (
        _axm_diag(c11, D2x)
        + _axm_diag(c12, Dxy)
        + _axm_diag(c21, Dyx)
        + _axm_diag(c22, D2y)
        + _axm_diag(qx, Dx)
        + _axm_diag(qy, Dy)
    ).tocsr()
