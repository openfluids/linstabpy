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
    J11, J12, J21, J22 : np.ndarray | None
        Computational-to-physical Jacobian entries on the used grid points.
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
    alpha_filter: float | str = "none"
    filters: MeshFilters | None = None
    J11: np.ndarray | None = None  # dX_phys/dxi_comp, shape (ngp,)
    J12: np.ndarray | None = None  # dY_phys/dxi_comp, shape (ngp,)
    J21: np.ndarray | None = None  # dX_phys/deta_comp, shape (ngp,)
    J22: np.ndarray | None = None  # dY_phys/deta_comp, shape (ngp,)

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
        alpha_filter: float | str = "none",
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
        y_min, y_max = yrange
        if y_symmetry and np.isclose(y_min, 0.0) and not np.isclose(y_max, 0.0):
            # MATLAB CreateMesh does not simply overwrite the first point with
            # ``dy/2``. It shortens the effective interval using
            # ``dy = y_max / (ny - 0.5)`` and then builds the full linspace
            # from ``dy/2`` to ``y_max``.
            dy = y_max / (ny - 0.5)
            y_min = y_min + dy / 2.0
        y = np.linspace(y_min, y_max, ny)
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
            J11=np.ones(used_ind.size),
            J12=np.zeros(used_ind.size),
            J21=np.zeros(used_ind.size),
            J22=np.ones(used_ind.size),
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

        Matches MATLAB ``DeformMesh.m`` exactly with three branches:

        - **X-only** (``X_new`` given, ``Y_new`` is None): 1-D chain rule
          on the x-derivative operators only.
        - **Y-only** (``Y_new`` given, ``X_new`` is None): 1-D chain rule
          on the y-derivative operators only.
        - **Full 2-D** (both given): full Jacobian transformation of all
          derivative operators.

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
        if Y_new is None and X_new is None:
            return self

        x_only = X_new is not None and Y_new is None
        y_only = Y_new is not None and X_new is None

        if X_new is not None and X_new.shape != self.X.shape:
            raise ValueError("X_new must have the same shape as X")
        if Y_new is not None and Y_new.shape != self.Y.shape:
            raise ValueError("Y_new must have the same shape as Y")

        X_out = X_new if X_new is not None else self.X
        Y_out = Y_new if Y_new is not None else self.Y

        p = self.used_ind
        Dx = self.Dx
        D2x = self.D2x
        Dy = self.Dy
        D2y = self.D2y

        if x_only:
            Dx_new, D2x_new, Dy_new, D2y_new, Dxy_new, Dyx_new, det, \
                Dy_symm_new, D2y_symm_new, Dy_asymm_new, D2y_asymm_new, \
                Dxy_symm_new, Dyx_symm_new, Dxy_asymm_new, Dyx_asymm_new, \
                J11_new, J12_new, J21_new, J22_new = \
                self._deform_x_only(X_new, p)
        elif y_only:
            Dx_new, D2x_new, Dy_new, D2y_new, Dxy_new, Dyx_new, det, \
                Dy_symm_new, D2y_symm_new, Dy_asymm_new, D2y_asymm_new, \
                Dxy_symm_new, Dyx_symm_new, Dxy_asymm_new, Dyx_asymm_new, \
                J11_new, J12_new, J21_new, J22_new = \
                self._deform_y_only(Y_new, p)
        else:
            Dx_new, D2x_new, Dy_new, D2y_new, Dxy_new, Dyx_new, det, \
                Dy_symm_new, D2y_symm_new, Dy_asymm_new, D2y_asymm_new, \
                Dxy_symm_new, Dyx_symm_new, Dxy_asymm_new, Dyx_asymm_new, \
                J11_new, J12_new, J21_new, J22_new = \
                self._deform_full_2d(X_new, Y_new, p)
            if self.y_symmetry and _is_separable_axis_deformation(X_new, Y_new):
                x_mesh = self.deform(X_new=X_new)
                _, _, _, _, _, _, _, \
                    Dy_symm_seq, D2y_symm_seq, Dy_asymm_seq, D2y_asymm_seq, \
                    Dxy_symm_seq, Dyx_symm_seq, Dxy_asymm_seq, Dyx_asymm_seq, \
                    _, _, _, _ = x_mesh._deform_y_only(Y_new, p)
                Dy_symm_new = Dy_symm_seq
                D2y_symm_new = D2y_symm_seq
                Dxy_symm_new = Dxy_symm_seq
                Dyx_symm_new = Dyx_symm_seq
                Dy_asymm_new = Dy_asymm_seq
                D2y_asymm_new = D2y_asymm_seq
                Dxy_asymm_new = Dxy_asymm_seq
                Dyx_asymm_new = Dyx_asymm_seq

        # Update integration weights: W_new = W * |det|
        W_new = self.W.copy()
        w_flat = W_new.ravel(order="F")
        w_flat[p] *= np.abs(det)
        W_new = w_flat.reshape(W_new.shape, order="F")

        W_symm_new = None
        W_asymm_new = None
        if self.W_symm is not None:
            W_symm_new = self.W_symm.copy()
            w_flat = W_symm_new.ravel(order="F")
            w_flat[p] *= np.abs(det)
            W_symm_new = w_flat.reshape(W_symm_new.shape, order="F")

        if self.W_asymm is not None:
            W_asymm_new = self.W_asymm.copy()
            w_flat = W_asymm_new.ravel(order="F")
            w_flat[p] *= np.abs(det)
            W_asymm_new = w_flat.reshape(W_asymm_new.shape, order="F")

        # Reconstruct the effective mask from used_ind so that internal
        # mask boundaries (e.g. cavity walls) are preserved after deformation.
        # On unmasked meshes used_ind = arange(ngp), so this is all-True.
        # Note: build flat then reshape because ravel("F") on C-order is a copy.
        eff_flat = np.zeros(self.X.size, dtype=bool)
        eff_flat[self.used_ind] = True
        effective_mask = eff_flat.reshape(self.X.shape, order="F")

        idx_new = _build_indices(
            X_out,
            Y_out,
            self.used_ind,
            x_periodicity=self.x_periodicity,
            y_symmetry=self.y_symmetry,
            mask=effective_mask,
        )
        # MATLAB parity behavior:
        # CreateMesh builds filter operators on the computational mesh, and
        # DeformMesh updates derivatives/weights only (filters are preserved).
        # Keep the existing filter operators instead of rebuilding them on
        # deformed coordinates.
        filters_new = self.filters

        return Mesh(
            X=X_out.copy(),
            Y=Y_out.copy(),
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
            J11=J11_new,
            J12=J12_new,
            J21=J21_new,
            J22=J22_new,
        )

    def _deform_x_only(self, X_new, p):
        """1-D chain-rule deformation in x (MATLAB DeformMesh branch 1)."""
        x = X_new.ravel(order="F")[p]
        Dx = self.Dx
        D2x = self.D2x
        Dy = self.Dy

        x2p = Dx @ x          # dx_new / dx_comp
        x2pp = D2x @ x        # d²x_new / dx_comp²

        inv_x2p = 1.0 / x2p
        Dx_new = _axm_diag(inv_x2p, Dx)
        D2x_new = _axm_diag(inv_x2p ** 2, D2x) - _axm_diag(x2pp * inv_x2p ** 3, Dx)
        Dy_new = Dy
        D2y_new = self.D2y
        Dxy_new = Dx_new @ Dy
        Dyx_new = Dy @ Dx_new
        det = x2p

        # y-symmetry variants: Dy unchanged, mixed derivatives use Dx_new
        Dy_symm_new = self.Dy_symm
        D2y_symm_new = self.D2y_symm
        Dy_asymm_new = self.Dy_asymm
        D2y_asymm_new = self.D2y_asymm
        Dxy_symm_new = None
        Dyx_symm_new = None
        Dxy_asymm_new = None
        Dyx_asymm_new = None

        if self.y_symmetry and self.Dy_symm is not None:
            Dxy_symm_new = Dx_new @ self.Dy_symm
            Dyx_symm_new = self.Dy_symm @ Dx_new
            Dxy_asymm_new = Dx_new @ self.Dy_asymm
            Dyx_asymm_new = self.Dy_asymm @ Dx_new

        return (Dx_new, D2x_new, Dy_new, D2y_new, Dxy_new, Dyx_new, det,
                Dy_symm_new, D2y_symm_new, Dy_asymm_new, D2y_asymm_new,
                Dxy_symm_new, Dyx_symm_new, Dxy_asymm_new, Dyx_asymm_new,
                x2p, np.zeros_like(x2p), np.zeros_like(x2p), np.ones_like(x2p))

    def _deform_y_only(self, Y_new, p):
        """1-D chain-rule deformation in y (MATLAB DeformMesh branch 2)."""
        y = Y_new.ravel(order="F")[p]
        Dx = self.Dx
        Dy = self.Dy
        D2y = self.D2y

        y2p = Dy @ y           # dy_new / dy_comp
        y2pp = D2y @ y         # d²y_new / dy_comp²

        inv_y2p = 1.0 / y2p
        Dx_new = Dx
        D2x_new = self.D2x
        Dy_new = _axm_diag(inv_y2p, Dy)
        D2y_new = _axm_diag(inv_y2p ** 2, D2y) - _axm_diag(y2pp * inv_y2p ** 3, Dy)
        Dxy_new = Dx @ Dy_new
        Dyx_new = Dy_new @ Dx
        det = y2p

        # y-symmetry variants
        Dy_symm_new = None
        D2y_symm_new = None
        Dy_asymm_new = None
        D2y_asymm_new = None
        Dxy_symm_new = None
        Dyx_symm_new = None
        Dxy_asymm_new = None
        Dyx_asymm_new = None

        if self.y_symmetry and self.Dy_symm is not None:
            Dy_symm_new = _axm_diag(inv_y2p, self.Dy_symm)
            D2y_symm_new = (_axm_diag(inv_y2p ** 2, self.D2y_symm)
                            - _axm_diag(y2pp * inv_y2p ** 3, self.Dy_symm))
            Dy_asymm_new = _axm_diag(inv_y2p, self.Dy_asymm)
            D2y_asymm_new = (_axm_diag(inv_y2p ** 2, self.D2y_asymm)
                             - _axm_diag(y2pp * inv_y2p ** 3, self.Dy_asymm))
            Dxy_symm_new = Dx @ Dy_symm_new
            Dyx_symm_new = Dy_symm_new @ Dx
            Dxy_asymm_new = Dx @ Dy_asymm_new
            Dyx_asymm_new = Dy_asymm_new @ Dx

        return (Dx_new, D2x_new, Dy_new, D2y_new, Dxy_new, Dyx_new, det,
                Dy_symm_new, D2y_symm_new, Dy_asymm_new, D2y_asymm_new,
                Dxy_symm_new, Dyx_symm_new, Dxy_asymm_new, Dyx_asymm_new,
                np.ones_like(y2p), np.zeros_like(y2p), np.zeros_like(y2p), y2p)

    def _deform_full_2d(self, X_new, Y_new, p):
        """Full 2-D Jacobian deformation (MATLAB DeformMesh branch 3)."""
        x = X_new.ravel(order="F")[p]
        y = Y_new.ravel(order="F")[p]

        Dx = self.Dx
        D2x = self.D2x
        Dy = self.Dy
        D2y = self.D2y

        # MATLAB DeformMesh.m Jacobian layout:
        #   J = [[dX/dxi,  dY/dxi],
        #        [dX/deta, dY/deta]]
        J11 = Dx @ x
        J12 = Dx @ y
        J21 = Dy @ x
        J22 = Dy @ y
        detJ = J11 * J22 - J12 * J21
        if np.any(np.abs(detJ) < 1e-15):
            raise ValueError("Degenerate mapping: Jacobian determinant is near zero")

        inv_det = 1.0 / detJ
        Ji11 = J22 * inv_det
        Ji12 = -J12 * inv_det
        Ji21 = -J21 * inv_det
        Ji22 = J11 * inv_det

        Dx_new = _axm_diag(Ji11, Dx) + _axm_diag(Ji12, Dy)
        Dy_new = _axm_diag(Ji21, Dx) + _axm_diag(Ji22, Dy)

        Dxy = Dx @ Dy
        Dyx = Dy @ Dx
        D2x_new, Dxy_new, Dyx_new, D2y_new = _transform_second_derivatives(
            Ji11, Ji12, Ji21, Ji22, D2x, D2y, Dxy, Dyx, Dx, Dy, x, y,
        )

        Dy_symm_new = None
        D2y_symm_new = None
        Dy_asymm_new = None
        D2y_asymm_new = None
        Dxy_symm_new = None
        Dyx_symm_new = None
        Dxy_asymm_new = None
        Dyx_asymm_new = None

        if self.y_symmetry and self.Dy_symm is not None and self.D2y_symm is not None:
            Dy_symm_new = _axm_diag(Ji21, Dx) + _axm_diag(Ji22, self.Dy_symm)
            Dy_asymm_new = _axm_diag(Ji21, Dx) + _axm_diag(Ji22, self.Dy_asymm)

            _, Dxy_symm_new, Dyx_symm_new, D2y_symm_new = _transform_second_derivatives(
                Ji11, Ji12, Ji21, Ji22, D2x, self.D2y_symm,
                self.Dxy_symm, self.Dyx_symm, Dx, self.Dy_symm, x, y,
            )
            _, Dxy_asymm_new, Dyx_asymm_new, D2y_asymm_new = _transform_second_derivatives(
                Ji11, Ji12, Ji21, Ji22, D2x, self.D2y_asymm,
                self.Dxy_asymm, self.Dyx_asymm, Dx, self.Dy_asymm, x, y,
            )

        det = detJ

        return (Dx_new, D2x_new, Dy_new, D2y_new, Dxy_new, Dyx_new, det,
                Dy_symm_new, D2y_symm_new, Dy_asymm_new, D2y_asymm_new,
                Dxy_symm_new, Dyx_symm_new, Dxy_asymm_new, Dyx_asymm_new,
                J11, J12, J21, J22)

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
        dx = self.X[0, 1] - self.X[0, 0] if self.X.shape[1] > 1 else 1.0
        dy = self.Y[1, 0] - self.Y[0, 0] if self.Y.shape[0] > 1 else 1.0

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
            J11=np.ones(used_ind.size),
            J12=np.zeros(used_ind.size),
            J21=np.zeros(used_ind.size),
            J22=np.ones(used_ind.size),
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
    if np.isclose(dx, 0.0):
        dx = 1.0
    if np.isclose(dy, 0.0):
        dy = 1.0

    # --- Fast path: Kronecker product for unmasked rectangular meshes ---
    if active_mask is None and ngp == nx * ny:
        return _build_operators_kron(
            X, Y, dx, dy, nx, ny, fd_order, x_periodicity, y_symmetry,
        )

    # --- Slow path: line-by-line LIL assembly for masked meshes ----------
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

    # Coordinates of used points (local indexing)
    x_used = x_flat[used_ind]
    y_used = y_flat[used_ind]

    periodic_period = None
    if x_periodicity and np.any(active_mask_1d):
        periodic_period = _build_periodic_1d_operators(nx, dx, fd_order)

    # --- X derivative matrices (loop over horizontal lines) -----------
    # Match MATLAB AddDW2mesh: iterate over unique y-values and collect
    # all used points at each y-level, then build 1-D FD along x.
    y_unique = np.sort(np.unique(y_used))
    for yval in y_unique:
        pos = np.where(y_used == yval)[0]
        if pos.size == 0:
            continue

        if x_periodicity and pos.size == nx:
            D1_1d, D2_1d = periodic_period
            Dx[np.ix_(pos, pos)] = D1_1d
            D2x[np.ix_(pos, pos)] = D2_1d
            w_full[used_ind[pos]] *= dx
            continue

        # Sort by x-coordinate and split by gaps.
        order = np.argsort(x_used[pos])
        pos = pos[order]
        x_line = x_used[pos]
        dx_line = np.diff(x_line) / dx
        breaks = _line_breaks(dx_line)
        dom_bounds = np.concatenate(([0], breaks + 1, [pos.size]))

        for i_seg in range(len(dom_bounds) - 1):
            seg = pos[dom_bounds[i_seg] : dom_bounds[i_seg + 1]]
            n_seg = seg.size
            if n_seg < 3:
                continue

            fd_ord = fd_order
            if n_seg < min_points(fd_ord):
                fd_ord = 2

            D1_1d, D2_1d = build_sbp_matrix(n_seg, dx, fd_ord)

            idx = np.ix_(seg, seg)
            Dx[idx] = D1_1d
            D2x[idx] = D2_1d

            global_seg = used_ind[seg]
            w_full[global_seg] *= dx
            w_full[global_seg[0]] *= 0.5
            w_full[global_seg[-1]] *= 0.5

    # --- Y derivative matrices (loop over vertical lines) -------------
    # Match MATLAB AddDW2mesh: iterate over unique x-values and collect
    # all used points at each x-level, then build 1-D FD along y.
    x_unique = np.sort(np.unique(x_used))
    for xval in x_unique:
        pos = np.where(x_used == xval)[0]
        if pos.size == 0:
            continue

        order = np.argsort(y_used[pos])
        pos = pos[order]
        y_line = y_used[pos]
        dy_line = np.diff(y_line) / dy
        breaks = _line_breaks(dy_line)
        dom_bounds = np.concatenate(([0], breaks + 1, [pos.size]))

        for i_seg in range(len(dom_bounds) - 1):
            seg = pos[dom_bounds[i_seg] : dom_bounds[i_seg + 1]]
            n_seg = seg.size
            if n_seg < 3:
                continue

            fd_ord = fd_order
            if n_seg < min_points(fd_ord):
                fd_ord = 2

            D1_base, D2_base = build_sbp_matrix(n_seg, dy, fd_ord)
            D1_sym = None
            D1_asym = None
            D2_sym = None
            D2_asym = None

            if y_symmetry and _starts_at_pole_line(y_line[dom_bounds[i_seg]], dy):
                D1_sym, D1_asym, D2_sym, D2_asym = build_sbp_pole_matrices(
                    n_seg,
                    dy,
                    fd_ord,
                )

            idx = np.ix_(seg, seg)
            Dy[idx] = D1_base
            D2y[idx] = D2_base
            if y_symmetry:
                Dy_symm[idx] = D1_sym if D1_sym is not None else D1_base
                D2y_symm[idx] = D2_sym if D2_sym is not None else D2_base
                Dy_asymm[idx] = D1_asym if D1_asym is not None else D1_base
                D2y_asymm[idx] = D2_asym if D2_asym is not None else D2_base

            global_seg = used_ind[seg]
            w_full[global_seg] *= dy
            w_full[global_seg[0]] *= 0.5
            w_full[global_seg[-1]] *= 0.5

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


def _starts_at_pole_line(y0: float, dy: float) -> bool:
    """Match MATLAB pole-line detection for symmetry meshes.

    ``AddDW2mesh.m`` applies the Mohseni pole closure when the first y-point
    on a line sits at the half-cell location ``dy/2`` (within ``0.1*dy``),
    not only when it is exactly zero.
    """
    dy_abs = abs(float(dy))
    tol = max(1e-12, 0.1 * dy_abs)
    y0 = float(y0)
    return abs(y0) <= tol or abs(y0 - 0.5 * dy_abs) <= tol


def _is_separable_axis_deformation(X_new: np.ndarray, Y_new: np.ndarray) -> bool:
    """Whether the supplied deformation is separable in x and y."""
    return (
        np.allclose(X_new, X_new[0:1, :], rtol=0.0, atol=1e-12)
        and np.allclose(Y_new, Y_new[:, 0:1], rtol=0.0, atol=1e-12)
    )


def _build_operators_kron(
    X: np.ndarray,
    Y: np.ndarray,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    fd_order: int,
    x_periodicity: bool,
    y_symmetry: bool,
) -> tuple:
    """Build derivative matrices via Kronecker products (unmasked fast path).

    For a full nx×ny grid, 2D operators are tensor products of 1D SBP stencils:
        Dx = I_ny ⊗ D1x,   Dy = D1y ⊗ I_nx
    This avoids the O(nx*ny) Python-level loop in the general LIL path.
    """
    I_nx = sp.eye(nx, format="csr")
    I_ny = sp.eye(ny, format="csr")

    # --- 1D operators in x ---
    if x_periodicity:
        D1x, D2x = _build_periodic_1d_operators(nx, dx, fd_order)
        wx = np.full(nx, dx)
    else:
        D1x, D2x = build_sbp_matrix(nx, dx, fd_order)
        wx = np.full(nx, dx)
        wx[0] *= 0.5
        wx[-1] *= 0.5

    # --- 1D operators in y ---
    D1y, D2y = build_sbp_matrix(ny, dy, fd_order)
    wy = np.full(ny, dy)
    wy[0] *= 0.5
    wy[-1] *= 0.5

    # --- 2D operators via kron (Fortran order: y varies fastest) ---
    Dx_g = sp.kron(D1x, I_ny, format="csr")
    D2x_g = sp.kron(D2x, I_ny, format="csr")
    Dy_g = sp.kron(I_nx, D1y, format="csr")
    D2y_g = sp.kron(I_nx, D2y, format="csr")
    Dxy_g = Dx_g @ Dy_g
    Dyx_g = Dy_g @ Dx_g

    # --- Quadrature weights (outer product) ---
    W_g = np.outer(wy, wx)  # shape (ny, nx), column j = wy * wx[j]

    # --- y-symmetry operators ---
    Dy_symm_g = None
    D2y_symm_g = None
    Dy_asymm_g = None
    D2y_asymm_g = None
    Dxy_symm_g = None
    Dyx_symm_g = None
    Dxy_asymm_g = None
    Dyx_asymm_g = None
    W_symm_g = None
    W_asymm_g = None

    if y_symmetry:
        y_col = Y[:, 0]
        has_pole = _starts_at_pole_line(y_col[0], dy)
        if has_pole:
            D1y_sym, D1y_asym, D2y_sym, D2y_asym = build_sbp_pole_matrices(
                ny, dy, fd_order,
            )
        else:
            D1y_sym = D1y
            D1y_asym = D1y
            D2y_sym = D2y
            D2y_asym = D2y

        Dy_symm_g = sp.kron(I_nx, D1y_sym, format="csr")
        D2y_symm_g = sp.kron(I_nx, D2y_sym, format="csr")
        Dy_asymm_g = sp.kron(I_nx, D1y_asym, format="csr")
        D2y_asymm_g = sp.kron(I_nx, D2y_asym, format="csr")
        Dxy_symm_g = Dx_g @ Dy_symm_g
        Dyx_symm_g = Dy_symm_g @ Dx_g
        Dxy_asymm_g = Dx_g @ Dy_asymm_g
        Dyx_asymm_g = Dy_asymm_g @ Dx_g
        W_symm_g = W_g.copy()
        W_asymm_g = W_g.copy()
        axis_nodes = np.isclose(Y, 0.0, atol=1e-12)
        W_asymm_g[axis_nodes] *= 0.75

    return (
        Dx_g, D2x_g, Dy_g, D2y_g,
        Dy_symm_g, D2y_symm_g, Dy_asymm_g, D2y_asymm_g,
        Dxy_g, Dyx_g,
        Dxy_symm_g, Dyx_symm_g, Dxy_asymm_g, Dyx_asymm_g,
        W_g, W_symm_g, W_asymm_g,
    )


def _build_indices(
    X: np.ndarray,
    Y: np.ndarray,
    used_ind: np.ndarray,
    x_periodicity: bool,
    y_symmetry: bool,
    mask: np.ndarray,
) -> dict:
    """Build reduced index sets for boundaries and mask edges.

    Indices are local (0-based into the ``used_ind`` array).  Boundary
    detection uses a 2-D boolean mask that is flattened in Fortran order
    to match the column-major convention used throughout the package.
    """
    del y_symmetry
    if np.asarray(mask).shape != X.shape:
        raise ValueError("mask must match X and Y shape")

    mask_2d = np.asarray(mask, dtype=bool)
    full_n = X.size
    global_to_local = np.full(full_n, -1, dtype=int)
    global_to_local[used_ind] = np.arange(used_ind.size)

    def _boundary_to_local(row_mask_2d: np.ndarray) -> np.ndarray:
        """Convert a 2-D boolean boundary mask to local indices.

        ``row_mask_2d`` must have the same shape as ``X``; True entries
        mark the boundary points.  The mask is flattened in Fortran order
        to obtain global flat indices, which are then mapped to local
        indices via ``global_to_local``.
        """
        global_idx = np.flatnonzero(row_mask_2d.ravel(order="F"))
        local = global_to_local[global_idx]
        return local[local >= 0]

    # Bottom row (row 0) and top row (row -1)
    bi_mask = np.zeros_like(mask_2d)
    bi_mask[0, :] = mask_2d[0, :]
    bi = _boundary_to_local(bi_mask)

    ti_mask = np.zeros_like(mask_2d)
    ti_mask[-1, :] = mask_2d[-1, :]
    ti = _boundary_to_local(ti_mask)

    if x_periodicity:
        li = np.array([], dtype=int)
        ri = np.array([], dtype=int)
    else:
        li_mask = np.zeros_like(mask_2d)
        li_mask[:, 0] = mask_2d[:, 0]
        li = _boundary_to_local(li_mask)

        ri_mask = np.zeros_like(mask_2d)
        ri_mask[:, -1] = mask_2d[:, -1]
        ri = _boundary_to_local(ri_mask)

    # Match MATLAB MeshMask.m exactly: ``mi`` is the set of *used* points
    # adjacent to the masked-out region, using a full 3x3 neighborhood on
    # the removed-point mask (including diagonals, without periodic wrap).
    removed = ~mask_2d
    removed_pad = np.pad(removed, 1, mode="constant", constant_values=False)
    near_removed = np.zeros_like(mask_2d, dtype=bool)
    for di in range(3):
        for dj in range(3):
            near_removed |= removed_pad[di : di + mask_2d.shape[0], dj : dj + mask_2d.shape[1]]

    mi_mask = mask_2d & near_removed
    mi = _boundary_to_local(mi_mask)

    return {"bi": bi, "ti": ti, "li": li, "ri": ri, "mi": mi}


def _line_breaks(deltas: np.ndarray, *, min_gap_ratio: float = 3.0) -> np.ndarray:
    """Return local-gap indices for segmenting 1-D derivative lines."""
    if deltas.size == 0:
        return np.array([], dtype=int)

    scale = np.median(np.abs(deltas))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.array([], dtype=int)

    threshold = max(1.5, min_gap_ratio * scale)
    return np.where(np.abs(deltas) > threshold)[0]


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

    Dyx_x = Dy @ (Dx @ x)
    Dyx_y = Dy @ (Dx @ y)
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
        a * d,
        b * c,
        b * d,
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
        c * b,
        d * a,
        d * b,
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
