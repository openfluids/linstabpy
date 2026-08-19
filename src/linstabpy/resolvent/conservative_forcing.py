# SPDX-License-Identifier: GPL-3.0-only
"""Conservative-momentum forcing space for the resolvent optimisation variable.

By default the resolvent forcing lives in the five PRIMITIVE
perturbation variables ``q' = (rho', u', v', w', T')``.
The region mask in ``resolvent_map._build_region_mask`` is a diagonal 0/1
selector on that space.

The physically motivated forcing for a shear-layer/shock-interaction
problem acts on three CONSERVATIVE momentum equations.
``(rho u)', (rho v)', (rho w)'``.
A conservative momentum forcing ``f_c = (f_rou, f_rov, f_row)`` maps
into the equivalent primitive-equation forcing.
The map is linear and POINTWISE (no coupling between grid points).
At each active grid point,

    e_rou -> (0, 1/rho_bar, 0, 0, -kappa*(kappa-1)*Ma^2 * u_bar/rho_bar).
    e_rov -> (0, 0, 1/rho_bar, 0, -kappa*(kappa-1)*Ma^2 * v_bar/rho_bar).
    e_row -> (0, 0, 0, 1/rho_bar, -kappa*(kappa-1)*Ma^2 * w_bar/rho_bar).

Primitive block order is ``(rho, u, v, w, T)``.
``rho_bar``, ``u_bar``, ``v_bar``, ``w_bar`` are base-flow fields.
``kappa``/``Ma`` are the same quantities
:func:`linstabpy.operators.energy_norm.compute_chu_norm` takes.

The map is a sparse matrix ``B`` (or a WHITENED version ``B_tilde``,
see below).
It plugs directly into the existing ``B_mat``-shaped forcing-matrix slot
used throughout ``linstabpy.resolvent`` and
``linstabpy.solvers.router.resolvent_map``.
The resolvent eigenproblem machinery does not need a change.

Norm: whitening instead of a new norm branch.
The physically meaningful norm on ``f_c`` is the mass-weighted L2 norm

    ||f_c||_F^2 = sum_j w_int_j * |f_c_j|^2.

This uses the SAME integration weights ``w_int`` that already enter
``compute_chu_norm``.
Rather than adding a third ``forcing_norm`` branch to
:class:`linstabpy.resolvent.gain.WeightedNormalOperator`, WHITEN the
columns of ``B``:

    B_tilde = B @ diag(w)^(-1/2).

The caller then runs the resolvent with ``forcing_norm="l2"`` (already
supported).
The plain L2 norm of the returned (whitened) coefficient vector ``g``
IS the mass-weighted conservative norm of the physical forcing.

    f_c = diag(w)^(-1/2) @ g.

That holds because

    ||g||_2^2 = ||diag(w)^(1/2) f_c||_2^2 = sum_j w_int_j |f_c_j|^2.

The same de-whitening vector ``diag(w)^(-1/2)`` therefore does double duty.
Construction bakes it into ``B_tilde``'s columns.
It is also the factor that converts an exported forcing coefficient
back into the physical conservative-momentum forcing
(``momentum_dewhiten`` below).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import scipy.sparse as sp

MOMENTUM_VARS: tuple[str, ...] = ("u", "v", "w")
_CONSERVATIVE_LABEL = {"u": "rou", "v": "rov", "w": "row"}


def validate_conservative_momentum_vars(raw_vars: object, *, name: str) -> tuple[str, ...]:
    """Validate the ``vars`` key of a conservative-momentum region spec.

    ``vars`` must be absent (meaning all three momentum components) or a
    non-empty subset of ``{"u", "v", "w"}``.
    Any density/temperature entry raises.
    The conservative-momentum forcing space has no rho'/T'
    optimisation variable of its own.
    Those are the OUTPUT of the mapping, not inputs to it.
    """
    if raw_vars is None:
        return MOMENTUM_VARS
    if isinstance(raw_vars, str):
        requested = (raw_vars,)
    else:
        try:
            requested = tuple(raw_vars)
        except TypeError as exc:
            raise ValueError(f"{name} vars must be an iterable of variable names") from exc
    if not requested:
        raise ValueError(f"{name} vars must select at least one momentum variable")
    requested = tuple(str(v) for v in requested)
    bad = [v for v in requested if v not in MOMENTUM_VARS]
    if bad:
        raise ValueError(
            f"{name} space='conservative_momentum' only accepts momentum vars "
            f"{MOMENTUM_VARS}; got unsupported entries {bad}. rho'/T' are outputs "
            "of the conservative-to-primitive map, not forcing inputs."
        )
    return requested


def conservative_momentum_coefficients(
    rho_bar: np.ndarray,
    u_bar: np.ndarray,
    v_bar: np.ndarray,
    w_bar: np.ndarray,
    kappa: float,
    ma: float,
    *,
    momentum_vars: tuple[str, ...] = MOMENTUM_VARS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pointwise coefficients mapping conservative momentum forcing to primitive.

    For each requested momentum var (``"u"``, ``"v"``, or ``"w"``),
    return a pair ``(own_coeff, T_coeff)``.
    The arrays have the same shape as ``rho_bar``.

        own_coeff = 1 / rho_bar.
        T_coeff   = -kappa * (kappa - 1) * Ma**2 * {u,v,w}_bar / rho_bar.

    Primitive forcing component ``own`` picks up
    ``own_coeff * f_{rou,rov,row}``.
    The temperature component picks up
    ``T_coeff * f_{rou,rov,row}`` (summed over the requested momentum vars).
    """
    rho_bar = np.asarray(rho_bar, dtype=float)
    if rho_bar.size and np.any(rho_bar == 0.0):
        raise ValueError(
            "conservative-momentum forcing map is singular where rho_bar == 0"
        )
    bars = {
        "u": np.asarray(u_bar, dtype=float),
        "v": np.asarray(v_bar, dtype=float),
        "w": np.asarray(w_bar, dtype=float),
    }
    prefactor = -float(kappa) * (float(kappa) - 1.0) * float(ma) ** 2
    coeffs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for var in momentum_vars:
        if var not in bars:
            raise ValueError(f"unknown momentum var {var!r}; expected one of {MOMENTUM_VARS}")
        own = 1.0 / rho_bar
        temp = prefactor * bars[var] / rho_bar
        coeffs[var] = (own, temp)
    return coeffs


def build_conservative_momentum_matrix(
    idx: Mapping[str, np.ndarray],
    ndof: int,
    point_indices: np.ndarray,
    rho_bar_active: np.ndarray,
    u_bar_active: np.ndarray,
    v_bar_active: np.ndarray,
    w_bar_active: np.ndarray,
    w_int_active: np.ndarray,
    kappa: float,
    ma: float,
    *,
    momentum_vars: tuple[str, ...] = MOMENTUM_VARS,
    excluded_dofs: np.ndarray | None = None,
) -> tuple[sp.csr_matrix, np.ndarray, dict[str, object]]:
    """Build the whitened conservative-momentum forcing matrix ``B_tilde``.

    Embed the (rectangular, in the physical sense) conservative-momentum
    forcing map into a SQUARE ``ndof x ndof`` matrix.
    Reuse the momentum block slots of the primitive DOF layout
    (``idx['u_j']``, ``idx['v_j']``, ``idx['w_j']``).
    Those slots store the whitened coefficient vector ``g``.
    Columns outside the selected region/vars, and the ``rho_j``/``T_j``
    columns, are identically zero.
    The existing square resolvent machinery
    (``ResolventOperator``, ``WeightedNormalOperator``) then needs no change.
    It already accepts an arbitrary ``ndof x ndof`` sparse/PETSc ``B_mat``.

    Parameters
    ----------
    idx : mapping
        LNSE index blocks (``{var}_j`` -> global dof array), same convention
        as ``resolvent_map._build_region_mask``.
    ndof : int
        Full state size (``5 * ngp``).
    point_indices : ndarray of int
        Indices into the ACTIVE-POINT arrays (``rho_bar_active`` and the
        ``idx['{var}_j']`` blocks).
        These select the spatial forcing region.
    rho_bar_active, u_bar_active, v_bar_active, w_bar_active : ndarray
        Base-flow fields restricted to active grid points, same ordering as
        ``idx['{var}_j']`` (i.e. ``mesh.X.ravel(order='F')[mesh.used_ind]``
        ordering).
    w_int_active : ndarray
        Integration weights restricted to active grid points, same ordering
        and the SAME array ``compute_chu_norm`` receives as ``W_int``.
    kappa, ma : float
        Heat capacity ratio and Mach number.
    momentum_vars : tuple of str
        Subset of ``("u", "v", "w")`` to force.
    excluded_dofs : ndarray of int, optional
        Global dof indices to drop from the region (e.g. boundary-condition
        rows).
        A given momentum var drops a point if that var's primitive dof
        at that point appears in ``excluded_dofs``.

    Returns
    -------
    B_tilde : scipy.sparse.csr_matrix, shape (ndof, ndof)
        Whitened conservative-momentum forcing matrix,
        ``B_tilde = B @ diag(w)^(-1/2)``.
    momentum_dewhiten : ndarray, shape (ndof,)
        De-whitening vector.
        At the momentum dofs used as coefficient slots this is
        ``1/sqrt(w_int)``.
        Elsewhere it is 1.0 and unused (the corresponding ``B_tilde``
        column is identically zero).
        Recover the physical conservative forcing from an exported
        coefficient vector ``g`` as ``f_c = momentum_dewhiten * g``
        restricted to ``momentum_dofs``.
    info : dict
        ``{"n_region": int, "momentum_dofs": ndarray, "vars": tuple}``.
    """
    point_indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    if point_indices.size == 0:
        raise ValueError("conservative_momentum region selects zero active points")

    rho_bar_active = np.asarray(rho_bar_active, dtype=float)
    n_total = rho_bar_active.size
    for name, arr in (
        ("u_bar_active", u_bar_active),
        ("v_bar_active", v_bar_active),
        ("w_bar_active", w_bar_active),
        ("w_int_active", w_int_active),
    ):
        if np.asarray(arr).size != n_total:
            raise ValueError(
                f"{name} has {np.asarray(arr).size} entries, expected {n_total} "
                "(same length as rho_bar_active)"
            )
    w_int_active = np.asarray(w_int_active, dtype=float)
    if np.any(w_int_active[point_indices] <= 0.0):
        raise ValueError(
            "conservative_momentum region: integration weight w_int must be > 0 "
            "at every selected point to whiten the forcing norm"
        )

    rho_sel = rho_bar_active[point_indices]
    coeffs = conservative_momentum_coefficients(
        rho_sel,
        u_bar_active[point_indices],
        v_bar_active[point_indices],
        w_bar_active[point_indices],
        kappa,
        ma,
        momentum_vars=momentum_vars,
    )
    dewhiten_local = 1.0 / np.sqrt(w_int_active[point_indices])

    if "T_j" not in idx:
        raise ValueError("conservative_momentum region: missing index block 'T_j'")
    T_j = np.asarray(idx["T_j"], dtype=np.int64).reshape(-1)
    if T_j.size != n_total:
        raise ValueError(
            f"conservative_momentum region: index block 'T_j' has {T_j.size} entries "
            f"for {n_total} active points"
        )
    T_dofs_sel = T_j[point_indices]

    excluded = None
    if excluded_dofs is not None:
        excluded = set(np.asarray(excluded_dofs, dtype=np.int64).reshape(-1).tolist())

    momentum_dewhiten = np.ones(int(ndof), dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    momentum_dofs: list[int] = []

    for var in momentum_vars:
        idx_key = f"{var}_j"
        if idx_key not in idx:
            raise ValueError(f"conservative_momentum region: missing index block {idx_key!r}")
        var_j = np.asarray(idx[idx_key], dtype=np.int64).reshape(-1)
        if var_j.size != n_total:
            raise ValueError(
                f"conservative_momentum region: index block {idx_key!r} has "
                f"{var_j.size} entries for {n_total} active points"
            )
        col_dofs = var_j[point_indices]
        own_coeff, T_coeff = coeffs[var]
        for k in range(col_dofs.size):
            col = int(col_dofs[k])
            if excluded is not None and col in excluded:
                continue
            wcol = dewhiten_local[k]
            rows.append(col)
            cols.append(col)
            data.append(float(own_coeff[k]) * wcol)
            row_T = int(T_dofs_sel[k])
            # The induced-T entry lands on the T EQUATION row; if that row is a
            # boundary-condition row it must stay untouched, exactly as the
            # diagonal-mask path zeroes forcing on every BC row. The momentum
            # entry above is safe: its row equals `col`, already screened.
            if excluded is None or row_T not in excluded:
                rows.append(row_T)
                cols.append(col)
                data.append(float(T_coeff[k]) * wcol)
            momentum_dewhiten[col] = wcol
            momentum_dofs.append(col)

    if not momentum_dofs:
        raise ValueError(
            "conservative_momentum region selects zero forcing dofs after composing "
            "with the boundary-condition mask; widen the region or set "
            "mask_bc_forcing=false"
        )

    B_tilde = sp.csr_matrix((data, (rows, cols)), shape=(int(ndof), int(ndof)))
    info = {
        "n_region": int(point_indices.size),
        "momentum_dofs": np.asarray(sorted(set(momentum_dofs)), dtype=np.int64),
        "vars": tuple(momentum_vars),
    }
    return B_tilde, momentum_dewhiten, info


def dewhiten_forcing(
    coefficient: np.ndarray, momentum_dewhiten: np.ndarray, momentum_dofs: np.ndarray
) -> np.ndarray:
    """Recover the physical conservative-momentum forcing from a whitened coefficient.

    ``f_c = diag(w)^(-1/2) @ g``, restricted to the dofs used as
    momentum-coefficient slots (``momentum_dofs``).
    All other entries of the returned array are zero by construction.
    ``g`` is zero there too, since ``B_tilde``'s columns are zero outside
    ``momentum_dofs``.
    """
    coefficient = np.asarray(coefficient)
    out = np.zeros_like(coefficient)
    out[momentum_dofs] = momentum_dewhiten[momentum_dofs] * coefficient[momentum_dofs]
    return out
