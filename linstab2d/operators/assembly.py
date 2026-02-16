"""LNSE operator assembly from coefficients and FD matrices.

Translates GetLinProblem.m: assembles the 5*ngp × 5*ngp linear operator
L0 such that dq/dt = L0 @ q, with q = [rho, u, v, w, T].
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..mesh.mesh import Mesh
from ..baseflow.baseflow import BaseFlow
from .coefficients import (
    compute_cartesian_coefficients,
    compute_cylindrical_coefficients,
    LNSECoefficients,
)


def build_lnse_operator(
    mesh: Mesh,
    baseflow: BaseFlow,
    kx: float = 0.0,
    model: str = "2D",
    m: float = 0,
    floquet_exp: float = 0.0,
) -> tuple[sp.csc_matrix, dict]:
    """Build the LNSE operator L0 as a scipy sparse matrix.

    Assembles L0 = inv(B/1i) @ (A0 + Ay*Dr + Ax*Dz + Ayy*D2r + Axx*D2z + Axy*D2rz)
    with coordinate remapping and sign conventions already absorbed into the
    coefficients, then subtracts the sponge function.

    Supported models:
    - ``2D``: cartesian BL form with Fourier spanwise wavenumber ``kx``.
    - ``axy``: cylindrical axisymmetric form with azimuthal mode ``m`` and
      optional Floquet shift.

    Parameters
    ----------
    mesh : Mesh
        Mesh object with FD matrices and integration weights.
    baseflow : BaseFlow
        Base flow with velocity, thermodynamic, and viscosity fields.
    kx : float
        Streamwise Fourier wavenumber (physical x-direction), used for
        the Cartesian model only.
    model : str
        Model selector ``"2D"`` or ``"axy"``.
    m : float
        Azimuthal Fourier mode for cylindrical model ``"axy"``.
    floquet_exp : float
        Floquet exponent used only by the cylindrical model.

    Returns
    -------
    L0 : csc_matrix, shape (5*ngp, 5*ngp)
        The assembled LNSE operator.
    idx : dict
        Index mapping for variables and boundaries. Keys include:
        - 'bi', 'ti', 'li', 'ri': boundary grid point indices (0-based)
        - 'rho_j', 'u_j', 'v_j', 'w_j', 'T_j': DOF index arrays for each variable
        - '{border}_{var}': DOF indices for each variable at each border
    """
    ngp = mesh.ngp
    p = mesh.used_ind  # active grid point indices

    # --- Extract base flow fields (flattened to 1D) ---
    U = baseflow.U.ravel(order="F")[p]
    V = baseflow.V.ravel(order="F")[p]
    W = baseflow.W.ravel(order="F")[p]
    RHO = baseflow.RHO.ravel(order="F")[p]
    T = baseflow.T.ravel(order="F")[p]
    MU = baseflow.MU.ravel(order="F")[p]
    dmudT = baseflow.dmudT.ravel(order="F")[p]
    d2mudT2 = baseflow.d2mudT2.ravel(order="F")[p]

    kappa = baseflow.kappa
    Ma = baseflow.Ma
    Re = baseflow.Re
    Pr = baseflow.Pr

    # Pressure from equation of state: P = RHO * T / (Ma^2 * kappa)
    P = RHO * T / (Ma**2 * kappa)

    # --- FD matrices (scipy sparse, ngp x ngp) ---
    Dx = mesh.Dx  # derivative along physical x
    D2x = mesh.D2x
    Dy = mesh.Dy  # derivative along physical y / radius
    D2y = mesh.D2y
    D2RZ = mesh.Dyx  # mixed derivative used by axisymmetric model

    model = model.lower()
    if model == "2d":
        # -- Cartesian model -------------------------------------------------
        # Compute derivatives in physical coordinates (x,y) used by
        # ``compute_cartesian_coefficients``.
        dUdx = Dx @ U;    dUdy = Dy @ U
        dVdx = Dx @ V;    dVdy = Dy @ V
        dWdx = Dx @ W;    dWdy = Dy @ W
        dRHOdx = Dx @ RHO;  dRHOdy = Dy @ RHO
        dTdx = Dx @ T;    dTdy = Dy @ T
        dPdx = Dx @ P;    dPdy = Dy @ P

        d2Udx2 = D2x @ U;  d2Udy2 = D2y @ U
        d2Vdx2 = D2x @ V;  d2Vdy2 = D2y @ V
        d2Wdx2 = D2x @ W;  d2Wdy2 = D2y @ W
        d2Tdx2 = D2x @ T;  d2Tdy2 = D2y @ T

        # Mixed derivatives: Dz @ (Dr @ f) in MATLAB = Dx @ (Dy @ f)
        d2Udxdy = Dx @ dUdy
        d2Vdxdy = Dx @ dVdy
        d2Wdxdy = Dx @ dWdy

        coeffs = compute_cartesian_coefficients(
            U=U, V=V, W=W, RHO=RHO, T=T, MU=MU, P=P,
            kx=kx, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa,
            dUdx=dUdx, dVdx=dVdx, dWdx=dWdx,
            dRHOdx=dRHOdx, dTdx=dTdx, dPdx=dPdx,
            dUdy=dUdy, dVdy=dVdy, dWdy=dWdy,
            dRHOdy=dRHOdy, dTdy=dTdy, dPdy=dPdy,
            d2Udx2=d2Udx2, d2Vdx2=d2Vdx2, d2Wdx2=d2Wdx2,
            d2Udy2=d2Udy2, d2Vdy2=d2Vdy2, d2Wdy2=d2Wdy2,
            d2Tdx2=d2Tdx2, d2Tdy2=d2Tdy2,
            d2Udxdy=d2Udxdy, d2Vdxdy=d2Vdxdy, d2Wdxdy=d2Wdxdy,
            dmudT=dmudT, d2mudT2=d2mudT2,
        )

        DR = Dy
        D2R = D2y
        DZ = Dx
        D2Z = D2x

    elif model == "axy":
        # -- Cylindrical axisymmetric model --------------------------------
        R = mesh.Y.ravel(order="F")[p]
        U_c = baseflow.V.ravel(order="F")[p]
        V_c = baseflow.W.ravel(order="F")[p]
        W_c = baseflow.U.ravel(order="F")[p]

        dUdr = Dy @ U_c
        dVdr = Dy @ V_c
        dWdr = Dy @ W_c
        dRHOdr = Dy @ RHO
        dTdr = Dy @ T
        dUdz = Dx @ U_c
        dVdz = Dx @ V_c
        dWdz = Dx @ W_c
        dRHOdz = Dx @ RHO
        dTdz = Dx @ T

        d2Udr2 = D2y @ U_c
        d2Vdr2 = D2y @ V_c
        d2Wdr2 = D2y @ W_c
        d2Tdr2 = D2y @ T
        d2Udz2 = D2x @ U_c
        d2Vdz2 = D2x @ V_c
        d2Wdz2 = D2x @ W_c
        d2Tdz2 = D2x @ T

        d2Udrz = Dx @ dUdr
        d2Vdrz = Dx @ dVdr
        d2Wdrz = Dx @ dWdr

        coeffs = compute_cylindrical_coefficients(
            U=U_c, V=V_c, W=W_c, R=R, RHO=RHO, T=T, MU=MU, m=m, Re=Re, Ma=Ma,
            Pr=Pr, kappa=kappa,
            dUdr=dUdr, dVdr=dVdr, dWdr=dWdr, dRHOdr=dRHOdr, dTdr=dTdr,
            dUdz=dUdz, dVdz=dVdz, dWdz=dWdz, dRHOdz=dRHOdz, dTdz=dTdz,
            d2Udr2=d2Udr2, d2Vdr2=d2Vdr2, d2Wdr2=d2Wdr2, d2Tdr2=d2Tdr2,
            d2Udz2=d2Udz2, d2Vdz2=d2Vdz2, d2Wdz2=d2Wdz2, d2Tdz2=d2Tdz2,
            d2Udrz=d2Udrz, d2Vdrz=d2Vdrz, d2Wdrz=d2Wdrz,
            dmudT=dmudT, d2mudT2=d2mudT2,
        )
        DR, D2R, DZ, D2Z, D2RZ = _create_axy_diff_matrices(mesh, m)

    else:
        raise ValueError("model must be '2D' or 'axy'")

    if floquet_exp != 0.0:
        alpha = 1j * floquet_exp
        DZ, D2Z = _apply_floquet_shift(DZ, D2Z, alpha, ngp)

    # --- Assemble the 5*ngp x 5*ngp operator ---
    # L0[i*ngp+p, j*ngp+q] = A0[i,j,p]*delta(p,q)
    #                       + Ay[i,j,p]*DR[p,q]
    #                       + Ax[i,j,p]*DZ[p,q]
    #                       + Ayy[i,j,p]*D2R[p,q]
    #                       + Axx[i,j,p]*D2Z[p,q]
    #                       + Axy[i,j,p]*D2RZ[p,q]
    L0 = _assemble_block_operator(ngp, coeffs, DR, D2R, DZ, D2Z, D2RZ)

    # --- Apply inv(B) and sign flip ---
    # B = diag(1i, 1i*RHO, 1i*RHO, 1i*RHO, 1i*RHO) blockwise
    # L0_final = inv(B/1i) @ LHS = [1, 1/RHO, 1/RHO, 1/RHO, 1/RHO] blockwise
    # Model-wide sign flip from GetLinProblem convention: multiply by -1
    # Combined: multiply row block i by {-1, -1/RHO, -1/RHO, -1/RHO, -1/RHO}
    inv_B_signs = np.empty(5 * ngp)
    inv_B_signs[0*ngp:1*ngp] = -1.0
    inv_B_signs[1*ngp:2*ngp] = -1.0 / RHO
    inv_B_signs[2*ngp:3*ngp] = -1.0 / RHO
    inv_B_signs[3*ngp:4*ngp] = -1.0 / RHO
    inv_B_signs[4*ngp:5*ngp] = -1.0 / RHO

    L0 = sp.diags(inv_B_signs) @ L0

    # --- Add sponge ---
    if mesh.sponge is not None:
        sponge = mesh.sponge.ravel(order="F")[p]
        sponge_diag = np.tile(sponge, 5)
        L0 = L0 - sp.diags(sponge_diag)

    L0 = L0.tocsc()

    # --- Build index structure ---
    idx = _build_index_structure(mesh.idx, ngp)

    return L0, idx


def _assemble_block_operator(
    ngp: int,
    coeffs: LNSECoefficients,
    DR: sp.spmatrix,
    D2R: sp.spmatrix,
    DZ: sp.spmatrix,
    D2Z: sp.spmatrix,
    D2RZ: sp.spmatrix,
) -> sp.csc_matrix:
    """Assemble the block operator from coefficient arrays and FD matrices.

    For each (i,j) variable pair, the (i,j) block of size ngp×ngp is:
        A0[i,j,:]*I + Ay[i,j,:]*DR + Ax[i,j,:]*DZ
        + Ayy[i,j,:]*D2R + Axx[i,j,:]*D2Z + Axy[i,j,:]*D2RZ
    where the coefficient vectors multiply the rows (left-multiply by diag).
    """
    blocks = [[None] * 5 for _ in range(5)]
    I_ngp = sp.eye(ngp, format="csc")
    DR_blocks = _ensure_5_block_derivatives(DR, ngp)
    D2R_blocks = _ensure_5_block_derivatives(D2R, ngp)
    DZ_blocks = _ensure_5_block_derivatives(DZ, ngp)
    D2Z_blocks = _ensure_5_block_derivatives(D2Z, ngp)
    D2RZ_blocks = _ensure_5_block_derivatives(D2RZ, ngp)

    for i in range(5):
        for j in range(5):
            a0 = coeffs.A0[i, j, :]
            ax = coeffs.Ax[i, j, :]
            ay = coeffs.Ay[i, j, :]
            axx = coeffs.Axx[i, j, :]
            ayy = coeffs.Ayy[i, j, :]
            axy = coeffs.Axy[i, j, :]

            # Check if block is entirely zero
            if (np.count_nonzero(a0) == 0 and np.count_nonzero(ax) == 0 and
                np.count_nonzero(ay) == 0 and np.count_nonzero(axx) == 0 and
                np.count_nonzero(ayy) == 0 and np.count_nonzero(axy) == 0):
                blocks[i][j] = sp.csc_matrix((ngp, ngp))
                continue

            block = sp.csc_matrix((ngp, ngp))

            if np.any(a0):
                block = block + sp.diags(a0) @ I_ngp
            if np.any(ay):
                block = block + sp.diags(ay) @ DR_blocks[j]
            if np.any(ax):
                block = block + sp.diags(ax) @ DZ_blocks[j]
            if np.any(ayy):
                block = block + sp.diags(ayy) @ D2R_blocks[j]
            if np.any(axx):
                block = block + sp.diags(axx) @ D2Z_blocks[j]
            if np.any(axy):
                block = block + sp.diags(axy) @ D2RZ_blocks[j]

            blocks[i][j] = block

    return sp.bmat(blocks, format="csc")


def _build_index_structure(mesh_idx: dict, ngp: int) -> dict:
    """Build the variable/boundary index structure for DOFs.

    Creates index arrays mapping boundary locations to DOF indices in the
    5*ngp state vector q = [rho, u, v, w, T].
    """
    idx = dict(mesh_idx)  # copy boundary indices

    variables = ["rho", "u", "v", "w", "T"]
    var_offsets = {v: i * ngp for i, v in enumerate(variables)}

    # Variable DOF ranges (full index arrays)
    for v in variables:
        idx[f"{v}_j"] = np.arange(var_offsets[v], var_offsets[v] + ngp)

    # Border-variable indices
    borders = {"b": "bi", "t": "ti", "l": "li", "r": "ri"}
    for b_char, b_key in borders.items():
        if b_key not in idx:
            continue
        grid_idx = idx[b_key]
        all_border = []
        for v in variables:
            border_var_key = f"{b_key}_{v}"
            dof_idx = grid_idx + var_offsets[v]
            idx[border_var_key] = dof_idx
            all_border.append(dof_idx)
        idx[f"{b_key}_all"] = np.concatenate(all_border)

    return idx


def _ensure_5_block_derivatives(
    op, ngp: int
) -> list[sp.csc_matrix]:
    """Return axis-specific derivative operators as a list of 5 blocks."""
    if isinstance(op, (list, tuple)):
        if len(op) != 5:
            raise ValueError("Expected five derivative blocks for axy mode")
        return [sp.csc_matrix(mat) for mat in op]

    if not sp.issparse(op):
        raise TypeError("Derivative operator must be a sparse matrix or 5-block list/tuple")

    mat = op.tocsc()
    return [mat for _ in range(5)]


def _apply_floquet_shift(
    DZ: list[sp.csc_matrix] | sp.spmatrix,
    D2Z: list[sp.spmatrix] | sp.spmatrix,
    alpha: float,
    ngp: int,
) -> tuple[list[sp.csc_matrix], list[sp.csc_matrix]]:
    """Shift axial derivatives with Floquet parameter `alpha`.

    Implements:
    DZ -> DZ + i alpha I
    D2Z -> D2Z + 2 i alpha DZ + (i alpha)^2 I
    """
    dz_blocks = _ensure_5_block_derivatives(DZ, ngp)
    d2z_blocks = _ensure_5_block_derivatives(D2Z, ngp)
    I = sp.eye(ngp, format="csc")
    shifted_dz = [mat + alpha * I for mat in dz_blocks]
    shifted_d2z = [d2z + 2.0 * alpha * dz + (alpha * alpha) * I for d2z, dz in zip(d2z_blocks, dz_blocks)]
    return shifted_dz, shifted_d2z


def _create_axy_diff_matrices(
    mesh: Mesh,
    m: float,
) -> tuple[
    tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix],
    tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix],
    tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix],
    tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix],
    tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix],
]:
    """Create radial/axial derivative blocks for the axy model.

    Parity selection follows MATLAB ``CreateDiffMatrices_Axy.m``, converted
    from coefficient ordering [rho, v_r, v_theta, v_z, T] to physical
    ordering [rho, u(axial), v(radial), w(azimuthal), T]:

    - m = 0: [symm, symm, asymm, asymm, symm]  (rho,u,T even; v,w odd)
    - m = 1: [asymm, asymm, symm, symm, asymm]  (v,w even; rho,u,T odd)
    - m >= 2: all asymm
    """
    if not np.isfinite(m):
        raise ValueError("m must be finite")

    m_int = int(np.rint(m))
    if m_int < 0 or not np.isclose(m, m_int):
        raise ValueError("m must be a non-negative integer mode number")

    if m_int == 0:
        parity = (True, True, False, False, True)
    elif m_int == 1:
        parity = (False, False, True, True, False)
    else:
        parity = (False, False, False, False, False)

    Dx = mesh.Dx.tocsc()
    D2x = mesh.D2x.tocsc()
    Dy = mesh.Dy.tocsc()
    D2y = mesh.D2y.tocsc()
    Dyx = mesh.Dyx.tocsc() if mesh.Dyx is not None else (mesh.Dy @ mesh.Dx).tocsc()

    DR_blocks: list[sp.csc_matrix] = []
    D2R_blocks: list[sp.csc_matrix] = []
    D2RZ_blocks: list[sp.csc_matrix] = []

    for use_symmetry in parity:
        if use_symmetry and mesh.y_symmetry and mesh.Dy_symm is not None and mesh.D2y_symm is not None:
            DR_blocks.append(mesh.Dy_symm.tocsc())
            D2R_blocks.append(mesh.D2y_symm.tocsc())
            if mesh.Dyx_symm is not None:
                D2RZ_blocks.append(mesh.Dyx_symm.tocsc())
            else:
                D2RZ_blocks.append((mesh.Dy_symm @ Dx).tocsc())
        elif mesh.y_symmetry and not use_symmetry and mesh.Dy_asymm is not None and mesh.D2y_asymm is not None:
            DR_blocks.append(mesh.Dy_asymm.tocsc())
            D2R_blocks.append(mesh.D2y_asymm.tocsc())
            if mesh.Dyx_asymm is not None:
                D2RZ_blocks.append(mesh.Dyx_asymm.tocsc())
            else:
                D2RZ_blocks.append((mesh.Dy_asymm @ Dx).tocsc())
        else:
            DR_blocks.append(Dy)
            D2R_blocks.append(D2y)
            D2RZ_blocks.append(Dyx)

    DZ_blocks = (Dx, Dx, Dx, Dx, Dx)
    D2Z_blocks = (D2x, D2x, D2x, D2x, D2x)

    return (
        tuple(DR_blocks),
        tuple(D2R_blocks),
        DZ_blocks,
        D2Z_blocks,
        tuple(D2RZ_blocks),
    )
