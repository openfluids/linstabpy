"""LNSE coefficient matrices for Cartesian compressible flows.

Translates ``getCoeffsLaminar_Cartesian.m`` from the MATLAB LinStab2D
codebase, absorbing the coordinate-system and variable-ordering
transformations from ``GetLinProblem.m``.

The MATLAB code uses an internal coordinate system (x-spanwise,
y-wall-normal, z-streamwise) that differs from the physical one
(x-streamwise, y-wall-normal, z-spanwise).  All inputs and outputs of
:func:`compute_cartesian_coefficients` use the *physical* coordinate
system, so callers never need to worry about the internal mapping.

Physical coordinate system::

           +y (wall-normal)
           |
           |______+x (streamwise)
          /
         /
        +z (spanwise, Fourier mode)

Variable ordering: q = [rho, u_x, v_y, w_z, T].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LNSECoefficients:
    """LNSE coefficient arrays in the physical coordinate system.

    Each array has shape ``(5, 5, ngp)``.  Entry ``[i, j, :]`` is a
    pointwise (diagonal) coefficient that multiplies the *j*-th
    perturbation variable in the *i*-th equation.

    The LNSE system is:

    .. math::

        B \\frac{\\partial q}{\\partial t}
        = \\bigl(A_0 + A_x D_x + A_y D_y
          + A_{xx} D_{xx} + A_{yy} D_{yy} + A_{xy} D_{xy}\\bigr) q

    Attributes
    ----------
    A0 : np.ndarray
        Pointwise coefficient (no spatial derivative).
    Ax : np.ndarray
        Multiplies the streamwise first derivative D_x.
    Ay : np.ndarray
        Multiplies the wall-normal first derivative D_y.
    Axx : np.ndarray
        Multiplies the streamwise second derivative D_{xx}.
    Ayy : np.ndarray
        Multiplies the wall-normal second derivative D_{yy}.
    Axy : np.ndarray
        Multiplies the mixed derivative D_{xy}.
    B : np.ndarray
        Time-derivative (LHS) coefficient.
    """

    A0: np.ndarray
    Ax: np.ndarray
    Ay: np.ndarray
    Axx: np.ndarray
    Ayy: np.ndarray
    Axy: np.ndarray
    B: np.ndarray


_SOURCE_TO_DEST = {
    "A0": "A0",
    "Ar": "Ay",
    "Az": "Ax",
    "Arr": "Ayy",
    "Azz": "Axx",
    "Arz": "Axy",
    "B": "B",
}


def _expected_coefficient_keys() -> list[str]:
    keys: list[str] = []
    for source_name in _SOURCE_TO_DEST:
        for ic in range(5):
            for jc in range(5):
                keys.append(f"c_{source_name}_{ic+1}{jc+1}")
    return keys


def _pack_coefficients_in_physical_order(
    local_vars: dict[str, object],
    *,
    ngp: int,
    c2p_idx: list[int],
    c2p_sgn: np.ndarray,
    source_signs: dict[str, complex] | None = None,
) -> LNSECoefficients:
    """Pack local ``c_<block>_<ij>`` arrays directly into physical ordering.

    This avoids the intermediate ``coeff_blocks -> result`` duplication while
    preserving the public ``LNSECoefficients`` tensor layout.
    """
    missing = [name for name in _expected_coefficient_keys() if name not in local_vars]
    if missing:
        preview = ", ".join(missing[:8])
        if len(missing) > 8:
            preview += ", ..."
        raise KeyError(
            "Missing coefficient locals for physical-order packing: "
            f"{preview} ({len(missing)} missing)"
        )

    out = {
        name: np.zeros((5, 5, ngp), dtype=complex)
        for name in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B")
    }
    signs = source_signs or {}

    for source_name, dest_name in _SOURCE_TO_DEST.items():
        dst = out[dest_name]
        source_sign = complex(signs.get(source_name, 1.0))
        for ic in range(5):
            ip = c2p_idx[ic]
            row_sign = source_sign * complex(c2p_sgn[ic])
            for jc in range(5):
                jp = c2p_idx[jc]
                dst[ip, jp, :] = row_sign * complex(c2p_sgn[jc]) * local_vars[f"c_{source_name}_{ic+1}{jc+1}"]

    return LNSECoefficients(
        A0=out["A0"],
        Ax=out["Ax"],
        Ay=out["Ay"],
        Axx=out["Axx"],
        Ayy=out["Ayy"],
        Axy=out["Axy"],
        B=out["B"],
    )


def compute_cartesian_coefficients(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    RHO: np.ndarray,
    T: np.ndarray,
    MU: np.ndarray,
    P: np.ndarray,
    kx: float,
    Re: float,
    Ma: float,
    Pr: float,
    kappa: float,
    dUdx: np.ndarray,
    dVdx: np.ndarray,
    dWdx: np.ndarray,
    dRHOdx: np.ndarray,
    dTdx: np.ndarray,
    dPdx: np.ndarray,
    dUdy: np.ndarray,
    dVdy: np.ndarray,
    dWdy: np.ndarray,
    dRHOdy: np.ndarray,
    dTdy: np.ndarray,
    dPdy: np.ndarray,
    d2Udx2: np.ndarray,
    d2Vdx2: np.ndarray,
    d2Wdx2: np.ndarray,
    d2Udy2: np.ndarray,
    d2Vdy2: np.ndarray,
    d2Wdy2: np.ndarray,
    d2Tdx2: np.ndarray,
    d2Tdy2: np.ndarray,
    d2Udxdy: np.ndarray,
    d2Vdxdy: np.ndarray,
    d2Wdxdy: np.ndarray,
    dmudT: np.ndarray,
    d2mudT2: np.ndarray,
) -> LNSECoefficients:
    r"""Compute LNSE coefficient matrices for a Cartesian 2-D base flow.

    All inputs use the physical coordinate system (x = streamwise,
    y = wall-normal, z = spanwise).  The returned coefficients are in
    physical variable ordering ``[rho, u, v, w, T]`` and physical
    derivative naming (``Ax`` = streamwise, ``Ay`` = wall-normal).

    Parameters
    ----------
    U, V, W : np.ndarray, shape ``(ngp,)``
        Streamwise, wall-normal, and spanwise base-flow velocity.
    RHO, T, MU, P : np.ndarray, shape ``(ngp,)``
        Density, temperature, viscosity, and pressure.
    kx : float
        Streamwise Fourier wavenumber (physical x-direction).
    Re, Ma, Pr, kappa : float
        Reynolds number, Mach number, Prandtl number, and heat-capacity
        ratio.
    dUdx, dVdx, ... : np.ndarray, shape ``(ngp,)``
        Base-flow spatial derivatives (x = streamwise, y = wall-normal).
    dmudT, d2mudT2 : np.ndarray, shape ``(ngp,)``
        First and second derivatives of viscosity with respect to
        temperature.

    Returns
    -------
    LNSECoefficients
        Coefficient arrays ``A0, Ax, Ay, Axx, Ayy, Axy, B``, each of
        shape ``(5, 5, ngp)``.

    Notes
    -----
    Internally, the function maps physical quantities to the MATLAB
    coefficient coordinate system (x_coeff = -z_phys, y_coeff = y_phys,
    z_coeff = x_phys) used in ``getCoeffsLaminar_Cartesian.m``, computes
    all 175 coefficient entries, then maps the results back to the
    physical variable ordering.  The ``-LHS`` sign flip from
    ``GetLinProblem.m`` (line 228) and the permutation matrix ``T``
    (lines 272--285) are both absorbed.
    """
    ngp = U.shape[0]

    # ------------------------------------------------------------------
    # Step 1: Map physical quantities to coefficient coordinate system.
    #
    # Coefficient CS:  x_c = -z_phys (spanwise, flipped)
    #                  y_c = y_phys  (wall-normal)
    #                  z_c = x_phys  (streamwise)
    #
    # Velocities:  U_c = -W_phys,  V_c = V_phys,  W_c = U_phys
    # Wavenumber:  kz_c = -kx_phys
    # Derivatives: r = y,  z = x  (in coefficient notation)
    # ------------------------------------------------------------------
    Uc = -W
    Vc = V
    Wc = U
    kz = -kx

    # Derivative mappings: dr = dy, dz = dx in physical space
    # Sign flips for U_c = -W_phys derivatives
    dUdr = -dWdy
    dUdz = -dWdx
    d2Udr2 = -d2Wdy2
    d2Udz2 = -d2Wdx2
    d2Udrz = -d2Wdxdy

    # V_c = V_phys (no sign flip)
    dVdr = dVdy
    dVdz = dVdx
    d2Vdr2 = d2Vdy2
    d2Vdz2 = d2Vdx2
    d2Vdrz = d2Vdxdy

    # W_c = U_phys (no sign flip)
    dWdr = dUdy
    dWdz = dUdx
    d2Wdr2 = d2Udy2
    d2Wdz2 = d2Udx2
    d2Wdrz = d2Udxdy

    # Scalars: no sign flip, just r=y, z=x
    dRHOdr = dRHOdy
    dRHOdz = dRHOdx
    dTdr = dTdy
    dTdz = dTdx
    d2Tdr2 = d2Tdy2
    d2Tdz2 = d2Tdx2
    dPdr = dPdy
    dPdz = dPdx

    # Rename viscosity derivatives to match MATLAB variable names
    dMUdT = dmudT
    d2MUdT2 = d2mudT2

    # ------------------------------------------------------------------
    # Step 2: Compute all coefficient entries.
    #
    # These formulas are transcribed verbatim from
    # getCoeffsLaminar_Cartesian.m.  Variable names (U, V, W, kz, r, z)
    # refer to the coefficient coordinate system.
    #
    # C = 0 in the Cartesian formulation (curvature term).
    # ------------------------------------------------------------------
    Z = np.zeros(ngp)
    I = np.ones(ngp)

    # ---- Row 1: continuity equation ------------------------------------
    c_A0_11 = 1j * kz * Uc + dVdr + dWdz
    c_A0_12 = 1j * kz * RHO
    c_A0_13 = dRHOdr
    c_A0_14 = dRHOdz
    c_A0_15 = Z
    c_Ar_11 = Vc
    c_Ar_12 = Z
    c_Ar_13 = RHO
    c_Ar_14 = Z
    c_Ar_15 = Z
    c_Az_11 = Wc   # C=0, so -C+W = W
    c_Az_12 = Z
    c_Az_13 = Z
    c_Az_14 = RHO
    c_Az_15 = Z
    c_Arz_11 = Z
    c_Arz_12 = Z
    c_Arz_13 = Z
    c_Arz_14 = Z
    c_Arz_15 = Z
    c_Arr_11 = Z
    c_Arr_12 = Z
    c_Arr_13 = Z
    c_Arr_14 = Z
    c_Arr_15 = Z
    c_Azz_11 = Z
    c_Azz_12 = Z
    c_Azz_13 = Z
    c_Azz_14 = Z
    c_Azz_15 = Z
    c_B_11 = 1j * I
    c_B_12 = Z
    c_B_13 = Z
    c_B_14 = Z
    c_B_15 = Z

    # ---- Row 2: U_c momentum (spanwise in physical space, flipped) -----
    # C = 0 throughout
    c_A0_21 = (
        -(RHO * 0 * dUdz - RHO * Vc * dUdr
          - RHO * Wc * dUdz - 1j * P * kz) / RHO
    )
    c_A0_22 = (
        kz * (4.0 * MU * kz + 3.0 * 1j * RHO * Uc * Re) / Re
    ) / 3.0
    c_A0_23 = (RHO * dUdr * Re - 1j * dMUdT * dTdr * kz) / Re
    c_A0_24 = (RHO * dUdz * Re - 1j * dMUdT * dTdz * kz) / Re
    c_A0_25 = -(
        (3.0 * dMUdT * d2Udr2 * T
         + 3.0 * dMUdT * d2Udz2 * T
         - 3.0 * 1j * P * kz * Re
         + 3.0 * d2MUdT2 * dTdr * dUdr * T
         + 3.0 * d2MUdT2 * dTdz * dUdz * T
         - 2.0 * 1j * dMUdT * kz * dVdr * T
         - 2.0 * 1j * dMUdT * kz * dWdz * T)
        / Re / T
    ) / 3.0

    c_Ar_21 = Z
    c_Ar_22 = (-dMUdT * dTdr + RHO * Vc * Re) / Re
    c_Ar_23 = -1j * kz * MU / Re / 3.0
    c_Ar_24 = Z
    c_Ar_25 = -1.0 / Re * dMUdT * dUdr

    c_Az_21 = Z
    c_Az_22 = -(dMUdT * dTdz - RHO * Wc * Re) / Re  # C=0
    c_Az_23 = Z
    c_Az_24 = -1j * kz * MU / Re / 3.0
    c_Az_25 = -1.0 / Re * dMUdT * dUdz

    c_Arz_21 = Z
    c_Arz_22 = Z
    c_Arz_23 = Z
    c_Arz_24 = Z
    c_Arz_25 = Z

    c_Arr_21 = Z
    c_Arr_22 = -1.0 / Re * MU
    c_Arr_23 = Z
    c_Arr_24 = Z
    c_Arr_25 = Z

    c_Azz_21 = Z
    c_Azz_22 = -1.0 / Re * MU
    c_Azz_23 = Z
    c_Azz_24 = Z
    c_Azz_25 = Z

    c_B_21 = Z
    c_B_22 = 1j * RHO
    c_B_23 = Z
    c_B_24 = Z
    c_B_25 = Z

    # ---- Row 3: V_c momentum (wall-normal) -----------------------------
    c_A0_31 = (
        -(P * dRHOdr
          + 0 * dVdz * RHO**2          # C=0
          - Vc * dVdr * RHO**2
          - Wc * dVdz * RHO**2
          - dPdr * RHO)
        / RHO**2
    )
    c_A0_32 = 2.0 / 3.0 * 1j * dTdr * kz * dMUdT / Re
    c_A0_33 = (
        MU * kz**2 + RHO * dVdr * Re + 1j * RHO * kz * Uc * Re
    ) / Re
    c_A0_34 = RHO * dVdz
    c_A0_35 = -(
        (3.0 * P * dTdr * Re
         + 4.0 * dMUdT * d2Vdr2 * T**2
         + dMUdT * d2Wdrz * T**2
         + 3.0 * dMUdT * d2Vdz2 * T**2
         - 3.0 * dPdr * T * Re
         + 4.0 * d2MUdT2 * dTdr * dVdr * T**2
         - 2.0 * d2MUdT2 * dTdr * dWdz * T**2
         + 3.0 * d2MUdT2 * dTdz * dVdz * T**2
         + 3.0 * d2MUdT2 * dTdz * dWdr * T**2
         + 3.0 * 1j * dMUdT * kz * dUdr * T**2)
        / T**2 / Re
    ) / 3.0

    c_Ar_31 = P / RHO
    c_Ar_32 = -1j * kz * MU / Re / 3.0
    c_Ar_33 = -(
        (4.0 * dMUdT * dTdr - 3.0 * RHO * Vc * Re) / Re
    ) / 3.0
    c_Ar_34 = -1.0 / Re * dMUdT * dTdz
    c_Ar_35 = (
        (-4.0 * dMUdT * dVdr * T
         + 2.0 * dMUdT * dWdz * T
         + 3.0 * P * Re)
        / Re / T
    ) / 3.0

    c_Az_31 = Z
    c_Az_32 = Z
    c_Az_33 = -(dMUdT * dTdz - RHO * Wc * Re) / Re  # C=0
    c_Az_34 = 2.0 / 3.0 * dTdr * dMUdT / Re
    c_Az_35 = -dMUdT * (dVdz + dWdr) / Re

    c_Arz_31 = Z
    c_Arz_32 = Z
    c_Arz_33 = Z
    c_Arz_34 = -MU / Re / 3.0
    c_Arz_35 = Z

    c_Arr_31 = Z
    c_Arr_32 = Z
    c_Arr_33 = -4.0 / 3.0 * MU / Re
    c_Arr_34 = Z
    c_Arr_35 = Z

    c_Azz_31 = Z
    c_Azz_32 = Z
    c_Azz_33 = -MU / Re
    c_Azz_34 = Z
    c_Azz_35 = Z

    c_B_31 = Z
    c_B_32 = Z
    c_B_33 = 1j * RHO
    c_B_34 = Z
    c_B_35 = Z

    # ---- Row 4: W_c momentum (streamwise in physical space) ------------
    c_A0_41 = (
        -(P * dRHOdz
          + 0 * dWdz * RHO**2           # C=0
          - Vc * dWdr * RHO**2
          - Wc * dWdz * RHO**2
          - dPdz * RHO)
        / RHO**2
    )
    c_A0_42 = 2.0 / 3.0 * 1j * dTdz * kz * dMUdT / Re
    c_A0_43 = RHO * dWdr
    c_A0_44 = (
        1j * RHO * kz * Uc * Re + MU * kz**2 + RHO * dWdz * Re
    ) / Re
    c_A0_45 = -(
        (3.0 * d2MUdT2 * dTdr * dVdz * T**2
         + 3.0 * d2MUdT2 * dTdr * dWdr * T**2
         + 4.0 * d2MUdT2 * dTdz * dWdz * T**2
         - 2.0 * d2MUdT2 * dTdz * dVdr * T**2
         + 3.0 * 1j * dMUdT * kz * dUdz * T**2
         + 3.0 * P * dTdz * Re
         + dMUdT * d2Vdrz * T**2
         + 3.0 * dMUdT * d2Wdr2 * T**2
         + 4.0 * dMUdT * d2Wdz2 * T**2
         - 3.0 * dPdz * Re * T)
        / Re / T**2
    ) / 3.0

    c_Ar_41 = Z
    c_Ar_42 = Z
    c_Ar_43 = 2.0 / 3.0 * dTdz * dMUdT / Re
    c_Ar_44 = -(dMUdT * dTdr - RHO * Vc * Re) / Re
    c_Ar_45 = -dMUdT * (dVdz + dWdr) / Re

    c_Az_41 = P / RHO
    c_Az_42 = -1j * kz * MU / Re / 3.0
    c_Az_43 = -1.0 / Re * dMUdT * dTdr
    c_Az_44 = -(
        (4.0 * dMUdT * dTdz - 3.0 * RHO * Wc * Re) / Re  # C=0
    ) / 3.0
    c_Az_45 = (
        (-4.0 * dMUdT * dWdz * T
         + 2.0 * dMUdT * dVdr * T
         + 3.0 * P * Re)
        / Re / T
    ) / 3.0

    c_Arz_41 = Z
    c_Arz_42 = Z
    c_Arz_43 = -MU / Re / 3.0
    c_Arz_44 = Z
    c_Arz_45 = Z

    c_Arr_41 = Z
    c_Arr_42 = Z
    c_Arr_43 = Z
    c_Arr_44 = -MU / Re
    c_Arr_45 = Z

    c_Azz_41 = Z
    c_Azz_42 = Z
    c_Azz_43 = Z
    c_Azz_44 = -4.0 / 3.0 * MU / Re
    c_Azz_45 = Z

    c_B_41 = Z
    c_B_42 = Z
    c_B_43 = Z
    c_B_44 = 1j * RHO
    c_B_45 = Z

    # ---- Row 5: energy equation ----------------------------------------
    c_A0_51 = (
        -(-kappa**2 * Ma**2 * P * dVdr
          - kappa**2 * Ma**2 * P * dWdz
          + kappa * Ma**2 * P * dVdr
          + kappa * Ma**2 * P * dWdz
          - RHO * Vc * dTdr       # C=0: +C*dTdz*RHO vanishes
          - RHO * Wc * dTdz) / RHO
    )
    c_A0_52 = (
        1j * kappa * Ma**2 * kz
        * (3.0 * kappa * P * Re - 3.0 * P * Re
           + 4.0 * kappa * MU * dVdr + 4.0 * kappa * MU * dWdz
           - 4.0 * MU * dVdr - 4.0 * MU * dWdz)
        / Re
    ) / 3.0
    c_A0_53 = -(
        2.0 * 1j * kappa**2 * Ma**2 * kz * MU * dUdr
        - 2.0 * 1j * kappa * Ma**2 * kz * MU * dUdr
        - RHO * dTdr * Re
    ) / Re
    c_A0_54 = -(
        2.0 * 1j * kappa**2 * Ma**2 * kz * MU * dUdz
        - 2.0 * 1j * kappa * Ma**2 * kz * MU * dUdz
        - RHO * dTdz * Re
    ) / Re
    c_A0_55 = (
        (4.0 * kappa * Ma**2 * dVdr**2 * dMUdT * T * Pr
         - 3.0 * kappa**2 * Ma**2 * dMUdT * dWdr**2 * T * Pr
         - 3.0 * kappa**2 * Ma**2 * dMUdT * dVdz**2 * T * Pr
         - 3.0 * kappa**2 * Ma**2 * dMUdT * dUdr**2 * T * Pr
         - 3.0 * kappa**2 * Ma**2 * dMUdT * dUdz**2 * T * Pr
         - 4.0 * kappa**2 * Ma**2 * dWdz**2 * dMUdT * T * Pr
         - 3.0 * kappa * d2MUdT2 * dTdr**2 * T
         - 3.0 * kappa * dMUdT * d2Tdr2 * T
         - 3.0 * kappa * d2MUdT2 * dTdz**2 * T
         - 3.0 * kappa * dMUdT * d2Tdz2 * T
         + 3.0 * kappa * MU * kz**2 * T
         - 4.0 * kappa**2 * Ma**2 * dVdr**2 * dMUdT * T * Pr
         + 3.0 * kappa * Ma**2 * dMUdT * dWdr**2 * T * Pr
         + 3.0 * kappa * Ma**2 * dMUdT * dVdz**2 * T * Pr
         + 3.0 * kappa * Ma**2 * dMUdT * dUdr**2 * T * Pr
         + 3.0 * kappa * Ma**2 * dMUdT * dUdz**2 * T * Pr
         + 3.0 * kappa**2 * Ma**2 * P * dVdr * Re * Pr
         + 3.0 * kappa**2 * Ma**2 * P * dWdz * Re * Pr
         - 3.0 * kappa * Ma**2 * P * dWdz * Re * Pr
         + 4.0 * kappa * Ma**2 * dWdz**2 * dMUdT * T * Pr
         - 3.0 * kappa * Ma**2 * P * dVdr * Re * Pr
         - 4.0 * kappa * Ma**2 * dMUdT * dVdr * dWdz * T * Pr
         - 6.0 * kappa**2 * Ma**2 * dMUdT * dWdr * dVdz * T * Pr
         + 6.0 * kappa * Ma**2 * dMUdT * dWdr * dVdz * T * Pr
         + 4.0 * kappa**2 * Ma**2 * dMUdT * dVdr * dWdz * T * Pr
         + 3.0 * 1j * kz * RHO * Uc * Re * T * Pr)
        / Pr / T / Re
    ) / 3.0

    c_Ar_51 = Z
    c_Ar_52 = -2.0 * kappa * Ma**2 * MU * dUdr * (kappa - 1.0) / Re
    c_Ar_53 = (
        kappa * Ma**2
        * (-4.0 * MU * dWdz + 4.0 * kappa * MU * dWdz
           - 3.0 * P * Re + 3.0 * kappa * P * Re
           + 8.0 * MU * dVdr - 8.0 * kappa * MU * dVdr)
        / Re
    ) / 3.0
    c_Ar_54 = (
        -2.0 * kappa * Ma**2 * MU
        * (-dVdz + kappa * dVdz - dWdr + kappa * dWdr) / Re
    )
    c_Ar_55 = (
        -2.0 * kappa * dMUdT * dTdr + RHO * Vc * Re * Pr
    ) / Re / Pr

    c_Az_51 = Z
    c_Az_52 = -2.0 * kappa * Ma**2 * MU * dUdz * (kappa - 1.0) / Re
    c_Az_53 = (
        -2.0 * kappa * Ma**2 * MU
        * (-dVdz + kappa * dVdz - dWdr + kappa * dWdr) / Re
    )
    c_Az_54 = (
        kappa * Ma**2
        * (4.0 * kappa * MU * dVdr - 4.0 * MU * dVdr
           - 3.0 * P * Re + 3.0 * kappa * P * Re
           - 8.0 * kappa * MU * dWdz + 8.0 * MU * dWdz)
        / Re
    ) / 3.0
    c_Az_55 = (
        -(2.0 * kappa * dMUdT * dTdz - RHO * Wc * Re * Pr) / Re / Pr
    )  # C=0

    c_Arz_51 = Z
    c_Arz_52 = Z
    c_Arz_53 = Z
    c_Arz_54 = Z
    c_Arz_55 = Z

    c_Arr_51 = Z
    c_Arr_52 = Z
    c_Arr_53 = Z
    c_Arr_54 = Z
    c_Arr_55 = -kappa * MU / Re / Pr

    c_Azz_51 = Z
    c_Azz_52 = Z
    c_Azz_53 = Z
    c_Azz_54 = Z
    c_Azz_55 = -kappa * MU / Re / Pr

    c_B_51 = Z
    c_B_52 = Z
    c_B_53 = Z
    c_B_54 = Z
    c_B_55 = 1j * RHO

    # ------------------------------------------------------------------
    # Pack directly into the final physical-order tensors, folding
    # the MATLAB sign flip and variable permutation into one pass.
    #
    # Permutation matrix T from GetLinProblem.m lines 273-277:
    #   T = [I   0   0   0   0 ;   row 0: rho -> rho
    #        0   0   0   I   0 ;   row 1: u_phys <- W_c (coeff row 3)
    #        0   0   I   0   0 ;   row 2: v_phys <- V_c (coeff row 2)
    #        0  -I   0   0   0 ;   row 3: w_phys <- -U_c (coeff row 1)
    #        0   0   0   0   I];   row 4: T -> T
    #
    # L0_phys = T @ L0_coeff @ T'
    #
    # For a block matrix A_coeff[i_c, j_c]:
    #   A_phys[i_p, j_p] = s_i * s_j * A_coeff[i_c, j_c]
    #
    # where the mapping from coefficient to physical index is:
    #   coeff 0 (rho) -> phys 0 (rho), sign +1
    #   coeff 1 (U_c) -> phys 3 (w),   sign -1
    #   coeff 2 (V_c) -> phys 2 (v),   sign +1
    #   coeff 3 (W_c) -> phys 1 (u),   sign +1
    #   coeff 4 (T)   -> phys 4 (T),   sign +1
    # ------------------------------------------------------------------
    c2p_idx = [0, 3, 2, 1, 4]
    c2p_sgn = np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    return _pack_coefficients_in_physical_order(
        locals(),
        ngp=ngp,
        c2p_idx=c2p_idx,
        c2p_sgn=c2p_sgn,
        source_signs={
            "A0": -1.0,
            "Ar": -1.0,
            "Az": -1.0,
            "Arr": -1.0,
            "Azz": -1.0,
            "Arz": -1.0,
        },
    )


def compute_cylindrical_coefficients(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    R: np.ndarray,
    RHO: np.ndarray,
    T: np.ndarray,
    MU: np.ndarray,
    m: float,
    Re: float,
    Ma: float,
    Pr: float,
    kappa: float,
    dUdr: np.ndarray,
    dVdr: np.ndarray,
    dWdr: np.ndarray,
    dRHOdr: np.ndarray,
    dTdr: np.ndarray,
    dUdz: np.ndarray,
    dVdz: np.ndarray,
    dWdz: np.ndarray,
    dRHOdz: np.ndarray,
    dTdz: np.ndarray,
    d2Udr2: np.ndarray,
    d2Vdr2: np.ndarray,
    d2Wdr2: np.ndarray,
    d2Tdr2: np.ndarray,
    d2Udz2: np.ndarray,
    d2Vdz2: np.ndarray,
    d2Wdz2: np.ndarray,
    d2Tdz2: np.ndarray,
    d2Udrz: np.ndarray,
    d2Vdrz: np.ndarray,
    d2Wdrz: np.ndarray,
    dmudT: np.ndarray,
    d2mudT2: np.ndarray,
) -> LNSECoefficients:
    r"""Compute LNSE coefficient matrices for a cylindrical 2-D base flow.

    Inputs and coefficients use the physical coordinate system (x = axial,
    y = radius, z = azimuth, with Fourier mode ``m``).  The returned
    coefficients are in physical variable ordering ``[rho, u, v, w, T]``.

    Parameters
    ----------
    U, V, W : np.ndarray, shape ``(ngp,)``
        Radial, azimuthal, and axial base-flow velocity.
    R, RHO, T, MU : np.ndarray, shape ``(ngp,)``
        Radial coordinate, density, temperature, and viscosity.
    m : float
        Azimuthal mode number.
    Re, Ma, Pr, kappa : float
        Reynolds number, Mach number, Prandtl number, and heat-capacity
        ratio.
    derivatives : np.ndarray, shape ``(ngp,)``
        Spatial derivatives of base-flow fields in radial (``r``) and
        axial (``z``) directions.
    dmudT, d2mudT2 : np.ndarray, shape ``(ngp,)``
        First and second derivatives of viscosity with respect to
        temperature.

    Returns
    -------
    LNSECoefficients
        Coefficient arrays ``A0, Ax, Ay, Axx, Ayy, Axy, B`` each with
        shape ``(5, 5, ngp)``.
    """
    ngp = U.shape[0]

    # ------------------------------------------------------------------
    # Step 1: Build derived coefficients from physical thermodynamic inputs.
    # ------------------------------------------------------------------
    R = np.asarray(R, dtype=float)
    # Regularize the axis singularity using the local radial mesh spacing as
    # a floor.  The old hard-coded 1e-14 floor allowed 1/R² ~ 10²⁸ near the
    # axis; using the smallest positive local spacing caps the metric terms at
    # the mesh resolution.  Degenerate single-point/all-axis inputs use a small
    # numeric floor rather than the old near-zero singular value.
    if R.size > 1:
        diffs = np.diff(np.sort(R))
        positive_diffs = diffs[diffs > 0]
        eps_reg = float(positive_diffs.min()) if positive_diffs.size > 0 else 1e-8
    else:
        eps_reg = 1e-8
    R = np.maximum(R, eps_reg)
    c1 = (kappa - 1.0) * Re * Pr * Ma**2
    c2 = kappa * Ma**2
    cv = 1.0 / (kappa * (kappa - 1.0) * Ma**2)

    # ------------------------------------------------------------------
    # Step 2: Compute all coefficient entries.
    # ------------------------------------------------------------------
    Z = np.zeros(ngp)
    I = np.ones(ngp)

    c_A0_11 = -(U + R * dUdr + 1j * m * V + R * dWdz) / R
    c_A0_12 = -(RHO + R * dRHOdr) / R
    c_A0_13 = -1j / R * RHO * m
    c_A0_14 = -dRHOdz
    c_A0_15 = Z
    c_Ar_11 = -U
    c_Ar_12 = -RHO
    c_Ar_13 = Z
    c_Ar_14 = Z
    c_Ar_15 = Z
    c_Arr_11 = Z
    c_Arr_12 = Z
    c_Arr_13 = Z
    c_Arr_14 = Z
    c_Arr_15 = Z
    c_Az_11 = -W
    c_Az_12 = Z
    c_Az_13 = Z
    c_Az_14 = -RHO
    c_Az_15 = Z
    c_Azz_11 = Z
    c_Azz_12 = Z
    c_Azz_13 = Z
    c_Azz_14 = Z
    c_Azz_15 = Z
    c_Arz_11 = Z
    c_Arz_12 = Z
    c_Arz_13 = Z
    c_Arz_14 = Z
    c_Arz_15 = Z
    c_A0_21 = -(
        2 * dUdr * U * R * c2 + W * dUdz * R * c2 + dWdz * U * R * c2
        + 1j * m * U * V * c2 + dTdr * R - V**2 * c2 + U**2 * c2
    ) / R / c2
    c_A0_22 = -(
        (3 * MU * m**2 + 6 * RHO * U * Re * R + 6 * dRHOdr * U * Re * R**2
         + 6 * dUdr * RHO * Re * R**2 + 3 * W * dRHOdz * Re * R**2
         + 3 * dWdz * RHO * Re * R**2 + 2 * dmudT * dTdr * R
         + 3 * 1j * RHO * m * V * Re * R + 4 * MU) / Re / R**2
    ) / 0.3e1
    c_A0_23 = -(
        (2 * 1j * dmudT * dTdr * m * R - 6 * RHO * V * Re * R
         + 7 * 1j * MU * m + 3 * 1j * RHO * U * m * Re * R) / Re / R**2
    ) / 0.3e1
    c_A0_24 = -RHO * dUdz - dRHOdz * U
    c_A0_25 = -(
        (2 * R**2 * c2 * dWdz * d2mudT2 * dTdr + 2 * R * c2 * U * d2mudT2 * dTdr
         - 4 * R**2 * c2 * dUdr * d2mudT2 * dTdr - 3 * R**2 * c2 * dWdr * d2mudT2 * dTdz
         - 3 * R**2 * c2 * dUdz * d2mudT2 * dTdz + 3 * 1j * dmudT * m * V * c2
         - 3 * 1j * dmudT * m * dVdr * R * c2 + 3 * Re * R**2 * dRHOdr)
        / Re / R**2 / c2
    ) / 0.3e1
    c_Ar_21 = -(T + U**2 * c2) / c2
    c_Ar_22 = 0.2e1 / 0.3e1 * (2 * dmudT * dTdr * R - 3 * RHO * U * Re * R + 2 * MU) / Re / R
    c_Ar_23 = 1j / Re / R * MU * m / 0.3e1
    c_Ar_24 = 0.1e1 / Re * dmudT * dTdz
    c_Ar_25 = -(
        (2 * dWdz * dmudT * R * c2 + 2 * U * dmudT * c2 - 4 * dUdr * dmudT * R * c2
         + 3 * RHO * Re * R) / Re / R / c2
    ) / 0.3e1
    c_Arr_21 = Z
    c_Arr_22 = 0.4e1 / 0.3e1 * MU / Re
    c_Arr_23 = Z
    c_Arr_24 = Z
    c_Arr_25 = Z
    c_Az_21 = -W * U
    c_Az_22 = -(W * RHO * Re - dmudT * dTdz) / Re
    c_Az_23 = Z
    c_Az_24 = -((2 * dmudT * dTdr + 3 * RHO * U * Re) / Re) / 0.3e1
    c_Az_25 = dmudT * (dWdr + dUdz) / Re
    c_Azz_21 = Z
    c_Azz_22 = MU / Re
    c_Azz_23 = Z
    c_Azz_24 = Z
    c_Azz_25 = Z
    c_Arz_21 = Z
    c_Arz_22 = Z
    c_Arz_23 = Z
    c_Arz_24 = MU / Re / 0.3e1
    c_Arz_25 = Z
    c_A0_31 = -(
        1j * m * V**2 * c2 + 2 * V * U * c2 + W * dVdz * R * c2 + dWdz * V * R * c2
        + dVdr * U * R * c2 + V * dUdr * R * c2 + 1j * m * T
    ) / R / c2
    c_A0_32 = (
        (-6 * RHO * V * Re * R - 3 * V * dRHOdr * Re * R**2 - 3 * dVdr * RHO * Re * R**2
         + 7 * 1j * MU * m + 3 * 1j * dmudT * dTdr * m * R) / Re / R**2
    ) / 0.3e1
    c_A0_33 = -(
        (3 * MU + 3 * dmudT * dTdr * R + 6 * RHO * U * Re * R + 3 * W * dRHOdz * Re * R**2
         + 3 * dWdz * RHO * Re * R**2 + 3 * dRHOdr * U * Re * R**2 + 3 * RHO * dUdr * Re * R**2
         + 4 * MU * m**2 + 6 * 1j * RHO * V * m * Re * R) / Re / R**2
    ) / 0.3e1
    c_A0_34 = -(RHO * dVdz * Re * R + dRHOdz * V * Re * R - 1j * dmudT * dTdz * m) / Re / R
    c_A0_35 = -(
        (3 * 1j * RHO * m * Re * R + 3 * R * c2 * V * d2mudT2 * dTdr
         - 3 * R**2 * c2 * dVdr * d2mudT2 * dTdr - 3 * R**2 * c2 * dVdz * d2mudT2 * dTdz
         - 4 * 1j * dmudT * m * U * c2 + 2 * 1j * dmudT * m * dUdr * R * c2
         + 2 * 1j * dmudT * m * dWdz * R * c2) / Re / R**2 / c2
    ) / 0.3e1
    c_Ar_31 = -V * U
    c_Ar_32 = -((3 * RHO * V * Re * R - 1j * MU * m) / Re / R) / 0.3e1
    c_Ar_33 = (MU + dmudT * dTdr * R - RHO * U * Re * R) / Re / R
    c_Ar_34 = Z
    c_Ar_35 = -dmudT * (V - dVdr * R) / Re / R
    c_Arr_31 = Z
    c_Arr_32 = Z
    c_Arr_33 = MU / Re
    c_Arr_34 = Z
    c_Arr_35 = Z
    c_Az_31 = -W * V
    c_Az_32 = Z
    c_Az_33 = (dmudT * dTdz - W * RHO * Re) / Re
    c_Az_34 = -((3 * RHO * V * Re * R - 1j * MU * m) / Re / R) / 0.3e1
    c_Az_35 = 0.1e1 / Re * dVdz * dmudT
    c_Azz_31 = Z
    c_Azz_32 = Z
    c_Azz_33 = MU / Re
    c_Azz_34 = Z
    c_Azz_35 = Z
    c_Arz_31 = Z
    c_Arz_32 = Z
    c_Arz_33 = Z
    c_Arz_34 = Z
    c_Arz_35 = Z
    c_A0_41 = -(
        dTdz * R + W * U * c2 + W * dUdr * R * c2 + 2 * dWdz * W * R * c2
        + dWdr * U * R * c2 + 1j * m * V * W * c2
    ) / R / c2
    c_A0_42 = -(
        (2 * dmudT * dTdz + 3 * W * RHO * Re + 3 * dWdr * RHO * R * Re + 3 * W * dRHOdr * R * Re)
        / R / Re
    ) / 0.3e1
    c_A0_43 = -(
        1j * m * (2 * dmudT * dTdz + 3 * W * RHO * Re) / R / Re
    ) / 0.3e1
    c_A0_44 = -(
        MU * m**2 + RHO * U * R * Re + RHO * dUdr * R**2 * Re + 2 * dWdz * RHO * R**2 * Re
        + dRHOdr * U * R**2 * Re + 2 * dRHOdz * W * R**2 * Re + 1j * RHO * V * m * R * Re
    ) / R**2 / Re
    c_A0_45 = -(
        (3 * dRHOdz * R * Re + 2 * dUdr * d2mudT2 * dTdz * R * c2
         - 4 * dWdz * d2mudT2 * dTdz * R * c2 - 3 * dWdr * d2mudT2 * dTdr * R * c2
         - 3 * dUdz * d2mudT2 * dTdr * R * c2 + 2 * U * d2mudT2 * dTdz * c2
         - 3 * 1j * dmudT * m * dVdz * c2) / R / Re / c2
    ) / 0.3e1
    c_Ar_41 = -W * U
    c_Ar_42 = -((2 * dmudT * dTdz + 3 * W * RHO * Re) / Re) / 0.3e1
    c_Ar_43 = Z
    c_Ar_44 = -(-MU - dmudT * dTdr * R + RHO * U * R * Re) / R / Re
    c_Ar_45 = dmudT * (dWdr + dUdz) / Re
    c_Arr_41 = Z
    c_Arr_42 = Z
    c_Arr_43 = Z
    c_Arr_44 = MU / Re
    c_Arr_45 = Z
    c_Az_41 = -(T + W**2 * c2) / c2
    c_Az_42 = ((MU + 3 * dmudT * dTdr * R) / R / Re) / 0.3e1
    c_Az_43 = 1j / R / Re * MU * m / 0.3e1
    c_Az_44 = 0.2e1 / 0.3e1 * (2 * dmudT * dTdz - 3 * W * RHO * Re) / Re
    c_Az_45 = -(
        (3 * RHO * R * Re + 2 * dUdr * dmudT * R * c2 - 4 * dWdz * dmudT * R * c2 + 2 * U * dmudT * c2)
        / R / Re / c2
    ) / 0.3e1
    c_Azz_41 = Z
    c_Azz_42 = Z
    c_Azz_43 = Z
    c_Azz_44 = 0.4e1 / 0.3e1 * MU / Re
    c_Azz_45 = Z
    c_Arz_41 = Z
    c_Arz_42 = MU / Re / 0.3e1
    c_Arz_43 = Z
    c_Arz_44 = Z
    c_Arz_45 = Z
    c_A0_51 = -(
        (1j * m * V**3 * c2 + 2 * cv * T * U * c2 + V**2 * dWdz * R * c2 + 3 * U**2 * dUdr * R * c2
         + 3 * W**2 * dWdz * R * c2 + V**2 * dUdr * R * c2 + U**2 * dWdz * R * c2
         + W**2 * dUdr * R * c2 + 2 * 1j * V * m * T + 2 * T * U + U**3 * c2
         + 2 * 1j * V * m * cv * T * c2 + 2 * cv * dTdz * W * R * c2 + 2 * dUdz * U * W * R * c2
         + 2 * cv * T * dWdz * R * c2 + 2 * dVdz * V * W * R * c2 + 2 * dVdr * V * U * R * c2
         + 2 * dWdr * U * W * R * c2 + 2 * cv * T * dUdr * R * c2 + 2 * cv * dTdr * U * R * c2
         + 1j * V * m * U**2 * c2 + 1j * V * m * W**2 * c2 + 2 * dTdz * W * R + 2 * T * dUdr * R
         + 2 * dTdr * U * R + V**2 * U * c2 + W**2 * U * c2 + 2 * T * dWdz * R
        ) / R / c2
    ) / 0.2e1
    c_A0_52 = (
        (-6 * T * RHO * Re * R - 8 * MU * dWdz * c2 * R + 6 * MU * d2Udz2 * R**2 * c2
         - 6 * T * dRHOdr * Re * R**2 - 6 * dTdr * RHO * Re * R**2 + 8 * MU * d2Udr2 * R**2 * c2
         + 2 * MU * d2Wdrz * R**2 * c2 - 6 * U * MU * m**2 * c2 - 6 * cv * T * RHO * Re * c2 * R
         - 6 * W * dRHOdz * U * Re * R**2 * c2 - 18 * dUdr * RHO * U * Re * R**2 * c2
         - 6 * dWdz * RHO * U * Re * R**2 * c2 - 6 * dUdz * W * RHO * Re * R**2 * c2
         - 6 * dVdr * V * RHO * Re * R**2 * c2 - 6 * dWdr * W * RHO * Re * R**2 * c2
         - 6 * cv * T * dRHOdr * Re * R**2 * c2 - 6 * cv * dTdr * RHO * Re * R**2 * c2
         + 12 * 1j * dVdr * MU * m * R * c2 - 9 * U**2 * RHO * Re * c2 * R
         - 3 * V**2 * RHO * Re * c2 * R - 3 * W**2 * RHO * Re * c2 * R
         - 4 * W * dmudT * dTdz * c2 * R - 8 * U * dmudT * dTdr * c2 * R + 2 * 1j * V * MU * m * c2
         - 3 * W**2 * dRHOdr * Re * R**2 * c2 - 9 * dRHOdr * U**2 * Re * R**2 * c2
         - 3 * V**2 * dRHOdr * Re * R**2 * c2 + 6 * dUdz * dmudT * dTdz * R**2 * c2
         + 6 * dWdr * dmudT * dTdz * R**2 * c2 + 8 * dUdr * dmudT * dTdr * R**2 * c2
         - 4 * dWdz * dmudT * dTdr * R**2 * c2 + 6 * 1j * V * dmudT * dTdr * m * R * c2
         - 6 * 1j * U * RHO * V * m * Re * R * c2) / Re / R**2 / c2
    ) / 0.6e1
    c_A0_53 = -(
        (-6 * MU * d2Vdr2 * R**2 * c2 - 6 * MU * d2Vdz2 * R**2 * c2 + 6 * dVdr * MU * R * c2
         + 8 * 1j * dUdr * MU * m * R * c2 + 3 * 1j * m * RHO * W**2 * Re * R * c2
         + 3 * 1j * m * RHO * U**2 * Re * R * c2 + 6 * 1j * m * RHO * cv * T * Re * R * c2
         + 8 * 1j * m * MU * dWdz * R * c2 - 2 * 1j * U * MU * m * c2
         + 6 * 1j * m * RHO * T * Re * R + 4 * 1j * W * dmudT * dTdz * m * R * c2
         + 8 * V * MU * m**2 * c2 + 12 * V * dmudT * dTdr * R * c2 - 6 * dVdz * dmudT * dTdz * R**2 * c2
         - 6 * dVdr * dmudT * dTdr * R**2 * c2 + 6 * U * RHO * V * Re * R * c2
         + 6 * dWdz * RHO * V * Re * R**2 * c2 + 6 * U * dRHOdr * V * Re * R**2 * c2
         + 6 * W * dRHOdz * V * Re * R**2 * c2 + 6 * dUdr * RHO * V * Re * R**2 * c2
         + 6 * dVdz * W * RHO * Re * R**2 * c2 + 6 * dVdr * U * RHO * Re * R**2 * c2
         + 4 * 1j * U * dmudT * dTdr * m * R * c2 + 9 * 1j * m * RHO * V**2 * Re * R * c2
        ) / Re / R**2 / c2
    ) / 0.6e1
    c_A0_54 = (
        6 * MU * d2Wdr2 * R**2 * c2 - 6 * T * dRHOdz * Re * R**2 - 6 * dTdz * RHO * Re * R**2
        + 2 * MU * d2Udrz * R**2 * c2 + 8 * MU * d2Wdz2 * R**2 * c2 + 6 * MU * dWdr * R * c2
        + 2 * MU * dUdz * R * c2 - 6 * W * MU * m**2 * c2 - 18 * dWdz * RHO * W * Re * R**2 * c2
        - 6 * dUdr * RHO * W * Re * R**2 * c2 - 6 * U * dRHOdr * W * Re * R**2 * c2
        - 6 * cv * dTdz * RHO * Re * R**2 * c2 - 6 * dUdz * U * RHO * Re * R**2 * c2
        - 6 * cv * T * dRHOdz * Re * R**2 * c2 - 6 * dVdz * V * RHO * Re * R**2 * c2
        - 6 * dWdr * U * RHO * Re * R**2 * c2 - 6 * U * RHO * W * Re * R * c2
        + 12 * 1j * m * MU * dVdz * R * c2 - 9 * dRHOdz * W**2 * Re * R**2 * c2
        - 3 * V**2 * dRHOdz * Re * R**2 * c2 - 3 * U**2 * dRHOdz * Re * R**2 * c2
        + 6 * dUdz * dmudT * dTdr * R**2 * c2 + 6 * dWdr * dmudT * dTdr * R**2 * c2
        - 4 * dUdr * dmudT * dTdz * R**2 * c2 + 8 * dWdz * dmudT * dTdz * R**2 * c2
        - 4 * U * dmudT * dTdz * R * c2 + 6 * 1j * V * dmudT * dTdz * m * R * c2
        - 6 * 1j * W * RHO * V * m * Re * R * c2
    ) / Re / R**2 / c2 / 0.6e1
    c_A0_55 = -(
        (2 * R * c1 * c2 * U**2 * d2mudT2 * dTdr + 3 * R * c1 * c2 * V**2 * d2mudT2 * dTdr
         - 3 * Re * R**2 * c2 * d2mudT2 * dTdr**2 - 3 * Re * R**2 * c2 * d2mudT2 * dTdz**2
         + 3 * Re * R * c1 * U * RHO + 3 * Re * R**2 * c1 * dUdr * RHO
         + 3 * Re * R**2 * c1 * U * dRHOdr + 3 * Re * R**2 * c1 * dWdz * RHO
         + 3 * Re * R * c1 * c2 * U * RHO * cv + 3 * Re * R**2 * c1 * c2 * W * dRHOdz * cv
         + 3 * Re * R**2 * c1 * c2 * dWdz * RHO * cv + 3 * Re * R**2 * c1 * c2 * dUdr * RHO * cv
         + 3 * Re * R**2 * c1 * c2 * U * dRHOdr * cv
         - 3 * R**2 * c1 * c2 * dUdz * W * d2mudT2 * dTdr - 3 * R**2 * c1 * c2 * dWdr * W * d2mudT2 * dTdr
         - 3 * R**2 * c1 * c2 * dUdz * U * d2mudT2 * dTdz - 3 * R**2 * c1 * c2 * dWdr * U * d2mudT2 * dTdz
         - 3 * R**2 * c1 * c2 * dVdz * V * d2mudT2 * dTdz - 3 * R**2 * c1 * c2 * dVdr * V * d2mudT2 * dTdr
         + 2 * R**2 * c1 * c2 * dUdr * W * d2mudT2 * dTdz - 4 * R**2 * c1 * c2 * dWdz * W * d2mudT2 * dTdz
         + 2 * R * c1 * c2 * U * W * d2mudT2 * dTdz - 4 * R**2 * c1 * c2 * dUdr * U * d2mudT2 * dTdr
         + 2 * R**2 * c1 * c2 * dWdz * U * d2mudT2 * dTdr + 3 * 1j * V * RHO * m * Re * R * c1
         - 1j * U * dmudT * m * V * c1 * c2 + 3 * MU * m**2 * Re * c2
         + 3 * 1j * V * RHO * cv * m * Re * R * c1 * c2 - 3 * 1j * U * dmudT * m * dVdr * R * c1 * c2
         - 3 * 1j * W * dmudT * m * dVdz * R * c1 * c2 + 2 * 1j * V * dmudT * m * dUdr * R * c1 * c2
         + 2 * 1j * V * dmudT * m * dWdz * R * c1 * c2) / Re / R**2 / c1 / c2
    ) / 0.3e1
    c_Ar_51 = -(U * (V**2 * c2 + W**2 * c2 + 2 * T + U**2 * c2 + 2 * cv * T * c2) / c2) / 0.2e1
    c_Ar_52 = (
        (16 * dUdr * MU * c2 * R - 9 * U**2 * RHO * Re * c2 * R - 3 * V**2 * RHO * Re * c2 * R
         - 3 * W**2 * RHO * Re * c2 * R - 6 * T * RHO * Re * R - 8 * MU * dWdz * c2 * R
         + 2 * 1j * V * MU * m * c2 - 6 * cv * T * RHO * Re * c2 * R - 4 * W * dmudT * dTdz * c2 * R
         + 8 * U * dmudT * dTdr * c2 * R) / Re / c2 / R
    ) / 0.6e1
    c_Ar_53 = (
        (6 * dVdr * MU * R - 3 * V * MU + 1j * U * MU * m + 3 * V * dmudT * dTdr * R - 3 * U * RHO * V * Re * R) / Re / R
    ) / 0.3e1
    c_Ar_54 = (2 * dWdr * MU * R + W * MU + 2 * MU * dUdz * R + W * dmudT * dTdr * R
               + U * dmudT * dTdz * R - W * RHO * U * Re * R) / Re / R
    c_Ar_55 = (
        (6 * dmudT * dTdr * Re * R * c2 - 3 * Re * R * c1 * U * RHO - 2 * U**2 * dmudT * c1 * c2
         - 3 * V**2 * dmudT * c1 * c2 + 3 * MU * Re * c2 + 3 * dUdz * W * dmudT * R * c1 * c2
         + 3 * dWdr * W * dmudT * R * c1 * c2 - 3 * Re * R * c1 * c2 * U * RHO * cv
         + 3 * dVdr * V * dmudT * R * c1 * c2 + 4 * dUdr * U * dmudT * R * c1 * c2
         - 2 * dWdz * U * dmudT * R * c1 * c2
        ) / Re / R / c1 / c2
    ) / 0.3e1
    c_Arr_51 = Z
    c_Arr_52 = 0.4e1 / 0.3e1 / Re * U * MU
    c_Arr_53 = 0.1e1 / Re * V * MU
    c_Arr_54 = 0.1e1 / Re * W * MU
    c_Arr_55 = 0.1e1 / c1 * MU
    c_Az_51 = -(W * (V**2 * c2 + W**2 * c2 + 2 * T + U**2 * c2 + 2 * cv * T * c2) / c2) / 0.2e1
    c_Az_52 = (
        (6 * MU * dUdz * R + W * MU + 6 * dWdr * MU * R + 3 * W * dmudT * dTdr * R
         + 3 * U * dmudT * dTdz * R - 3 * W * RHO * U * Re * R) / Re / R
    ) / 0.3e1
    c_Az_53 = (
        (6 * dVdz * MU * R + 1j * W * MU * m + 3 * V * dmudT * dTdz * R - 3 * W * RHO * V * Re * R) / Re / R
    ) / 0.3e1
    c_Az_54 = (
        (16 * MU * dWdz * c2 * R - 3 * V**2 * RHO * Re * c2 * R - 9 * W**2 * RHO * Re * c2 * R
         - 3 * U**2 * RHO * Re * c2 * R - 8 * U * MU * c2 - 8 * dUdr * MU * c2 * R
         - 6 * T * RHO * Re * R + 2 * 1j * V * MU * m * c2 - 6 * cv * T * RHO * Re * c2 * R
         + 8 * W * dmudT * dTdz * c2 * R - 4 * U * dmudT * dTdr * c2 * R) / Re / c2 / R
    ) / 0.6e1
    c_Az_55 = -(
        (-6 * dmudT * dTdz * Re * R * c2 + 3 * W * RHO * Re * R * c1 - 3 * dUdz * U * dmudT * R * c1 * c2
         - 3 * dWdr * U * dmudT * R * c1 * c2 - 3 * dVdz * V * dmudT * R * c1 * c2
         + 3 * W * RHO * cv * Re * R * c1 * c2 + 2 * dUdr * W * dmudT * R * c1 * c2
         - 4 * dWdz * W * dmudT * R * c1 * c2 + 2 * U * W * dmudT * c1 * c2
        ) / Re / R / c1 / c2
    ) / 0.3e1
    c_Azz_51 = Z
    c_Azz_52 = 0.1e1 / Re * U * MU
    c_Azz_53 = 0.1e1 / Re * V * MU
    c_Azz_54 = 0.4e1 / 0.3e1 / Re * W * MU
    c_Azz_55 = 0.1e1 / c1 * MU
    c_Arz_51 = Z
    c_Arz_52 = 0.1e1 / Re * W * MU / 0.3e1
    c_Arz_53 = Z
    c_Arz_54 = 0.1e1 / Re * U * MU / 0.3e1
    c_Arz_55 = Z
    c_B_11 = 1j * I
    c_B_12 = Z
    c_B_13 = Z
    c_B_14 = Z
    c_B_15 = Z
    c_B_21 = 1j * U
    c_B_22 = 1j * RHO
    c_B_23 = Z
    c_B_24 = Z
    c_B_25 = Z
    c_B_31 = 1j * V
    c_B_32 = Z
    c_B_33 = 1j * RHO
    c_B_34 = Z
    c_B_35 = Z
    c_B_41 = 1j * W
    c_B_42 = Z
    c_B_43 = Z
    c_B_44 = 1j * RHO
    c_B_45 = Z
    c_B_51 = (1j * (V**2 + W**2 + U**2 + 2 * cv * T)) / 0.2e1
    c_B_52 = 1j * U * RHO
    c_B_53 = 1j * V * RHO
    c_B_54 = 1j * W * RHO
    c_B_55 = 1j * RHO * cv

    # ------------------------------------------------------------------
    # Step 3: Pack directly into the final physical-order tensors.
    #
    # Mapping from coefficient ordering [rho, v_r, v_theta, v_z, T] to
    # physical ordering [rho, u, v, w, T] where u=axial(v_z), v=radial(v_r),
    # w=azimuthal(v_theta):
    #   coeff[0]->phys[0], coeff[1]->phys[2], coeff[2]->phys[3],
    #   coeff[3]->phys[1], coeff[4]->phys[4]
    # No sign flips for model 'axy' in GetLinProblem.m.
    # ------------------------------------------------------------------
    c2p_idx = [0, 2, 3, 1, 4]
    c2p_sgn = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    return _pack_coefficients_in_physical_order(
        locals(),
        ngp=ngp,
        c2p_idx=c2p_idx,
        c2p_sgn=c2p_sgn,
    )
