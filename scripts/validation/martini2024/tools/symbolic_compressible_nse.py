#!/usr/bin/env python3
"""SymPy templates for compressible-LNSE and curvilinear metric parity."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import sympy as sp

_ASSIGN_RE = re.compile(r"c_(A0|Ar|Az|Arr|Azz|Arz|B)_([1-5])([1-5])$")
_BLOCK_ORDER = ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cartesian_symbol_table() -> dict[str, sp.Symbol]:
    names = [
        "U",
        "V",
        "W",
        "RHO",
        "T",
        "MU",
        "P",
        "kx",
        "Re",
        "Ma",
        "Pr",
        "kappa",
        "dUdx",
        "dVdx",
        "dWdx",
        "dRHOdx",
        "dTdx",
        "dPdx",
        "dUdy",
        "dVdy",
        "dWdy",
        "dRHOdy",
        "dTdy",
        "dPdy",
        "d2Udx2",
        "d2Vdx2",
        "d2Wdx2",
        "d2Udy2",
        "d2Vdy2",
        "d2Wdy2",
        "d2Tdx2",
        "d2Tdy2",
        "d2Udxdy",
        "d2Vdxdy",
        "d2Wdxdy",
        "dmudT",
        "d2mudT2",
    ]
    return {name: sp.Symbol(name, complex=True) for name in names}


def cylindrical_symbol_table() -> dict[str, sp.Symbol]:
    names = [
        "U",
        "V",
        "W",
        "R",
        "RHO",
        "T",
        "MU",
        "m",
        "Re",
        "Ma",
        "Pr",
        "kappa",
        "dUdr",
        "dVdr",
        "dWdr",
        "dRHOdr",
        "dTdr",
        "dUdz",
        "dVdz",
        "dWdz",
        "dRHOdz",
        "dTdz",
        "d2Udr2",
        "d2Vdr2",
        "d2Wdr2",
        "d2Tdr2",
        "d2Udz2",
        "d2Vdz2",
        "d2Wdz2",
        "d2Tdz2",
        "d2Udrz",
        "d2Vdrz",
        "d2Wdrz",
        "dmudT",
        "d2mudT2",
    ]
    return {name: sp.Symbol(name, complex=True) for name in names}


def curvilinear_symbol_table() -> dict[str, sp.Symbol]:
    names = [
        "a",
        "b",
        "c",
        "d",
        "dJ11_xi",
        "dJ12_xi",
        "dJ21_xi",
        "dJ22_xi",
        "dJ11_eta",
        "dJ12_eta",
        "dJ21_eta",
        "dJ22_eta",
        "x2p",
        "x2pp",
        "y2p",
        "y2pp",
    ]
    return {name: sp.Symbol(name, complex=True) for name in names}


def continuity_symbol_table() -> dict[str, sp.Symbol]:
    names = [
        "rho",
        "U",
        "V",
        "W",
        "kx",
        "dUdx",
        "dVdy",
        "dRHOdx",
        "dRHOdy",
    ]
    return {name: sp.Symbol(name, complex=True) for name in names}


def continuity_operator_coefficients(symbols: dict[str, sp.Symbol]) -> dict[str, sp.Expr]:
    rho = symbols["rho"]
    U = symbols["U"]
    V = symbols["V"]
    W = symbols["W"]
    kx = symbols["kx"]
    dUdx = symbols["dUdx"]
    dVdy = symbols["dVdy"]
    dRHOdx = symbols["dRHOdx"]
    dRHOdy = symbols["dRHOdy"]

    I = sp.I
    return {
        "A0_00": -(I * kx * W + dUdx + dVdy),
        "A0_01": -dRHOdx,
        "A0_02": -dRHOdy,
        "A0_03": -(I * kx * rho),
        "A0_04": sp.Integer(0),
        "Ax_00": -U,
        "Ax_01": -rho,
        "Ay_00": -V,
        "Ay_02": -rho,
        "B_00": I,
    }



def cartesian_lnse_coefficients_from_nse(symbols: dict[str, sp.Symbol]) -> dict[str, sp.Expr]:
    """Independent first-principles derivation of all 175 cartesian LNSE coefficients.

    Governing equations: non-dimensional compressible Navier-Stokes in primitive
    variables q = [rho, u, v, w, T].  Physical coordinates: x=streamwise,
    y=wall-normal, z=spanwise.  Linearization about a 2-D steady base flow
    (Q_bar = RHO, U, V, W, T, MU, P) with all base-flow z-derivatives zero.

    Fourier ansatz in z: d/dz on perturbations -> I*kx  (kx = spanwise wavenumber).
    System convention: B*dq/dt = L*q, so B[i,j] = sp.I*(coeff of dq_j/dt in eq_i).

    The continuity row (row 0) is verified to match continuity_operator_coefficients
    exactly.

    Nondimensionalisation: lengths by L_ref, velocities by U_inf, density by
    rho_inf, temperature by T_inf.  Pressure: p = rho*T/(kappa*Ma^2).
    Viscous stress: tau_ij = mu*(dui/dxj + duj/dxi) - 2/3*mu*div(u)*delta_ij.
    Energy (temperature form):
        rho*DT/Dt = -(kappa-1)*T*rho*div(u)
                  + kappa/(Re*Pr) * div(mu*grad(T))
                  + kappa*(kappa-1)*Ma^2/Re * Phi
    where Phi = mu*[2*(du/dx)^2 + 2*(dv/dy)^2 + 2*(dw/dz)^2 + (du/dy+dv/dx)^2
                    + (du/dz+dw/dx)^2 + (dv/dz+dw/dy)^2 - 2/3*div(u)^2].
    """
    I = sp.I
    Z = sp.Integer(0)

    U = symbols["U"]; V = symbols["V"]; W = symbols["W"]
    RHO = symbols["RHO"]; T = symbols["T"]; MU = symbols["MU"]
    P = symbols["P"]; kx = symbols["kx"]
    Re = symbols["Re"]; Ma = symbols["Ma"]; Pr = symbols["Pr"]; kappa = symbols["kappa"]
    dUdx = symbols["dUdx"]; dVdx = symbols["dVdx"]; dWdx = symbols["dWdx"]
    dRHOdx = symbols["dRHOdx"]; dTdx = symbols["dTdx"]; dPdx = symbols["dPdx"]
    dUdy = symbols["dUdy"]; dVdy = symbols["dVdy"]; dWdy = symbols["dWdy"]
    dRHOdy = symbols["dRHOdy"]; dTdy = symbols["dTdy"]; dPdy = symbols["dPdy"]
    d2Udx2 = symbols["d2Udx2"]; d2Vdx2 = symbols["d2Vdx2"]; d2Wdx2 = symbols["d2Wdx2"]
    d2Udy2 = symbols["d2Udy2"]; d2Vdy2 = symbols["d2Vdy2"]; d2Wdy2 = symbols["d2Wdy2"]
    d2Tdx2 = symbols["d2Tdx2"]; d2Tdy2 = symbols["d2Tdy2"]
    d2Udxdy = symbols["d2Udxdy"]; d2Vdxdy = symbols["d2Vdxdy"]; d2Wdxdy = symbols["d2Wdxdy"]
    dmudT = symbols["dmudT"]; d2mudT2 = symbols["d2mudT2"]

    # dMU/dx = (dMU/dT)*(dT/dx), similarly for y.  2D base: no z-derivatives.
    dMUdx = dmudT * dTdx
    dMUdy = dmudT * dTdy
    # 1/(kappa*Ma^2) = P/(RHO*T) from ideal gas p = rho*T/(kappa*Ma^2)
    # Used for pressure gradient linearization:
    #   dp'/dx = P/RHO * drho'/dx + rho'*(dPdx/RHO - P*dRHOdx/RHO^2) + P/T * dT'/dx + T'*dRHOdx*P/(RHO*T)
    # (with P=RHO*T/(kappa*Ma^2), dT/dx/(kappa*Ma^2) = dPdx/RHO - P*dRHOdx/RHO^2)
    kappa_1 = kappa - sp.Integer(1)
    # kappa*(kappa-1)*Ma^2/Re — prefactor for viscous dissipation in energy
    C1 = kappa * kappa_1 * Ma ** 2 / Re

    out: dict[str, sp.Expr] = {}

    # ------------------------------------------------------------------
    # Initialise all 175 entries to zero, then assign nonzero terms.
    # Keys: "{block}_{ij}" where block in (A0,Ax,Ay,Axx,Ayy,Axy,B),
    # i in 0..4, j in 0..4.
    # ------------------------------------------------------------------
    for block in ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B"):
        for i in range(5):
            for j in range(5):
                out[f"{block}_{i}{j}"] = Z

    # ==================================================================
    # ROW 0: CONTINUITY
    #
    # drho/dt + d(RHO*u' + U*rho')/dx + d(RHO*v' + V*rho')/dy
    #         + I*kx*(RHO*w' + W*rho') = 0
    #
    # Moving all terms to the RHS (L*q = B*dq/dt convention):
    # B[0,0] = I  (coeff of drho'/dt, times i)
    # Exactly reproduces continuity_operator_coefficients (rho -> RHO).
    # ==================================================================
    out["B_00"] = I
    out["A0_00"] = -(I * kx * W + dUdx + dVdy)
    out["A0_01"] = -dRHOdx
    out["A0_02"] = -dRHOdy
    out["A0_03"] = -I * kx * RHO
    # A0_04 = 0 (already set)
    out["Ax_00"] = -U
    out["Ax_01"] = -RHO
    out["Ay_00"] = -V
    out["Ay_02"] = -RHO
    # All second-derivative blocks for row 0 remain zero.

    # ==================================================================
    # ROW 1: X-MOMENTUM (streamwise, perturbation u', i=1)
    #
    # RHO*du'/dt + RHO*(U*du'/dx + V*du'/dy + I*kx*W*u')
    #            + RHO*(u'*dU/dx + v'*dU/dy)
    #            + rho'*(U*dU/dx + V*dU/dy)
    # = -dp'/dx + (1/Re)*(d(tau_xx')/dx + d(tau_xy')/dy + I*kx*tau_xz')
    #
    # Pressure:
    #   dp'/dx = P/RHO*drho'/dx + (dPdx/RHO-P*dRHOdx/RHO^2)*rho'
    #          + P/T*dT'/dx + dRHOdx*P/(RHO*T)*T'
    #
    # Viscous stress (2D base, dU/dz=0):
    #   tau_xx' = MU*(4/3*du'/dx - 2/3*dv'/dy - 2/3*I*kx*w')
    #             + T'*dmudT*(4/3*dUdx - 2/3*dVdy)
    #   tau_xy' = MU*(du'/dy + dv'/dx) + T'*dmudT*(dUdy + dVdx)
    #   tau_xz' = MU*(I*kx*u' + dw'/dx) + T'*dmudT*dWdx
    # ==================================================================
    out["B_11"] = I * RHO

    # --- A0 (no-derivative) ---
    # rho' terms: advection -(U*dUdx+V*dUdy) + pressure -(dPdx/RHO - P*dRHOdx/RHO^2)
    out["A0_10"] = -(U * dUdx + V * dUdy + dPdx / RHO - P * dRHOdx / RHO ** 2)
    # u' terms: advection -RHO*I*kx*W + base gradient -RHO*dUdx
    #           + viscous -kx^2*MU/Re (from I*kx*tau_xz' = -kx^2*MU*u' + ...)
    out["A0_11"] = -I * kx * W * RHO - RHO * dUdx - kx ** 2 * MU / Re
    # v' terms: advection -RHO*dUdy
    out["A0_12"] = -RHO * dUdy
    # w' terms: viscous from d(tau_xx')/dx: dMUdx*(-2/3*I*kx)*w'
    out["A0_13"] = sp.Rational(-2, 3) * I * kx * dMUdx / Re
    # T' terms: pressure -dRHOdx*P/(RHO*T) + viscous variable-mu second-deriv-of-base terms
    out["A0_14"] = (
        (P * dTdx - T * dPdx) / T ** 2
        + (
            dmudT * (sp.Rational(4, 3) * d2Udx2 - sp.Rational(2, 3) * d2Vdxdy)
            + d2mudT2 * dTdx * (sp.Rational(4, 3) * dUdx - sp.Rational(2, 3) * dVdy)
            + dmudT * (d2Udy2 + d2Vdxdy)
            + d2mudT2 * dTdy * (dUdy + dVdx)
            + I * kx * dmudT * dWdx
        )
        / Re
    )

    # --- Ax (d/dx coefficients) ---
    out["Ax_10"] = -P / RHO  # pressure: -P/RHO * drho'/dx
    # advection -RHO*U + viscous dMUdx*(4/3)*du'/dx from d(tau_xx')/dx
    out["Ax_11"] = -RHO * U + sp.Rational(4, 3) * dMUdx / Re
    # from d(tau_xy')/dy: dMUdy*dv'/dx -> Ax[1,2]
    out["Ax_12"] = dMUdy / Re
    # I*kx*tau_xz' + d(tau_xx')/dx: net coeff of dw'/dx is I*kx*MU/Re - 2/3*I*kx*MU/Re
    out["Ax_13"] = sp.Rational(1, 3) * I * kx * MU / Re
    # pressure -P/T + viscous dmudT*(4/3*dUdx-2/3*dVdy)*dT'/dx
    out["Ax_14"] = -P / T + dmudT * (sp.Rational(4, 3) * dUdx - sp.Rational(2, 3) * dVdy) / Re

    # --- Ay (d/dy coefficients) ---
    # Ay[1,0] = 0 (pressure has no drho'/dy in x-momentum)
    # advection -RHO*V + viscous dMUdy*du'/dy from d(tau_xy')/dy
    out["Ay_11"] = -RHO * V + dMUdy / Re
    # from d(tau_xx')/dx: dMUdx*(-2/3)*dv'/dy
    out["Ay_12"] = sp.Rational(-2, 3) * dMUdx / Re
    # Ay[1,3] = 0 (no dw'/dy in x-momentum viscous)
    # viscous dmudT*(dUdy+dVdx)*dT'/dy from d(tau_xy')/dy
    out["Ay_14"] = dmudT * (dUdy + dVdx) / Re

    # --- Second-derivative blocks ---
    out["Axx_11"] = sp.Rational(4, 3) * MU / Re   # 4/3*MU * d2u'/dx2
    out["Ayy_11"] = MU / Re                         # MU * d2u'/dy2
    out["Axy_12"] = MU / (3 * Re)                  # net coeff of d2v'/dxdy: MU/Re - 2/3*MU/Re

    # ==================================================================
    # ROW 2: Y-MOMENTUM (wall-normal, perturbation v', i=2)
    #
    # By y-momentum symmetry (swap x<->y roles):
    # RHO*dv'/dt + RHO*(U*dv'/dx + V*dv'/dy + I*kx*W*v')
    #            + RHO*(u'*dV/dx + v'*dV/dy)
    #            + rho'*(U*dV/dx + V*dV/dy)
    # = -dp'/dy + (1/Re)*(d(tau_xy')/dx + d(tau_yy')/dy + I*kx*tau_yz')
    #
    #   tau_yy' = MU*(4/3*dv'/dy - 2/3*du'/dx - 2/3*I*kx*w')
    #             + T'*dmudT*(4/3*dVdy - 2/3*dUdx)
    #   tau_yz' = MU*(I*kx*v' + dw'/dy) + T'*dmudT*dWdy
    # ==================================================================
    out["B_22"] = I * RHO

    # --- A0 ---
    out["A0_20"] = -(U * dVdx + V * dVdy + dPdy / RHO - P * dRHOdy / RHO ** 2)
    out["A0_21"] = -RHO * dVdx
    out["A0_22"] = -I * kx * W * RHO - RHO * dVdy - kx ** 2 * MU / Re
    out["A0_23"] = sp.Rational(-2, 3) * I * kx * dMUdy / Re
    out["A0_24"] = (
        (P * dTdy - T * dPdy) / T ** 2
        + (
            dmudT * (sp.Rational(4, 3) * d2Vdy2 - sp.Rational(2, 3) * d2Udxdy)
            + d2mudT2 * dTdy * (sp.Rational(4, 3) * dVdy - sp.Rational(2, 3) * dUdx)
            + dmudT * (d2Vdx2 + d2Udxdy)
            + d2mudT2 * dTdx * (dVdx + dUdy)
            + I * kx * dmudT * dWdy
        )
        / Re
    )

    # --- Ax ---
    # from d(tau_yy')/dy: dMUdy*(-2/3)*du'/dx -> Ax[2,1]
    out["Ax_21"] = sp.Rational(-2, 3) * dMUdy / Re
    # advection -RHO*U + viscous from d(tau_xy')/dx: dMUdx*dv'/dx
    out["Ax_22"] = -RHO * U + dMUdx / Re
    # Ax[2,3] = 0 (no dw'/dx in y-momentum viscous)
    # from d(tau_xy')/dx: dmudT*(dUdy+dVdx)*dT'/dx
    out["Ax_24"] = dmudT * (dUdy + dVdx) / Re

    # --- Ay ---
    out["Ay_20"] = -P / RHO  # pressure: -P/RHO * drho'/dy
    # from d(tau_xy')/dx: dMUdx*du'/dy
    out["Ay_21"] = dMUdx / Re
    # advection -RHO*V + viscous d(tau_yy')/dy: dMUdy*(4/3)*dv'/dy
    out["Ay_22"] = -RHO * V + sp.Rational(4, 3) * dMUdy / Re
    # I*kx*tau_yz' + d(tau_yy')/dy: net I*kx*MU/(3*Re) for dw'/dy
    out["Ay_23"] = sp.Rational(1, 3) * I * kx * MU / Re
    out["Ay_24"] = -P / T + dmudT * (sp.Rational(4, 3) * dVdy - sp.Rational(2, 3) * dUdx) / Re

    # --- Second-derivative blocks ---
    out["Axx_22"] = MU / Re
    out["Ayy_22"] = sp.Rational(4, 3) * MU / Re
    out["Axy_21"] = MU / (3 * Re)  # net coeff of d2u'/dxdy: MU/Re - 2/3*MU/Re

    # ==================================================================
    # ROW 3: Z-MOMENTUM (spanwise, perturbation w', i=3)
    #
    # After Fourier in z (d/dz -> I*kx on perturbations, d/dz of base = 0):
    # -dp'/dz = -I*kx * p' = -I*kx*(P/RHO*rho' + P/T*T')
    #
    #   tau_xz' = MU*(I*kx*u' + dw'/dx) + T'*dmudT*dWdx
    #   tau_yz' = MU*(I*kx*v' + dw'/dy) + T'*dmudT*dWdy
    #   tau_zz' = MU*(4/3*I*kx*w' - 2/3*du'/dx - 2/3*dv'/dy)
    #             + T'*dmudT*(-2/3*(dUdx+dVdy))  [since 2*dW/dz=0 for 2D base]
    #
    # I*kx*tau_zz': -4/3*kx^2*MU*w' - 2/3*I*kx*MU*du'/dx - 2/3*I*kx*MU*dv'/dy
    # ==================================================================
    out["B_33"] = I * RHO

    # --- A0 ---
    out["A0_30"] = -(U * dWdx + V * dWdy + I * kx * P / RHO)
    out["A0_31"] = -RHO * dWdx + I * kx * dMUdx / Re
    out["A0_32"] = -RHO * dWdy + I * kx * dMUdy / Re
    out["A0_33"] = -I * kx * W * RHO - sp.Rational(4, 3) * kx ** 2 * MU / Re
    out["A0_34"] = (
        -I * kx * P / T
        + (
            dmudT * d2Wdx2
            + d2mudT2 * dTdx * dWdx
            + dmudT * d2Wdy2
            + d2mudT2 * dTdy * dWdy
            - sp.Rational(2, 3) * I * kx * dmudT * (dUdx + dVdy)
        )
        / Re
    )

    # --- Ax ---
    # d(tau_xz')/dx: I*kx*MU*du'/dx; I*kx*tau_zz': -2/3*I*kx*MU*du'/dx  -> net I*kx*MU/(3*Re)
    out["Ax_31"] = sp.Rational(1, 3) * I * kx * MU / Re
    # advection -RHO*U + viscous dMUdx*dw'/dx from d(tau_xz')/dx
    out["Ax_33"] = -RHO * U + dMUdx / Re
    # d(tau_xz')/dx: dmudT*dWdx*dT'/dx
    out["Ax_34"] = dmudT * dWdx / Re

    # --- Ay ---
    # d(tau_yz')/dy: I*kx*MU*dv'/dy; I*kx*tau_zz': -2/3*I*kx*MU*dv'/dy -> net I*kx*MU/(3*Re)
    out["Ay_32"] = sp.Rational(1, 3) * I * kx * MU / Re
    # advection -RHO*V + viscous dMUdy*dw'/dy from d(tau_yz')/dy
    out["Ay_33"] = -RHO * V + dMUdy / Re
    # d(tau_yz')/dy: dmudT*dWdy*dT'/dy
    out["Ay_34"] = dmudT * dWdy / Re

    # --- Second-derivative blocks ---
    out["Axx_33"] = MU / Re
    out["Ayy_33"] = MU / Re
    # Axy row 3 all zero (no cross-derivative terms in z-momentum)

    # ==================================================================
    # ROW 4: TEMPERATURE / ENERGY EQUATION (i=4)
    #
    # RHO*dT'/dt + RHO*(U*dT'/dx + V*dT'/dy + I*kx*W*T')
    #           + RHO*(u'*dTdx + v'*dTdy)
    #           + rho'*(U*dTdx + V*dTdy)
    # = -(kappa-1)*T*RHO*(du'/dx + dv'/dy + I*kx*w')
    #   -(kappa-1)*T*(dUdx+dVdy)*rho'
    #   -(kappa-1)*(dUdx+dVdy)*RHO*T'
    #   + kappa/(Re*Pr)*[d(MU*dT'/dx)/dx + d(MU*dT'/dy)/dy - kx^2*MU*T'
    #                    + dmudT*(dTdx+dTdx)*dT'/dx + d(mu'*dTdx)/dx terms]
    #   + C1*Phi'
    #
    # Heat conduction (full linearization):
    #   d/dx[mu*dT/dx] linearized:
    #     MU*d2T'/dx2 + dMUdx*dT'/dx + dmudT*dTdx*dT'/dx
    #     + dmudT*d2Tdx2*T' + d2mudT2*dTdx^2*T'
    #   similarly for y; for z: -kx^2*MU*T' (dT_base/dz=0)
    #
    # Viscous dissipation linearized (2D base, dW/dz=0, dU/dz=dV/dz=0):
    #   Phi' = MU*[4*dUdx*du'/dx + 4*dVdy*dv'/dy
    #              + 2*(dUdy+dVdx)*(du'/dy+dv'/dx)
    #              + 2*dWdx*(I*kx*u'+dw'/dx) + 2*dWdy*(I*kx*v'+dw'/dy)
    #              - 4/3*(dUdx+dVdy)*(du'/dx+dv'/dy+I*kx*w')]
    #         + dmudT*T'*[2*dUdx^2 + 2*dVdy^2 + (dUdy+dVdx)^2
    #                     + dWdx^2 + dWdy^2 - 2/3*(dUdx+dVdy)^2]
    # ==================================================================
    out["B_44"] = I * RHO

    div_U_base = dUdx + dVdy  # 2D base-flow divergence

    # --- A0 ---
    # rho': advection -(U*dTdx+V*dTdy) + pressure-dilatation -(kappa-1)*T*(dUdx+dVdy)
    out["A0_40"] = -(U * dTdx + V * dTdy + kappa_1 * T * div_U_base)
    # u': advection -RHO*dTdx + dissipation C1*MU*2*I*kx*dWdx
    out["A0_41"] = -RHO * dTdx + C1 * MU * 2 * I * kx * dWdx
    # v': advection -RHO*dTdy + dissipation C1*MU*2*I*kx*dWdy
    out["A0_42"] = -RHO * dTdy + C1 * MU * 2 * I * kx * dWdy
    # w': pressure-dilatation -(kappa-1)*T*RHO*I*kx + dissipation C1*MU*(-4/3*div_U)*I*kx
    out["A0_43"] = -(kappa_1 * T * RHO * I * kx) + C1 * MU * (-sp.Rational(4, 3) * div_U_base * I * kx)
    # T': advection -I*kx*W*RHO + pressure-dilatation -(kappa-1)*RHO*div_U
    #   + heat conduction: -kappa*kx^2*MU/(Re*Pr) + kappa*(...)/(Re*Pr)
    #   + dissipation: C1*dmudT*[...]
    out["A0_44"] = (
        -I * kx * W * RHO
        - kappa_1 * RHO * div_U_base
        - kappa * kx ** 2 * MU / (Re * Pr)
        + kappa
        * (
            dmudT * d2Tdx2
            + d2mudT2 * dTdx ** 2
            + dmudT * d2Tdy2
            + d2mudT2 * dTdy ** 2
        )
        / (Re * Pr)
        + C1
        * dmudT
        * (
            2 * dUdx ** 2
            + 2 * dVdy ** 2
            + (dUdy + dVdx) ** 2
            + dWdx ** 2
            + dWdy ** 2
            - sp.Rational(2, 3) * div_U_base ** 2
        )
    )

    # --- Ax ---
    # pressure-dilatation contribution -(kappa-1)*T*RHO*du'/dx + dissipation C1*MU*(8/3*dUdx-4/3*dVdy)
    out["Ax_41"] = (
        -kappa_1 * T * RHO
        + C1 * MU * (sp.Rational(8, 3) * dUdx - sp.Rational(4, 3) * dVdy)
    )
    # dissipation: C1*MU*2*(dUdy+dVdx)
    out["Ax_42"] = C1 * MU * 2 * (dUdy + dVdx)
    # dissipation: C1*MU*2*dWdx
    out["Ax_43"] = C1 * MU * 2 * dWdx
    # advection -RHO*U + heat conduction 2*kappa*dMUdx/(Re*Pr) = 2*kappa*dmudT*dTdx/(Re*Pr)
    out["Ax_44"] = -RHO * U + 2 * kappa * dMUdx / (Re * Pr)

    # --- Ay ---
    # dissipation: C1*MU*2*(dUdy+dVdx)
    out["Ay_41"] = C1 * MU * 2 * (dUdy + dVdx)
    # pressure-dilatation -(kappa-1)*T*RHO + dissipation C1*MU*(8/3*dVdy - 4/3*dUdx)
    out["Ay_42"] = (
        -kappa_1 * T * RHO
        + C1 * MU * (sp.Rational(8, 3) * dVdy - sp.Rational(4, 3) * dUdx)
    )
    # dissipation: C1*MU*2*dWdy
    out["Ay_43"] = C1 * MU * 2 * dWdy
    # advection -RHO*V + heat conduction 2*kappa*dMUdy/(Re*Pr)
    out["Ay_44"] = -RHO * V + 2 * kappa * dMUdy / (Re * Pr)

    # --- Second-derivative blocks (heat conduction only) ---
    out["Axx_44"] = kappa * MU / (Re * Pr)
    out["Ayy_44"] = kappa * MU / (Re * Pr)
    # Axy row 4 all zero (no cross-derivative in energy equation)

    return out



def _coeff_source_path() -> Path:
    # Parse the FROZEN pre-fusion reference (snapshot of coefficients.py at commit
    # 4d4d07b), which still exposes the 175 named ``c_*_ij`` assignments this tool
    # needs. The live production coefficients.py was fused (P1) and no longer
    # exposes them; comparing the live fused numeric output against the symbolic
    # evaluation of these frozen expressions is exactly the "fusion preserved the
    # validated algebra" guard. Sibling-relative, so no repo-root guessing.
    reference = Path(__file__).resolve().parent / "coefficients_reference_base.py"
    if reference.exists():
        return reference
    raise FileNotFoundError(f"Frozen coefficient reference not found: {reference}")


@lru_cache(maxsize=4)
def _assignment_expressions(function_name: str) -> dict[tuple[str, int, int], str]:
    source = _coeff_source_path().read_text(encoding="utf-8")
    tree = ast.parse(source)

    fn_node: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            fn_node = node
            break
    if fn_node is None:
        raise RuntimeError(f"Could not locate {function_name} in coefficients.py")

    out: dict[tuple[str, int, int], str] = {}
    for stmt in fn_node.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        m = _ASSIGN_RE.match(target.id)
        if m is None:
            continue
        block = m.group(1)
        i = int(m.group(2)) - 1
        j = int(m.group(3)) - 1
        out[(block, i, j)] = ast.unparse(stmt.value)

    expected_terms = 7 * 5 * 5
    if len(out) != expected_terms:
        raise RuntimeError(
            f"Expected {expected_terms} assignments in {function_name}, found {len(out)}."
        )
    return out


def _cartesian_eval_namespace(symbols: dict[str, sp.Symbol]) -> dict[str, Any]:
    U = symbols["U"]
    V = symbols["V"]
    W = symbols["W"]
    RHO = symbols["RHO"]
    T = symbols["T"]
    MU = symbols["MU"]
    P = symbols["P"]
    kx = symbols["kx"]

    ns: dict[str, Any] = {
        "sp": sp,
        "U": U,
        "V": V,
        "W": W,
        "RHO": RHO,
        "T": T,
        "MU": MU,
        "P": P,
        "kx": kx,
        "Re": symbols["Re"],
        "Ma": symbols["Ma"],
        "Pr": symbols["Pr"],
        "kappa": symbols["kappa"],
        "Uc": -W,
        "Vc": V,
        "Wc": U,
        "kz": -kx,
        "dUdr": -symbols["dWdy"],
        "dUdz": -symbols["dWdx"],
        "d2Udr2": -symbols["d2Wdy2"],
        "d2Udz2": -symbols["d2Wdx2"],
        "d2Udrz": -symbols["d2Wdxdy"],
        "dVdr": symbols["dVdy"],
        "dVdz": symbols["dVdx"],
        "d2Vdr2": symbols["d2Vdy2"],
        "d2Vdz2": symbols["d2Vdx2"],
        "d2Vdrz": symbols["d2Vdxdy"],
        "dWdr": symbols["dUdy"],
        "dWdz": symbols["dUdx"],
        "d2Wdr2": symbols["d2Udy2"],
        "d2Wdz2": symbols["d2Udx2"],
        "d2Wdrz": symbols["d2Udxdy"],
        "dRHOdr": symbols["dRHOdy"],
        "dRHOdz": symbols["dRHOdx"],
        "dTdr": symbols["dTdy"],
        "dTdz": symbols["dTdx"],
        "d2Tdr2": symbols["d2Tdy2"],
        "d2Tdz2": symbols["d2Tdx2"],
        "dPdr": symbols["dPdy"],
        "dPdz": symbols["dPdx"],
        "dMUdT": symbols["dmudT"],
        "d2MUdT2": symbols["d2mudT2"],
        "Z": sp.Integer(0),
        "I": sp.Integer(1),
    }
    return ns


def _cylindrical_eval_namespace(symbols: dict[str, sp.Symbol]) -> dict[str, Any]:
    kappa = symbols["kappa"]
    Re = symbols["Re"]
    Pr = symbols["Pr"]
    Ma = symbols["Ma"]

    c1 = (kappa - 1) * Re * Pr * Ma**2
    c2 = kappa * Ma**2
    cv = 1 / (kappa * (kappa - 1) * Ma**2)

    ns: dict[str, Any] = {
        "sp": sp,
        "c1": c1,
        "c2": c2,
        "cv": cv,
        "Z": sp.Integer(0),
        "I": sp.Integer(1),
    }
    ns.update(symbols)
    return ns


def _evaluate_coeff_blocks(
    function_name: str,
    symbols: dict[str, sp.Symbol],
    namespace_builder: Callable[[dict[str, sp.Symbol]], dict[str, Any]],
) -> dict[str, dict[tuple[int, int], sp.Expr]]:
    raw = _assignment_expressions(function_name)
    ns = namespace_builder(symbols)
    blocks = {name: {} for name in ("A0", "Ar", "Az", "Arr", "Azz", "Arz", "B")}

    for (block, i, j), expr_src in raw.items():
        expr = sp.sympify(eval(expr_src, {"__builtins__": {}}, ns))
        blocks[block][(i, j)] = expr

    return blocks


def _map_coeff_to_physical(
    blocks: dict[str, dict[tuple[int, int], sp.Expr]],
    *,
    sign_flip_non_b: bool,
    c2p_idx: list[int],
    c2p_sgn: list[int],
) -> dict[str, dict[tuple[int, int], sp.Expr]]:
    signed: dict[str, dict[tuple[int, int], sp.Expr]] = {}
    for name, entries in blocks.items():
        if name == "B" or not sign_flip_non_b:
            signed[name] = dict(entries)
        else:
            signed[name] = {(i, j): -expr for (i, j), expr in entries.items()}

    permuted: dict[str, dict[tuple[int, int], sp.Expr]] = {}
    for name, entries in signed.items():
        mapped: dict[tuple[int, int], sp.Expr] = {}
        for ic in range(5):
            ip = c2p_idx[ic]
            for jc in range(5):
                jp = c2p_idx[jc]
                mapped[(ip, jp)] = c2p_sgn[ic] * c2p_sgn[jc] * entries[(ic, jc)]
        permuted[name] = mapped

    return {
        "A0": permuted["A0"],
        "Ax": permuted["Az"],
        "Ay": permuted["Ar"],
        "Axx": permuted["Azz"],
        "Ayy": permuted["Arr"],
        "Axy": permuted["Arz"],
        "B": permuted["B"],
    }


def _flatten_blocks(blocks: dict[str, dict[tuple[int, int], sp.Expr]]) -> dict[str, sp.Expr]:
    out: dict[str, sp.Expr] = {}
    for block in _BLOCK_ORDER:
        for i in range(5):
            for j in range(5):
                out[f"{block}_{i}{j}"] = blocks[block][(i, j)]
    return out


def cartesian_operator_coefficients(symbols: dict[str, sp.Symbol]) -> dict[str, sp.Expr]:
    coeff_blocks = _evaluate_coeff_blocks(
        "compute_cartesian_coefficients",
        symbols,
        _cartesian_eval_namespace,
    )
    phys_blocks = _map_coeff_to_physical(
        coeff_blocks,
        sign_flip_non_b=True,
        c2p_idx=[0, 3, 2, 1, 4],
        c2p_sgn=[1, -1, 1, 1, 1],
    )
    return _flatten_blocks(phys_blocks)


def cylindrical_operator_coefficients(symbols: dict[str, sp.Symbol]) -> dict[str, sp.Expr]:
    coeff_blocks = _evaluate_coeff_blocks(
        "compute_cylindrical_coefficients",
        symbols,
        _cylindrical_eval_namespace,
    )
    phys_blocks = _map_coeff_to_physical(
        coeff_blocks,
        sign_flip_non_b=False,
        c2p_idx=[0, 2, 3, 1, 4],
        c2p_sgn=[1, 1, 1, 1, 1],
    )
    return _flatten_blocks(phys_blocks)


def curvilinear_operator_coefficients(symbols: dict[str, sp.Symbol]) -> dict[str, sp.Expr]:
    """Return symbolic chain-rule coefficients for curvilinear transforms."""
    a = symbols["a"]
    b = symbols["b"]
    c = symbols["c"]
    d = symbols["d"]

    dJ_xi = sp.Matrix(
        [
            [symbols["dJ11_xi"], symbols["dJ12_xi"]],
            [symbols["dJ21_xi"], symbols["dJ22_xi"]],
        ]
    )
    dJ_eta = sp.Matrix(
        [
            [symbols["dJ11_eta"], symbols["dJ12_eta"]],
            [symbols["dJ21_eta"], symbols["dJ22_eta"]],
        ]
    )
    Ji = sp.Matrix([[a, b], [c, d]])

    dJi_xi = -Ji * dJ_xi * Ji
    dJi_eta = -Ji * dJ_eta * Ji

    q11_x = a * dJi_xi[0, 0] + b * dJi_eta[0, 0]
    q11_y = a * dJi_xi[0, 1] + b * dJi_eta[0, 1]
    q12_x = c * dJi_xi[0, 0] + d * dJi_eta[0, 0]
    q12_y = c * dJi_xi[0, 1] + d * dJi_eta[0, 1]
    q21_x = a * dJi_xi[1, 0] + b * dJi_eta[1, 0]
    q21_y = a * dJi_xi[1, 1] + b * dJi_eta[1, 1]
    q22_x = c * dJi_xi[1, 0] + d * dJi_eta[1, 0]
    q22_y = c * dJi_xi[1, 1] + d * dJi_eta[1, 1]

    out: dict[str, sp.Expr] = {
        # Full-2D first derivatives
        "full2d_Dx_Dx": a,
        "full2d_Dx_Dy": b,
        "full2d_Dy_Dx": c,
        "full2d_Dy_Dy": d,
        # Full-2D second derivatives: [D2x, Dxy, Dyx, D2y, Dx, Dy] coefficients
        "full2d_D2x_c11": a * a,
        "full2d_D2x_c12": a * b,
        "full2d_D2x_c21": b * a,
        "full2d_D2x_c22": b * b,
        "full2d_D2x_qx": q11_x,
        "full2d_D2x_qy": q11_y,
        "full2d_Dxy_c11": a * c,
        "full2d_Dxy_c12": c * b,
        "full2d_Dxy_c21": d * a,
        "full2d_Dxy_c22": d * b,
        "full2d_Dxy_qx": q12_x,
        "full2d_Dxy_qy": q12_y,
        "full2d_Dyx_c11": c * a,
        "full2d_Dyx_c12": c * d,
        "full2d_Dyx_c21": d * a,
        "full2d_Dyx_c22": d * d,
        "full2d_Dyx_qx": q21_x,
        "full2d_Dyx_qy": q21_y,
        "full2d_D2y_c11": c * c,
        "full2d_D2y_c12": c * d,
        "full2d_D2y_c21": d * c,
        "full2d_D2y_c22": d * d,
        "full2d_D2y_qx": q22_x,
        "full2d_D2y_qy": q22_y,
    }

    x2p = symbols["x2p"]
    x2pp = symbols["x2pp"]
    y2p = symbols["y2p"]
    y2pp = symbols["y2pp"]

    out.update(
        {
            # X-only branch
            "x_only_Dx_Dx": 1 / x2p,
            "x_only_D2x_D2x": 1 / x2p**2,
            "x_only_D2x_Dx": -(x2pp / x2p**3),
            # Y-only branch
            "y_only_Dy_Dy": 1 / y2p,
            "y_only_D2y_D2y": 1 / y2p**2,
            "y_only_D2y_Dy": -(y2pp / y2p**3),
        }
    )

    return out


def _find_unresolved(
    exprs: dict[str, sp.Expr],
    allowed_symbols: dict[str, sp.Symbol],
) -> dict[str, list[str]]:
    allowed = set(allowed_symbols.values())
    unresolved: dict[str, list[str]] = {}
    for key, expr in exprs.items():
        unknown = sorted(str(sym) for sym in expr.free_symbols if sym not in allowed)
        if unknown:
            unresolved[key] = unknown
    return unresolved


def _has_nonfinite(expr: sp.Expr) -> bool:
    return bool(expr.has(sp.oo) or expr.has(-sp.oo) or expr.has(sp.zoo) or expr.has(sp.nan))


def build_symbolic_report(mode: str = "all") -> dict[str, Any]:
    tasks: list[
        tuple[
            str,
            Callable[[], dict[str, sp.Symbol]],
            Callable[[dict[str, sp.Symbol]], dict[str, sp.Expr]],
        ]
    ] = []

    if mode in ("cartesian", "both", "all"):
        tasks.append(
            ("cartesian_full_5x5", cartesian_symbol_table, cartesian_operator_coefficients)
        )
    if mode in ("cylindrical", "both", "all"):
        tasks.append(
            ("cylindrical_full_5x5", cylindrical_symbol_table, cylindrical_operator_coefficients)
        )
    if mode in ("curvilinear", "all"):
        tasks.append(
            (
                "curvilinear_metric_terms",
                curvilinear_symbol_table,
                curvilinear_operator_coefficients,
            )
        )

    model_reports: dict[str, Any] = {}
    combined_unresolved: dict[str, list[str]] = {}
    combined_nonfinite: list[str] = []
    total_terms = 0

    for scope, sym_builder, coeff_builder in tasks:
        syms = sym_builder()
        coeffs = coeff_builder(syms)
        unresolved = _find_unresolved(coeffs, syms)
        nonfinite = [k for k, expr in coeffs.items() if _has_nonfinite(expr)]
        total_terms += len(coeffs)

        model_reports[scope] = {
            "n_terms": len(coeffs),
            "n_unresolved_terms": len(unresolved),
            "n_nonfinite_terms": len(nonfinite),
            "pass": bool(not unresolved and not nonfinite),
        }

        for key, value in unresolved.items():
            combined_unresolved[f"{scope}:{key}"] = value
        combined_nonfinite.extend(f"{scope}:{key}" for key in nonfinite)

    passed = bool(not combined_unresolved and not combined_nonfinite)
    return {
        "generated_at_utc": _now_utc(),
        "mode": mode,
        "scope": "full_symbolic_coverage" if mode in ("all", "both") else mode,
        "n_terms_total": total_terms,
        "models": model_reports,
        "n_unresolved_terms": len(combined_unresolved),
        "n_nonfinite_terms": len(combined_nonfinite),
        "unresolved": combined_unresolved,
        "nonfinite": combined_nonfinite,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cartesian", "cylindrical", "curvilinear", "both", "all"),
        default="all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/validation/martini2024/output/operator_symbolic_templates.json"),
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    report = build_symbolic_report(mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        "symbolic templates:",
        f"mode={report['mode']}",
        f"pass={report['pass']}",
        f"n_terms={report['n_terms_total']}",
        f"unresolved={report['n_unresolved_terms']}",
        f"nonfinite={report['n_nonfinite_terms']}",
        f"output={args.output}",
    )
    if not report["pass"] and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
