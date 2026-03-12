"""Spatial eigenproblem assembly for LNSE spatial stability analysis."""

from __future__ import annotations

import scipy.sparse as sp

from ..mesh.mesh import Mesh
from ..baseflow.baseflow import BaseFlow
from .assembly import (
    _prepare_coefficients,
    _assemble_from_prepared,
    _build_index_structure,
)


def build_spatial_operator(
    mesh: Mesh,
    baseflow: BaseFlow,
    model: str = "2D",
    floquet_exp: float = 0.0,
    *,
    kx: float = 0.0,
    m: float = 0,
    delta: float = 1.0,
) -> tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, dict]:
    """Build doubled linear operator blocks for the spatial EVP.

    The spatial formulation is constructed from finite-difference derivatives of
    the LNSE operator with respect to a Floquet-like axial exponent ``alpha``:

    - R0 = L(alpha=0)
    - R1 = (L(+delta) - L(-delta)) / (2*delta)
    - R2 = (L(+delta) - R0 - R1*delta) / delta**2

    And the generalized polynomial EVP is assembled as:

    [ 0,  I] q = alpha [ 0, 0 ] q + [ I, 0 ] q  + terms
    [-R0, 0]    [R1, R2]   [0, I]

    implemented via
        Lc = [[0, I], [-R0, 0]]
        R  = [[I, 0], [R1, R2]]
        Lw = [[0, 0], [-iI, 0]]
    """
    if delta <= 0:
        raise ValueError("delta must be positive")

    n = mesh.ngp * 5

    # Compute coefficients once, assemble three operators with different shifts
    prep = _prepare_coefficients(mesh, baseflow, kx=kx, model=model, m=m)

    L0 = _assemble_from_prepared(prep, floquet_exp=floquet_exp)
    Lp = _assemble_from_prepared(prep, floquet_exp=floquet_exp + delta)
    Lm = _assemble_from_prepared(prep, floquet_exp=floquet_exp - delta)

    R0 = L0.tocsc()
    R1 = (Lp - Lm) / (2.0 * delta)
    R2 = (Lp - R0 - R1 * delta) / delta**2

    I = sp.eye(n, format="csc")
    Z = sp.csc_matrix((n, n))

    Lc = sp.bmat([[Z, I], [-R0, Z]], format="csc")
    R = sp.bmat([[I, Z], [R1, R2]], format="csc")
    Lw = sp.bmat([[Z, Z], [-1j * I, Z]], format="csc")

    idx = _build_index_structure(mesh.idx, mesh.ngp)
    return Lc, R, Lw, idx
