#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Independent geometric closed form for Neumann BC rows.

A homogeneous Neumann row enforces zero *normal* gradient at a boundary point.
Divided by the penalty it is a first-derivative stencil pointing along the
outward boundary normal.
The production rows are built from mesh metrics (``gradX``/``gradY`` on
rectangular meshes, the ``J`` Jacobian arrays on mapped meshes).
Any check that reuses those metrics would just re-derive the implementation.

This closed form instead takes the boundary normal from the *physical
boundary polyline*.
The tangent is the finite difference of consecutive physical boundary points.
The normal is the perpendicular of that tangent.
This form does not use the production normal formula.
A normal-derivative row has these geometric properties.

* It annihilates a field that varies purely along the boundary tangent
  (``n . grad(f_tangent) = n . t = 0``).
* It responds to a field that varies along the normal
  (``n . grad(f_normal) = |n|^2 != 0``).
* It annihilates constants.
* Its nonzeros stay inside the constrained variable block.

A row built along the tangent instead of the normal fails here.
A swapped Jacobian component fails here.
A mismatched variable block fails here.
The check covers the rectangular (gradient) branch and the curvilinear
(Jacobian) branch.

The non-curvilinear branch uses the physical-axis normal.
That normal equals the true normal only on axis-aligned boundaries.
This check therefore uses a rectangular mesh.
The curvilinear branch carries true mapped normals and this check uses a
curved mesh.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from linstabpy.boundary_conditions.neumann import apply_neumann
from linstabpy.mesh.mesh import Mesh
from linstabpy.operators.assembly import _build_index_structure

_BORDER_KEY = {"l": "li", "r": "ri", "b": "bi", "t": "ti"}
_VAR_OFFSET = {"r": 0, "u": 1, "v": 2, "w": 3, "T": 4}


def _rect_mesh(nx: int = 11, ny: int = 9) -> Mesh:
    return Mesh.create((0.0, 3.0), (0.0, 2.0), nx, ny, fd_order=4)


def _curved_mesh(nx: int = 11, ny: int = 9) -> Mesh:
    base = _rect_mesh(nx, ny)
    # Nonlinear, genuinely 2-D map so the boundary normal is not axis-aligned
    # and varies along each boundary.
    Xn = base.X + 0.10 * base.Y + 0.03 * base.Y**2
    Yn = base.Y + 0.05 * base.X
    return base.deform(Xn, Yn)


def _check_border(
    mesh: Mesh,
    idx: dict,
    border: str,
    variable: str,
    *,
    curvilinear: bool,
) -> dict:
    ngp = mesh.ngp
    X = mesh.X.ravel(order="F")
    Y = mesh.Y.ravel(order="F")
    var = _VAR_OFFSET[variable]

    L0 = sps.csc_matrix((5 * ngp, 5 * ngp), dtype=complex)
    L0b, bc = apply_neumann(
        L0, mesh, idx, border, variable, penalty=1.0, curvilinear_normals=curvilinear
    )
    L = L0b.tocsr()

    # Interior boundary point (avoid corners) and its along-boundary neighbours.
    pts = np.sort(idx[_BORDER_KEY[border]])
    mid = len(pts) // 2
    imesh = int(pts[mid])
    prev_pt = int(pts[mid - 1])
    next_pt = int(pts[mid + 1])

    # Independent tangent/normal from the physical boundary polyline.
    tx = X[next_pt] - X[prev_pt]
    ty = Y[next_pt] - Y[prev_pt]
    tnorm = float(np.hypot(tx, ty))
    tx, ty = tx / tnorm, ty / tnorm
    nx_, ny_ = -ty, tx  # perpendicular to the tangent

    dof = var * ngp + imesh
    row_full = L.getrow(dof).toarray().ravel()
    row = row_full[var * ngp : (var + 1) * ngp]

    # Variable-block confinement: nonzeros only inside the constrained block.
    outside = np.concatenate([row_full[: var * ngp], row_full[(var + 1) * ngp :]])
    block_confined = bool(not np.any(outside))

    f_tan = tx * X + ty * Y
    f_norm = nx_ * X + ny_ * Y
    resp_tan = float(np.real(row @ f_tan))
    resp_norm = float(np.real(row @ f_norm))
    resp_const = float(np.real(row @ np.ones(ngp)))

    denom = max(abs(resp_norm), 1e-300)
    return {
        "border": border,
        "variable": variable,
        "dof_ok": bool(dof in set(bc.tolist())),
        "block_confined": block_confined,
        "tangent_to_normal_ratio": abs(resp_tan) / denom,
        "normal_response": abs(resp_norm),
        "const_response": abs(resp_const),
    }


def _passes(rec: dict) -> bool:
    # Measured tangent/normal and constant responses are ~5e-15 on BOTH the
    # rectangular (gradient) and curved (Jacobian) branches, so the check asserts
    # ~1e-12 (the ~1e-15 property, with ~200x margin) rather than the
    # old 1e-9. A tangent-aligned or swapped-Jacobian mutation drives the ratio to
    # O(1), far above either threshold.
    return bool(
        rec["dof_ok"]
        and rec["block_confined"]
        and rec["tangent_to_normal_ratio"] < 1e-12
        and rec["normal_response"] > 0.1
        and rec["const_response"] < 1e-12
    )


def test_neumann_rectangular_is_normal_derivative():
    # Non-curvilinear (gradient) branch on an axis-aligned mesh: the row points
    # along the physical-axis normal, which is the true normal here.
    mesh = _rect_mesh()
    idx = _build_index_structure(mesh.idx, mesh.ngp)
    for border in "lrbt":
        for variable in ("r", "u", "T"):
            rec = _check_border(mesh, idx, border, variable, curvilinear=False)
            assert _passes(rec), rec


def test_neumann_curvilinear_is_normal_derivative_on_curved_mesh():
    # Curvilinear (Jacobian) branch on a curved mesh: the row must follow the
    # true mapped boundary normal, verified against the physical polyline normal.
    mesh = _curved_mesh()
    idx = _build_index_structure(mesh.idx, mesh.ngp)
    for border in "lrbt":
        for variable in ("r", "v", "T"):
            rec = _check_border(mesh, idx, border, variable, curvilinear=True)
            assert _passes(rec), rec


@pytest.mark.parametrize("edge", ["l", "r", "b", "t"])
def test_neumann_rows_add_no_spurious_unstable_mode(edge: str) -> None:
    """The penalty row must damp: an outward normal gave Im(omega) ~ penalty / h."""
    from linstabpy.baseflow import BaseFlow, sutherland_air
    from linstabpy.boundary_conditions.neumann import apply_neumann
    from linstabpy.mesh import Mesh
    from linstabpy.operators import build_lnse_operator

    mesh = Mesh.create(xrange=(0.0, 2.0), yrange=(0.0, 1.0), nx=10, ny=9, fd_order=4)
    T = 1.0 + 0.3 * np.exp(-3.0 * mesh.Y)
    zero = np.zeros_like(T)
    bf = BaseFlow(
        Re=500.0,
        Ma=0.5,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
        RHO=1.0 / T,
        U=np.tanh(4.0 * mesh.Y),
        V=zero,
        W=zero,
        T=T,
    )
    sutherland_air(bf)
    L0, idx = build_lnse_operator(mesh, bf, model="2D")
    growth_free = np.linalg.eigvals(1j * L0.toarray()).imag.max()
    L, _ = apply_neumann(L0, mesh, idx, edge, "T", penalty=1.0e4)
    growth = np.linalg.eigvals(1j * L.toarray()).imag.max()
    assert growth < growth_free + 1.0
