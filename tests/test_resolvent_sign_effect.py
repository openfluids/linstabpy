# SPDX-License-Identifier: GPL-3.0-only
"""Effect of the resolvent sign convention on gains and modes of an axisymmetric jet.

``H_new(omega) = -H_old(-omega)`` for every operator.
Take ``W = 0`` and a nonzero azimuthal wavenumber.
The gains agree.
The modes differ by a conjugation and a sign change of the spanwise velocity.
Take ``W != 0``.
The gain map then reverses in ``omega``.
``docs/resolvent_sign_convention.md`` gives the effect table.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
import numpy as np
import scipy.sparse as sp

from linstabpy.baseflow import sutherland_air
from linstabpy.boundary_conditions import apply_dirichlet
from linstabpy.operators import build_lnse_operator
from linstabpy.resolvent.analysis import resolvent_analysis
from tests.conftest import build_synthetic_jet


def _jet_with_swirl(swirl: float) -> tuple[sp.spmatrix, np.ndarray, np.ndarray]:
    """Build an axisymmetric jet with azimuthal wavenumber 3 and a swirl of the given size.

    Returns
    -------
    tuple
        The operator, the Chu weight, and a sign vector that flips the spanwise velocity.
    """
    jet = build_synthetic_jet(nx=15, ny=10, m=3)
    bf, mesh = jet["bf"], jet["mesh"]
    bf.W = swirl * mesh.Y / np.cosh(mesh.Y) ** 2
    sutherland_air(bf)
    L0, idx = build_lnse_operator(mesh, bf, model="axy", m=3)
    L0, _ = apply_dirichlet(L0, idx, "ltr", "uvwT", mesh.ngp)
    flip = np.ones(L0.shape[0])
    flip[np.asarray(idx["w_j"]).ravel()] = -1.0
    return L0, jet["W_diag"], flip


def _modes(
    L0: sp.spmatrix, W_diag: np.ndarray, omega: float, *, parity: bool
) -> tuple[float, np.ndarray, np.ndarray]:
    cfg = {"resolvent_sign_convention": "matlab_parity"} if parity else {}
    gains, responses, forcings = resolvent_analysis(
        L0, omega=omega, n_modes=1, W_diag=W_diag, solver_cfg=cfg
    )
    return gains[0], responses[:, 0], forcings[:, 0]


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_nonzero_azimuthal_wavenumber_without_swirl_conjugates_modes_and_flips_w() -> None:
    # beta != 0 with W = 0: same gain; the new mode is conj(old) with the sign of the
    # spanwise (azimuthal) velocity component reversed, up to a unit phase.
    L0, W_diag, flip = _jet_with_swirl(0.0)
    g_new, u_new, f_new = _modes(L0, W_diag, 2.0, parity=False)
    g_old, u_old, f_old = _modes(L0, W_diag, 2.0, parity=True)

    assert g_new == pytest.approx(g_old, rel=1e-8)
    assert _overlap(f_new, flip * np.conj(f_old)) == pytest.approx(1.0, abs=1e-6)
    assert _overlap(u_new, flip * np.conj(u_old)) == pytest.approx(1.0, abs=1e-6)
    assert _overlap(f_new, np.conj(f_old)) < 0.95


def test_nonzero_azimuthal_wavenumber_with_swirl_mirrors_the_gain_map() -> None:
    # beta != 0 with W != 0: gain_new(omega) = gain_old(-omega), and the map is not even.
    L0, W_diag, _ = _jet_with_swirl(0.5)
    for omega in (-2.0, 2.0):
        g_new = _modes(L0, W_diag, omega, parity=False)[0]
        g_old_mirrored = _modes(L0, W_diag, -omega, parity=True)[0]
        assert g_new == pytest.approx(g_old_mirrored, rel=1e-8)
    assert (
        abs(_modes(L0, W_diag, 2.0, parity=False)[0] - _modes(L0, W_diag, -2.0, parity=False)[0])
        > 1.0
    )
