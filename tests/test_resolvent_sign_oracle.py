# SPDX-License-Identifier: GPL-3.0-only
"""Closed form that pins the resolvent frequency-axis SIGN convention.

Physical convention (default): ``H(omega) = [(gamma - i*omega) I - L0]^-1``.
It is the response to forcing ``f*exp(-i*omega*t)`` of ``dq/dt = L0 q + f``,
with ``q ~ exp(-i*omega*t)`` as in the stability solver.
Take an ASYMMETRIC operator with dominant eigenvalue
``lambda = -gamma + i*Omega`` (Omega != 0).
The EVP returns ``omega_eig = eig(i*L0) = i*lambda = -Omega - i*gamma``.
The gain is ``sigma(omega) = 1 / sigma_min((-i*omega) I - L0)``.
The operator is closest to singular when ``-i*omega`` meets the eigenvalue.
The leading gain therefore peaks at ``omega = Re(omega_eig) = -Omega``.
``solver.resolvent_sign_convention: matlab_parity`` restores the LinStab2D matrix
``L0 - (gamma + i*omega) I``, whose gain peaks at ``+Omega``.
See ``wiki/resolvent-sign-convention.md``.

The sign matters only for genuinely asymmetric / k_z != 0 base flows
(the production SWBLI / Savarino-like gain map).
For a real operator the gain does not change when omega changes sign,
so both conventions give the same gain.
The shipped examples with a spanwise-homogeneous base flow
(Blasius, jet m=0, paraboloid m=5) are therefore gain-invariant.

This is an independent closed form: a tiny diagonal complex operator whose
resolvent peak frequency is known exactly.
CI only; no external data.
A flipped shift sign trips these tests.

Measured both-signs table, asymmetric operator below (gain at the physical peak -Omega
and at +Omega, physical convention): gain(-Omega)=5.000, gain(+Omega)=0.167, 30x.
Symmetric operator: gain(+Omega)=0.329, gain(-Omega)=0.329, 1.0x.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
import numpy as np
import scipy.sparse as sp

from linstabpy.resolvent.analysis import resolvent_analysis

_CFG = {"resolvent": {"eigen": {"eps_tol": 1e-12}}}
_CFG_PARITY = {
    "resolvent": {"eigen": {"eps_tol": 1e-12}},
    "resolvent_sign_convention": "matlab_parity",
}
_GAMMA = 0.2
_OMEGA = 3.0


def _leading_gain(L0: sp.spmatrix, omega: float, solver_cfg: dict = _CFG) -> float:
    gains, *_ = resolvent_analysis(
        L0,
        omega=omega,
        n_modes=1,
        W_diag=np.ones(L0.shape[0]),
        solver_cfg=solver_cfg,
    )
    return float(gains[0])


def _asymmetric_operator() -> sp.spmatrix:
    # Dominant near-resonant complex eigenvalue -gamma + i*Omega plus far-off,
    # off-axis eigenvalues so the gain peak is unambiguous and sign-bearing.
    diag = np.array(
        [
            -_GAMMA + 1j * _OMEGA,
            -5 - 8j,
            -6 + 9j,
            -7 - 4j,
            -8 + 11j,
            -9 - 6j,
            -10 + 13j,
            -11 - 7j,
        ],
        dtype=complex,
    )
    return sp.diags(diag, format="csc", dtype=complex)


def _symmetric_operator() -> sp.spmatrix:
    # Real eigenvalues: resolvent gain does not change when omega changes sign.
    return sp.diags(
        np.array([-0.5, -2, -3, -4, -5, -6, -7, -8], dtype=complex),
        format="csc",
        dtype=complex,
    )


def test_resolvent_gain_peaks_at_re_omega_eig_for_asymmetric_operator():
    """Physical (gamma - i*omega) I - L0 must peak at -Omega = Re(eig(i*L0))."""
    L0 = _asymmetric_operator()
    omegas = np.linspace(-6.0, 6.0, 49)
    gains = np.array([_leading_gain(L0, w) for w in omegas])
    w_peak = omegas[int(np.argmax(gains))]
    assert abs(w_peak + _OMEGA) < abs(w_peak - _OMEGA), (
        f"resolvent gain peaks at omega={w_peak}, expected near -Omega={-_OMEGA} "
        f"for the physical convention; the old LinStab2D sign would peak near +Omega={_OMEGA}"
    )
    assert abs(w_peak + _OMEGA) <= (omegas[1] - omegas[0]) + 1e-9, (
        f"peak omega={w_peak} not within one sweep step of -Omega={-_OMEGA}"
    )


def test_resolvent_gain_peaks_at_positive_omega_under_matlab_parity():
    """matlab_parity (L0 - (gamma + i*omega) I) keeps the old peak at +Omega."""
    L0 = _asymmetric_operator()
    omegas = np.linspace(-6.0, 6.0, 49)
    gains = np.array([_leading_gain(L0, w, _CFG_PARITY) for w in omegas])
    w_peak = omegas[int(np.argmax(gains))]
    assert abs(w_peak - _OMEGA) <= (omegas[1] - omegas[0]) + 1e-9, (
        f"peak omega={w_peak} not within one sweep step of +Omega={_OMEGA}"
    )


def test_resolvent_gain_sign_distinguishes_plus_minus_omega_asymmetric():
    """gain(-Omega) must dominate gain(+Omega) for the asymmetric operator."""
    L0 = _asymmetric_operator()
    g_peak = _leading_gain(L0, -_OMEGA)
    g_off = _leading_gain(L0, _OMEGA)
    assert g_peak > 5.0 * g_off, (
        f"gain(-Omega)={g_peak:.4f} should dominate gain(+Omega)={g_off:.4f} "
        "(sign-distinguishing); near-equal would mean the convention is undetectable"
    )
    # matlab_parity mirrors it exactly: gain_parity(w) = gain_physical(-w).
    assert _leading_gain(L0, _OMEGA, _CFG_PARITY) == pytest.approx(g_peak, rel=1e-9)
    assert _leading_gain(L0, -_OMEGA, _CFG_PARITY) == pytest.approx(g_off, rel=1e-9)


def test_real_operator_gain_equal_and_modes_conjugate_under_both_conventions():
    """Real L0: same gain; physical modes are the complex conjugates of the parity modes."""
    rng = np.random.default_rng(3)
    n = 8
    L0 = sp.csc_matrix(rng.standard_normal((n, n)) - 4.0 * np.eye(n), dtype=complex)
    omega = 1.3
    out = {}
    for name, cfg in (("physical", _CFG), ("parity", _CFG_PARITY)):
        gains, responses, forcings = resolvent_analysis(
            L0, omega=omega, n_modes=1, W_diag=np.ones(n), solver_cfg=cfg
        )
        out[name] = (gains[0], responses[:, 0], forcings[:, 0])
    assert out["physical"][0] == pytest.approx(out["parity"][0], rel=1e-9)
    # A singular vector is defined up to a unit phase: |<a, conj(b)>| = 1 for unit vectors.
    for k in (1, 2):
        a, b = out["physical"][k], out["parity"][k]
        overlap = abs(np.vdot(a, np.conj(b))) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert overlap == pytest.approx(1.0, abs=1e-8)


def test_resolvent_gain_is_even_in_omega_for_symmetric_operator():
    """Real/symmetric operator: gain(omega) == gain(-omega), convention-invariant."""
    L0 = _symmetric_operator()
    for w in (1.0, 2.5, 4.0):
        g_plus = _leading_gain(L0, w)
        g_minus = _leading_gain(L0, -w)
        assert abs(g_plus - g_minus) / max(g_plus, 1e-300) < 1e-9, (
            f"symmetric gain not even in omega at w={w}: "
            f"gain(+w)={g_plus:.6e} vs gain(-w)={g_minus:.6e}"
        )
