# SPDX-License-Identifier: GPL-3.0-only
"""Front-F oracle: pin the resolvent frequency-axis SIGN convention.

Production factors the resolvent as ``H(omega) = (L0 - i*omega*I)^{-1}``
(solvers/resolvent.py ``_shifted_operator``). For an ASYMMETRIC operator with a
dominant eigenvalue ``lambda = -gamma + i*Omega`` (Omega != 0) the leading gain
``sigma(omega) = 1 / sigma_min(L0 - i*omega*I)`` peaks at ``omega = +Omega =
Im(lambda)``: the operator is closest to singular when ``i*omega`` meets the
eigenvalue. The OPPOSITE shift ``(L0 + i*omega*I)`` would peak at ``-Omega`` -- a
flipped frequency axis. This matters only for genuinely asymmetric / k_z != 0
base flows (the production SWBLI / Savarino-like gain map); for a real/symmetric
operator the gain is even in omega and both conventions agree, so every shipped
symmetric example (Blasius spanwise-homogeneous, jet m=0, paraboloid m=5) is
convention-invariant.

This is an INDEPENDENT closed-form oracle (a tiny diagonal complex operator whose
resolvent peak frequency is known exactly), CI-only, no external data. It PINS
the as-implemented convention so a wrong shift sign trips here. It does NOT edit
the production shift -- that is STOP-gated (Front-F). If the intended convention
is ever changed (Ricardo's call), the expected sign below flips with it.

Measured both-signs table (production H = (L0 - i*omega*I)^{-1}):
    asymmetric  gain(+Omega)=5.000  gain(-Omega)=0.167  ratio 30x  -> DIFFER
    symmetric   gain(+Omega)=0.329  gain(-Omega)=0.329  ratio  1.0 -> AGREE
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from linstabpy.solvers.resolvent import resolvent_analysis

_CFG = {"resolvent": {"eigen": {"eps_tol": 1e-12}}}
_GAMMA = 0.2
_OMEGA = 3.0


def _leading_gain(L0: sp.spmatrix, omega: float) -> float:
    gains, *_ = resolvent_analysis(
        L0,
        omega=omega,
        n_modes=1,
        W_diag=np.ones(L0.shape[0]),
        solver_cfg=_CFG,
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
    # Real eigenvalues -> resolvent gain is even in omega.
    return sp.diags(
        np.array([-0.5, -2, -3, -4, -5, -6, -7, -8], dtype=complex),
        format="csc",
        dtype=complex,
    )


def test_resolvent_gain_peaks_at_positive_omega_for_asymmetric_operator():
    """Production (L0 - i*omega*I)^-1 must peak at +Omega = +Im(lambda)."""
    L0 = _asymmetric_operator()
    omegas = np.linspace(-6.0, 6.0, 49)
    gains = np.array([_leading_gain(L0, w) for w in omegas])
    w_peak = omegas[int(np.argmax(gains))]
    # Peak must be on the +Omega side, not the flipped -Omega side.
    assert abs(w_peak - _OMEGA) < abs(w_peak + _OMEGA), (
        f"resolvent gain peaks at omega={w_peak}, expected near +Omega={_OMEGA} "
        f"for the (L0 - i*omega*I) convention; a flipped shift sign would peak "
        f"near -Omega={-_OMEGA}"
    )
    assert abs(w_peak - _OMEGA) <= (omegas[1] - omegas[0]) + 1e-9, (
        f"peak omega={w_peak} not within one sweep step of +Omega={_OMEGA}"
    )


def test_resolvent_gain_sign_distinguishes_plus_minus_omega_asymmetric():
    """gain(+Omega) must dominate gain(-Omega) for the asymmetric operator."""
    L0 = _asymmetric_operator()
    g_plus = _leading_gain(L0, _OMEGA)
    g_minus = _leading_gain(L0, -_OMEGA)
    assert g_plus > 5.0 * g_minus, (
        f"gain(+Omega)={g_plus:.4f} should dominate gain(-Omega)={g_minus:.4f} "
        "(sign-distinguishing); near-equal would mean the convention is undetectable"
    )


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
