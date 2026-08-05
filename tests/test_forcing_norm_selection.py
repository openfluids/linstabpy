# SPDX-License-Identifier: GPL-3.0-only
"""The resolvent's forcing (input) norm must be selectable: Chu or plain L2.

The optimal gain is ``sigma^2 = max_f ||H f||_E^2 / ||f||_F^2``. We have always used the
Chu weight ``W`` for *both* norms, which solves the eigenproblem of ``invW H^H W H``. The
reference we reproduce uses Chu on the response and plain **L2** on the forcing, which is
the eigenproblem of ``H^H W H`` -- the same operator with the leading ``invW`` dropped.

Two properties of the fixture below are load-bearing, and both were got wrong on a first
attempt at these tests:

**It must be big enough for SLEPc.** ``_resolve_resolvent_ncv`` computes
``min(max(2*nev + 3, 20), ndof - 1)``, so on a tiny problem the subspace collapses to
``ndof - 1``. With ``ndof = 3`` and ``nev = 2`` that gives ``ncv = 2 = nev``, and SLEPc
refuses with "The value of ncv must be at least nev+1" before solving anything. Six degrees
of freedom leaves ``ncv = 5``.

**W must not be the identity.** With ``W = I`` the two norms coincide, ``invW`` disappears,
and a wrong implementation passes every check. ``_W`` below spans a factor of 800 and is
arranged so the two norms peak on *different* diagonal entries, so the gains differ for a
real reason rather than by rounding.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.resolvent import resolvent_analysis

# --------------------------------------------------------------------------------------
# Fixture, and the gains worked out by hand from it
# --------------------------------------------------------------------------------------

#: L0 = -diag(1..6), so H = (L0 - i*omega*I)^-1 is diagonal with H_kk = 1/(-k - i*omega).
_DIAG = -np.arange(1.0, 7.0)
_OMEGA = 0.1

#: Non-uniform on purpose, and tuned so the Chu-optimal index (k=1) differs from the
#: L2-optimal index (k=3).
_W = np.array([0.01, 0.5, 8.0, 0.25, 4.0, 1.5])

_NMODES = 2


def _L0():
    return sp.diags(_DIAG.astype(complex), format="csc")


def _cfg(forcing_norm: str | None = None) -> dict:
    eigen: dict = {"eps_tol": 1e-12}
    if forcing_norm is not None:
        eigen["forcing_norm"] = forcing_norm
    return {"resolvent": {"eigen": eigen}}


def _abs_H_squared() -> np.ndarray:
    """|H_kk|^2 = 1 / (k^2 + omega^2) for this diagonal fixture."""
    return 1.0 / (_DIAG**2 + _OMEGA**2)


def _expected_gain(forcing_norm: str) -> float:
    """The leading sigma, derived analytically rather than measured.

    H and W are diagonal, so the normal operator is diagonal too and its eigenvalues are
    just its diagonal entries::

        chu:  M_kk = |H_kk|^2 * W_kk / W_kk = |H_kk|^2
        l2:   M_kk = |H_kk|^2 * W_kk

    and the leading gain is the square root of the largest.
    """
    m = _abs_H_squared()
    if forcing_norm == "l2":
        m = m * _W
    return float(np.sqrt(m.max()))


def test_the_fixture_makes_the_two_norms_peak_on_different_modes():
    """Guards the fixture itself: if these coincided, the tests below would prove nothing."""
    assert int(np.argmax(_abs_H_squared())) == 0
    assert int(np.argmax(_abs_H_squared() * _W)) == 2
    assert _expected_gain("chu") != pytest.approx(_expected_gain("l2"), rel=1e-6)


# --------------------------------------------------------------------------------------
# The option is wired, and the default has not moved
# --------------------------------------------------------------------------------------


def test_chu_and_l2_give_different_gains():
    """If these agree, the option is not reaching the operator."""
    g_chu = resolvent_analysis(
        _L0(), omega=_OMEGA, n_modes=_NMODES, W_diag=_W, solver_cfg=_cfg("chu")
    )[0]
    g_l2 = resolvent_analysis(
        _L0(), omega=_OMEGA, n_modes=_NMODES, W_diag=_W, solver_cfg=_cfg("l2")
    )[0]
    assert g_chu[0] != pytest.approx(g_l2[0], rel=1e-6)


def test_default_is_chu_so_validated_results_do_not_move():
    """Omitting the option must reproduce the previous behaviour exactly."""
    g_default = resolvent_analysis(
        _L0(), omega=_OMEGA, n_modes=_NMODES, W_diag=_W, solver_cfg=_cfg()
    )[0]
    g_explicit = resolvent_analysis(
        _L0(), omega=_OMEGA, n_modes=_NMODES, W_diag=_W, solver_cfg=_cfg("chu")
    )[0]
    assert g_default[0] == pytest.approx(g_explicit[0], rel=1e-12)


@pytest.mark.parametrize("forcing_norm", ["chu", "l2"])
def test_gain_matches_the_hand_computed_value(forcing_norm):
    """Correctness rather than plumbing: compare against the analytic answer."""
    gains = resolvent_analysis(
        _L0(), omega=_OMEGA, n_modes=_NMODES, W_diag=_W, solver_cfg=_cfg(forcing_norm)
    )[0]
    assert gains[0] == pytest.approx(_expected_gain(forcing_norm), rel=1e-9)


def test_an_unknown_forcing_norm_is_refused():
    """A typo must fail loudly rather than silently falling back to Chu."""
    with pytest.raises(ValueError, match="forcing_norm"):
        resolvent_analysis(_L0(), omega=_OMEGA, n_modes=_NMODES, W_diag=_W, solver_cfg=_cfg("chew"))


# --------------------------------------------------------------------------------------
# The residual gate -- where the real trap lives
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("forcing_norm", ["chu", "l2"])
def test_residual_gate_passes_for_both_norms(forcing_norm):
    """Forward AND adjoint residuals must be small under both choices.

    The adjoint relation the singular triplet satisfies differs between the two, by
    exactly ``invW``::

        chu:  invW H^H (W u) = sigma v
        l2:        H^H (W u) = sigma v

    Measuring the chu relation on an l2 triplet evaluates to ``sigma*||(invW - I) v||``,
    which is large for any non-identity W. This test therefore only has teeth because
    ``_W`` is not the identity: on a dense 6x6 check with W spanning 1e-3..1e3, the wrong
    form gives 6e+04 where the right one gives 1e-13.
    """
    *_, rho_fwd, rho_adj = resolvent_analysis(
        _L0(),
        omega=_OMEGA,
        n_modes=_NMODES,
        W_diag=_W,
        solver_cfg=_cfg(forcing_norm),
        residuals=True,
    )
    assert np.max(rho_fwd) < 1e-10, f"forward residuals too large: {rho_fwd}"
    assert np.max(rho_adj) < 1e-10, f"adjoint residuals too large: {rho_adj}"


# --------------------------------------------------------------------------------------
# Mode normalisation follows the norm that was actually used
# --------------------------------------------------------------------------------------


def test_l2_forcing_modes_are_unit_in_the_two_norm():
    forcings = resolvent_analysis(
        _L0(), omega=_OMEGA, n_modes=_NMODES, W_diag=_W, solver_cfg=_cfg("l2")
    )[2]
    for i in range(_NMODES):
        assert np.linalg.norm(forcings[:, i]) == pytest.approx(1.0, rel=1e-10)


def test_chu_forcing_modes_are_unit_in_the_W_norm():
    forcings = resolvent_analysis(
        _L0(), omega=_OMEGA, n_modes=_NMODES, W_diag=_W, solver_cfg=_cfg("chu")
    )[2]
    for i in range(_NMODES):
        w_norm = np.sqrt(np.real(forcings[:, i].conj() @ (_W * forcings[:, i])))
        assert w_norm == pytest.approx(1.0, rel=1e-10)
