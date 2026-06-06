#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Parity + structural oracle for Chu's compressible energy norm.

The Chu norm is a canonical diagonal quadratic form on the primitive
perturbation variables ``[rho, u, v, w, T]``:

    W = 0.5 * [ T/(rho*kappa*Ma^2),  rho,  rho,  rho,
                rho/(kappa*(kappa-1)*T*Ma^2) ] (.) W_int

with ``W_int -> W_int * R`` for the axisymmetric (``axy``) model. The three
distinct coefficients are not a multi-step derivation that could silently drop a
term -- they are a short canonical form, identical across the production code,
the checked-in upstream reference
``scripts/validation/song_2023/reference/matlab/matlab_code/SolverMartini/aux_matlab/ConstructOperators/GetCompEnergyNorm.m``,
and the Chu (1965) literature. So a from-scratch re-derivation here would only
risk being tuned to match.

Instead this oracle (1) pins production to the canonical coefficient form -- any
drift (e.g. a kappa/(kappa-1) slip) diverges and fails -- and (2) independently
verifies the assembly plumbing that genuinely can break: block concatenation
order, the 0.5 factor, the W_int multiplication, the axy radial factor, and the
inverse relationship invW = 1/W.
"""

from __future__ import annotations

import numpy as np

from linstabpy.operators.energy_norm import compute_chu_norm

_KAPPA = 1.4
_MA = 0.9


def _canonical_blocks(
    RHO: np.ndarray, T: np.ndarray, W_int: np.ndarray, kappa: float, Ma: float
) -> list[np.ndarray]:
    """Canonical Chu weight blocks (GetCompEnergyNorm.m), built independently."""
    Ma2 = Ma**2
    return [
        0.5 * T / (RHO * kappa * Ma2) * W_int,  # rho
        0.5 * RHO * W_int,  # u
        0.5 * RHO * W_int,  # v
        0.5 * RHO * W_int,  # w
        0.5 * RHO / (kappa * (kappa - 1.0) * T * Ma2) * W_int,  # T
    ]


def _run_norm_oracle(ngp: int = 16, seed: int = 20260601) -> dict:
    rng = np.random.default_rng(seed)
    RHO = rng.uniform(0.5, 2.0, ngp)
    T = rng.uniform(0.5, 2.0, ngp)
    W_int = rng.uniform(0.1, 1.0, ngp)
    kappa, Ma = _KAPPA, _MA

    W_diag, invW_diag = compute_chu_norm(W_int, RHO, T, kappa, Ma, ngp)

    expected = np.concatenate(_canonical_blocks(RHO, T, W_int, kappa, Ma))
    value_diff = float(np.max(np.abs(W_diag - expected)))
    inv_diff = float(np.max(np.abs(invW_diag - 1.0 / W_diag)))

    # block layout: u, v, w blocks must be identical (all 0.5*RHO*W_int)
    blocks = W_diag.reshape(5, ngp)
    uvw_identical = bool(
        np.array_equal(blocks[1], blocks[2]) and np.array_equal(blocks[2], blocks[3])
    )
    # and rho/T blocks must differ from the momentum blocks (catches a flat layout)
    rho_T_distinct = bool(
        not np.allclose(blocks[0], blocks[1]) and not np.allclose(blocks[4], blocks[1])
    )

    # energy-equivalence: quadratic form on a random complex state equals the
    # per-component sum built from the independent blocks.
    q = rng.normal(size=5 * ngp) + 1j * rng.normal(size=5 * ngp)
    E_form = float(np.real(np.sum(np.conj(q) * (W_diag * q))))
    E_comp = float(
        np.sum(
            [np.sum(_canonical_blocks(RHO, T, W_int, kappa, Ma)[v] * np.abs(q[v * ngp : (v + 1) * ngp]) ** 2) for v in range(5)]
        )
    )
    energy_diff = abs(E_form - E_comp) / max(abs(E_comp), 1e-300)

    # axisymmetric radial factor: W_int -> W_int * R per block
    R = rng.uniform(0.2, 3.0, ngp)
    W_axy, _ = compute_chu_norm(W_int, RHO, T, kappa, Ma, ngp, model="axy", R=R)
    expected_axy = np.concatenate(_canonical_blocks(RHO, T, W_int * R, kappa, Ma))
    axy_diff = float(np.max(np.abs(W_axy - expected_axy)))

    return {
        "value_diff": value_diff,
        "inv_diff": inv_diff,
        "uvw_identical": uvw_identical,
        "rho_T_distinct": rho_T_distinct,
        "energy_equiv_rel": energy_diff,
        "axy_diff": axy_diff,
        "pass": bool(
            value_diff < 1e-12
            # invW is the exact reciprocal of W (measured |invW - 1/W| == 0.0),
            # so this is held at 1e-12 like the other checks, not the looser 1e-9.
            and inv_diff < 1e-12
            and uvw_identical
            and rho_T_distinct
            and energy_diff < 1e-12
            and axy_diff < 1e-12
        ),
    }


def test_chu_norm_matches_canonical_form_and_layout():
    # Pins production to the canonical Chu coefficient form and independently
    # checks block layout, scaling, weights, axy factor, and the inverse.
    report = _run_norm_oracle()
    assert report["pass"], report


def test_chu_norm_oracle_stable_across_seeds():
    for seed in (1, 13, 101, 20260601):
        report = _run_norm_oracle(ngp=20, seed=seed)
        assert report["pass"], (seed, report)
