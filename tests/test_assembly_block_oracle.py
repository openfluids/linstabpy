#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Independent oracle for the LNSE block assembly.

The raw block operator stacks five equations and five variables into a
``(5*ngp, 5*ngp)`` matrix whose ``(i, j)`` block is

    A0[i,j]*I + Ay[i,j]*DR + Ax[i,j]*DZ + Ayy[i,j]*D2R + Axx[i,j]*D2Z
        + Axy[i,j]*D2RZ

with each coefficient vector left-multiplying (row-scaling) its derivative
operator. The production assembler builds this via COO/CSR scatter and
``bmat``/``hstack``-``vstack``. Those paths share the same placement logic, so
checking one against the other is self-referential.

This oracle instead computes the operator *action* on a random stacked state
through a completely separate route -- explicit per-(equation, variable)
coefficient multiply and derivative application -- and asserts the assembled
matrix reproduces it. A transposed block, a coefficient paired with the wrong
derivative, or a misplaced ``(i, j)`` block diverges between the two paths and
fails here.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from linstabpy.operators.assembly import (
    _assemble_block_operator,
    _assemble_block_operator_csr,
)
from linstabpy.operators.coefficients import LNSECoefficients


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_block(ngp: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(5, 5, ngp)) + 1j * rng.normal(size=(5, 5, ngp))


def _random_coeffs(ngp: int, rng: np.random.Generator) -> LNSECoefficients:
    return LNSECoefficients(
        A0=_random_block(ngp, rng),
        Ax=_random_block(ngp, rng),
        Ay=_random_block(ngp, rng),
        Axx=_random_block(ngp, rng),
        Ayy=_random_block(ngp, rng),
        Axy=_random_block(ngp, rng),
        B=_random_block(ngp, rng),  # unused by the raw block assembler
    )


def _random_operator(ngp: int, rng: np.random.Generator, density: float = 0.4) -> sps.csr_matrix:
    mask = rng.random((ngp, ngp)) < density
    dense = (rng.normal(size=(ngp, ngp)) + 1j * rng.normal(size=(ngp, ngp))) * mask
    return sps.csr_matrix(dense)


def _oracle_action(
    coeffs: LNSECoefficients,
    q: np.ndarray,
    ngp: int,
    DR: sps.spmatrix,
    D2R: sps.spmatrix,
    DZ: sps.spmatrix,
    D2Z: sps.spmatrix,
    D2RZ: sps.spmatrix,
) -> np.ndarray:
    """Apply the LNSE block operator to ``q`` without assembling any matrix.

    Mirrors the documented pairing (A0->I, Ay->DR, Ax->DZ, Ayy->D2R,
    Axx->D2Z, Axy->D2RZ) one equation/variable at a time.
    """
    qj = [q[j * ngp : (j + 1) * ngp] for j in range(5)]
    out = np.zeros(5 * ngp, dtype=complex)
    for i in range(5):
        acc = np.zeros(ngp, dtype=complex)
        for j in range(5):
            v = qj[j]
            acc = acc + coeffs.A0[i, j, :] * v
            acc = acc + coeffs.Ay[i, j, :] * (DR @ v)
            acc = acc + coeffs.Ax[i, j, :] * (DZ @ v)
            acc = acc + coeffs.Ayy[i, j, :] * (D2R @ v)
            acc = acc + coeffs.Axx[i, j, :] * (D2Z @ v)
            acc = acc + coeffs.Axy[i, j, :] * (D2RZ @ v)
        out[i * ngp : (i + 1) * ngp] = acc
    return out


def _run_assembly_oracle(ngp: int = 12, seed: int = 20260601) -> dict:
    rng = _rng(seed)
    coeffs = _random_coeffs(ngp, rng)
    DR = _random_operator(ngp, rng)
    D2R = _random_operator(ngp, rng)
    DZ = _random_operator(ngp, rng)
    D2Z = _random_operator(ngp, rng)
    D2RZ = _random_operator(ngp, rng)

    q = rng.normal(size=5 * ngp) + 1j * rng.normal(size=5 * ngp)
    expected = _oracle_action(coeffs, q, ngp, DR, D2R, DZ, D2Z, D2RZ)
    scale = max(np.max(np.abs(expected)), 1e-300)

    L_csr = _assemble_block_operator_csr(ngp, coeffs, DR, D2R, DZ, D2Z, D2RZ)
    L_bmat = _assemble_block_operator(ngp, coeffs, DR, D2R, DZ, D2Z, D2RZ)

    rel_csr = float(np.max(np.abs((L_csr @ q) - expected)) / scale)
    rel_bmat = float(np.max(np.abs((L_bmat @ q) - expected)) / scale)
    max_rel = max(rel_csr, rel_bmat)
    return {
        "ngp": ngp,
        "rel_csr": rel_csr,
        "rel_bmat": rel_bmat,
        "max_rel_residual": max_rel,
        "pass": bool(max_rel < 1e-12),
    }


def test_block_assembly_matches_independent_action_oracle():
    # Independent oracle: the assembled raw block operator must reproduce the
    # per-(equation, variable) action computed without assembly. Covers both
    # production paths (direct CSR scatter and bmat/hstack-vstack). A misplaced
    # or transposed block, or a coefficient paired with the wrong derivative,
    # diverges from the explicit action and fails here -- not a second
    # transcription of the same placement logic.
    report = _run_assembly_oracle(ngp=12, seed=20260601)
    assert report["pass"], report


def test_block_assembly_oracle_stable_across_seeds():
    for seed in (1, 7, 99, 20260601):
        report = _run_assembly_oracle(ngp=10, seed=seed)
        assert report["pass"], (seed, report)
