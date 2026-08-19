#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""PERF1 assembly profiling record and equivalence guard.

Representative profile case used for r3uu.14.2 measurements.
Blasius 2-D operator, fd6 mesh, nx=100, ny=80 (ngp=8000, ndof=40000),
kx=0.25.
The mesh and baseflow are built once.
Then five timed repeats of ``build_lnse_operator`` run on the shared dev
box.

Before optimization (baseline assembly.py).
Times [s] = 0.107545, 0.100286, 0.103195, 0.103143, 0.101942.
Median = 0.103143 s.
cProfile hot paths by cumulative time.
``_assemble_block_operator`` 0.057 s.
scipy ``csr_tocsc`` 0.035 s total.
``_build_block_csr`` 0.026 s.
2-D ``_apply_inv_B`` sparse ``diag @ L0`` 0.026 s.

After optimization (CSR raw orchestration plus direct 2-D row scaling).
Times [s] = 0.075539, 0.079442, 0.076685, 0.076899, 0.076183.
Median = 0.076685 s.
cProfile hot paths by cumulative time.
``_assemble_block_operator_csr`` 0.049 s.
Final ``csr_tocsc`` 0.014 s.
``_build_block_csr`` 0.028 s.
``hstack``/``vstack`` CSR stacking 0.021 s total.
The measured median reduction on this noisy shared box was 0.026458 s
(about 25.7%).

The check gates correctness only.
It has no wall-time assertion.
The reference below uses the already validated raw block assembly route
plus an explicit sparse diagonal ``inv(B)`` multiply.
It is not a snapshot of the optimized full-assembly output.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from examples.martini.matlab_baseflow import load_blasius
from linstabpy.baseflow import sutherland_air
from linstabpy.mesh import Mesh
from linstabpy.operators.assembly import (
    _assemble_block_operator,
    _assemble_from_prepared,
    _prepare_coefficients,
)


def _max_abs_sparse_delta(a: sp.spmatrix, b: sp.spmatrix) -> float:
    delta = (a - b).tocoo()
    if delta.nnz == 0:
        return 0.0
    return float(np.max(np.abs(delta.data)))


def test_assembly_orchestration_matches_validated_block_reference_within_1e_10() -> None:
    mesh = Mesh.create(
        xrange=(200.0, 300.0),
        yrange=(0.0, 30.0),
        nx=30,
        ny=20,
        fd_order=6,
    )
    baseflow = load_blasius(
        mesh.X,
        mesh.Y,
        Re=1e4,
        Ma=0.3,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
    )
    sutherland_air(baseflow)

    prep = _prepare_coefficients(mesh, baseflow, kx=0.25, model="2D", m=0.0)
    actual = _assemble_from_prepared(prep, floquet_exp=0.0)

    raw_reference = _assemble_block_operator(
        prep["ngp"],
        prep["coeffs"],
        prep["DR"],
        prep["D2R"],
        prep["DZ"],
        prep["D2Z"],
        prep["D2RZ"],
    )
    row_scaling = np.empty(5 * prep["ngp"])
    row_scaling[0 * prep["ngp"] : 1 * prep["ngp"]] = 1.0
    inv_rho = 1.0 / prep["RHO"]
    for block in range(1, 5):
        row_scaling[block * prep["ngp"] : (block + 1) * prep["ngp"]] = inv_rho
    reference = (sp.diags(row_scaling, format="csc") @ raw_reference).tocsc()

    max_abs_delta = _max_abs_sparse_delta(actual, reference)
    assert max_abs_delta <= 1e-10
