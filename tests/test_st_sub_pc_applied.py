# SPDX-License-Identifier: GPL-3.0-only
"""Check that the ST sub-preconditioner is the one the config asked for.

Background.
``_configure_st_for_eigenvalue`` used to configure the ASM/BJacobi sub-PC by
calling ``getASMSubKSP()`` immediately, before ``eps.setUp()``.
PETSc only materialises sub-solvers once the caller attaches the operator,
so that call raises ``ierr=58``.
The except-path swallowed the error and discarded both ``sub_pc_type``
and ``pc_factor_levels``.

The failure was invisible for a specific reason worth remembering.
PETSc's *default* sub-PC is also ``ilu``, so the configured **type**
appeared to match.
Only the fill level differed -- a config asking for ILU(2) silently ran
with ILU(0).
Nothing was off about the converged eigenvalues.
What was off was the iteration count, and therefore any cost figure
attributed to that configuration.

These checks pin both halves: that deferral is *necessary*, and that after
deferral the realized stack matches the requested one.
"""

from __future__ import annotations

import pytest

petsc4py = pytest.importorskip("petsc4py", reason="SKIPPED: petsc4py not installed")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from petsc4py import PETSc  # noqa: E402

from linstabpy.solvers.eigenvalue import _make_sub_pc_applier  # noqa: E402
from linstabpy.solvers.petsc_utils import scipy_to_petsc  # noqa: E402


def _asm_ksp(n: int = 40):
    """A small KSP with an ASM preconditioner, operator attached but NOT set up."""
    operator = (sp.diags([2.0] * n) + sp.eye(n) * 0.1).tocsr().astype(complex)
    mat = scipy_to_petsc(operator)
    ksp = PETSc.KSP().create()
    ksp.setOperators(mat)
    pc = ksp.getPC()
    pc.setType("asm")
    pc.setASMOverlap(1)
    return ksp, pc


def test_sub_ksp_query_fails_before_setup() -> None:
    """Pin why the applier waits until after setUp, as a fact rather than a comment.

    If a future PETSc makes this work before setUp, this check fails.
    Then we can revisit the deferral deliberately -- instead of the
    code carrying a stale rationale.
    """
    _ksp, pc = _asm_ksp()
    with pytest.raises(PETSc.Error):
        pc.getASMSubKSP()


@pytest.mark.parametrize("levels", [0, 1, 2, 3])
def test_configured_fill_level_is_realized(levels: int) -> None:
    """ILU(k) requested must be ILU(k) applied -- this is the mismatch that shipped."""
    ksp, pc = _asm_ksp()
    apply = _make_sub_pc_applier(pc=pc, pc_type="asm", sub_pc_type="ilu", factor_levels=levels)
    ksp.setUp()

    realized = apply()

    assert realized["sub_pc_type"] == "ilu"
    assert realized["pc_factor_levels"] == levels
    assert realized["n_sub_ksp"] >= 1

    sub_pc = pc.getASMSubKSP()[0].getPC()
    assert sub_pc.getType() == "ilu"


def test_unrealizable_sub_pc_raises_instead_of_downgrading() -> None:
    """A sub-PC that cannot exist must raise, not fall back to a default.

    The original code's ``except PETSc.Error: pass`` is precisely the shape
    excluded here.
    A preconditioner quietly weaker than the one requested makes reported
    iteration counts unattributable to the configuration that supposedly
    produced them.
    """
    ksp, pc = _asm_ksp()
    apply = _make_sub_pc_applier(
        pc=pc, pc_type="asm", sub_pc_type="definitely-not-a-pc", factor_levels=0
    )
    ksp.setUp()

    with pytest.raises((RuntimeError, PETSc.Error, ValueError)):
        apply()


def test_no_silent_default_when_levels_differ() -> None:
    """0 and 2 must produce different realized values, not both the built-in default.

    A first version of this verification passed the config keys at the
    InexactCfg nesting, so every run reported factor_levels=2.
    That is the InexactCfg default, and it looked correct.
    Distinguishing two values is what makes the check meaningful.
    """
    seen = []
    for levels in (0, 2):
        ksp, pc = _asm_ksp()
        apply = _make_sub_pc_applier(pc=pc, pc_type="asm", sub_pc_type="ilu", factor_levels=levels)
        ksp.setUp()
        seen.append(apply()["pc_factor_levels"])
    assert seen == [0, 2], seen
    assert len(set(seen)) == 2, "realized levels collapsed to a single value"
