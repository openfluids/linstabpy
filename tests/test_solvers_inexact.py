"""Plumb-tests for the inexact shift-invert ST configuration.

These tests verify that ``_configure_st_for_eigenvalue`` wires up the
SLEPc KSP/PC chain correctly for each ``st_solver`` value.  No actual
eigensolve is performed; we only check that the PETSc/SLEPc objects are
configured with the expected types and tolerances.
"""

from __future__ import annotations

import pytest

# Guard: skip the entire module if petsc4py / slepc4py are not available.
# On machines without MPI (e.g. macOS CI without openmpi) these imports fail.
pytest.importorskip("petsc4py", reason="petsc4py not available")
pytest.importorskip("slepc4py", reason="slepc4py not available")

from petsc4py import PETSc  # noqa: E402
from slepc4py import SLEPc  # noqa: E402

from linstab2d.solvers.eigenvalue import _configure_st_for_eigenvalue  # noqa: E402


def _make_st_ksp_pc():
    """Create a minimal SLEPc EPS with SINVERT ST for config-only tests.

    Returns
    -------
    eps, st, ksp, pc, A
        All SLEPc/PETSc objects needed to call ``_configure_st_for_eigenvalue``.
        The matrix ``A`` is a 4×4 diagonal (values 1–4) in AIJ format.
    """
    A = PETSc.Mat().create()
    A.setSizes([4, 4])
    A.setType(PETSc.Mat.Type.AIJ)
    A.setUp()
    for i in range(4):
        A.setValue(i, i, float(i + 1))
    A.assemble()

    eps = SLEPc.EPS().create()
    eps.setOperators(A)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(0.0 + 0.0j)
    ksp = st.getKSP()
    pc = ksp.getPC()
    return eps, st, ksp, pc, A


class TestDefaultMumpsPath:
    """Verify that st_solver='mumps' wires up PREONLY + LU."""

    def test_default_st_solver_is_mumps(self):
        """With st_solver='mumps' the KSP must be PREONLY and PC must be LU."""
        _, st, ksp, pc, _ = _make_st_ksp_pc()
        _configure_st_for_eigenvalue("mumps", {}, ksp, pc, st)
        assert ksp.getType() == PETSc.KSP.Type.PREONLY
        assert pc.getType() == PETSc.PC.Type.LU

    def test_direct_alias(self):
        """st_solver='direct' is an alias for MUMPS-style LU (no factor solver)."""
        _, st, ksp, pc, _ = _make_st_ksp_pc()
        _configure_st_for_eigenvalue("direct", {}, ksp, pc, st)
        assert ksp.getType() == PETSc.KSP.Type.PREONLY
        assert pc.getType() == PETSc.PC.Type.LU


class TestInexactBranchPlumbs:
    """Verify inexact-SI default config: FGMRES + ASM."""

    def test_inexact_ksp_type(self):
        """Default ksp_type for inexact must be fgmres."""
        _, st, ksp, pc, _ = _make_st_ksp_pc()
        _configure_st_for_eigenvalue("inexact", {}, ksp, pc, st)
        assert "fgmres" in ksp.getType().lower()

    def test_inexact_pc_type(self):
        """Default pc_type for inexact must be asm."""
        _, st, ksp, pc, _ = _make_st_ksp_pc()
        _configure_st_for_eigenvalue("inexact", {}, ksp, pc, st)
        assert "asm" in pc.getType().lower()

    def test_inexact_sub_ksp_ilu(self):
        """After setUp the ASM sub-PCs must have ILU type."""
        _, st, ksp, pc, _ = _make_st_ksp_pc()
        _configure_st_for_eigenvalue("inexact", {}, ksp, pc, st)
        # _configure_st_for_eigenvalue calls pc.setUp() internally for ASM.
        try:
            sub_ksps = pc.getASMSubKSP()
        except PETSc.Error:
            pytest.skip("getASMSubKSP not available in this PETSc build")
        if sub_ksps:
            sub_pc = sub_ksps[0].getPC()
            assert "ilu" in sub_pc.getType().lower()


class TestInexactOverrides:
    """Verify custom values propagate through the inexact config dict."""

    def test_custom_ksp_type(self):
        """A custom ksp_type in the inexact sub-dict must be applied."""
        _, st, ksp, pc, _ = _make_st_ksp_pc()
        ecfg = {"inexact": {"ksp_type": "fgmres", "pc_type": "bjacobi", "ksp_max_it": 100}}
        _configure_st_for_eigenvalue("inexact", ecfg, ksp, pc, st)
        assert "fgmres" in ksp.getType().lower()

    def test_custom_pc_type_bjacobi(self):
        """pc_type='bjacobi' in inexact sub-dict must be applied."""
        _, st, ksp, pc, _ = _make_st_ksp_pc()
        ecfg = {"inexact": {"pc_type": "bjacobi"}}
        _configure_st_for_eigenvalue("inexact", ecfg, ksp, pc, st)
        assert "bjacobi" in pc.getType().lower()

    def test_tolerances_applied(self):
        """ksp_rtol / ksp_atol / ksp_max_it must propagate."""
        _, st, ksp, pc, _ = _make_st_ksp_pc()
        ecfg = {"inexact": {"ksp_rtol": 1e-6, "ksp_atol": 1e-9, "ksp_max_it": 42}}
        _configure_st_for_eigenvalue("inexact", ecfg, ksp, pc, st)
        rtol, atol, dtol, max_it = ksp.getTolerances()
        assert max_it == 42
        assert abs(rtol - 1e-6) < 1e-12
        assert abs(atol - 1e-9) < 1e-13

    def test_unknown_st_solver_raises(self):
        """An unrecognized st_solver value must raise ValueError."""
        _, st, ksp, pc, _ = _make_st_ksp_pc()
        with pytest.raises(ValueError, match="inexact"):
            _configure_st_for_eigenvalue("bogus_solver", {}, ksp, pc, st)
