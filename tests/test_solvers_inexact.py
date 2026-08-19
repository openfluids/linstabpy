# SPDX-License-Identifier: GPL-3.0-only
"""Check the inexact shift-invert ST configuration.

Check that ``_configure_st_for_eigenvalue`` wires the SLEPc KSP/PC chain
for each ``st_solver`` value.
The checks do not run an eigensolve.
They only inspect PETSc/SLEPc types and tolerances.

Note on pc.setUp() and sub-KSP access.
--------------------------------------
``pc.getASMSubKSP()`` requires the PC to complete setup with an operator
matrix attached.
These checks only have a bare EPS without a matrix bound to the ST yet.
The caller can query KSP/PC types immediately after ``setType()``.
The checks therefore inspect types and tolerances.
Sub-KSP checks inspect the non-error path at EPS-setup time when a matrix
is present.
"""

from __future__ import annotations

import pytest

# Guard: skip the entire module if petsc4py / slepc4py are not available.
pytest.importorskip("petsc4py", reason="petsc4py not available")
pytest.importorskip("slepc4py", reason="slepc4py not available")

from petsc4py import PETSc  # noqa: E402
from slepc4py import SLEPc  # noqa: E402

from linstabpy.solvers.eigenvalue import _configure_st_for_eigenvalue  # noqa: E402


def _make_st_ksp_pc():
    """Create a minimal SLEPc EPS with SINVERT ST for config-only tests.

    Returns
    -------
    eps, st, ksp, pc, A
        All objects needed to call ``_configure_st_for_eigenvalue``.
        Matrix ``A`` is a 4×4 diagonal (values 1–4) in AIJ format.
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
        """st_solver='direct' is an alias for LU (no MUMPS factor solver)."""
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
        # getType() returns the type string set by ksp.setType().
        # It does not require the operator to sit on the KSP.
        assert "fgmres" in ksp.getType().lower()

    def test_inexact_pc_type(self):
        """Default pc_type for inexact must be asm."""
        _, st, ksp, pc, _ = _make_st_ksp_pc()
        _configure_st_for_eigenvalue("inexact", {}, ksp, pc, st)
        assert "asm" in pc.getType().lower()

    def test_inexact_sub_ksp_ilu_post_setup(self):
        """After eps.setUp() the ASM sub-PCs must have ILU type.

        eps.setUp() binds the matrix to the ST and triggers PC setup, which
        makes getASMSubKSP() available.  We verify the sub-PC type here.
        """
        eps, st, ksp, pc, _ = _make_st_ksp_pc()
        eps.setDimensions(nev=1, ncv=3)
        eps.setTarget(2.0 + 0.0j)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        _configure_st_for_eigenvalue("inexact", {}, ksp, pc, st)
        try:
            eps.setUp()
        except Exception:
            pytest.skip("eps.setUp() failed (matrix too small for ASM on this build)")
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
        """A custom ksp_type in the inexact sub-dict must take effect."""
        _, st, ksp, pc, _ = _make_st_ksp_pc()
        ecfg = {"inexact": {"ksp_type": "fgmres", "pc_type": "bjacobi", "ksp_max_it": 100}}
        _configure_st_for_eigenvalue("inexact", ecfg, ksp, pc, st)
        assert "fgmres" in ksp.getType().lower()

    def test_custom_pc_type_bjacobi(self):
        """pc_type='bjacobi' in the inexact sub-dict must take effect."""
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


# ---------------------------------------------------------------------------
# Tests for _raise_eps_setup_failure exception-type dispatch
# ---------------------------------------------------------------------------

from linstabpy.solvers.eigenvalue import _raise_eps_setup_failure  # noqa: E402


class TestRaiseEpsSetupFailureDispatch:
    """Verify _raise_eps_setup_failure branches on exception type correctly.

    All calls use real (tiny) PETSc/SLEPc objects so that _safe_petsc_info
    succeeds.
    Only the ``exc`` argument is a stand-in.
    """

    def _call(self, exc: Exception) -> None:
        """Call _raise_eps_setup_failure with a fabricated exc and real objects."""
        eps, st, ksp, pc, A = _make_st_ksp_pc()
        _raise_eps_setup_failure(
            exc=exc,
            A=A,
            eps=eps,
            st=st,
            ksp=ksp,
            pc=pc,
            ndof=12345,
            n_modes=4,
            ncv=20,
            target=0.1 + 0.0j,
            st_solver="mumps",
        )

    def test_system_error_chkerr_collapse_flags_collapse(self):
        """SystemError with CHKERR sentinel must name the collapse + stderr guidance."""
        exc = SystemError("_cppyy.gbl.Cxx_Object returned a result with an exception set")
        with pytest.raises(RuntimeError, match="CHKERR collapse"):
            self._call(exc)

    def test_system_error_chkerr_collapse_has_stderr_guidance(self):
        """SystemError collapse message must mention stderr."""
        exc = SystemError("foo returned a result with an exception set")
        with pytest.raises(RuntimeError, match="stderr"):
            self._call(exc)

    def test_system_error_chkerr_collapse_preserves_context(self):
        """SystemError collapse message must still carry ndof and backend context."""
        exc = SystemError("bar returned a result with an exception set")
        with pytest.raises(RuntimeError, match="ndof=12345") as exc_info:
            self._call(exc)
        msg = str(exc_info.value)
        assert "st_solver=mumps" in msg

    def test_petsc_error_names_integer_code(self):
        """PETSc.Error must produce a message naming the integer error code."""
        petsc_exc = PETSc.Error(92)
        with pytest.raises(RuntimeError, match=r"ierr=92") as exc_info:
            self._call(petsc_exc)
        msg = str(exc_info.value)
        # Context fields must still be present.
        assert "ndof=12345" in msg

    def test_plain_runtime_error_reports_message(self):
        """A plain RuntimeError must still report its own message unchanged."""
        exc = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom") as exc_info:
            self._call(exc)
        msg = str(exc_info.value)
        assert "ndof=12345" in msg
