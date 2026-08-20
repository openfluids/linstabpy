# SPDX-License-Identifier: GPL-3.0-only
"""Compressed-anchor sweep tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from linstabpy.linalg.backends import factor_backend_available
from linstabpy.linalg.factorize import factorize
from linstabpy.solvers.router import stable_sweep


class _FakeCompressedFactor:
    def __init__(self, matrix: sp.spmatrix, *, backend: str) -> None:
        self.matrix = sp.csc_matrix(matrix, dtype=complex)
        self.lu = spla.splu(self.matrix)
        self.backend = backend.upper()
        self.metrics = {
            "memory_bytes": 40.0 if backend == "strumpack" else 80.0,
            "effective_rank": 3.0 if backend == "strumpack" else -1.0,
            "refactor_count": 0,
            "refactor_seconds": 0.0,
            "solve_block_calls": 0,
            "solve_block_rhs": 0,
            "solve_block_seconds": 0.0,
        }

    def solve(self, rhs):  # noqa: ANN001, ANN201
        return self.lu.solve(np.asarray(rhs, dtype=complex))

    def solve_adjoint(self, rhs):  # noqa: ANN001, ANN201
        return self.lu.solve(np.asarray(rhs, dtype=complex), trans="H")

    def solve_block(self, rhs_block: np.ndarray) -> np.ndarray:
        block = np.asarray(rhs_block, dtype=complex)
        return np.column_stack([self.solve(block[:, j]) for j in range(block.shape[1])])

    def destroy(self) -> None:
        return None


def _operator(n: int = 6) -> sp.csc_matrix:
    diagonal = 3.0 + 0.25 * np.arange(n) + 1j * (0.15 - 0.03 * np.arange(n))
    lower = 0.04 - 0.02j + 0.005 * np.arange(n - 1)
    upper = -0.03 + 0.015j - 0.004 * np.arange(n - 1)
    far = np.full(n - 2, 0.01 + 0.01j, dtype=complex)
    return sp.diags(
        [far, lower, diagonal, upper, -0.5 * far], [-2, -1, 0, 1, 2], format="csc", dtype=complex
    )


def _cfg(n: int) -> dict:
    eye = np.eye(n, dtype=complex)
    return {
        "sigma_tol": 1.0e-6,
        "inv_norm_est": 1.0,
        "actual_solve_tol": 1.0e-10,
        "fgmres_rtol": 1.0e-11,
        "fgmres_maxiter": n,
        "n_modes": 1,
        "oversample": n,
        "seeds": (11, 12, 13),
        "probes_a": (eye[0], eye[n // 2]),
        "probes_b": (eye[1], eye[-1]),
        "W": np.ones(n, dtype=complex),
        "invW": np.ones(n, dtype=complex),
        "linearity_tol": 1.0e-8,
        "seed_gain_tol": 1.0e-6,
        "seed_mode_tol": 1.0e-5,
        "compressed_anchor": {
            "enabled": True,
            "compression": "blr",
            "rel_tol": 1.0e-8,
        },
    }


def _backend_from_solver_cfg(solver_cfg: dict | None) -> str:
    raw = (solver_cfg or {}).get("solver", solver_cfg or {})
    section = raw.get("resolvent", raw.get("linsolve", {}))
    return section.get("factor_backend", "mumps")


def test_compressed_anchor_branch_uses_approx_factor_adjoint_and_reports_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    L0 = _operator()
    omegas = np.array([0.95, 1.0], dtype=complex)
    anchors = np.array([1.0], dtype=complex)
    cfg = _cfg(L0.shape[0])
    cfg["reuse_symbolic"] = False
    seen_cert_cfgs = []
    real_certified_rsvd = stable_sweep.certified_rsvd

    def fake_factorize(operator, *, solver_cfg=None, **_kwargs):  # noqa: ANN001, ANN003
        return _FakeCompressedFactor(operator, backend=_backend_from_solver_cfg(solver_cfg))

    def recording_certified_rsvd(apply_h, apply_h_adj, W, invW, cert_cfg):  # noqa: ANN001
        seen_cert_cfgs.append(cert_cfg)
        return real_certified_rsvd(apply_h, apply_h_adj, W, invW, cert_cfg)

    monkeypatch.setattr(stable_sweep, "factorize", fake_factorize)
    monkeypatch.setattr(
        stable_sweep, "factor_backend_available", lambda backend: backend == "mumps"
    )
    monkeypatch.setattr(stable_sweep, "certified_rsvd", recording_certified_rsvd)

    result = stable_sweep.sweep_stable_regime(L0, omegas, anchors, cfg)

    assert seen_cert_cfgs
    for cert_cfg in seen_cert_cfgs:
        assert cert_cfg["approx_solve"].__self__ is cert_cfg["approx_solve_adjoint"].__self__
        assert cert_cfg["approx_solve_adjoint"].__name__ == "solve_adjoint"
        assert cert_cfg["adjoint_factor_tol"] == pytest.approx(cfg["actual_solve_tol"])
    assert all(
        report.checks["adjoint_of_approx_factor"].passed for report in result.certification_reports
    )
    assert result.metrics["compressed_anchor_requested"] is True
    assert result.metrics["compression_ratio"] == pytest.approx(0.5)
    assert result.metrics["effective_rank"] == pytest.approx(3.0)
    assert "compression helped" in result.metrics["compression_note"]


def _strumpack_adjoint_identity_defect(L0: sp.spmatrix, *, compressed: bool) -> float:
    """Relative defect of the adjoint identity <v, A^{-1}u> == <(A^{-H})v, u> for the
    STRUMPACK factor's own forward/adjoint solves.  This is the quantity S7 check #3
    certifies on the *approximate* factor."""
    solver_cfg = stable_sweep._with_direct_factor_backend(
        {"solver": {"resolvent": {"type": "direct", "factor_backend": "strumpack"}}}, "strumpack"
    )
    factorize_cfg = stable_sweep._factorize_cfg_with_backend(None, solver_cfg, "strumpack")
    options = (
        stable_sweep._compressed_anchor_controls(
            {"compressed_anchor": {"enabled": True, "compression": "blr", "rel_tol": 1.0e-8}}
        )[1]
        if compressed
        else {}
    )
    rng = np.random.default_rng(0)
    n = L0.shape[0]
    u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    with stable_sweep._temporary_petsc_options(options):
        factor = factorize(
            stable_sweep.shifted_operator(L0, 1.0 + 0j),
            env_prefix="LINSTABPY_RESOLVENT",
            solver_cfg=solver_cfg,
            cfg=factorize_cfg,
        )
    try:
        lhs = np.vdot(v, factor.solve(u))
        rhs = np.vdot(factor.solve_adjoint(v), u)
    finally:
        factor.destroy()
    return float(abs(lhs - rhs) / max(abs(lhs), np.finfo(float).tiny))


def test_compressed_anchor_strumpack_certifies_via_dual_factor_heal() -> None:
    """UPDATED 2026-06-23 (bead r3uu.14.17 — dual-factor adjoint heal) — validated on c9.

    HISTORICAL FINDING (bead r3uu.14.9, measured on c9 2026-06-16).
    STRUMPACK's complex transpose-solve is a non-conjugate transpose.
    The bare adjoint identity defect was ~3.2e-2 (MUMPS ~3e-16).
    The compressed-anchor sweep correctly REFUSED rather than use an
    uncertified adjoint.

    That backend-adjoint blocker is now REMOVED.
    ``factorize`` auto-detects the adjoint defect on first adjoint use.
    For a defective backend it FORWARD-solves an explicitly factored ``K^H``
    (the dual factorization).
    Measured on real STRUMPACK (c9, 2026-06-23).
    The raw transpose-solve defect is ~0.72.
    The heal activates.
    The healed adjoint identity drops to ~6e-16.
    The compressed-anchor sweep now CERTIFIES (returns a ``SweepResult``)
    instead of refusing.

    CAVEAT (capacity lever).
    The heal builds the ``K^H`` factor at full accuracy (uncompressed).
    The certified adjoint is exact but does NOT inherit BLR compression.
    The memory saving applies to the forward factor, not the healed adjoint.
    """
    if not factor_backend_available("strumpack"):
        pytest.skip("SKIPPED: strumpack backend unavailable")
    if not factor_backend_available("mumps"):
        pytest.skip("SKIPPED: mumps backend unavailable")

    L0 = _operator()
    omegas = np.array([1.0], dtype=complex)
    anchors = np.array([1.0], dtype=complex)
    cfg = _cfg(L0.shape[0])

    # The dual-factor heal certifies the adjoint, so the sweep no longer refuses.
    result = stable_sweep.sweep_stable_regime(L0, omegas, anchors, cfg)
    assert result is not None

    # The healed adjoint identity is at machine precision (was ~3.2e-2 unhealed).
    defect = _strumpack_adjoint_identity_defect(L0, compressed=False)
    assert defect < 1.0e-10
