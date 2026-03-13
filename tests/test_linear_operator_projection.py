"""Tests for ProjectionOperator.

A4: Oblique projection, complementary projection, adjoint, idempotency,
    orthogonal special case, rank > 1.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as la

from linstab2d.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from linstab2d.linear_operators import ProjectionOperator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_to_vec(v: np.ndarray) -> PETSc.Vec:
    return PETSc.Vec().createWithArray(np.ascontiguousarray(v, dtype=complex))


def _vec_to_np(v: PETSc.Vec) -> np.ndarray:
    return np.array(v.array_r, copy=True)


# ---------------------------------------------------------------------------
# A4: Projection operator tests
# ---------------------------------------------------------------------------

class TestProjectionOperator:
    """Test oblique and orthogonal projection operators."""

    def test_apply_oblique(self):
        """P x = Φ (Ψ*Φ)^{-1} Ψ* x matches dense reference."""
        rng = np.random.default_rng(1000)
        N, r = 20, 3
        Phi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
        Psi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))

        op = ProjectionOperator(Phi, Psi, complement=False)

        x_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = _np_to_vec(x_np)

        y = op.apply(x)
        y_np = _vec_to_np(y)

        # Dense reference.
        S = Psi.conj().T @ Phi
        Sinv = la.inv(S)
        P = Phi @ Sinv @ Psi.conj().T
        y_ref = P @ x_np

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()

    def test_apply_complement(self):
        """(I - P) x matches dense reference."""
        rng = np.random.default_rng(1010)
        N, r = 20, 3
        Phi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
        Psi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))

        op = ProjectionOperator(Phi, Psi, complement=True)

        x_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = _np_to_vec(x_np)

        y = op.apply(x)
        y_np = _vec_to_np(y)

        S = Psi.conj().T @ Phi
        Sinv = la.inv(S)
        P = Phi @ Sinv @ Psi.conj().T
        y_ref = x_np - P @ x_np

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()

    def test_apply_adjoint(self):
        """P^H x = Ψ S^{-H} Φ* x matches dense reference."""
        rng = np.random.default_rng(1020)
        N, r = 20, 3
        Phi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
        Psi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))

        op = ProjectionOperator(Phi, Psi, complement=False)

        x_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = _np_to_vec(x_np)

        y = op.apply_adjoint(x)
        y_np = _vec_to_np(y)

        # Dense reference: P^H = (Φ S^{-1} Ψ*)^H = Ψ S^{-H} Φ*
        S = Psi.conj().T @ Phi
        P = Phi @ la.inv(S) @ Psi.conj().T
        y_ref = P.conj().T @ x_np

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()

    def test_apply_adjoint_complement(self):
        """(I - P)^H x matches dense reference."""
        rng = np.random.default_rng(1025)
        N, r = 20, 3
        Phi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
        Psi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))

        op = ProjectionOperator(Phi, Psi, complement=True)

        x_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = _np_to_vec(x_np)

        y = op.apply_adjoint(x)
        y_np = _vec_to_np(y)

        S = Psi.conj().T @ Phi
        P = Phi @ la.inv(S) @ Psi.conj().T
        IminusP_H = np.eye(N) - P.conj().T
        y_ref = IminusP_H @ x_np

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()

    def test_idempotency(self):
        """P^2 = P: applying the projector twice gives the same result."""
        rng = np.random.default_rng(1030)
        N, r = 15, 2
        Phi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
        Psi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))

        op = ProjectionOperator(Phi, Psi, complement=False)

        x_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = _np_to_vec(x_np)

        y1 = op.apply(x)
        y2 = op.apply(y1)

        np.testing.assert_allclose(
            _vec_to_np(y2), _vec_to_np(y1), atol=1e-13,
            err_msg="Projector is not idempotent"
        )
        y1.destroy()
        y2.destroy()
        x.destroy()

    def test_complement_idempotency(self):
        """(I-P)^2 = (I-P)."""
        rng = np.random.default_rng(1035)
        N, r = 15, 2
        Phi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
        Psi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))

        op = ProjectionOperator(Phi, Psi, complement=True)

        x_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = _np_to_vec(x_np)

        y1 = op.apply(x)
        y2 = op.apply(y1)

        np.testing.assert_allclose(
            _vec_to_np(y2), _vec_to_np(y1), atol=1e-13,
            err_msg="Complement projector is not idempotent"
        )
        y1.destroy()
        y2.destroy()
        x.destroy()

    def test_adjoint_inner_product_consistency(self):
        """<Px, z> = <x, P^H z> for all x, z."""
        rng = np.random.default_rng(1040)
        N, r = 15, 2
        Phi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
        Psi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))

        for complement in [False, True]:
            op = ProjectionOperator(Phi, Psi, complement=complement)

            x_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            z_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            x = _np_to_vec(x_np)
            z = _np_to_vec(z_np)

            Px = op.apply(x)
            PHz = op.apply_adjoint(z)

            lhs = np.vdot(_vec_to_np(Px), z_np)  # <Px, z>
            rhs = np.vdot(x_np, _vec_to_np(PHz))  # <x, P^H z>

            assert abs(lhs - rhs) < 1e-12, (
                f"Inner product inconsistency for complement={complement}: "
                f"|<Px,z> - <x,P^H z>| = {abs(lhs - rhs):.2e}"
            )
            Px.destroy()
            PHz.destroy()
            x.destroy()
            z.destroy()

    def test_orthogonal_projector(self):
        """When Phi = Psi, the projector is orthogonal: P = P^H."""
        rng = np.random.default_rng(1050)
        N, r = 15, 2
        Phi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
        # Orthogonalise Phi for a clean orthogonal projector.
        Phi, _ = la.qr(Phi, mode="economic")

        op = ProjectionOperator(Phi, Phi, complement=False)

        x_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = _np_to_vec(x_np)

        Px = op.apply(x)
        PHx = op.apply_adjoint(x)

        np.testing.assert_allclose(
            _vec_to_np(Px), _vec_to_np(PHx), atol=1e-13,
            err_msg="Orthogonal projector should be self-adjoint"
        )
        Px.destroy()
        PHx.destroy()
        x.destroy()

    def test_rank1_vector_input(self):
        """Accept 1-D arrays as rank-1 projector basis."""
        rng = np.random.default_rng(1060)
        N = 10
        phi = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        psi = rng.standard_normal(N) + 1j * rng.standard_normal(N)

        op = ProjectionOperator(phi, psi, complement=False)
        assert op.rank == 1

        x_np = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = _np_to_vec(x_np)

        y = op.apply(x)
        y_np = _vec_to_np(y)

        # Dense reference for rank-1: P x = phi * (psi^H x) / (psi^H phi)
        s = np.vdot(psi, phi)
        y_ref = phi * (np.vdot(psi, x_np) / s)

        assert np.linalg.norm(y_np - y_ref) / np.linalg.norm(y_ref) < 1e-12
        y.destroy()
        x.destroy()

    def test_complement_annihilates_range(self):
        """(I-P) Φ = 0: the complement projector annihilates vectors in range(P)."""
        rng = np.random.default_rng(1070)
        N, r = 15, 3
        Phi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
        Psi = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))

        op = ProjectionOperator(Phi, Psi, complement=True)

        # Any vector in range(P) = span(Phi) should be annihilated.
        for j in range(r):
            col = Phi[:, j]
            x = _np_to_vec(col)
            y = op.apply(x)
            assert np.linalg.norm(_vec_to_np(y)) < 1e-12, (
                f"Complement projector did not annihilate column {j} of Phi"
            )
            y.destroy()
            x.destroy()

    def test_shape_mismatch_raises(self):
        """Mismatched Phi/Psi dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="same number of rows"):
            ProjectionOperator(np.ones((10, 2)), np.ones((8, 2)))

        with pytest.raises(ValueError, match="same number of columns"):
            ProjectionOperator(np.ones((10, 2)), np.ones((10, 3)))
