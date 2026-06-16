# SPDX-License-Identifier: GPL-3.0-only
"""Fixed-subspace recycling helpers for stable resolvent sweeps.

The routines here implement the algebraic core needed by the sweep router: a
right-preconditioned GMRES solve augmented by a *fixed* recycle subspace.  The
subspace is prepared outside each RSVD/certification pass and is never adapted
inside a solve call, preserving the linear map checked by S7 certification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray
Apply = Callable[[Array], Array]


@dataclass(frozen=True)
class PreparedRecycleSubspace:
    """Orthonormal image-space recycle basis for one shifted operator."""

    basis: Array
    image: Array
    setup_matvecs: int = 0

    @property
    def rank(self) -> int:
        return int(self.image.shape[1])


@dataclass(frozen=True)
class RecycledGmresResult:
    """Result and accounting from one recycled GMRES solve."""

    x: Array
    iterations: int
    residual: float
    matvecs: int
    recycle_rank: int
    candidate_basis: Array


def empty_recycle_basis(n: int) -> Array:
    """Return an empty ``n x 0`` complex recycle basis."""
    return np.empty((int(n), 0), dtype=complex)


def _as_basis(basis: Array | None, *, n: int | None) -> Array:
    if basis is None:
        if n is None:
            raise ValueError("n is required when basis is None")
        return empty_recycle_basis(n)
    arr = np.asarray(basis, dtype=complex)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"recycle basis must be 1-D or 2-D, got shape {arr.shape}")
    if n is not None and arr.shape[0] != int(n):
        raise ValueError(f"recycle basis row count {arr.shape[0]} does not match n={int(n)}")
    return arr


def combine_recycle_bases(
    *bases: Array | None,
    max_dim: int,
    n: int | None = None,
    orth_tol: float = 1.0e-12,
) -> Array:
    """Merge candidate bases by modified Gram-Schmidt, capped at ``max_dim``.

    Earlier bases have priority.  This keeps the carried recycle space bounded
    and deterministic while avoiding near-dependent columns.
    """
    max_dim = int(max_dim)
    if max_dim < 0:
        raise ValueError("max_dim must be nonnegative")
    inferred_n = n
    for basis in bases:
        if basis is not None:
            arr = np.asarray(basis, dtype=complex)
            inferred_n = int(arr.shape[0]) if arr.ndim > 1 else int(arr.size)
            break
    if inferred_n is None:
        raise ValueError("n is required when all bases are None")
    if max_dim == 0:
        return empty_recycle_basis(inferred_n)

    q_cols: list[Array] = []
    tol = float(orth_tol)
    for basis in bases:
        arr = _as_basis(basis, n=inferred_n)
        for j in range(arr.shape[1]):
            v = np.asarray(arr[:, j], dtype=complex).reshape(-1).copy()
            start_norm = float(np.linalg.norm(v))
            if not np.isfinite(start_norm) or start_norm == 0.0:
                continue
            for _ in range(2):
                for q in q_cols:
                    v = v - np.vdot(q, v) * q
            norm = float(np.linalg.norm(v))
            if norm > tol * max(start_norm, 1.0):
                q_cols.append(v / norm)
                if len(q_cols) >= max_dim:
                    return np.column_stack(q_cols)
    if not q_cols:
        return empty_recycle_basis(inferred_n)
    return np.column_stack(q_cols)


def prepare_recycle_subspace(
    matvec: Apply,
    basis: Array | None,
    *,
    n: int,
    max_dim: int,
    orth_tol: float = 1.0e-12,
) -> PreparedRecycleSubspace:
    """Prepare a fixed recycle subspace for the current shifted operator.

    ``basis`` contains solution-space vectors carried from a previous shifted
    system.  The returned ``image`` columns are orthonormal vectors ``C`` with
    matching transformed solution vectors ``U`` such that ``A @ U[:, j]`` is
    ``C[:, j]`` up to roundoff.  The setup matvecs are reported separately so
    recycling accounting includes the cost of forming ``C``.
    """
    raw = combine_recycle_bases(basis, max_dim=max_dim, n=n, orth_tol=orth_tol)
    if raw.shape[1] == 0:
        empty = empty_recycle_basis(n)
        return PreparedRecycleSubspace(basis=empty, image=empty, setup_matvecs=0)

    u_cols: list[Array] = []
    c_cols: list[Array] = []
    setup_matvecs = 0
    tol = float(orth_tol)
    for j in range(raw.shape[1]):
        u = np.asarray(raw[:, j], dtype=complex).reshape(-1).copy()
        c = np.asarray(matvec(u), dtype=complex).reshape(-1)
        setup_matvecs += 1
        if c.size != int(n):
            raise ValueError(f"matvec returned length {c.size}, expected {int(n)}")
        start_norm = float(np.linalg.norm(c))
        if not np.isfinite(start_norm) or start_norm == 0.0:
            continue
        for _ in range(2):
            for q_c, q_u in zip(c_cols, u_cols, strict=True):
                alpha = np.vdot(q_c, c)
                c = c - alpha * q_c
                u = u - alpha * q_u
        norm = float(np.linalg.norm(c))
        if norm > tol * max(start_norm, 1.0):
            c_cols.append(c / norm)
            u_cols.append(u / norm)

    if not c_cols:
        empty = empty_recycle_basis(n)
        return PreparedRecycleSubspace(basis=empty, image=empty, setup_matvecs=setup_matvecs)
    return PreparedRecycleSubspace(
        basis=np.column_stack(u_cols),
        image=np.column_stack(c_cols),
        setup_matvecs=setup_matvecs,
    )


def recycled_right_preconditioned_gmres(
    matvec: Apply,
    precondition: Apply,
    rhs: Array,
    *,
    rtol: float,
    maxiter: int,
    recycle: PreparedRecycleSubspace | None = None,
    candidate_dim: int = 0,
    orth_tol: float = 1.0e-12,
) -> RecycledGmresResult:
    """Right-preconditioned GMRES with a fixed GCRO-style recycle subspace."""
    b = np.asarray(rhs, dtype=complex).reshape(-1)
    n = b.size
    rhs_norm = float(np.linalg.norm(b))
    if rhs_norm == 0.0:
        return RecycledGmresResult(
            x=np.zeros(n, dtype=complex),
            iterations=0,
            residual=0.0,
            matvecs=0,
            recycle_rank=0 if recycle is None else recycle.rank,
            candidate_basis=empty_recycle_basis(n),
        )

    maxiter = int(maxiter)
    if maxiter < 0:
        raise ValueError("maxiter must be nonnegative")

    if recycle is None or recycle.rank == 0:
        u = empty_recycle_basis(n)
        c = empty_recycle_basis(n)
    else:
        u = _as_basis(recycle.basis, n=n)
        c = _as_basis(recycle.image, n=n)
        if u.shape[1] != c.shape[1]:
            raise ValueError("recycle basis and image ranks differ")

    if c.shape[1] > 0:
        initial_coeff = c.conj().T @ b
        x0 = u @ initial_coeff
        r = b - c @ initial_coeff
    else:
        x0 = np.zeros(n, dtype=complex)
        r = b.copy()

    beta = float(np.linalg.norm(r))
    if beta / rhs_norm <= float(rtol) or maxiter == 0:
        return RecycledGmresResult(
            x=x0,
            iterations=0,
            residual=beta / rhs_norm,
            matvecs=0,
            recycle_rank=int(c.shape[1]),
            candidate_basis=empty_recycle_basis(n),
        )

    q_vectors = [r / beta]
    z_vectors: list[Array] = []
    h = np.zeros((maxiter + 1, maxiter), dtype=complex)
    g = np.zeros(maxiter + 1, dtype=complex)
    g[0] = beta
    best_y = np.zeros(0, dtype=complex)
    best_residual = beta / rhs_norm
    best_iter = 0
    matvecs = 0

    for j in range(maxiter):
        z_raw = np.asarray(precondition(q_vectors[j]), dtype=complex).reshape(-1)
        v = np.asarray(matvec(z_raw), dtype=complex).reshape(-1)
        matvecs += 1
        if c.shape[1] > 0:
            coeff = c.conj().T @ v
            v = v - c @ coeff
            z = z_raw - u @ coeff
        else:
            z = z_raw
        z_vectors.append(z)

        for i, q in enumerate(q_vectors):
            h[i, j] = np.vdot(q, v)
            v = v - h[i, j] * q
        h[j + 1, j] = np.linalg.norm(v)
        if h[j + 1, j] > 0.0 and j + 1 < maxiter:
            q_vectors.append(v / h[j + 1, j])

        y, *_ = np.linalg.lstsq(h[: j + 2, : j + 1], g[: j + 2], rcond=None)
        residual = float(np.linalg.norm(g[: j + 2] - h[: j + 2, : j + 1] @ y) / rhs_norm)
        best_y = y
        best_residual = residual
        best_iter = j + 1
        if residual <= float(rtol):
            break
        if h[j + 1, j] == 0.0:
            break

    x = x0.copy()
    for coeff, z in zip(best_y, z_vectors, strict=False):
        x = x + coeff * z

    candidate_basis = (
        combine_recycle_bases(
            np.column_stack(z_vectors) if z_vectors else None,
            max_dim=int(candidate_dim),
            n=n,
            orth_tol=orth_tol,
        )
        if int(candidate_dim) > 0
        else empty_recycle_basis(n)
    )
    return RecycledGmresResult(
        x=x,
        iterations=best_iter,
        residual=best_residual,
        matvecs=matvecs,
        recycle_rank=int(c.shape[1]),
        candidate_basis=candidate_basis,
    )


__all__ = [
    "PreparedRecycleSubspace",
    "RecycledGmresResult",
    "combine_recycle_bases",
    "empty_recycle_basis",
    "prepare_recycle_subspace",
    "recycled_right_preconditioned_gmres",
]
