# SPDX-License-Identifier: GPL-3.0-only
"""Harmonic resolvent analysis via resolvent4py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import petsc4py.PETSc as PETSc


def harmonic_resolvent_gains(
    coo_fnames: list[tuple[str, str, str]],
    n: int,
    *,
    perts_freqs: np.ndarray,
    n_modes: int = 10,
    n_rand_cycles: int = 3,
    seed: int = 10,
    kappa: float = 0.0,
    eps_reg: float = 1e-7,
    dq_fnames: list[tuple[str, str, str]] | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute harmonic resolvent gains via resolvent4py.

    Assembles the block-Toeplitz operator T and computes singular values
    of T^{-1} (or projected T^{-1} if dQ is given).

    Parameters
    ----------
    coo_fnames : list[tuple[str, str, str]]
        List of (rows_path, cols_path, vals_path) for k=-N_h..N_h,
        written by write_resolvent4py_coo.
    n : int
        DOF per harmonic block (n_dof / (2*N_p+1) where N_p = len(perts_freqs)//2).
    perts_freqs : np.ndarray
        Perturbation frequencies omega_j, length 2*N_p+1.
        Default ladder: arange(-N_p, N_p+1)*omega0 with N_p >= N_h+3.
    n_modes : int, optional
        Number of leading singular values/modes. Default 10.
    n_rand_cycles : int, optional
        Randomized SVD cycles. Default 3.
    seed : int, optional
        Random seed for SVD. Default 10.
    kappa : float, optional
        Discount parameter (shift A -> A - kappa*I). Default 0.
    eps_reg : float, optional
        Regularization epsilon added to T diagonal before factoring.
        Default 1e-7 (same as resolvent4py toy).
    dq_fnames : list[tuple[str, str, str]], optional
        If given, list of (rows, cols, vals) paths for dQ harmonics.
        Enables phase-shift projectors P_d and P_r.

    Returns
    -------
    gains : np.ndarray
        Shape (n_modes,), leading singular values of T^{-1} or P_r T^{-1} P_d.
    info : dict
        Metadata: eps_reg, kappa, projected (bool), N_p, N_h, sizes.

    Raises
    ------
    ImportError
        If resolvent4py or petsc4py not available.
    """
    try:
        from resolvent4py.solvers.harmonic_resolvent import (
            assemble_harmonic_resolvent_generator,
            check_lu_factorization,
            create_mumps_solver,
        )
        from resolvent4py.utils.io import read_harmonic_balanced_matrix
        from resolvent4py.utils.operator import (
            MatrixLinearOperator,
            ProductLinearOperator,
        )
        from resolvent4py.utils.svd import randomized_svd
    except ImportError as e:
        raise ImportError(
            "resolvent4py not found. Install with: pip install linstabpy[harmonic]"
        ) from e

    try:
        import petsc4py.PETSc as PETSc
    except ImportError as e:
        raise ImportError(
            "petsc4py not found. Install with: pip install linstabpy[harmonic]"
        ) from e

    N_h = len(coo_fnames) - 1
    N_p = (len(perts_freqs) - 1) // 2

    # Full system DOF
    N_full = (2 * N_p + 1) * n

    # Read harmonic balanced matrix
    block_sizes = ((N_full, n), (N_full, n))
    full_sizes = ((N_full, N_full), (N_full, N_full))

    # Read with real_bflow=False because the blocks store ±k independently.
    # At kx=0, Ahat[-k] = conj(Ahat[k]), but at kx!=0 they differ by ~6.6e-2
    # because A(t; kx) is genuinely complex through i*kx derivatives.
    A = read_harmonic_balanced_matrix(
        coo_fnames,
        real_bflow=False,
        block_sizes=block_sizes,
        full_sizes=full_sizes,
    )

    # Discount shifts the operator: A -> A - kappa*I. This gives the k=0
    # block a global shift of -(kappa + i*omega_j) at each frequency omega_j,
    # matching the convention of the classical L_sigma - (gamma + i*omega)*I.
    if kappa > 0:
        A.shift(-kappa)

    # Assemble generator T = A - i*omega*I
    T = assemble_harmonic_resolvent_generator(A, perts_freqs, M=None)

    # Scale by -1 and add regularization. This exactly mirrors resolvent4py's
    # toy convention: T.scale(-1.0) produces -(A - i*omega*I) = -A + i*omega*I,
    # then +eps*I adds the regularization. Drifting from this will break parity.
    # The k=0 block after scaling is -(A_0 - i*omega*I), which gains a global
    # sign against linstabpy's L_sigma. Gains are sign-invariant; modes carry
    # only a global phase and are physically equivalent.
    T.scale(-1.0)
    T += eps_reg * PETSc.Mat().createIdentity(T.getSize()[0])

    # MUMPS factorization
    ksp = create_mumps_solver(T)
    check_lu_factorization(ksp)

    # Operator interface
    Top = MatrixLinearOperator(T, ksp, nblocks=2 * N_p + 1)

    # Projectors (if dQ given)
    projected = False
    if dq_fnames is not None:
        from resolvent4py.utils.operator import ProjectionLinearOperator

        dQ_vec = read_harmonic_balanced_vector(
            dq_fnames,
            block_sizes=block_sizes,
            full_sizes=full_sizes,
        )

        # Normalize dQ
        dQ_norm = np.sqrt(np.real(dQ_vec.dot(dQ_vec)))
        dQ_vec = dQ_vec / dQ_norm

        # Left projector: solve T^H dQ
        dQ_left = Top.solve_hermitian_transpose(dQ_vec)
        dQ_left_norm = np.sqrt(np.real(dQ_left.dot(dQ_left)))
        dQ_left = dQ_left / dQ_left_norm

        # Build projectors
        Pd = ProjectionLinearOperator(dQ_vec, True, nblocks=2 * N_p + 1)
        Pr = ProjectionLinearOperator(dQ_left, True, nblocks=2 * N_p + 1)

        # Composed: H = Pr * T^{-1} * Pd
        H = ProductLinearOperator([Pr, Top, Pd], nblocks=2 * N_p + 1)
        projected = True
    else:
        # Unprojected: H = T^{-1}
        H = Top

    # Randomized SVD
    gains, _, _ = randomized_svd(
        H,
        n_components=min(n_modes, N_full),
        n_iter=n_rand_cycles,
        random_state=seed,
    )

    info = {
        "eps_reg": eps_reg,
        "kappa": kappa,
        "projected": projected,
        "N_p": N_p,
        "N_h": N_h,
        "N_full": N_full,
        "n": n,
        "n_modes": n_modes,
    }

    return gains, info


def read_harmonic_balanced_vector(
    coo_fnames: list[tuple[str, str, str]],
    block_sizes: tuple[tuple[int, int], tuple[int, int]],
    full_sizes: tuple[tuple[int, int], tuple[int, int]],
) -> PETSc.Vec:
    """Read a harmonic-mode vector from COO files.

    Wrapper around resolvent4py's reader for dQ-like vectors.

    Parameters
    ----------
    coo_fnames : list[tuple[str, str, str]]
        List of (rows, cols, vals) paths for harmonics.
    block_sizes : tuple
        Passed to resolvent4py.
    full_sizes : tuple
        Passed to resolvent4py.

    Returns
    -------
    PETSc.Vec
        Stacked vector of length (2*N_p+1)*n.
    """
    from resolvent4py.utils.io import read_harmonic_balanced_vector as r4py_read

    return r4py_read(coo_fnames, block_sizes, full_sizes)
