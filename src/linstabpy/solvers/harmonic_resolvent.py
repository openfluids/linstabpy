# SPDX-License-Identifier: GPL-3.0-only
"""Harmonic resolvent analysis via resolvent4py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from petsc4py import PETSc


def harmonic_resolvent_gains(
    coo_fnames: list[tuple[str, str, str]],
    n: int,
    *,
    perts_freqs: np.ndarray,
    n_modes: int = 10,
    n_rand_cycles: int = 3,
    n_rand: int = 30,
    gamma: float = 0.0,
    eps_reg: float = 1e-7,
    dq_fnames: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute harmonic resolvent gains via resolvent4py.

    Time convention: the same as the classical resolvent, ``exp(-i*omega*t)``.
    The perturbation is ``q(t) = sum_j q_j exp(-i*omega_j*t)`` and the base
    flow is ``A(t) = sum_k Ahat_k exp(-i*k*omega0*t)``. With
    ``omega_j = omega + j*omega0`` the ladder solves
    ``(gamma - i*omega_j) q_j - sum_k Ahat_k q_{j-k} = f_j``. At ``N_h = 0``
    the harmonic gain at ``omega_j`` equals the classical gain at the same
    ``omega_j``. resolvent4py uses ``exp(+i*omega*t)``. ``_harmonic_generator``
    maps the problem onto it, so the gains are those of the
    ``exp(-i*omega*t)`` problem.
    See ``docs/resolvent_sign_convention.md``.

    Assembles the block-Toeplitz operator T and computes singular values
    of T^{-1} (or the projected T^{-1} when the caller supplies dQ).

    The gains are in the plain (unweighted) 2-norm on the stacked ladder
    vector. The classical resolvent map reports Chu-norm gains by default,
    so at N_h = 0 the two agree only when the classical run uses
    ``W_diag = 1`` (identity weight). This path runs on one MPI rank.

    Parameters
    ----------
    coo_fnames : list[tuple[str, str, str]]
        List of (rows_path, cols_path, vals_path) for k=-N_h..N_h, where
        Ahat_k is the coefficient of ``exp(-i*k*omega0*t)``, written by
        write_resolvent4py_coo (two-sided storage).
    n : int
        DOF per harmonic block at single frequency.
    perts_freqs : np.ndarray
        Perturbation frequencies omega_j (perturbation ``exp(-i*omega_j*t)``),
        ascending with spacing omega0, length 2*N_p+1.
        Default ladder: arange(-N_p, N_p+1)*omega0 with N_p >= N_h+3.
    n_modes : int, optional
        Number of leading singular values to extract. Default 10.
    n_rand_cycles : int, optional
        Randomized SVD number of subspace iterations. Default 3.
        The sigma_1 estimate converges through the sigma_1/sigma_2 gap,
        so nearly degenerate leading singular values need more cycles and
        a larger sketch. Measured on a gap-free spectrum (sigma_1/sigma_2 =
        1.0): relative error 3.6e-2 at the defaults, 1.4e-16 with n_rand=100
        and n_rand_cycles=20 (measured on the reference compute stack).
    n_rand : int, optional
        Oversampling dimension for randomized SVD. Default 30.
        Clamped internally to min(n_rand, N_full).
    gamma : float, optional
        Discount (shifts A -> A - gamma*I), the same quantity as ``gamma``
        in the classical ``(gamma - i*omega) I - L0``. Any sign is applied.
        Default 0.
    eps_reg : float, optional
        Regularization epsilon added to T diagonal before factoring.
        Default 1e-7 (matches resolvent4py toy). Set to 0.0 for
        unperturbed solve (e.g., steady-state gate).
    dq_fnames : list[str], optional
        One Vec file per base harmonic k=-N_h..N_h
        (2*N_h+1 files), the coefficient of ``exp(-i*k*omega0*t)`` of dQ/dt.
        Reader centers them on the perturbation ladder (2*N_p+1 blocks).

    Returns
    -------
    gains : np.ndarray
        Leading singular values of T^{-1} or P_r T^{-1} P_d. The length is
        min(n_modes, n_rand, N_full): a randomized sketch cannot return
        more values than its own dimension.
    info : dict
        Metadata: eps_reg, gamma, projected (bool), N_p, N_h, N_full, n,
        n_modes, n_rand (used).

    Raises
    ------
    ImportError
        If resolvent4py or petsc4py not available.
    RuntimeError
        If more than one MPI rank runs it.
    """
    try:
        from resolvent4py import (
            check_lu_factorization,
            create_mumps_solver,
            read_harmonic_balanced_vector,
        )
        from resolvent4py.linalg import randomized_svd
        from resolvent4py.linear_operators import (
            MatrixLinearOperator,
            ProductLinearOperator,
            ProjectionLinearOperator,
        )
    except ImportError as e:
        message = "resolvent4py not found. Install with: pip install linstabpy[harmonic]"
        raise ImportError(message) from e

    try:
        from petsc4py import PETSc
    except ImportError as e:
        message = (
            "petsc4py not found. The [harmonic] extra does not install it; "
            "build the complex PETSc/SLEPc stack with: bash install.sh"
        )
        raise ImportError(message) from e

    # The local sizes below assume one rank; under mpiexec every rank would
    # build the whole ladder on COMM_WORLD with wrong layouts.
    if PETSc.COMM_WORLD.getSize() > 1:
        message = "harmonic_resolvent_gains runs on one MPI rank; run it without mpiexec"
        raise RuntimeError(message)

    # coo_fnames stores k=-N_h..N_h (two-sided, 2*N_h+1 blocks)
    N_h = (len(coo_fnames) - 1) // 2
    N_p = (len(perts_freqs) - 1) // 2

    # Full system DOF
    N_full = (2 * N_p + 1) * n

    # Local sizes for parallel PETSc (here: single-rank for simplicity)
    from resolvent4py import compute_local_size

    nl = compute_local_size(n)
    Nl = compute_local_size(N_full)

    T = _harmonic_generator(coo_fnames, n, perts_freqs, gamma, eps_reg)

    # MUMPS LU factorization
    ksp = create_mumps_solver(T)
    check_lu_factorization(T, ksp)

    # Operator interface: wrap T for action T^{-1}
    Top = MatrixLinearOperator(T, ksp, nblocks=2 * N_p + 1)

    # Projectors (if dQ given)
    projected = False
    if dq_fnames is not None:
        try:
            from slepc4py import SLEPc
        except ImportError as e:
            message = "slepc4py not found. The phase-shift projectors need SLEPc BV objects."
            raise ImportError(message) from e

        # Read dQ vector (single PETSc Vec file per harmonic, two-sided order).
        # The reader expects a flat list of file paths (one per harmonic).
        dQ = read_harmonic_balanced_vector(
            dq_fnames[::-1],
            real_bflow=False,
            block_sizes=(nl, n),
            full_sizes=(Nl, N_full),
        )
        dQ.scale(1.0 / dQ.norm())

        # Left direction: w = T^{-H} dQ, normalized. PETSc Vec has no "/"
        # operator; normalization goes through scale().
        w = Top.solve_hermitian_transpose(dQ)
        w.scale(1.0 / w.norm())

        # ProjectionLinearOperator takes one-column SLEPc BVs, not bare Vecs.
        # Mirror the toy exactly: Pd = I - w w* (from Psi), Pr = I - dQ dQ*
        # (from Phi). Padovan 2020 JFM section 2.1 defines these directions.
        Phi = SLEPc.BV().create(dQ.getComm())
        Phi.setSizes(dQ.getSizes(), 1)
        Phi.setType("mat")
        Psi = Phi.copy()
        Phi.insertVec(0, dQ)
        Psi.insertVec(0, w)

        Pd = ProjectionLinearOperator(Psi, Psi, complement=True, nblocks=2 * N_p + 1)
        Pr = ProjectionLinearOperator(Phi, Phi, complement=True, nblocks=2 * N_p + 1)

        # Composed operator: H = Pr * T^{-1} * Pd. The middle action is
        # Top.solve, so the product operator's APPLY is already the
        # projected inverse; its own solve_mat would invert the singular
        # projectors and must never be the SVD action.
        H = ProductLinearOperator(
            [Pr, Top, Pd],
            [Pr.apply, Top.solve, Pd.apply],
            nblocks=2 * N_p + 1,
        )
        action = H.apply_mat
        projected = True
    else:
        # Unprojected: H = T^{-1}, so the SVD action inverts T directly.
        H = Top
        action = Top.solve_mat

    # Randomized SVD with oversampling parameter clamped to available DOF.
    # The sketch delivers at most n_rand_used singular values, so the
    # request clamps to it as well; info reports the delivered count.
    n_rand_used = min(n_rand, N_full)
    n_svals_used = min(n_modes, n_rand_used)
    _u, S, _v = randomized_svd(
        H,
        action,
        n_rand=n_rand_used,
        n_loops=n_rand_cycles,
        n_svals=n_svals_used,
        verbose=0,
    )

    # Extract gains from singular values (already real for a normal operator)
    gains = np.diag(S).real

    info = {
        "eps_reg": eps_reg,
        "gamma": gamma,
        "projected": projected,
        "N_p": N_p,
        "N_h": N_h,
        "N_full": N_full,
        "n": n,
        "n_modes": n_svals_used,
        "n_rand": n_rand_used,
    }

    return gains, info


def _harmonic_generator(
    coo_fnames: list[tuple[str, str, str]],
    n: int,
    perts_freqs: np.ndarray,
    gamma: float,
    eps_reg: float,
) -> PETSc.Mat:
    """Assemble the resolvent4py matrix of the ``exp(-i*omega*t)`` ladder.

    The target ladder is ``(gamma - i*omega_j) q_j - sum_k Ahat_k q_{j-k} = f_j``
    with ``omega_j = perts_freqs[j + N_p]``. resolvent4py assembles
    ``(gamma + i*w_p) q_p - sum_k X_k q_{p-k}`` with the file k holding ``X_k``.
    Put ``w_p = -omega_{-p}`` and ``X_k = Ahat_{-k}``. Block ``p`` of the
    resolvent4py ladder is then harmonic ``j = -p`` and the equations are
    identical. Both maps reverse an ordered list: the COO files, and the
    negated frequencies. A block reversal does not change singular values.

    Block ``p``, counted from the centre, carries the harmonic ``j = -p``
    of the target ladder.

    Parameters
    ----------
    coo_fnames : list[tuple[str, str, str]]
        COO files of ``Ahat_k`` for k=-N_h..N_h.
    n : int
        DOF per harmonic block.
    perts_freqs : np.ndarray
        Ascending frequencies omega_j, length 2*N_p+1.
    gamma : float
        Discount.
    eps_reg : float
        Regularization added to the diagonal. 0 skips it.

    Returns
    -------
    PETSc.Mat
        The matrix whose inverse is the harmonic resolvent.
    """
    from petsc4py import PETSc
    from resolvent4py import (
        assemble_harmonic_resolvent_generator,
        compute_local_size,
        create_AIJ_identity,
        read_harmonic_balanced_matrix,
    )

    N_full = len(perts_freqs) * n
    nl = compute_local_size(n)
    Nl = compute_local_size(N_full)

    # Two-sided storage: at kx != 0 Ahat[-k] differs from conj(Ahat[k]),
    # so real_bflow=False reads all 2*N_h+1 files independently.
    A = read_harmonic_balanced_matrix(
        coo_fnames[::-1],
        real_bflow=False,
        block_sizes=((nl, n), (nl, n)),
        full_sizes=((Nl, N_full), (Nl, N_full)),
    )

    # The discount shifts A -> A - gamma*I. After the sign flip below, each
    # diagonal block is (gamma + i*w_p) I - A with w_p = -omega_{-p}.
    if gamma != 0.0:
        A.shift(-gamma)

    # The resolvent4py 1.0.1 wheel takes exactly two positional args
    # (A, freqs); upstream git main added a Mass parameter, but we pin 1.0.1.
    T = assemble_harmonic_resolvent_generator(A, -np.asarray(perts_freqs)[::-1])
    T.scale(-1.0)

    # Build the regularization as an identity and axpy it in: MatShift needs
    # diagonal entries that padded harmonics may lack.
    if eps_reg > 0.0:
        Id = create_AIJ_identity(PETSc.COMM_WORLD, T.getSizes())
        Id.scale(eps_reg)
        T.axpy(1.0, Id)
        Id.destroy()
    return T
