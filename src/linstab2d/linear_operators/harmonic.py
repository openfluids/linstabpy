"""Harmonic resolvent infrastructure.

Provides harmonic layout, block assembly, and weight lifting for
harmonic-balanced operator formulations.

Mathematical background
-----------------------
Given a T-periodic linear operator A(t) with Fourier coefficients
{A_k}, k = -Nf, ..., 0, ..., +Nf, the harmonic-balanced operator is
the block-Toeplitz matrix:

    A_hat[i,j] = A_{i-j}     for |i-j| <= Nf, else 0

The harmonic resolvent generator is:

    T = -diag(i k omega I) + A_hat

where k ranges over -Nh, ..., 0, ..., +Nh (the perturbation harmonics)
and Nh >= Nf.

For a real-valued base flow, A_{-k} = conj(A_k), so only positive
harmonics need to be stored.

References
----------
Padovan, A., Otto, S. E., & Rowley, C. W. (2020). Analysis of
    amplification mechanisms and cross-frequency interactions in
    nonlinear flows via the harmonic resolvent. J. Fluid Mech., 900.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from ..utils.petsc_init import init_slepc_from_argv

init_slepc_from_argv()
from slepc4py import SLEPc

from ..utils.petsc_core import scipy_to_petsc, petsc_vec, vec_to_numpy
from .matrix import MatrixOperator
from .product import ProductOperator
from .petsc_shell import PetscShellOperator
from .adapters import scipy_to_solve_operator, DiagonalOperator
from .projection import ProjectionOperator


# ---------------------------------------------------------------------------
# C1: Harmonic layout / indexing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HarmonicLayout:
    """Bookkeeping for harmonic-balanced block operators.

    Parameters
    ----------
    Nh :
        Number of positive harmonics. The full harmonic set is
        k = -Nh, ..., 0, ..., +Nh.
    n :
        DOF per harmonic block (e.g. 5*ngp for 5-variable compressible LNSE).
    omega :
        Fundamental angular frequency of the base flow.
    """

    Nh: int
    n: int
    omega: float

    @property
    def nblocks(self) -> int:
        """Total number of harmonic blocks: 2*Nh + 1."""
        return 2 * self.Nh + 1

    @property
    def N(self) -> int:
        """Total DOF of the harmonic system."""
        return self.nblocks * self.n

    @property
    def harmonics(self) -> np.ndarray:
        """Array of harmonic indices: [-Nh, ..., 0, ..., +Nh]."""
        return np.arange(-self.Nh, self.Nh + 1)

    def block_index(self, k: int) -> int:
        """Block index (0-based) for harmonic *k* in [-Nh, Nh].

        Block ordering: k=-Nh is block 0, k=0 is block Nh, k=+Nh is block 2*Nh.
        """
        if abs(k) > self.Nh:
            raise ValueError(f"Harmonic k={k} is outside [-{self.Nh}, {self.Nh}]")
        return k + self.Nh

    def block_offset(self, k: int) -> int:
        """Global row offset for harmonic *k*."""
        return self.block_index(k) * self.n

    def block_slice(self, k: int) -> slice:
        """Global row slice for harmonic *k*."""
        off = self.block_offset(k)
        return slice(off, off + self.n)

    def lift_indices(self, base_idx: np.ndarray, k: int) -> np.ndarray:
        """Lift base-DOF indices to harmonic block *k*.

        Parameters
        ----------
        base_idx :
            Array of indices in [0, n).
        k :
            Harmonic index in [-Nh, Nh].

        Returns
        -------
        ndarray
            Indices shifted by ``block_offset(k)``.
        """
        return np.asarray(base_idx) + self.block_offset(k)

    def harmonic_frequencies(self) -> np.ndarray:
        """Array of k*omega for each harmonic: [-Nh*omega, ..., 0, ..., +Nh*omega]."""
        return self.harmonics * self.omega


# ---------------------------------------------------------------------------
# C2: Harmonic weight builder
# ---------------------------------------------------------------------------

def build_harmonic_weights(
    layout: HarmonicLayout,
    W_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build block-diagonal harmonic energy weights.

    Replicates the per-block weight vector *W_diag* across all harmonics.
    This is the simplest (block-diagonal) harmonic norm — no cross-harmonic
    coupling.

    Parameters
    ----------
    layout :
        Harmonic layout defining the block structure.
    W_diag :
        Diagonal of the single-harmonic energy norm (length ``layout.n``).

    Returns
    -------
    W_harm :
        Diagonal of the harmonic weight matrix (length ``layout.N``).
    invW_harm :
        Diagonal of the inverse weight matrix (length ``layout.N``).
    """
    W_diag = np.asarray(W_diag, dtype=float)
    if W_diag.size != layout.n:
        raise ValueError(
            f"W_diag has {W_diag.size} entries but layout.n={layout.n}"
        )
    W_harm = np.tile(W_diag, layout.nblocks)
    invW_harm = 1.0 / W_harm
    return W_harm, invW_harm


# ---------------------------------------------------------------------------
# C3: Harmonic block assembly
# ---------------------------------------------------------------------------

def assemble_harmonic_operator(
    layout: HarmonicLayout,
    operator_harmonics: list[sp.spmatrix],
    *,
    real_baseflow: bool = True,
    include_time_derivative: bool = True,
) -> sp.spmatrix:
    r"""Assemble the harmonic-balanced operator from operator Fourier coefficients.

    Given operator harmonics {L_0, L_1, ..., L_Nf}, builds the block-Toeplitz
    matrix A_hat and optionally subtracts the time-derivative diagonal
    to form the generator T.

    Parameters
    ----------
    layout :
        Harmonic layout defining block structure and frequencies.
    operator_harmonics :
        List of SciPy sparse matrices [L_0, L_1, ..., L_Nf].
        L_0 is the mean-flow operator; L_k for k>0 are the k-th harmonic
        contributions.  For ``real_baseflow=True``, the negative harmonics
        are computed as L_{-k} = conj(L_k).
    real_baseflow :
        If True (default), assume L_{-k} = conj(L_k).
        If False, ``operator_harmonics`` must contain all 2*Nf+1 coefficients
        ordered as [L_{-Nf}, ..., L_0, ..., L_{Nf}].
    include_time_derivative :
        If True (default), subtract the block-diagonal time-derivative term
        diag(i*k*omega*I) to form the full generator T.
        If False, return only the block-Toeplitz A_hat.

    Returns
    -------
    sp.spmatrix
        The assembled harmonic operator (CSC format), shape (N, N).
    """
    Nh = layout.Nh
    n = layout.n
    nblocks = layout.nblocks

    # Build full harmonic coefficient list: [L_{-Nf}, ..., L_0, ..., L_{Nf}].
    if real_baseflow:
        Nf = len(operator_harmonics) - 1
        # operator_harmonics = [L_0, L_1, ..., L_Nf]
        full_harmonics: dict[int, sp.spmatrix] = {}
        full_harmonics[0] = operator_harmonics[0]
        for k in range(1, Nf + 1):
            full_harmonics[k] = operator_harmonics[k]
            full_harmonics[-k] = operator_harmonics[k].conj()
    else:
        Nf = (len(operator_harmonics) - 1) // 2
        # operator_harmonics = [L_{-Nf}, ..., L_0, ..., L_{Nf}]
        full_harmonics = {}
        for idx, k in enumerate(range(-Nf, Nf + 1)):
            full_harmonics[k] = operator_harmonics[idx]

    if Nf > Nh:
        raise ValueError(
            f"Number of operator harmonics Nf={Nf} exceeds layout Nh={Nh}"
        )

    # Build block-Toeplitz matrix A_hat.
    # Block (i,j) = L_{k_i - k_j} where k_i = i - Nh.
    blocks = [[None] * nblocks for _ in range(nblocks)]
    zero_block = sp.csc_matrix((n, n), dtype=complex)

    for i in range(nblocks):
        for j in range(nblocks):
            diff = i - j  # k_i - k_j
            if diff in full_harmonics:
                blocks[i][j] = full_harmonics[diff]
            else:
                blocks[i][j] = zero_block

    A_hat = sp.bmat(blocks, format="csc")

    if not include_time_derivative:
        return A_hat

    # Subtract time-derivative: T = A_hat - diag(i*k*omega*I).
    freqs = layout.harmonic_frequencies()
    diag_vals = np.zeros(layout.N, dtype=complex)
    for idx, k_omega in enumerate(freqs):
        diag_vals[idx * n : (idx + 1) * n] = 1j * k_omega

    T = A_hat - sp.diags(diag_vals, format="csc")
    return T


# ---------------------------------------------------------------------------
# C4: Harmonic BC application
# ---------------------------------------------------------------------------

def apply_harmonic_bcs(
    T: sp.spmatrix,
    layout: HarmonicLayout,
    bc_indices: np.ndarray,
    penalty: float = 1e4,
) -> sp.spmatrix:
    """Apply Dirichlet boundary conditions across all harmonic blocks.

    Lifts the base-system BC indices into each harmonic block and enforces
    the penalty-method Dirichlet condition on the full harmonic operator.
    This zeros all rows and columns at BC DOFs (including off-diagonal
    coupling blocks) and sets the diagonal to *penalty*.

    Parameters
    ----------
    T :
        Assembled harmonic operator (from :func:`assemble_harmonic_operator`),
        shape ``(N, N)`` where ``N = layout.N``.
    layout :
        Harmonic layout defining block structure.
    bc_indices :
        1-D array of base-system BC DOF indices (in ``[0, layout.n)``).
        These are the indices returned by ``apply_bc_recipe`` or
        ``apply_dirichlet`` on the single-harmonic operator.
    penalty :
        Diagonal penalty value (default ``1e4``).

    Returns
    -------
    sp.spmatrix
        Modified harmonic operator with BCs applied (CSC format).
    """
    bc_indices = np.asarray(bc_indices, dtype=int)
    if bc_indices.size == 0:
        return sp.csc_matrix(T)

    # Lift base BC indices to every harmonic block.
    lifted = np.concatenate([
        layout.lift_indices(bc_indices, k) for k in layout.harmonics
    ])
    lifted = np.unique(lifted)

    # Zero rows (CSR) then columns (CSC).
    T = T.tocsr()
    starts = T.indptr[lifted]
    ends = T.indptr[lifted + 1]
    idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    if idx.size:
        T.data[idx] = 0

    T = T.tocsc()
    starts = T.indptr[lifted]
    ends = T.indptr[lifted + 1]
    idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    if idx.size:
        T.data[idx] = 0

    # Set diagonal to penalty.
    T = T + sp.csc_matrix(
        (np.full(lifted.size, complex(penalty)), (lifted, lifted)),
        shape=T.shape,
    )
    T.eliminate_zeros()
    return T


def assemble_harmonic_operator_petsc(
    layout: HarmonicLayout,
    operator_harmonics: list[sp.spmatrix],
    *,
    real_baseflow: bool = True,
    include_time_derivative: bool = True,
    solver_cfg: dict | None = None,
) -> MatrixOperator:
    """Assemble harmonic operator and wrap as a MatrixOperator with MUMPS KSP.

    Convenience function combining :func:`assemble_harmonic_operator` with
    PETSc conversion and MUMPS factorization.

    Returns
    -------
    MatrixOperator
        The harmonic generator as a solve-capable operator.
    """
    T_sp = assemble_harmonic_operator(
        layout,
        operator_harmonics,
        real_baseflow=real_baseflow,
        include_time_derivative=include_time_derivative,
    )
    A = scipy_to_petsc(T_sp, solver_cfg)

    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    ksp.setUp()

    return MatrixOperator(A, ksp, nblocks=layout.nblocks)


# ---------------------------------------------------------------------------
# C5: Harmonic resolvent analysis
# ---------------------------------------------------------------------------

def harmonic_resolvent_analysis(
    T: sp.spmatrix,
    layout: HarmonicLayout,
    n_modes: int,
    W_diag: np.ndarray,
    *,
    solver_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute harmonic resolvent gains and optimal forcing/response modes.

    The harmonic resolvent operator is :math:`H = T^{-1}` where *T* is
    the harmonic generator (block-Toeplitz LNSE operator minus the
    time-derivative diagonal).  Gains and modes are computed from the
    weighted Hermitian operator

    .. math::
        \mathcal{H}^+\mathcal{H}(v) = W^{-1} T^{-H} W T^{-1} v

    using SLEPc EPS with Krylov-Schur, following the same ``HtrH``
    eigenvalue route as the standard resolvent analysis.

    Parameters
    ----------
    T :
        Assembled harmonic generator with BCs applied, shape ``(N, N)``.
        Typically produced by :func:`assemble_harmonic_operator` followed
        by :func:`apply_harmonic_bcs`.
    layout :
        Harmonic layout (defines block structure and frequencies).
    n_modes :
        Number of leading resolvent modes to compute.
    W_diag :
        Diagonal of the **base-system** energy-norm weight vector
        (length ``layout.n``).  It is tiled across harmonics internally
        via :func:`build_harmonic_weights`.
    solver_cfg :
        Optional solver configuration dict.

    Returns
    -------
    gains : ndarray, shape (n_modes,)
        Harmonic resolvent gains (singular values), sorted descending.
    responses : ndarray, shape (N, n_modes)
        Optimal response modes (columns), W-normalised.
    forcings : ndarray, shape (N, n_modes)
        Optimal forcing modes (columns), W-normalised.
    """
    N = layout.N

    # Build harmonic weights from per-block weights.
    W_harm, invW_harm = build_harmonic_weights(layout, W_diag)

    # Factorize harmonic generator with MUMPS.
    T_op = scipy_to_solve_operator(sp.csc_matrix(T), solver_cfg=solver_cfg)

    # Compose HtrH = invW * T^{-H} * W * T^{-1}.
    W_op = DiagonalOperator(W_harm)
    invW_op = DiagonalOperator(invW_harm)
    htrh = ProductOperator(
        [T_op, W_op, T_op, invW_op],
        [T_op.solve, W_op.apply, T_op.solve_adjoint, invW_op.apply],
    )
    htrh_shell = PetscShellOperator.as_mat(htrh)

    # SLEPc EPS on HtrH.
    ncv = min(max(2 * n_modes + 3, 20), N - 1)
    eps = SLEPc.EPS().create()
    eps.setOperators(htrh_shell)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev=n_modes, ncv=ncv)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    eps.setTolerances(tol=1e-12, max_it=1000)
    eps.solve()

    nconv = eps.getConverged()
    if nconv == 0:
        raise RuntimeError("SLEPc EPS did not converge for any HtrH eigenpair")
    n_extract = min(nconv, n_modes)

    # Extract gains, forcings, responses.
    gains = np.empty(n_extract, dtype=float)
    forcings = np.empty((N, n_extract), dtype=complex)
    responses = np.empty((N, n_extract), dtype=complex)

    vr = htrh_shell.createVecRight()
    vi = htrh_shell.createVecRight()
    for i in range(n_extract):
        eig = eps.getEigenpair(i, vr, vi)
        sigma = float(np.sqrt(max(float(np.real(eig)), 0.0)))
        gains[i] = sigma

        # Extract and W-normalise forcing.
        forcing = vec_to_numpy(vr)
        norm_sq = float(np.real(np.vdot(forcing, W_harm * forcing)))
        if norm_sq > 0.0:
            forcing /= np.sqrt(norm_sq)
        forcings[:, i] = forcing

        # Response = T^{-1} forcing / sigma.
        f_vec = petsc_vec(forcing)
        r_vec = T_op.solve(f_vec)
        response = vec_to_numpy(r_vec)
        if sigma > 0.0:
            response /= sigma
        responses[:, i] = response
        f_vec.destroy()
        r_vec.destroy()

    # Sort by descending gain.
    order = np.argsort(gains)[::-1]
    gains = gains[order]
    forcings = forcings[:, order]
    responses = responses[:, order]

    # Cleanup.
    vr.destroy()
    vi.destroy()
    eps.destroy()
    htrh_shell.destroy()
    htrh.destroy()

    return gains, responses, forcings


# ---------------------------------------------------------------------------
# C6: Projected harmonic resolvent analysis
# ---------------------------------------------------------------------------

def projected_harmonic_resolvent_analysis(
    T: sp.spmatrix,
    layout: HarmonicLayout,
    n_modes: int,
    W_diag: np.ndarray,
    dQdt: np.ndarray,
    *,
    solver_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute projected harmonic resolvent gains and modes.

    For time-periodic base flows, the harmonic generator *T* is singular
    (it has a zero eigenvalue in the phase-shift direction ``dQ/dt``).
    This function removes the neutral direction via complementary projectors
    before computing the SVD, following Padovan et al. (2020).

    The projected harmonic resolvent is

    .. math::
        H_{\perp} = P_r \, T^{-1} \, P_d

    where :math:`P_d = I - w w^*` (domain projector) and
    :math:`P_r = I - v v^*` (range projector) with:

    - :math:`v = dQ/d\tau` (normalised time derivative of the base flow)
    - :math:`w = T^{-H} v / \|T^{-H} v\|` (adjoint null-vector approximation)

    The weighted operator for the eigenproblem is

    .. math::
        \mathcal{H}_\perp^+ \mathcal{H}_\perp(f)
            = W^{-1} P_d^H T^{-H} P_r^H W \, P_r T^{-1} P_d \, f

    Parameters
    ----------
    T :
        Assembled harmonic generator with BCs applied, shape ``(N, N)``.
    layout :
        Harmonic layout.
    n_modes :
        Number of leading resolvent modes to compute.
    W_diag :
        Base-system energy-norm weight vector (length ``layout.n``).
    dQdt :
        Time derivative of the base-flow Fourier expansion in the harmonic
        ordering, shape ``(N,)``.  This is the neutral (phase-shift)
        direction of *T*.  It will be normalised internally.
    solver_cfg :
        Optional solver configuration dict.

    Returns
    -------
    gains : ndarray, shape (n_modes,)
        Projected harmonic resolvent gains, sorted descending.
    responses : ndarray, shape (N, n_modes)
        Projected optimal response modes (columns), W-normalised.
    forcings : ndarray, shape (N, n_modes)
        Projected optimal forcing modes (columns), W-normalised.
    """
    N = layout.N
    dQdt = np.asarray(dQdt, dtype=complex).ravel()
    if dQdt.size != N:
        raise ValueError(
            f"dQdt has {dQdt.size} entries but layout.N={N}"
        )

    # Build harmonic weights.
    W_harm, invW_harm = build_harmonic_weights(layout, W_diag)

    # Factorize harmonic generator.
    T_op = scipy_to_solve_operator(sp.csc_matrix(T), solver_cfg=solver_cfg)

    # Build projector directions.
    # v = dQdt / ||dQdt||
    v = dQdt / np.linalg.norm(dQdt)

    # w = T^{-H} v / ||T^{-H} v||
    v_vec = petsc_vec(v)
    w_vec = T_op.solve_adjoint(v_vec)
    w = vec_to_numpy(w_vec)
    w /= np.linalg.norm(w)
    v_vec.destroy()
    w_vec.destroy()

    # Build complementary projectors.
    # Pd = I - w w*  (orthogonal, domain)
    # Pr = I - v v*  (orthogonal, range)
    Pd = ProjectionOperator(w, w, complement=True)
    Pr = ProjectionOperator(v, v, complement=True)

    # Compose HtrH = invW * Pd^H * T^{-H} * Pr^H * W * Pr * T^{-1} * Pd
    W_op = DiagonalOperator(W_harm)
    invW_op = DiagonalOperator(invW_harm)

    htrh = ProductOperator(
        [Pd, T_op, Pr, W_op, Pr, T_op, Pd, invW_op],
        [Pd.apply, T_op.solve, Pr.apply, W_op.apply,
         Pr.apply_adjoint, T_op.solve_adjoint, Pd.apply_adjoint, invW_op.apply],
    )
    htrh_shell = PetscShellOperator.as_mat(htrh)

    # SLEPc EPS.
    ncv = min(max(2 * n_modes + 3, 20), N - 1)
    eps = SLEPc.EPS().create()
    eps.setOperators(htrh_shell)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev=n_modes, ncv=ncv)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    eps.setTolerances(tol=1e-12, max_it=1000)
    eps.solve()

    nconv = eps.getConverged()
    if nconv == 0:
        raise RuntimeError(
            "SLEPc EPS did not converge for any projected HtrH eigenpair"
        )
    n_extract = min(nconv, n_modes)

    # Extract gains, forcings, responses.
    gains = np.empty(n_extract, dtype=float)
    forcings = np.empty((N, n_extract), dtype=complex)
    responses = np.empty((N, n_extract), dtype=complex)

    vr = htrh_shell.createVecRight()
    vi = htrh_shell.createVecRight()
    for i in range(n_extract):
        eig = eps.getEigenpair(i, vr, vi)
        sigma = float(np.sqrt(max(float(np.real(eig)), 0.0)))
        gains[i] = sigma

        forcing = vec_to_numpy(vr)
        norm_sq = float(np.real(np.vdot(forcing, W_harm * forcing)))
        if norm_sq > 0.0:
            forcing /= np.sqrt(norm_sq)
        forcings[:, i] = forcing

        # Response = Pr T^{-1} Pd forcing / sigma.
        f_vec = petsc_vec(forcing)
        Pd_f = Pd.apply(f_vec)
        Tinv_Pd_f = T_op.solve(Pd_f)
        Pr_Tinv_Pd_f = Pr.apply(Tinv_Pd_f)
        response = vec_to_numpy(Pr_Tinv_Pd_f)
        if sigma > 0.0:
            response /= sigma
        responses[:, i] = response
        f_vec.destroy()
        Pd_f.destroy()
        Tinv_Pd_f.destroy()
        Pr_Tinv_Pd_f.destroy()

    order = np.argsort(gains)[::-1]
    gains = gains[order]
    forcings = forcings[:, order]
    responses = responses[:, order]

    vr.destroy()
    vi.destroy()
    eps.destroy()
    htrh_shell.destroy()
    htrh.destroy()

    return gains, responses, forcings
