# SPDX-License-Identifier: GPL-3.0-only
"""Harmonic resolvent operator blocks from periodic base flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from linstabpy.baseflow.periodic import PeriodicBaseFlow
    from linstabpy.mesh import Mesh


@dataclass(frozen=True)
class HarmonicBlocks:
    """Fourier blocks of LNSE operator at harmonics k=-N_h..N_h.

    All blocks share the same union sparsity pattern to enable FFT of
    data arrays while preserving alignment.

    Attributes
    ----------
    Ahat : list[sp.csr_matrix]
        Length 2*N_h+1, CSR matrices for k=-N_h..N_h, all sharing indices/indptr.
        Index j = k + N_h, so k=-N_h..0..N_h maps to j=0..N_h..2*N_h.
        Shape (n_dof, n_dof).
    N_h : int
        Highest harmonic index (stored as both +N_h and -N_h).
    N_t : int
        Number of time samples per period used in FFT.
    tail_ratio : float | None
        ||Ahat_{N_h+1}||_F / ||Ahat_0||_F when N_t large enough to expose
        harmonic N_h+1; else None. Aliasing diagnostic.
    conjugation_asymmetry : float | None
        max over k=1..N_h of ||Ahat_{-k} - conj(Ahat_k)||_F / ||Ahat_k||_F.
        Measures asymmetry from two-sided storage (should be ~0 at kx=0,
        nonzero at kx!=0 because the operator is genuinely complex).
    """

    Ahat: list[sp.csr_matrix]
    N_h: int
    N_t: int
    tail_ratio: float | None = None
    conjugation_asymmetry: float | None = None

    def __post_init__(self) -> None:
        """Validate harmonic blocks."""
        if len(self.Ahat) != 2 * self.N_h + 1:
            raise ValueError(
                f"Ahat must have {2 * self.N_h + 1} blocks (k=-N_h..N_h), got {len(self.Ahat)}"
            )


def assemble_harmonic_blocks(
    mesh: Mesh,
    pbf: PeriodicBaseFlow,
    *,
    N_h: int,
    N_t: int,
    kx: float = 0.0,
    model: str = "2D",
    bc_recipe: list[dict] | None = None,
    bc_penalty: float = 1e4,
    dissipation_epsilon: float = 0.0,
    solver_cfg: dict | None = None,
) -> HarmonicBlocks:
    """Assemble harmonic blocks of LNSE operator from periodic base flow.

    For each sample time t_n in [0, T_period), reconstruct the real base flow
    Q(t_n), call build_lnse_system, and collect the operators. Then align
    all matrices onto a union sparsity pattern and FFT the data arrays to
    extract k=-N_h..N_h (full two-sided spectrum).

    Union pattern alignment is critical because assembly() calls
    eliminate_zeros(), making the pattern depend on base-flow values.
    A full FFT allows independent handling of positive and negative
    harmonics, avoiding the assumption that Ahat_{-k} = conj(Ahat_k).
    That assumption fails when kx != 0 because A(t; kx) is genuinely
    complex through its i*kx derivatives.

    Parameters
    ----------
    mesh : Mesh
        Computational mesh.
    pbf : PeriodicBaseFlow
        Periodic base flow with harmonics.
    N_h : int
        Highest harmonic to keep (must be <= pbf.N_h).
    N_t : int
        Number of time samples per period. Must be even and > 2*N_h + 6
        (or check tail_ratio afterwards). Also must be >= 2*N_h + 1
        to avoid aliasing of ±N_h.
    kx : float, optional
        Streamwise wavenumber [rad/m]. Default 0.
    model : str, optional
        Dynamical model ("2D", "3D", "axy"). Default "2D".
    bc_recipe : list[dict], optional
        Boundary condition specification.
    bc_penalty : float, optional
        Penalty parameter for BC enforcement. Default 1e4.
    dissipation_epsilon : float, optional
        Artificial dissipation coefficient. Default 0.
    solver_cfg : dict, optional
        Solver configuration dictionary.

    Returns
    -------
    HarmonicBlocks
        Harmonic blocks Ahat[k] for k=-N_h..N_h (index j = k + N_h),
        shared pattern, tail_ratio, and conjugation_asymmetry diagnostic.
    """
    from linstabpy.operators.assembly import build_lnse_system

    if N_h >= N_t:
        raise ValueError(
            f"N_h={N_h} must be less than N_t={N_t} (need N_t samples to resolve ±N_h)"
        )
    if N_t % 2 != 0:
        raise ValueError(f"N_t must be even, got {N_t}")

    # Sample times over [0, T_period)
    times = pbf.sample_times(N_t)

    # Assemble L0(Q(t_n)) at each instant
    A_list = []
    for t_n in times:
        bf = pbf.reconstruct(t_n)
        L0, _, _ = build_lnse_system(
            mesh,
            bf,
            kx=kx,
            model=model,
            bc_recipe=bc_recipe,
            bc_penalty=bc_penalty,
            dissipation_epsilon=dissipation_epsilon,
            solver_cfg=solver_cfg,
        )
        # Convert to CSR if needed
        if not isinstance(L0, sp.csr_matrix):
            L0 = L0.tocsr()
        A_list.append(L0)

    n_dof = A_list[0].shape[0]

    # Build union sparsity pattern from all instants
    coo_list = [A.tocoo() for A in A_list]
    rows_cat = np.concatenate([c.row for c in coo_list])
    cols_cat = np.concatenate([c.col for c in coo_list])

    union_coo = sp.csr_matrix(
        (np.ones_like(rows_cat, dtype=float), (rows_cat, cols_cat)),
        shape=(n_dof, n_dof),
    )
    union_coo.sum_duplicates()
    union_coo.sort_indices()

    # Extract union pattern
    union_indices = union_coo.indices
    union_indptr = union_coo.indptr
    nnz_union = union_coo.nnz

    # Scatter each instant's data into union slots
    stacked_data = np.zeros((N_t, nnz_union), dtype=complex)

    for inst, A_n in enumerate(A_list):
        out = np.zeros(nnz_union, dtype=complex)

        for i in range(n_dof):
            # Row segment in A_n
            a0 = A_n.indptr[i]
            a1 = A_n.indptr[i + 1]
            # Union row segment
            u0 = union_indptr[i]
            u1 = union_indptr[i + 1]

            if a1 > a0:
                # Find position in union for each col in A_n[i,:]
                A_cols = A_n.indices[a0:a1]
                union_cols = union_indices[u0:u1]
                pos = np.searchsorted(union_cols, A_cols)
                out[u0 + pos] = A_n.data[a0:a1]

        # Validate reconstruction: check that sparse representation matches
        reconstructed = sp.csr_matrix((out, union_indices, union_indptr), shape=(n_dof, n_dof))
        diff = A_n - reconstructed
        if diff.nnz > 0 and np.max(np.abs(diff.data)) > 0:
            raise RuntimeError(
                f"Union pattern alignment failed at instant {inst}: "
                f"reconstructed matrix differs from original in {diff.nnz} entries"
            )

        stacked_data[inst, :] = out

    # Full FFT over time (not rfft): forward 1/N_t normalization, keep all k
    Ahat_full = (1.0 / N_t) * np.fft.fft(stacked_data, axis=0)

    # Extract k = -N_h..N_h; negative k are in bins N_t+k (e.g., k=-1 -> bin N_t-1)
    ahat_list = []
    for k in range(-N_h, N_h + 1):
        if k >= 0:
            bin_idx = k
        else:
            bin_idx = N_t + k  # Negative frequency: bin N_t - |k|
        ahat_k = sp.csr_matrix(
            (Ahat_full[bin_idx, :], union_indices, union_indptr),
            shape=(n_dof, n_dof),
        )
        ahat_list.append(ahat_k)

    # Compute conjugation asymmetry: max_k ||Ahat[-k] - conj(Ahat[k])|| / ||Ahat[k]||
    conjugation_asymmetry = None
    if N_h > 0:
        asym_vals = []
        for k in range(1, N_h + 1):
            idx_pos = N_h + k  # Ahat[k] is at position N_h + k in the output list
            idx_neg = N_h - k  # Ahat[-k] is at position N_h - k
            diff_asym = ahat_list[idx_neg] - ahat_list[idx_pos].conj()
            norm_asym = np.linalg.norm(diff_asym.data) if diff_asym.nnz > 0 else 0.0
            norm_pos = (
                np.linalg.norm(ahat_list[idx_pos].data) if ahat_list[idx_pos].nnz > 0 else 1.0
            )
            if norm_pos > 0:
                asym_vals.append(norm_asym / norm_pos)
        conjugation_asymmetry = max(asym_vals) if asym_vals else 0.0

    # Compute tail ratio if N_t allows harmonic N_h+1
    tail_ratio = None
    if N_t > 2 * (N_h + 1):
        norm_tail = np.linalg.norm(Ahat_full[N_h + 1, :], ord=2)
        norm_0 = np.linalg.norm(Ahat_full[0, :], ord=2)
        tail_ratio = norm_tail / norm_0 if norm_0 > 0 else np.nan

    return HarmonicBlocks(
        Ahat=ahat_list,
        N_h=N_h,
        N_t=N_t,
        tail_ratio=tail_ratio,
        conjugation_asymmetry=conjugation_asymmetry,
    )


def write_resolvent4py_coo(
    blocks: HarmonicBlocks,
    out_dir: str,
) -> list[tuple[str, str, str]]:
    """Write harmonic blocks in resolvent4py COO format.

    Each harmonic k=-N_h..N_h is written as three binary files:
      rows_XX.dat, cols_XX.dat, vals_XX.dat
    where XX = k + N_h (i.e., k ranges [0, 2*N_h] in the file indices).

    Indices are int64; values are complex128. Uses resolvent4py's own
    write_to_file so binary layout is compatible. The file ordering
    (k=-N_h to +N_h) matches resolvent4py's expectation for
    real_bflow=False: read_harmonic_balanced_matrix(..., real_bflow=False)
    expects files in order {..., A_{-1}, A_0, A_1, ...}.

    Parameters
    ----------
    blocks : HarmonicBlocks
        Harmonic blocks to write (k=-N_h..N_h in storage).
    out_dir : str
        Output directory (created if needed).

    Returns
    -------
    list[tuple[str, str, str]]
        List of (rows_path, cols_path, vals_path) for k=-N_h..N_h
        in that order (j = 0..2*N_h corresponds to k=-N_h..N_h).
    """
    try:
        from resolvent4py.utils import io as r4py_io
    except ImportError:
        raise ImportError("resolvent4py not found. Install with: pip install linstabpy[harmonic]")

    import os

    os.makedirs(out_dir, exist_ok=True)

    result = []
    for j, Ahat_k in enumerate(blocks.Ahat):
        # j = 0..2*N_h corresponds to k = -N_h..N_h
        # Convert to COO
        coo_k = Ahat_k.tocoo()

        # Build file names with j index (0-padded)
        fmt = f"{j:02d}"
        rows_path = os.path.join(out_dir, f"rows_{fmt}.dat")
        cols_path = os.path.join(out_dir, f"cols_{fmt}.dat")
        vals_path = os.path.join(out_dir, f"vals_{fmt}.dat")

        # Write using resolvent4py's writer
        r4py_io.write_to_file(
            rows_path,
            coo_k.row.astype(np.int64),
        )
        r4py_io.write_to_file(
            cols_path,
            coo_k.col.astype(np.int64),
        )
        r4py_io.write_to_file(
            vals_path,
            coo_k.data.astype(np.complex128),
        )

        result.append((rows_path, cols_path, vals_path))

    return result
