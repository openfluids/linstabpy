# SPDX-License-Identifier: GPL-3.0-only
"""Harmonic resolvent operator blocks from periodic base flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from linstabpy.baseflow.periodic import PeriodicBaseFlow
    from linstabpy.mesh import Mesh


@dataclass(frozen=True)
class HarmonicBlocks:
    """Fourier blocks of LNSE operator at harmonics k=-N_h..N_h.

    The time dependence is ``A(t) = sum_k Ahat_k exp(-i*k*omega0*t)``, the
    ``exp(-i*omega*t)`` convention of the classical resolvent.

    All blocks share the same union sparsity pattern to enable FFT of
    data arrays while preserving alignment.

    Attributes
    ----------
    Ahat : list[sp.csr_matrix]
        Length 2*N_h+1, CSR matrices for k=-N_h..N_h, all sharing indices/indptr.
        ``Ahat[k]`` is the coefficient of ``exp(-i*k*omega0*t)``.
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
    extract k=-N_h..N_h (full two-sided spectrum). ``Ahat_k`` is the
    coefficient of ``exp(-i*k*omega0*t)``.

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
        (or check tail_ratio afterwards). Also must be > 2*N_h
        to avoid aliasing of ±N_h.
    kx : float, optional
        Spanwise wavenumber beta (z direction, normal to the mesh plane),
        nondimensional (case units). Default 0.
    model : str, optional
        Dynamical model ("2D", "3D", "axy"). Default "2D".
    bc_recipe : list[dict], optional
        Boundary condition specification.
    bc_penalty : float, optional
        Penalty parameter for BC enforcement. Default 1e4.
    dissipation_epsilon : float, optional
        Artificial dissipation coefficient. Default 0. A positive value
        needs a mesh built with ``numerics_operator_dissipation`` enabled;
        ``build_lnse_system`` refuses it otherwise.
    solver_cfg : dict, optional
        Solver configuration dictionary.

    Returns
    -------
    HarmonicBlocks
        Harmonic blocks Ahat[k] for k=-N_h..N_h (index j = k + N_h),
        shared pattern, tail_ratio, and conjugation_asymmetry diagnostic.
    """
    from linstabpy.operators.assembly import build_lnse_system

    # Aliasing guard: extracting bins k=-N_h..N_h from an N_t-point FFT
    # needs N_t > 2*N_h, else bin N_t-|k| wraps back and aliases with a
    # positive harmonic (silent data corruption).
    if N_t <= 2 * N_h:
        message = f"N_t={N_t} must be > 2*N_h={2 * N_h} to avoid aliasing of harmonics ±{N_h}"
        raise ValueError(message)
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

    # Union sparsity pattern of all instants. assembly() calls
    # eliminate_zeros(), so the pattern depends on the base-flow values.
    # Stored zeros count as entries, so the scatter below always finds a slot.
    rows_cat = np.concatenate([np.repeat(np.arange(n_dof), np.diff(A.indptr)) for A in A_list])
    cols_cat = np.concatenate([A.indices for A in A_list])
    union_csr = sp.csr_matrix(
        (np.ones_like(rows_cat, dtype=float), (rows_cat, cols_cat)),
        shape=(n_dof, n_dof),
    )
    del rows_cat, cols_cat
    union_csr.sum_duplicates()
    union_csr.sort_indices()

    # Extract union pattern
    union_indices = union_csr.indices
    union_indptr = union_csr.indptr
    nnz_union = union_csr.nnz

    # After sort_indices(), rows ascend globally and columns ascend within
    # each row, so the key row*n_dof + col is globally ascending and
    # searchsorted maps each A_n entry to its union slot.
    union_rows = np.repeat(np.arange(n_dof), np.diff(union_indptr))
    union_keys = union_rows.astype(np.int64) * n_dof + union_indices

    # Harmonics to keep: k = -N_h..N_h, plus N_h+1 for the aliasing
    # diagnostic when N_t resolves it. Accumulating the discrete Fourier
    # sum one instant at a time holds (2*N_h+2) data arrays instead of the
    # (N_t, nnz) array a full FFT needs. The normalization is the 1/N_t of
    # np.fft.fft. The kernel is exp(+2*pi*i*k*n/N_t), the inverse-FFT sign,
    # so bin k is the coefficient of exp(-i*k*omega0*t) (np.fft.fft gives
    # exp(+i*k*omega0*t)).
    with_tail = N_t > 2 * (N_h + 1)
    k_values = list(range(-N_h, N_h + 1)) + ([N_h + 1] if with_tail else [])
    k_array = np.asarray(k_values, dtype=float)
    acc = np.zeros((len(k_values), nnz_union), dtype=complex)

    for inst, A_n in enumerate(A_list):
        out = np.zeros(nnz_union, dtype=complex)
        A_n_coo = A_n.tocoo()
        # int64 keys avoid overflow for n_dof above 46341.
        A_n_keys = A_n_coo.row.astype(np.int64) * n_dof + A_n_coo.col
        pos = np.searchsorted(union_keys, A_n_keys)
        out[pos] = A_n_coo.data

        # Validate reconstruction: check that sparse representation matches
        # original. This gate falsifies the optimization; a mismatch signals
        # a keying or searchsorted error.
        reconstructed = sp.csr_matrix((out, union_indices, union_indptr), shape=(n_dof, n_dof))
        diff = A_n - reconstructed
        # The union pattern holds every entry of A_n by construction, so
        # this raise fires only on an internal defect. No public input can
        # reach it, which is why coverage excludes the branch.
        if diff.nnz > 0 and np.max(np.abs(diff.data)) > 0:  # pragma: no cover
            message = (
                f"Union pattern alignment failed at instant {inst}: "
                f"reconstructed matrix differs from original in {diff.nnz} entries"
            )
            raise RuntimeError(message)

        phase = np.exp(2j * np.pi * k_array * inst / N_t) / N_t
        acc += phase[:, None] * out[None, :]
        # The instant is folded into every kept harmonic; release it.
        A_list[inst] = None

    ahat_list = [
        sp.csr_matrix((acc[j, :], union_indices, union_indptr), shape=(n_dof, n_dof))
        for j in range(2 * N_h + 1)
    ]

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
    if with_tail:
        norm_tail = np.linalg.norm(acc[-1, :], ord=2)
        norm_0 = np.linalg.norm(acc[N_h, :], ord=2)
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

    The writer stores three binary files per harmonic:
      rows_XX.dat, cols_XX.dat, vals_XX.dat.
    The index XX = k + N_h runs 0..2*N_h for k=-N_h..N_h. File k holds
    Ahat_k, the coefficient of exp(-i*k*omega0*t). resolvent4py reads the
    files with exp(+i*k*omega0*t); harmonic_resolvent_gains reverses the
    file order to map one convention onto the other.

    resolvent4py.write_to_file() stores PETSc objects, so the indices
    ride as PETSc scalars. The reader later extracts them as integers
    via .getArray().real. resolvent4py expects the file order
    k=-N_h..+N_h when real_bflow=False:
    read_harmonic_balanced_matrix(..., real_bflow=False) reads files
    in order {..., A_{-1}, A_0, A_1, ...}.

    Parameters
    ----------
    blocks : HarmonicBlocks
        Harmonic blocks to write (k=-N_h..N_h in storage).
    out_dir : str
        Output directory (created if needed).

    Returns
    -------
    list[tuple[str, str, str]]
        The (rows_path, cols_path, vals_path) tuples in order
        j = 0..2*N_h (k=-N_h..N_h).
    """
    try:
        from petsc4py import PETSc
    except ImportError as e:
        message = (
            "petsc4py not found. The [harmonic] extra does not install it; "
            "build the complex PETSc/SLEPc stack with: bash install.sh"
        )
        raise ImportError(message) from e

    try:
        from resolvent4py import write_to_file
    except ImportError as e:
        message = "resolvent4py not found. Install with: pip install linstabpy[harmonic]"
        raise ImportError(message) from e

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    result = []
    for j, Ahat_k in enumerate(blocks.Ahat):
        # j = 0..2*N_h corresponds to k = -N_h..N_h
        # Convert to COO
        coo_k = Ahat_k.tocoo()

        # Build file names with j index (0-padded)
        fmt = f"{j:02d}"
        rows_path = out_dir_path / f"rows_{fmt}.dat"
        cols_path = out_dir_path / f"cols_{fmt}.dat"
        vals_path = out_dir_path / f"vals_{fmt}.dat"

        # Write indices and values via resolvent4py's write_to_file.
        # Indices must be cast to PETSc.ScalarType so the buffer dtype
        # matches (on a complex PETSc build, scalars are complex, so
        # integers are stored as complex with zero imaginary part).
        # The reader recovers them via .getArray().real -> int.

        arr_rows = np.asarray(coo_k.row).ravel().astype(PETSc.ScalarType)
        vec_rows = PETSc.Vec().createWithArray(arr_rows, len(arr_rows), None, PETSc.COMM_SELF)
        write_to_file(str(rows_path), vec_rows)
        vec_rows.destroy()

        arr_cols = np.asarray(coo_k.col).ravel().astype(PETSc.ScalarType)
        vec_cols = PETSc.Vec().createWithArray(arr_cols, len(arr_cols), None, PETSc.COMM_SELF)
        write_to_file(str(cols_path), vec_cols)
        vec_cols.destroy()

        arr_vals = np.asarray(coo_k.data).ravel().astype(PETSc.ScalarType)
        vec_vals = PETSc.Vec().createWithArray(arr_vals, len(arr_vals), None, PETSc.COMM_SELF)
        write_to_file(str(vals_path), vec_vals)
        vec_vals.destroy()

        result.append((str(rows_path), str(cols_path), str(vals_path)))

    return result


def write_resolvent4py_vecs(
    vecs: list[np.ndarray],
    out_dir: str,
    prefix: str = "dQ",
) -> list[str]:
    """Write harmonic vectors in resolvent4py format.

    Each harmonic k=-N_h..N_h is written as a single PETSc binary Vec file
    in two-sided order (j=0..2*N_h corresponds to k=-N_h..N_h).

    Parameters
    ----------
    vecs : list[np.ndarray]
        List of 1-D complex arrays (one per harmonic, in two-sided order).
    out_dir : str
        Output directory (created if needed).
    prefix : str, optional
        File prefix (default "dQ"). The writer names each file
        prefix_XX.dat with XX = 00..2*N_h.

    Returns
    -------
    list[str]
        Paths to written files in order (j=0..2*N_h).
    """
    try:
        from petsc4py import PETSc
    except ImportError as e:
        message = (
            "petsc4py not found. The [harmonic] extra does not install it; "
            "build the complex PETSc/SLEPc stack with: bash install.sh"
        )
        raise ImportError(message) from e

    try:
        from resolvent4py import write_to_file
    except ImportError as e:
        message = "resolvent4py not found. Install with: pip install linstabpy[harmonic]"
        raise ImportError(message) from e

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    result = []
    for j, vec_j in enumerate(vecs):
        # j = 0..2*N_h corresponds to k = -N_h..N_h

        # Flatten and cast to PETSc.ScalarType
        arr = np.asarray(vec_j).ravel().astype(PETSc.ScalarType)

        # Create PETSc Vec from array
        pv = PETSc.Vec().createWithArray(arr, len(arr), None, PETSc.COMM_SELF)

        # Write via resolvent4py
        fmt = f"{j:02d}"
        fpath = out_dir_path / f"{prefix}_{fmt}.dat"
        write_to_file(str(fpath), pv)
        pv.destroy()

        result.append(str(fpath))

    return result
