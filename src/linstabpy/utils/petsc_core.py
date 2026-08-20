# SPDX-License-Identifier: GPL-3.0-only
"""SciPy <-> PETSc conversion utilities (local and distributed scaffolding).

Shared by both mesh (filters) and solver modules to avoid circular
dependencies between the mesh and solver layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    # Type-only import so ruff/mypy/pyright see ``PETSc`` names; at runtime,
    # ``TYPE_CHECKING`` is ``False`` so this never fires and PETSc stays
    # un-initialized until the lazy ``__getattr__`` below is hit.
    from petsc4py import PETSc  # noqa: F401

    from linstabpy.config.typed import MpiCfg

_petsc_configured = False


def mumps_icntl_petsc_options(mpi: "MpiCfg") -> dict[str, str]:
    """Map the typed MUMPS ICNTL config to PETSc option-database entries.

    ICNTL(7) is not set here. Use ``mumps_ordering`` / ``LINSTABPY_MUMPS_ORDERING``.
    ICNTL(14)/ICNTL(24) get project defaults.
    Leave them to PETSc/MUMPS only when the config sets ``None``.
    """
    opts: dict[str, str] = {}
    if mpi.mumps_icntl_14 is not None:
        opts["mat_mumps_icntl_14"] = str(mpi.mumps_icntl_14)
    if mpi.mumps_icntl_24 is not None:
        opts["mat_mumps_icntl_24"] = str(mpi.mumps_icntl_24)
    return opts


# Lazy PETSc accessor: ``import linstabpy.utils.petsc_core`` stays PETSc-free.
# External callers get ``PETSc`` via PEP 562 ``__getattr__`` below. Internal
# functions in this module must bind ``PETSc`` locally via ``_get_petsc()``:
# PEP 562 module ``__getattr__`` only fires for external attribute access, not
# for ``LOAD_GLOBAL`` lookups inside the same module's function bodies.
def __getattr__(name):
    if name == "PETSc":
        from .petsc_init import init_petsc_from_argv

        init_petsc_from_argv()
        from petsc4py import PETSc as _imp_PETSc

        globals()["PETSc"] = _imp_PETSc
        return _imp_PETSc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_petsc():
    """Return PETSc, triggering PEP 562 lazy init on first call.

    Required because bare ``PETSc`` references inside this module's own
    function bodies do NOT fire ``__getattr__`` (PEP 562 limitation). Each
    function that touches PETSc must start with ``PETSc = _get_petsc()``.
    """
    return globals().get("PETSc") or __getattr__("PETSc")


def _is_petsc_matrix(operator) -> bool:
    """Return ``True`` when ``operator`` is already a PETSc matrix.

    Uses a duck-type check on ``type(operator).__module__`` so callers do not
    need PETSc to be importable. Pure-numpy / SciPy operators return ``False``
    without ever loading ``petsc4py``.
    """
    cls = type(operator)
    return cls.__module__.startswith("petsc4py") and cls.__name__ == "Mat"


def _operator_shape(operator) -> tuple[int, int]:
    """Return matrix shape for SciPy or PETSc operators."""
    if _is_petsc_matrix(operator):
        nrows, ncols = operator.getSize()
        return int(nrows), int(ncols)
    return int(operator.shape[0]), int(operator.shape[1])


def _matrix_nnz(operator) -> int:
    """Return the stored nonzero count for SciPy or PETSc matrices."""
    if _is_petsc_matrix(operator):
        return int(operator.getInfo().get("nz_used", 0))
    return int(operator.nnz)


def _petsc_matrix_mode(solver_cfg: dict | None = None) -> str:
    """Resolve PETSc matrix conversion mode from YAML/env policy."""
    from linstabpy.config import Config

    raw = Config.load(solver_cfg if solver_cfg is not None else {}).mpi.petsc_matrix_mode
    if raw in {"auto", "local", "serial", "distributed", "mpi", "parallel"}:
        return raw
    raise ValueError(
        f"Invalid petsc_matrix_mode="
        f"{raw!r}; expected one of auto|local|serial|distributed|mpi|parallel"
    )


def _use_distributed_matrix(solver_cfg: dict | None = None) -> bool:
    """Return whether SciPy-to-PETSc conversion should use MPI AIJ ownership."""
    mode = _petsc_matrix_mode(solver_cfg)
    if mode in {"distributed", "mpi", "parallel"}:
        return True
    if mode in {"local", "serial"}:
        return False
    # auto: one rank when PETSc is absent
    try:
        PETSc = _get_petsc()
    except ModuleNotFoundError:
        return False
    return PETSc.COMM_WORLD.getSize() > 1


def require_complex(solver_cfg: dict | None = None) -> None:
    """Assert complex-scalar PETSc and install one-time direct-solver defaults.

    PETSc's options database is process-global.
    The call sets only the project MUMPS defaults, and only once.
    Importing multiple solver modules then does not rewrite options
    behind an already-created KSP/EPS object.
    """
    PETSc = _get_petsc()
    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        raise RuntimeError(
            "PETSc must be built with --with-scalar-type=complex for stability analysis. "
            f"Current scalar type: {PETSc.ScalarType}"
        )
    global _petsc_configured
    if not _petsc_configured:
        from linstabpy.config import Config

        cfg = Config.load(solver_cfg if solver_cfg is not None else {})
        opts = PETSc.Options()
        for key, value in mumps_icntl_petsc_options(cfg.mpi).items():
            opts.setValue(key, value)
        _petsc_configured = True


def scipy_to_petsc(mat: sp.spmatrix, solver_cfg: dict | None = None) -> PETSc.Mat:
    """Convert a SciPy sparse matrix to a PETSc AIJ Mat.

    ``petsc_matrix_mode`` (YAML) or ``LINSTABPY_PETSC_MATRIX_MODE`` (env)
    selects the layout.

    - ``local``/``serial``: local AIJ from full CSR (single-rank style).
    - ``distributed``/``mpi``/``parallel``: ownership-aware distributed AIJ.
    - ``auto`` (default): distributed when MPI size > 1, else local.
    """
    PETSc = _get_petsc()
    if _use_distributed_matrix(solver_cfg):
        return scipy_to_petsc_distributed(mat, comm=PETSc.COMM_WORLD)

    csr = sp.csr_matrix(mat, dtype=complex)
    A = PETSc.Mat().createAIJ(
        size=csr.shape,
        csr=(csr.indptr, csr.indices, csr.data),
        comm=PETSc.COMM_SELF,
    )
    A.assemble()
    return A


def compact_petsc_matrix(mat: PETSc.Mat, solver_cfg: dict | None = None) -> PETSc.Mat:
    """Rebuild a sequential PETSc AIJ matrix into canonical compact form.

    Use this for matrices after in-place PETSc row/column modification.
    Those paths may retain stored structural zeros or duplicate
    allocation patterns.
    The algebraic operator stays the same, but those extras can perturb
    downstream direct factorization.
    """
    PETSc = _get_petsc()
    csr = petsc_to_scipy(mat, format="csr")
    compact = scipy_to_petsc(csr, solver_cfg)

    prefix = mat.getOptionsPrefix()
    if prefix:
        compact.setOptionsPrefix(prefix)

    try:
        row_bs, col_bs = mat.getBlockSizes()
        if int(row_bs) > 0 and int(col_bs) > 0:
            compact.setBlockSizes(int(row_bs), int(col_bs))
    except (PETSc.Error, AttributeError, TypeError):
        pass

    nullspace = mat.getNullSpace()
    if nullspace is not None:
        compact.setNullSpace(nullspace)

    near_nullspace = getattr(mat, "getNearNullSpace", None)
    if callable(near_nullspace):
        ns = near_nullspace()
        if ns is not None:
            compact.setNearNullSpace(ns)

    compact.assemble()
    return compact


def create_distributed_aij(
    shape: tuple[int, int],
    *,
    comm: PETSc.Comm | None = None,
    nnz_diag: int | np.ndarray | None = None,
    nnz_offdiag: int | np.ndarray | None = None,
) -> PETSc.Mat:
    """Create an MPI AIJ matrix with optional preallocation.

    ``nnz_diag`` and ``nnz_offdiag`` may be:

    * scalar: constant upper bound per row.
      Simple, but can overflow ``PetscInt`` for wide stencils on large matrices.
    * array of length nrows: exact per-row count.
      Avoids the ``nrows × max`` product that causes int32 overflow.
      Also allocates less memory.

    For MPI AIJ matrices, ``nnz_diag`` is the diagonal block
    (columns owned by this rank).
    ``nnz_offdiag`` is the off-diagonal block.
    Per-row arrays are automatically clipped to the local block
    widths so that PETSc's preallocation check does not reject them.
    """
    PETSc = _get_petsc()
    if len(shape) != 2:
        raise ValueError(f"shape must be 2D, got {shape}")
    nrows, ncols = int(shape[0]), int(shape[1])
    if nrows <= 0 or ncols <= 0:
        raise ValueError(f"shape entries must be positive, got {shape}")

    matrix = PETSc.Mat().create(comm=comm or PETSc.COMM_WORLD)
    matrix.setType(PETSc.Mat.Type.AIJ)
    matrix.setSizes((nrows, ncols))

    if nnz_diag is not None or nnz_offdiag is not None:
        d = nnz_diag if nnz_diag is not None else nnz_offdiag
        o = nnz_offdiag if nnz_offdiag is not None else nnz_diag
        if isinstance(d, np.ndarray) or isinstance(o, np.ndarray):
            # Per-row arrays: clip to the local block widths so PETSc's
            # preallocation sanity check does not reject oversized entries.
            r0, r1 = matrix.getOwnershipRange()
            local_nrows = r1 - r0
            local_ncols = local_nrows  # square partition for square matrices
            offdiag_ncols = ncols - local_ncols

            d_arr = np.asarray(d, dtype=np.int64)
            o_arr = np.asarray(o, dtype=np.int64)
            # Slice to locally-owned rows if the caller passes full-size arrays.
            if d_arr.size == nrows:
                d_arr = d_arr[r0:r1]
            if o_arr.size == nrows:
                o_arr = o_arr[r0:r1]
            np.clip(d_arr, 0, local_ncols, out=d_arr)
            np.clip(o_arr, 0, max(offdiag_ncols, 1), out=o_arr)
            matrix.setPreallocationNNZ((d_arr.astype(PETSc.IntType), o_arr.astype(PETSc.IntType)))
            # Per-row counts are upper-bound estimates; allow PETSc to
            # dynamically reallocate the few rows where the estimate is
            # slightly low rather than aborting on insertion.
            matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        else:
            matrix.setPreallocationNNZ((int(d), int(o)))

    matrix.setUp()
    return matrix


def ownership_range(matrix: PETSc.Mat) -> tuple[int, int]:
    """Return global owned row range for this rank: [rstart, rend)."""
    rstart, rend = matrix.getOwnershipRange()
    return int(rstart), int(rend)


def petsc_owned_rows(mat: PETSc.Mat, *, copy: bool = True) -> dict:
    """Return CSR data for the rows owned by the current rank.

    For sequential AIJ matrices this is the full matrix. For MPI AIJ matrices,
    PETSc exposes the locally-owned row block through ``getValuesCSR()``, with
    column indices reported in the global numbering.
    ``copy=False`` avoids copies when PETSc already exposes compatible
    NumPy arrays.
    Callers should still keep the matrix alive while reading the returned buffers.
    """
    rstart, rend = ownership_range(mat)
    row_ids = np.arange(rstart, rend, dtype=np.int64)

    try:
        indptr, indices, data = mat.getValuesCSR()
    except Exception as exc:  # pragma: no cover - PETSc type guard
        raise TypeError("PETSc matrix does not support CSR extraction via getValuesCSR()") from exc

    if copy:
        indptr_arr = np.array(indptr, dtype=np.int64, copy=True)
        indices_arr = np.array(indices, dtype=np.int64, copy=True)
        data_arr = np.array(data, copy=True)
    else:
        indptr_arr = np.asarray(indptr, dtype=np.int64)
        indices_arr = np.asarray(indices, dtype=np.int64)
        data_arr = np.asarray(data)

    expected_indptr_size = int(row_ids.size) + 1
    if indptr_arr.size != expected_indptr_size:
        raise RuntimeError(
            "PETSc CSR extraction returned an unexpected local row count: "
            f"ownership_range={(rstart, rend)}, len(indptr)={indptr_arr.size}"
        )

    return {
        "ownership_range": (rstart, rend),
        "row_ids": row_ids,
        "indptr": indptr_arr,
        "indices": indices_arr,
        "data": data_arr,
    }


def insert_local_csr_rows(
    matrix: PETSc.Mat,
    local_rows: sp.spmatrix,
    *,
    row_start: int,
) -> None:
    """Insert one contiguous block of locally-owned CSR rows into PETSc Mat.

    Parameters
    ----------
    matrix:
        Target PETSc matrix.
    local_rows:
        Sparse rows to insert; row ``i`` maps to global row ``row_start + i``.
    row_start:
        Global first row index represented by ``local_rows``.

    Notes
    -----
    This function only inserts values.
    The caller must assemble with ``assemble_matrix`` after all insertions.
    The explicit ownership check is intentional.
    Distributed assembly errors are much easier to diagnose here
    than after PETSc accepts off-rank values into a global assembly.
    """
    PETSc = _get_petsc()
    csr = sp.csr_matrix(local_rows, dtype=complex)
    if csr.shape[1] != matrix.getSize()[1]:
        raise ValueError(
            f"Column mismatch: local_rows has {csr.shape[1]} columns, "
            f"matrix has {matrix.getSize()[1]}"
        )
    if row_start < 0:
        raise ValueError(f"row_start must be non-negative, got {row_start}")

    r0, r1 = ownership_range(matrix)
    row_stop = int(row_start) + int(csr.shape[0])
    if row_start < r0 or row_stop > r1:
        raise ValueError(
            f"Row block [{row_start}, {row_stop}) is outside local ownership [{r0}, {r1})"
        )

    for local_i in range(csr.shape[0]):
        global_i = row_start + local_i
        p0 = int(csr.indptr[local_i])
        p1 = int(csr.indptr[local_i + 1])
        if p1 == p0:
            continue
        cols = csr.indices[p0:p1]
        vals = csr.data[p0:p1]
        matrix.setValues(global_i, cols, vals, addv=PETSc.InsertMode.INSERT_VALUES)


def assemble_matrix(matrix: PETSc.Mat) -> PETSc.Mat:
    """Assemble a PETSc matrix after row insertions."""
    matrix.assemblyBegin()
    matrix.assemblyEnd()
    return matrix


def _local_row_nnz_split(
    local_csr: sp.csr_matrix, r0: int, r1: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row diagonal/off-diagonal nonzero counts for a locally-owned CSR slice.

    Columns in ``[r0, r1)`` land in the MPI-AIJ diagonal block; all others in the
    off-diagonal block. Returns two ``int64`` arrays of length ``r1 - r0`` ready
    for ``create_distributed_aij(nnz_diag=..., nnz_offdiag=...)``. The cumulative
    sum keyed on ``indptr`` counts per row correctly even for empty rows.
    """
    indptr = local_csr.indptr
    in_diag = (local_csr.indices >= r0) & (local_csr.indices < r1)
    csum = np.concatenate(([0], np.cumsum(in_diag, dtype=np.int64)))
    diag = csum[indptr[1:]] - csum[indptr[:-1]]
    offdiag = np.diff(indptr).astype(np.int64) - diag
    return diag, offdiag


def scipy_to_petsc_distributed(
    mat: sp.spmatrix,
    *,
    comm: PETSc.Comm | None = None,
) -> PETSc.Mat:
    """Convert SciPy sparse matrix into distributed PETSc AIJ by row ownership.

    Each rank still needs a SciPy matrix object, but inserts only its
    locally-owned row slice into the PETSc matrix.

    Preallocation uses the exact per-row diagonal/off-diagonal nonzero counts.
    Without this, the row-by-row insert reallocates on nearly every row
    (~O(nnz^2)) and stalls for hours at full resolution
    (e.g. ~9e5 DOF SWBLI operators) before factorization can begin.
    """
    PETSc = _get_petsc()
    comm = comm or PETSc.COMM_WORLD
    csr = sp.csr_matrix(mat, dtype=complex)
    nrows = csr.shape[0]
    # Row ownership matching PETSc's default layout (PetscSplitOwnership): the
    # first (nrows % size) ranks own one extra row. Computing it here lets each
    # rank split its rows into diagonal/off-diagonal blocks consistently with the
    # partition create_distributed_aij derives internally.
    size, rank = comm.getSize(), comm.getRank()
    base, rem = divmod(nrows, size)
    local_n = base + (1 if rank < rem else 0)
    r0 = rank * base + min(rank, rem)
    r1 = r0 + local_n

    local = csr[r0:r1]
    d_nnz, o_nnz = _local_row_nnz_split(local, r0, r1)
    matrix = create_distributed_aij(csr.shape, comm=comm, nnz_diag=d_nnz, nnz_offdiag=o_nnz)

    rm0, rm1 = ownership_range(matrix)
    if (rm0, rm1) != (r0, r1):  # defensive: layouts must agree (raises on all ranks)
        raise RuntimeError(f"ownership mismatch: computed [{r0}, {r1}) vs PETSc [{rm0}, {rm1})")
    insert_local_csr_rows(matrix, local, row_start=r0)
    return assemble_matrix(matrix)


def scipy_to_petsc_distributed_from_root(
    mat: sp.spmatrix | None,
    *,
    shape: tuple[int, int] | None = None,
    root: int = 0,
    comm: PETSc.Comm | None = None,
) -> PETSc.Mat:
    """Build distributed PETSc AIJ by scattering CSR row blocks from ``root``.

    Parameters
    ----------
    mat:
        Global SciPy sparse matrix on ``root`` rank. Non-root ranks may pass
        ``None``.
    shape:
        Global matrix shape. Optional on ``root`` (inferred from ``mat``).
        Required on non-root ranks when ``mat`` is ``None``.
    root:
        Root rank holding the global SciPy matrix.
    comm:
        PETSc communicator to use.

    Notes
    -----
    Avoid a full SciPy matrix on every rank.
    Use this for MPI shift-invert paths that prefer root-owned assembly.
    """
    PETSc = _get_petsc()
    petsc_comm = comm or PETSc.COMM_WORLD
    mpi_comm = petsc_comm.tompi4py()
    rank = int(mpi_comm.Get_rank())

    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f"shape must be 2D, got {shape}")
        shape = (int(shape[0]), int(shape[1]))

    root_csr: sp.csr_matrix | None = None
    root_shape: tuple[int, int] | None
    if rank == int(root):
        if mat is None:
            raise ValueError("Root rank must provide a SciPy matrix for rooted distribution")
        root_csr = sp.csr_matrix(mat, dtype=complex)
        root_shape = root_csr.shape
        if shape is not None and root_shape != shape:
            raise ValueError(
                f"Provided shape {shape} does not match root matrix shape {root_shape}"
            )
    else:
        root_shape = shape
        if root_shape is None:
            raise ValueError("Non-root ranks must provide shape when mat=None")

    global_shape = mpi_comm.bcast(root_shape if rank == int(root) else None, root=int(root))
    if global_shape is None:
        raise ValueError("Failed to broadcast global matrix shape from root")

    matrix = create_distributed_aij(global_shape, comm=petsc_comm)
    r0, r1 = ownership_range(matrix)
    owned_ranges = mpi_comm.gather((int(r0), int(r1)), root=int(root))

    send_payloads = None
    if rank == int(root):
        assert root_csr is not None
        send_payloads = []
        for rr0, rr1 in owned_ranges:
            block = root_csr[rr0:rr1]
            send_payloads.append(
                (
                    int(rr0),
                    np.asarray(block.indptr, dtype=np.int64),
                    np.asarray(block.indices, dtype=np.int64),
                    np.asarray(block.data, dtype=complex),
                    int(block.shape[0]),
                )
            )

    row_start, indptr, indices, data, nrows = mpi_comm.scatter(send_payloads, root=int(root))
    local_rows = sp.csr_matrix(
        (data, indices, indptr),
        shape=(int(nrows), int(global_shape[1])),
        dtype=complex,
    )
    if int(nrows) > 0:
        insert_local_csr_rows(matrix, local_rows, row_start=int(row_start))
    return assemble_matrix(matrix)


def petsc_to_scipy(mat: PETSc.Mat, *, format: str = "csr") -> sp.csr_matrix | sp.csc_matrix:
    """Convert a sequential PETSc AIJ matrix into a SciPy sparse matrix.

    Parameters
    ----------
    mat:
        PETSc matrix to extract.
    format:
        Output sparse format, either ``"csr"`` or ``"csc"``.

    Notes
    -----
    The initial implementation supports only sequential matrices.
    That is enough for parity tests and local development.
    Distributed-gather support can wait.
    """
    out_format = format.lower()
    if out_format not in {"csr", "csc"}:
        raise ValueError(f"Unsupported sparse format {format!r}; expected 'csr' or 'csc'")

    comm = mat.getComm()
    if comm.getSize() != 1:
        raise NotImplementedError(
            "petsc_to_scipy currently supports only sequential PETSc matrices"
        )

    try:
        indptr, indices, data = mat.getValuesCSR()
    except Exception as exc:  # pragma: no cover - PETSc type guard
        raise TypeError("PETSc matrix does not support CSR extraction via getValuesCSR()") from exc

    csr = sp.csr_matrix(
        (
            np.array(data, copy=True),
            np.array(indices, dtype=np.int64, copy=True),
            np.array(indptr, dtype=np.int64, copy=True),
        ),
        shape=mat.getSize(),
        dtype=complex,
    )
    csr.sum_duplicates()
    csr.eliminate_zeros()
    if out_format == "csr":
        return csr
    return csr.tocsc()


def _canonical_sparse(mat: sp.spmatrix) -> sp.csr_matrix:
    """Return a canonical CSR matrix for sparse comparisons."""
    csr = sp.csr_matrix(mat, dtype=complex).copy()
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    return csr


def _matrix_entry(mat: sp.csr_matrix, row: int, col: int) -> complex:
    """Return a scalar sparse entry as a Python complex number."""
    value = mat[row, col]
    if hasattr(value, "item"):
        value = value.item()
    return complex(value)


def compare_sparse_matrices(
    reference: sp.spmatrix,
    candidate: sp.spmatrix,
    *,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-12,
    check_structure: bool = False,
    max_terms: int = 20,
) -> dict:
    """Compare two sparse matrices and return a structured parity report."""
    ref = _canonical_sparse(reference)
    cand = _canonical_sparse(candidate)

    shape_match = ref.shape == cand.shape
    if not shape_match:
        return {
            "shape_match": False,
            "reference_shape": ref.shape,
            "candidate_shape": cand.shape,
            "nnz_reference": int(ref.nnz),
            "nnz_candidate": int(cand.nnz),
            "nnz_delta": None,
            "structure_match": False,
            "max_abs_residual": float("inf"),
            "max_rel_residual": float("inf"),
            "frob_rel_residual": float("inf"),
            "threshold": None,
            "largest_mismatches": [],
            "pass": False,
        }

    structure_match = bool(
        np.array_equal(ref.indptr, cand.indptr) and np.array_equal(ref.indices, cand.indices)
    )

    delta = (cand - ref).tocsr()
    delta.sum_duplicates()
    delta.eliminate_zeros()

    if delta.nnz:
        max_abs_residual = float(np.max(np.abs(delta.data)))
        ref_scale = float(np.max(np.abs(ref.data))) if ref.nnz else 0.0
        delta_norm = float(np.linalg.norm(delta.data))
        ref_norm = float(np.linalg.norm(ref.data)) if ref.nnz else 0.0

        # Work entirely in COO so row/col/data arrays share the same ordering.
        delta_coo = delta.tocoo()
        abs_delta_coo = np.abs(delta_coo.data)
        order = np.argsort(abs_delta_coo)[::-1]
        largest_mismatches = []
        for idx in order[:max_terms]:
            row = int(delta_coo.row[idx])
            col = int(delta_coo.col[idx])
            ref_val = _matrix_entry(ref, row, col)
            cand_val = _matrix_entry(cand, row, col)
            resid = cand_val - ref_val
            largest_mismatches.append(
                {
                    "row": row,
                    "col": col,
                    "reference": ref_val,
                    "candidate": cand_val,
                    "abs_residual": float(abs(resid)),
                    "rel_residual": float(abs(resid) / max(abs(ref_val), 1e-30)),
                }
            )
    else:
        max_abs_residual = 0.0
        ref_scale = float(np.max(np.abs(ref.data))) if ref.nnz else 0.0
        delta_norm = 0.0
        ref_norm = float(np.linalg.norm(ref.data)) if ref.nnz else 0.0
        largest_mismatches = []

    max_rel_residual = float(max_abs_residual / max(ref_scale, 1e-30))
    frob_rel_residual = float(delta_norm / max(ref_norm, 1e-30))
    threshold = float(abs_tol + rel_tol * ref_scale)
    passed = bool(max_abs_residual <= threshold and (structure_match or not check_structure))

    return {
        "shape_match": True,
        "reference_shape": ref.shape,
        "candidate_shape": cand.shape,
        "nnz_reference": int(ref.nnz),
        "nnz_candidate": int(cand.nnz),
        "nnz_delta": int(delta.nnz),
        "structure_match": structure_match,
        "max_abs_residual": max_abs_residual,
        "max_rel_residual": max_rel_residual,
        "frob_rel_residual": frob_rel_residual,
        "threshold": threshold,
        "largest_mismatches": largest_mismatches,
        "pass": passed,
    }


def compare_petsc_against_scipy(
    reference: sp.spmatrix,
    candidate: PETSc.Mat,
    *,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-12,
    check_structure: bool = False,
    max_terms: int = 20,
) -> dict:
    """Compare a PETSc matrix against a SciPy sparse reference matrix."""
    candidate_sp = petsc_to_scipy(candidate, format="csr")
    report = compare_sparse_matrices(
        reference,
        candidate_sp,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        check_structure=check_structure,
        max_terms=max_terms,
    )
    report["candidate_format"] = "petsc"
    return report


def petsc_matvec_residual(
    reference: sp.spmatrix,
    candidate: PETSc.Mat,
    *,
    seed: int = 0,
    trials: int = 3,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-12,
) -> dict:
    """Compare PETSc and SciPy matrix-vector products over random trials."""
    ref = _canonical_sparse(reference)
    if ref.shape != candidate.getSize():
        raise ValueError(
            f"Shape mismatch: reference has shape {ref.shape}, candidate has shape {candidate.getSize()}"
        )
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")

    rng = np.random.default_rng(seed)
    max_abs_residual = 0.0
    max_rel_residual = 0.0
    trial_reports = []

    for trial in range(trials):
        x = rng.standard_normal(ref.shape[1]) + 1j * rng.standard_normal(ref.shape[1])
        y_ref = ref @ x

        x_vec = petsc_vec(x)
        y_vec = candidate.createVecLeft()
        try:
            candidate.mult(x_vec, y_vec)
            y_petsc = vec_to_numpy(y_vec)
        finally:
            x_vec.destroy()
            y_vec.destroy()

        residual = y_petsc - y_ref
        trial_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
        trial_rel = float(np.linalg.norm(residual) / max(np.linalg.norm(y_ref), 1e-30))
        trial_reports.append(
            {
                "trial": trial,
                "max_abs_residual": trial_abs,
                "rel_residual": trial_rel,
            }
        )
        max_abs_residual = max(max_abs_residual, trial_abs)
        max_rel_residual = max(max_rel_residual, trial_rel)

    ref_scale = float(np.max(np.abs(ref.data))) if ref.nnz else 0.0
    threshold = float(abs_tol + rel_tol * ref_scale)
    passed = bool(max_abs_residual <= threshold or max_rel_residual <= rel_tol)

    return {
        "trials": trial_reports,
        "max_abs_residual": max_abs_residual,
        "max_rel_residual": max_rel_residual,
        "threshold": threshold,
        "pass": passed,
    }


def petsc_vec(arr: np.ndarray) -> PETSc.Vec:
    """Convert a full-global 1-D numpy array to a PETSc Vec.

    Under MPI the input ``arr`` is a full-global array (every rank holds the
    whole vector).
    ``createWithArray`` with no communicator/size split would treat each
    rank's entire copy as its local slice.
    That produces a vec of global size ``N * nranks`` (the array replicated
    across ranks).
    That replicated vec then mismatches any distributed operator that
    receives it.
    Build a vec of the correct global size ``N`` instead, with each rank
    holding only its ownership slice.
    See ``tests/test_petsc_vec_mpi.py``.
    """
    PETSc = _get_petsc()
    arr = np.ascontiguousarray(arr, dtype=complex)
    if PETSc.COMM_WORLD.getSize() == 1:
        return PETSc.Vec().createWithArray(arr)
    n = int(arr.shape[0])
    v = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    v.setSizes((PETSc.DECIDE, n))
    v.setType("standard")
    lo, hi = v.getOwnershipRange()
    v.getArray()[:] = arr[lo:hi]
    return v


def vec_to_numpy(v: PETSc.Vec) -> np.ndarray:
    """Copy a PETSc Vec to a numpy array.

    Use read-only access (``array_r``) so the function works correctly
    inside PETSc MatShell callbacks where PETSc locks the input vector.
    For distributed vectors, gathers local slices from all ranks in
    ownership order and returns a global 1-D array.
    """
    PETSc = _get_petsc()
    local = np.array(v.array_r, copy=True)
    comm = v.getComm()
    if comm.getSize() == 1:
        return local
    scatter, gathered = PETSc.Scatter.toAll(v)
    try:
        scatter.scatter(
            v,
            gathered,
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.Scatter.Mode.FORWARD,
        )
        return np.array(gathered.array_r, copy=True)
    finally:
        scatter.destroy()
        gathered.destroy()


def numpy_columns_to_petsc_vecs(
    template: PETSc.Mat | PETSc.Vec, modes: np.ndarray
) -> list[PETSc.Vec]:
    """Convert dense full-state mode columns into ownership-compatible PETSc Vecs."""
    PETSc = _get_petsc()
    arr = np.asarray(modes, dtype=complex)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("modes must be a 1-D vector or a 2-D column matrix")

    make_vec = (
        template.createVecRight if hasattr(template, "createVecRight") else template.duplicate
    )
    probe = make_vec()
    try:
        r0, r1 = probe.getOwnershipRange()
        ndof = int(probe.getSize())
    finally:
        probe.destroy()

    if arr.shape[0] != ndof:
        raise ValueError(
            f"initial-space row count {arr.shape[0]} does not match operator size {ndof}"
        )

    vecs: list[PETSc.Vec] = []
    rows = np.arange(r0, r1, dtype=PETSc.IntType) if r1 > r0 else np.array([], dtype=PETSc.IntType)
    for j in range(arr.shape[1]):
        vec = make_vec()
        column = np.asarray(arr[:, j], dtype=complex)
        norm = float(np.linalg.norm(column))
        if norm > 0.0:
            column = column / norm
        if r1 > r0:
            vec.setValues(rows, column[r0:r1], addv=PETSc.InsertMode.INSERT_VALUES)
        vec.assemblyBegin()
        vec.assemblyEnd()
        vecs.append(vec)
    return vecs
