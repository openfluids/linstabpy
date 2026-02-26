"""SciPy <-> PETSc conversion utilities (local and distributed scaffolding).

Shared by both mesh (filters) and solver modules to avoid circular
dependencies between the mesh and solver layers.
"""

from __future__ import annotations

import os
import numpy as np
import scipy.sparse as sp

from .petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc


_petsc_configured = False


def _petsc_matrix_mode() -> str:
    """Resolve PETSc matrix conversion mode."""
    raw = os.getenv("LINSTAB2D_PETSC_MATRIX_MODE", "auto").strip().lower()
    if raw in {"auto", "local", "serial", "distributed", "mpi", "parallel"}:
        return raw
    raise ValueError(
        "Invalid LINSTAB2D_PETSC_MATRIX_MODE="
        f"{raw!r}; expected one of auto|local|serial|distributed|mpi|parallel"
    )


def _use_distributed_matrix() -> bool:
    mode = _petsc_matrix_mode()
    if mode in {"distributed", "mpi", "parallel"}:
        return True
    if mode in {"local", "serial"}:
        return False
    # auto
    return PETSc.COMM_WORLD.getSize() > 1


def require_complex() -> None:
    """Assert PETSc was built with complex scalar type and set defaults."""
    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        raise RuntimeError(
            "PETSc must be built with --with-scalar-type=complex for stability analysis. "
            f"Current scalar type: {PETSc.ScalarType}"
        )
    global _petsc_configured
    if not _petsc_configured:
        if os.getenv("LINSTAB2D_PETSC_SET_MUMPS_ICNTL_7", "0") == "1":
            PETSc.Options().setValue("mat_mumps_icntl_7", "5")  # METIS ordering
        _petsc_configured = True


def scipy_to_petsc(mat: sp.spmatrix) -> PETSc.Mat:
    """Convert a SciPy sparse matrix to a PETSc AIJ Mat.

    Behavior is controlled by ``LINSTAB2D_PETSC_MATRIX_MODE``:
    - ``local``/``serial``: local AIJ from full CSR (single-rank style)
    - ``distributed``/``mpi``/``parallel``: ownership-aware distributed AIJ
    - ``auto`` (default): distributed when MPI size > 1, else local
    """
    if _use_distributed_matrix():
        return scipy_to_petsc_distributed(mat, comm=PETSc.COMM_WORLD)

    csr = sp.csr_matrix(mat, dtype=complex)
    A = PETSc.Mat().createAIJ(
        size=csr.shape,
        csr=(csr.indptr, csr.indices, csr.data),
    )
    A.assemble()
    return A


def create_distributed_aij(
    shape: tuple[int, int],
    *,
    comm: PETSc.Comm | None = None,
    nnz_diag: int | None = None,
    nnz_offdiag: int | None = None,
) -> PETSc.Mat:
    """Create an MPI AIJ matrix with optional constant preallocation."""
    if len(shape) != 2:
        raise ValueError(f"shape must be 2D, got {shape}")
    nrows, ncols = int(shape[0]), int(shape[1])
    if nrows <= 0 or ncols <= 0:
        raise ValueError(f"shape entries must be positive, got {shape}")

    matrix = PETSc.Mat().create(comm=comm or PETSc.COMM_WORLD)
    matrix.setType(PETSc.Mat.Type.AIJ)
    matrix.setSizes((nrows, ncols))

    if nnz_diag is not None or nnz_offdiag is not None:
        diag_nnz = int(nnz_diag if nnz_diag is not None else nnz_offdiag)
        offdiag_nnz = int(nnz_offdiag if nnz_offdiag is not None else diag_nnz)
        matrix.setPreallocationNNZ((diag_nnz, offdiag_nnz))

    matrix.setUp()
    return matrix


def ownership_range(matrix: PETSc.Mat) -> tuple[int, int]:
    """Return global owned row range for this rank: [rstart, rend)."""
    rstart, rend = matrix.getOwnershipRange()
    return int(rstart), int(rend)


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
    This function only inserts values; caller must assemble with
    ``assemble_matrix`` after all insertions.
    """
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
            f"Row block [{row_start}, {row_stop}) is outside local ownership "
            f"[{r0}, {r1})"
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


def scipy_to_petsc_distributed(
    mat: sp.spmatrix,
    *,
    comm: PETSc.Comm | None = None,
) -> PETSc.Mat:
    """Convert SciPy sparse matrix into distributed PETSc AIJ by row ownership.

    This is a transitional helper: it still requires a SciPy matrix object but
    inserts only the locally-owned row slice into the PETSc matrix on each rank.
    """
    csr = sp.csr_matrix(mat, dtype=complex)
    matrix = create_distributed_aij(csr.shape, comm=comm)
    r0, r1 = ownership_range(matrix)
    insert_local_csr_rows(matrix, csr[r0:r1], row_start=r0)
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
    This helper avoids requiring a full SciPy matrix on every rank and is
    intended for MPI shift-invert paths where root-owned assembly is preferred.
    """
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
            raise ValueError(f"Provided shape {shape} does not match root matrix shape {root_shape}")
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


def petsc_vec(arr: np.ndarray) -> PETSc.Vec:
    """Convert a 1-D numpy array to a PETSc Vec."""
    v = PETSc.Vec().createWithArray(np.ascontiguousarray(arr, dtype=complex))
    return v


def vec_to_numpy(v: PETSc.Vec) -> np.ndarray:
    """Copy a PETSc Vec to a numpy array.

    Uses read-only access (``array_r``) so the function works correctly
    inside PETSc MatShell callbacks where the input vector is locked.
    For distributed vectors, gathers local slices from all ranks in
    ownership order and returns a global 1-D array.
    """
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
