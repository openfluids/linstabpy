"""Spatial eigenproblem assembly for LNSE spatial stability analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from ..mesh.mesh import Mesh
from ..baseflow.baseflow import BaseFlow
from .assembly import (
    _prepare_coefficients,
    _assemble_from_prepared,
    _assemble_from_prepared_petsc,
    _build_index_structure,
    _validate_direct_petsc_backend,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only optional dependency
    from petsc4py import PETSc


def build_spatial_operator(
    mesh: Mesh,
    baseflow: BaseFlow,
    model: str = "2D",
    floquet_exp: float = 0.0,
    *,
    kx: float = 0.0,
    m: float = 0,
    delta: float = 1.0,
) -> tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, dict]:
    """Build doubled linear operator blocks for the spatial EVP.

    The spatial formulation is constructed from finite-difference derivatives of
    the LNSE operator with respect to a Floquet-like axial exponent ``alpha``:

    - R0 = L(alpha=0)
    - R1 = (L(+delta) - L(-delta)) / (2*delta)
    - R2 = (L(+delta) - R0 - R1*delta) / delta**2

    And the generalized polynomial EVP is assembled as:

    [ 0,  I] q = alpha [ 0, 0 ] q + [ I, 0 ] q  + terms
    [-R0, 0]    [R1, R2]   [0, I]

    implemented via
        Lc = [[0, I], [-R0, 0]]
        R  = [[I, 0], [R1, R2]]
        Lw = [[0, 0], [-iI, 0]]
    """
    if delta <= 0:
        raise ValueError("delta must be positive")

    n = mesh.ngp * 5

    # Compute coefficients once, assemble three operators with different shifts
    prep = _prepare_coefficients(mesh, baseflow, kx=kx, model=model, m=m)

    L0 = _assemble_from_prepared(prep, floquet_exp=floquet_exp)
    Lp = _assemble_from_prepared(prep, floquet_exp=floquet_exp + delta)
    Lm = _assemble_from_prepared(prep, floquet_exp=floquet_exp - delta)

    R0 = L0.tocsc()
    R1 = (Lp - Lm) / (2.0 * delta)
    R2 = (Lp - R0 - R1 * delta) / delta**2

    I = sp.eye(n, format="csc")
    Z = sp.csc_matrix((n, n))

    Lc = sp.bmat([[Z, I], [-R0, Z]], format="csc")
    R = sp.bmat([[I, Z], [R1, R2]], format="csc")
    Lw = sp.bmat([[Z, Z], [-1j * I, Z]], format="csc")

    idx = _build_index_structure(mesh.idx, mesh.ngp)
    return Lc, R, Lw, idx


def build_spatial_operator_petsc(
    mesh: Mesh,
    baseflow: BaseFlow,
    model: str = "2D",
    floquet_exp: float = 0.0,
    *,
    kx: float = 0.0,
    m: float = 0,
    delta: float = 1.0,
    solver_cfg: dict | None = None,
) -> tuple["PETSc.Mat", "PETSc.Mat", "PETSc.Mat", dict]:
    """Build doubled spatial-EVP matrices directly as PETSc AIJ matrices.

    This currently targets the validated local/sequential 2D direct-PETSc path.
    """
    from petsc4py import PETSc

    if delta <= 0:
        raise ValueError("delta must be positive")

    _validate_direct_petsc_backend(model=model, solver_cfg=solver_cfg)

    n = mesh.ngp * 5
    prep = _prepare_coefficients(mesh, baseflow, kx=kx, model=model, m=m)

    L0 = _assemble_from_prepared_petsc(prep, floquet_exp=floquet_exp, solver_cfg=solver_cfg)
    Lp = _assemble_from_prepared_petsc(prep, floquet_exp=floquet_exp + delta, solver_cfg=solver_cfg)
    Lm = _assemble_from_prepared_petsc(prep, floquet_exp=floquet_exp - delta, solver_cfg=solver_cfg)

    R0 = L0  # L0 is not used separately after this point
    R1 = Lp.copy()
    R1.axpy(-1.0, Lm, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    R1.scale(1.0 / (2.0 * delta))

    R2 = Lp.copy()
    R2.axpy(-1.0, R0, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    R2.axpy(-delta, R1, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    R2.scale(1.0 / delta**2)

    I = _create_identity_petsc(n, comm=L0.getComm())
    minus_R0 = R0.copy()
    minus_R0.scale(-1.0)
    minus_iI = _create_identity_petsc(n, comm=L0.getComm(), scale=-1j)

    Lc = _assemble_block_matrix_petsc([[None, I], [minus_R0, None]])
    R = _assemble_block_matrix_petsc([[I, None], [R1, R2]])
    Lw = _assemble_block_matrix_petsc([[None, None], [minus_iI, None]])

    # Destroy intermediate PETSc matrices no longer needed after block assembly.
    # R0 is L0 (alias, not a copy). R1/R2 were copied into output R by
    # _assemble_block_matrix_petsc. Destroy all intermediates.
    for mat in (R0, Lp, Lm, R1, R2, I, minus_R0, minus_iI):
        mat.destroy()

    idx = _build_index_structure(mesh.idx, mesh.ngp)
    return Lc, R, Lw, idx


def _create_identity_petsc(
    n: int,
    *,
    comm: "PETSc.Comm",
    scale: complex = 1.0,
) -> "PETSc.Mat":
    """Create a sequential/local PETSc AIJ identity matrix with optional scaling."""
    from petsc4py import PETSc

    indptr = np.arange(n + 1, dtype=np.int32)
    indices = np.arange(n, dtype=np.int32)
    data = np.full(n, complex(scale), dtype=complex)
    I = PETSc.Mat().createAIJ(size=(n, n), csr=(indptr, indices, data), comm=comm)
    I.assemble()
    return I


def _assemble_block_matrix_petsc(blocks: list[list["PETSc.Mat | None"]]) -> "PETSc.Mat":
    """Assemble a PETSc AIJ matrix from PETSc sub-blocks by CSR row insertion."""
    from petsc4py import PETSc
    from ..utils.petsc_core import assemble_matrix, create_distributed_aij

    if not blocks or not blocks[0]:
        raise ValueError("blocks must be a non-empty 2D matrix of PETSc submatrices")

    n_block_rows = len(blocks)
    n_block_cols = len(blocks[0])
    if any(len(row) != n_block_cols for row in blocks):
        raise ValueError("All block rows must have the same length")

    row_sizes: list[int | None] = [None] * n_block_rows
    col_sizes: list[int | None] = [None] * n_block_cols
    comm = None

    for i, row in enumerate(blocks):
        for j, block in enumerate(row):
            if block is None:
                continue
            brow, bcol = block.getSize()
            comm = block.getComm() if comm is None else comm
            if row_sizes[i] is None:
                row_sizes[i] = int(brow)
            elif row_sizes[i] != int(brow):
                raise ValueError("Inconsistent PETSc block row sizes")
            if col_sizes[j] is None:
                col_sizes[j] = int(bcol)
            elif col_sizes[j] != int(bcol):
                raise ValueError("Inconsistent PETSc block column sizes")

    if comm is None:
        raise ValueError("At least one PETSc submatrix is required")

    known_row_sizes = {size for size in row_sizes if size is not None}
    known_col_sizes = {size for size in col_sizes if size is not None}
    if not known_row_sizes or not known_col_sizes:
        raise ValueError("Unable to infer PETSc block sizes")
    if any(size is None for size in row_sizes):
        if len(known_row_sizes) != 1:
            raise ValueError("Cannot infer missing PETSc block row sizes from non-uniform blocks")
        fill_row = next(iter(known_row_sizes))
        row_sizes = [fill_row if size is None else size for size in row_sizes]
    if any(size is None for size in col_sizes):
        if len(known_col_sizes) != 1:
            raise ValueError("Cannot infer missing PETSc block column sizes from non-uniform blocks")
        fill_col = next(iter(known_col_sizes))
        col_sizes = [fill_col if size is None else size for size in col_sizes]

    row_sizes_i = [int(size) for size in row_sizes]
    col_sizes_i = [int(size) for size in col_sizes]
    row_offsets = np.cumsum([0, *row_sizes_i])
    col_offsets = np.cumsum([0, *col_sizes_i])

    matrix = create_distributed_aij((int(row_offsets[-1]), int(col_offsets[-1])), comm=comm)

    for i, row in enumerate(blocks):
        for j, block in enumerate(row):
            if block is None:
                continue
            indptr, indices, data = block.getValuesCSR()
            row_offset = int(row_offsets[i])
            col_offset = int(col_offsets[j])
            nrows = int(block.getSize()[0])
            for local_row in range(nrows):
                p0 = int(indptr[local_row])
                p1 = int(indptr[local_row + 1])
                if p0 == p1:
                    continue
                matrix.setValues(
                    row_offset + local_row,
                    np.asarray(indices[p0:p1], dtype=np.int32) + col_offset,
                    np.asarray(data[p0:p1], dtype=complex),
                    addv=PETSc.InsertMode.INSERT_VALUES,
                )

    return assemble_matrix(matrix)

