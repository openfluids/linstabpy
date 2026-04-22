"""Minimal slepc4py non-Hermitian EPS reproducer for macOS install debugging."""

from __future__ import annotations

from petsc4py import PETSc
from slepc4py import SLEPc


def build_matrix() -> PETSc.Mat:
    mat = PETSc.Mat().createAIJ([2, 2], nnz=2, comm=PETSc.COMM_SELF)
    mat.setUp()
    mat[0, 0] = 1.0
    mat[0, 1] = 2.0
    mat[1, 1] = 3.0
    mat.assemble()
    return mat


def main() -> None:
    mat = build_matrix()
    eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
    eps.setOperators(mat)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(1)
    print("before solve", flush=True)
    eps.solve()
    print(f"after solve nconv={eps.getConverged()}", flush=True)


if __name__ == "__main__":
    main()
