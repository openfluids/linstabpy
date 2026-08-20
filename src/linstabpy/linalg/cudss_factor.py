# SPDX-License-Identifier: GPL-3.0-only
"""cuDSS sparse-direct factorization through nvmath-python.

This module wraps NVIDIA cuDSS for the resolvent and shift-invert solves.
The wrapper gives the four operations the solvers need:

1. plan: the analysis (reordering) phase, once per sparsity pattern;
2. factorize: the numeric factorization;
3. refactorize: new values on the kept pattern, without a new analysis;
4. solve: one right-hand side per call.

The module is PETSc-free.  The PETSc coupling (a Python preconditioner
inside FGMRES) lives in :mod:`linstabpy.linalg.factorize`.

Two nvmath rules shape the code.  A host operand is copied to the device
when it is set, not on each call, so ``solve`` must call
``reset_operands(b=...)`` and ``refactorize`` must call
``reset_operands(a=...)``.  And iterative refinement is turned off
(``ir_num_steps = 0``): the outer FGMRES owns the accuracy.

Precision policy, measured on a 450k-unknown SWBLI resolvent operator:

- ``double``: complex128 factor, no matching, no equilibration.  FGMRES
  needs 1-2 iterations and matches a MUMPS reference to 1e-12.
- ``single``: complex64 factor.  The factorization is 6-9x faster and
  uses half the device memory, but it fails near zero frequency unless
  two measures are both active: MC64-style matching
  (``MAX_DIAG_PRODUCT``) and double-precision equilibration before the
  cast.  This class force-enables both; they are not options.

Environment overrides:

- ``LINSTABPY_CUDSS_MT_LIB``: path to the cuDSS threading layer, or
  ``none`` to disable it.  Default: auto-discovery inside the installed
  NVIDIA wheels.  The threading layer halves the analysis time.
- ``LINSTABPY_CUDSS_THREADS``: host threads for the analysis phase.
  Default: the CPU affinity count of the process.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

_EQUILIBRATION_FLOOR = 1.0e-300
_MT_LIB_ENV = "LINSTABPY_CUDSS_MT_LIB"
_THREADS_ENV = "LINSTABPY_CUDSS_THREADS"
_PRECISIONS = frozenset({"double", "single"})

_MISSING_NVMATH_MSG = (
    "factor_backend='cudss' needs nvmath-python with the cuDSS wheels. "
    "Install the optional dependency group: pip install 'linstabpy[cudss]'. "
    "A CUDA 12 runtime and an NVIDIA GPU are required."
)


def _import_nvmath() -> Any:  # ruff: ignore[any-type] -- external module without stubs
    """Import nvmath, or raise with the install command.

    Returns
    -------
    module
        The imported ``nvmath`` module.

    Raises
    ------
    RuntimeError
        When the nvmath wheels are not installed.
    """
    try:
        import nvmath  # ruff: ignore[import-outside-top-level] # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(_MISSING_NVMATH_MSG) from exc
    return nvmath


def find_threading_layer() -> str | None:
    """Locate the cuDSS OpenMP threading layer inside the installed wheels.

    The layer (``libcudss_mtlayer_gomp``) lets the analysis phase use host
    threads.  Measured on the 450k-unknown SWBLI operator: analysis drops
    from 6.7-8.1 s to 3.3-3.4 s.  ``LINSTABPY_CUDSS_MT_LIB`` overrides the
    search; the value ``none`` disables the layer.

    Returns
    -------
    str or None
        Path to the threading layer, or None when it is absent or disabled.
    """
    override = os.environ.get(_MT_LIB_ENV)
    if override is not None:
        text = override.strip()
        if text.lower() in {"", "none", "off"}:
            return None
        return text
    try:
        import nvidia  # ruff: ignore[import-outside-top-level] # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    candidates: list[Path] = []
    for root in nvidia.__path__:
        candidates.extend(Path(root).glob("*/lib/libcudss_mtlayer_gomp.so*"))
    if not candidates:
        logger.debug("No cuDSS threading layer found in the nvidia wheels")
        return None
    return str(min(candidates))


def _default_host_threads() -> int:
    """Resolve host threads for the analysis phase.

    Returns
    -------
    int
        The ``LINSTABPY_CUDSS_THREADS`` value, or the CPU affinity count.
    """
    if _THREADS_ENV in os.environ:
        return max(1, int(os.environ[_THREADS_ENV]))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _csr_int32(kmat: sp.spmatrix) -> sp.csr_matrix:
    """Copy a sparse matrix into the canonical form cuDSS wants.

    cuDSS wants int32 indices.  Sorting fixes one canonical entry order,
    so value updates land on the entries they belong to.

    Returns
    -------
    scipy.sparse.csr_matrix
        A sorted complex128 CSR copy with int32 index arrays.
    """
    csr = sp.csr_matrix(kmat, dtype=np.complex128)
    out = sp.csr_matrix(
        (csr.data.copy(), csr.indices.astype(np.int32), csr.indptr.astype(np.int32)),
        shape=csr.shape,
    )
    out.sort_indices()
    return out


def equilibrate(kmat: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Scale rows and columns to inf-norm one, in double precision.

    Row scaling first (inf-norm), then column scaling on the row-scaled
    matrix.  The scalings condition the matrix before a single-precision
    cast; the outer solve must apply ``D_r`` to the right-hand side and
    ``D_c`` to the solution.

    Returns
    -------
    tuple
        ``(D_r K D_c, r, c)``: the scaled matrix and the two scalings.
    """
    row_max = np.asarray(abs(kmat).max(axis=1).todense()).ravel()
    r = 1.0 / np.maximum(row_max, _EQUILIBRATION_FLOOR)
    scaled = sp.diags(r) @ kmat
    col_max = np.asarray(abs(scaled).max(axis=0).todense()).ravel()
    c = 1.0 / np.maximum(col_max, _EQUILIBRATION_FLOOR)
    scaled = (scaled @ sp.diags(c)).tocsr()
    return scaled, r, c


def _tiny_probe_solve(kmat: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
    """Factor and solve one tiny system, releasing the device resources.

    Returns
    -------
    numpy.ndarray
        The solution of ``kmat x = rhs``.
    """
    solver = CudssSolver(kmat)
    try:
        return solver.solve(rhs)
    finally:
        solver.free()


@functools.cache
def cudss_available() -> bool:
    """Report whether a tiny complex cuDSS factorization works here.

    The probe imports nvmath, factors a 2x2 complex matrix on the GPU and
    checks the solution.  Any failure (no wheel, no driver, no device)
    means the backend is not available; the reason is logged at DEBUG.

    Returns
    -------
    bool
        True when the full stack (wheels, driver, GPU) works.
    """
    try:
        import nvmath  # ruff: ignore[unused-import]  # ruff: ignore[import-outside-top-level] # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    kmat = sp.csr_matrix(np.array([[2.0 + 0.0j, 1.0 + 0.0j], [0.0 + 0.0j, 2.0 + 0.0j]]))
    rhs = np.array([1.0 + 0.0j, 1.0 + 0.0j])
    try:
        x = _tiny_probe_solve(kmat, rhs)
    except Exception:  # any CUDA/driver failure means "not available"
        logger.debug("cuDSS availability probe failed", exc_info=True)
        return False
    return bool(np.allclose(kmat @ x, rhs))


class CudssSolver:
    """One cuDSS LU of a square complex CSR matrix, ready for many solves.

    Parameters
    ----------
    kmat:
        Square complex sparse matrix.  Copied into a canonical sorted
        int32 CSR form; the caller's matrix is not modified.
    precision:
        ``"double"`` (default) or ``"single"``.  Single precision
        force-enables MAX_DIAG_PRODUCT matching and double-precision
        equilibration; see the module docstring for the measurement.
    host_threads:
        Threads for the analysis phase.  Default: process CPU affinity,
        overridable with ``LINSTABPY_CUDSS_THREADS``.
    mt_lib:
        Path to the cuDSS threading layer, ``"auto"`` (default) to
        discover it in the installed wheels, or ``None`` to disable.

    Attributes
    ----------
    plan_seconds, factor_seconds, refactor_seconds:
        Wall time of the last analysis / factorization / refactorization.
    lu_nnz:
        Nonzeros in L+U as reported by cuDSS, or None if not reported.
    """

    def __init__(
        self,
        kmat: sp.spmatrix,
        *,
        precision: str = "double",
        host_threads: int | None = None,
        mt_lib: str | None = "auto",
    ) -> None:
        """Build the plan and the first factorization.

        Raises
        ------
        ValueError
            When the matrix is not square or the precision is unknown.
        """
        nrows, ncols = kmat.shape
        if nrows != ncols:
            msg = f"cuDSS needs a square matrix, got shape {kmat.shape}"
            raise ValueError(msg)
        if precision not in _PRECISIONS:
            msg = f"precision must be one of {sorted(_PRECISIONS)}, got {precision!r}"
            raise ValueError(msg)
        self.n = int(nrows)
        self.precision = precision
        self._dtype = np.complex64 if precision == "single" else np.complex128
        self._row_scale: np.ndarray | None = None
        self._col_scale: np.ndarray | None = None

        nvmath = _import_nvmath()
        adv = nvmath.sparse.advanced

        base = _csr_int32(kmat)
        self._indptr = base.indptr
        self._indices = base.indices
        self._a = sp.csr_matrix(
            (self._prepare_data(base), self._indices, self._indptr), shape=(self.n, self.n)
        )
        self._b = np.zeros(self.n, dtype=self._dtype)

        execution = adv.ExecutionCUDA(device_id=0)
        layer = find_threading_layer() if mt_lib == "auto" else mt_lib
        options = adv.DirectSolverOptions(multithreading_lib=str(layer)) if layer else None
        self._solver = adv.DirectSolver(self._a, self._b, options=options, execution=execution)

        plan_config = self._solver.plan_config
        if precision == "single":
            # Measured requirement, not a tuning knob: without matching the
            # single-precision factor stalls FGMRES near zero frequency.
            plan_config.matching_algorithm = nvmath.bindings.cudss.MatchingAlg.MAX_DIAG_PRODUCT
        plan_config.host_nthreads = int(host_threads) if host_threads else _default_host_threads()
        # No hidden iterative refinement: the outer FGMRES owns the accuracy.
        self._solver.solution_config.ir_num_steps = 0

        start = time.perf_counter()
        self._solver.plan()
        self.plan_seconds = time.perf_counter() - start
        start = time.perf_counter()
        info = self._solver.factorize()
        self.factor_seconds = time.perf_counter() - start
        self.refactor_seconds = float("nan")
        self.lu_nnz = self._read_lu_nnz(info)
        logger.info(
            "cuDSS %s factor: n=%d, plan %.2f s, factor %.2f s, lu_nnz=%s, threading_layer=%s",
            precision,
            self.n,
            self.plan_seconds,
            self.factor_seconds,
            self.lu_nnz,
            layer or "off",
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve for one right-hand side.

        For a single-precision factor the equilibration scalings wrap the
        call: ``x = D_c F^{-1} (D_r b)``.

        Returns
        -------
        numpy.ndarray
            The solution, always in complex128.

        Raises
        ------
        ValueError
            When the right-hand side has the wrong size.
        """
        rhs_arr = np.asarray(rhs).reshape(-1)
        if rhs_arr.size != self.n:
            msg = f"rhs must have size {self.n}, got {rhs_arr.size}"
            raise ValueError(msg)
        work = rhs_arr if self._row_scale is None else self._row_scale * rhs_arr
        self._b[:] = work
        # A host operand is copied to the device when it is (re)set, not read
        # per call: without reset_operands every solve reuses the first copy.
        self._solver.reset_operands(b=self._b)
        x = np.asarray(self._solver.solve()).reshape(-1).astype(np.complex128)
        if self._col_scale is not None:
            x = self._col_scale * x
        return x

    def refactorize(self, kmat: sp.spmatrix) -> None:
        """Load new values on the kept pattern and redo the numeric factor.

        The analysis (plan) is kept; only the numeric phase runs.  The new
        matrix must have exactly the nonzero pattern of the original one.

        Raises
        ------
        ValueError
            When the new matrix has a different nonzero pattern.
        """
        base = _csr_int32(kmat)
        if not (
            np.array_equal(base.indptr, self._indptr)
            and np.array_equal(base.indices, self._indices)
        ):
            msg = "refactorize requires an identical nonzero pattern"
            raise ValueError(msg)
        self._a.data[:] = self._prepare_data(base)
        start = time.perf_counter()
        # Same host-operand rule as solve: set the operand so the new values
        # reach the device, then rerun the numeric factorization.
        self._solver.reset_operands(a=self._a)
        self._solver.factorize()
        self.refactor_seconds = time.perf_counter() - start

    def free(self) -> None:
        """Release the device resources held by the solver."""
        self._solver.free()

    def _prepare_data(self, base: sp.csr_matrix) -> np.ndarray:
        """Prepare factor-ready values for ``base`` in the canonical order.

        Double precision: a plain complex128 copy.  Single precision:
        equilibrate in double, store the scalings, then cast to complex64.

        Returns
        -------
        numpy.ndarray
            The value array to load into the cuDSS operand.

        Raises
        ------
        RuntimeError
            When equilibration changes the nonzero pattern (a bug guard).
        """
        if self.precision == "double":
            return base.data.astype(np.complex128, copy=True)
        scaled, row_scale, col_scale = equilibrate(base)
        canonical = _csr_int32(scaled)
        if not (
            np.array_equal(canonical.indptr, base.indptr)
            and np.array_equal(canonical.indices, base.indices)
        ):
            msg = "equilibration changed the nonzero pattern; this is a bug"
            raise RuntimeError(msg)
        self._row_scale = row_scale
        self._col_scale = col_scale
        return canonical.data.astype(np.complex64)

    @staticmethod
    def _read_lu_nnz(info: Any) -> int | None:  # ruff: ignore[any-type] -- nvmath info, no stubs
        """Read the L+U nonzero count from the factorization info.

        Returns
        -------
        int or None
            The count, or None when cuDSS does not report it.
        """
        try:
            return int(info.lu_nnz)
        except (AttributeError, TypeError, ValueError):
            return None


__all__ = [
    "CudssSolver",
    "cudss_available",
    "equilibrate",
    "find_threading_layer",
]
