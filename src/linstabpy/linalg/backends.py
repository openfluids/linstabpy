# SPDX-License-Identifier: GPL-3.0-only
"""Sparse-direct factor backend selection and typed registry for PETSc LU paths."""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Mapping
from enum import StrEnum
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

from linstabpy.mumps_options import MUMPS_ORDERING_TO_ICNTL_7
from linstabpy.utils._solver_cfg import ITERATIVE_EIGEN_SOLVER_TYPES

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Type-only import so annotations resolve for type checkers; at runtime
    # ``TYPE_CHECKING`` is False, so importing this module stays PETSc-free.
    from petsc4py import PETSc  # noqa: F401


# Lazy PETSc accessor, mirroring ``linstabpy.utils.petsc_core``. External
# callers reach ``PETSc`` through PEP 562 ``__getattr__``; functions *inside*
# functions here must bind it locally via ``PETSc = _get_petsc()`` because
# ``__getattr__`` does not fire for same-module ``LOAD_GLOBAL`` lookups.
def __getattr__(name):  # noqa: ANN001, ANN201
    if name == "PETSc":
        from linstabpy.utils.petsc_init import init_petsc_from_argv

        init_petsc_from_argv()
        from petsc4py import PETSc as _imp_PETSc

        globals()["PETSc"] = _imp_PETSc
        return _imp_PETSc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_petsc():  # noqa: ANN202
    """Return PETSc, triggering the lazy import on first use."""
    from linstabpy.utils.petsc_init import init_petsc_from_argv

    init_petsc_from_argv()
    from petsc4py import PETSc as _imp_PETSc

    globals()["PETSc"] = _imp_PETSc
    return _imp_PETSc


DEFAULT_FACTOR_BACKEND = "mumps"
DEFAULT_FACTOR_PRECISION = "double"
DEFAULT_MUMPS_BLR = False
DEFAULT_MUMPS_BLR_TOL = 1.0e-4
ERR_SUP = 92
_PROBE_DENOMINATOR_TINY = 1.0e-300
PRECONDITIONER_PROBE_RESIDUAL_REL_MAX = 1.0e-4

# DOF threshold below which MUMPS beats STRUMPACK for GPU shift-invert.
# Measured on c9 (2x NVIDIA A6000, 96 GB GPU RAM): MUMPS wins below ~224k DOF and
# STRUMPACK wins at ~900k DOF (3.2x eigensolve speedup, |dω| = 2.4e-13).
# The true crossover between 224k and 900k has no measurement; this constant is a
# tunable approximation, NOT a universal hardware-independent threshold.
DEFAULT_AUTO_ROUTE_DOF_THRESHOLD = 224_000


class Backend(StrEnum):
    """Known PETSc sparse-direct factor backends."""

    MUMPS = "mumps"
    SUPERLU = "superlu"
    SUPERLU_DIST = "superlu_dist"
    STRUMPACK = "strumpack"
    CUDSS = "cudss"

    @classmethod
    def _missing_(cls, value: object) -> Backend:
        text = str(value).strip().lower()
        backend = str.__new__(cls, text)
        backend._name_ = text.upper()
        backend._value_ = text
        return backend


def resolve_factor_backend(
    cfg: Mapping[str, Any] | None = None,
    *,
    env_var: str | None = None,
    default: str | None = DEFAULT_FACTOR_BACKEND,
) -> str | None:
    """Resolve a direct-factor backend from YAML, env vars, then default."""
    if cfg is not None and "factor_backend" in cfg:
        return _normalize_backend(cfg["factor_backend"])
    if env_var is not None and env_var in os.environ:
        return _normalize_backend(os.environ[env_var])
    if "LINSTABPY_FACTOR_BACKEND" in os.environ:
        return _normalize_backend(os.environ["LINSTABPY_FACTOR_BACKEND"])
    return _normalize_backend(default)


def resolve_factor_precision(
    cfg: Mapping[str, Any] | None = None,
    *,
    env_var: str = "LINSTABPY_FACTOR_PRECISION",
    default: str = DEFAULT_FACTOR_PRECISION,
) -> str:
    """Resolve direct-factor precision with precedence env > YAML > default."""
    if env_var in os.environ:
        return _normalize_factor_precision(os.environ[env_var])
    if cfg is not None and "factor_precision" in cfg:
        return _normalize_factor_precision(cfg["factor_precision"])
    return _normalize_factor_precision(default)


def resolve_mumps_blr(
    cfg: Mapping[str, Any] | None = None,
    *,
    env_var: str = "LINSTABPY_MUMPS_BLR",
    default: bool = DEFAULT_MUMPS_BLR,
) -> bool:
    """Resolve the MUMPS BLR switch with precedence env > YAML > default."""
    if env_var in os.environ:
        return _as_bool(os.environ[env_var])
    if cfg is not None and "mumps_blr" in cfg:
        return _as_bool(cfg["mumps_blr"])
    return bool(default)


def resolve_mumps_blr_tol(
    cfg: Mapping[str, Any] | None = None,
    *,
    env_var: str = "LINSTABPY_MUMPS_BLR_TOL",
    default: float = DEFAULT_MUMPS_BLR_TOL,
) -> float:
    """Resolve the MUMPS BLR tolerance with precedence env > YAML > default."""
    if env_var in os.environ:
        return _normalize_mumps_blr_tol(os.environ[env_var])
    if cfg is not None and "mumps_blr_tol" in cfg:
        return _normalize_mumps_blr_tol(cfg["mumps_blr_tol"])
    return _normalize_mumps_blr_tol(default)


def resolve_mumps_ordering(
    cfg: Mapping[str, Any] | None = None,
    *,
    env_var: str = "LINSTABPY_MUMPS_ORDERING",
    default: str | None = None,
) -> str | None:
    """Resolve MUMPS ordering with precedence env > YAML > default."""
    if env_var in os.environ:
        return _normalize_mumps_ordering(os.environ[env_var])
    if cfg is not None and "mumps_ordering" in cfg:
        return _normalize_mumps_ordering(cfg["mumps_ordering"])
    return _normalize_mumps_ordering(default)


def resolve_auto_factor_backend(
    ndof: int,
    *,
    threshold: int = DEFAULT_AUTO_ROUTE_DOF_THRESHOLD,
    strumpack_available: bool,
) -> str:
    """Resolve the concrete backend for ``factor_backend="auto"`` at solve time.

    Parameters
    ----------
    ndof:
        Number of degrees of freedom of the operator under solve.
    threshold:
        DOF count at or above which STRUMPACK wins when available.
        See ``DEFAULT_AUTO_ROUTE_DOF_THRESHOLD`` for provenance (c9 A6000
        measured lower bound — tunable, not a universal constant).
    strumpack_available:
        Whether STRUMPACK is usable in the current PETSc build (result of
        ``factor_backend_available("strumpack")``).  When False, the function
        always returns ``"mumps"`` regardless of ``ndof`` — ensuring safe
        CPU-only fallback without crashing.

    Returns
    -------
    str
        Either ``"strumpack"`` (iff ndof >= threshold AND strumpack_available)
        or ``"mumps"`` (all other cases, including strumpack unavailable).

    Notes
    -----
    The 224k default threshold is the measured c9 A6000 lower bound where
    MUMPS stops winning.
    The true crossover between 224k and ~900k DOF has no measurement.
    Treat this as a tunable approximation, not a universal constant.
    When STRUMPACK is not available, return ``"mumps"`` silently.
    Never raise or attempt to apply an absent backend.
    """
    if ndof >= threshold and strumpack_available:
        return "strumpack"
    return "mumps"


# STRUMPACK is not thread safe. Its own TODO file says so, and c9 jobs 4525
# and 4526 measured it on this build: with ten host threads, two solves in
# eight gave a factorization wrong by 4.4e-9 and 6.9e-9 on a problem where the
# correct answer is 2.2e-11. One host thread gave 2.209e-11 eight times out of
# eight, to the last bit. STRUMPACK uses the libgomp OpenMP runtime, which
# PETSc links, so one call to that runtime makes the backend repeatable. The
# call has no effect on MUMPS or on the BLAS, which use a different runtime.
_GOMP_LIBRARY = "libgomp.so.1"


@cache
def limit_strumpack_host_threads() -> bool:
    """Set the OpenMP thread count to one, for the STRUMPACK factorization.

    The function runs its effect one time for each process.

    Returns
    -------
    bool
        True if the thread count is one. False if this system has no libgomp,
        which keeps STRUMPACK at the thread count of the environment.
    """
    try:
        gomp = ctypes.CDLL(_GOMP_LIBRARY)
    except OSError as exc:
        logger.warning(
            "STRUMPACK needs one host thread to give a repeatable factorization, "
            "but %s does not load here (%s). Set OMP_NUM_THREADS=1 before the run.",
            _GOMP_LIBRARY,
            exc,
        )
        return False
    set_num_threads = gomp.omp_set_num_threads
    set_num_threads.argtypes = [ctypes.c_int]
    set_num_threads.restype = None
    set_num_threads(1)
    logger.info("STRUMPACK: host OpenMP threads set to 1; the library is not thread safe.")
    return True


def apply_factor_backend(
    pc: PETSc.PC,
    backend: str | None,
    *,
    options_prefix: str = "",
    factor_precision: str | None = None,
    mumps_blr: bool | None = None,
    mumps_blr_tol: float | None = None,
    mumps_ordering: str | None = None,
) -> str | None:
    """Apply ``backend`` to a PCLU after ``pc.setType('lu')``."""
    requested_backend = backend
    backend = _normalize_backend(backend)
    if backend == "auto":
        raise ValueError(
            "factor_backend='auto' must be resolved to a concrete backend before "
            "apply_factor_backend() is called.  Use resolve_auto_factor_backend() "
            "at the solve site where ndof is known."
        )
    blr_enabled = resolve_mumps_blr(None) if mumps_blr is None else bool(mumps_blr)
    ordering = (
        resolve_mumps_ordering(None)
        if mumps_ordering is None
        else _normalize_mumps_ordering(mumps_ordering)
    )
    if blr_enabled and backend != "mumps":
        raise ValueError(
            "LINSTABPY_MUMPS_BLR is only supported with backend 'mumps' "
            f"(got backend {requested_backend!r})"
        )
    if ordering is not None and backend != "mumps":
        raise ValueError(
            "LINSTABPY_MUMPS_ORDERING is only supported with backend 'mumps' "
            f"(got backend {requested_backend!r})"
        )
    if backend is None:
        return None
    precision = (
        resolve_factor_precision(None)
        if factor_precision is None
        else _normalize_factor_precision(factor_precision)
    )
    blr_tol = (
        resolve_mumps_blr_tol(None)
        if mumps_blr_tol is None
        else _normalize_mumps_blr_tol(mumps_blr_tol)
    )
    if precision == "single" and backend != "superlu_dist":
        raise ValueError(
            "LINSTABPY_FACTOR_PRECISION=single is only supported with "
            f"backend 'superlu_dist' (got backend {backend!r}). The working "
            "single-precision route is the GPU backend factor_backend='cudss' "
            "on the iterative path."
        )
    if backend == "cudss":
        # After the knob-specific rejections above, so a BLR/ordering/precision
        # misconfiguration keeps its own actionable message.
        raise ValueError(
            "factor_backend='cudss' is not a PETSc factor type; the cuDSS LU runs "
            "as a preconditioner inside the iterative path. Configure solver type "
            "'iterative' with ksp_type 'fgmres', pc_type 'lu' and "
            "factor_backend 'cudss'."
        )
    if precision == "single":
        # MEASURED 2026-07-30 on PETSc 3.24.2 (c9, job 536): the only lever this
        # code has for single-precision factors is the `-pc_precision` option
        # written below, and PETSc never consumes it -- after pc.setFromOptions()
        # + ksp.setUp() the key is still sitting unread in the options database.
        # The observable consequence (job 535) is that "single" and "double"
        # produce BIT-IDENTICAL results: residual 1.852e-06 either way, and
        # eigenvalues agreeing to all 12 printed digits.
        #
        # So the knob did nothing while reporting success -- the caller asked for
        # a single-precision factor and silently got a double-precision one. That
        # is the failure mode AGENTS.md calls out (a default that silently lies),
        # and it is worse than no feature at all because provenance then records
        # `factor_precision=single` for a run that was double throughout.
        #
        # Refuse instead. When a real mechanism exists (a PETSc release exposing
        # single-precision SuperLU_DIST factors through petsc4py, or a direct
        # SuperLU_DIST option), wire it here and delete this guard -- do NOT
        # delete the guard on its own.
        #
        # NOTE: this makes the `precision == "single"` branches further down
        # (writing and then deleting `pc_precision`) unreachable. They are kept
        # on purpose as the wiring point for that future mechanism, so the shape
        # of the fix is obvious; they are not live code today.
        raise NotImplementedError(
            "factor_precision='single' is not supported: this build drives it "
            "through the PETSc '-pc_precision' option, which PETSc "
            f"{'.'.join(str(v) for v in _get_petsc().Sys.getVersion())} ignores, "
            "so the factor would silently be computed in double precision. "
            "Use factor_precision='double' (the default). See "
            "docs/COMPUTE_BACKENDS.md."
        )
    if backend == "strumpack":
        limit_strumpack_host_threads()
    ensure_factor_backend_available(backend)
    PETSc = _get_petsc()
    opts = PETSc.Options()
    key = f"{options_prefix}pc_factor_mat_solver_type"
    precision_key = f"{options_prefix}pc_precision"
    blr_mode_key = f"{options_prefix}mat_mumps_icntl_35"
    blr_tol_key = f"{options_prefix}mat_mumps_cntl_7"
    ordering_key = f"{options_prefix}mat_mumps_icntl_7"
    opts.setValue(key, backend)
    if precision == "single":
        opts.setValue(precision_key, "single")
    if blr_enabled:
        opts.setValue(blr_mode_key, "1")
        opts.setValue(blr_tol_key, f"{blr_tol:g}")
    elif backend == "mumps":
        # BLR options persist in the process-global options DB so MUMPS can
        # consume them at factorization setup; an off-request must clear any
        # leftovers from an earlier BLR-enabled setup or BLR becomes sticky.
        opts.delValue(blr_mode_key)
        opts.delValue(blr_tol_key)
    if ordering is not None:
        opts.setValue(ordering_key, str(MUMPS_ORDERING_TO_ICNTL_7[ordering]))
    # Unlike the BLR keys (which only linstabpy writes), mat_mumps_icntl_7 has
    # legitimate external writers: user-supplied CLI options. When this knob
    # is unset it expresses no opinion and must NOT clear the key.
    try:
        pc.setFromOptions()
        pc.setFactorSolverType(backend)
    finally:
        if precision == "single":
            opts.delValue(precision_key)
        opts.delValue(key)
    return backend


def ensure_factor_backend_available(backend: str) -> None:
    """Raise a clear error if PETSc cannot set up a tiny LU with ``backend``."""
    if not factor_backend_available(backend):
        PETSc = _get_petsc()
        version = PETSc.Sys.getVersion()
        raise RuntimeError(
            f"PETSc factor backend '{backend}' is not available in this build "
            f"(PETSc {version}). Rebuild PETSc/petsc4py with that sparse-direct "
            "solver enabled, then rerun. See docs/COMPUTE_BACKENDS.md."
        )


FGMRES_DIRECT_PC_EIGEN_SOLVER_TYPES = frozenset({"fgmres_lu", "strumpack_fgmres"})
PRECONDITIONED_EIGEN_SOLVER_TYPES = (
    ITERATIVE_EIGEN_SOLVER_TYPES | FGMRES_DIRECT_PC_EIGEN_SOLVER_TYPES
)


def backend_probe_is_preconditioner_role(st_solver: str | None) -> bool:
    """Return whether the factor backend is only an iterative-outer preconditioner.

    In this role the backend probe's bare direct-solve residual is diagnostic.
    The outer Krylov/SLEPc convergence test, not a standalone PREONLY+LU
    apply, is the primary acceptance gate for the shifted solve.
    A separate relaxed sanity ceiling still catches catastrophically
    ineffective preconditioners.
    """
    return str(st_solver or "").strip().lower() in PRECONDITIONED_EIGEN_SOLVER_TYPES


def should_probe_backend(
    resolved_backend: str | None,
    probe_cfg: Mapping[str, Any] | None = None,
    *,
    st_solver: str | None = None,
) -> bool:
    """Return whether a resolved factor backend should run the accuracy probe.

    Genuine direct solves keep the historical strict direct-residual gate and
    the default-MUMPS exemption.
    Iterative-outer eigensolver families use the factor backend as a
    preconditioner.
    They still run the probe as a diagnostic even for default MUMPS unless
    explicitly disabled.
    """
    backend = _normalize_backend(resolved_backend)
    if backend is None:
        return False
    preconditioner_role = backend_probe_is_preconditioner_role(st_solver)
    if backend == DEFAULT_FACTOR_BACKEND and not preconditioner_role:
        return False
    if probe_cfg is not None and "enabled" in probe_cfg:
        return _as_bool(probe_cfg["enabled"])
    return True


def shifted_ksp_probe_residual(
    ksp: PETSc.KSP,
    *,
    rhs_seed: int = 17,
) -> dict[str, float | int | str | None]:
    """Solve one deterministic shifted-system RHS and return the true residual.

    This is a diagnostic probe for an already-configured shift-invert KSP.
    It does not know about eigenpairs or acceptance policy.
    It only measures
    ``||Kx-b|| / ||b||`` for the KSP's current operator using PETSc collective
    vector norms.
    """
    operators = ksp.getOperators()
    A = operators[0]
    b = A.createVecRight()
    x = A.createVecRight()
    r = A.createVecLeft()
    old_nonzero_guess = _safe_call(ksp.getInitialGuessNonzero)
    try:
        _fill_deterministic_rhs(b, rhs_seed=rhs_seed)
        x.set(0)
        ksp.setInitialGuessNonzero(False)
        ksp.solve(b, x)
        A.mult(x, r)
        r.axpy(-1, b)
        residual_abs = float(r.norm())
        rhs_norm = float(b.norm())
        residual_rel = residual_abs / max(rhs_norm, _PROBE_DENOMINATOR_TINY)
        pc = ksp.getPC()
        return {
            "residual_abs": residual_abs,
            "residual_rel": residual_rel,
            "ksp_iterations": int(ksp.getIterationNumber()),
            "ksp_reason": int(ksp.getConvergedReason()),
            "ksp_type": ksp.getType(),
            "pc_type": pc.getType(),
            "pc_factor_solver_type": _safe_call(pc.getFactorSolverType),
        }
    finally:
        if old_nonzero_guess is not None:
            ksp.setInitialGuessNonzero(bool(old_nonzero_guess))
        r.destroy()
        x.destroy()
        b.destroy()


@cache
def factor_backend_available(backend: str) -> bool:
    """Return whether a tiny complex AIJ LU can be set up with ``backend``."""
    backend = _normalize_backend(backend)
    if backend is None:
        return True
    if backend == "cudss":
        # cuDSS is not a PETSc factor package; availability means the nvmath
        # wheels are installed and a tiny factorization runs on the GPU.
        from .cudss_factor import cudss_available

        return cudss_available()
    if backend == "strumpack":
        limit_strumpack_host_threads()
    PETSc = _get_petsc()
    A = PETSc.Mat().createAIJ(size=(1, 1), comm=PETSc.COMM_SELF)
    A.setValue(0, 0, 1.0 + 0.0j)
    A.assemble()
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    try:
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType(backend)
        ksp.setUp()
        return True
    except PETSc.Error as exc:
        if _petsc_error_code(exc) == ERR_SUP:
            return False
        raise
    finally:
        ksp.destroy()
        A.destroy()


def select(
    cfg: Mapping[str, Any] | None = None,
    *,
    env_var: str | None = None,
    default: str | Backend | None = Backend.MUMPS,
) -> Backend | None:
    """Resolve the configured factor backend as a typed registry entry."""
    resolved = resolve_factor_backend(cfg, env_var=env_var, default=default)
    if resolved is None:
        return None
    return Backend(resolved)


def ensure(backend: str | Backend | None) -> None:
    """Raise if ``backend`` is not available in this PETSc build."""
    if backend is not None:
        ensure_factor_backend_available(str(backend))


def available() -> dict[Backend, bool]:
    """Return availability for known registry backends."""
    return {backend: factor_backend_available(str(backend)) for backend in Backend}


def _normalize_backend(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "default", "petsc"}:
        return None
    # "auto" is a recognized intent token, not a concrete backend — carry it through.
    # It must be resolved to a concrete backend at solve time via
    # resolve_auto_factor_backend() once ndof is known.
    return text


def _normalize_factor_precision(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"double", "single"}:
        return text
    raise ValueError(f"Invalid LINSTABPY_FACTOR_PRECISION={value!r}; expected 'double' or 'single'")


def _normalize_mumps_ordering(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "default", "off"}:
        return None
    if text in MUMPS_ORDERING_TO_ICNTL_7:
        return text
    allowed = ", ".join(sorted(MUMPS_ORDERING_TO_ICNTL_7))
    raise ValueError(f"Invalid LINSTABPY_MUMPS_ORDERING={value!r}; allowed values: {allowed}")


def _normalize_mumps_blr_tol(value: object) -> float:
    try:
        tol = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid LINSTABPY_MUMPS_BLR_TOL={value!r}; expected a positive finite number"
        ) from exc
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(
            f"Invalid LINSTABPY_MUMPS_BLR_TOL={value!r}; expected a positive finite number"
        )
    return tol


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid backend_probe.enabled={value!r}; expected boolean")


def _fill_deterministic_rhs(vec: PETSc.Vec, *, rhs_seed: int) -> None:
    rstart, rend = vec.getOwnershipRange()
    for row in range(rstart, rend):
        real = np.sin((row + 1) * (rhs_seed + 0.5))
        imag = np.cos((row + 1) * (rhs_seed + 1.5))
        vec.setValue(row, complex(real, imag))
    vec.assemble()


def _safe_call(method):  # noqa: ANN001
    PETSc = _get_petsc()
    try:
        return method()
    except (AttributeError, PETSc.Error):
        return None


def _petsc_error_code(exc: PETSc.Error) -> int | None:
    return getattr(exc, "ierr", None) or (exc.args[0] if exc.args else None)


__all__ = [
    "Backend",
    "DEFAULT_FACTOR_BACKEND",
    "DEFAULT_FACTOR_PRECISION",
    "DEFAULT_MUMPS_BLR",
    "DEFAULT_MUMPS_BLR_TOL",
    "DEFAULT_AUTO_ROUTE_DOF_THRESHOLD",
    "PRECONDITIONER_PROBE_RESIDUAL_REL_MAX",
    "MUMPS_ORDERING_TO_ICNTL_7",
    "apply_factor_backend",
    "available",
    "ensure",
    "ensure_factor_backend_available",
    "limit_strumpack_host_threads",
    "backend_probe_is_preconditioner_role",
    "factor_backend_available",
    "resolve_auto_factor_backend",
    "resolve_factor_backend",
    "resolve_factor_precision",
    "resolve_mumps_blr",
    "resolve_mumps_blr_tol",
    "resolve_mumps_ordering",
    "select",
    "shifted_ksp_probe_residual",
    "should_probe_backend",
]
