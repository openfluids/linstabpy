"""LNSE operator assembly from coefficients and FD matrices.

Translates GetLinProblem.m: assembles the 5*ngp × 5*ngp linear operator
L0 such that dq/dt = L0 @ q, with eig(i*L0) = omega_code = -omega_OS.
Unstable modes have Im(omega_code) < 0.  State: q = [rho, u, v, w, T].
"""

from __future__ import annotations

import json
import logging
import os
import resource
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import scipy.sparse as sp

from ..utils._solver_cfg import _scfg, _scfg_bool
from ..utils.petsc_core import _matrix_nnz
from ..mesh.mesh import Mesh
from ..baseflow.baseflow import BaseFlow
from .coefficients import (
    compute_cartesian_coefficients,
    compute_cylindrical_coefficients,
    LNSECoefficients,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only optional dependency
    from petsc4py import PETSc


logger = logging.getLogger(__name__)

_CURRENT_ASSEMBLY_METRICS: ContextVar["_AssemblyMetricsCollector | None"] = ContextVar(
    "_CURRENT_ASSEMBLY_METRICS",
    default=None,
)
_COEFF_TENSOR_NAMES = ("A0", "Ax", "Ay", "Axx", "Ayy", "Axy", "B")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _jsonify_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify_metric_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_metric_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _raw_cfg_value(cfg: dict | None, key: str, env_var: str) -> str | None:
    if cfg is not None and key in cfg:
        value = cfg[key]
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    raw = os.getenv(env_var)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _detect_rank_size() -> tuple[int, int]:
    try:
        from petsc4py import PETSc

        comm = PETSc.COMM_WORLD
        return int(comm.getRank()), int(comm.getSize())
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("PETSc rank/size probe failed; falling back to (0, 1): %s", exc)
        return 0, 1


@dataclass(frozen=True)
class _AssemblyMetricsConfig:
    enabled: bool
    output: str | None = None
    directory: str | None = None
    tag: str | None = None
    log: bool = False


@dataclass
class _AssemblyMetricsCollector:
    config: _AssemblyMetricsConfig
    metadata: dict[str, Any]
    rank: int
    size: int
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    started_at_utc: str = field(default_factory=_now_utc)
    start_time: float = field(default_factory=time.perf_counter)

    def record_stage(self, name: str, payload: dict[str, Any]) -> None:
        self.stages[name] = _jsonify_metric_value(payload)

    def update_metadata(self, *, overwrite: bool = True, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if value is None:
                continue
            if overwrite or key not in self.metadata:
                self.metadata[key] = _jsonify_metric_value(value)

    def _output_path(self) -> Path | None:
        if self.config.output:
            template = self.config.output
            try:
                rendered = template.format(rank=self.rank, size=self.size)
            except KeyError as exc:  # pragma: no cover - config guard
                raise ValueError(
                    f"Invalid assembly_metrics_output placeholder {exc!s}; "
                    "supported placeholders are {rank} and {size}"
                ) from exc
            path = Path(rendered)
            if path.suffix == "":
                path = path.with_suffix(".json")
            if self.size > 1 and "{rank" not in template and "{size" not in template:
                path = path.with_name(f"{path.stem}_rank{self.rank:04d}{path.suffix}")
            return path
        if not self.config.directory:
            return None
        stem = "assembly_metrics"
        if self.config.tag:
            stem = f"{stem}_{self.config.tag}"
        filename = f"{stem}_rank{self.rank:04d}.json"
        return Path(self.config.directory) / filename

    def _log_summary(self, payload: dict[str, Any], output_path: Path | None) -> None:
        if not self.config.log:
            return
        logger.info(
            "Assembly metrics: backend=%s resolved=%s ndof=%s nnz=%s wall=%.3fs path=%s",
            payload.get("requested_backend", "?"),
            payload.get("resolved_backend", "?"),
            payload.get("ndof", "?"),
            payload.get("nnz", "?"),
            payload.get("wall_seconds", float("nan")),
            output_path if output_path is not None else "<log-only>",
        )
        for name, stage in payload.get("stages", {}).items():
            logger.info(
                "  %s: %.3fs peak_rss_delta=%.1f MiB%s",
                name,
                float(stage.get("seconds", 0.0)),
                float(stage.get("rss_peak_delta_bytes", 0)) / (1024.0 * 1024.0),
                " [skipped]" if stage.get("skipped") else "",
            )

    def finish(self) -> None:
        payload = {
            **self.metadata,
            "rank": self.rank,
            "size": self.size,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": _now_utc(),
            "wall_seconds": time.perf_counter() - self.start_time,
            "stages": self.stages,
        }
        output_path = self._output_path()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
        self._log_summary(payload, output_path)


def _resolve_assembly_metrics_config(solver_cfg: dict | None) -> _AssemblyMetricsConfig:
    output = _raw_cfg_value(
        solver_cfg,
        "assembly_metrics_output",
        "LINSTAB2D_ASSEMBLY_METRICS_OUTPUT",
    )
    directory = _raw_cfg_value(
        solver_cfg,
        "assembly_metrics_dir",
        "LINSTAB2D_ASSEMBLY_METRICS_DIR",
    )
    tag = _raw_cfg_value(
        solver_cfg,
        "assembly_metrics_tag",
        "LINSTAB2D_ASSEMBLY_METRICS_TAG",
    )
    enabled = (
        _scfg_bool(solver_cfg, "assembly_metrics", "LINSTAB2D_ASSEMBLY_METRICS", False)
        or output is not None
        or directory is not None
    )
    log_enabled = _scfg_bool(
        solver_cfg,
        "assembly_metrics_log",
        "LINSTAB2D_ASSEMBLY_METRICS_LOG",
        enabled,
    )
    return _AssemblyMetricsConfig(
        enabled=enabled,
        output=output,
        directory=directory,
        tag=tag,
        log=log_enabled,
    )


def _current_assembly_metrics() -> _AssemblyMetricsCollector | None:
    return _CURRENT_ASSEMBLY_METRICS.get()


def _note_assembly_metadata(*, overwrite: bool = True, **kwargs: Any) -> None:
    collector = _current_assembly_metrics()
    if collector is None:
        return
    collector.update_metadata(overwrite=overwrite, **kwargs)


@contextmanager
def _assembly_metrics_session(
    *,
    solver_cfg: dict | None,
    **metadata: Any,
) -> Iterator[_AssemblyMetricsCollector | None]:
    existing = _current_assembly_metrics()
    if existing is not None:
        existing.update_metadata(overwrite=False, **metadata)
        yield existing
        return

    config = _resolve_assembly_metrics_config(solver_cfg)
    if not config.enabled:
        yield None
        return

    rank, size = _detect_rank_size()
    collector = _AssemblyMetricsCollector(
        config=config,
        metadata={},
        rank=rank,
        size=size,
    )
    collector.update_metadata(
        rank=rank,
        size=size,
        status="running",
        **metadata,
    )
    token = _CURRENT_ASSEMBLY_METRICS.set(collector)
    try:
        yield collector
    except Exception as exc:
        collector.update_metadata(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    else:
        collector.update_metadata(status="ok")
    finally:
        _CURRENT_ASSEMBLY_METRICS.reset(token)
        collector.finish()


@contextmanager
def _assembly_stage(name: str) -> Iterator[dict[str, Any]]:
    stage: dict[str, Any] = {}
    collector = _current_assembly_metrics()
    rss0 = _peak_rss_bytes()
    t0 = time.perf_counter()
    try:
        yield stage
    except Exception as exc:
        stage["status"] = "error"
        stage["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if collector is not None:
            rss1 = _peak_rss_bytes()
            stage.setdefault("seconds", time.perf_counter() - t0)
            stage.setdefault("rss_peak_before_bytes", rss0)
            stage.setdefault("rss_peak_after_bytes", rss1)
            stage.setdefault("rss_peak_delta_bytes", max(0, rss1 - rss0))
            collector.record_stage(name, stage)


@dataclass(frozen=True)
class _CSRDerivativeCache:
    csr: sp.csr_matrix
    row_nnz: np.ndarray
    row_indices: np.ndarray


def _make_csr_derivative_cache(op: sp.spmatrix) -> _CSRDerivativeCache:
    """Convert a derivative operator into a canonical CSR cache."""
    if not sp.issparse(op):
        raise TypeError("Derivative operator must be a sparse matrix")
    csr = op if sp.isspmatrix_csr(op) else op.tocsr()
    if not getattr(csr, "has_canonical_format", False):
        csr = csr.copy()
        csr.sum_duplicates()
        csr.sort_indices()
    if not np.issubdtype(csr.dtype, np.complexfloating):
        csr = csr.astype(complex)
    row_nnz = np.diff(csr.indptr).astype(np.int64, copy=False)
    row_indices = np.repeat(np.arange(csr.shape[0], dtype=np.int64), row_nnz)
    return _CSRDerivativeCache(csr=csr, row_nnz=row_nnz, row_indices=row_indices)


def _as_csr_derivative_cache(op: _CSRDerivativeCache | sp.spmatrix) -> _CSRDerivativeCache:
    if isinstance(op, _CSRDerivativeCache):
        return op
    return _make_csr_derivative_cache(op)


def _cache_5_block_derivatives(
    op: list[sp.spmatrix] | tuple[sp.spmatrix, ...] | sp.spmatrix,
    ngp: int,
) -> list[_CSRDerivativeCache]:
    if isinstance(op, (list, tuple)):
        if len(op) != 5:
            raise ValueError("Expected five derivative blocks for axy mode")
        return [_make_csr_derivative_cache(mat) for mat in op]

    if not sp.issparse(op):
        raise TypeError("Derivative operator must be a sparse matrix or 5-block list/tuple")

    cache = _make_csr_derivative_cache(op)
    if cache.csr.shape != (ngp, ngp):
        raise ValueError(f"Expected derivative operator with shape {(ngp, ngp)}, got {cache.csr.shape}")
    return [cache] * 5


def build_lnse_operator(
    mesh: Mesh,
    baseflow: BaseFlow,
    kx: float = 0.0,
    model: str = "2D",
    m: float = 0,
    floquet_exp: float = 0.0,
) -> tuple[sp.csc_matrix, dict]:
    """Build the LNSE operator L0 as a scipy sparse matrix.

    Assembles L0 = inv(B/1i) @ (A0 + Ay*Dr + Ax*Dz + Ayy*D2r + Axx*D2z + Axy*D2rz)
    with coordinate remapping and sign conventions already absorbed into the
    coefficients, then subtracts the sponge function.

    Supported models:
    - ``2D``: cartesian BL form with Fourier spanwise wavenumber ``kx``.
    - ``axy``: cylindrical axisymmetric form with azimuthal mode ``m`` and
      optional Floquet shift.

    Parameters
    ----------
    mesh : Mesh
        Mesh object with FD matrices and integration weights.
    baseflow : BaseFlow
        Base flow with velocity, thermodynamic, and viscosity fields.
    kx : float
        Streamwise Fourier wavenumber (physical x-direction), used for
        the Cartesian model only.
    model : str
        Model selector ``"2D"`` or ``"axy"``.
    m : float
        Azimuthal Fourier mode for cylindrical model ``"axy"``.
    floquet_exp : float
        Floquet exponent used only by the cylindrical model.

    Returns
    -------
    L0 : csc_matrix, shape (5*ngp, 5*ngp)
        The assembled LNSE operator.
    idx : dict
        Index mapping for variables and boundaries. Keys include:
        - 'bi', 'ti', 'li', 'ri': boundary grid point indices (0-based)
        - 'rho_j', 'u_j', 'v_j', 'w_j', 'T_j': DOF index arrays for each variable
        - '{border}_{var}': DOF indices for each variable at each border
    """
    model_name = str(model).lower()
    with _assembly_metrics_session(
        solver_cfg=None,
        path_kind="operator",
        requested_backend="scipy",
        resolved_backend="scipy",
        distributed_requested=False,
        direct_petsc_supported=False,
        model=model_name,
        kx=float(kx),
        m=float(m),
        floquet_exp=float(floquet_exp),
    ):
        prep = _prepare_coefficients(mesh, baseflow, kx=kx, model=model, m=m)
        L0 = _assemble_from_prepared(prep, floquet_exp=floquet_exp)
        idx = _build_index_structure(mesh.idx, mesh.ngp)
        _note_assembly_metadata(
            nnz_pre_bc=_matrix_nnz(L0),
            matrix_type="scipy",
        )
        return L0, idx


def build_lnse_operator_petsc(
    mesh: Mesh,
    baseflow: BaseFlow,
    kx: float = 0.0,
    model: str = "2D",
    m: float = 0,
    floquet_exp: float = 0.0,
    *,
    solver_cfg: dict | None = None,
) -> tuple["PETSc.Mat", dict]:
    """Build the pre-BC LNSE operator L0 directly as a PETSc AIJ matrix.

    This prototype preserves the existing coefficient-preparation path but avoids
    the final SciPy-assembled global matrix. It currently targets the same
    pre-BC operator returned by :func:`build_lnse_operator`.
    """
    from ..utils.petsc_core import require_complex

    model_name = str(model).lower()
    with _assembly_metrics_session(
        solver_cfg=solver_cfg,
        path_kind="operator",
        requested_backend="petsc",
        resolved_backend="petsc",
        distributed_requested=None,
        direct_petsc_supported=True,
        model=model_name,
        kx=float(kx),
        m=float(m),
        floquet_exp=float(floquet_exp),
    ):
        _validate_direct_petsc_backend(model=model, solver_cfg=solver_cfg)
        require_complex(solver_cfg)
        prep = _prepare_coefficients(mesh, baseflow, kx=kx, model=model, m=m)
        L0 = _assemble_from_prepared_petsc(prep, floquet_exp=floquet_exp, solver_cfg=solver_cfg)
        idx = _build_index_structure(mesh.idx, mesh.ngp)
        _note_assembly_metadata(
            nnz_pre_bc=_matrix_nnz(L0),
            matrix_type="petsc",
        )
        return L0, idx


def _validate_direct_petsc_backend(
    *,
    model: str,
    solver_cfg: dict | None = None,
) -> None:
    """Reject direct-PETSc combinations that are not yet validated."""
    model_name = str(model).lower()
    if model_name not in {"2d", "axy"}:
        raise NotImplementedError(
            "Direct PETSc LNSE assembly is currently supported only for model='2D' or model='axy'"
        )

    from ..utils.petsc_core import _use_distributed_matrix

    if _use_distributed_matrix(solver_cfg) and model_name != "2d":
        raise NotImplementedError(
            "Direct distributed PETSc LNSE assembly is currently supported only for model='2D'; "
            "model='axy' still requires local/sequential PETSc matrices"
        )


def _direct_petsc_backend_supported(
    *,
    model: str,
    solver_cfg: dict | None = None,
) -> bool:
    """Return whether direct PETSc LNSE assembly is validated for this case."""
    try:
        _validate_direct_petsc_backend(model=model, solver_cfg=solver_cfg)
    except NotImplementedError:
        return False
    return True


def _requested_lnse_backend(
    backend: str | None,
    *,
    solver_cfg: dict | None = None,
) -> str:
    """Return the normalized backend request string for LNSE workflow helpers."""
    raw = (
        _scfg(solver_cfg, "assembly_backend", "LINSTAB2D_ASSEMBLY_BACKEND", "scipy")
        if backend is None
        else backend
    )
    normalized = str(raw).strip().lower()
    if normalized not in {"scipy", "petsc", "auto"}:
        raise ValueError(
            f"Invalid assembly backend {normalized!r}; expected one of scipy|petsc|auto"
        )
    return normalized


def _resolve_lnse_backend(
    backend: str | None,
    *,
    model: str,
    solver_cfg: dict | None = None,
) -> str:
    """Resolve LNSE assembly backend selection for workflow helpers."""
    raw = _requested_lnse_backend(backend, solver_cfg=solver_cfg)
    if raw == "petsc":
        return "petsc"
    if raw == "scipy":
        return "scipy"

    if _direct_petsc_backend_supported(model=model, solver_cfg=solver_cfg):
        return "petsc"
    return "scipy"


def _build_lnse_system_scipy(
    mesh: Mesh,
    baseflow: BaseFlow,
    *,
    kx: float = 0.0,
    model: str = "2D",
    m: float = 0,
    floquet_exp: float = 0.0,
    bc_recipe: list[dict] | None = None,
    bc_penalty: float = 1e4,
) -> tuple[sp.csc_matrix, dict, np.ndarray]:
    """Build the LNSE system through the canonical SciPy assembly/BC path."""
    from ..boundary_conditions.recipe import apply_bc_recipe

    L0, idx = build_lnse_operator(
        mesh,
        baseflow,
        kx=kx,
        model=model,
        m=m,
        floquet_exp=floquet_exp,
    )
    with _assembly_stage("apply_bc") as stage:
        if bc_recipe:
            L0, bc_indices = apply_bc_recipe(
                L0,
                mesh,
                idx,
                bc_recipe,
                penalty=bc_penalty,
            )
        else:
            stage["skipped"] = True
            bc_indices = np.array([], dtype=np.int64)
        stage["bc_indices_count"] = int(bc_indices.size)
        stage["nnz"] = _matrix_nnz(L0)
    return L0, idx, bc_indices


def build_lnse_system(
    mesh: Mesh,
    baseflow: BaseFlow,
    *,
    kx: float = 0.0,
    model: str = "2D",
    m: float = 0,
    floquet_exp: float = 0.0,
    bc_recipe: list[dict] | None = None,
    bc_penalty: float = 1e4,
    backend: str | None = None,
    solver_cfg: dict | None = None,
) -> tuple[sp.csc_matrix | "PETSc.Mat", dict, np.ndarray]:
    """Build an LNSE operator and optionally apply BCs with one backend selector.

    This workflow helper keeps the existing SciPy path as the default while
    allowing direct-PETSc assembly/BC application when ``backend='petsc'``.
    In distributed PETSc mode, unsupported direct-assembly cases use the
    validated SciPy assembly + SciPy->PETSc conversion fallback. That fallback
    returns PETSc matrices, but it is not true distributed direct assembly.
    """
    from ..utils.petsc_core import _use_distributed_matrix, compact_petsc_matrix, scipy_to_petsc

    requested_backend = _requested_lnse_backend(backend, solver_cfg=solver_cfg)
    resolved_backend = _resolve_lnse_backend(backend, model=model, solver_cfg=solver_cfg)
    distributed_requested = _use_distributed_matrix(solver_cfg)
    direct_supported = _direct_petsc_backend_supported(model=model, solver_cfg=solver_cfg)
    use_petsc_fallback = (
        requested_backend in {"petsc", "auto"}
        and distributed_requested
        and resolved_backend == "scipy"
    )

    with _assembly_metrics_session(
        solver_cfg=solver_cfg,
        path_kind="system",
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        distributed_requested=distributed_requested,
        direct_petsc_supported=direct_supported,
        use_petsc_fallback=use_petsc_fallback,
        model=str(model).lower(),
        kx=float(kx),
        m=float(m),
        floquet_exp=float(floquet_exp),
        bc_recipe_enabled=bool(bc_recipe),
    ):
        if use_petsc_fallback:
            L0_sp, idx, bc_indices = _build_lnse_system_scipy(
                mesh,
                baseflow,
                kx=kx,
                model=model,
                m=m,
                floquet_exp=floquet_exp,
                bc_recipe=bc_recipe,
                bc_penalty=bc_penalty,
            )
            nnz_post_bc = _matrix_nnz(L0_sp)
            with _assembly_stage("scipy_to_petsc") as stage:
                L0 = scipy_to_petsc(L0_sp, solver_cfg)
                stage["nnz"] = _matrix_nnz(L0)
            with _assembly_stage("petsc_compaction") as stage:
                stage["skipped"] = True
            _note_assembly_metadata(
                nnz_post_bc=nnz_post_bc,
                nnz_final=_matrix_nnz(L0),
                nnz=_matrix_nnz(L0),
                bc_indices_count=int(bc_indices.size),
                matrix_type="petsc",
            )
            return L0, idx, bc_indices

        if resolved_backend == "petsc":
            from ..boundary_conditions.recipe import apply_bc_recipe_petsc

            L0, idx = build_lnse_operator_petsc(
                mesh,
                baseflow,
                kx=kx,
                model=model,
                m=m,
                floquet_exp=floquet_exp,
                solver_cfg=solver_cfg,
            )
            with _assembly_stage("apply_bc") as stage:
                if bc_recipe:
                    L0, bc_indices = apply_bc_recipe_petsc(
                        L0,
                        mesh,
                        idx,
                        bc_recipe,
                        penalty=bc_penalty,
                    )
                else:
                    stage["skipped"] = True
                    bc_indices = np.array([], dtype=np.int64)
                stage["bc_indices_count"] = int(bc_indices.size)
                stage["nnz"] = _matrix_nnz(L0)
            nnz_post_bc = _matrix_nnz(L0)
            with _assembly_stage("scipy_to_petsc") as stage:
                stage["skipped"] = True
            with _assembly_stage("petsc_compaction") as stage:
                if not distributed_requested and direct_supported:
                    L0 = compact_petsc_matrix(L0, solver_cfg)
                    stage["nnz"] = _matrix_nnz(L0)
                else:
                    stage["skipped"] = True
            _note_assembly_metadata(
                nnz_post_bc=nnz_post_bc,
                nnz_final=_matrix_nnz(L0),
                nnz=_matrix_nnz(L0),
                bc_indices_count=int(bc_indices.size),
                matrix_type="petsc",
            )
            return L0, idx, bc_indices

        with _assembly_stage("scipy_to_petsc") as stage:
            stage["skipped"] = True
        with _assembly_stage("petsc_compaction") as stage:
            stage["skipped"] = True
        L0, idx, bc_indices = _build_lnse_system_scipy(
            mesh,
            baseflow,
            kx=kx,
            model=model,
            m=m,
            floquet_exp=floquet_exp,
            bc_recipe=bc_recipe,
            bc_penalty=bc_penalty,
        )
        _note_assembly_metadata(
            nnz_post_bc=_matrix_nnz(L0),
            nnz_final=_matrix_nnz(L0),
            nnz=_matrix_nnz(L0),
            bc_indices_count=int(bc_indices.size),
            matrix_type="scipy",
        )
        return L0, idx, bc_indices


def _prepare_coefficients(
    mesh: Mesh,
    baseflow: BaseFlow,
    *,
    kx: float = 0.0,
    model: str = "2D",
    m: float = 0,
) -> dict:
    """Compute LNSE coefficients and base derivative matrices (once).

    Returns a dict with everything needed by ``_assemble_from_prepared``,
    separated from the Floquet shift so the spatial EVP can reuse it.
    """
    with _assembly_stage("prepare_coefficients") as stage:
        ngp = mesh.ngp
        p = mesh.used_ind

        U = baseflow.U.ravel(order="F")[p]
        V = baseflow.V.ravel(order="F")[p]
        W = baseflow.W.ravel(order="F")[p]
        RHO = baseflow.RHO.ravel(order="F")[p]
        T = baseflow.T.ravel(order="F")[p]
        MU = baseflow.MU.ravel(order="F")[p]
        dmudT = baseflow.dmudT.ravel(order="F")[p]
        d2mudT2 = baseflow.d2mudT2.ravel(order="F")[p]

        kappa = baseflow.kappa
        Ma = baseflow.Ma
        Re = baseflow.Re
        Pr = baseflow.Pr

        P = RHO * T / (Ma**2 * kappa)

        Dx = mesh.Dx
        D2x = mesh.D2x
        Dy = mesh.Dy
        D2y = mesh.D2y
        # MATLAB 2D path uses Dzr = Dz*Dr (GetLinProblem.m: D2RZ=blkdiag(Dzr)).
        # In this codebase that is Dyx = Dy @ Dx, not Dxy.
        D2RZ = mesh.Dyx

        model = model.lower()
        if model == "2d":
            dUdx = Dx @ U;    dUdy = Dy @ U
            dVdx = Dx @ V;    dVdy = Dy @ V
            dWdx = Dx @ W;    dWdy = Dy @ W
            dRHOdx = Dx @ RHO;  dRHOdy = Dy @ RHO
            dTdx = Dx @ T;    dTdy = Dy @ T
            dPdx = Dx @ P;    dPdy = Dy @ P

            d2Udx2 = D2x @ U;  d2Udy2 = D2y @ U
            d2Vdx2 = D2x @ V;  d2Vdy2 = D2y @ V
            d2Wdx2 = D2x @ W;  d2Wdy2 = D2y @ W
            d2Tdx2 = D2x @ T;  d2Tdy2 = D2y @ T

            d2Udxdy = Dx @ dUdy
            d2Vdxdy = Dx @ dVdy
            d2Wdxdy = Dx @ dWdy

            coeffs = compute_cartesian_coefficients(
                U=U, V=V, W=W, RHO=RHO, T=T, MU=MU, P=P,
                kx=kx, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa,
                dUdx=dUdx, dVdx=dVdx, dWdx=dWdx,
                dRHOdx=dRHOdx, dTdx=dTdx, dPdx=dPdx,
                dUdy=dUdy, dVdy=dVdy, dWdy=dWdy,
                dRHOdy=dRHOdy, dTdy=dTdy, dPdy=dPdy,
                d2Udx2=d2Udx2, d2Vdx2=d2Vdx2, d2Wdx2=d2Wdx2,
                d2Udy2=d2Udy2, d2Vdy2=d2Vdy2, d2Wdy2=d2Wdy2,
                d2Tdx2=d2Tdx2, d2Tdy2=d2Tdy2,
                d2Udxdy=d2Udxdy, d2Vdxdy=d2Vdxdy, d2Wdxdy=d2Wdxdy,
                dmudT=dmudT, d2mudT2=d2mudT2,
            )

            DR = Dy
            D2R = D2y
            DZ = Dx
            D2Z = D2x

        elif model == "axy":
            R = mesh.Y.ravel(order="F")[p]
            U_c = baseflow.V.ravel(order="F")[p]
            V_c = baseflow.W.ravel(order="F")[p]
            W_c = baseflow.U.ravel(order="F")[p]

            dUdr = Dy @ U_c;    dVdr = Dy @ V_c;    dWdr = Dy @ W_c
            dRHOdr = Dy @ RHO;  dTdr = Dy @ T
            dUdz = Dx @ U_c;    dVdz = Dx @ V_c;    dWdz = Dx @ W_c
            dRHOdz = Dx @ RHO;  dTdz = Dx @ T

            d2Udr2 = D2y @ U_c;  d2Vdr2 = D2y @ V_c;  d2Wdr2 = D2y @ W_c
            d2Tdr2 = D2y @ T
            d2Udz2 = D2x @ U_c;  d2Vdz2 = D2x @ V_c;  d2Wdz2 = D2x @ W_c
            d2Tdz2 = D2x @ T

            d2Udrz = Dx @ dUdr;  d2Vdrz = Dx @ dVdr;  d2Wdrz = Dx @ dWdr

            coeffs = compute_cylindrical_coefficients(
                U=U_c, V=V_c, W=W_c, R=R, RHO=RHO, T=T, MU=MU, m=m, Re=Re, Ma=Ma,
                Pr=Pr, kappa=kappa,
                dUdr=dUdr, dVdr=dVdr, dWdr=dWdr, dRHOdr=dRHOdr, dTdr=dTdr,
                dUdz=dUdz, dVdz=dVdz, dWdz=dWdz, dRHOdz=dRHOdz, dTdz=dTdz,
                d2Udr2=d2Udr2, d2Vdr2=d2Vdr2, d2Wdr2=d2Wdr2, d2Tdr2=d2Tdr2,
                d2Udz2=d2Udz2, d2Vdz2=d2Vdz2, d2Wdz2=d2Wdz2, d2Tdz2=d2Tdz2,
                d2Udrz=d2Udrz, d2Vdrz=d2Vdrz, d2Wdrz=d2Wdrz,
                dmudT=dmudT, d2mudT2=d2mudT2,
            )
            DR, D2R, DZ, D2Z, D2RZ = _create_axy_diff_matrices(mesh, m)

        else:
            raise ValueError("model must be '2D' or 'axy'")

        coeff_tensor_bytes = {
            name: int(getattr(coeffs, name).nbytes)
            for name in _COEFF_TENSOR_NAMES
        }
        coeff_tensor_bytes_total = int(sum(coeff_tensor_bytes.values()))
        stage["ngp"] = int(ngp)
        stage["ndof"] = int(5 * ngp)
        stage["coeff_tensor_bytes_by_array"] = coeff_tensor_bytes
        stage["coeff_tensor_bytes_total"] = coeff_tensor_bytes_total
        _note_assembly_metadata(
            ngp=int(ngp),
            ndof=int(5 * ngp),
            coeff_tensor_bytes_by_array=coeff_tensor_bytes,
            coeff_tensor_bytes_total=coeff_tensor_bytes_total,
        )

        return {
            "ngp": ngp, "model": model, "coeffs": coeffs,
            "DR": DR, "D2R": D2R, "DZ": DZ, "D2Z": D2Z, "D2RZ": D2RZ,
            "RHO": RHO, "U": U, "V": V, "W": W, "T": T,
            "kappa": kappa, "Ma": Ma, "mesh": mesh,
        }


def _assemble_from_prepared(prep: dict, *, floquet_exp: float = 0.0) -> sp.csc_matrix:
    """Assemble the LNSE operator from precomputed coefficients and derivatives.

    Applies the Floquet shift, block assembly, inv(B), and sponge.
    """
    ngp = prep["ngp"]
    model = prep["model"]
    coeffs = prep["coeffs"]
    DR = prep["DR"]
    D2R = prep["D2R"]
    DZ = prep["DZ"]
    D2Z = prep["D2Z"]
    D2RZ = prep["D2RZ"]
    mesh = prep["mesh"]

    # Floquet shift: Dx → Dx + iα, D2x → D2x + 2iα Dx + (iα)²
    # Without this (floquet_exp=0), no streamwise wavenumber is imposed.
    if floquet_exp != 0.0:
        i_alpha = 1j * floquet_exp
        DZ, D2Z = _apply_floquet_shift(DZ, D2Z, i_alpha, ngp)

    with _assembly_stage("assemble_raw_blocks") as stage:
        L0 = _assemble_block_operator(ngp, coeffs, DR, D2R, DZ, D2Z, D2RZ)
        stage["nnz"] = _matrix_nnz(L0)

    with _assembly_stage("apply_invB") as stage:
        L0 = _apply_inv_B(
            L0, model, ngp,
            prep["RHO"], prep["U"], prep["V"], prep["W"], prep["T"],
            prep["kappa"], prep["Ma"],
        )
        stage["nnz"] = _matrix_nnz(L0)

    with _assembly_stage("apply_sponge") as stage:
        if mesh.sponge is not None:
            p = mesh.used_ind
            sponge = mesh.sponge.ravel(order="F")[p]
            sponge_diag = np.tile(sponge, 5)
            L0 = L0 - sp.diags(sponge_diag)
        else:
            stage["skipped"] = True
        stage["nnz"] = _matrix_nnz(L0)

    return L0.tocsc()


def _assemble_from_prepared_petsc(
    prep: dict,
    *,
    floquet_exp: float = 0.0,
    solver_cfg: dict | None = None,
) -> "PETSc.Mat":
    """Assemble the LNSE operator from precomputed data directly into PETSc."""
    from petsc4py import PETSc
    from ..utils.petsc_core import _use_distributed_matrix, assemble_matrix, ownership_range

    ngp = prep["ngp"]
    model = prep["model"]
    coeffs = prep["coeffs"]
    DR = prep["DR"]
    D2R = prep["D2R"]
    DZ = prep["DZ"]
    D2Z = prep["D2Z"]
    D2RZ = prep["D2RZ"]
    mesh = prep["mesh"]

    if floquet_exp != 0.0:
        i_alpha = 1j * floquet_exp
        DZ, D2Z = _apply_floquet_shift(DZ, D2Z, i_alpha, ngp)

    with _assembly_stage("assemble_raw_blocks") as stage:
        L0_petsc = _assemble_block_operator_petsc(
            ngp,
            coeffs,
            DR,
            D2R,
            DZ,
            D2Z,
            D2RZ,
            solver_cfg=solver_cfg,
        )
        stage["nnz"] = _matrix_nnz(L0_petsc)
    with _assembly_stage("apply_invB") as stage:
        _apply_inv_B_petsc(
            L0_petsc,
            model,
            ngp,
            prep["RHO"], prep["U"], prep["V"], prep["W"], prep["T"],
            prep["kappa"], prep["Ma"],
        )
        stage["nnz"] = _matrix_nnz(L0_petsc)

    with _assembly_stage("apply_sponge") as stage:
        if mesh.sponge is not None:
            p = mesh.used_ind
            sponge = np.asarray(mesh.sponge.ravel(order="F")[p], dtype=complex)
            sponge_diag = np.tile(sponge, 5)
            if _use_distributed_matrix(solver_cfg):
                r0, r1 = ownership_range(L0_petsc)
                local_diag = L0_petsc.createVecLeft()
                try:
                    local_diag.set(0.0)
                    local_values = -sponge_diag[r0:r1]
                    if local_values.size:
                        local_rows = np.arange(r0, r1, dtype=PETSc.IntType)
                        local_diag.setValues(local_rows, np.asarray(local_values, dtype=complex))
                    local_diag.assemble()
                    L0_petsc.setDiagonal(local_diag, addv=PETSc.InsertMode.ADD_VALUES)
                finally:
                    local_diag.destroy()
            else:
                diag = L0_petsc.getDiagonal()
                try:
                    diag_arr = diag.getArray()
                    diag_arr -= sponge_diag
                    L0_petsc.setDiagonal(diag, addv=PETSc.InsertMode.INSERT_VALUES)
                finally:
                    diag.destroy()
        else:
            stage["skipped"] = True
        stage["nnz"] = _matrix_nnz(L0_petsc)

    return assemble_matrix(L0_petsc)


def _assemble_block_operator(
    ngp: int,
    coeffs: LNSECoefficients,
    DR: sp.spmatrix,
    D2R: sp.spmatrix,
    DZ: sp.spmatrix,
    D2Z: sp.spmatrix,
    D2RZ: sp.spmatrix,
) -> sp.csc_matrix:
    """Assemble the block operator from coefficient arrays and FD matrices.

    For each (i,j) variable pair, the (i,j) block of size ngp×ngp is:
        A0[i,j,:]*I + Ay[i,j,:]*DR + Ax[i,j,:]*DZ
        + Ayy[i,j,:]*D2R + Axx[i,j,:]*D2Z + Axy[i,j,:]*D2RZ
    where the coefficient vectors multiply the rows (left-multiply by diag).
    """
    blocks = [[None] * 5 for _ in range(5)]
    DR_blocks = _cache_5_block_derivatives(DR, ngp)
    D2R_blocks = _cache_5_block_derivatives(D2R, ngp)
    DZ_blocks = _cache_5_block_derivatives(DZ, ngp)
    D2Z_blocks = _cache_5_block_derivatives(D2Z, ngp)
    D2RZ_blocks = _cache_5_block_derivatives(D2RZ, ngp)

    for i in range(5):
        for j in range(5):
            blocks[i][j] = _build_block_csr(
                ngp,
                coeffs.A0[i, j, :],
                coeffs.Ax[i, j, :],
                coeffs.Ay[i, j, :],
                coeffs.Axx[i, j, :],
                coeffs.Ayy[i, j, :],
                coeffs.Axy[i, j, :],
                DR_blocks[j],
                D2R_blocks[j],
                DZ_blocks[j],
                D2Z_blocks[j],
                D2RZ_blocks[j],
            )

    return sp.bmat(blocks, format="csc")


def _assemble_block_operator_csr(
    ngp: int,
    coeffs: LNSECoefficients,
    DR: sp.spmatrix,
    D2R: sp.spmatrix,
    DZ: sp.spmatrix,
    D2Z: sp.spmatrix,
    D2RZ: sp.spmatrix,
) -> sp.csr_matrix:
    """Assemble the LNSE raw block operator directly into a global CSR matrix."""
    block_rows: list[sp.csr_matrix] = []
    DR_blocks = _cache_5_block_derivatives(DR, ngp)
    D2R_blocks = _cache_5_block_derivatives(D2R, ngp)
    DZ_blocks = _cache_5_block_derivatives(DZ, ngp)
    D2Z_blocks = _cache_5_block_derivatives(D2Z, ngp)
    D2RZ_blocks = _cache_5_block_derivatives(D2RZ, ngp)

    for i in range(5):
        row_parts = []
        for j in range(5):
            row_parts.append(
                _build_block_csr(
                    ngp,
                    coeffs.A0[i, j, :],
                    coeffs.Ax[i, j, :],
                    coeffs.Ay[i, j, :],
                    coeffs.Axx[i, j, :],
                    coeffs.Ayy[i, j, :],
                    coeffs.Axy[i, j, :],
                    DR_blocks[j],
                    D2R_blocks[j],
                    DZ_blocks[j],
                    D2Z_blocks[j],
                    D2RZ_blocks[j],
                )
            )
        block_rows.append(sp.hstack(row_parts, format="csr"))

    return sp.vstack(block_rows, format="csr")


def _assemble_block_operator_petsc(
    ngp: int,
    coeffs: LNSECoefficients,
    DR: sp.spmatrix,
    D2R: sp.spmatrix,
    DZ: sp.spmatrix,
    D2Z: sp.spmatrix,
    D2RZ: sp.spmatrix,
    *,
    solver_cfg: dict | None = None,
) -> "PETSc.Mat":
    """Assemble the LNSE block operator directly into PETSc via row insertion."""
    from petsc4py import PETSc
    from ..utils.petsc_core import (
        _use_distributed_matrix,
        assemble_matrix,
        create_distributed_aij,
        insert_local_csr_rows,
        ownership_range,
    )

    total_size = 5 * ngp
    distributed_matrix = _use_distributed_matrix(solver_cfg)
    if not distributed_matrix and PETSc.COMM_WORLD.getSize() == 1:
        csr = _assemble_block_operator_csr(ngp, coeffs, DR, D2R, DZ, D2Z, D2RZ)
        _note_assembly_metadata(raw_block_assembly_mode="petsc_local_full_csr")
        mat = PETSc.Mat().createAIJ(
            size=csr.shape,
            csr=(csr.indptr, csr.indices, csr.data),
        )
        mat.assemble()
        return mat

    DR_blocks = _cache_5_block_derivatives(DR, ngp)
    D2R_blocks = _cache_5_block_derivatives(D2R, ngp)
    DZ_blocks = _cache_5_block_derivatives(DZ, ngp)
    D2Z_blocks = _cache_5_block_derivatives(D2Z, ngp)
    D2RZ_blocks = _cache_5_block_derivatives(D2RZ, ngp)

    # Per-row preallocation: cache row nnz once instead of recalculating
    # ``getnnz(axis=1)`` inside every block-row estimate.
    block_row_nnz = np.ones(ngp, dtype=np.int64)
    for op_blocks in (DR_blocks, D2R_blocks, DZ_blocks, D2Z_blocks, D2RZ_blocks):
        for cache in op_blocks:
            if cache.row_nnz.size:
                block_row_nnz += cache.row_nnz
    d_nnz = np.tile(block_row_nnz, 5)
    np.clip(d_nnz, 1, total_size, out=d_nnz)

    matrix = create_distributed_aij(
        (total_size, total_size),
        nnz_diag=d_nnz,
        nnz_offdiag=d_nnz,
    )

    if distributed_matrix:
        _note_assembly_metadata(raw_block_assembly_mode="petsc_distributed_owned_rows")
        r0, r1 = ownership_range(matrix)
        owned_blocks = []
        for i in range(5):
            row_start = max(0, r0 - i * ngp)
            row_stop = min(ngp, r1 - i * ngp)
            if row_start >= row_stop:
                continue
            owned_blocks.append((i, row_start, row_stop))

        for i, row_start, row_stop in owned_blocks:
            # Build each (i,j) block ONCE and slice the owned rows.
            row_parts: list[sp.csr_matrix] = []
            for j in range(5):
                block = _build_block_csr(
                    ngp,
                    coeffs.A0[i, j, :],
                    coeffs.Ax[i, j, :],
                    coeffs.Ay[i, j, :],
                    coeffs.Axx[i, j, :],
                    coeffs.Ayy[i, j, :],
                    coeffs.Axy[i, j, :],
                    DR_blocks[j],
                    D2R_blocks[j],
                    DZ_blocks[j],
                    D2Z_blocks[j],
                    D2RZ_blocks[j],
                )
                row_parts.append(block[row_start:row_stop, :])

            local_rows = sp.hstack(row_parts, format="csr") if row_parts else sp.csr_matrix((0, total_size), dtype=complex)
            if local_rows.shape[0] > 0:
                insert_local_csr_rows(matrix, local_rows, row_start=i * ngp + row_start)
        return assemble_matrix(matrix)

    _note_assembly_metadata(raw_block_assembly_mode="petsc_local_row_insert_fallback")
    for i in range(5):
        for j in range(5):
            block = _build_block_csr(
                ngp,
                coeffs.A0[i, j, :],
                coeffs.Ax[i, j, :],
                coeffs.Ay[i, j, :],
                coeffs.Axx[i, j, :],
                coeffs.Ayy[i, j, :],
                coeffs.Axy[i, j, :],
                DR_blocks[j],
                D2R_blocks[j],
                DZ_blocks[j],
                D2Z_blocks[j],
                D2RZ_blocks[j],
            )
            if block.nnz == 0:
                continue
            col_offset = j * ngp
            for local_row in range(ngp):
                p0 = int(block.indptr[local_row])
                p1 = int(block.indptr[local_row + 1])
                if p0 == p1:
                    continue
                cols = block.indices[p0:p1] + col_offset
                vals = block.data[p0:p1]
                matrix.setValues(
                    i * ngp + local_row,
                    cols,
                    vals,
                    addv=PETSc.InsertMode.INSERT_VALUES,
                )

    return assemble_matrix(matrix)


def _build_index_structure(mesh_idx: dict, ngp: int) -> dict:
    """Build the variable/boundary index structure for DOFs.

    Creates index arrays mapping boundary locations to DOF indices in the
    5*ngp state vector q = [rho, u, v, w, T].
    """
    idx = dict(mesh_idx)  # copy boundary indices

    variables = ["rho", "u", "v", "w", "T"]
    var_offsets = {v: i * ngp for i, v in enumerate(variables)}

    # Variable DOF ranges (full index arrays)
    for v in variables:
        idx[f"{v}_j"] = np.arange(var_offsets[v], var_offsets[v] + ngp)

    # Border-variable indices
    borders = {"b": "bi", "t": "ti", "l": "li", "r": "ri"}
    for b_char, b_key in borders.items():
        if b_key not in idx:
            continue
        grid_idx = idx[b_key]
        all_border = []
        for v in variables:
            border_var_key = f"{b_key}_{v}"
            dof_idx = grid_idx + var_offsets[v]
            idx[border_var_key] = dof_idx
            all_border.append(dof_idx)
        idx[f"{b_key}_all"] = np.concatenate(all_border)

    return idx


def _build_block_csr(
    ngp: int,
    a0: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    axx: np.ndarray,
    ayy: np.ndarray,
    axy: np.ndarray,
    DR: _CSRDerivativeCache | sp.spmatrix,
    D2R: _CSRDerivativeCache | sp.spmatrix,
    DZ: _CSRDerivativeCache | sp.spmatrix,
    D2Z: _CSRDerivativeCache | sp.spmatrix,
    D2RZ: _CSRDerivativeCache | sp.spmatrix,
) -> sp.csr_matrix:
    """Build one ngp×ngp block in CSR format from row-scaled derivative terms."""
    # Compute flags once to avoid scanning each coefficient array twice.
    has_a0 = np.any(a0)
    has_ay = np.any(ay)
    has_ax = np.any(ax)
    has_ayy = np.any(ayy)
    has_axx = np.any(axx)
    has_axy = np.any(axy)

    if not (has_a0 or has_ay or has_ax or has_ayy or has_axx or has_axy):
        return sp.csr_matrix((ngp, ngp), dtype=complex)

    dr_cache = _as_csr_derivative_cache(DR)
    d2r_cache = _as_csr_derivative_cache(D2R)
    dz_cache = _as_csr_derivative_cache(DZ)
    d2z_cache = _as_csr_derivative_cache(D2Z)
    d2rz_cache = _as_csr_derivative_cache(D2RZ)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    if has_a0:
        diag_idx = np.arange(ngp, dtype=np.int64)
        row_parts.append(diag_idx)
        col_parts.append(diag_idx)
        data_parts.append(np.asarray(a0, dtype=complex))

    def _append_scaled_term(coeff: np.ndarray, cache: _CSRDerivativeCache) -> None:
        if cache.csr.nnz == 0:
            return
        row_parts.append(cache.row_indices)
        col_parts.append(cache.csr.indices)
        data_parts.append(coeff[cache.row_indices] * cache.csr.data)

    if has_ay:
        _append_scaled_term(ay, dr_cache)
    if has_ax:
        _append_scaled_term(ax, dz_cache)
    if has_ayy:
        _append_scaled_term(ayy, d2r_cache)
    if has_axx:
        _append_scaled_term(axx, d2z_cache)
    if has_axy:
        _append_scaled_term(axy, d2rz_cache)

    if not data_parts:
        return sp.csr_matrix((ngp, ngp), dtype=complex)

    block = sp.csr_matrix(
        (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(ngp, ngp),
        dtype=complex,
    )
    block.sum_duplicates()
    block.eliminate_zeros()
    block.sort_indices()
    return block


def _apply_floquet_shift(
    DZ: list[sp.csc_matrix] | sp.spmatrix,
    D2Z: list[sp.spmatrix] | sp.spmatrix,
    i_alpha: complex,
    ngp: int,
) -> tuple[list[sp.csc_matrix], list[sp.csc_matrix]]:
    """Shift axial derivatives with Floquet parameter ``iα``.

    Implements:
    DZ  -> DZ + iα I
    D2Z -> D2Z + 2 iα DZ + (iα)² I
    """
    dz_blocks = [cache.csr.tocsc() for cache in _cache_5_block_derivatives(DZ, ngp)]
    d2z_blocks = [cache.csr.tocsc() for cache in _cache_5_block_derivatives(D2Z, ngp)]
    I = sp.eye(ngp, format="csc")
    shifted_dz = [mat + i_alpha * I for mat in dz_blocks]
    shifted_d2z = [d2z + 2.0 * i_alpha * dz + i_alpha**2 * I for d2z, dz in zip(d2z_blocks, dz_blocks)]
    return shifted_dz, shifted_d2z


def _apply_inv_B(
    L0: sp.spmatrix,
    model: str,
    ngp: int,
    RHO: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    T: np.ndarray,
    kappa: float,
    Ma: float,
) -> sp.spmatrix:
    """Left-multiply L0 by ``inv(B/1j)`` (analytical block inverse).

    For the Cartesian model, B/1j = diag(1, RHO, ..., RHO), so the inverse
    is diagonal.  For the cylindrical (axy) model, B/1j is lower-triangular
    due to the conservative-to-primitive transformation, and the 5×5 block
    inverse is computed analytically at each grid point.
    """
    if model == "2d":
        # B/1j = diag(1, RHO, RHO, RHO, RHO)  →  inv = diag(1, 1/RHO, ...)
        inv_B_signs = np.empty(5 * ngp)
        inv_B_signs[0*ngp:1*ngp] = 1.0
        inv_B_signs[1*ngp:2*ngp] = 1.0 / RHO
        inv_B_signs[2*ngp:3*ngp] = 1.0 / RHO
        inv_B_signs[3*ngp:4*ngp] = 1.0 / RHO
        inv_B_signs[4*ngp:5*ngp] = 1.0 / RHO
        return sp.diags(inv_B_signs) @ L0

    # --- Cylindrical model: B/1j is lower-triangular ---
    #
    # In physical ordering [rho, u, v, w, T]:
    #
    #   B/1j = [ 1     0       0       0       0      ]
    #          [ U     RHO     0       0       0      ]
    #          [ V     0       RHO     0       0      ]
    #          [ W     0       0       RHO     0      ]
    #          [ KE    U*RHO   V*RHO   W*RHO   RHO*cv ]
    #
    # with KE = (U²+V²+W² + 2*cv*T)/2.
    #
    # The analytical inverse (also lower-triangular):
    #
    #   inv(B/1j) = [ 1        0        0        0        0          ]
    #              [ -U/RHO    1/RHO    0        0        0          ]
    #              [ -V/RHO    0        1/RHO    0        0          ]
    #              [ -W/RHO    0        0        1/RHO    0          ]
    #              [ e40       e41      e42      e43      1/(RHO*cv) ]
    #
    # where e4j = -vel_j / (RHO*cv)  for j=1,2,3  (vel = U, V, W)
    #       e40 = (U²+V²+W²)/(2*RHO*cv) - T/RHO
    #
    # Build inv(B/1j) as a 5×5 block matrix with diagonal blocks.
    cv = 1.0 / (kappa * (kappa - 1.0) * Ma**2)
    invRHO = 1.0 / RHO
    invRHOcv = invRHO / cv
    KE_half = 0.5 * (U**2 + V**2 + W**2)

    Z = sp.csc_matrix((ngp, ngp))

    def D(v):
        return sp.diags(v, format="csc")

    blocks = [
        [D(np.ones(ngp)),   Z,               Z,               Z,               Z                     ],
        [D(-U * invRHO),    D(invRHO),       Z,               Z,               Z                     ],
        [D(-V * invRHO),    Z,               D(invRHO),       Z,               Z                     ],
        [D(-W * invRHO),    Z,               Z,               D(invRHO),       Z                     ],
        [D(KE_half * invRHOcv - T * invRHO),
                            D(-U * invRHOcv),
                                            D(-V * invRHOcv),
                                                             D(-W * invRHOcv),
                                                                              D(invRHOcv)           ],
    ]
    inv_B = sp.bmat(blocks, format="csc")
    return inv_B @ L0


def _apply_inv_B_petsc(
    L0: "PETSc.Mat",
    model: str,
    ngp: int,
    RHO: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    T: np.ndarray,
    kappa: float,
    Ma: float,
) -> None:
    """Apply the analytical ``inv(B/1j)`` left scaling directly to PETSc rows."""
    from petsc4py import PETSc
    from ..utils.petsc_core import petsc_to_scipy

    row_scaling = np.ones(5 * ngp, dtype=complex)
    if model == "2d":
        row_scaling[1*ngp:2*ngp] = 1.0 / RHO
        row_scaling[2*ngp:3*ngp] = 1.0 / RHO
        row_scaling[3*ngp:4*ngp] = 1.0 / RHO
        row_scaling[4*ngp:5*ngp] = 1.0 / RHO
        diag = L0.createVecLeft()
        try:
            r0, r1 = diag.getOwnershipRange()
            if r1 > r0:
                rows = np.arange(r0, r1, dtype=PETSc.IntType)
                diag.setValues(
                    rows,
                    np.ascontiguousarray(row_scaling[r0:r1], dtype=complex),
                    addv=PETSc.InsertMode.INSERT_VALUES,
                )
            diag.assemblyBegin()
            diag.assemblyEnd()
            L0.diagonalScale(L=diag)
        finally:
            diag.destroy()
        L0.assemble()
        return

    L0_sp = petsc_to_scipy(L0, format="csr")
    transformed_csr = sp.csr_matrix(
        _apply_inv_B(L0_sp, model, ngp, RHO, U, V, W, T, kappa, Ma),
        dtype=complex,
    )
    transformed_csr.sum_duplicates()
    transformed_csr.eliminate_zeros()
    rebuilt = PETSc.Mat().createAIJ(
        size=L0.getSize(),
        csr=(transformed_csr.indptr, transformed_csr.indices, transformed_csr.data),
        comm=L0.getComm(),
    )
    rebuilt.assemble()
    # Allow new nonzero positions from the lower-triangular inv(B) multiply.
    L0.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    try:
        L0.zeroEntries()
        L0.axpy(1.0, rebuilt, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    finally:
        rebuilt.destroy()
    L0.assemble()


def _create_axy_diff_matrices(
    mesh: Mesh,
    m: float,
) -> tuple[
    tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix],
    tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix],
    tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix],
    tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix],
    tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix],
]:
    """Create radial/axial derivative blocks for the axy model.

    Parity selection follows MATLAB ``CreateDiffMatrices_Axy.m``, converted
    from coefficient ordering [rho, v_r, v_theta, v_z, T] to physical
    ordering [rho, u(axial), v(radial), w(azimuthal), T]:

    - m = 0: [symm, symm, asymm, asymm, symm]  (rho,u,T even; v,w odd)
    - m = 1: [asymm, asymm, symm, symm, asymm]  (v,w even; rho,u,T odd)
    - m >= 2: all asymm
    """
    if not np.isfinite(m):
        raise ValueError("m must be finite")

    m_int = int(np.rint(m))
    if m_int < 0 or not np.isclose(m, m_int):
        raise ValueError("m must be a non-negative integer mode number")

    if m_int == 0:
        parity = (True, True, False, False, True)
    elif m_int == 1:
        parity = (False, False, True, True, False)
    else:
        parity = (False, False, False, False, False)

    Dx = mesh.Dx.tocsc()
    D2x = mesh.D2x.tocsc()
    Dy = mesh.Dy.tocsc()
    D2y = mesh.D2y.tocsc()
    Dyx = mesh.Dyx.tocsc() if mesh.Dyx is not None else (mesh.Dy @ mesh.Dx).tocsc()

    DR_blocks: list[sp.csc_matrix] = []
    D2R_blocks: list[sp.csc_matrix] = []
    D2RZ_blocks: list[sp.csc_matrix] = []

    for use_symmetry in parity:
        if use_symmetry and mesh.y_symmetry and mesh.Dy_symm is not None and mesh.D2y_symm is not None:
            DR_blocks.append(mesh.Dy_symm.tocsc())
            D2R_blocks.append(mesh.D2y_symm.tocsc())
            if mesh.Dyx_symm is not None:
                D2RZ_blocks.append(mesh.Dyx_symm.tocsc())
            else:
                D2RZ_blocks.append((mesh.Dy_symm @ Dx).tocsc())
        elif mesh.y_symmetry and not use_symmetry and mesh.Dy_asymm is not None and mesh.D2y_asymm is not None:
            DR_blocks.append(mesh.Dy_asymm.tocsc())
            D2R_blocks.append(mesh.D2y_asymm.tocsc())
            if mesh.Dyx_asymm is not None:
                D2RZ_blocks.append(mesh.Dyx_asymm.tocsc())
            else:
                D2RZ_blocks.append((mesh.Dy_asymm @ Dx).tocsc())
        else:
            DR_blocks.append(Dy)
            D2R_blocks.append(D2y)
            D2RZ_blocks.append(Dyx)

    DZ_blocks = (Dx, Dx, Dx, Dx, Dx)
    D2Z_blocks = (D2x, D2x, D2x, D2x, D2x)

    return (
        tuple(DR_blocks),
        tuple(D2R_blocks),
        DZ_blocks,
        D2Z_blocks,
        tuple(D2RZ_blocks),
    )
