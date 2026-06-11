# SPDX-License-Identifier: GPL-3.0-only
"""Shared case-pipeline orchestration driver."""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import resource as _resource_module

try:
    import resource as _resource
except ImportError:  # pragma: no cover - Windows fallback
    _resource: "_resource_module | None" = None

import numpy as np
import scipy.sparse as sp

from linstabpy.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from mpi4py import MPI
from petsc4py import PETSc

from linstabpy.baseflow import load_hamada_case, sutherland_air
from linstabpy.boundary_conditions import analyze_bc_recipe, dirichlet_free_dofs
from linstabpy.config import Config
from linstabpy.linalg.backends import FACTOR_PRECISION, resolve_factor_backend
from linstabpy.mesh import Mesh
from linstabpy.mesh.sbp import DISSIPATION_DEFAULTS
from linstabpy.operators.assembly import build_lnse_system_result
from linstabpy.operators.energy_norm import compute_chu_norm
from linstabpy.resolvent.analysis import resolvent_analysis
from linstabpy.results_runtime import (
    EMPTY_COMPLEX,
    EMPTY_FLOAT,
    EMPTY_INT32,
    EMPTY_INT64,
)
from linstabpy.solvers.petsc_utils import _is_petsc_matrix, scipy_to_petsc, vec_to_numpy
from linstabpy.stability._eigenvalues import _solve_near_target
from linstabpy.stability._eigenvalues import eigenvalues as temporal_eigenvalues
from linstabpy.stability.acceptance import (
    _postcheck_target_distance,
    check_conjugate_pair_completeness,
    evaluate_stability_acceptance,
)
from linstabpy.utils.jsonc import load_jsonc

_STAGE_WALL: dict[str, float] = {}


def _array_from_mapping(
    mapping: dict[str, object],
    key: str,
    *,
    dtype,
    default: np.ndarray,
) -> np.ndarray:
    """Return a typed array from a runtime info mapping without allocating a default."""
    return np.asarray(mapping.get(key, default), dtype=dtype)


def _config_hash(config_dict: dict) -> str:
    """Return a 64-char hex SHA-256 of the sorted-key JSON serialization."""
    import hashlib
    import json

    return hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()


def _git_head_sha() -> str:
    """Return the current git HEAD SHA, or 'unknown' on failure."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _first_env(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment value across ``names``."""
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        text = raw.strip()
        if text:
            return text
    return default


def _output_npz_path() -> Path:
    """Return the per-run output archive path.

    Defaults to the current study working directory, but allows the timing
    harness to isolate concurrent runs by providing a unique target path.
    """
    raw = _first_env(
        "LINSTABPY_CASE_OUTPUT_NPZ",
        "LINSTABPY_CASE2_OUTPUT_NPZ",
        "LINSTABPY_CASE1_OUTPUT_NPZ",
        default=str(Path.cwd() / f"{_log_prefix()}.npz"),
    )
    return Path(raw).resolve()


def _log_prefix() -> str:
    """Return the runtime log prefix for the current case runner."""
    return os.getenv("LINSTABPY_CASE_LOG_PREFIX", "case1").strip() or "case1"


def _log(message: str) -> None:
    """Small helper for consistent prefixed progress messages."""
    if PETSc.COMM_WORLD.getRank() == 0:
        print(f"[{_log_prefix()}] {message}", flush=True)


def _peak_rss_bytes() -> int:
    """Return this rank's peak resident set size in bytes."""
    if _resource is None:
        return 0
    rss = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _stage_start() -> tuple[float, int]:
    """Capture timing and memory counters for a runtime stage."""
    return time.perf_counter(), _peak_rss_bytes()


def _log_stage(stage: str, stage_start: float | tuple[float, int]) -> None:
    """Emit a machine-parseable stage timing line."""
    if isinstance(stage_start, tuple):
        start_time, rss0 = stage_start
        elapsed = time.perf_counter() - start_time
        _STAGE_WALL[stage] = _STAGE_WALL.get(stage, 0.0) + elapsed
        rss_delta = max(0, _peak_rss_bytes() - rss0)
        _log(
            f"stage complete: stage={stage} elapsed={elapsed:.2f}s rss_peak_delta_bytes={rss_delta}"
        )
        return
    elapsed = time.perf_counter() - stage_start
    _STAGE_WALL[stage] = _STAGE_WALL.get(stage, 0.0) + elapsed
    _log(f"stage complete: stage={stage} elapsed={elapsed:.2f}s")


def _default_stage_heartbeat_seconds() -> float:
    """Return the default heartbeat interval for long-running stages."""
    raw = os.getenv("LINSTABPY_STAGE_HEARTBEAT_SECONDS", "30")
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            "LINSTABPY_STAGE_HEARTBEAT_SECONDS must be a float number of seconds"
        ) from None


def _parse_alpha_filter(alpha_filter_raw: object) -> float | str:
    """Parse the configured alpha-filter value into its runtime form."""
    try:
        return float(alpha_filter_raw)
    except (ValueError, TypeError):
        alpha_filter = str(alpha_filter_raw).strip()
        if alpha_filter.lower() == "none" or alpha_filter == "":
            return "none"
        return alpha_filter


def _effective_alpha_filter(
    alpha_filter_raw: object,
    *,
    run_resolvent: bool,
    context: str = "config",
) -> float | str:
    """Parse the configured alpha-filter, honoring it for every analysis stage.

    The previous behaviour silently overrode a numeric ``alpha_filter`` to
    ``"none"`` whenever ``run_resolvent=False``, as a build-time optimisation
    for EVP-only runs. That override had a real numerical side-effect: SBP
    diagonal-norm operators have a near-null-space mode at the 2-Δ Nyquist
    wavenumber, and the Mattsson operator-level filter is the designed
    damping for that mode. Skipping it for EVP-only runs made those runs
    produce pixel-scale saw-tooth eigenvectors localised near the boundary
    closure — a purely numerical artefact that also contaminated resolvent
    extraction whenever someone happened to set ``alpha_filter: 0.0``.

    The filter is an operator-level property and should not depend on which
    analysis stage runs. We now honour the configured value for every stage;
    the ``run_resolvent`` argument is kept so the context parameter remains
    informative in any future logging.
    """
    del run_resolvent  # kept in signature for back-compat; no longer branches.
    del context
    return _parse_alpha_filter(alpha_filter_raw)


@contextmanager
def _stage_heartbeat(stage: str, *, interval_seconds: float | None = None):
    """Emit periodic 'still alive' logs while a long stage is running."""
    interval = (
        _default_stage_heartbeat_seconds() if interval_seconds is None else float(interval_seconds)
    )
    if interval <= 0.0:
        yield
        return

    started = time.perf_counter()
    stop_event = threading.Event()

    def _worker() -> None:
        while not stop_event.wait(interval):
            _log(f"stage heartbeat: stage={stage} elapsed={time.perf_counter() - started:.2f}s")

    thread = threading.Thread(
        target=_worker,
        name=f"stage-heartbeat-{stage}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=max(interval, 0.1) + 1.0)


def _restrict_operator_for_stability(
    operator: sp.spmatrix | PETSc.Mat,
    W_diag: np.ndarray,
    constrained_indices: np.ndarray,
) -> tuple[sp.spmatrix | PETSc.Mat, np.ndarray, np.ndarray]:
    """Remove constrained DOFs from the stability EVP."""
    ndof = int(operator.getSize()[0]) if _is_petsc_matrix(operator) else int(operator.shape[0])
    free_dofs = dirichlet_free_dofs(ndof, constrained_indices)
    W_reduced = np.asarray(W_diag, dtype=float)[free_dofs]
    if _is_petsc_matrix(operator):
        r0, r1 = operator.getOwnershipRange()
        free_local = free_dofs[(free_dofs >= r0) & (free_dofs < r1)].astype(
            PETSc.IntType, copy=False
        )
        is_free = PETSc.IS().createGeneral(free_local, comm=operator.getComm())
        try:
            reduced = operator.createSubMatrix(is_free, is_free)
        finally:
            is_free.destroy()
        return reduced, W_reduced, free_dofs
    reduced = operator.tocsc()[free_dofs][:, free_dofs].tocsc()
    return reduced, W_reduced, free_dofs


def _expand_reduced_modes(
    modes_reduced: np.ndarray, n_full: int, free_dofs: np.ndarray
) -> np.ndarray:
    """Expand reduced eigenvectors back into the full state space."""
    if modes_reduced.size == 0:
        return np.empty((n_full, 0), dtype=complex)
    modes_full = np.zeros((n_full, modes_reduced.shape[1]), dtype=complex)
    modes_full[free_dofs, :] = modes_reduced
    return modes_full


def _build_distributed_vec_from_global_array(mat: PETSc.Mat, values: np.ndarray) -> PETSc.Vec:
    """Create a PETSc vector matching ``mat`` ownership from a global array."""
    vec = mat.createVecRight()
    r0, r1 = vec.getOwnershipRange()
    if r1 > r0:
        rows = np.arange(r0, r1, dtype=PETSc.IntType)
        vec.setValues(
            rows, np.asarray(values[r0:r1], dtype=complex), addv=PETSc.InsertMode.INSERT_VALUES
        )
    vec.assemblyBegin()
    vec.assemblyEnd()
    return vec


def _postcheck_stability_modes(
    L0: sp.spmatrix | PETSc.Mat,
    omega: np.ndarray,
    modes: np.ndarray,
    *,
    target: complex,
    bc_indices: np.ndarray,
    constrained_dirichlet_indices: np.ndarray,
    conjugate_aware: bool = False,
) -> dict[str, object]:
    """Compute explicit residual and BC-leakage diagnostics for returned modes."""
    n_modes = int(np.asarray(omega).size)
    if n_modes == 0 or modes.size == 0:
        return {
            "postcheck_residual_abs": EMPTY_FLOAT,
            "postcheck_residual_rel": EMPTY_FLOAT,
            "postcheck_residual_interior_abs": EMPTY_FLOAT,
            "postcheck_residual_bc_abs": EMPTY_FLOAT,
            "postcheck_bc_leak_max": EMPTY_FLOAT,
            "postcheck_bc_leak_p99": EMPTY_FLOAT,
            "postcheck_dirichlet_leak_max": EMPTY_FLOAT,
            "postcheck_dirichlet_leak_p99": EMPTY_FLOAT,
            "postcheck_target_distance": EMPTY_FLOAT,
        }

    bc_idx = np.unique(np.asarray(bc_indices, dtype=int))
    dir_idx = np.unique(np.asarray(constrained_dirichlet_indices, dtype=int))
    n_full = int(modes.shape[0])
    mask = np.ones(n_full, dtype=bool)
    mask[bc_idx] = False

    residual_abs = np.full(n_modes, np.nan, dtype=float)
    residual_rel = np.full(n_modes, np.nan, dtype=float)
    residual_interior_abs = np.full(n_modes, np.nan, dtype=float)
    residual_bc_abs = np.full(n_modes, np.nan, dtype=float)
    bc_leak_max = np.full(n_modes, np.nan, dtype=float)
    bc_leak_p99 = np.full(n_modes, np.nan, dtype=float)
    dir_leak_max = np.full(n_modes, np.nan, dtype=float)
    dir_leak_p99 = np.full(n_modes, np.nan, dtype=float)
    target_distance = _postcheck_target_distance(
        np.asarray(omega, dtype=complex), complex(target), conjugate_aware=conjugate_aware
    )

    for i in range(n_modes):
        mode = np.asarray(modes[:, i], dtype=complex)
        if bc_idx.size:
            bc_amp = np.abs(mode[bc_idx])
            bc_leak_max[i] = float(np.max(bc_amp))
            bc_leak_p99[i] = float(np.quantile(bc_amp, 0.99))
        if dir_idx.size:
            dir_amp = np.abs(mode[dir_idx])
            dir_leak_max[i] = float(np.max(dir_amp))
            dir_leak_p99[i] = float(np.quantile(dir_amp, 0.99))

        if _is_petsc_matrix(L0):
            x = _build_distributed_vec_from_global_array(L0, mode)
            y = x.duplicate()
            try:
                L0.mult(x, y)
                y.scale(1j)
                y.axpy(-complex(omega[i]), x)
                res = vec_to_numpy(y)
            finally:
                x.destroy()
                y.destroy()
        else:
            res = 1j * (L0 @ mode) - complex(omega[i]) * mode

        residual_abs[i] = float(np.linalg.norm(res))
        mode_norm = float(np.linalg.norm(mode))
        residual_rel[i] = residual_abs[i] / max(mode_norm, np.finfo(float).eps)
        residual_interior_abs[i] = float(np.linalg.norm(res[mask]))
        residual_bc_abs[i] = float(np.linalg.norm(res[bc_idx])) if bc_idx.size else 0.0

    return {
        "postcheck_residual_abs": residual_abs,
        "postcheck_residual_rel": residual_rel,
        "postcheck_residual_interior_abs": residual_interior_abs,
        "postcheck_residual_bc_abs": residual_bc_abs,
        "postcheck_bc_leak_max": bc_leak_max,
        "postcheck_bc_leak_p99": bc_leak_p99,
        "postcheck_dirichlet_leak_max": dir_leak_max,
        "postcheck_dirichlet_leak_p99": dir_leak_p99,
        "postcheck_target_distance": np.asarray(target_distance, dtype=float),
    }


def _resolve_omega_target(raw: object, default: complex) -> complex:
    """Parse an optional explicit omega target override from config."""
    if raw is None:
        return complex(default)
    if isinstance(raw, (int, float, np.floating)):
        return complex(float(raw), 0.0)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return complex(default)
        return complex(text)
    if isinstance(raw, dict):
        real = float(raw.get("real", np.real(default)))
        imag = float(raw.get("imag", np.imag(default)))
        return complex(real, imag)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return complex(float(raw[0]), float(raw[1]))
    raise ValueError(
        "analysis.omega_target must be null, a number, a complex-like string, "
        "a [real, imag] pair, or an object with {real, imag}"
    )


def _resolve_optional_float(raw: object) -> float:
    """Parse an optional scalar threshold from config, returning NaN when unset."""
    if raw is None:
        return float("nan")
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() in {"none", "null", "nan"}:
            return float("nan")
        return float(text)
    return float(raw)


def _cfg_first(cfg_fn, *paths: str) -> object:
    """Return the first configured value across a list of dotted config paths."""
    for path in paths:
        value = cfg_fn(path, None)
        if value is not None:
            return value
    return None


def _cfg_get(config: dict, path: str, default):
    """Read a dotted JSONC path from the loaded runtime config."""
    node = config
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


@dataclass(frozen=True)
class _RuntimeConfigSource:
    config_path: Path
    config: dict
    config_hash: str
    git_sha: str


def _stage_load_runtime_config() -> _RuntimeConfigSource:
    """Load the JSONC runtime config and immutable provenance strings."""
    config_path = Path(
        _first_env(
            "LINSTABPY_CASE_CONFIG",
            "LINSTABPY_CASE2_CONFIG",
            "LINSTABPY_CASE1_CONFIG",
            default=str(Path.cwd() / "configs" / "parity_config.jsonc"),
        )
    )
    config: dict = {}
    if config_path.exists():
        config = load_jsonc(config_path)
        from linstabpy.config import validate_config

        validate_config(config, source=str(config_path))
        _log(f"loaded config: {config_path}")
    elif any(
        name in os.environ
        for name in (
            "LINSTABPY_CASE_CONFIG",
            "LINSTABPY_CASE2_CONFIG",
            "LINSTABPY_CASE1_CONFIG",
        )
    ):
        raise FileNotFoundError(f"case config file not found: {config_path}")

    return _RuntimeConfigSource(
        config_path=config_path,
        config=config,
        config_hash=_config_hash(config),
        git_sha=_git_head_sha(),
    )


@dataclass(frozen=True)
class _LoadedCase:
    X: np.ndarray
    Y: np.ndarray
    baseflow: object
    sponge: np.ndarray
    sponge_enabled: bool
    sponge_source: str


def _stage_load_case(
    *,
    case_dir: Path,
    Re: float,
    Ma: float,
    Pr: float,
    kappa: float,
    T_0: float,
    stride_ksi: int,
    stride_eta: int,
    zero_spanwise: bool,
    sponge_mat_path: Path,
    prefer_matlab_sponge_aux: bool,
    native_sponge_enabled: bool,
    sponge_cfg: dict | None,
    zero_mu_t_derivs: bool,
    heartbeat_seconds: float,
) -> _LoadedCase:
    """Load the base-flow snapshot and apply transport-property conventions."""
    stage_start = _stage_start()
    with _stage_heartbeat("load_case", interval_seconds=heartbeat_seconds):
        X, Y, baseflow, sponge = load_hamada_case(
            case_dir,
            Re=Re,
            Ma=Ma,
            Pr=Pr,
            kappa=kappa,
            T_0=T_0,
            stride_ksi=stride_ksi,
            stride_eta=stride_eta,
            transpose_input=True,
            zero_spanwise=zero_spanwise,
            sponge_mat_path=sponge_mat_path,
            prefer_matlab_sponge_aux=prefer_matlab_sponge_aux,
            native_sponge_cfg=sponge_cfg,
        )

    sutherland_air(baseflow)
    if zero_mu_t_derivs:
        baseflow.dmudT[:] = 0.0
        baseflow.d2mudT2[:] = 0.0
    sponge_enabled = bool(np.count_nonzero(sponge))
    if native_sponge_enabled:
        sponge_source = "native"
    elif prefer_matlab_sponge_aux:
        sponge_source = "matlab_aux"
    elif sponge_enabled:
        sponge_source = "baseflow_file"
    else:
        sponge_source = "off"
    _log(
        f"loaded case arrays: shape={X.shape}, points={X.size}, "
        f"sponge_source={sponge_source}, "
        f"sponge_max={float(np.max(sponge)):.3g}, sponge_nnz={int(np.count_nonzero(sponge))}"
    )
    _log_stage("load_case", stage_start)
    return _LoadedCase(X, Y, baseflow, sponge, sponge_enabled, sponge_source)


def _stage_build_mesh(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    sponge: np.ndarray,
    fd_order: int,
    alpha_filter: float | str,
    od_cfg: dict,
    od_enabled: bool,
    heartbeat_seconds: float,
) -> Mesh:
    """Build the computational mesh and attach the runtime sponge field."""
    stage_start = _stage_start()
    ny, nx = X.shape
    with _stage_heartbeat("build_mesh", interval_seconds=heartbeat_seconds):
        cmesh = Mesh.create(
            (0.0, 1.0),
            (0.0, 0.5),
            nx=nx,
            ny=ny,
            fd_order=fd_order,
            alpha_filter=alpha_filter,
            numerics_operator_dissipation=od_cfg if od_enabled else None,
        )
        mesh = cmesh.deform(X_new=X, Y_new=Y)
        mesh.sponge = sponge
    _log_stage("build_mesh", stage_start)
    return mesh


def _stage_compute_chu_norm(
    *,
    mesh: Mesh,
    baseflow,
    heartbeat_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Chu energy weights for stability and resolvent stages."""
    stage_start = _stage_start()
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO_p = baseflow.RHO.ravel(order="F")[p]
    T_p = baseflow.T.ravel(order="F")[p]
    with _stage_heartbeat("compute_chu_norm", interval_seconds=heartbeat_seconds):
        W_diag, invW_diag = compute_chu_norm(
            W_int, RHO_p, T_p, baseflow.kappa, baseflow.Ma, mesh.ngp
        )
    _log_stage("compute_chu_norm", stage_start)
    return W_diag, invW_diag


@dataclass(frozen=True)
class _AssemblyStageResult:
    L0: object
    L0_resolv: object
    idx: dict
    bc_indices: np.ndarray
    total_bc_indices: np.ndarray
    effective_dirichlet_indices: np.ndarray
    effective_neumann_indices: np.ndarray
    overlap_indices: np.ndarray
    column_zeroed_dirichlet_indices: np.ndarray
    constrained_dirichlet_indices: np.ndarray


def _stage_assemble_operators(
    *,
    mesh: Mesh,
    baseflow,
    beta: float,
    bc_recipe,
    penalty: float,
    assembly_backend,
    solver_cfg: dict,
    eps_stab: float,
    eps_resolv: float,
    run_resolvent: bool,
    use_root_operator: bool,
    rank: int,
    reduce_dirichlet_set: str,
    heartbeat_seconds: float,
) -> _AssemblyStageResult:
    """Assemble stability/resolvent operators and derive BC provenance arrays."""
    if use_root_operator and rank != 0:
        # Root-assembly mode for MPI scalability:
        # non-root ranks keep only a shape-compatible placeholder; the real
        # SciPy->PETSc operator transfer happens from root inside eigenvalue.py.
        ndof = mesh.ngp * 5
        L0 = sp.csc_matrix((ndof, ndof), dtype=complex)
        idx = {
            "rho_j": np.arange(0, mesh.ngp, dtype=np.int64),
            "u_j": np.arange(mesh.ngp, 2 * mesh.ngp, dtype=np.int64),
            "v_j": np.arange(2 * mesh.ngp, 3 * mesh.ngp, dtype=np.int64),
            "w_j": np.arange(3 * mesh.ngp, 4 * mesh.ngp, dtype=np.int64),
            "T_j": np.arange(4 * mesh.ngp, 5 * mesh.ngp, dtype=np.int64),
        }
        bc_indices = EMPTY_INT64
        L0_resolv = L0
        bc_analysis = analyze_bc_recipe(mesh, idx, bc_recipe)
    else:
        stage_start = _stage_start()
        _log("assembling LNSE operator")
        with _stage_heartbeat("assemble_operator_bc", interval_seconds=heartbeat_seconds):
            lnse_system = build_lnse_system_result(
                mesh,
                baseflow,
                kx=beta,
                model="2D",
                bc_recipe=bc_recipe,
                bc_penalty=penalty,
                backend=assembly_backend,
                solver_cfg=solver_cfg,
                dissipation_epsilon=eps_stab,
            )
            L0 = lnse_system.operator
            idx = lnse_system.idx
            bc_indices = lnse_system.bc_indices
        nnz = int(L0.getInfo()["nz_used"]) if hasattr(L0, "getInfo") else int(L0.nnz)
        _log(f"operator assembled with BCs: shape={(mesh.ngp * 5, mesh.ngp * 5)}, nnz={nnz}")
        _log_stage("assemble_operator_bc", stage_start)

        if run_resolvent and eps_resolv != eps_stab:
            stage_start = _stage_start()
            _log(f"assembling separate resolvent LNSE operator (dissipation_epsilon={eps_resolv})")
            with _stage_heartbeat(
                "assemble_resolvent_operator_bc",
                interval_seconds=heartbeat_seconds,
            ):
                resolvent_system = build_lnse_system_result(
                    mesh,
                    baseflow,
                    kx=beta,
                    model="2D",
                    bc_recipe=bc_recipe,
                    bc_penalty=penalty,
                    backend=assembly_backend,
                    solver_cfg=solver_cfg,
                    dissipation_epsilon=eps_resolv,
                )
                L0_resolv = resolvent_system.operator
            nnz_resolv = (
                int(L0_resolv.getInfo()["nz_used"])
                if hasattr(L0_resolv, "getInfo")
                else int(L0_resolv.nnz)
            )
            _log(
                "resolvent operator assembled with BCs: "
                f"shape={(mesh.ngp * 5, mesh.ngp * 5)}, nnz={nnz_resolv}"
            )
            _log_stage("assemble_resolvent_operator_bc", stage_start)
        elif run_resolvent:
            L0_resolv = L0
            _log(
                "reusing stability LNSE operator for resolvent "
                f"(eps_stab == eps_resolv == {eps_stab})"
            )
        else:
            L0_resolv = L0

        bc_analysis = (
            lnse_system.bc_analysis
            if lnse_system.bc_analysis is not None
            else analyze_bc_recipe(mesh, idx, bc_recipe)
        )

    effective_dirichlet_indices = np.asarray(
        bc_analysis.get("effective_dirichlet_indices", EMPTY_INT64),
        dtype=np.int64,
    )
    effective_neumann_indices = np.asarray(
        bc_analysis.get("effective_neumann_indices", EMPTY_INT64),
        dtype=np.int64,
    )
    overlap_indices = np.asarray(
        bc_analysis.get("overlap_indices", EMPTY_INT64),
        dtype=np.int64,
    )
    column_zeroed_dirichlet_indices = np.asarray(
        bc_analysis.get("column_zeroed_dirichlet_indices", EMPTY_INT64),
        dtype=np.int64,
    )
    total_bc_indices = np.asarray(
        bc_analysis.get("total_bc_indices", EMPTY_INT64),
        dtype=np.int64,
    )
    constrained_dirichlet_indices = (
        column_zeroed_dirichlet_indices
        if reduce_dirichlet_set == "column_zeroed"
        else effective_dirichlet_indices
    )
    if rank == 0:
        if total_bc_indices.size and not np.array_equal(np.unique(bc_indices), total_bc_indices):
            _log(
                "warning: analyzed BC index set differs from assembled BC index set "
                f"(assembled={np.unique(bc_indices).size}, analyzed={total_bc_indices.size})"
            )
        _log(
            "bc analysis: "
            f"dirichlet_total={column_zeroed_dirichlet_indices.size} "
            f"effective_dirichlet={effective_dirichlet_indices.size} "
            f"effective_neumann={effective_neumann_indices.size} "
            f"cross_type_overlap={overlap_indices.size}"
        )

    return _AssemblyStageResult(
        L0=L0,
        L0_resolv=L0_resolv,
        idx=idx,
        bc_indices=bc_indices,
        total_bc_indices=total_bc_indices,
        effective_dirichlet_indices=effective_dirichlet_indices,
        effective_neumann_indices=effective_neumann_indices,
        overlap_indices=overlap_indices,
        column_zeroed_dirichlet_indices=column_zeroed_dirichlet_indices,
        constrained_dirichlet_indices=constrained_dirichlet_indices,
    )


@dataclass(frozen=True)
class _ResolventStageResult:
    gains: np.ndarray
    responses: np.ndarray
    forcings: np.ndarray
    rho_fwd: np.ndarray
    rho_adj: np.ndarray


def _empty_resolvent_result(ndof: int) -> _ResolventStageResult:
    """Return shape-stable empty arrays for runs with resolvent disabled."""
    return _ResolventStageResult(
        gains=EMPTY_FLOAT,
        responses=np.empty((ndof, 0), dtype=complex),
        forcings=np.empty((ndof, 0), dtype=complex),
        rho_fwd=EMPTY_FLOAT,
        rho_adj=EMPTY_FLOAT,
    )


def _stage_run_resolvent(
    *,
    run_resolvent: bool,
    L0_resolv,
    mesh: Mesh,
    W_diag: np.ndarray,
    omega_target: complex,
    n_modes_resolvent: int,
    total_bc_indices: np.ndarray,
    solver_cfg: dict,
    cfg: Config | None = None,
    heartbeat_seconds: float,
) -> _ResolventStageResult:
    """Run resolvent analysis and its SVD postcheck for one target frequency."""
    ndof = mesh.ngp * 5
    if not run_resolvent:
        return _empty_resolvent_result(ndof)

    stage_start = _stage_start()
    _log(f"running resolvent: omega_target={omega_target:.8g}, modes={n_modes_resolvent}")
    B_diag = np.ones(ndof, dtype=float)
    B_diag[total_bc_indices] = 0.0
    B_mat_sp = sp.diags(B_diag, format="csc")
    B_mat = scipy_to_petsc(B_mat_sp, solver_cfg) if hasattr(L0_resolv, "getInfo") else B_mat_sp
    C_mat = B_mat.copy()
    with _stage_heartbeat("resolvent", interval_seconds=heartbeat_seconds):
        gains, responses, forcings, rho_fwd, rho_adj = resolvent_analysis(
            L0_resolv,
            omega_target,
            n_modes_resolvent,
            W_diag,
            B_mat=B_mat,
            C_mat=C_mat,
            filter_fn=mesh.filters.filter if mesh.filters is not None else None,
            filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
            solver_cfg=solver_cfg,
            cfg=cfg,
            residuals=True,
        )
    _log("resolvent analysis finished")
    _log_stage("resolvent", stage_start)

    if gains.size > 0:
        _log(
            "resolvent postcheck: "
            f"rho_fwd_max={float(np.max(rho_fwd)):.3e} "
            f"rho_adj_max={float(np.max(rho_adj)):.3e}"
        )

    return _ResolventStageResult(gains, responses, forcings, rho_fwd, rho_adj)


@dataclass(frozen=True)
class _StabilityStageResult:
    omega: np.ndarray
    direct_modes: np.ndarray
    stability_info: dict[str, object]


def _stage_run_stability(
    *,
    run_stability: bool,
    use_raw_eigs: bool,
    reduce_dirichlet: bool,
    use_root_operator: bool,
    reduce_dirichlet_set: str,
    constrained_dirichlet_indices: np.ndarray,
    effective_dirichlet_indices: np.ndarray,
    effective_neumann_indices: np.ndarray,
    overlap_indices: np.ndarray,
    L0,
    W_diag: np.ndarray,
    omega_target: complex,
    n_modes: int,
    mesh: Mesh,
    solver_cfg: dict,
    cfg: Config | None = None,
    total_bc_indices: np.ndarray,
    stability_acceptance_conjugate_aware: bool,
    heartbeat_seconds: float,
) -> _StabilityStageResult:
    """Run the optional temporal stability stage and postcheck returned modes."""
    omega = EMPTY_COMPLEX
    direct_modes = np.empty((mesh.ngp * 5, 0), dtype=complex)
    stability_info: dict[str, object] = {}
    if not run_stability:
        return _StabilityStageResult(omega, direct_modes, stability_info)

    stage_start = _stage_start()
    stability_meta = {
        "reduce_dirichlet": bool(reduce_dirichlet and not use_root_operator),
        "reduce_dirichlet_set": reduce_dirichlet_set,
        "effective_dirichlet_count": int(effective_dirichlet_indices.size),
        "constrained_dirichlet_count": int(constrained_dirichlet_indices.size),
        "effective_neumann_count": int(effective_neumann_indices.size),
        "cross_type_overlap_count": int(overlap_indices.size),
    }
    stability_info.update(stability_meta)
    if use_raw_eigs:
        _log(
            "running raw eigensolve on L0 "
            f"(diagnostic path): target={omega_target:.8g}, modes={n_modes}"
        )
        with _stage_heartbeat("stability", interval_seconds=heartbeat_seconds):
            omega, direct_modes, stability_info = _solve_near_target(
                L0.tocsc(),
                n_modes=n_modes,
                target=omega_target,
                solver_cfg=solver_cfg,
                cfg=cfg,
                return_info=True,
            )
        order = np.argsort(np.abs(omega - omega_target))
        omega = omega[order]
        direct_modes = direct_modes[:, order]
        for key in (
            "residual_estimates",
            "residual_absolute",
            "residual_relative",
            "residual_backward",
        ):
            residuals_raw = np.asarray(stability_info.get(key, EMPTY_FLOAT), dtype=float)
            if residuals_raw.size == omega.size:
                stability_info[key] = residuals_raw[order]
    else:
        if reduce_dirichlet and use_root_operator:
            _log("reduced stability EVP requested but disabled with root-assembly mode")
        if reduce_dirichlet and constrained_dirichlet_indices.size and not use_root_operator:
            _log(
                "running reduced temporal eigensolve on i*L0 "
                f"(reduce_set={reduce_dirichlet_set}, constrained={constrained_dirichlet_indices.size}, modes={n_modes})"
            )
            L0_stab, W_stab, free_dofs = _restrict_operator_for_stability(
                L0,
                W_diag,
                constrained_dirichlet_indices,
            )
            try:
                with _stage_heartbeat("stability", interval_seconds=heartbeat_seconds):
                    omega, modes_reduced, stability_info = temporal_eigenvalues(
                        L0_stab,
                        n_modes=n_modes,
                        target=omega_target,
                        W_diag=W_stab,
                        solver_cfg=solver_cfg,
                        cfg=cfg,
                        return_info=True,
                    )
            finally:
                if _is_petsc_matrix(L0_stab):
                    L0_stab.destroy()
            direct_modes = _expand_reduced_modes(modes_reduced, mesh.ngp * 5, free_dofs)
            stability_info["free_dof_count"] = int(free_dofs.size)
            stability_info["constrained_dirichlet_indices"] = np.asarray(
                constrained_dirichlet_indices, dtype=np.int64
            )
        else:
            _log(
                "running temporal eigensolve on i*L0 "
                f"(MATLAB convention): target={omega_target:.8g}, modes={n_modes}"
            )
            with _stage_heartbeat("stability", interval_seconds=heartbeat_seconds):
                omega, direct_modes, stability_info = temporal_eigenvalues(
                    L0,
                    n_modes=n_modes,
                    target=omega_target,
                    W_diag=W_diag,
                    solver_cfg=solver_cfg,
                    cfg=cfg,
                    return_info=True,
                )
    if not use_root_operator:
        stability_info.update(
            _postcheck_stability_modes(
                L0,
                omega,
                direct_modes,
                target=omega_target,
                bc_indices=total_bc_indices,
                constrained_dirichlet_indices=constrained_dirichlet_indices,
                conjugate_aware=stability_acceptance_conjugate_aware,
            )
        )
    stability_info.update(stability_meta)
    _log(f"stability analysis finished: converged_modes={omega.size}")
    residuals = _array_from_mapping(
        stability_info, "residual_estimates", dtype=float, default=EMPTY_FLOAT
    )
    if residuals.size:
        finite = residuals[np.isfinite(residuals)]
        if finite.size:
            _log(
                "stability eig convergence: "
                f"nev={stability_info.get('nev_requested')} "
                f"ncv={stability_info.get('ncv_used')} "
                f"nconv={stability_info.get('nconv')} "
                f"eps_iterations={stability_info.get('eps_iterations')} "
                f"max_residual={float(np.max(finite)):.3e} "
                f"median_residual={float(np.median(finite)):.3e}"
            )
    postcheck_rel = np.asarray(
        stability_info.get("postcheck_residual_rel", EMPTY_FLOAT),
        dtype=float,
    )
    if postcheck_rel.size:
        finite_post = postcheck_rel[np.isfinite(postcheck_rel)]
        if finite_post.size:
            _log(
                "stability postcheck: "
                f"max_rel={float(np.max(finite_post)):.3e} "
                f"median_rel={float(np.median(finite_post)):.3e}"
            )
    _log_stage("stability", stage_start)
    return _StabilityStageResult(omega, direct_modes, stability_info)


@dataclass(frozen=True)
class _StabilityAcceptanceStageResult:
    residual_estimates: np.ndarray
    residual_absolute: np.ndarray
    residual_relative: np.ndarray
    residual_backward: np.ndarray
    postcheck_residual_abs: np.ndarray
    postcheck_residual_rel: np.ndarray
    postcheck_residual_interior_abs: np.ndarray
    postcheck_residual_bc_abs: np.ndarray
    postcheck_bc_leak_max: np.ndarray
    postcheck_bc_leak_p99: np.ndarray
    postcheck_dirichlet_leak_max: np.ndarray
    postcheck_dirichlet_leak_p99: np.ndarray
    postcheck_target_distance: np.ndarray
    acceptance: dict
    conjugate_diagnostic: dict


def _stage_evaluate_stability_acceptance(
    *,
    omega: np.ndarray,
    stability_info: dict[str, object],
    stability_acceptance_postcheck_rel_max: float,
    stability_acceptance_target_distance_max: float,
    stability_acceptance_omega_real_abs_max: float,
    stability_acceptance_omega_imag_abs_max: float,
    stability_acceptance_conjugate_aware: bool,
) -> _StabilityAcceptanceStageResult:
    """Prepare stability postcheck arrays, acceptance verdicts, and conjugate diagnostics."""
    stability_residual_estimates = _array_from_mapping(
        stability_info, "residual_estimates", dtype=float, default=EMPTY_FLOAT
    )
    stability_residual_absolute = _array_from_mapping(
        stability_info, "residual_absolute", dtype=float, default=EMPTY_FLOAT
    )
    stability_residual_relative = _array_from_mapping(
        stability_info, "residual_relative", dtype=float, default=EMPTY_FLOAT
    )
    stability_residual_backward = _array_from_mapping(
        stability_info, "residual_backward", dtype=float, default=EMPTY_FLOAT
    )
    stability_postcheck_residual_abs = _array_from_mapping(
        stability_info, "postcheck_residual_abs", dtype=float, default=EMPTY_FLOAT
    )
    stability_postcheck_residual_rel = _array_from_mapping(
        stability_info, "postcheck_residual_rel", dtype=float, default=EMPTY_FLOAT
    )
    stability_postcheck_residual_interior_abs = _array_from_mapping(
        stability_info,
        "postcheck_residual_interior_abs",
        dtype=float,
        default=EMPTY_FLOAT,
    )
    stability_postcheck_residual_bc_abs = _array_from_mapping(
        stability_info, "postcheck_residual_bc_abs", dtype=float, default=EMPTY_FLOAT
    )
    stability_postcheck_bc_leak_max = _array_from_mapping(
        stability_info, "postcheck_bc_leak_max", dtype=float, default=EMPTY_FLOAT
    )
    stability_postcheck_bc_leak_p99 = _array_from_mapping(
        stability_info, "postcheck_bc_leak_p99", dtype=float, default=EMPTY_FLOAT
    )
    stability_postcheck_dirichlet_leak_max = _array_from_mapping(
        stability_info, "postcheck_dirichlet_leak_max", dtype=float, default=EMPTY_FLOAT
    )
    stability_postcheck_dirichlet_leak_p99 = _array_from_mapping(
        stability_info, "postcheck_dirichlet_leak_p99", dtype=float, default=EMPTY_FLOAT
    )
    stability_postcheck_target_distance = _array_from_mapping(
        stability_info, "postcheck_target_distance", dtype=float, default=EMPTY_FLOAT
    )
    stability_acceptance = evaluate_stability_acceptance(
        omega,
        stability_postcheck_residual_rel,
        stability_postcheck_target_distance,
        postcheck_rel_max=stability_acceptance_postcheck_rel_max,
        target_distance_max=stability_acceptance_target_distance_max,
        omega_real_abs_max=stability_acceptance_omega_real_abs_max,
        omega_imag_abs_max=stability_acceptance_omega_imag_abs_max,
        conjugate_aware=stability_acceptance_conjugate_aware,
    )
    if stability_acceptance:
        accepted_count = int(stability_acceptance.get("accepted_count", 0))
        _log(
            "stability acceptance: "
            f"accepted_modes={accepted_count} "
            f"postcheck_rel_max={stability_acceptance_postcheck_rel_max:.3e} "
            f"target_distance_max={stability_acceptance_target_distance_max:.3e} "
            f"omega_real_abs_max={stability_acceptance_omega_real_abs_max:.3e} "
            f"omega_imag_abs_max={stability_acceptance_omega_imag_abs_max:.3e}"
        )
    accepted_idx_for_diag = np.asarray(
        stability_acceptance.get("accepted_indices", EMPTY_INT32),
        dtype=np.int32,
    )
    conjugate_diagnostic = check_conjugate_pair_completeness(omega, accepted_idx_for_diag)
    if conjugate_diagnostic["missing_partner_indices"].size > 0:
        _log(
            f"warning: {conjugate_diagnostic['missing_partner_indices'].size} accepted mode(s) "
            "have no conjugate partner in returned spectrum: "
            f"indices={conjugate_diagnostic['missing_partner_indices'].tolist()}"
        )
    return _StabilityAcceptanceStageResult(
        residual_estimates=stability_residual_estimates,
        residual_absolute=stability_residual_absolute,
        residual_relative=stability_residual_relative,
        residual_backward=stability_residual_backward,
        postcheck_residual_abs=stability_postcheck_residual_abs,
        postcheck_residual_rel=stability_postcheck_residual_rel,
        postcheck_residual_interior_abs=stability_postcheck_residual_interior_abs,
        postcheck_residual_bc_abs=stability_postcheck_residual_bc_abs,
        postcheck_bc_leak_max=stability_postcheck_bc_leak_max,
        postcheck_bc_leak_p99=stability_postcheck_bc_leak_p99,
        postcheck_dirichlet_leak_max=stability_postcheck_dirichlet_leak_max,
        postcheck_dirichlet_leak_p99=stability_postcheck_dirichlet_leak_p99,
        postcheck_target_distance=stability_postcheck_target_distance,
        acceptance=stability_acceptance,
        conjugate_diagnostic=conjugate_diagnostic,
    )


def run() -> None:
    """Run the shared case pipeline and write the requested NPZ artifact."""
    t0 = time.perf_counter()
    output_npz = _output_npz_path()
    _log("starting compute")
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()

    config_source = _stage_load_runtime_config()
    config = config_source.config
    config_hash = config_source.config_hash
    git_sha = config_source.git_sha
    cfg = lambda path, default: _cfg_get(config, path, default)

    heartbeat_seconds = float(
        cfg("logging.stage_heartbeat_seconds", _default_stage_heartbeat_seconds())
    )

    # ------------------------------------------------------------------
    # 1) Read runtime configuration from environment variables.
    #    Defaults here mirror the reference MATLAB setup.
    # ------------------------------------------------------------------
    case_dir = Path(str(cfg("paths.case_dir", str(Path.cwd() / "data" / "baseflow"))))
    sponge_mat_path = Path(
        str(cfg("paths.case_inputs_mat", "MATLAB_exports_resolvent/case_inputs_used.mat"))
    )
    Re = float(cfg("physics.Re", 395000.0))
    Ma = float(cfg("physics.Ma", 2.0))
    Pr = float(cfg("physics.Pr", 0.747))
    # MATLAB export note:
    # - `case_inputs_used.mat/baseFlow.kappa` is 1.4
    # - `run_metadata.mat/kappa` is 1.31
    # The assembled reference operators/spectrum align with baseFlow.kappa.
    kappa = float(cfg("physics.kappa", 1.4))
    T_0 = float(cfg("physics.T_0", 1540.0))

    stride_ksi = int(cfg("numerics.stride_ksi", 1))
    stride_eta = int(cfg("numerics.stride_eta", 1))
    fd_order = int(cfg("numerics.fd_order", 4))
    beta = float(cfg("numerics.beta", 0.0))
    penalty = float(cfg("bc.penalty", 1e8))

    ratio_u = float(cfg("analysis.ratio_uinf", 1.0))
    ratio_l = float(cfg("analysis.ratio_l", 0.1))
    St = float(cfg("analysis.St", 0.1))

    # Target angular frequency used by resolvent and shift-invert EVP.
    omega_target = _resolve_omega_target(
        cfg("analysis.omega_target", None),
        2.0 * np.pi * St * ratio_u / ratio_l,
    )
    stability_acceptance_postcheck_rel_max = _resolve_optional_float(
        cfg("analysis.stability_acceptance.postcheck_rel_max", None)
    )
    stability_acceptance_target_distance_max = _resolve_optional_float(
        cfg("analysis.stability_acceptance.target_distance_max", None)
    )
    stability_acceptance_omega_real_abs_max = _resolve_optional_float(
        _cfg_first(
            cfg,
            "analysis.stability_acceptance.omega_real_abs_max",
            "analysis.stability_acceptance.frequency_abs_max",
            "analysis.stability_acceptance.omega_abs_max",
        )
    )
    stability_acceptance_omega_imag_abs_max = _resolve_optional_float(
        _cfg_first(
            cfg,
            "analysis.stability_acceptance.omega_imag_abs_max",
            "analysis.stability_acceptance.growth_abs_max",
            "analysis.stability_acceptance.sigma_abs_max",
        )
    )
    # Default True: real-valued baseflows have a conjugate-pair spectrum, and
    # admitting the conjugate partner at the same target_distance_max is the
    # physically correct metric (see _postcheck_target_distance docstring and
    # SWBLI.md for the mode-family rationale). Set to False in the JSONC only
    # for complex baseflows (Floquet, complex perturbation-tracking).
    stability_acceptance_conjugate_aware: bool = bool(
        cfg("analysis.stability_acceptance.conjugate_aware", True)
    )

    n_modes = int(cfg("analysis.n_modes_stability", 100))
    n_modes_resolvent = int(cfg("analysis.n_modes_resolvent", 10))
    run_resolvent = bool(cfg("analysis.run_resolvent", True))
    run_stability = bool(cfg("analysis.run_stability", True))
    zero_spanwise = bool(cfg("physics.zero_spanwise", False))
    sponge_cfg_raw = cfg("physics.sponge", None)
    sponge_cfg = sponge_cfg_raw if isinstance(sponge_cfg_raw, dict) else None
    native_sponge_enabled = bool((sponge_cfg or {}).get("enabled", False))
    use_matlab_sponge_aux = bool(cfg("physics.use_matlab_sponge_aux", False))
    alpha_filter_raw = cfg("numerics.alpha_filter", "none")
    alpha_filter = _effective_alpha_filter(
        alpha_filter_raw,
        run_resolvent=run_resolvent,
        context="config",
    )
    od_raw = cfg("numerics.operator_dissipation", None)
    od_cfg = od_raw if isinstance(od_raw, dict) else {}
    od_enabled = bool(od_cfg.get("enabled", False))
    od_epsilon = float(od_cfg.get("epsilon", 0.0))
    od_order_diss = int(od_cfg.get("order_diss", fd_order))
    od_apply_to_stability = bool(od_cfg.get("apply_to_stability", True))
    od_apply_to_resolvent = bool(od_cfg.get("apply_to_resolvent", False))
    od_transition_fraction = float(
        od_cfg.get("transition_fraction", DISSIPATION_DEFAULTS["transition_fraction"])
    )
    od_endpoint_exponent = float(
        od_cfg.get("endpoint_exponent", DISSIPATION_DEFAULTS["endpoint_exponent"])
    )
    od_interior_value = float(od_cfg.get("interior_value", DISSIPATION_DEFAULTS["interior_value"]))
    # Effective epsilon per analysis stage
    eps_stab = od_epsilon if (od_enabled and od_apply_to_stability) else 0.0
    eps_resolv = (
        od_epsilon if (od_enabled and od_apply_to_resolvent) else 0.0
    )  # Phase 4: use when wiring resolvent
    if od_enabled:
        _log(
            f"numerics.operator_dissipation effective: enabled=True, epsilon={od_epsilon}, order_diss={od_order_diss}, "
            f"apply_to_stability={od_apply_to_stability}, apply_to_resolvent={od_apply_to_resolvent}, "
            f"transition_fraction={od_transition_fraction}, endpoint_exponent={od_endpoint_exponent}, "
            f"interior_value={od_interior_value}"
        )
    # MATLAB export note:
    # case_inputs_used.mat stores uppercase dMUdT/d2MUdT2 as zeros; GetLinProblem.m
    # consumes those names in getCoeffsLaminar_Cartesian.m. Keep this default True
    # for strict parity with the exported reference matrices/spectrum.
    zero_mu_t_derivs = bool(cfg("physics.zero_mu_t_derivs", True))
    # MATLAB stability convention:
    #   L0_matlab = L0 / (-1i)  => omega = eig(L0_matlab) = eig(1j * L0)
    # Keep this as default for parity with exported MATLAB spectrum.
    use_raw_eigs = bool(cfg("analysis.use_raw_eigs", False))
    prefer_matlab_sponge_aux = (
        (not native_sponge_enabled) and use_matlab_sponge_aux and sponge_mat_path.exists()
    )
    solver_cfg = config.get("solver")
    if solver_cfg is None:
        solver_cfg = {}
        config["solver"] = solver_cfg
    cfg_obj = Config.load(config)

    # Read solver knobs from JSONC (env-var fallback for backward compat).
    _ecfg = solver_cfg.get("eigen", {})
    st_solver = (
        _ecfg.get("st_solver", os.getenv("LINSTABPY_EIGEN_ST_SOLVER", "mumps")).strip().lower()
    )
    iterative_st_solver = st_solver in {"iterative", "krylov", "slepc"}
    _root_raw = _first_env(
        "LINSTABPY_CASE_ROOT_ASSEMBLY",
        "LINSTABPY_CASE2_ROOT_ASSEMBLY",
        "LINSTABPY_CASE1_ROOT_ASSEMBLY",
    )
    requested_root_operator = size > 1 and (
        (_root_raw.strip().lower() in {"1", "true", "yes", "on"})
        if _root_raw is not None
        else iterative_st_solver
    )
    use_root_operator = requested_root_operator and iterative_st_solver
    if requested_root_operator and not iterative_st_solver:
        _log(
            "root-assembly mode requested but disabled because "
            f"eigen.st_solver={st_solver!r} is not iterative"
        )
    if use_root_operator and run_resolvent:
        raise RuntimeError(
            "Root-only operator assembly mode is currently intended for stability solves only. "
            "Set LINSTABPY_CASE_ROOT_ASSEMBLY=0 when running resolvent in MPI."
        )
    if use_root_operator:
        solver_cfg.setdefault("root_scatter", True)

    _linsolve = solver_cfg.get("linsolve", {})
    _resolvent_cfg = solver_cfg.get("resolvent", _linsolve)
    resolvent_type = _resolvent_cfg.get("type", _linsolve.get("type", "mumps"))
    factor_backend_provenance = resolve_factor_backend(solver_cfg, default="mumps") or "petsc"
    matrix_mode = solver_cfg.get("petsc_matrix_mode", "auto")
    assembly_backend = cfg("assembly_backend", None)
    reduce_dirichlet = bool(_ecfg.get("reduce_dirichlet", False))
    reduce_dirichlet_set = str(_ecfg.get("reduce_dirichlet_set", "column_zeroed")).strip().lower()
    if reduce_dirichlet_set not in {"effective", "column_zeroed"}:
        raise ValueError("solver.eigen.reduce_dirichlet_set must be 'effective' or 'column_zeroed'")

    _log(
        f"config: case_dir={case_dir} fd_order={fd_order} n_modes={n_modes} "
        f"run_stability={run_stability} run_resolvent={run_resolvent} "
        f"raw_eigs={use_raw_eigs} zero_mu_t_derivs={zero_mu_t_derivs} "
        f"alpha_filter={alpha_filter} "
        f"sponge_request={'native' if native_sponge_enabled else ('matlab_aux' if prefer_matlab_sponge_aux else 'baseflow_file')}"
    )
    _log(
        "solver setup: "
        f"mpi_ranks={size} "
        f"st_solver={st_solver} "
        f"resolvent_solver={resolvent_type} "
        f"root_assembly={use_root_operator} "
        f"matrix_mode={matrix_mode}"
    )

    # ------------------------------------------------------------------
    # 2) Load one case snapshot (x, y, rho, u, v, w, T, sponge) and
    #    compute viscosity/transport via Sutherland law.
    # ------------------------------------------------------------------
    loaded_case = _stage_load_case(
        case_dir=case_dir,
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        T_0=T_0,
        stride_ksi=stride_ksi,
        stride_eta=stride_eta,
        zero_spanwise=zero_spanwise,
        sponge_mat_path=sponge_mat_path,
        prefer_matlab_sponge_aux=prefer_matlab_sponge_aux,
        native_sponge_enabled=native_sponge_enabled,
        sponge_cfg=sponge_cfg,
        zero_mu_t_derivs=zero_mu_t_derivs,
        heartbeat_seconds=heartbeat_seconds,
    )
    X = loaded_case.X
    Y = loaded_case.Y
    baseflow = loaded_case.baseflow
    sponge = loaded_case.sponge
    sponge_enabled = loaded_case.sponge_enabled
    sponge_source = loaded_case.sponge_source

    # ------------------------------------------------------------------
    # 3) Build a computational mesh, then deform it to physical X/Y.
    # ------------------------------------------------------------------
    mesh = _stage_build_mesh(
        X=X,
        Y=Y,
        sponge=sponge,
        fd_order=fd_order,
        alpha_filter=alpha_filter,
        od_cfg=od_cfg,
        od_enabled=od_enabled,
        heartbeat_seconds=heartbeat_seconds,
    )

    # ------------------------------------------------------------------
    # 4) Assemble LNSE operator and impose boundary conditions.
    # ------------------------------------------------------------------
    bc_recipe = cfg(
        "bc.recipe",
        [
            {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
            {"type": "dirichlet", "borders": "b", "variables": "ruvw"},
            {"type": "neumann", "borders": "b", "variables": "T", "exclude_corners": True},
            {"type": "neumann", "borders": "r", "variables": "ruvwT", "exclude_corners": True},
        ],
    )
    assembly_result = _stage_assemble_operators(
        mesh=mesh,
        baseflow=baseflow,
        beta=beta,
        bc_recipe=bc_recipe,
        penalty=penalty,
        assembly_backend=assembly_backend,
        solver_cfg=solver_cfg,
        eps_stab=eps_stab,
        eps_resolv=eps_resolv,
        run_resolvent=run_resolvent,
        use_root_operator=use_root_operator,
        rank=rank,
        reduce_dirichlet_set=reduce_dirichlet_set,
        heartbeat_seconds=heartbeat_seconds,
    )
    L0 = assembly_result.L0
    L0_resolv = assembly_result.L0_resolv
    idx = assembly_result.idx
    bc_indices = assembly_result.bc_indices
    total_bc_indices = assembly_result.total_bc_indices
    effective_dirichlet_indices = assembly_result.effective_dirichlet_indices
    effective_neumann_indices = assembly_result.effective_neumann_indices
    overlap_indices = assembly_result.overlap_indices
    column_zeroed_dirichlet_indices = assembly_result.column_zeroed_dirichlet_indices
    constrained_dirichlet_indices = assembly_result.constrained_dirichlet_indices

    # ------------------------------------------------------------------
    # 5) Build Chu energy norm weights. These weights are required for
    #    weighted analyses (resolvent and temporal formulation).
    # ------------------------------------------------------------------
    W_diag, invW_diag = _stage_compute_chu_norm(
        mesh=mesh,
        baseflow=baseflow,
        heartbeat_seconds=heartbeat_seconds,
    )

    # ------------------------------------------------------------------
    # 6) Optional resolvent analysis at omega_target.
    # ------------------------------------------------------------------
    resolvent_result = _stage_run_resolvent(
        run_resolvent=run_resolvent,
        L0_resolv=L0_resolv,
        mesh=mesh,
        W_diag=W_diag,
        omega_target=omega_target,
        n_modes_resolvent=n_modes_resolvent,
        total_bc_indices=total_bc_indices,
        solver_cfg=solver_cfg,
        cfg=cfg_obj,
        heartbeat_seconds=heartbeat_seconds,
    )
    gains = resolvent_result.gains
    responses = resolvent_result.responses
    forcings = resolvent_result.forcings
    resolvent_rho_fwd = resolvent_result.rho_fwd
    resolvent_rho_adj = resolvent_result.rho_adj

    # ------------------------------------------------------------------
    # 7) Optional stability analysis.
    #    - raw_eigs: eig(L0) directly (diagnostic-only path, not MATLAB parity)
    #    - temporal: eig(i*L0), which matches MATLAB's L0 = L0 / -1i convention
    # ------------------------------------------------------------------
    stability_result = _stage_run_stability(
        run_stability=run_stability,
        use_raw_eigs=use_raw_eigs,
        reduce_dirichlet=reduce_dirichlet,
        use_root_operator=use_root_operator,
        reduce_dirichlet_set=reduce_dirichlet_set,
        constrained_dirichlet_indices=constrained_dirichlet_indices,
        effective_dirichlet_indices=effective_dirichlet_indices,
        effective_neumann_indices=effective_neumann_indices,
        overlap_indices=overlap_indices,
        L0=L0,
        W_diag=W_diag,
        omega_target=omega_target,
        n_modes=n_modes,
        mesh=mesh,
        solver_cfg=solver_cfg,
        cfg=cfg_obj,
        total_bc_indices=total_bc_indices,
        stability_acceptance_conjugate_aware=stability_acceptance_conjugate_aware,
        heartbeat_seconds=heartbeat_seconds,
    )
    omega = stability_result.omega
    direct_modes = stability_result.direct_modes
    stability_info = stability_result.stability_info

    # ------------------------------------------------------------------
    # 8) Save full run state to one NPZ for downstream parity/plots.
    # ------------------------------------------------------------------
    # Cross-rank peak-RSS aggregation MUST run on EVERY rank (collective),
    # BEFORE the rank-0-only save block below -- otherwise the allreduce
    # deadlocks every multi-rank run. The aggregated scalar is threaded into the
    # rank-0-only save (see _stage_write_npz / collect_cost_metrics).
    _peak_rss_total_bytes = MPI.COMM_WORLD.allreduce(int(_peak_rss_bytes()), op=MPI.SUM)
    if rank == 0:
        acceptance_result = _stage_evaluate_stability_acceptance(
            omega=omega,
            stability_info=stability_info,
            stability_acceptance_postcheck_rel_max=stability_acceptance_postcheck_rel_max,
            stability_acceptance_target_distance_max=stability_acceptance_target_distance_max,
            stability_acceptance_omega_real_abs_max=stability_acceptance_omega_real_abs_max,
            stability_acceptance_omega_imag_abs_max=stability_acceptance_omega_imag_abs_max,
            stability_acceptance_conjugate_aware=stability_acceptance_conjugate_aware,
        )
        stability_residual_estimates = acceptance_result.residual_estimates
        stability_residual_absolute = acceptance_result.residual_absolute
        stability_residual_relative = acceptance_result.residual_relative
        stability_residual_backward = acceptance_result.residual_backward
        stability_postcheck_residual_abs = acceptance_result.postcheck_residual_abs
        stability_postcheck_residual_rel = acceptance_result.postcheck_residual_rel
        stability_postcheck_residual_interior_abs = (
            acceptance_result.postcheck_residual_interior_abs
        )
        stability_postcheck_residual_bc_abs = acceptance_result.postcheck_residual_bc_abs
        stability_postcheck_bc_leak_max = acceptance_result.postcheck_bc_leak_max
        stability_postcheck_bc_leak_p99 = acceptance_result.postcheck_bc_leak_p99
        stability_postcheck_dirichlet_leak_max = acceptance_result.postcheck_dirichlet_leak_max
        stability_postcheck_dirichlet_leak_p99 = acceptance_result.postcheck_dirichlet_leak_p99
        stability_postcheck_target_distance = acceptance_result.postcheck_target_distance
        stability_acceptance = acceptance_result.acceptance
        _conj_diag = acceptance_result.conjugate_diagnostic
        from linstabpy.results_runtime import _stage_write_npz

        _stage_write_npz(
            output_npz=output_npz,
            heartbeat_seconds=heartbeat_seconds,
            peak_rss_bytes=_peak_rss_total_bytes,
            stage_wall=_STAGE_WALL,
            payload=dict(
                omega=omega,
                direct_modes=direct_modes,
                gains=gains,
                responses=responses,
                forcings=forcings,
                resolvent_rho_fwd=resolvent_rho_fwd,
                resolvent_rho_adj=resolvent_rho_adj,
                omega_target=omega_target,
                use_raw_eigs=use_raw_eigs,
                St=St,
                beta=beta,
                fd_order=fd_order,
                alpha_filter=np.array(str(alpha_filter)),
                use_matlab_sponge_aux=use_matlab_sponge_aux,
                sponge_enabled=sponge_enabled,
                sponge_source=np.array(sponge_source),
                sponge_preset=np.array(str((sponge_cfg or {}).get("preset", ""))),
                native_sponge_requested=native_sponge_enabled,
                matlab_sponge_aux_requested=use_matlab_sponge_aux,
                n_modes_stability_requested=n_modes,
                n_modes_resolvent_requested=n_modes_resolvent,
                Re=Re,
                Ma=Ma,
                Pr=Pr,
                kappa=kappa,
                T_0=T_0,
                X=mesh.X,
                Y=mesh.Y,
                U=baseflow.U,
                sponge=mesh.sponge,
                idx_rho_j=idx["rho_j"],
                idx_u_j=idx["u_j"],
                idx_v_j=idx["v_j"],
                idx_w_j=idx["w_j"],
                idx_T_j=idx["T_j"],
                bc_indices=bc_indices,
                bc_total_indices=total_bc_indices,
                bc_effective_dirichlet_indices=effective_dirichlet_indices,
                bc_column_zeroed_dirichlet_indices=column_zeroed_dirichlet_indices,
                bc_effective_neumann_indices=effective_neumann_indices,
                bc_overlap_indices=overlap_indices,
                W_diag=W_diag,
                invW_diag=invW_diag,
                stability_eps_nev=int(stability_info.get("nev_requested", -1)),
                stability_eps_ncv=int(stability_info.get("ncv_used", -1)),
                stability_eps_nconv=int(stability_info.get("nconv", -1)),
                stability_eps_iterations=int(stability_info.get("eps_iterations", -1)),
                stability_eps_reason=np.array(str(stability_info.get("eps_reason", ""))),
                stability_ksp_reason=np.array(str(stability_info.get("ksp_reason", ""))),
                stability_eps_conv_test=np.array(str(stability_info.get("conv_test", ""))),
                stability_eps_tol=float(stability_info.get("eps_tol", np.nan)),
                stability_eps_max_it=int(stability_info.get("eps_max_it", -1)),
                stability_eps_setup_seconds=float(stability_info.get("eps_setup_seconds", np.nan)),
                stability_eps_solve_seconds=float(stability_info.get("eps_solve_seconds", np.nan)),
                stability_true_residual=bool(stability_info.get("true_residual", False)),
                stability_reduce_dirichlet=bool(stability_info.get("reduce_dirichlet", False)),
                stability_reduce_dirichlet_set=np.array(
                    str(stability_info.get("reduce_dirichlet_set", ""))
                ),
                stability_free_dof_count=int(stability_info.get("free_dof_count", -1)),
                stability_constrained_dirichlet_count=int(
                    stability_info.get("constrained_dirichlet_count", -1)
                ),
                stability_residual_error_type=np.array(
                    str(stability_info.get("residual_error_type", ""))
                ),
                stability_residual_estimates=stability_residual_estimates,
                stability_residual_absolute=stability_residual_absolute,
                stability_residual_relative=stability_residual_relative,
                stability_residual_backward=stability_residual_backward,
                stability_postcheck_residual_abs=stability_postcheck_residual_abs,
                stability_postcheck_residual_rel=stability_postcheck_residual_rel,
                stability_postcheck_residual_interior_abs=stability_postcheck_residual_interior_abs,
                stability_postcheck_residual_bc_abs=stability_postcheck_residual_bc_abs,
                stability_postcheck_bc_leak_max=stability_postcheck_bc_leak_max,
                stability_postcheck_bc_leak_p99=stability_postcheck_bc_leak_p99,
                stability_postcheck_dirichlet_leak_max=stability_postcheck_dirichlet_leak_max,
                stability_postcheck_dirichlet_leak_p99=stability_postcheck_dirichlet_leak_p99,
                stability_postcheck_target_distance=stability_postcheck_target_distance,
                # Full pre-acceptance dump for every ARPACK-converged mode (Front A).
                # These arrays cover ALL converged modes, not just accepted ones.
                stability_all_omega=np.asarray(omega, dtype=complex),
                stability_all_postcheck_rel=stability_postcheck_residual_rel,
                stability_all_target_distance=stability_postcheck_target_distance,
                stability_all_reject_reason_code=np.asarray(
                    stability_acceptance.get("reject_reason_code", EMPTY_INT32),
                    dtype=np.int32,
                ),
                stability_acceptance_postcheck_rel_max=float(
                    stability_acceptance_postcheck_rel_max
                ),
                stability_acceptance_target_distance_max=float(
                    stability_acceptance_target_distance_max
                ),
                stability_acceptance_omega_real_abs_max=float(
                    stability_acceptance_omega_real_abs_max
                ),
                stability_acceptance_omega_imag_abs_max=float(
                    stability_acceptance_omega_imag_abs_max
                ),
                stability_acceptance_conjugate_aware=bool(
                    stability_acceptance.get("acceptance_conjugate_aware", False)
                ),
                stability_accepted_count=int(stability_acceptance.get("accepted_count", -1)),
                stability_accepted_indices=np.asarray(
                    stability_acceptance.get("accepted_indices", EMPTY_INT32),
                    dtype=np.int32,
                ),
                stability_rejected_indices=np.asarray(
                    stability_acceptance.get("rejected_indices", EMPTY_INT32),
                    dtype=np.int32,
                ),
                stability_accepted_postcheck_rel_min=float(
                    stability_acceptance.get("accepted_postcheck_rel_min", np.nan)
                ),
                stability_accepted_postcheck_rel_median=float(
                    stability_acceptance.get("accepted_postcheck_rel_median", np.nan)
                ),
                stability_accepted_target_distance_min=float(
                    stability_acceptance.get("accepted_target_distance_min", np.nan)
                ),
                stability_accepted_target_distance_median=float(
                    stability_acceptance.get("accepted_target_distance_median", np.nan)
                ),
                stability_accepted_omega_real_min=float(
                    stability_acceptance.get("accepted_omega_real_min", np.nan)
                ),
                stability_accepted_omega_real_max=float(
                    stability_acceptance.get("accepted_omega_real_max", np.nan)
                ),
                stability_accepted_omega_imag_min=float(
                    stability_acceptance.get("accepted_omega_imag_min", np.nan)
                ),
                stability_accepted_omega_imag_max=float(
                    stability_acceptance.get("accepted_omega_imag_max", np.nan)
                ),
                stability_leading_weighted_norm_error=float(
                    stability_info.get("leading_weighted_norm_error", np.nan)
                ),
                stability_weighted_norm_tol=float(stability_info.get("weighted_norm_tol", np.nan)),
                numerics_operator_dissipation_enabled=np.bool_(od_enabled),
                numerics_operator_dissipation_epsilon=np.float64(od_epsilon),
                numerics_operator_dissipation_order_diss=np.int64(od_order_diss),
                numerics_operator_dissipation_apply_to_stability=np.bool_(od_apply_to_stability),
                numerics_operator_dissipation_apply_to_resolvent=np.bool_(od_apply_to_resolvent),
                numerics_operator_dissipation_transition_fraction=np.float64(
                    od_transition_fraction
                ),
                numerics_operator_dissipation_endpoint_exponent=np.float64(od_endpoint_exponent),
                numerics_operator_dissipation_interior_value=np.float64(od_interior_value),
                solver_eigen_st_solver_used=str(st_solver),
                solver_apply_to_resolvent_effective=np.bool_(eps_resolv > 0.0),
                provenance_config_hash=np.bytes_(config_hash[:16].encode()),
                provenance_linstabpy_git_sha=np.bytes_(git_sha.encode()),
                provenance_factor_backend=np.array(str(factor_backend_provenance)),
                provenance_factor_precision=np.array(FACTOR_PRECISION),
                stability_missing_conjugate_partner_indices=np.asarray(
                    _conj_diag["missing_partner_indices"], dtype=np.int32
                ),
                stability_partner_rejected_indices=np.asarray(
                    _conj_diag["partner_rejected_indices"], dtype=np.int32
                ),
            ),
        )
    elapsed = time.perf_counter() - t0
    _log(f"results saved to {output_npz}")
    _log(f"compute finished in {elapsed:.2f}s")
    comm.barrier()
