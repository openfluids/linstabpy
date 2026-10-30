# SPDX-License-Identifier: GPL-3.0-only
"""Shared case-pipeline orchestration driver."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from linstabpy.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from mpi4py import MPI
from petsc4py import PETSc

from linstabpy.baseflow import load_case_fields, sutherland_air
from linstabpy.boundary_conditions import analyze_bc_recipe, dirichlet_free_dofs
from linstabpy.config import Config, ConfigValidationError
from linstabpy.config.defaults import DISSIPATION_APPLY_DEFAULTS, stage_dissipation_epsilon
from linstabpy.driver.payload import (
    RUNTIME_NPZ_CONTRACT_KEYS as RUNTIME_NPZ_CONTRACT_KEYS,  # noqa: F401
)
from linstabpy.driver.payload import build_runtime_payload
from linstabpy.driver.preflight import (
    baseflow_advisories,
    config_advisories,
    format_advisories,
    format_banner,
    format_summary,
    mesh_advisories,
    result_advisories,
)
from linstabpy.linalg.backends import resolve_factor_backend
from linstabpy.mesh import Mesh
from linstabpy.mesh.sbp import DISSIPATION_DEFAULTS
from linstabpy.operators.assembly import build_lnse_system_result
from linstabpy.operators.energy_norm import compute_chu_norm
from linstabpy.resolvent.analysis import resolvent_analysis
from linstabpy.solvers.petsc_utils import _is_petsc_matrix, vec_to_numpy
from linstabpy.solvers.router.resolvent_map import _forcing_response_mats
from linstabpy.stability._eigenvalues import _solve_near_target
from linstabpy.stability._eigenvalues import eigenvalues as temporal_eigenvalues
from linstabpy.stability.acceptance import (
    _postcheck_target_distance,
    check_conjugate_pair_completeness,
    evaluate_stability_acceptance,
)
from linstabpy.utils._solver_cfg import ITERATIVE_EIGEN_SOLVER_TYPES
from linstabpy.utils.cost_metrics import peak_rss_bytes as _peak_rss_bytes
from linstabpy.utils.provenance import _git_head_sha, collect_provenance  # noqa: F401
from linstabpy.utils.yaml_io import load_config


def _readonly_empty(dtype) -> np.ndarray:
    """Return a shared empty array that callers must not mutate."""
    array = np.array([], dtype=dtype)
    array.setflags(write=False)
    return array


EMPTY_FLOAT = _readonly_empty(float)
EMPTY_COMPLEX = _readonly_empty(complex)
EMPTY_INT32 = _readonly_empty(np.int32)
EMPTY_INT64 = _readonly_empty(np.int64)
# Shared empty arrays used as read-only defaults in hot runtime paths.

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


def _first_env(name: str) -> str | None:
    """Return the stripped environment value of ``name``, or None if unset or empty."""
    return (os.getenv(name) or "").strip() or None


def _runtime_config_path() -> Path:
    """Return the configured runtime config path without loading it."""
    raw = _first_env("LINSTABPY_CASE_CONFIG")
    if raw is None:
        raise ConfigValidationError(
            "LINSTABPY_CASE_CONFIG is not set. Set it to the path of a YAML case file."
        )
    return Path(raw)


def _output_npz_path(provenance: dict[str, str] | None = None) -> Path:
    """Return the per-run output archive path.

    Defaults to the current study working directory with provenance embedded.
    Concurrent timing runs can isolate themselves with a unique target path.
    """
    raw = _first_env("LINSTABPY_CASE_OUTPUT_NPZ")
    if raw is not None:
        return Path(raw).resolve()
    if provenance is None:
        provenance = collect_provenance(_runtime_config_path())
    config_md5 = provenance["config_md5"]
    sha8 = provenance["git_sha"][:8]
    return (Path.cwd() / f"{_log_prefix()}_{config_md5}_{sha8}.npz").resolve()


def _log_prefix() -> str:
    """Return the runtime log prefix for the current case runner."""
    return os.getenv("LINSTABPY_CASE_LOG_PREFIX", "linstabpy").strip() or "linstabpy"


def _log(message: str) -> None:
    """Print a consistent prefixed progress message."""
    if PETSc.COMM_WORLD.getRank() == 0:
        print(f"[{_log_prefix()}] {message}", flush=True)


def _stage_start() -> tuple[float, int]:
    """Capture timing and memory counters for a runtime stage."""
    return time.perf_counter(), _peak_rss_bytes()


def _log_stage(stage: str, stage_start: tuple[float, int]) -> None:
    """Emit a machine-parseable stage timing line."""
    start_time, rss0 = stage_start
    elapsed = time.perf_counter() - start_time
    _STAGE_WALL[stage] = _STAGE_WALL.get(stage, 0.0) + elapsed
    rss_delta = max(0, _peak_rss_bytes() - rss0)
    _log(f"stage complete: stage={stage} elapsed={elapsed:.2f}s rss_peak_delta_bytes={rss_delta}")


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
    """Read a dotted YAML path from the loaded runtime config."""
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
    provenance: dict[str, str]


def _stage_load_runtime_config() -> _RuntimeConfigSource:
    """Load the YAML runtime config and immutable provenance strings."""
    config_path = _runtime_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"case config file not found: {config_path}")
    config = load_config(config_path)
    from linstabpy.config import validate_config
    from linstabpy.config.schema import check_cli_ignored_keys

    validate_config(config, source=str(config_path))
    check_cli_ignored_keys(config, source=str(config_path))
    _log(f"loaded config: {config_path}")

    provenance = collect_provenance(config_path)
    return _RuntimeConfigSource(
        config_path=config_path,
        config=config,
        config_hash=_config_hash(config),
        git_sha=provenance["git_sha"],
        provenance=provenance,
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
    sponge_mat_path: Path | None,
    prefer_matlab_sponge_aux: bool,
    native_sponge_enabled: bool,
    sponge_cfg: dict | None,
    preprocess_cfg: dict | None,
    zero_mu_t_derivs: bool,
    heartbeat_seconds: float,
) -> _LoadedCase:
    """Load the base-flow snapshot and apply transport-property conventions."""
    stage_start = _stage_start()
    with _stage_heartbeat("load_case", interval_seconds=heartbeat_seconds):
        X, Y, baseflow, sponge, _info = load_case_fields(
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
            preprocess=preprocess_cfg,
        )
        if native_sponge_enabled and sponge_cfg is not None:
            from linstabpy.baseflow import sponge_from_config

            sponge = sponge_from_config(sponge.shape, sponge_cfg, X=X, Y=Y)
        elif prefer_matlab_sponge_aux and sponge_mat_path is not None:
            from linstabpy.baseflow.sponge import load_sponge_aux

            sponge_aux = load_sponge_aux(Path(sponge_mat_path))
            if sponge_aux.shape != sponge.shape:
                if sponge_aux.T.shape == sponge.shape:
                    sponge_aux = sponge_aux.T
                else:
                    raise ValueError(
                        "sponge_aux shape mismatch: "
                        f"{sponge_aux.shape} vs field shape {sponge.shape}"
                    )
            sponge = sponge_aux

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
    #: Nonzeros in the assembled stability operator (after BCs).
    #:
    #: Carried out of this stage rather than only written to the log line
    #: "operator assembled with BCs: shape=..., nnz=...".  It is a required
    #: field of the metrics sidecar's `problem` block, and because it was
    #: local to this function the sidecar shipped it missing -- so the gain-map
    #: sweep had to recover it by grepping the driver's own log output.  A
    #: number the cost study depends on should come from this stage, not parsed back out
    #: of prose (Front H).
    nnz: int


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
        # This rank holds a placeholder, so its nonzero count is 0 and is NOT
        # the operator's real nnz. Only rank 0 writes the metrics sidecar, and
        # rank 0 always takes the else-branch below, so the reported nnz is
        # always the measured one. Set here purely so the field exists on
        # every rank rather than raising NameError at the return.
        nnz = int(L0.nnz)
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
        nnz=nnz,
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
    idx: dict,
    config: dict,
    W_diag: np.ndarray,
    omega_target: complex,
    n_modes_resolvent: int,
    total_bc_indices: np.ndarray,
    solver_cfg: dict,
    cfg: Config | None = None,
    heartbeat_seconds: float,
    base_flow: Any | None = None,
) -> _ResolventStageResult:
    """Run resolvent analysis and its SVD postcheck for one target frequency."""
    ndof = mesh.ngp * 5
    if not run_resolvent:
        return _empty_resolvent_result(ndof)

    stage_start = _stage_start()
    _log(f"running resolvent: omega_target={omega_target:.8g}, modes={n_modes_resolvent}")
    # Reuse the resolvent_map region-masking logic so this single-point path
    # honours `input_region`/`output_region` and independent B/C the same way
    # the multi-point sweep does, instead of hand-rolling a BC-only mask that
    # silently ignores those config keys (B == C == BC mask when unset).  When
    # `input_region.space == "conservative_momentum"` this also needs the base
    # flow (rho_bar, u_bar, v_bar, w_bar, kappa, Ma); `base_flow=None` is fine
    # for every other region kind (the default primitive diagonal mask).
    B_mat, C_mat = _forcing_response_mats(
        L0_resolv,
        ndof,
        total_bc_indices,
        config,
        mesh=mesh,
        idx=idx,
        base_flow=base_flow,
    )
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
        # omega = eig(i*L0) = i*lambda for lambda = eig(L0), so the search
        # centre for L0 is -i*omega_target. The solver takes a scipy or a
        # PETSc matrix as it is.
        with _stage_heartbeat("stability", interval_seconds=heartbeat_seconds):
            lam, direct_modes, stability_info = _solve_near_target(
                L0,
                n_modes=n_modes,
                target=-1j * omega_target,
                solver_cfg=solver_cfg,
                cfg=cfg,
                return_info=True,
            )
        omega = 1j * lam
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
    if use_root_operator:
        # The residual needs the operator, the target distance does not.
        stability_info["postcheck_target_distance"] = _postcheck_target_distance(
            omega, omega_target, conjugate_aware=stability_acceptance_conjugate_aware
        )
    else:
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
    _log("starting compute")
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()

    config_source = _stage_load_runtime_config()
    output_npz = _output_npz_path(config_source.provenance)
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
    sponge_mat_raw = cfg("paths.case_inputs_mat", None)
    sponge_mat_path = Path(str(sponge_mat_raw)) if sponge_mat_raw is not None else None
    Re_raw = cfg("physics.Re", None)
    if Re_raw is None:
        raise ConfigValidationError(
            f"physics.Re is required: set it in the case file {config_source.config_path}."
        )
    Re = float(Re_raw)
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
    floquet_shift = cfg("numerics.floquet_shift", None)
    if floquet_shift is not None:
        # YAML over env, as for every other solver knob (setdefault let a stale env win).
        os.environ["LINSTABPY_FLOQUET_SHIFT"] = str(floquet_shift)
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
    # None means "decide from the base flow" once it is loaded (see below).
    _conjugate_aware_raw = cfg("analysis.stability_acceptance.conjugate_aware", None)

    n_modes = int(cfg("analysis.n_modes_stability", 100))
    n_modes_resolvent = int(cfg("analysis.n_modes_resolvent", 10))
    run_resolvent = bool(cfg("analysis.run_resolvent", True))
    run_stability = bool(cfg("analysis.run_stability", True))
    zero_spanwise = bool(cfg("physics.zero_spanwise", False))
    sponge_cfg_raw = cfg("physics.sponge", None)
    sponge_cfg = sponge_cfg_raw if isinstance(sponge_cfg_raw, dict) else None
    native_sponge_enabled = bool((sponge_cfg or {}).get("enabled", False))
    use_matlab_sponge_aux = bool(cfg("physics.use_matlab_sponge_aux", False))
    if (
        use_matlab_sponge_aux
        and not native_sponge_enabled
        and (sponge_mat_path is None or not sponge_mat_path.exists())
    ):
        raise ConfigValidationError(
            "physics.use_matlab_sponge_aux is true, but paths.case_inputs_mat "
            f"({sponge_mat_raw!r}) does not name an existing file."
        )
    alpha_filter_raw = cfg("numerics.alpha_filter", "none")
    # Honoured for every stage: the filter damps the SBP 2-dx mode in EVP-only runs too.
    alpha_filter = _parse_alpha_filter(alpha_filter_raw)
    od_raw = cfg("numerics.operator_dissipation", None)
    od_cfg = od_raw if isinstance(od_raw, dict) else {}
    od_enabled = bool(od_cfg.get("enabled", False))
    od_epsilon = float(od_cfg.get("epsilon", 0.0))
    od_order_diss = int(od_cfg.get("order_diss", fd_order))
    # Defaults live in mesh.sbp, not here: re-declaring them locally is what let the
    # stability/resolvent asymmetry drift apart unnoticed.
    od_apply_to_stability = bool(
        od_cfg.get("apply_to_stability", DISSIPATION_APPLY_DEFAULTS["apply_to_stability"])
    )
    od_apply_to_resolvent = bool(
        od_cfg.get("apply_to_resolvent", DISSIPATION_APPLY_DEFAULTS["apply_to_resolvent"])
    )
    od_transition_fraction = float(
        od_cfg.get("transition_fraction", DISSIPATION_DEFAULTS["transition_fraction"])
    )
    od_endpoint_exponent = float(
        od_cfg.get("endpoint_exponent", DISSIPATION_DEFAULTS["endpoint_exponent"])
    )
    od_interior_value = float(od_cfg.get("interior_value", DISSIPATION_DEFAULTS["interior_value"]))
    # Effective epsilon per analysis stage
    eps_stab = stage_dissipation_epsilon(od_cfg, "stability")
    eps_resolv = stage_dissipation_epsilon(od_cfg, "resolvent")
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
        (not native_sponge_enabled)
        and use_matlab_sponge_aux
        and sponge_mat_path is not None
        and sponge_mat_path.exists()
    )
    solver_cfg = config.get("solver")
    if solver_cfg is None:
        solver_cfg = {}
        config["solver"] = solver_cfg
    cfg_obj = Config.load(config)

    # Read solver knobs from YAML (env-var fallback for backward compat).
    _ecfg = solver_cfg.get("eigen", {})
    st_solver = cfg_obj.eigen.st_solver
    iterative_st_solver = st_solver in ITERATIVE_EIGEN_SOLVER_TYPES
    _root_raw = _first_env("LINSTABPY_CASE_ROOT_ASSEMBLY")
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
    if use_root_operator and np.isfinite(stability_acceptance_postcheck_rel_max):
        # Non-root ranks hold no operator, so the residual postcheck cannot run; without
        # this check every mode would be rejected with a "postcheck failed" code.
        raise ValueError(
            "analysis.stability_acceptance.postcheck_rel_max needs the residual postcheck, "
            "which root-only assembly skips. Set LINSTABPY_CASE_ROOT_ASSEMBLY=0 or drop the gate."
        )
    if use_root_operator:
        solver_cfg.setdefault("root_scatter", True)
        # The eigen solve reads cfg_obj, which is frozen; reload it so root_scatter applies.
        cfg_obj = Config.load(config)

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

    # Banner + preflight. Reports EFFECTIVE values, because the recurring failure mode is
    # a config that asks for something the driver then declines to do -- most of all the
    # per-stage dissipation, whose effective epsilon is what the operator actually sees.
    if rank == 0:
        for line in format_banner(
            {
                "case_dir": case_dir,
                "mpi_ranks": size,
                "stride (ksi, eta)": f"{stride_ksi}, {stride_eta}",
                "fd_order": fd_order,
                "beta": beta,
                "omega_target": omega_target,
                "stages": f"stability={run_stability} resolvent={run_resolvent}",
                "dissipation enabled": od_enabled,
                "epsilon (requested)": od_epsilon,
                "epsilon -> stability": eps_stab,
                "epsilon -> resolvent": eps_resolv,
                "st_solver": st_solver,
                "factor_backend": factor_backend_provenance,
                "git_sha": git_sha,
            }
        ).splitlines():
            _log(line)
        for line in format_advisories(
            config_advisories(config), title="preflight advisories"
        ).splitlines():
            _log(line)

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
        preprocess_cfg=cfg("preprocess", None),
        zero_mu_t_derivs=zero_mu_t_derivs,
        heartbeat_seconds=heartbeat_seconds,
    )
    X = loaded_case.X
    Y = loaded_case.Y
    baseflow = loaded_case.baseflow
    sponge = loaded_case.sponge
    sponge_enabled = loaded_case.sponge_enabled
    sponge_source = loaded_case.sponge_source
    # The spectrum is closed under omega -> -conj(omega) when L0 is real (beta = 0), and
    # also for beta != 0 when W = 0 (conjugation maps beta -> -beta, z-reflection maps it
    # back). A spanwise base velocity with beta != 0 breaks the pairing, so the default
    # turns conjugate-aware acceptance off there. Floquet or other complex formulations
    # must set it to False in the YAML.
    if _conjugate_aware_raw is None:
        stability_acceptance_conjugate_aware = not (beta != 0.0 and bool(np.any(baseflow.W)))
    else:
        stability_acceptance_conjugate_aware = bool(_conjugate_aware_raw)

    # Freestream normalisation gate. The canonical convention is rho = u = T = 1 in the
    # freestream; a base flow scaled any other way runs to completion and returns numbers
    # that mean nothing. Checked here, immediately after load, so it costs one pass over
    # the three fields and reports before any operator is built.
    if rank == 0:
        for line in format_advisories(
            baseflow_advisories(baseflow.RHO, baseflow.U, baseflow.T),
            title="base-flow normalisation",
        ).splitlines():
            _log(line)
        # The dissipation verdict needs the physical mesh, so it comes here and
        # not with the config advisories above.
        for line in format_advisories(
            mesh_advisories(X, Y, dissipation_enabled=od_enabled),
            title="mesh and dissipation",
        ).splitlines():
            _log(line)

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
    # 5) Build Chu energy norm weights. Weighted analyses need these for
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
        idx=idx,
        config=config,
        W_diag=W_diag,
        omega_target=omega_target,
        n_modes_resolvent=n_modes_resolvent,
        total_bc_indices=total_bc_indices,
        solver_cfg=solver_cfg,
        cfg=cfg_obj,
        heartbeat_seconds=heartbeat_seconds,
        base_flow=baseflow,
    )
    gains = resolvent_result.gains
    responses = resolvent_result.responses
    forcings = resolvent_result.forcings
    resolvent_rho_fwd = resolvent_result.rho_fwd
    resolvent_rho_adj = resolvent_result.rho_adj

    # ------------------------------------------------------------------
    # 7) Optional stability analysis.
    #    - raw_eigs: eig(L0) directly, reported as omega = i*lambda (diagnostic path,
    #      no BC reduction or weighted normalisation)
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
    # deadlocks every multi-rank run. The aggregated scalar goes into the
    # rank-0-only save (see save_results / collect_cost_metrics).
    _peak_rss_total_bytes = MPI.COMM_WORLD.allreduce(int(_peak_rss_bytes()), op=MPI.SUM)
    # The sum answers how much memory the job took on the node. The maximum
    # answers how much one rank needs, which is what sizing a run asks for.
    _peak_rss_max_bytes = MPI.COMM_WORLD.allreduce(int(_peak_rss_bytes()), op=MPI.MAX)
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

        # Closing summary: what came out, and whether to trust it. This belongs in the
        # LOG and not only in the NPZ -- reading verdicts back out of artifacts with
        # ad-hoc scripts is precisely the cost this layer removes.
        _post = result_advisories(
            gains=gains if np.size(gains) else None,
            residuals=[
                r
                for r in (
                    np.max(resolvent_rho_fwd) if np.size(resolvent_rho_fwd) else None,
                    np.max(resolvent_rho_adj) if np.size(resolvent_rho_adj) else None,
                )
                if r is not None
            ]
            or None,
            dissipation_effective=bool(eps_resolv > 0.0) if run_resolvent else None,
        )
        for line in format_summary(
            {
                "ndof": mesh.ngp * 5,
                "leading gain": f"{float(np.max(gains)):.6g}" if np.size(gains) else "n/a",
                "eigenvalues returned": int(np.size(omega)),
                "resolvent residual (fwd, adj)": (
                    f"{float(np.max(resolvent_rho_fwd)):.3e}, "
                    f"{float(np.max(resolvent_rho_adj)):.3e}"
                    if np.size(resolvent_rho_fwd) and np.size(resolvent_rho_adj)
                    else "n/a"
                ),
                "epsilon -> resolvent (effective)": eps_resolv,
                "peak RSS [bytes]": _peak_rss_total_bytes,
                "peak RSS per rank, max [bytes]": _peak_rss_max_bytes,
            },
            _post,
        ).splitlines():
            _log(line)

        from linstabpy.driver.io import save_results

        payload = build_runtime_payload(
            spectra={
                "omega": omega,
                "direct_modes": direct_modes,
                "gains": gains,
                "responses": responses,
                "forcings": forcings,
                "resolvent_rho_fwd": resolvent_rho_fwd,
                "resolvent_rho_adj": resolvent_rho_adj,
                "omega_target": omega_target,
                "use_raw_eigs": use_raw_eigs,
                "St": St,
                "beta": beta,
                "fd_order": fd_order,
            },
            problem={
                "case": case_dir.name,
                "nx": int(mesh.X.shape[1]),
                "ny": int(mesh.X.shape[0]),
                "ndof": int(mesh.ngp * 5),
                "nnz": int(assembly_result.nnz),
                "lambda_over_L": float(2.0 * np.pi / beta) if beta != 0.0 else float("inf"),
                "stride": int(stride_ksi) if stride_ksi == stride_eta else -1,
                "stride_ksi": int(stride_ksi),
                "stride_eta": int(stride_eta),
                "alpha_filter": np.array(str(alpha_filter)),
                "use_matlab_sponge_aux": use_matlab_sponge_aux,
                "sponge_enabled": sponge_enabled,
                "sponge_source": np.array(sponge_source),
                "sponge_preset": np.array(str((sponge_cfg or {}).get("preset", ""))),
                "native_sponge_requested": native_sponge_enabled,
                "matlab_sponge_aux_requested": use_matlab_sponge_aux,
                "n_modes_stability_requested": n_modes,
                "n_modes_resolvent_requested": n_modes_resolvent,
                "Re": Re,
                "Ma": Ma,
                "Pr": Pr,
                "kappa": kappa,
                "T_0": T_0,
            },
            fields={
                "X": mesh.X,
                "Y": mesh.Y,
                "U": baseflow.U,
                "sponge": mesh.sponge,
                "idx_rho_j": idx["rho_j"],
                "idx_u_j": idx["u_j"],
                "idx_v_j": idx["v_j"],
                "idx_w_j": idx["w_j"],
                "idx_T_j": idx["T_j"],
                "bc_indices": bc_indices,
                "bc_total_indices": total_bc_indices,
                "bc_effective_dirichlet_indices": effective_dirichlet_indices,
                "bc_column_zeroed_dirichlet_indices": column_zeroed_dirichlet_indices,
                "bc_effective_neumann_indices": effective_neumann_indices,
                "bc_overlap_indices": overlap_indices,
                "W_diag": W_diag,
                "invW_diag": invW_diag,
            },
            stability={
                "stability_eps_nev": int(stability_info.get("nev_requested", -1)),
                "stability_eps_ncv": int(stability_info.get("ncv_used") or -1),
                "stability_eps_nconv": int(stability_info.get("nconv", -1)),
                "stability_eps_iterations": int(stability_info.get("eps_iterations", -1)),
                "stability_eps_reason": np.array(str(stability_info.get("eps_reason", ""))),
                "stability_ksp_reason": np.array(str(stability_info.get("ksp_reason", ""))),
                "stability_eps_conv_test": np.array(str(stability_info.get("conv_test", ""))),
                "stability_eps_tol": float(stability_info.get("eps_tol") or np.nan),
                "stability_eps_max_it": int(stability_info.get("eps_max_it") or -1),
                "stability_eps_setup_seconds": float(
                    stability_info.get("eps_setup_seconds", np.nan)
                ),
                "stability_eps_solve_seconds": float(
                    stability_info.get("eps_solve_seconds", np.nan)
                ),
                "stability_true_residual": bool(stability_info.get("true_residual", False)),
                "stability_reduce_dirichlet": bool(stability_info.get("reduce_dirichlet", False)),
                "stability_reduce_dirichlet_set": np.array(
                    str(stability_info.get("reduce_dirichlet_set", ""))
                ),
                "stability_free_dof_count": int(stability_info.get("free_dof_count", -1)),
                "stability_constrained_dirichlet_count": int(
                    stability_info.get("constrained_dirichlet_count", -1)
                ),
                "stability_residual_error_type": np.array(
                    str(stability_info.get("residual_error_type", ""))
                ),
                "stability_residual_estimates": stability_residual_estimates,
                "stability_residual_absolute": stability_residual_absolute,
                "stability_residual_relative": stability_residual_relative,
                "stability_residual_backward": stability_residual_backward,
                "stability_postcheck_residual_abs": stability_postcheck_residual_abs,
                "stability_postcheck_residual_rel": stability_postcheck_residual_rel,
                "stability_postcheck_residual_interior_abs": (
                    stability_postcheck_residual_interior_abs
                ),
                "stability_postcheck_residual_bc_abs": stability_postcheck_residual_bc_abs,
                "stability_postcheck_bc_leak_max": stability_postcheck_bc_leak_max,
                "stability_postcheck_bc_leak_p99": stability_postcheck_bc_leak_p99,
                "stability_postcheck_dirichlet_leak_max": stability_postcheck_dirichlet_leak_max,
                "stability_postcheck_dirichlet_leak_p99": stability_postcheck_dirichlet_leak_p99,
                "stability_postcheck_target_distance": stability_postcheck_target_distance,
                "stability_all_omega": np.asarray(omega, dtype=complex),
                "stability_all_postcheck_rel": stability_postcheck_residual_rel,
                "stability_all_target_distance": stability_postcheck_target_distance,
                "stability_all_reject_reason_code": np.asarray(
                    stability_acceptance.get("reject_reason_code", EMPTY_INT32),
                    dtype=np.int32,
                ),
                "stability_acceptance_postcheck_rel_max": float(
                    stability_acceptance_postcheck_rel_max
                ),
                "stability_acceptance_target_distance_max": float(
                    stability_acceptance_target_distance_max
                ),
                "stability_acceptance_omega_real_abs_max": float(
                    stability_acceptance_omega_real_abs_max
                ),
                "stability_acceptance_omega_imag_abs_max": float(
                    stability_acceptance_omega_imag_abs_max
                ),
                "stability_acceptance_conjugate_aware": bool(stability_acceptance_conjugate_aware),
                "stability_accepted_count": int(stability_acceptance.get("accepted_count", -1)),
                "stability_accepted_indices": np.asarray(
                    stability_acceptance.get("accepted_indices", EMPTY_INT32),
                    dtype=np.int32,
                ),
                "stability_rejected_indices": np.asarray(
                    stability_acceptance.get("rejected_indices", EMPTY_INT32),
                    dtype=np.int32,
                ),
                "stability_accepted_postcheck_rel_min": float(
                    stability_acceptance.get("accepted_postcheck_rel_min", np.nan)
                ),
                "stability_accepted_postcheck_rel_median": float(
                    stability_acceptance.get("accepted_postcheck_rel_median", np.nan)
                ),
                "stability_accepted_target_distance_min": float(
                    stability_acceptance.get("accepted_target_distance_min", np.nan)
                ),
                "stability_accepted_target_distance_median": float(
                    stability_acceptance.get("accepted_target_distance_median", np.nan)
                ),
                "stability_accepted_omega_real_min": float(
                    stability_acceptance.get("accepted_omega_real_min", np.nan)
                ),
                "stability_accepted_omega_real_max": float(
                    stability_acceptance.get("accepted_omega_real_max", np.nan)
                ),
                "stability_accepted_omega_imag_min": float(
                    stability_acceptance.get("accepted_omega_imag_min", np.nan)
                ),
                "stability_accepted_omega_imag_max": float(
                    stability_acceptance.get("accepted_omega_imag_max", np.nan)
                ),
                "stability_leading_weighted_norm_error": float(
                    stability_info.get("leading_weighted_norm_error", np.nan)
                ),
                "stability_weighted_norm_tol": float(
                    stability_info.get("weighted_norm_tol", np.nan)
                ),
                "stability_missing_conjugate_partner_indices": np.asarray(
                    _conj_diag["missing_partner_indices"], dtype=np.int32
                ),
                "stability_partner_rejected_indices": np.asarray(
                    _conj_diag["partner_rejected_indices"], dtype=np.int32
                ),
            },
            dissipation={
                "numerics_operator_dissipation_enabled": np.bool_(od_enabled),
                "numerics_operator_dissipation_epsilon": np.float64(od_epsilon),
                "numerics_operator_dissipation_order_diss": np.int64(od_order_diss),
                "numerics_operator_dissipation_apply_to_stability": np.bool_(od_apply_to_stability),
                "numerics_operator_dissipation_apply_to_resolvent": np.bool_(od_apply_to_resolvent),
                "numerics_operator_dissipation_transition_fraction": np.float64(
                    od_transition_fraction
                ),
                "numerics_operator_dissipation_endpoint_exponent": np.float64(od_endpoint_exponent),
                "numerics_operator_dissipation_interior_value": np.float64(od_interior_value),
            },
            provenance={
                "solver_eigen_st_solver_used": str(st_solver),
                "solver_apply_to_resolvent_effective": np.bool_(eps_resolv > 0.0),
                "provenance_config_hash": np.bytes_(config_hash[:16].encode()),
                "provenance_linstabpy_git_sha": np.bytes_(git_sha.encode()),
                "resolvent_sign_convention": np.array(cfg_obj.resolvent.sign_convention),
                "provenance_factor_backend": str(factor_backend_provenance),
                "provenance_factor_precision": np.array(
                    cfg_obj.eigen.factor_precision
                    if run_stability
                    else cfg_obj.resolvent.linear.factor_precision
                ),
            },
        )
        save_results(
            output_npz=output_npz,
            heartbeat_seconds=heartbeat_seconds,
            peak_rss_bytes=_peak_rss_total_bytes,
            stage_wall=_STAGE_WALL,
            provenance=config_source.provenance,
            payload=payload,
        )
    elapsed = time.perf_counter() - t0
    _log(f"results saved to {output_npz}")
    _log(f"compute finished in {elapsed:.2f}s")
    comm.barrier()
