# SPDX-License-Identifier: GPL-3.0-only
"""LNSE operator assembly from coefficients and FD matrices.

Translates GetLinProblem.m: assembles the 5*ngp × 5*ngp linear operator
L0 such that dq/dt = L0 @ q, with eig(i*L0) = omega_code = -omega_OS.
Unstable modes have Im(omega_code) < 0.  State: q = [rho, u, v, w, T].

PETSc assembly sequencing
-------------------------
The PETSc operator pipeline applies transformations in a fixed order that
cannot be permuted.  The canonical sequence inside
``build_lnse_operator_petsc`` is:

1. ``_assemble_block_operator_petsc`` — builds the raw LNSE block matrix
   L_raw from coefficient arrays and FD stencil matrices.
2. ``_apply_inv_B_petsc`` — applies the analytical ``inv(B/1j)`` left
   scaling row-by-row, converting L_raw to the B-preconditioned form
   L0 = inv(B/1j) @ L_raw.  This step must come before the sponge and
   artificial dissipation additions, which add to L0 rather than to L_raw.
3. Sponge layer addition — adds the diagonal sponge damping to L0.
4. Artificial dissipation addition — adds the FD dissipation operator to L0.
5. ``apply_bc_recipe`` / ``apply_bc_recipe_petsc`` — overwrites boundary
   rows of L0 with the chosen boundary conditions.  This must be the
   last step: BC rows must not be touched by subsequent sponge or
   dissipation additions.

This ordering is non-interchangeable.  Applying BCs before inv(B) scaling
would corrupt the BC rows; applying inv(B) after BCs would overwrite them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from linstabpy.config import Config

from ..baseflow.baseflow import BaseFlow
from ..mesh.mesh import Mesh
from ..utils.petsc_core import _matrix_nnz
from .coefficients import (
    LNSECoefficients,
    compute_cartesian_coefficients,
    compute_cylindrical_coefficients,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only optional dependency
    from petsc4py import PETSc


from .metrics import (
    _COEFF_TENSOR_NAMES,
    _as_csr_derivative_cache,
    _assembly_metrics_session,
    _assembly_stage,
    _cache_5_block_derivatives,
    _CSRDerivativeCache,
    _note_assembly_metadata,
)
from .metrics import (
    AssemblyMetrics as AssemblyMetrics,
)
from .metrics import (
    _AssemblyMetricsCollector as _AssemblyMetricsCollector,
)
from .metrics import (
    _make_csr_derivative_cache as _make_csr_derivative_cache,
)


@dataclass(frozen=True)
class _LNSEAssemblyRequest:
    """Normalized input bundle for the LNSE system assembly workflow.

    Keep workflow policy here rather than spreading backend flags across
    runtime callers.  The lower-level ``build_lnse_operator*`` functions remain
    pre-BC builders; this request represents the full system path that may also
    apply boundary conditions and backend conversion.
    """

    mesh: Mesh
    baseflow: BaseFlow
    kx: float = 0.0
    model: str = "2D"
    m: float = 0.0
    floquet_exp: float = 0.0
    dissipation_epsilon: float = 0.0
    bc_recipe: Any = None
    bc_penalty: float = 1e4
    backend: str | None = None
    solver_cfg: dict | None = None

    @property
    def model_name(self) -> str:
        return str(self.model).lower()

    @property
    def has_bc_recipe(self) -> bool:
        return bool(self.bc_recipe)


@dataclass(frozen=True)
class LNSESystem:
    """Assembled LNSE operator plus the metadata needed by downstream solvers.

    The compatibility wrapper still returns ``(operator, idx, bc_indices)``.
    New runtime code should keep this object intact so backend decisions,
    effective BC ownership, and future secondary operators stay traceable.
    """

    operator: sp.csc_matrix | "PETSc.Mat"
    idx: dict
    bc_indices: np.ndarray
    bc_analysis: dict[str, Any] | None
    backend_metadata: dict[str, Any]
    resolvent_operator_data: dict[str, Any] | None = None


_LNSEAssemblyResult = LNSESystem


@dataclass(frozen=True)
class _LNSEBackendDecision:
    """Resolved backend choice plus the reason for any fallback path."""

    requested_backend: str
    resolved_backend: str
    distributed_requested: bool
    direct_petsc_supported: bool
    use_petsc_fallback: bool
    reason: str

    def as_metadata(self) -> dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "distributed_requested": self.distributed_requested,
            "direct_petsc_supported": self.direct_petsc_supported,
            "use_petsc_fallback": self.use_petsc_fallback,
            "backend_reason": self.reason,
        }


def _record_prepare_derivative_metrics(
    stage: dict[str, Any],
    derivatives: dict[str, np.ndarray],
) -> None:
    """Record materialized derivative-vector cost for P3 kernel work.

    The current coefficient API receives derivative arrays, so
    ``prepare_coefficients`` must keep these vectors alive until the coefficient
    tensors have been written.  P3 reductions should lower these counters, not
    just move the allocations to a different line.
    """
    bytes_by_name = {name: int(np.asarray(value).nbytes) for name, value in derivatives.items()}
    total_bytes = int(sum(bytes_by_name.values()))
    stage["derivative_vector_count"] = int(len(bytes_by_name))
    stage["derivative_vector_bytes_by_name"] = bytes_by_name
    stage["derivative_vector_bytes_total"] = total_bytes


def _recorded_derivative_matvec(
    stage: dict[str, Any],
    name: str,
    operator: sp.spmatrix,
    values: np.ndarray,
) -> np.ndarray:
    """Apply one sparse derivative operator and record the materialized matvec."""
    names = stage.setdefault("derivative_matvec_names", [])
    names.append(name)
    stage["derivative_matvec_count"] = int(len(names))
    return operator @ values


def build_lnse_operator(
    mesh: Mesh,
    baseflow: BaseFlow,
    kx: float = 0.0,
    model: str = "2D",
    m: float = 0,
    floquet_exp: float = 0.0,
    dissipation_epsilon: float = 0.0,
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
    dissipation_epsilon : float
        Artificial dissipation strength. When positive and mesh dissipation
        matrices are available, subtracts epsilon times the 5-variable block
        dissipation operator from the pre-BC LNSE operator.

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
        if dissipation_epsilon > 0.0 and mesh.Q_diss_x is not None and mesh.Q_diss_y is not None:
            # Artificial dissipation is an L0-level correction. It belongs
            # after inv(B) and sponge, but before any BC recipe overwrites rows.
            ny, nx = mesh.X.shape
            I_ny = sp.eye(ny, format="csr")
            I_nx = sp.eye(nx, format="csr")
            Q_diss_2D = sp.kron(mesh.Q_diss_x, I_ny, format="csr") + sp.kron(
                I_nx, mesh.Q_diss_y, format="csr"
            )
            Q_diss_block = sp.block_diag([Q_diss_2D] * 5, format="csr")
            L0 = (L0 - dissipation_epsilon * Q_diss_block).tocsc()
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
    dissipation_epsilon: float = 0.0,
    solver_cfg: dict | None = None,
) -> tuple["PETSc.Mat", dict]:
    """Build the pre-BC LNSE operator L0 directly as a PETSc AIJ matrix.

    This prototype preserves the existing coefficient-preparation path but avoids
    the final SciPy-assembled global matrix. It currently targets the same
    pre-BC operator returned by :func:`build_lnse_operator`. Boundary conditions
    are intentionally outside this function so the same post-operator ordering
    is used by SciPy, PETSc, and fallback paths.
    """
    if dissipation_epsilon != 0.0:
        # There is no validated PETSc-side Q_diss insertion yet. Falling back
        # through the system builder keeps the established SciPy ordering rather
        # than assembling a PETSc operator with a missing numerical term.
        raise NotImplementedError(
            "Direct PETSc backend does not yet support dissipation_epsilon. "
            'Use backend="auto" or "scipy", or implement Q_diss subtraction '
            "in build_lnse_operator_petsc()."
        )

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
        Config.load(solver_cfg if solver_cfg is not None else {}).mesh.assembly_backend
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


def _decide_lnse_backend(request: _LNSEAssemblyRequest) -> _LNSEBackendDecision:
    """Resolve backend selection once and keep the fallback reason inspectable."""
    from ..utils.petsc_core import _use_distributed_matrix

    requested_backend = _requested_lnse_backend(request.backend, solver_cfg=request.solver_cfg)
    resolved_backend = _resolve_lnse_backend(
        request.backend, model=request.model, solver_cfg=request.solver_cfg
    )
    distributed_requested = _use_distributed_matrix(request.solver_cfg)
    direct_supported = _direct_petsc_backend_supported(
        model=request.model, solver_cfg=request.solver_cfg
    )
    use_petsc_fallback = (
        requested_backend in {"petsc", "auto"}
        and distributed_requested
        and resolved_backend == "scipy"
    )

    # Backend selection is part of the provenance. Avoid returning only
    # "scipy"/"petsc"; callers and metrics need to know whether PETSc was a
    # true direct assembly path or a SciPy assembly converted for MPI ownership.
    if requested_backend == "scipy":
        reason = "requested_scipy"
    elif requested_backend == "petsc":
        reason = "requested_petsc" if direct_supported else "requested_petsc_direct_unsupported"
    elif direct_supported:
        reason = "auto_direct_petsc_supported"
    else:
        reason = "auto_fallback_scipy_direct_petsc_unsupported"
    if use_petsc_fallback:
        reason = f"{reason}_distributed_conversion"

    return _LNSEBackendDecision(
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        distributed_requested=distributed_requested,
        direct_petsc_supported=direct_supported,
        use_petsc_fallback=use_petsc_fallback,
        reason=reason,
    )


def _build_lnse_system_scipy(
    mesh: Mesh,
    baseflow: BaseFlow,
    *,
    kx: float = 0.0,
    model: str = "2D",
    m: float = 0,
    floquet_exp: float = 0.0,
    dissipation_epsilon: float = 0.0,
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
        dissipation_epsilon=dissipation_epsilon,
    )
    with _assembly_stage("apply_bc") as stage:
        if bc_recipe:
            # BC application is the final mutation in the operator pipeline.
            # Applying it earlier would let inv(B), sponge, or dissipation alter
            # boundary rows that should be exact replacement equations.
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


def _build_lnse_system_from_request(request: _LNSEAssemblyRequest) -> LNSESystem:
    """Build an LNSE system from the normalized internal request object."""
    from ..utils.petsc_core import compact_petsc_matrix, scipy_to_petsc

    backend_decision = _decide_lnse_backend(request)
    backend_metadata = backend_decision.as_metadata()

    with _assembly_metrics_session(
        solver_cfg=request.solver_cfg,
        path_kind="system",
        requested_backend=backend_decision.requested_backend,
        resolved_backend=backend_decision.resolved_backend,
        distributed_requested=backend_decision.distributed_requested,
        direct_petsc_supported=backend_decision.direct_petsc_supported,
        use_petsc_fallback=backend_decision.use_petsc_fallback,
        backend_reason=backend_decision.reason,
        model=request.model_name,
        kx=float(request.kx),
        m=float(request.m),
        floquet_exp=float(request.floquet_exp),
        bc_recipe_enabled=request.has_bc_recipe,
    ):
        if backend_decision.use_petsc_fallback:
            # Distributed PETSc callers sometimes request a PETSc-owned matrix
            # for a case whose direct assembly path is not validated. Assemble
            # with the SciPy reference path first, then convert, so numerical
            # ordering remains identical to the baseline.
            L0_sp, idx, bc_indices = _build_lnse_system_scipy(
                request.mesh,
                request.baseflow,
                kx=request.kx,
                model=request.model,
                m=request.m,
                floquet_exp=request.floquet_exp,
                dissipation_epsilon=request.dissipation_epsilon,
                bc_recipe=request.bc_recipe,
                bc_penalty=request.bc_penalty,
            )
            nnz_post_bc = _matrix_nnz(L0_sp)
            with _assembly_stage("scipy_to_petsc") as stage:
                L0 = scipy_to_petsc(L0_sp, request.solver_cfg)
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
            return LNSESystem(
                operator=L0,
                idx=idx,
                bc_indices=bc_indices,
                bc_analysis=_analyze_lnse_bc_recipe(request, idx),
                backend_metadata={**backend_metadata, "matrix_type": "petsc"},
            )

        if backend_decision.resolved_backend == "petsc":
            from ..boundary_conditions.recipe import apply_bc_recipe_petsc

            # This is the true direct-PETSc path: raw operator, inv(B), sponge,
            # then PETSc BC rows. Keep this path separate from the conversion
            # fallback so benchmarks can tell which assembly route ran.
            L0, idx = build_lnse_operator_petsc(
                request.mesh,
                request.baseflow,
                kx=request.kx,
                model=request.model,
                m=request.m,
                floquet_exp=request.floquet_exp,
                dissipation_epsilon=request.dissipation_epsilon,
                solver_cfg=request.solver_cfg,
            )
            with _assembly_stage("apply_bc") as stage:
                if request.bc_recipe:
                    # Same final-step rule as SciPy: PETSc BC rows are applied
                    # only after every L0-level operator contribution is present.
                    L0, bc_indices = apply_bc_recipe_petsc(
                        L0,
                        request.mesh,
                        idx,
                        request.bc_recipe,
                        penalty=request.bc_penalty,
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
                if (
                    not backend_decision.distributed_requested
                    and backend_decision.direct_petsc_supported
                ):
                    L0 = compact_petsc_matrix(L0, request.solver_cfg)
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
            return LNSESystem(
                operator=L0,
                idx=idx,
                bc_indices=bc_indices,
                bc_analysis=_analyze_lnse_bc_recipe(request, idx),
                backend_metadata={**backend_metadata, "matrix_type": "petsc"},
            )

        with _assembly_stage("scipy_to_petsc") as stage:
            stage["skipped"] = True
        with _assembly_stage("petsc_compaction") as stage:
            stage["skipped"] = True
        # Default SciPy system path. It remains the reference for parity tests
        # and for backend fallbacks that need unsupported numerical terms.
        L0, idx, bc_indices = _build_lnse_system_scipy(
            request.mesh,
            request.baseflow,
            kx=request.kx,
            model=request.model,
            m=request.m,
            floquet_exp=request.floquet_exp,
            dissipation_epsilon=request.dissipation_epsilon,
            bc_recipe=request.bc_recipe,
            bc_penalty=request.bc_penalty,
        )
        _note_assembly_metadata(
            nnz_post_bc=_matrix_nnz(L0),
            nnz_final=_matrix_nnz(L0),
            nnz=_matrix_nnz(L0),
            bc_indices_count=int(bc_indices.size),
            matrix_type="scipy",
        )
        return LNSESystem(
            operator=L0,
            idx=idx,
            bc_indices=bc_indices,
            bc_analysis=_analyze_lnse_bc_recipe(request, idx),
            backend_metadata={**backend_metadata, "matrix_type": "scipy"},
        )


def _analyze_lnse_bc_recipe(
    request: _LNSEAssemblyRequest,
    idx: dict,
) -> dict[str, Any] | None:
    if not request.bc_recipe:
        return None
    from ..boundary_conditions.recipe import analyze_bc_recipe

    return analyze_bc_recipe(request.mesh, idx, request.bc_recipe)


def build_lnse_system_result(
    mesh: Mesh,
    baseflow: BaseFlow,
    *,
    kx: float = 0.0,
    model: str = "2D",
    m: float = 0,
    floquet_exp: float = 0.0,
    dissipation_epsilon: float = 0.0,
    bc_recipe: list[dict] | None = None,
    bc_penalty: float = 1e4,
    backend: str | None = None,
    solver_cfg: dict | None = None,
) -> LNSESystem:
    """Build an LNSE system and return the operator with assembly metadata.

    This workflow helper keeps the existing SciPy path as the default while
    allowing direct-PETSc assembly/BC application when ``backend='petsc'``.
    In distributed PETSc mode, unsupported direct-assembly cases use the
    validated SciPy assembly + SciPy->PETSc conversion fallback. That fallback
    returns PETSc matrices, but it is not true distributed direct assembly.
    """
    request = _LNSEAssemblyRequest(
        mesh=mesh,
        baseflow=baseflow,
        kx=kx,
        model=model,
        m=m,
        floquet_exp=floquet_exp,
        dissipation_epsilon=dissipation_epsilon,
        bc_recipe=bc_recipe,
        bc_penalty=bc_penalty,
        backend=backend,
        solver_cfg=solver_cfg,
    )
    return _build_lnse_system_from_request(request)


def build_lnse_system(
    mesh: Mesh,
    baseflow: BaseFlow,
    *,
    kx: float = 0.0,
    model: str = "2D",
    m: float = 0,
    floquet_exp: float = 0.0,
    dissipation_epsilon: float = 0.0,
    bc_recipe: list[dict] | None = None,
    bc_penalty: float = 1e4,
    backend: str | None = None,
    solver_cfg: dict | None = None,
) -> tuple[sp.csc_matrix | "PETSc.Mat", dict, np.ndarray]:
    """Build an LNSE operator and optionally apply BCs with one backend selector.

    This compatibility wrapper preserves the historical tuple return. New
    internal callers should use :func:`build_lnse_system_result` so backend
    metadata and BC analysis stay attached to the assembled operator.
    """
    result = build_lnse_system_result(
        mesh,
        baseflow,
        kx=kx,
        model=model,
        m=m,
        floquet_exp=floquet_exp,
        dissipation_epsilon=dissipation_epsilon,
        bc_recipe=bc_recipe,
        bc_penalty=bc_penalty,
        backend=backend,
        solver_cfg=solver_cfg,
    )
    return result.operator, result.idx, result.bc_indices


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
        unused_derivative = np.empty(0, dtype=float)
        # MATLAB 2D path uses Dzr = Dz*Dr (GetLinProblem.m: D2RZ=blkdiag(Dzr)).
        # Dz*Dr applied to a vector applies Dr first (r-direction=y), then Dz
        # (z-direction=x): that is Dx @ Dy = mesh.Dxy, not Dy @ Dx = mesh.Dyx.
        D2RZ = mesh.Dxy

        model = model.lower()
        if model == "2d":
            dUdx = _recorded_derivative_matvec(stage, "dUdx", Dx, U)
            dUdy = _recorded_derivative_matvec(stage, "dUdy", Dy, U)
            dVdx = _recorded_derivative_matvec(stage, "dVdx", Dx, V)
            dVdy = _recorded_derivative_matvec(stage, "dVdy", Dy, V)
            dWdx = _recorded_derivative_matvec(stage, "dWdx", Dx, W)
            dWdy = _recorded_derivative_matvec(stage, "dWdy", Dy, W)
            dRHOdx = _recorded_derivative_matvec(stage, "dRHOdx", Dx, RHO)
            dRHOdy = _recorded_derivative_matvec(stage, "dRHOdy", Dy, RHO)
            dTdx = _recorded_derivative_matvec(stage, "dTdx", Dx, T)
            dTdy = _recorded_derivative_matvec(stage, "dTdy", Dy, T)
            dPdx = _recorded_derivative_matvec(stage, "dPdx", Dx, P)
            dPdy = _recorded_derivative_matvec(stage, "dPdy", Dy, P)

            d2Udx2 = _recorded_derivative_matvec(stage, "d2Udx2", D2x, U)
            d2Udy2 = _recorded_derivative_matvec(stage, "d2Udy2", D2y, U)
            d2Vdx2 = _recorded_derivative_matvec(stage, "d2Vdx2", D2x, V)
            d2Vdy2 = _recorded_derivative_matvec(stage, "d2Vdy2", D2y, V)
            d2Wdx2 = _recorded_derivative_matvec(stage, "d2Wdx2", D2x, W)
            d2Wdy2 = _recorded_derivative_matvec(stage, "d2Wdy2", D2y, W)
            d2Tdx2 = _recorded_derivative_matvec(stage, "d2Tdx2", D2x, T)
            d2Tdy2 = _recorded_derivative_matvec(stage, "d2Tdy2", D2y, T)

            d2Udxdy = _recorded_derivative_matvec(stage, "d2Udxdy", Dx, dUdy)
            d2Vdxdy = _recorded_derivative_matvec(stage, "d2Vdxdy", Dx, dVdy)

            _record_prepare_derivative_metrics(
                stage,
                {
                    "dUdx": dUdx,
                    "dUdy": dUdy,
                    "dVdx": dVdx,
                    "dVdy": dVdy,
                    "dWdx": dWdx,
                    "dWdy": dWdy,
                    "dRHOdx": dRHOdx,
                    "dRHOdy": dRHOdy,
                    "dTdx": dTdx,
                    "dTdy": dTdy,
                    "dPdx": dPdx,
                    "dPdy": dPdy,
                    "d2Udx2": d2Udx2,
                    "d2Udy2": d2Udy2,
                    "d2Vdx2": d2Vdx2,
                    "d2Vdy2": d2Vdy2,
                    "d2Wdx2": d2Wdx2,
                    "d2Wdy2": d2Wdy2,
                    "d2Tdx2": d2Tdx2,
                    "d2Tdy2": d2Tdy2,
                    "d2Udxdy": d2Udxdy,
                    "d2Vdxdy": d2Vdxdy,
                },
            )

            coeffs = compute_cartesian_coefficients(
                U=U,
                V=V,
                W=W,
                RHO=RHO,
                T=T,
                MU=MU,
                P=P,
                kx=kx,
                Re=Re,
                Ma=Ma,
                Pr=Pr,
                kappa=kappa,
                dUdx=dUdx,
                dVdx=dVdx,
                dWdx=dWdx,
                dRHOdx=dRHOdx,
                dTdx=dTdx,
                dPdx=dPdx,
                dUdy=dUdy,
                dVdy=dVdy,
                dWdy=dWdy,
                dRHOdy=dRHOdy,
                dTdy=dTdy,
                dPdy=dPdy,
                d2Udx2=d2Udx2,
                d2Vdx2=d2Vdx2,
                d2Wdx2=d2Wdx2,
                d2Udy2=d2Udy2,
                d2Vdy2=d2Vdy2,
                d2Wdy2=d2Wdy2,
                d2Tdx2=d2Tdx2,
                d2Tdy2=d2Tdy2,
                d2Udxdy=d2Udxdy,
                d2Vdxdy=d2Vdxdy,
                d2Wdxdy=unused_derivative,
                dmudT=dmudT,
                d2mudT2=d2mudT2,
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

            dUdr = _recorded_derivative_matvec(stage, "dUdr", Dy, U_c)
            dVdr = _recorded_derivative_matvec(stage, "dVdr", Dy, V_c)
            dWdr = _recorded_derivative_matvec(stage, "dWdr", Dy, W_c)
            dRHOdr = _recorded_derivative_matvec(stage, "dRHOdr", Dy, RHO)
            dTdr = _recorded_derivative_matvec(stage, "dTdr", Dy, T)
            dUdz = _recorded_derivative_matvec(stage, "dUdz", Dx, U_c)
            dVdz = _recorded_derivative_matvec(stage, "dVdz", Dx, V_c)
            dWdz = _recorded_derivative_matvec(stage, "dWdz", Dx, W_c)
            dRHOdz = _recorded_derivative_matvec(stage, "dRHOdz", Dx, RHO)
            dTdz = _recorded_derivative_matvec(stage, "dTdz", Dx, T)

            d2Udr2 = _recorded_derivative_matvec(stage, "d2Udr2", D2y, U_c)
            d2Vdr2 = _recorded_derivative_matvec(stage, "d2Vdr2", D2y, V_c)
            d2Wdr2 = _recorded_derivative_matvec(stage, "d2Wdr2", D2y, W_c)
            d2Udz2 = _recorded_derivative_matvec(stage, "d2Udz2", D2x, U_c)
            d2Vdz2 = _recorded_derivative_matvec(stage, "d2Vdz2", D2x, V_c)
            d2Wdz2 = _recorded_derivative_matvec(stage, "d2Wdz2", D2x, W_c)

            d2Udrz = _recorded_derivative_matvec(stage, "d2Udrz", Dx, dUdr)
            d2Wdrz = _recorded_derivative_matvec(stage, "d2Wdrz", Dx, dWdr)

            _record_prepare_derivative_metrics(
                stage,
                {
                    "dUdr": dUdr,
                    "dVdr": dVdr,
                    "dWdr": dWdr,
                    "dRHOdr": dRHOdr,
                    "dTdr": dTdr,
                    "dUdz": dUdz,
                    "dVdz": dVdz,
                    "dWdz": dWdz,
                    "dRHOdz": dRHOdz,
                    "dTdz": dTdz,
                    "d2Udr2": d2Udr2,
                    "d2Vdr2": d2Vdr2,
                    "d2Wdr2": d2Wdr2,
                    "d2Udz2": d2Udz2,
                    "d2Vdz2": d2Vdz2,
                    "d2Wdz2": d2Wdz2,
                    "d2Udrz": d2Udrz,
                    "d2Wdrz": d2Wdrz,
                },
            )

            coeffs = compute_cylindrical_coefficients(
                U=U_c,
                V=V_c,
                W=W_c,
                R=R,
                RHO=RHO,
                T=T,
                MU=MU,
                m=m,
                Re=Re,
                Ma=Ma,
                Pr=Pr,
                kappa=kappa,
                dUdr=dUdr,
                dVdr=dVdr,
                dWdr=dWdr,
                dRHOdr=dRHOdr,
                dTdr=dTdr,
                dUdz=dUdz,
                dVdz=dVdz,
                dWdz=dWdz,
                dRHOdz=dRHOdz,
                dTdz=dTdz,
                d2Udr2=d2Udr2,
                d2Vdr2=d2Vdr2,
                d2Wdr2=d2Wdr2,
                d2Tdr2=unused_derivative,
                d2Udz2=d2Udz2,
                d2Vdz2=d2Vdz2,
                d2Wdz2=d2Wdz2,
                d2Tdz2=unused_derivative,
                d2Udrz=d2Udrz,
                d2Vdrz=unused_derivative,
                d2Wdrz=d2Wdrz,
                dmudT=dmudT,
                d2mudT2=d2mudT2,
            )
            DR, D2R, DZ, D2Z, D2RZ = _create_axy_diff_matrices(mesh, m)

        else:
            raise ValueError("model must be '2D' or 'axy'")

        coeff_tensor_bytes = {
            name: int(getattr(coeffs, name).nbytes) for name in _COEFF_TENSOR_NAMES
        }
        coeff_tensor_bytes_total = int(sum(coeff_tensor_bytes.values()))
        stage["ngp"] = int(ngp)
        stage["ndof"] = int(5 * ngp)
        stage["coeff_tensor_bytes_by_array"] = coeff_tensor_bytes
        stage["coeff_tensor_bytes_total"] = coeff_tensor_bytes_total
        _note_assembly_metadata(
            ngp=int(ngp),
            ndof=int(5 * ngp),
            derivative_vector_count=stage.get("derivative_vector_count"),
            derivative_matvec_count=stage.get("derivative_matvec_count"),
            derivative_vector_bytes_total=stage.get("derivative_vector_bytes_total"),
            coeff_tensor_bytes_by_array=coeff_tensor_bytes,
            coeff_tensor_bytes_total=coeff_tensor_bytes_total,
        )

        return {
            "ngp": ngp,
            "model": model,
            "coeffs": coeffs,
            "DR": DR,
            "D2R": D2R,
            "DZ": DZ,
            "D2Z": D2Z,
            "D2RZ": D2RZ,
            "RHO": RHO,
            "U": U,
            "V": V,
            "W": W,
            "T": T,
            "kappa": kappa,
            "Ma": Ma,
            "mesh": mesh,
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
            L0,
            model,
            ngp,
            prep["RHO"],
            prep["U"],
            prep["V"],
            prep["W"],
            prep["T"],
            prep["kappa"],
            prep["Ma"],
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
            prep["RHO"],
            prep["U"],
            prep["V"],
            prep["W"],
            prep["T"],
            prep["kappa"],
            prep["Ma"],
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
                    # PETSc 3.24 can abort here on matrices that were built
                    # without an explicit diagonal slot in every row, even when
                    # the diagonal update is logically benign.
                    L0_petsc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
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

    return sp.bmat(blocks, format="csr").tocsc()


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
        # Some PETSc builds keep the CSR pattern exact and reject later
        # diagonal updates when a row has no explicit diagonal slot yet.
        # Song's sponge update uses ``MatDiagonalSet`` after assembly, so allow
        # PETSc to insert those missing diagonal entries instead of aborting.
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
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
        _note_assembly_metadata(raw_block_assembly_mode="petsc_distributed_owned_rows_direct")
        r0, r1 = ownership_range(matrix)
        owned_blocks = []
        for i in range(5):
            row_start = max(0, r0 - i * ngp)
            row_stop = min(ngp, r1 - i * ngp)
            if row_start >= row_stop:
                continue
            owned_blocks.append((i, row_start, row_stop))

        for i, row_start, row_stop in owned_blocks:
            _insert_owned_block_rows_petsc(
                matrix,
                ngp,
                coeffs,
                i,
                row_start,
                row_stop,
                DR_blocks,
                D2R_blocks,
                DZ_blocks,
                D2Z_blocks,
                D2RZ_blocks,
            )
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


def _insert_owned_block_rows_petsc(
    matrix: "PETSc.Mat",
    ngp: int,
    coeffs: LNSECoefficients,
    row_block: int,
    row_start: int,
    row_stop: int,
    DR: list[_CSRDerivativeCache],
    D2R: list[_CSRDerivativeCache],
    DZ: list[_CSRDerivativeCache],
    D2Z: list[_CSRDerivativeCache],
    D2RZ: list[_CSRDerivativeCache],
) -> None:
    """Insert owned PETSc rows without building throw-away SciPy block slices."""
    from petsc4py import PETSc

    # Keep this order paired with _build_block_csr: Ay*DR, Ax*DZ,
    # Ayy*D2R, Axx*D2Z, Axy*D2RZ.  PETSc ADD_VALUES performs the same
    # duplicate-column summation that CSR sum_duplicates does in the SciPy path.
    caches_by_term = (DR, DZ, D2R, D2Z, D2RZ)
    coeff_arrays = (coeffs.Ay, coeffs.Ax, coeffs.Ayy, coeffs.Axx, coeffs.Axy)

    for local_row in range(row_start, row_stop):
        global_row = row_block * ngp + local_row
        for col_block in range(5):
            col_offset = col_block * ngp
            a0 = coeffs.A0[row_block, col_block, local_row]
            if a0 != 0:
                matrix.setValue(
                    global_row,
                    col_offset + local_row,
                    a0,
                    addv=PETSc.InsertMode.ADD_VALUES,
                )

            for coeff_tensor, caches in zip(coeff_arrays, caches_by_term, strict=True):
                coeff = coeff_tensor[row_block, col_block, local_row]
                if coeff == 0:
                    continue
                cache = caches[col_block]
                p0 = int(cache.csr.indptr[local_row])
                p1 = int(cache.csr.indptr[local_row + 1])
                if p0 == p1:
                    continue
                matrix.setValues(
                    global_row,
                    col_offset + cache.csr.indices[p0:p1],
                    coeff * cache.csr.data[p0:p1],
                    addv=PETSc.InsertMode.ADD_VALUES,
                )


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

    derivative_terms = [
        (has_ay, ay, dr_cache),
        (has_ax, ax, dz_cache),
        (has_ayy, ayy, d2r_cache),
        (has_axx, axx, d2z_cache),
        (has_axy, axy, d2rz_cache),
    ]
    total_nnz = (ngp if has_a0 else 0) + sum(
        cache.csr.nnz for enabled, _, cache in derivative_terms if enabled and cache.csr.nnz
    )
    if total_nnz == 0:
        return sp.csr_matrix((ngp, ngp), dtype=complex)

    rows = np.empty(total_nnz, dtype=np.int64)
    cols = np.empty(total_nnz, dtype=np.int64)
    data = np.empty(total_nnz, dtype=complex)
    offset = 0

    if has_a0:
        stop = offset + ngp
        diag_idx = np.arange(ngp, dtype=np.int64)
        rows[offset:stop] = diag_idx
        cols[offset:stop] = diag_idx
        data[offset:stop] = a0
        offset = stop

    for enabled, coeff, cache in derivative_terms:
        if not enabled or cache.csr.nnz == 0:
            continue
        stop = offset + cache.csr.nnz
        rows[offset:stop] = cache.row_indices
        cols[offset:stop] = cache.csr.indices
        np.multiply(coeff[cache.row_indices], cache.csr.data, out=data[offset:stop])
        offset = stop

    block = sp.csr_matrix(
        (data, (rows, cols)),
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
    shifted_d2z = [
        d2z + 2.0 * i_alpha * dz + i_alpha**2 * I for d2z, dz in zip(d2z_blocks, dz_blocks)
    ]
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
        inv_B_signs[0 * ngp : 1 * ngp] = 1.0
        inv_B_signs[1 * ngp : 2 * ngp] = 1.0 / RHO
        inv_B_signs[2 * ngp : 3 * ngp] = 1.0 / RHO
        inv_B_signs[3 * ngp : 4 * ngp] = 1.0 / RHO
        inv_B_signs[4 * ngp : 5 * ngp] = 1.0 / RHO
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
        [D(np.ones(ngp)), Z, Z, Z, Z],
        [D(-U * invRHO), D(invRHO), Z, Z, Z],
        [D(-V * invRHO), Z, D(invRHO), Z, Z],
        [D(-W * invRHO), Z, Z, D(invRHO), Z],
        [
            D(KE_half * invRHOcv - T * invRHO),
            D(-U * invRHOcv),
            D(-V * invRHOcv),
            D(-W * invRHOcv),
            D(invRHOcv),
        ],
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
    """Apply the analytical ``inv(B/1j)`` left scaling directly to PETSc rows.

    This is step 2 of the PETSc assembly sequence (see module docstring).
    It must be called after ``_assemble_block_operator_petsc`` and before
    the sponge, dissipation, and boundary-condition steps.

    For ``model='2d'``: the inverse B matrix is block-diagonal and applied
    via PETSc diagonal scaling --- efficient and parallel-safe.

    For ``model='axy'``: the lower-triangular inv(B) couples different
    variable blocks, so the matrix is first converted to SciPy CSR on rank 0,
    transformed in scipy, then rebuilt as a PETSc AIJ matrix.  This path
    requires a sequential (single-rank) PETSc matrix; it raises
    ``RuntimeError`` for distributed matrices.
    """
    from petsc4py import PETSc

    from ..utils.petsc_core import petsc_to_scipy

    # Hard-fail early: the axisymmetric path falls back to petsc_to_scipy which
    # converts the distributed matrix to a local SciPy CSR (only valid on one
    # rank).  Raise a clean error before any silent data corruption can occur.
    if model == "axy" and L0.getComm().getSize() > 1:
        comm_size = L0.getComm().getSize()
        raise RuntimeError(
            "_apply_inv_B_petsc: model='axy' is not supported with distributed "
            f"PETSc matrices (comm size = {comm_size}).  Assemble with backend='scipy' "
            "or run on a single MPI rank for axisymmetric resolvent/stability."
        )

    row_scaling = np.ones(5 * ngp, dtype=complex)
    if model == "2d":
        row_scaling[1 * ngp : 2 * ngp] = 1.0 / RHO
        row_scaling[2 * ngp : 3 * ngp] = 1.0 / RHO
        row_scaling[3 * ngp : 4 * ngp] = 1.0 / RHO
        row_scaling[4 * ngp : 5 * ngp] = 1.0 / RHO
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
        if (
            use_symmetry
            and mesh.y_symmetry
            and mesh.Dy_symm is not None
            and mesh.D2y_symm is not None
        ):
            DR_blocks.append(mesh.Dy_symm.tocsc())
            D2R_blocks.append(mesh.D2y_symm.tocsc())
            if mesh.Dyx_symm is not None:
                D2RZ_blocks.append(mesh.Dyx_symm.tocsc())
            else:
                D2RZ_blocks.append((mesh.Dy_symm @ Dx).tocsc())
        elif (
            mesh.y_symmetry
            and not use_symmetry
            and mesh.Dy_asymm is not None
            and mesh.D2y_asymm is not None
        ):
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
