# SPDX-License-Identifier: GPL-3.0-only
"""Check candidate sparse-direct backends against the MUMPS reference.

CPU skeleton for the future GPU/mixed-precision parity gate.
The reference is the repository's validated MUMPS shifted-invert path.
A candidate backend runs on the same small Blasius temporal-LNSE EVP.
It must satisfy the same acceptance quantities already used by the
stability checks.

* ``postcheck_residual_rel <= 1e-6`` bounds each returned eigenpair by the
  explicit full-operator residual from ``_postcheck_stability_modes``.
  This is the primary quality floor.
  A backend cannot be parity-equivalent if either its modes or the
  MUMPS modes fail the existing postcheck residual gate.
* ``postcheck_target_distance <= 0.11`` bounds that both spectra came from
  the intended shift-invert target neighbourhood.
  The tolerance matches the small Blasius backend-comparison convention
  in ``tests/test_gpu_backend_comparison.py``.
* Matching is conjugate-aware: the assignment cost between candidate and
  MUMPS eigenvalues is ``min(|omega_c - omega_m|, |omega_c + conj(omega_m)|)``.
  The ``-conj(omega)`` mirror is the temporal-spectrum convention.
  The backend parity path in ``tests/test_gpu_backend_comparison.py``
  already uses it (reflection across the imaginary axis: opposite real
  frequency, same growth rate).
* The candidate-to-MUMPS matched eigenvalue distance must be no larger
  than the postcheck residual gate (``1e-6``).
  That ties the parity distance to the already accepted eigenpair-defect
  scale instead of introducing a separate backend-specific epsilon.

Each comparison also creates a structured quality/cost record.
It holds matrix size, nonzeros, factor-memory estimate, and setup/solve
wall times.
It also holds rank/GPU counts, exact solver options, backend-probe
readbacks, postcheck maxima, target-distance maxima, and matched
eigenvalue distances.
The record prints with a stable ``PARITY_RECORD`` prefix.
Pytest capture stays off so batch scripts can collect it from normal
test output.

GPU execution is intentionally deferred.
The non-MUMPS self-check here uses the CPU HARD path requested by bead
r3uu.1.6.
That path is SuperLU_DIST with PETSc ``-pc_precision single`` (complex64
factor with FP64 refinement) when that PETSc build supports it.
If SuperLU_DIST is absent, the skip reason is loud and explicit.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from petsc4py import PETSc
from scipy.optimize import linear_sum_assignment

from examples.martini.matlab_baseflow import load_blasius
from linstabpy.baseflow import sutherland_air
from linstabpy.boundary_conditions import apply_dirichlet
from linstabpy.driver.run import _postcheck_stability_modes
from linstabpy.linalg.backends import apply_factor_backend, factor_backend_available
from linstabpy.mesh import Mesh
from linstabpy.operators import build_lnse_operator
from linstabpy.solvers.petsc_utils import scipy_to_petsc
from linstabpy.stability import eigenvalues as temporal_eigenvalues
from linstabpy.stability.acceptance import evaluate_stability_acceptance

TARGET = -0.1j
N_MODES = 5
NCV = 30
POSTCHECK_REL_MAX = 1.0e-6
TARGET_DISTANCE_MAX = 0.11
EIGENVALUE_MATCH_MAX = POSTCHECK_REL_MAX


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [[float(np.real(item)), float(np.imag(item))] for item in value.reshape(-1)]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    return str(value)


def _build_blasius_temporal_problem() -> tuple[sp.spmatrix, np.ndarray]:
    mesh = Mesh.create(xrange=(200, 300), yrange=(0, 30), nx=15, ny=10, fd_order=4)
    y_new = 30.0 * np.sinh(3.0 * mesh.Y / 30.0) / np.sinh(3.0)
    mesh = mesh.deform(Y_new=y_new)

    bf = load_blasius(mesh.X, mesh.Y, Re=1e4, Ma=0.3, Pr=0.7, kappa=1.4, T_0=288.15)
    sutherland_air(bf)

    operator, idx = build_lnse_operator(mesh, bf, kx=0.0)
    operator, bc_indices = apply_dirichlet(operator, idx, "bt", "uvwT", mesh.ngp)
    return operator, bc_indices


def _solver_options(backend: str, *, factor_precision: str = "double") -> dict[str, Any]:
    return {
        "st_solver": "mumps",
        "factor_backend": backend,
        "factor_precision": factor_precision,
        "eps_tol": 1.0e-10,
        "eps_max_it": 7500,
        "backend_probe": {"enabled": backend != "mumps", "residual_rel_max": POSTCHECK_REL_MAX},
    }


def _solve_with_backend(
    operator: sp.spmatrix,
    bc_indices: np.ndarray,
    backend: str,
    *,
    factor_precision: str = "double",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    omega, modes, info = temporal_eigenvalues(
        operator,
        N_MODES,
        TARGET,
        ncv=NCV,
        solver_cfg={"eigen": _solver_options(backend, factor_precision=factor_precision)},
        return_info=True,
    )
    postcheck = _postcheck_stability_modes(
        operator,
        omega,
        modes,
        target=TARGET,
        bc_indices=bc_indices,
        constrained_dirichlet_indices=bc_indices,
        conjugate_aware=True,
    )
    info.update(postcheck)
    return omega, modes, info


def _assert_existing_acceptance_gate_passes(
    omega: np.ndarray, info: dict[str, Any]
) -> dict[str, Any]:
    result = evaluate_stability_acceptance(
        omega,
        np.asarray(info["postcheck_residual_rel"], dtype=float),
        np.asarray(info["postcheck_target_distance"], dtype=float),
        postcheck_rel_max=POSTCHECK_REL_MAX,
        target_distance_max=TARGET_DISTANCE_MAX,
        conjugate_aware=True,
    )
    assert result["accepted_count"] == N_MODES, result
    return result


def _conjugate_aware_match(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    direct = np.abs(candidate[:, None] - reference[None, :])
    # Temporal mirror partner is -conj(omega) (same convention as
    # tests/test_gpu_backend_comparison.py), not conj(omega).
    mirror = np.abs(candidate[:, None] + np.conjugate(reference[None, :]))
    cost = np.minimum(direct, mirror)
    rows, cols = linear_sum_assignment(cost)
    errors = np.empty(candidate.size, dtype=float)
    errors[rows] = cost[rows, cols]
    return errors


def _factor_memory_probe(
    operator: sp.spmatrix,
    backend: str,
    *,
    factor_precision: str = "double",
) -> dict[str, Any]:
    shifted = (1j * operator - TARGET * sp.eye(operator.shape[0], format="csc")).tocsc()
    mat = scipy_to_petsc(
        shifted, {"eigen": _solver_options(backend, factor_precision=factor_precision)}
    )
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    prefix = f"parity_{backend}_"
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(mat)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    t0 = time.perf_counter()
    try:
        resolved = apply_factor_backend(
            pc,
            backend,
            options_prefix=prefix,
            factor_precision=factor_precision,
        )
        ksp.setUp()
        setup_seconds = time.perf_counter() - t0
        factor = pc.getFactorMatrix()
        info = factor.getInfo() if factor is not None else {}
        verified_precision = factor_precision == "double"
        if factor_precision == "single":
            # Verify from PETSc consumption, not request state: PCSetUp must
            # read the option (Options.used), and the factor solver
            # readback must be SuperLU_DIST.  An ignored/unsupported
            # -pc_precision leaves used() False and fails this check.
            verified_precision = (
                PETSc.Options().used(f"{prefix}pc_precision")
                and resolved == "superlu_dist"
                and pc.getFactorSolverType() == "superlu_dist"
            )
        return {
            "factor_probe_backend_readback": pc.getFactorSolverType(),
            "factor_probe_precision_requested": factor_precision,
            "factor_probe_precision_verified": bool(verified_precision),
            "factor_setup_seconds": float(setup_seconds),
            "factor_memory_bytes": int(info.get("memory", -1)) if info else -1,
            "factor_nz_allocated": int(info.get("nz_allocated", -1)) if info else -1,
            "factor_nz_used": int(info.get("nz_used", -1)) if info else -1,
        }
    finally:
        ksp.destroy()
        mat.destroy()


def _solver_state(info: dict[str, Any]) -> dict[str, Any]:
    """Subspace and convergence state, so two comparisons are provably alike."""
    return {
        key: info.get(key)
        for key in (
            "nev_requested",
            "ncv_used",
            "nconv",
            "eps_iterations",
            "eps_reason",
            "ksp_reason",
            "eps_tol",
            "eps_max_it",
            "st_solver",
        )
    }


def _comparison_record(
    name: str,
    operator: sp.spmatrix,
    reference_omega: np.ndarray,
    reference_info: dict[str, Any],
    candidate_omega: np.ndarray,
    candidate_info: dict[str, Any],
    *,
    candidate_backend: str,
    factor_precision: str,
) -> dict[str, Any]:
    match_errors = _conjugate_aware_match(candidate_omega, reference_omega)
    record = {
        "comparison": name,
        "matrix_size": list(operator.shape),
        "matrix_nnz": int(operator.nnz),
        "ranks": int(PETSc.COMM_WORLD.getSize()),
        "gpus": 0,
        "target": TARGET,
        "tolerances": {
            "postcheck_rel_max": POSTCHECK_REL_MAX,
            "target_distance_max": TARGET_DISTANCE_MAX,
            "eigenvalue_match_max": EIGENVALUE_MATCH_MAX,
            "conjugate_aware_matching": True,
        },
        "reference_options": _solver_options("mumps"),
        "candidate_options": _solver_options(candidate_backend, factor_precision=factor_precision),
        "reference_solver": _solver_state(reference_info),
        "candidate_solver": _solver_state(candidate_info),
        "reference_wall_seconds": {
            "eps_setup": float(reference_info["eps_setup_seconds"]),
            "eps_solve": float(reference_info["eps_solve_seconds"]),
        },
        "candidate_wall_seconds": {
            "eps_setup": float(candidate_info["eps_setup_seconds"]),
            "eps_solve": float(candidate_info["eps_solve_seconds"]),
        },
        "reference_backend_probe": {
            key: value for key, value in reference_info.items() if key.startswith("backend_probe_")
        },
        "candidate_backend_probe": {
            key: value for key, value in candidate_info.items() if key.startswith("backend_probe_")
        },
        "reference_quality": {
            "postcheck_residual_rel_max": float(np.max(reference_info["postcheck_residual_rel"])),
            "target_distance_max": float(np.max(reference_info["postcheck_target_distance"])),
        },
        "candidate_quality": {
            "postcheck_residual_rel_max": float(np.max(candidate_info["postcheck_residual_rel"])),
            "target_distance_max": float(np.max(candidate_info["postcheck_target_distance"])),
            "matched_eigenvalue_distance_max": float(np.max(match_errors)),
            "matched_eigenvalue_distances": match_errors,
        },
    }
    record.update(
        _factor_memory_probe(operator, candidate_backend, factor_precision=factor_precision)
    )
    return record


def _assert_parity_record(record: dict[str, Any]) -> None:
    assert record["candidate_quality"]["postcheck_residual_rel_max"] <= POSTCHECK_REL_MAX
    assert record["reference_quality"]["postcheck_residual_rel_max"] <= POSTCHECK_REL_MAX
    assert record["candidate_quality"]["target_distance_max"] <= TARGET_DISTANCE_MAX
    assert record["reference_quality"]["target_distance_max"] <= TARGET_DISTANCE_MAX
    assert record["candidate_quality"]["matched_eigenvalue_distance_max"] <= EIGENVALUE_MATCH_MAX
    if record["candidate_options"]["factor_precision"] == "single":
        assert record["factor_probe_precision_verified"] is True


def _emit(record: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    with capsys.disabled():
        print(
            "PARITY_RECORD " + json.dumps(record, default=_json_default, sort_keys=True), flush=True
        )


def test_mumps_vs_mumps_self_consistency(capsys: pytest.CaptureFixture[str]) -> None:
    if not factor_backend_available("mumps"):
        pytest.skip("SKIPPED: mumps not in this PETSc build")

    operator, bc_indices = _build_blasius_temporal_problem()
    reference_omega, _, reference_info = _solve_with_backend(operator, bc_indices, "mumps")
    candidate_omega, _, candidate_info = _solve_with_backend(operator, bc_indices, "mumps")

    _assert_existing_acceptance_gate_passes(reference_omega, reference_info)
    _assert_existing_acceptance_gate_passes(candidate_omega, candidate_info)
    record = _comparison_record(
        "mumps_vs_mumps",
        operator,
        reference_omega,
        reference_info,
        candidate_omega,
        candidate_info,
        candidate_backend="mumps",
        factor_precision="double",
    )
    _emit(record, capsys)
    _assert_parity_record(record)


def test_factor_precision_single_is_refused_rather_than_silently_downgraded() -> None:
    """Asking for a single-precision factor must fail loudly, not lie.

    This replaces ``test_superlu_dist_single_precision_vs_mumps_cpu_hard_path``,
    whose premise a c9 run (PETSc 3.24.2) showed to be false.
    That check believed it was comparing an FP32 factor against the MUMPS
    FP64 reference.
    It was not.

    * job 535 -- SuperLU_DIST with ``factor_precision`` "single" and "double"
      returned BIT-IDENTICAL results.
      ``||Kx-b||/||b||`` = 1.852e-06 both ways, eigenvalues agreeing to all
      12 printed digits.
    * job 536 -- the only lever in the code is the PETSc ``-pc_precision``
      option.
      After ``pc.setFromOptions()`` + ``ksp.setUp()`` that key is still
      sitting unconsumed in the options database.

    So the knob never did anything.
    The 1.852e-06 the old check tripped over was SuperLU_DIST's ordinary
    *double*-precision accuracy at this shift -- not an FP32 effect.
    Relaxing the probe threshold to make it pass is what the failure
    superficially invites.
    That would hide a real accuracy signal to accommodate a feature that
    does not exist.

    ``apply_factor_backend`` now refuses the request outright.
    This check pins that refusal.
    When a genuine mechanism lands, it will fail and must become a real
    FP32-vs-FP64 comparison.
    """
    from linstabpy.linalg.backends import apply_factor_backend

    if not factor_backend_available("superlu_dist"):
        pytest.skip("SKIPPED: superlu_dist not in this PETSc build")

    # No operator needed: the refusal fires before any factorization work.
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    try:
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        with pytest.raises(NotImplementedError) as excinfo:
            apply_factor_backend(pc, "superlu_dist", factor_precision="single")
    finally:
        ksp.destroy()

    message = str(excinfo.value)
    assert "factor_precision='single' is not supported" in message
    assert "silently be computed in double precision" in message


def test_superlu_dist_is_rejected_by_the_backend_probe_at_a_hard_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accuracy gate must actually reject a backend that solves badly.

    Measured on c9 at this problem and shift (TARGET = -0.1j).
    That target sits close to the spectrum and so makes
    ``K = L0 - sigma*I`` ill-conditioned.
    Job 535 numbers.

        mumps         probe skipped (trusted default).
        strumpack     ||Kx-b||/||b|| = 1.979e-12.
        superlu_dist  ||Kx-b||/||b|| = 1.852e-06   <- six decades worse.

    SuperLU_DIST also returned eigenvalues that do not match there.
    Imaginary parts differ from the MUMPS and STRUMPACK answers in the
    third significant digit.
    This is backend-specific and shift-specific.
    At an easier shift the same backend reaches 1.189e-13.
    That is exactly why the per-solve probe exists rather than a static
    allow-list.

    Guarding the rejection keeps the safety net honest.
    If the probe is ever loosened or bypassed, this check fails.
    A mismatched spectrum then cannot flow silently into a gain map.
    """
    if not factor_backend_available("superlu_dist"):
        pytest.skip("SKIPPED: superlu_dist not in this PETSc build")

    monkeypatch.delenv("LINSTABPY_FACTOR_PRECISION", raising=False)
    operator, bc_indices = _build_blasius_temporal_problem()

    with pytest.raises(RuntimeError) as excinfo:
        _solve_with_backend(operator, bc_indices, "superlu_dist")

    message = str(excinfo.value)
    assert "factor backend 'superlu_dist' failed accuracy probe" in message
    assert "set eigen.backend_probe.enabled=false to override" in message
