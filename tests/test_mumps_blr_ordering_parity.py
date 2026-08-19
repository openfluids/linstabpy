# SPDX-License-Identifier: GPL-3.0-only
"""MUMPS BLR/order parity checks against the baseline MUMPS EVP path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from petsc4py import PETSc

from linstabpy.linalg.backends import MUMPS_ORDERING_TO_ICNTL_7, factor_backend_available
from linstabpy.solvers.petsc_utils import scipy_to_petsc
from tests.test_parity_vs_mumps import (
    TARGET,
    _assert_existing_acceptance_gate_passes,
    _assert_parity_record,
    _build_blasius_temporal_problem,
    _comparison_record,
    _conjugate_aware_match,
    _emit,
    _json_default,
    _solve_with_backend,
    _solver_options,
)


@dataclass(frozen=True)
class MumpsVariant:
    name: str
    blr: bool = False
    blr_tol: float | None = None
    ordering: str | None = None


_VARIANTS = (
    MumpsVariant("mumps_blr_tol_1e-4", blr=True, blr_tol=1.0e-4),
    MumpsVariant("mumps_ordering_pord", ordering="pord"),
    MumpsVariant("mumps_blr_tol_1e-4_ordering_pord", blr=True, blr_tol=1.0e-4, ordering="pord"),
)

_MUMPS_OPTION_KEYS = (
    "mat_mumps_icntl_35",
    "mat_mumps_cntl_7",
    "mat_mumps_icntl_7",
)


def _clear_mumps_options(prefix: str = "") -> None:
    opts = PETSc.Options()
    for key in _MUMPS_OPTION_KEYS:
        opts.delValue(f"{prefix}{key}")


def _set_variant_env(monkeypatch: pytest.MonkeyPatch, variant: MumpsVariant) -> None:
    if variant.blr:
        monkeypatch.setenv("LINSTABPY_MUMPS_BLR", "1")
        monkeypatch.setenv("LINSTABPY_MUMPS_BLR_TOL", f"{variant.blr_tol:g}")
    else:
        monkeypatch.setenv("LINSTABPY_MUMPS_BLR", "0")
        monkeypatch.delenv("LINSTABPY_MUMPS_BLR_TOL", raising=False)
    if variant.ordering is None:
        monkeypatch.delenv("LINSTABPY_MUMPS_ORDERING", raising=False)
    else:
        monkeypatch.setenv("LINSTABPY_MUMPS_ORDERING", variant.ordering)


def _mumps_probe(
    operator: sp.spmatrix,
    variant: MumpsVariant,
    *,
    prefix: str,
) -> dict[str, Any]:
    shifted = (1j * operator - TARGET * sp.eye(operator.shape[0], format="csc")).tocsc()
    mat = scipy_to_petsc(shifted, {"eigen": _solver_options("mumps")})
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(mat)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    try:
        from linstabpy.linalg.backends import apply_factor_backend

        resolved = apply_factor_backend(pc, "mumps", options_prefix=prefix)
        try:
            setup_t0 = time.perf_counter()
            ksp.setUp()
            setup_seconds = time.perf_counter() - setup_t0
        except PETSc.Error as exc:
            pytest.skip(f"SKIPPED: MUMPS variant {variant.name} unsupported during setup: {exc}")
        factor = pc.getFactorMatrix()
        info = factor.getInfo() if factor is not None else {}
        icntl7 = factor.getMumpsIcntl(7) if factor is not None else None
        opts = PETSc.Options()
        return {
            "factor_probe_backend_readback": pc.getFactorSolverType(),
            "factor_setup_seconds": float(setup_seconds),
            "factor_memory_bytes": int(info.get("memory", -1)) if info else -1,
            "factor_nz_allocated": int(info.get("nz_allocated", -1)) if info else -1,
            "factor_nz_used": int(info.get("nz_used", -1)) if info else -1,
            "mumps_backend_resolved": resolved,
            "mumps_icntl_7": None if icntl7 is None else int(icntl7),
            "mumps_ordering_requested": variant.ordering,
            "mumps_ordering_expected_icntl_7": (
                None if variant.ordering is None else MUMPS_ORDERING_TO_ICNTL_7[variant.ordering]
            ),
            "mumps_ordering_verified": (
                True
                if variant.ordering is None
                else int(icntl7) == MUMPS_ORDERING_TO_ICNTL_7[variant.ordering]
            ),
            "mumps_blr_requested": bool(variant.blr),
            "mumps_blr_tol": variant.blr_tol,
            "mumps_blr_option_consumed": (
                opts.used(f"{prefix}mat_mumps_icntl_35") if variant.blr else None
            ),
            "mumps_blr_tol_option_consumed": (
                opts.used(f"{prefix}mat_mumps_cntl_7") if variant.blr else None
            ),
        }
    finally:
        ksp.destroy()
        mat.destroy()
        _clear_mumps_options(prefix)


def _mumps_variant_record(
    variant: MumpsVariant,
    operator: sp.spmatrix,
    reference_omega: np.ndarray,
    reference_info: dict[str, Any],
    candidate_omega: np.ndarray,
    candidate_info: dict[str, Any],
) -> dict[str, Any]:
    try:
        record = _comparison_record(
            variant.name,
            operator,
            reference_omega,
            reference_info,
            candidate_omega,
            candidate_info,
            candidate_backend="mumps",
            factor_precision="double",
        )
    except PETSc.Error as exc:
        pytest.skip(f"SKIPPED: MUMPS variant {variant.name} unsupported during factor probe: {exc}")
    except RuntimeError as exc:
        if "MUMPS" in str(exc) or "mumps" in str(exc):
            pytest.skip(
                f"SKIPPED: MUMPS variant {variant.name} unsupported during factor probe: {exc}"
            )
        raise
    # Recompute the MUMPS-specific factor probe under the same variant knobs so
    # the record contains ICNTL(7), BLR tolerance, factor nonzeros, and memory.
    record.update(_mumps_probe(operator, variant, prefix=f"{variant.name}_"))
    record["candidate_options"].update(
        {
            "mumps_blr": bool(variant.blr),
            "mumps_blr_tol": variant.blr_tol,
            "mumps_ordering": variant.ordering,
        }
    )
    match_errors = _conjugate_aware_match(candidate_omega, reference_omega)
    record["candidate_quality"]["matched_eigenvalue_distances"] = match_errors
    record["candidate_quality"]["matched_eigenvalue_distance_max"] = float(np.max(match_errors))
    return record


def _assert_mumps_variant_record(record: dict[str, Any], variant: MumpsVariant) -> None:
    _assert_parity_record(record)
    assert record["factor_probe_backend_readback"] == "mumps"
    if variant.ordering is not None:
        if not record["mumps_ordering_verified"]:
            pytest.skip(
                "SKIPPED: MUMPS ordering variant "
                f"{variant.ordering!r} was not realized as ICNTL(7)="
                f"{record['mumps_ordering_expected_icntl_7']} "
                f"(got {record['mumps_icntl_7']})"
            )
    if variant.blr:
        if not record["mumps_blr_option_consumed"] or not record["mumps_blr_tol_option_consumed"]:
            pytest.skip(f"SKIPPED: MUMPS BLR variant {variant.name} unsupported or not consumed")


def _run_variant_in_child_process(variant: MumpsVariant) -> dict[str, Any]:
    if not factor_backend_available("mumps"):
        pytest.skip("SKIPPED: mumps not in this PETSc build")

    monkeypatch = pytest.MonkeyPatch()
    operator, bc_indices = _build_blasius_temporal_problem()
    try:
        _clear_mumps_options()
        monkeypatch.setenv("LINSTABPY_MUMPS_BLR", "0")
        monkeypatch.delenv("LINSTABPY_MUMPS_BLR_TOL", raising=False)
        monkeypatch.delenv("LINSTABPY_MUMPS_ORDERING", raising=False)
        reference_omega, _, reference_info = _solve_with_backend(operator, bc_indices, "mumps")

        _clear_mumps_options()
        _set_variant_env(monkeypatch, variant)
        try:
            candidate_omega, _, candidate_info = _solve_with_backend(operator, bc_indices, "mumps")
        except PETSc.Error as exc:
            pytest.skip(
                f"SKIPPED: MUMPS variant {variant.name} unsupported during EVP solve: {exc}"
            )
        except RuntimeError as exc:
            if "MUMPS" in str(exc) or "mumps" in str(exc):
                pytest.skip(
                    f"SKIPPED: MUMPS variant {variant.name} unsupported during EVP solve: {exc}"
                )
            raise

        _assert_existing_acceptance_gate_passes(reference_omega, reference_info)
        _assert_existing_acceptance_gate_passes(candidate_omega, candidate_info)
        record = _mumps_variant_record(
            variant,
            operator,
            reference_omega,
            reference_info,
            candidate_omega,
            candidate_info,
        )
        _assert_mumps_variant_record(record, variant)
        return record
    finally:
        _clear_mumps_options()
        monkeypatch.undo()


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src:." if not existing else f"src:.:{existing}"
    return env


@pytest.mark.parametrize("variant", _VARIANTS, ids=[variant.name for variant in _VARIANTS])
def test_mumps_blr_ordering_variant_matches_baseline(
    capsys: pytest.CaptureFixture[str],
    variant: MumpsVariant,
) -> None:
    # MUMPS/PETSc options are process-global.  Run each lossy BLR/ordering
    # variant in a child process so the rest of the suite still sees the
    # frozen, knobs-off baseline path.
    #
    # Imported here, not at module scope: this file re-executes itself as a plain
    # script (see __main__ below), where the `tests` package is not importable.
    from tests._subprocess_timeout import load_aware_timeout

    proc = subprocess.run(
        [sys.executable, __file__, variant.name],
        cwd=os.getcwd(),
        env=_subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # Hang guard, not a performance assertion. The former hard 120 s left ~1.4x
        # headroom over this variant's own runtime on a loaded machine, so it timed
        # out spuriously whenever another job used the node. See tests/_subprocess_timeout.py.
        timeout=load_aware_timeout(300),
        check=False,
    )
    if proc.returncode == 77:
        pytest.skip(proc.stdout.strip() or f"SKIPPED: MUMPS variant {variant.name} unsupported")
    assert proc.returncode == 0, proc.stdout
    records = [
        line.removeprefix("PARITY_RECORD ")
        for line in proc.stdout.splitlines()
        if line.startswith("PARITY_RECORD ")
    ]
    assert len(records) == 1, proc.stdout
    record = json.loads(records[0])
    _assert_mumps_variant_record(record, variant)
    _emit(record, capsys)


def test_mumps_variant_record_is_json_serializable() -> None:
    record = {
        "comparison": "schema_only",
        "target": TARGET,
        "matched_eigenvalue_distances": np.array([0.0]),
    }
    json.dumps(record, default=_json_default, sort_keys=True)


def _main() -> int:
    variants = {variant.name: variant for variant in _VARIANTS}
    if len(sys.argv) != 2 or sys.argv[1] not in variants:
        print("usage: test_mumps_blr_ordering_parity.py <variant>", file=sys.stderr)
        return 2
    try:
        record = _run_variant_in_child_process(variants[sys.argv[1]])
    except pytest.skip.Exception as exc:
        print(str(exc), flush=True)
        return 77
    print("PARITY_RECORD " + json.dumps(record, default=_json_default, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
