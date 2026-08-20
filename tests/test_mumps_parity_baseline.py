# SPDX-License-Identifier: GPL-3.0-only
"""Regression guard for the default CPU/MUMPS shift-invert baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")

from tests.regenerate_mumps_parity_reference import (
    REFERENCE_PATH,
    RELATIVE_DRIFT_TOL,
    compute_reference_quantities,
    refuse_polluted_linstabpy_env,
)


def _load_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    if not path.exists():
        pytest.skip("SKIPPED: missing tests/fixtures/mumps_parity_reference.npz")
    with np.load(path, allow_pickle=False) as data:
        return {
            "leading_eigenvalues": np.asarray(data["leading_eigenvalues"], dtype=complex),
            "metadata": json.loads(str(np.asarray(data["metadata_json"]).item())),
            "relative_drift_tol": float(np.asarray(data["relative_drift_tol"]).item()),
        }


def assert_within_relative_drift(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float = RELATIVE_DRIFT_TOL,
) -> None:
    """Assert every frozen complex value has a unique nearby current match."""
    actual = np.asarray(actual, dtype=complex).reshape(-1)
    expected = np.asarray(expected, dtype=complex).reshape(-1)
    assert actual.size >= expected.size, (
        f"current solve returned fewer modes than frozen reference: "
        f"actual={actual.size}, expected={expected.size}"
    )

    unmatched_actual = set(range(actual.size))
    matched_actual: list[int] = []
    rel_drifts: list[float] = []
    pairs: list[tuple[complex, complex, float]] = []
    for frozen in expected:
        nearest_index = min(unmatched_actual, key=lambda idx: abs(actual[idx] - frozen))
        unmatched_actual.remove(nearest_index)
        matched_actual.append(nearest_index)
        current = actual[nearest_index]
        scale = max(abs(frozen), np.finfo(float).tiny)
        rel_drift = float(abs(current - frozen) / scale)
        rel_drifts.append(rel_drift)
        pairs.append((current, frozen, rel_drift))

    assert len(matched_actual) == expected.size
    assert len(set(matched_actual)) == expected.size
    assert all(rel_drift <= rtol for rel_drift in rel_drifts), (
        f"relative drift exceeds {rtol:.1e}: matched pairs (actual, expected, rel_drift)={pairs}"
    )


def _clear_linstabpy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("LINSTABPY_"):
            monkeypatch.delenv(name, raising=False)


def test_default_mumps_parity_matches_frozen_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_linstabpy_env(monkeypatch)
    reference = _load_reference()

    # Ask for every frozen mode, not only the requested five: the reference
    # build converged an incidental sixth (a +/- pair at identical distance to
    # the target) and froze it; on the MKL build (2026-08-18) nev=5 converges
    # exactly five and this guard tripped on the count, with all five values
    # inside 1e-6. nev=6 reproduces all six on both builds.
    reference_metadata = reference["metadata"]
    current = compute_reference_quantities(
        n_modes=max(int(reference_metadata["n_modes"]), int(reference_metadata["stored_n_modes"]))
    )

    assert_within_relative_drift(
        current["leading_eigenvalues"],
        reference["leading_eigenvalues"],
        rtol=reference["relative_drift_tol"],
    )
    current_metadata = current["metadata"]
    assert reference_metadata["ndof"] == 750
    assert reference_metadata["case_params"]["case_id"] == "mumps_parity_blasius_lnse_v1"
    assert current_metadata["st_solver"] == "mumps"
    assert current_metadata["factor_backend"] == "mumps"
    assert reference_metadata["st_solver"] == current_metadata["st_solver"]
    assert reference_metadata["factor_backend"] == current_metadata["factor_backend"]
    assert reference_metadata["ncv_used"] == current_metadata["ncv_used"]
    assert reference_metadata["n_modes"] == reference_metadata["case_params"]["n_modes"]
    assert reference_metadata["nconv"] >= reference_metadata["n_modes"]
    assert reference_metadata["stored_n_modes"] == reference_metadata["nconv"]
    assert len(reference["leading_eigenvalues"]) == reference_metadata["nconv"]
    assert reference_metadata["eps_tol"] == current_metadata["eps_tol"]


def test_regeneration_refuses_polluted_linstabpy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINSTABPY_EIGEN_EPS_TOL", "1e-4")
    with pytest.raises(RuntimeError, match="refusing to regenerate MUMPS parity fixture"):
        refuse_polluted_linstabpy_env()


def test_perturbed_reference_trips_relative_drift_guard() -> None:
    reference = _load_reference()
    perturbed = reference["leading_eigenvalues"].copy()
    # Perturbation must exceed RELATIVE_DRIFT_TOL (1e-6) for the guard to trip;
    # 1e-4 is still far below the ~1e-3 shift a real operator regression causes.
    perturbed[0] *= 1.0 + 1.0e-4

    with pytest.raises(AssertionError, match="relative drift exceeds"):
        assert_within_relative_drift(
            reference["leading_eigenvalues"],
            perturbed,
            rtol=reference["relative_drift_tol"],
        )
