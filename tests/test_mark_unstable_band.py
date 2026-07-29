# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.solvers.router import band_marking
from linstabpy.solvers.router.band_marking import (
    NEAR_POLE,
    STABLE,
    BandMarks,
    label_band,
    mark_unstable_band,
)


def _assert_same_band_marks(actual: BandMarks, expected: BandMarks) -> None:
    np.testing.assert_array_equal(actual.omegas, expected.omegas)
    np.testing.assert_array_equal(actual.labels, expected.labels)
    np.testing.assert_allclose(actual.distances, expected.distances)
    assert actual.danger_radius == pytest.approx(expected.danger_radius)


@pytest.mark.parametrize(
    ("band_cfg", "expected"),
    [
        (
            {
                "n_modes": 3,
                "evp_n_modes": 9,
                "ncv": 11,
                "evp_ncv": 19,
                "target": 1.25,
                "evp_target": -7.0,
            },
            {"n_modes": 3, "ncv": 11, "target": 1.25 + 0.0j},
        ),
        (
            {"evp_n_modes": 4, "evp_ncv": 13, "evp_target": -2.5},
            {"n_modes": 4, "ncv": 13, "target": -2.5 + 0.0j},
        ),
    ],
)
def test_mark_unstable_band_forwards_cfg_aliases_to_evp_seam(
    monkeypatch: pytest.MonkeyPatch,
    band_cfg: dict[str, float],
    expected: dict[str, object],
) -> None:
    captured: dict[str, object] = {}
    known_spectrum = np.array([-1.0 + 0.0j, -2.0 + 0.2j, 4.0 - 0.5j])

    def fake_evp(L0_beta: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        captured["L0_beta"] = L0_beta
        captured.update(kwargs)
        return known_spectrum, np.empty((0, 0), dtype=complex)

    monkeypatch.setattr(band_marking, "_evp_eigenvalues", fake_evp)
    operator = object()
    f_range = np.array([1.0, 2.0, 3.0])
    cfg = {"band_marking": {**band_cfg, "danger_radius": 0.25}}

    marks = mark_unstable_band(operator, f_range, cfg)

    assert captured["L0_beta"] is operator
    assert captured["n_modes"] == expected["n_modes"]
    assert captured["ncv"] == expected["ncv"]
    assert captured["target"] == expected["target"]
    _assert_same_band_marks(marks, label_band(known_spectrum, f_range, cfg))


def test_mark_unstable_band_uses_negated_asymmetric_median_target_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    known_spectrum = np.array([-2.0 + 0.0j])

    def fake_evp(_L0_beta: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        captured.update(kwargs)
        return known_spectrum, np.empty((0, 0), dtype=complex)

    monkeypatch.setattr(band_marking, "_evp_eigenvalues", fake_evp)
    f_range = np.array([1.0, 2.0, 3.0])
    cfg = {"band_marking": {"danger_radius": 1.0e-6}}

    marks = mark_unstable_band(object(), f_range, cfg)

    assert captured["target"] == -np.median(f_range) + 0.0j
    assert captured["n_modes"] == 8
    assert captured["ncv"] is None
    _assert_same_band_marks(marks, label_band(known_spectrum, f_range, cfg))


def test_mark_unstable_band_real_evp_seam_smoke_on_tiny_diagonal_operator() -> None:
    pytest.importorskip(
        "petsc4py",
        reason="SKIPPED: missing petsc4py complex PETSc stack for real _evp_eigenvalues smoke",
    )
    pytest.importorskip(
        "slepc4py",
        reason="SKIPPED: missing slepc4py complex SLEPc stack for real _evp_eigenvalues smoke",
    )

    # stability.eigenvalues solves eig(i*L0).  This diagonal therefore has
    # temporal eigenvalues [-2, -10, 5], so the pole at omega=2 should be marked.
    operator = sp.diags([2.0j, 10.0j, -5.0j], format="csr", dtype=complex)
    f_range = np.array([2.0, 5.0])
    cfg = {"band_marking": {"n_modes": 1, "danger_radius": 1.0e-8}}

    try:
        marks = mark_unstable_band(operator, f_range, cfg)
    except RuntimeError as exc:
        if "PETSc must be built with --with-scalar-type=complex" in str(exc):
            pytest.skip(f"SKIPPED: PETSc/SLEPc stack is not complex-scalar: {exc}")
        raise

    assert isinstance(marks, BandMarks)
    assert marks.omegas.shape == f_range.shape
    assert marks.labels.shape == f_range.shape
    assert marks.distances.shape == f_range.shape
    assert marks.labels.tolist() == [NEAR_POLE, STABLE]
    assert np.all(np.isfinite(marks.distances))
