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


@pytest.mark.parametrize("parity", [False, True], ids=["physical", "matlab_parity"])
def test_mark_unstable_band_default_targets_cover_the_range(
    monkeypatch: pytest.MonkeyPatch, parity: bool
) -> None:
    targets: list[complex] = []
    known_spectrum = np.array([-2.0 + 0.0j])

    def fake_evp(_L0_beta: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        targets.append(kwargs["target"])
        assert kwargs["n_modes"] == 8
        assert kwargs["ncv"] is None
        return known_spectrum, np.empty((0, 0), dtype=complex)

    monkeypatch.setattr(band_marking, "_evp_eigenvalues", fake_evp)
    f_range = np.array([1.0, 2.0, 3.0])
    cfg = {"band_marking": {"danger_radius": 1.0}}
    if parity:
        cfg["solver"] = {"resolvent_sign_convention": "matlab_parity"}

    marks = mark_unstable_band(object(), f_range, cfg)

    # Physical poles sit at +omega_eig, matlab_parity poles at -omega_eig: both ends
    # and the centre of the range are searched, the repeated eigenvalue is merged.
    sign = -1.0 if parity else 1.0
    assert sorted(t.real for t in targets) == sorted(sign * f_range)
    _assert_same_band_marks(marks, label_band(known_spectrum, f_range, cfg, matlab_parity=parity))


@pytest.mark.parametrize("parity", [False, True], ids=["physical", "matlab_parity"])
def test_mark_unstable_band_finds_poles_at_the_edges_of_a_wide_range(
    monkeypatch: pytest.MonkeyPatch, parity: bool
) -> None:
    sign = -1.0 if parity else 1.0
    # Poles near the two ends of the range; none near the median.
    spectrum = sign * np.array([1.0 + 0.1j, 99.0 + 0.1j])

    def nearest_modes(_L0_beta: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(np.abs(spectrum - kwargs["target"]))[: kwargs["n_modes"]]
        return spectrum[order], np.empty((0, 0), dtype=complex)

    monkeypatch.setattr(band_marking, "_evp_eigenvalues", nearest_modes)
    f_range = np.linspace(0.0, 100.0, 101)
    cfg = {"band_marking": {"n_modes": 1, "danger_radius": 0.5}}
    if parity:
        cfg["solver"] = {"resolvent_sign_convention": "matlab_parity"}

    marks = mark_unstable_band(object(), f_range, cfg)

    assert marks.labels[1] == NEAR_POLE
    assert marks.labels[99] == NEAR_POLE
    assert marks.distances[1] == pytest.approx(0.1)
    assert marks.distances[99] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("parity", "f_range"),
    [(False, [-2.0, -5.0]), (True, [2.0, 5.0])],
    ids=["physical", "matlab_parity"],
)
def test_mark_unstable_band_real_evp_seam_on_tiny_diagonal_operator(
    parity: bool, f_range: list[float]
) -> None:
    pytest.importorskip(
        "petsc4py",
        reason="SKIPPED: missing petsc4py complex PETSc stack for real _evp_eigenvalues check",
    )
    pytest.importorskip(
        "slepc4py",
        reason="SKIPPED: missing slepc4py complex SLEPc stack for real _evp_eigenvalues check",
    )

    # stability.eigenvalues solves eig(i*L0).  This diagonal therefore has
    # temporal eigenvalues [-2, -10, 5].  The physical pole sits at +omega_eig = -2;
    # under matlab_parity it sits at -omega_eig = +2.  Each case mirrors the other.
    operator = sp.diags([2.0j, 10.0j, -5.0j], format="csr", dtype=complex)
    f_range = np.array(f_range)
    cfg = {"band_marking": {"n_modes": 1, "danger_radius": 1.0e-8}}
    if parity:
        cfg["solver"] = {"resolvent_sign_convention": "matlab_parity"}

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


def test_router_band_marking_reads_sign_from_explicit_solver_cfg(monkeypatch) -> None:
    """The router band marking takes its sign from the solver config of the solves."""
    import linstabpy.solvers.router.band_marking as band_marking
    import importlib

    router_map = importlib.import_module("linstabpy.solvers.router.resolvent_map")

    seen: dict[str, object] = {}

    def capture(L0_beta, f_range, cfg=None, *, matlab_parity=None):
        seen["matlab_parity"] = matlab_parity

    monkeypatch.setattr(band_marking, "mark_unstable_band", capture)
    cfg = {"solver_cfg": {"resolvent_sign_convention": "matlab_parity"}}
    router_map._mark_unstable_band(None, [1.0, 2.0], cfg)
    assert seen["matlab_parity"] is True
    router_map._mark_unstable_band(None, [1.0, 2.0], {"solver": {}})
    assert seen["matlab_parity"] is False


def test_band_marking_targets_include_the_range_centre() -> None:
    """An even target count would leave out the centre of the range."""
    from linstabpy.solvers.router.band_marking import _evp_targets

    omegas = np.linspace(0.0, 30.0, 61)
    # radius 1.25 gives 4 targets and 0.75 gives 6 before the count is made odd.
    for radius in (0.0, 0.75, 1.25):
        targets = _evp_targets(omegas, radius, 8, 1.0)
        assert any(abs(t.real - 15.0) < 1e-12 for t in targets), (radius, targets)
