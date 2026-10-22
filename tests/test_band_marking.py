# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from linstabpy.solvers.router import band_marking
from linstabpy.solvers.router.band_marking import NEAR_POLE, SMOOTH, STABLE, label_band
from tests._matlab_parity import matlab_parity, require_matlab_reference


def test_label_band_distances_labels_and_threshold_are_config_driven() -> None:
    eigs = np.array([10.0 + 0.0j])
    omegas = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])

    marks = label_band(eigs, omegas, {"danger_radius": 2.0})

    np.testing.assert_allclose(marks.distances, np.abs(omegas - 10.0))
    assert marks.danger_radius == pytest.approx(2.0)
    # Strict '< radius': the two points exactly at the radius remain stable.
    assert marks.labels.tolist() == [
        STABLE,
        STABLE,
        NEAR_POLE,
        NEAR_POLE,
        NEAR_POLE,
        STABLE,
        STABLE,
    ]

    narrower = label_band(eigs, omegas, {"danger_radius": 0.5})
    assert narrower.labels.tolist() == [STABLE, STABLE, STABLE, NEAR_POLE, STABLE, STABLE, STABLE]


def test_label_band_distance_is_monotone_away_from_planted_eigenvalue() -> None:
    eigs = np.array([2.0 + 0.0j])
    omegas = np.array([2.0, 2.01, 2.05, 2.20])

    marks = label_band(eigs, omegas, {"danger_radius": 1.0e-3})

    assert np.all(np.diff(marks.distances) > 0.0)


def test_default_danger_radius_uses_half_median_sweep_spacing_and_fires_near_pole() -> None:
    eigs = np.array([1.3 + 0.0j])
    omegas = np.array([0.0, 1.0, 2.0])

    marks = label_band(eigs, omegas)

    assert marks.danger_radius == pytest.approx(0.5)
    assert marks.distances[1] == pytest.approx(0.3)
    # This same case was inert under the old residual-scale default because
    # the pole distance is much larger than the historical ~1e-8 fallback.
    assert marks.distances[1] > 1.0e-8
    assert marks.labels.tolist() == [STABLE, NEAR_POLE, STABLE]


def test_default_danger_radius_uses_finite_unique_median_spacing() -> None:
    eigs = np.array([1.4 + 0.0j])
    omegas = np.array([np.nan, 0.0, 1.0, 1.0, np.inf, 4.0])

    marks = label_band(eigs, omegas)

    # finite unique grid [0, 1, 4] -> median spacing 2 -> default radius 1.
    assert marks.danger_radius == pytest.approx(1.0)
    assert marks.labels.tolist() == [STABLE, STABLE, NEAR_POLE, NEAR_POLE, STABLE, STABLE]


def test_default_danger_radius_is_zero_when_no_sweep_scale_available() -> None:
    for omegas in ([1.0], [1.0, 1.0, np.nan, np.inf], [np.nan, np.inf]):
        marks = label_band([1.0 + 0.0j], omegas)
        assert marks.danger_radius == pytest.approx(0.0)
        assert not np.any(marks.labels == NEAR_POLE)


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        (
            {
                "danger_radius": 0.0,
                "band_marking": {"danger_radius": 0.25},
                "solver": {"router": {"band_marking": {"danger_radius": 0.5}}},
            },
            0.0,
        ),
        (
            {
                "band_marking": {"danger_radius": 0.25},
                "solver": {"router": {"band_marking": {"danger_radius": 0.5}}},
            },
            0.25,
        ),
        ({"solver": {"router": {"band_marking": {"danger_radius": 0.5}}}}, 0.5),
    ],
)
def test_explicit_danger_radius_precedence_and_zero_value(
    cfg: dict[str, object], expected: float
) -> None:
    marks = label_band([1.0 + 0.0j], [1.0], cfg)

    assert marks.danger_radius == pytest.approx(expected)
    if expected == 0.0:
        assert marks.labels.tolist() == [STABLE]
    else:
        assert marks.labels.tolist() == [NEAR_POLE]


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_explicit_danger_radius_validation_still_applies(value: float) -> None:
    with pytest.raises(ValueError, match="danger_radius must be a finite non-negative float"):
        label_band([-1.0 + 0.0j], [1.0, 2.0], {"danger_radius": value})


def test_label_band_asymmetric_spectrum_uses_resolvent_pole_sign() -> None:
    # Physical convention: eig = 3 + 0.2j is the resolvent pole 3 + 0.2j.
    marks = label_band([3.0 + 0.2j], [3.0, -3.0], {"danger_radius": 0.3})

    np.testing.assert_allclose(marks.distances, [0.2, np.hypot(6.0, 0.2)])
    assert marks.labels.tolist() == [NEAR_POLE, STABLE]


def test_label_band_matlab_parity_places_the_pole_at_minus_eig() -> None:
    # matlab_parity: eig = -3 - 0.2j maps to the resolvent pole 3 + 0.2j.
    marks = label_band([-3.0 - 0.2j], [3.0, -3.0], {"danger_radius": 0.3}, matlab_parity=True)

    np.testing.assert_allclose(marks.distances, [0.2, np.hypot(6.0, 0.2)])
    assert marks.labels.tolist() == [NEAR_POLE, STABLE]
    # The same eigenvalue under the physical convention puts the pole at -3 - 0.2j.
    physical = label_band([-3.0 - 0.2j], [3.0, -3.0], {"danger_radius": 0.3})
    assert physical.labels.tolist() == [STABLE, NEAR_POLE]


def test_label_band_is_pure_and_does_not_call_evp(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("label_band must not call an EVP entry point")

    monkeypatch.setattr(band_marking, "_evp_eigenvalues", fail_if_called)

    marks = label_band([0.0 + 0.0j], [0.0], {"danger_radius": 1.0e-6})

    assert marks.labels.tolist() == [NEAR_POLE]


def test_explicit_radius_label_band_import_does_not_initialize_petsc() -> None:
    code = """
import sys
from linstabpy.solvers.router import band_marking
marks = band_marking.label_band([1+0j], [1.0], {"danger_radius": 0.1})
assert marks.labels.tolist() == ["near-pole"]
assert "petsc4py.PETSc" not in sys.modules, sorted(k for k in sys.modules if k.startswith("petsc4py"))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_label_band_smooth_region_is_explicit_only() -> None:
    eigs = np.array([-10.0 + 0.0j])
    omegas = np.array([0.0, 0.5, 1.0])

    default_marks = label_band(eigs, omegas, {"danger_radius": 0.1})
    configured_marks = label_band(
        eigs,
        omegas,
        {"danger_radius": 0.1, "smooth_regions": [{"min": 0.25, "max": 0.75}]},
    )

    assert default_marks.labels.tolist() == [STABLE, STABLE, STABLE]
    assert configured_marks.labels.tolist() == [STABLE, SMOOTH, STABLE]


def _load_reference_spectrum_from_env() -> np.ndarray:
    ref = os.environ.get("LINSTABPY_TEST_REFERENCE_SPECTRUM")
    if not ref:
        require_matlab_reference(
            Path("LINSTABPY_TEST_REFERENCE_SPECTRUM-is-unset"),
            description="the MATLAB reference spectrum CSV (LINSTABPY_TEST_REFERENCE_SPECTRUM)",
        )
    require_matlab_reference(Path(ref), description="the MATLAB reference spectrum CSV")
    return np.loadtxt(ref, comments="#")


# Compares band marking against the MATLAB reference spectrum, which is not committed.
@matlab_parity
def test_reference_spectrum_marks_unstable_stripe_and_excludes_far_field() -> None:
    data = np.atleast_2d(_load_reference_spectrum_from_env())
    growth = data[:, 1]
    eigs = 1j * growth
    omegas = np.unique(np.r_[np.linspace(-0.2, 0.2, 41), 0.0])

    # The private reference stores the Song 32-deg stationary global-mode growth
    # against spanwise wavelength.
    # Radius 6e-2 must exceed the reference unstable peak (+0.0558).
    # Then the marker labels omega_req=0 for every unstable row.
    radius = 6.0e-2
    zero_index = int(np.flatnonzero(omegas == 0.0)[0])

    unstable = growth > 0.0
    far_field = np.abs(growth) > 0.10

    assert np.any(unstable)
    assert np.any(far_field)
    for eig in eigs[unstable]:
        marks = label_band([eig], omegas, {"danger_radius": radius})
        assert marks.labels[zero_index] == NEAR_POLE

    for eig in eigs[far_field]:
        marks = label_band([eig], omegas, {"danger_radius": radius})
        assert not np.any(marks.labels == NEAR_POLE)
