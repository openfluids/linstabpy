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


def test_label_band_distances_labels_and_threshold_are_config_driven() -> None:
    eigs = np.array([-10.0 + 0.0j])
    omegas = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])

    marks = label_band(eigs, omegas, {"danger_radius": 2.0})

    np.testing.assert_allclose(marks.distances, np.abs(omegas - 10.0))
    assert marks.danger_radius == pytest.approx(2.0)
    # Strict '< radius': the two points exactly at the radius remain stable.
    assert marks.labels.tolist() == [STABLE, STABLE, NEAR_POLE, NEAR_POLE, NEAR_POLE, STABLE, STABLE]

    narrower = label_band(eigs, omegas, {"danger_radius": 0.5})
    assert narrower.labels.tolist() == [STABLE, STABLE, STABLE, NEAR_POLE, STABLE, STABLE, STABLE]


def test_label_band_distance_is_monotone_away_from_planted_eigenvalue() -> None:
    eigs = np.array([-2.0 + 0.0j])
    omegas = np.array([2.0, 2.01, 2.05, 2.20])

    marks = label_band(eigs, omegas, {"danger_radius": 1.0e-3})

    assert np.all(np.diff(marks.distances) > 0.0)


def test_label_band_asymmetric_spectrum_uses_resolvent_pole_sign() -> None:
    # eig = -3 - 0.2j maps to the resolvent pole 3 + 0.2j.
    marks = label_band([-3.0 - 0.2j], [3.0, -3.0], {"danger_radius": 0.3})

    np.testing.assert_allclose(marks.distances, [0.2, np.hypot(6.0, 0.2)])
    assert marks.labels.tolist() == [NEAR_POLE, STABLE]


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
marks = band_marking.label_band([-1+0j], [1.0], {"danger_radius": 0.1})
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
    if not ref or not Path(ref).exists():
        pytest.skip(f"SKIPPED: missing reference CSV LINSTABPY_TEST_REFERENCE_SPECTRUM={ref!r}")
    return np.loadtxt(ref, comments="#")


def test_reference_spectrum_marks_unstable_stripe_and_excludes_far_field() -> None:
    data = np.atleast_2d(_load_reference_spectrum_from_env())
    growth = data[:, 1]
    eigs = 1j * growth
    omegas = np.unique(np.r_[np.linspace(-0.2, 0.2, 41), 0.0])

    # The private oracle stores the Song 32-deg stationary global-mode growth
    # against spanwise wavelength.  Radius 6e-2 must exceed the reference
    # unstable peak (+0.0558) so omega_req=0 is marked for every unstable row.
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
