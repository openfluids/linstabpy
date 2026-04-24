from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from linstab2d.baseflow import case1_sponge_from_config, generate_case1_sponge, load_hamada_case
from tests._paths import case_fixture


CASE1_DIR = case_fixture("case1")
LEGACY_SPONGE = CASE1_DIR / "workflow" / "sponge_generator.py"


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_case1_sponge", LEGACY_SPONGE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _mat_sponge(path: Path) -> np.ndarray:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "r") as handle:
        sponge = np.asarray(handle["sponge_aux"], dtype=float)
    return sponge


def test_native_sponge_matches_legacy_workflow_generator() -> None:
    legacy = _load_legacy_module()
    shape = (12, 10)
    pts_dir = np.array([2, 3, 4, 1], dtype=int)
    amp_dir = np.array([5.0, 7.0, 11.0, 13.0], dtype=float)

    for sponge_type in ("constant", "parabolic", "localized"):
        native = generate_case1_sponge(
            shape,
            sponge_type=sponge_type,
            pts_dir=pts_dir,
            amp_dir=amp_dir,
        )
        legacy_out = legacy.generate_sponge(np.zeros(shape, dtype=float), sponge_type, pts_dir, amp_dir)
        np.testing.assert_allclose(native, legacy_out, atol=0.0, rtol=0.0)


def test_case1_sponge_presets_match_matlab_exports() -> None:
    stability = _mat_sponge(CASE1_DIR / "MATLAB_exports" / "case_inputs_used.mat")
    resolvent = _mat_sponge(CASE1_DIR / "MATLAB_exports_resolvent" / "case_inputs_used.mat")

    native_stability = case1_sponge_from_config(
        stability.shape,
        {"enabled": True, "preset": "case1_stability"},
    )
    native_resolvent = case1_sponge_from_config(
        resolvent.shape,
        {"enabled": True, "preset": "case1_resolvent"},
    )

    np.testing.assert_allclose(native_stability, stability, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(native_resolvent, resolvent, atol=0.0, rtol=0.0)


def test_load_hamada_case_can_inject_native_sponge() -> None:
    X, Y, baseflow, sponge = load_hamada_case(
        CASE1_DIR,
        transpose_input=True,
        native_sponge_cfg={"enabled": True, "preset": "case1_stability"},
    )
    expected = _mat_sponge(CASE1_DIR / "MATLAB_exports" / "case_inputs_used.mat")
    np.testing.assert_equal(X.shape, expected.shape)
    np.testing.assert_equal(Y.shape, expected.shape)
    np.testing.assert_equal(baseflow.RHO.shape, expected.shape)
    np.testing.assert_allclose(sponge, expected, atol=0.0, rtol=0.0)


def test_smooth_boundary_sponge_is_progressive_toward_selected_edge() -> None:
    nx, ny = 11, 7
    x_line = np.linspace(0.0, 1.0, nx)
    y_line = np.linspace(0.0, 0.6, ny)
    X = np.repeat(x_line[:, None], ny, axis=1)
    Y = np.repeat(y_line[None, :], nx, axis=0)

    sponge = case1_sponge_from_config(
        X.shape,
        {
            "enabled": True,
            "type": "smooth_boundary",
            "width_dir": (0.0, 0.3, 0.0, 0.0),
            "amp_dir": (0.0, 5.0, 0.0, 0.0),
            "profile": "smoothstep",
            "blend": "max",
        },
        X=X,
        Y=Y,
    )

    assert np.all(sponge >= 0.0)
    np.testing.assert_allclose(sponge[-1, :], 5.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(sponge[:7, :], 0.0, atol=1e-12, rtol=0.0)
    diffs = np.diff(sponge[:, ny // 2])
    assert np.all(diffs >= -1e-12)


def test_smooth_localized_sponge_stays_in_target_corner() -> None:
    nx, ny = 21, 17
    x_line = np.linspace(0.0, 1.0, nx)
    y_line = np.linspace(0.0, 0.8, ny)
    X = np.repeat(x_line[:, None], ny, axis=1)
    Y = np.repeat(y_line[None, :], nx, axis=0)

    sponge = case1_sponge_from_config(
        X.shape,
        {
            "enabled": True,
            "type": "smooth_localized",
            "width_frac_dir": (0.0, 0.30, 0.30, 0.0),
            "amp_dir": (0.0, 6.0, 6.0, 0.0),
            "profile": "smootherstep",
        },
        X=X,
        Y=Y,
    )

    assert np.all(sponge >= 0.0)
    np.testing.assert_allclose(float(np.max(sponge)), 6.0, atol=1e-12, rtol=0.0)
    assert np.allclose(sponge[:10, :], 0.0, atol=1e-12)
    assert np.allclose(sponge[:, 6:], 0.0, atol=1e-12)
    assert float(sponge[-1, 0]) > 0.0
    assert float(sponge[-1, ny // 2]) == 0.0
    assert float(sponge[nx // 2, 0]) == 0.0


def test_smooth_boundary_requires_coordinates() -> None:
    with pytest.raises(ValueError, match="requires X and Y"):
        generate_case1_sponge(
            (8, 6),
            sponge_type="smooth_boundary",
            amp_dir=(0.0, 1.0, 0.0, 0.0),
            width_dir=(0.0, 0.2, 0.0, 0.0),
        )
