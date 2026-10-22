# SPDX-License-Identifier: GPL-3.0-only
"""Sponge generators and named presets."""

from __future__ import annotations

import numpy as np
import pytest

from linstabpy.baseflow.sponge import (
    SPONGE_PRESETS,
    _boundary_distance_fields,
    _cumulative_distances,
    _smooth_profile,
    generate_sponge,
    load_sponge_aux,
    sponge_from_config,
)


def test_generate_sponge_localized_nonzero() -> None:
    field = generate_sponge(
        (12, 20),
        sponge_type="localized",
        pts_dir=(0, 4, 3, 0),
        amp_dir=(2.0, 2.0, 1.0, 0.0),
    )
    assert field.shape == (12, 20)
    assert float(field.max()) > 0.0
    assert float(np.max(field[:-4, 3:])) == 0.0


def test_sponge_presets_drop_case1_names() -> None:
    assert "localized_stability" in SPONGE_PRESETS
    assert "smooth_soft" in SPONGE_PRESETS
    assert not any(name.startswith("case1_") for name in SPONGE_PRESETS)


def test_sponge_from_config_uses_preset() -> None:
    x = np.linspace(0.0, 1.0, 16)
    y = np.linspace(0.0, 0.5, 10)
    X, Y = np.meshgrid(x, y, indexing="xy")
    field = sponge_from_config(
        X.shape,
        {"enabled": True, "preset": "localized_stability"},
        X=X,
        Y=Y,
    )
    assert field.shape == X.shape
    assert float(field.max()) > 0.0


def test_generate_sponge_constant_and_parabolic() -> None:
    const = generate_sponge(
        (8, 10),
        sponge_type="constant",
        pts_dir=(2, 0, 0, 0),
        amp_dir=(3.0, 0.0, 0.0, 0.0),
    )
    assert float(const[0, -1]) == 3.0
    assert float(const[0, 0]) == 0.0
    para = generate_sponge(
        (8, 10),
        sponge_type="parabolic",
        pts_dir=(0, 0, 3, 0),
        amp_dir=(0.0, 0.0, 4.0, 0.0),
    )
    assert float(para.max()) == 4.0
    assert float(para[0, -1]) == 0.0


def test_generate_sponge_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="sponge_type"):
        generate_sponge((4, 4), sponge_type="not_a_type")


def test_sponge_from_config_unknown_preset() -> None:
    with pytest.raises(ValueError, match="preset"):
        sponge_from_config((4, 4), {"enabled": True, "preset": "case1_stability"})


def _xy(nx: int = 21, ny: int = 13) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 0.5, ny)
    return np.meshgrid(x, y, indexing="xy")


def test_smooth_soft_preset_is_nonzero_only_near_active_edges() -> None:
    X, Y = _xy()
    field = sponge_from_config(
        X.shape,
        {"enabled": True, "preset": "smooth_soft"},
        X=X,
        Y=Y,
    )
    assert field.shape == X.shape
    assert float(field.max()) > 0.0
    # smooth_soft damps right (last rows) and bottom (first columns).
    # The opposite corner sits outside both strips.
    assert float(field[0, -1]) == pytest.approx(0.0, abs=1e-12)
    assert float(field[-1, 0]) == pytest.approx(float(field.max()), rel=1e-12, abs=1e-12)


def test_smooth_localized_requires_matching_xy() -> None:
    X, Y = _xy(8, 6)
    with pytest.raises(ValueError, match="requires X and Y"):
        generate_sponge(X.shape, sponge_type="smooth_localized", amp_dir=(0.0, 1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="X/Y shapes must match"):
        generate_sponge(
            X.shape,
            sponge_type="smooth_localized",
            amp_dir=(0.0, 1.0, 0.0, 0.0),
            width_dir=(0.0, 0.2, 0.0, 0.0),
            X=X,
            Y=Y[:, :4],
        )


def test_smooth_width_dir_must_be_four_nonnegative() -> None:
    X, Y = _xy(6, 5)
    with pytest.raises(ValueError, match="width_dir"):
        generate_sponge(
            X.shape,
            sponge_type="smooth",
            width_dir=(0.1, 0.1, 0.1),
            amp_dir=(1.0, 1.0, 1.0, 1.0),
            X=X,
            Y=Y,
        )
    with pytest.raises(ValueError, match="non-negative"):
        generate_sponge(
            X.shape,
            sponge_type="smooth",
            width_dir=(0.1, -0.2, 0.0, 0.0),
            amp_dir=(1.0, 1.0, 0.0, 0.0),
            X=X,
            Y=Y,
        )


def test_smooth_profile_and_blend_loud_fail() -> None:
    X, Y = _xy(6, 5)
    kwargs = {
        "shape": X.shape,
        "sponge_type": "smooth",
        "width_dir": (0.0, 0.2, 0.2, 0.0),
        "amp_dir": (0.0, 1.0, 1.0, 0.0),
        "X": X,
        "Y": Y,
    }
    with pytest.raises(ValueError, match="steepness"):
        generate_sponge(**kwargs, profile="tanh", steepness=0.0)
    with pytest.raises(ValueError, match="profile"):
        generate_sponge(**kwargs, profile="linear")
    with pytest.raises(ValueError, match="blend"):
        generate_sponge(**kwargs, blend="min")
    with pytest.raises(ValueError, match="power"):
        generate_sponge(**kwargs, power=0.0)


@pytest.mark.parametrize("profile", ["smoothstep", "smootherstep", "cosine", "tanh"])
def test_smooth_boundary_ramp_is_monotone_along_one_edge(profile: str) -> None:
    X, Y = _xy(16, 12)
    field = generate_sponge(
        X.shape,
        sponge_type="smooth_boundary",
        width_dir=(0.0, 0.25, 0.0, 0.0),
        amp_dir=(0.0, 2.0, 0.0, 0.0),
        profile=profile,
        X=X,
        Y=Y,
    )
    strip = field[:, X.shape[1] // 2]
    assert float(strip.min()) >= 0.0
    assert float(strip.max()) <= 2.0 + 1e-12
    assert np.all(np.diff(strip) >= -1e-12)
    assert float(strip[-1]) == pytest.approx(2.0, rel=1e-12, abs=1e-12)
    assert float(strip[0]) == pytest.approx(0.0, abs=1e-12)


def test_smooth_boundary_additive_blend_exceeds_max_in_overlap() -> None:
    X, Y = _xy(18, 14)
    kwargs = {
        "shape": X.shape,
        "sponge_type": "smooth",
        "width_dir": (0.0, 0.3, 0.3, 0.0),
        "amp_dir": (0.0, 1.0, 1.0, 0.0),
        "X": X,
        "Y": Y,
        "profile": "smoothstep",
    }
    field_max = generate_sponge(**kwargs, blend="max")
    field_add = generate_sponge(**kwargs, blend="additive")
    field_rss = generate_sponge(**kwargs, blend="quadrature")
    np.testing.assert_allclose(np.maximum(field_max, field_add), field_add, atol=1e-12)
    overlap = (field_max > 1e-12) & (field_add > field_max + 1e-12)
    assert bool(np.any(overlap))
    np.testing.assert_allclose(np.maximum(field_max, field_rss), field_rss, atol=1e-12)
    assert float(np.max(field_rss - field_max)) > 0.0


def test_sponge_from_config_disabled_is_zero() -> None:
    zeros = sponge_from_config((5, 7), None)
    np.testing.assert_array_equal(zeros, np.zeros((5, 7)))
    disabled = sponge_from_config((5, 7), {"enabled": False, "preset": "smooth_soft"})
    np.testing.assert_array_equal(disabled, np.zeros((5, 7)))


def test_sponge_from_config_without_preset_uses_explicit_type() -> None:
    field = sponge_from_config(
        (6, 8),
        {
            "enabled": True,
            "type": "constant",
            "pts_dir": (0, 2, 0, 0),
            "amp_dir": (0.0, 5.0, 0.0, 0.0),
        },
    )
    assert float(field[-1, 3]) == 5.0
    assert float(field[0, 3]) == 0.0


def test_constant_sponge_covers_each_edge() -> None:
    right = generate_sponge(
        (6, 8), sponge_type="constant", pts_dir=(0, 2, 0, 0), amp_dir=(0, 5, 0, 0)
    )
    assert float(right[-1, 3]) == 5.0
    assert float(right[0, 3]) == 0.0
    bottom = generate_sponge(
        (6, 8), sponge_type="constant", pts_dir=(0, 0, 3, 0), amp_dir=(0, 0, 4, 0)
    )
    assert float(bottom[2, 0]) == 4.0
    assert float(bottom[2, -1]) == 0.0
    left = generate_sponge(
        (6, 8), sponge_type="constant", pts_dir=(0, 0, 0, 2), amp_dir=(0, 0, 0, 7)
    )
    assert float(left[0, 3]) == 7.0
    assert float(left[-1, 3]) == 0.0


def test_parabolic_sponge_peaks_on_the_damped_edge() -> None:
    top = generate_sponge(
        (8, 10), sponge_type="parabolic", pts_dir=(3, 0, 0, 0), amp_dir=(9.0, 0, 0, 0)
    )
    assert float(top[0, -1]) == pytest.approx(9.0)
    assert float(top[0, 0]) == pytest.approx(0.0)
    right = generate_sponge(
        (8, 10), sponge_type="parabolic", pts_dir=(0, 2, 0, 0), amp_dir=(0, 4.0, 0, 0)
    )
    assert float(right[-1, 4]) == pytest.approx(4.0)
    left = generate_sponge(
        (8, 10), sponge_type="parabolic", pts_dir=(0, 0, 0, 2), amp_dir=(0, 0, 0, 1.0)
    )
    assert float(left[0, 4]) == pytest.approx(1.0)


def test_smooth_localized_zero_widths_return_zeros() -> None:
    X, Y = _xy(6, 5)
    field = generate_sponge(
        X.shape,
        sponge_type="smooth_localized",
        width_dir=(0.0, 0.0, 0.0, 0.0),
        amp_dir=(1.0, 1.0, 1.0, 1.0),
        X=X,
        Y=Y,
    )
    np.testing.assert_array_equal(field, np.zeros_like(X))


def test_cumulative_distances_need_a_mesh_axis() -> None:
    X, Y = _xy(4, 3)
    with pytest.raises(ValueError, match="axis"):
        _cumulative_distances(X, Y, axis=2)


def test_boundary_distance_fields_need_matching_xy() -> None:
    X, Y = _xy(4, 3)
    with pytest.raises(ValueError, match="matching shapes"):
        _boundary_distance_fields(X, Y[:, :2])


def test_smooth_profile_with_zero_width_is_zero() -> None:
    distance = np.linspace(0.0, 1.0, 5)
    width = np.zeros_like(distance)
    ramp = _smooth_profile(distance, width, profile="smoothstep", power=1.0, steepness=4.0)
    np.testing.assert_array_equal(ramp, np.zeros_like(distance))


def test_pts_dir_must_be_four_nonnegative() -> None:
    with pytest.raises(ValueError, match="pts_dir"):
        generate_sponge((4, 4), sponge_type="constant", pts_dir=(1, 1, 1))
    with pytest.raises(ValueError, match="non-negative"):
        generate_sponge((4, 4), sponge_type="constant", pts_dir=(1, -1, 0, 0))


def test_amp_dir_and_width_frac_must_be_four_nonnegative() -> None:
    X, Y = _xy(6, 5)
    with pytest.raises(ValueError, match="amp_dir"):
        generate_sponge((4, 4), sponge_type="constant", amp_dir=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="width_frac_dir"):
        generate_sponge(
            X.shape,
            sponge_type="smooth_localized",
            width_frac_dir=(0.1, 0.1, 0.1),
            amp_dir=(0.0, 1.0, 0.0, 0.0),
            X=X,
            Y=Y,
        )
    with pytest.raises(ValueError, match="non-negative"):
        generate_sponge(
            X.shape,
            sponge_type="smooth_localized",
            width_frac_dir=(0.1, -0.2, 0.0, 0.0),
            amp_dir=(0.0, 1.0, 0.0, 0.0),
            X=X,
            Y=Y,
        )


def test_localized_sponge_covers_top_and_left() -> None:
    top = generate_sponge(
        (6, 8), sponge_type="localized", pts_dir=(3, 0, 0, 0), amp_dir=(5.0, 0, 0, 0)
    )
    assert float(top[2, -1]) == 5.0
    assert float(top[2, 0]) == 0.0
    left = generate_sponge(
        (6, 8), sponge_type="localized", pts_dir=(0, 0, 0, 2), amp_dir=(4.0, 0, 0, 0)
    )
    assert float(left[0, 3]) == 4.0
    assert float(left[-1, 3]) == 0.0


def test_smooth_power_steepens_the_ramp() -> None:
    X, Y = _xy(16, 12)
    kwargs = {
        "shape": X.shape,
        "sponge_type": "smooth_boundary",
        "width_dir": (0.0, 0.25, 0.0, 0.0),
        "amp_dir": (0.0, 1.0, 0.0, 0.0),
        "profile": "smoothstep",
        "X": X,
        "Y": Y,
    }
    linear = generate_sponge(**kwargs, power=1.0)
    steep = generate_sponge(**kwargs, power=2.0)
    strip = linear[:, 4]
    interior = np.where((strip > 0.1) & (strip < 0.9))[0]
    assert interior.size > 0
    i = int(interior[0])
    assert 0.0 < float(steep[i, 4]) < float(linear[i, 4])
    assert float(steep[-1, 4]) == pytest.approx(1.0)


def test_load_sponge_aux_reads_hdf5_and_loud_fails(tmp_path, monkeypatch) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "case.mat"
    expected = np.arange(12.0).reshape(3, 4)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sponge_aux", data=expected)
    np.testing.assert_allclose(load_sponge_aux(path), expected)

    empty = tmp_path / "empty.mat"
    with h5py.File(empty, "w") as handle:
        handle.create_dataset("other", data=expected)
    with pytest.raises(KeyError, match="sponge_aux"):
        load_sponge_aux(empty)

    rank1 = tmp_path / "rank1.mat"
    with h5py.File(rank1, "w") as handle:
        handle.create_dataset("sponge_aux", data=np.arange(4.0))
    with pytest.raises(ValueError, match="2-D"):
        load_sponge_aux(rank1)

    monkeypatch.delitem(__import__("sys").modules, "h5py", raising=False)

    class _BlockH5py:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "h5py":
                raise ModuleNotFoundError("h5py")
            return None

    import sys

    monkeypatch.setattr(sys, "meta_path", [_BlockH5py(), *sys.meta_path])
    with pytest.raises(ModuleNotFoundError, match="h5py"):
        load_sponge_aux(path)


@pytest.mark.parametrize("sponge_type", ["constant", "parabolic"])
def test_edge_sponges_do_not_erase_each_other(sponge_type: str) -> None:
    pts = (0, 3, 3, 0)
    both = generate_sponge((12, 10), sponge_type=sponge_type, pts_dir=pts, amp_dir=(0, 1, 1, 0))
    right = generate_sponge(
        (12, 10), sponge_type=sponge_type, pts_dir=(0, 3, 0, 0), amp_dir=(0, 1, 0, 0)
    )
    bottom = generate_sponge(
        (12, 10), sponge_type=sponge_type, pts_dir=(0, 0, 3, 0), amp_dir=(0, 0, 1, 0)
    )
    np.testing.assert_array_equal(both, np.maximum(right, bottom))
