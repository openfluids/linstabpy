# SPDX-License-Identifier: GPL-3.0-only
"""Sponge generators and named presets."""

from __future__ import annotations

import numpy as np
import pytest

from linstabpy.baseflow.sponge import SPONGE_PRESETS, generate_sponge, sponge_from_config


def test_generate_sponge_localized_nonzero() -> None:
    field = generate_sponge(
        (12, 20),
        sponge_type="localized",
        pts_dir=(0, 4, 3, 0),
        amp_dir=(2.0, 2.0, 1.0, 0.0),
    )
    assert field.shape == (12, 20)
    assert float(field.max()) > 0.0
    assert float(np.max(field[: -4, 3:])) == 0.0


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
