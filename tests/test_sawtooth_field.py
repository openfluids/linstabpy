# SPDX-License-Identifier: GPL-3.0-only
"""Packed Lui stride-2 field used by examples/sawtooth."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from linstabpy.baseflow.grid_fields import FIELD_NAMES, read_grid_fields

SAWTOOTH = Path(__file__).resolve().parents[1] / "examples" / "sawtooth" / "sawtooth.npz"


def test_sawtooth_npz_is_stride_two_shape() -> None:
    fields = read_grid_fields(SAWTOOTH)
    assert set(fields) == set(FIELD_NAMES)
    for name, arr in fields.items():
        assert arr.shape == (300, 340), name
        assert arr.dtype == np.float64, name
