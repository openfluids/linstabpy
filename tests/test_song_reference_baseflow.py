# SPDX-License-Identifier: GPL-3.0-only
"""The shipped Song base flow reaches the solver in velocity / U_inf scaling."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples.song.prepare_baseflow import main, read_reference_baseflow
from examples.song.reference_paths import BASEFLOW_NPZ


def test_reference_baseflow_velocity_is_scaled_by_freestream_velocity() -> None:
    if not BASEFLOW_NPZ.is_file():
        pytest.skip("song_32deg.npz is not in this checkout")
    fields = read_reference_baseflow()
    with np.load(BASEFLOW_NPZ, allow_pickle=False) as packed:
        mach = float(packed["Ma"])
        for name in ("u", "v", "w"):
            np.testing.assert_allclose(fields[name] * mach, packed[name], rtol=0, atol=1e-14)
        for name in ("x", "y", "rho", "T", "sponge"):
            assert np.array_equal(fields[name], packed[name])
    # Free stream outside the boundary layer: u / U_inf = 1.
    assert abs(float(fields["u"].max()) - 1.0) < 1e-3


def test_from_reference_writes_plots_of_the_fields_it_has(tmp_path: Path) -> None:
    if not BASEFLOW_NPZ.is_file():
        pytest.skip("song_32deg.npz is not in this checkout")
    plt.switch_backend("Agg")
    assert main(["--from-reference", "--output-dir", str(tmp_path)]) == 0
    assert {p.name for p in tmp_path.glob("*.png")} == {"rho.png", "u.png", "v.png", "T.png"}
    assert (tmp_path / "rho.npy").is_file()
