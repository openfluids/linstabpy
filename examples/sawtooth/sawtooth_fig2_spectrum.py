# SPDX-License-Identifier: GPL-3.0-only
"""Fig 2: temporal spectrum from the sawtooth EVP run npz."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from examples.figure_style import apply, save

SRC = HERE / "sawtooth_evp.npz"
DST = HERE / "sawtooth_fig2_spectrum.png"


def main() -> int:
    apply()
    if not SRC.exists():
        raise FileNotFoundError(f"missing EVP run file: {SRC}")
    data = np.load(SRC)
    if "omega" not in data.files:
        raise KeyError(f"{SRC} has no omega array")
    omega = np.asarray(data["omega"], dtype=complex).reshape(-1)
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.plot(omega.real, omega.imag, "+", ms=5, mew=0.8, color="0.15")
    ax.axhline(0.0, color="0.75", lw=0.5)
    ax.axvline(0.0, color="0.75", lw=0.5)
    ax.set_xlabel(r"$\omega_r$")
    ax.set_ylabel(r"$\omega_i$")
    ax.tick_params(top=True, right=True)
    save(fig, DST)
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
