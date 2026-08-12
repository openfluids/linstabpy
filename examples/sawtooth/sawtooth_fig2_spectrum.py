# SPDX-License-Identifier: GPL-3.0-only
"""Fig 2: temporal spectrum from the sawtooth EVP run npz."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "sawtooth_evp.npz"
DST = HERE / "sawtooth_fig2_spectrum.png"


def main() -> int:
    if not SRC.exists():
        raise FileNotFoundError(f"missing EVP run file: {SRC}")
    data = np.load(SRC)
    if "omega" not in data.files:
        raise KeyError(f"{SRC} has no omega array")
    omega = np.asarray(data["omega"], dtype=complex).reshape(-1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    ax.plot(omega.real, omega.imag, "o", ms=4.0, color="0.15")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.axvline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$\Re(\omega)$")
    ax.set_ylabel(r"$\Im(\omega)$")
    ax.grid(True, alpha=0.3)
    fig.savefig(DST, dpi=300)
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
