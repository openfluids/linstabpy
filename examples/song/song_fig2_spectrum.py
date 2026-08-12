#!/usr/bin/env python3
"""Fig S2: quoted lead eigenvalues at beta=19.2, plus Song's omega_i curve."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
QUOTED = HERE / "song_32deg_spectrum_quoted.csv"
DISPERSION = HERE / "song_omega_i.csv"
DST = HERE / "song_fig2_spectrum.png"


def _quoted() -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    with QUOTED.open(encoding="utf-8") as handle:
        for row in csv.DictReader(
            (line for line in handle if not line.startswith("#"))
        ):
            rows.append((row["source"], float(row["omega_r"]), float(row["omega_i"])))
    return rows


def _dispersion() -> tuple[np.ndarray, np.ndarray]:
    lam = []
    imag = []
    with DISPERSION.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            left, right = [item.strip() for item in text.split(",")]
            lam.append(float(left))
            imag.append(float(right))
    lam_arr = np.asarray(lam, dtype=float)
    beta = 2.0 * np.pi / lam_arr
    return beta, np.asarray(imag, dtype=float)


def main() -> int:
    pairs = _quoted()
    beta, omega_i = _dispersion()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)

    ax = axes[0]
    markers = {"song": "o", "linstabpy": "s"}
    for name, real, imag in pairs:
        ax.plot(real, imag, markers.get(name, "x"), label=name, ms=8)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.axvline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$\omega_r$")
    ax.set_ylabel(r"$\omega_i$")
    ax.set_title(r"Quoted lead pair at $\beta=19.2$")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(beta, omega_i, "o-", ms=4)
    ax.axvline(19.2, color="0.4", ls="--", lw=0.8)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$\beta X_{\mathrm{sh}}$")
    ax.set_ylabel(r"$\omega_i$")
    ax.set_title("Song growth-rate curve")

    fig.savefig(DST, dpi=300)
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
