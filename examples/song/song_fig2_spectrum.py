#!/usr/bin/env python3
"""Fig S2: Song lead plus the linstabpy spectrum from a run npz."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
QUOTED = HERE / "song_32deg_spectrum_quoted.csv"
DST = HERE / "song_fig2_spectrum.png"


def _quoted() -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    with QUOTED.open(encoding="utf-8") as handle:
        for row in csv.DictReader((line for line in handle if not line.startswith("#"))):
            rows.append((row["source"], float(row["omega_r"]), float(row["omega_i"])))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-npz", type=Path, default=HERE / "song_32deg_paper.npz")
    parser.add_argument("-o", "--output", type=Path, default=DST)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    for name, real, imag in _quoted():
        if name == "song":
            ax.plot(real, imag, "o", ms=8, mfc="none", mec="k", label="Song lead")

    if args.run_npz.exists():
        omega = np.asarray(np.load(args.run_npz)["omega"]).reshape(-1)
        ax.plot(omega.real, omega.imag, "s", ms=5, label="linstabpy")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.axvline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$\omega_r$")
    ax.set_ylabel(r"$\omega_i$")
    ax.set_title(r"Temporal spectrum at $\beta=19.2$")
    ax.legend(frameon=False)
    fig.savefig(args.output, dpi=300)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
