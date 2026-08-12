#!/usr/bin/env python3
"""Fig S2: Song lead plus the packed linstabpy spectrum CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.figure_style import apply, save

HERE = Path(__file__).resolve().parent
QUOTED = HERE / "song_32deg_spectrum_quoted.csv"
SPECTRUM = HERE / "song_32deg_spectrum.csv"
DST = HERE / "song_fig2_spectrum.png"


def _quoted() -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    with QUOTED.open(encoding="utf-8") as handle:
        for row in csv.DictReader((line for line in handle if not line.startswith("#"))):
            rows.append((row["source"], float(row["omega_r"]), float(row["omega_i"])))
    return rows


def _spectrum(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader((line for line in handle if not line.startswith("#"))):
            rows.append((float(row["omega_r"]), float(row["omega_i"])))
    return rows


def main() -> int:
    apply()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", type=Path, default=SPECTRUM)
    parser.add_argument("-o", "--output", type=Path, default=DST)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for name, real, imag in _quoted():
        if name == "song":
            ax.plot(
                real,
                imag,
                "o",
                ms=6,
                mfc="none",
                mec="0.1",
                mew=1.0,
                label="Song",
                zorder=3,
            )

    if args.spectrum.exists():
        omega = _spectrum(args.spectrum)
        ax.plot(
            [row[0] for row in omega],
            [row[1] for row in omega],
            "+",
            ms=5,
            mew=0.8,
            color="0.15",
            label="linstabpy",
            zorder=2,
        )
    ax.axhline(0.0, color="0.75", lw=0.5)
    ax.axvline(0.0, color="0.75", lw=0.5)
    ax.set_xlabel(r"$\omega_r$")
    ax.set_ylabel(r"$\omega_i$")
    ax.tick_params(top=True, right=True)
    ax.legend(loc="upper right", handletextpad=0.4)
    save(fig, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
