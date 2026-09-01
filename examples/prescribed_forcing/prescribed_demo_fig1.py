#!/usr/bin/env python3
"""
Plot gain vs omega from prescribed_demo.py CSV output.

USAGE:   python prescribed_demo_fig1.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "prescribed_demo_gains.csv"
OUTPUT_PNG = SCRIPT_DIR / "prescribed_demo_fig1.png"


def main():
    """Plot gain vs omega from the CSV."""
    if not CSV_PATH.exists():
        print(f"{CSV_PATH} not found; run prescribed_demo.py first")
        return

    # Read CSV. Skip a row with a missing column or text in a numeric
    # column.
    import contextlib
    import csv

    records = []
    with Path(CSV_PATH).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            with contextlib.suppress(ValueError, KeyError):
                omega = float(row["omega"])
                gain = float(row["gain"]) if row["gain"] and row["gain"] != "None" else None
                if gain is not None:
                    records.append((omega, gain))

    if not records:
        print("No valid gain data in CSV")
        return

    omegas, gains = zip(*records)
    omegas = np.asarray(omegas)
    gains = np.asarray(gains)

    # Plot.
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.semilogy(omegas, gains, "o-", linewidth=1.5, markersize=6, label="Prescribed gain")
    ax.set_xlabel(r"Frequency $\omega$ (nondimensional)", fontsize=11)
    # The demo sets forcing_norm "l2": Chu-weighted response over plain L2
    # forcing norm, so the label names the two norms it actually plots.
    ax.set_ylabel(r"Gain $\|\hat q\|_W / \|\hat f\|_2$", fontsize=11)
    ax.set_title("Prescribed-Forcing Resolvent Response", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"Saved {OUTPUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
