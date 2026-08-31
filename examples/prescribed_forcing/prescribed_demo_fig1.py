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

    # Read CSV.
    import csv

    records = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                omega = float(row["omega"])
                gain = float(row["gain"]) if row["gain"] and row["gain"] != "None" else None
                if gain is not None:
                    records.append((omega, gain))
            except (ValueError, KeyError):
                pass

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
    ax.set_ylabel(r"Gain $\|\hat q\|_W / \|\hat f\|_W$", fontsize=11)
    ax.set_title("Prescribed-Forcing Resolvent Response", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"Saved {OUTPUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
