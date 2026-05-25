#!/usr/bin/env python3
"""Plot the resolvent gain curve σ(ω) from a scan campaign.

Usage:
    .venv/bin/python examples/case_studies/case2_lui/phase1p5_plot_resolvent_gain.py \\
        <campaign-runs-dir>

Outputs:
  - phase1p5/resolvent_<campaign>_gain_curve.png — σ₁(ω), σ₂(ω), ...
  - phase1p5/resolvent_<campaign>_gain_table.txt — tabular data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
OUT_DIR = CASE_DIR / "artifacts" / "phase1p5"
OUT_DIR.mkdir(exist_ok=True)


def load_run(npz: Path) -> dict:
    cfg = json.loads(npz.with_suffix(".config.json").read_text())
    with np.load(npz, allow_pickle=True) as d:
        gains = np.asarray(d.get("gains", np.array([])))
        omega_target = complex(d.get("omega_target", 0)) if "omega_target" in d.files else None
    ct = cfg.get("analysis", {}).get("omega_target")
    if omega_target is None and isinstance(ct, dict):
        omega_target = complex(ct.get("real", 0.0), ct.get("imag", 0.0))
    return {"npz": npz, "omega": omega_target, "gains": gains}


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: phase1p5_plot_resolvent_gain.py <runs_dir>")
        sys.exit(2)
    runs_dir = Path(sys.argv[1])
    npz_paths = sorted(runs_dir.glob("*.npz"))
    runs = [load_run(p) for p in npz_paths]
    runs = [r for r in runs if r["omega"] is not None and r["gains"].size > 0]
    if not runs:
        print(f"No valid resolvent runs in {runs_dir}")
        sys.exit(1)
    runs.sort(key=lambda r: r["omega"].real)

    omegas = np.array([r["omega"].real for r in runs])
    max_modes = max(len(r["gains"]) for r in runs)
    gain_matrix = np.full((len(runs), max_modes), np.nan)
    for i, r in enumerate(runs):
        gain_matrix[i, : len(r["gains"])] = r["gains"]

    print(f"{'ω':>8s}  {'St=ω/2π':>8s}  " + "  ".join(f"σ_{k + 1}" for k in range(max_modes)))
    for i, r in enumerate(runs):
        row = f"{omegas[i]:8.3f}  {omegas[i] / (2 * np.pi):8.4f}  "
        row += "  ".join(f"{g:7.3e}" for g in r["gains"])
        print(row)

    campaign = runs_dir.parent.name
    # Gain curve plot
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for k in range(max_modes):
        mask = ~np.isnan(gain_matrix[:, k])
        if mask.any():
            ax.plot(
                omegas[mask],
                gain_matrix[mask, k],
                marker="o",
                color=colors[k % len(colors)],
                lw=1.5,
                label=f"σ_{k + 1}",
            )
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"singular value $\sigma_k$")
    ax.set_yscale("log")
    ax.set_title(f"Resolvent gain curve — {campaign}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Annotate likely peak
    sigma1 = gain_matrix[:, 0]
    peak_idx = int(np.nanargmax(sigma1))
    ax.axvline(omegas[peak_idx], color="k", ls="--", lw=0.8, alpha=0.5)
    ax.annotate(
        f"σ₁ peak:\nω={omegas[peak_idx]:.3f}\nσ={sigma1[peak_idx]:.3g}",
        xy=(omegas[peak_idx], sigma1[peak_idx]),
        xytext=(10, -40),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="k", alpha=0.8),
    )

    # Also mark Dussauge SWBLI low-frequency range for context
    ax.axvspan(0.2 * 2 * np.pi, 0.05 * 2 * np.pi, alpha=0.1, color="green", label="_nolegend_")
    # Dussauge St 0.02–0.05 → ω 0.126–0.314
    ax.axvspan(0.02 * 2 * np.pi, 0.05 * 2 * np.pi, alpha=0.15, color="green")
    ax.text(
        0.03 * 2 * np.pi,
        ax.get_ylim()[0] * 2,
        "Dussauge\nSWBLI range",
        fontsize=8,
        color="darkgreen",
        alpha=0.8,
    )

    out = OUT_DIR / f"resolvent_{campaign}_gain_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}")

    # Tabular output
    tbl = OUT_DIR / f"resolvent_{campaign}_gain_table.txt"
    with tbl.open("w") as f:
        f.write(f"Resolvent gain curve — {campaign}\n")
        f.write("=" * 60 + "\n\n")
        f.write(
            f"{'ω':>8s}  {'St=ω/2π':>8s}  "
            + "  ".join(f"σ_{k + 1:<8d}" for k in range(max_modes))
            + "\n"
        )
        for i, r in enumerate(runs):
            row = f"{omegas[i]:8.3f}  {omegas[i] / (2 * np.pi):8.4f}  "
            row += "  ".join(f"{g:9.3e}" for g in r["gains"])
            f.write(row + "\n")
        f.write(
            f"\nσ₁ peak at ω = {omegas[peak_idx]:.4f}  "
            f"(St = {omegas[peak_idx] / (2 * np.pi):.4f})  "
            f"σ₁ = {sigma1[peak_idx]:.4e}\n"
        )
    print(f"Wrote {tbl}")


if __name__ == "__main__":
    main()
