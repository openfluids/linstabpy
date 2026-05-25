#!/usr/bin/env python3
"""Overlay EVP run spectra on the complex plane with targets.

Usage:
    .venv/bin/python examples/case_studies/case2_lui/phase1p5_plot_spectrum.py \\
        <label>=<npz_path> [<label>=<npz_path> ...]

Produces examples/case_studies/case2_lui/phase1p5/evp_spectra_overview.png
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
        omegas = np.asarray(d.get("omega", np.array([], dtype=complex)))
    ct = cfg.get("analysis", {}).get("omega_target")
    target = complex(ct.get("real", 0.0), ct.get("imag", 0.0)) if isinstance(ct, dict) else None
    return {"omegas": omegas, "target": target, "cfg": cfg}


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("usage: <label>=<npz_path> ...")
        sys.exit(2)
    runs = {}
    for a in args:
        label, path = a.split("=", 1)
        runs[label] = load_run(Path(path))

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.axvline(0, color="k", lw=0.4, alpha=0.4)
    markers = ["o", "s", "^", "D", "v"]
    colors = ["C0", "C1", "C2", "C3"]
    targets_seen = set()
    for i, (label, run) in enumerate(runs.items()):
        w = run["omegas"]
        ax.scatter(
            w.real, w.imag,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            s=80, edgecolor="k", lw=0.6,
            label=f"{label}  ({len(w)} mode{'s' if len(w)!=1 else ''})",
        )
        if run["target"] is not None:
            t = run["target"]
            key = (round(t.real, 4), round(t.imag, 4))
            if key not in targets_seen:
                targets_seen.add(key)
                ax.scatter(
                    t.real, t.imag,
                    marker="x", color=colors[i % len(colors)],
                    s=150, lw=2.0, label=f"target ({label})",
                )
                # circle of radius 0.1*|target|
                r = 0.1 * abs(t)
                theta = np.linspace(0, 2 * np.pi, 128)
                ax.plot(t.real + r * np.cos(theta), t.imag + r * np.sin(theta),
                        color=colors[i % len(colors)], lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"Re $\omega$")
    ax.set_ylabel(r"Im $\omega$")
    ax.set_title("Case 2 EVP spectra — target vs returned modes\n"
                 "(dashed circle = PLAN target-distance gate: 0.1·|ω_target|)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    out = OUT_DIR / "evp_spectra_overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
