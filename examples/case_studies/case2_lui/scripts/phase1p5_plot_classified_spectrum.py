#!/usr/bin/env python3
"""Paper-ready spectrum plot with modes color-coded by Chu-energy class.

Usage:
    .venv/bin/python examples/case_studies/case2_lui/phase1p5_plot_classified_spectrum.py \\
        <npz> [<npz> ...]
Writes phase1p5/<basename>_classified_spectrum.png per input and a
combined overlay if multiple inputs are passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1p5_classify_modes import classify_one  # noqa: E402

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
OUT_DIR = CASE_DIR / "artifacts" / "phase1p5"
OUT_DIR.mkdir(exist_ok=True)

CLASS_COLOR = {
    "WALL_LOCAL": "#d62728",  # red — contamination
    "GLOBAL": "#2ca02c",  # green — physical
    "INTERMEDIATE": "#7f7f7f",  # gray
    "ZERO_ENERGY": "#000000",
}
CLASS_MARKER = {
    "WALL_LOCAL": "X",
    "GLOBAL": "o",
    "INTERMEDIATE": "s",
    "ZERO_ENERGY": ".",
}


def load_and_classify(npz: Path) -> list[dict]:
    d_obj = np.load(npz, allow_pickle=True)
    d = {k: d_obj[k] for k in d_obj.files}
    n = d["direct_modes"].shape[1] if d["direct_modes"].ndim == 2 else 0
    rows = []
    for j in range(n):
        r = classify_one(d, j)
        r["j"] = j
        r["npz"] = npz.name
        rows.append(r)
    return rows


def plot_one(rows: list[dict], title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.axvline(0, color="k", lw=0.4, alpha=0.4)

    # Dussauge SWBLI band (St 0.02–0.05 → |Re ω| 0.126–0.314)
    ax.axvspan(0.02 * 2 * np.pi, 0.05 * 2 * np.pi, alpha=0.08, color="green")
    ax.axvspan(-0.05 * 2 * np.pi, -0.02 * 2 * np.pi, alpha=0.08, color="green")
    ax.text(
        0.035 * 2 * np.pi,
        ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.5,
        "Dussauge\nSWBLI",
        fontsize=7,
        color="darkgreen",
        ha="center",
    )

    classes_seen = set()
    for r in rows:
        c = r["klass"]
        w = r["omega"]
        ax.scatter(
            w.real,
            w.imag,
            marker=CLASS_MARKER.get(c, "s"),
            color=CLASS_COLOR.get(c, "gray"),
            s=100,
            edgecolor="k",
            lw=0.5,
            label=c if c not in classes_seen else None,
            zorder=3,
        )
        classes_seen.add(c)
        ax.annotate(
            f"{r['j']}",
            xy=(w.real, w.imag),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            alpha=0.6,
            zorder=4,
        )

    ax.set_xlabel(r"Re $\omega$ (frequency)")
    ax.set_ylabel(r"Im $\omega$ (-growth rate)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: phase1p5_plot_classified_spectrum.py <npz> [...]")
        sys.exit(2)
    all_rows: list[dict] = []
    for p in sys.argv[1:]:
        npz = Path(p)
        rows = load_and_classify(npz)
        all_rows.extend(rows)
        plot_one(
            rows,
            f"Case 2 M2 Qraw — classified EVP spectrum\n{npz.stem}",
            OUT_DIR / f"{npz.stem}_classified_spectrum.png",
        )

    if len(sys.argv) > 2:
        plot_one(
            all_rows,
            "Case 2 M2 Qraw — combined classified spectrum across probes",
            OUT_DIR / "combined_classified_spectrum.png",
        )
        # Count classes
        from collections import Counter

        counts = Counter(r["klass"] for r in all_rows)
        total = len(all_rows)
        print("\nCombined count (all modes across inputs):")
        for k, v in counts.most_common():
            print(f"  {k}: {v} ({v / total * 100:.0f}%)")


if __name__ == "__main__":
    main()
