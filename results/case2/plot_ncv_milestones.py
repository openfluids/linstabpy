#!/usr/bin/env python3
"""Plot Case 2 Krylov-space saturation milestones."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parent
OUTPUT_DIR = CASE_DIR / "output" / "ncv_milestones"
SUMMARY_CSV = OUTPUT_DIR / "case2_ncv_milestones_summary.csv"
SUMMARY_PNG = OUTPUT_DIR / "case2_ncv_milestones_summary.png"
SPECTRUM_PNG = OUTPUT_DIR / "case2_ncv_milestones_spectra.png"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    rows = _load_rows(SUMMARY_CSV)
    if not rows:
        raise SystemExit(f"no rows in {SUMMARY_CSV}")

    fd_values = sorted({int(row["fd_order"]) for row in rows})
    for fd in fd_values:
        fd_rows = [row for row in rows if int(row["fd_order"]) == fd]
        fd_rows.sort(key=lambda row: (int(row.get("stage_index", 0)), int(row["ncv"])))
        ncv = np.array([int(row["ncv"]) for row in fd_rows], dtype=float)
        requested = np.array([int(row["requested_nev"]) for row in fd_rows], dtype=float)
        post_1e6 = np.array([int(row["post_rel_count_le_1e6"]) for row in fd_rows], dtype=float)
        post_1e5 = np.array([int(row["post_rel_count_le_1e5"]) for row in fd_rows], dtype=float)
        nconv = np.array([int(row["nconv"]) for row in fd_rows], dtype=float)
        elapsed = np.array([float(row["stage_seconds"]) / 60.0 for row in fd_rows], dtype=float)
        stage_idx = np.arange(len(fd_rows), dtype=float)
        stage_labels = [
            f"{row.get('stage_name', f'ncv{row['ncv']}')}\nnev={row['requested_nev']} ncv={row['ncv']}\ntol={float(row['eps_tol']):.0e}"
            for row in fd_rows
        ]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
        axes[0].plot(stage_idx, requested, marker="o", linewidth=2, color="#9f9f9f", label="requested nev")
        axes[0].plot(stage_idx, nconv, marker="o", linewidth=2, color="#00798c", label="EPS nconv")
        axes[0].plot(stage_idx, post_1e5, marker="s", linewidth=2, color="#c76b00", label="postcheck <= 1e-5")
        axes[0].plot(stage_idx, post_1e6, marker="^", linewidth=2, color="#7a3e9d", label="postcheck <= 1e-6")
        axes[0].set_title(f"FD{fd} Accepted Modes by Stage")
        axes[0].set_xlabel("Stage")
        axes[0].set_ylabel("Mode count")
        axes[0].set_xticks(stage_idx, stage_labels, rotation=35, ha="right")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        axes[1].plot(stage_idx, elapsed, marker="o", linewidth=2, color="#1f3b4d")
        axes[1].set_title(f"FD{fd} Runtime by Stage")
        axes[1].set_xlabel("Stage")
        axes[1].set_ylabel("Stage minutes")
        axes[1].set_xticks(stage_idx, stage_labels, rotation=35, ha="right")
        axes[1].grid(True, alpha=0.25)

        fig.savefig(OUTPUT_DIR / f"case2_fd{fd}_ncv_summary.png", dpi=220)
        if fd == fd_values[0]:
            fig.savefig(SUMMARY_PNG, dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0.1, 0.95, len(fd_rows)))
        sc = None
        spectral_rows = [row for row in fd_rows if row.get("status", "ok") == "ok" and row.get("artifact_npz", "")]
        for color, row in zip(colors[: len(spectral_rows)], spectral_rows):
            npz = Path(row["artifact_npz"])
            with np.load(npz, allow_pickle=False) as data:
                omega = np.asarray(data["omega"], dtype=complex).reshape(-1)
                post_rel = np.asarray(data["stability_postcheck_residual_rel"], dtype=float).reshape(-1)
            label = (
                f"{row.get('stage_name', f'ncv{row['ncv']}')}  "
                f"nev={row['requested_nev']} ncv={row['ncv']}  "
                f"post<=1e-6={row['post_rel_count_le_1e6']}"
            )
            good = np.isfinite(post_rel)
            if np.any(good):
                sc = ax.scatter(
                    np.real(omega[good]),
                    np.imag(omega[good]),
                    c=np.log10(np.maximum(post_rel[good], 1e-30)),
                    cmap="plasma",
                    vmin=-8,
                    vmax=-3,
                    s=18,
                    alpha=0.8,
                    label=label,
                )
            else:
                ax.scatter(np.real(omega), np.imag(omega), s=18, alpha=0.8, color=color, label=label)
        ax.set_title(f"Case 2 FD{fd} Spectra by NCV")
        ax.set_xlabel(r"$\Re(\omega)$")
        ax.set_ylabel(r"$\Im(\omega)$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(r"$\log_{10}$ postcheck residual")
        fig.savefig(OUTPUT_DIR / f"case2_fd{fd}_ncv_spectra.png", dpi=220)
        if fd == fd_values[0]:
            fig.savefig(SPECTRUM_PNG, dpi=220)
        plt.close(fig)

    print(SUMMARY_PNG)
    print(SPECTRUM_PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
