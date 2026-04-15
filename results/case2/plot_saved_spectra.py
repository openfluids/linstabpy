#!/usr/bin/env python3
"""Compare saved EVP spectra from Case 2 NPZ artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ROOT / "output" / "evp_stride4_localize" / "runs"


def _collect_npz_paths(paths: list[Path], run_dir: Path, pattern: str) -> list[Path]:
    if paths:
        resolved = [path.resolve() for path in paths]
    else:
        resolved = sorted(run_dir.glob(pattern))
    return [path for path in resolved if path.suffix == ".npz" and path.exists()]


def _case_label(path: Path) -> str:
    stem = path.stem
    prefix = "case2_perf_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_paths", nargs="*", type=Path)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default="Case 2 Saved Temporal EVP Spectra")
    args = parser.parse_args()

    npz_paths = _collect_npz_paths(args.npz_paths, args.run_dir.resolve(), args.pattern)
    if not npz_paths:
        raise SystemExit("no NPZ files found")

    output = args.output
    if output is None:
        output = args.run_dir.resolve().parent / "plots" / "spectra" / "saved_spectra.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    for idx, path in enumerate(npz_paths):
        color = cmap(idx % 10)
        label = _case_label(path)
        with np.load(path, allow_pickle=False) as data:
            omega = np.asarray(data["omega"]).reshape(-1)
            target = complex(np.asarray(data["omega_target"]).reshape(-1)[0])
            accepted_indices = np.asarray(
                data.get("stability_accepted_indices", np.array([], dtype=np.int32)),
                dtype=np.int32,
            ).reshape(-1)
            accepted_mask = np.zeros(omega.size, dtype=bool)
            accepted_mask[accepted_indices[(accepted_indices >= 0) & (accepted_indices < omega.size)]] = True

            ax.scatter(
                np.real(omega),
                np.imag(omega),
                label=label,
                color=color,
                s=70,
                alpha=0.85,
            )
            if np.any(accepted_mask):
                ax.scatter(
                    np.real(omega[accepted_mask]),
                    np.imag(omega[accepted_mask]),
                    facecolors="none",
                    edgecolors="black",
                    s=180,
                    linewidths=1.5,
                )

            ax.scatter(
                [np.real(target)],
                [np.imag(target)],
                color=color,
                marker="x",
                s=120,
                linewidths=2.0,
            )

    ax.set_xlabel("omega_r = Re(omega)")
    ax.set_ylabel("omega_i = Im(omega) [growth/decay]")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.legend(loc="best", fontsize=8)

    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
