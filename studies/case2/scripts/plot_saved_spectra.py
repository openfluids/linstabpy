#!/usr/bin/env python3
"""Compare saved EVP spectra from Case 2 NPZ artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
DEFAULT_RUN_DIR = CASE_DIR / "output" / "evp_stride2x1_oscillatory_target_ncv500_tol1e4" / "runs"
CASE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")


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


def _residual_log_norm(npz_paths: list[Path]) -> tuple[mpl.colors.Normalize | None, str]:
    values: list[np.ndarray] = []
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as data:
            residual_rel = np.asarray(
                data.get("stability_postcheck_residual_rel", np.array([], dtype=float)),
                dtype=float,
            ).reshape(-1)
            residual_rel = residual_rel[np.isfinite(residual_rel) & (residual_rel > 0.0)]
            if residual_rel.size:
                values.append(np.log10(residual_rel))

    if not values:
        return None, ""

    merged = np.concatenate(values)
    vmin = float(np.min(merged))
    vmax = float(np.max(merged))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax), "log10(postcheck residual rel)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_paths", nargs="*", type=Path)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default="Case 2 Saved Temporal EVP Spectra")
    parser.add_argument(
        "--show-target",
        action="store_true",
        help="Plot omega_target markers. Default is hidden to reduce clutter.",
    )
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
    residual_norm, residual_label = _residual_log_norm(npz_paths)
    residual_cmap = plt.get_cmap("gray")
    colorbar_mappable: mpl.cm.ScalarMappable | None = None

    for idx, path in enumerate(npz_paths):
        color = cmap(idx % 10)
        marker = CASE_MARKERS[idx % len(CASE_MARKERS)]
        with np.load(path, allow_pickle=False) as data:
            omega = np.asarray(data["omega"]).reshape(-1)
            target = complex(np.asarray(data["omega_target"]).reshape(-1)[0])
            label = f"{_case_label(path)} (n={omega.size})"
            accepted_indices = np.asarray(
                data.get("stability_accepted_indices", np.array([], dtype=np.int32)),
                dtype=np.int32,
            ).reshape(-1)
            accepted_mask = np.zeros(omega.size, dtype=bool)
            accepted_mask[accepted_indices[(accepted_indices >= 0) & (accepted_indices < omega.size)]] = True
            residual_rel = np.asarray(
                data.get("stability_postcheck_residual_rel", np.array([], dtype=float)),
                dtype=float,
            ).reshape(-1)

            facecolors: str | np.ndarray = "white"
            if residual_norm is not None and residual_rel.size == omega.size:
                log_residual = np.full(omega.size, residual_norm.vmax, dtype=float)
                valid = np.isfinite(residual_rel) & (residual_rel > 0.0)
                log_residual[valid] = np.log10(residual_rel[valid])
                facecolors = residual_cmap(residual_norm(log_residual))
                colorbar_mappable = mpl.cm.ScalarMappable(norm=residual_norm, cmap=residual_cmap)

            ax.scatter(
                np.real(omega),
                np.imag(omega),
                label=label,
                facecolors=facecolors,
                edgecolors=color,
                marker=marker,
                s=70,
                linewidths=1.0,
                alpha=0.95,
            )
            if np.any(accepted_mask):
                ax.scatter(
                    np.real(omega[accepted_mask]),
                    np.imag(omega[accepted_mask]),
                    facecolors="none",
                    edgecolors="red",
                    marker=marker,
                    s=180,
                    linewidths=1.7,
                )

            if args.show_target:
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
    if colorbar_mappable is not None:
        fig.colorbar(colorbar_mappable, ax=ax, label=residual_label)

    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
