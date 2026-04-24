#!/usr/bin/env python3
"""Plot saved Case 2 EVP modes from Case 2 NPZ artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
DEFAULT_RUN_DIR = CASE_DIR / "output" / "evp_stride2x1_oscillatory_target_ncv500_tol1e4" / "runs"
COMPONENT_NAMES = ("rho", "u", "v", "w", "T")


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


def _split_mode_blocks(mode: np.ndarray, shape: tuple[int, int]) -> list[np.ndarray]:
    ngp = int(np.prod(shape))
    blocks = []
    for idx in range(5):
        block = mode[idx * ngp : (idx + 1) * ngp].reshape(shape, order="F")
        blocks.append(block)
    return blocks


def _mode_indices(requested: list[int] | None, n_modes: int) -> list[int]:
    if requested:
        return [idx for idx in requested if 0 <= idx < n_modes]
    return list(range(n_modes))


def _log_residual_norm(values: np.ndarray) -> mpl.colors.Normalize | None:
    valid = values[np.isfinite(values) & (values > 0.0)]
    if valid.size == 0:
        return None
    log_values = np.log10(valid)
    vmin = float(np.min(log_values))
    vmax = float(np.max(log_values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_paths", nargs="*", type=Path)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument(
        "--mode-index",
        type=int,
        action="append",
        default=None,
        help="Mode index to plot. Repeat to request multiple; default is all saved modes.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title-prefix", default="Case 2 EVP Mode")
    parser.add_argument(
        "--show-target",
        action="store_true",
        help="Plot omega_target in the spectrum panel. Default is hidden.",
    )
    args = parser.parse_args()

    npz_paths = _collect_npz_paths(args.npz_paths, args.run_dir.resolve(), args.pattern)
    if not npz_paths:
        raise SystemExit("no NPZ files found")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.run_dir.resolve().parent / "plots" / "modes"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as data:
            omega = np.asarray(data["omega"]).reshape(-1)
            if omega.size == 0:
                continue

            X = np.asarray(data["X"])
            Y = np.asarray(data["Y"])
            direct_modes = np.asarray(data["direct_modes"])
            target = complex(np.asarray(data["omega_target"]).reshape(-1)[0])
            post_dist = np.asarray(data.get("stability_postcheck_target_distance", np.array([])), dtype=float).reshape(-1)
            post_rel = np.asarray(data.get("stability_postcheck_residual_rel", np.array([])), dtype=float).reshape(-1)
            post_rel_norm = _log_residual_norm(post_rel)
            residual_cmap = plt.get_cmap("gray")
            label = _case_label(path)
            case_output_dir = output_dir / path.stem
            case_output_dir.mkdir(parents=True, exist_ok=True)

            for mode_index in _mode_indices(args.mode_index, omega.size):
                mode = direct_modes[:, mode_index]
                blocks = _split_mode_blocks(mode, X.shape)

                fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
                axes_flat = list(axes.flat)

                for ax, name, block in zip(axes_flat[:5], COMPONENT_NAMES, blocks, strict=True):
                    magnitude = np.abs(block)
                    vmax = float(np.max(magnitude))
                    normed = magnitude / vmax if vmax > 0.0 else magnitude
                    im = ax.pcolormesh(X, Y, normed, shading="auto", cmap="magma")
                    fig.colorbar(im, ax=ax)
                    ax.set_title(f"{name} |.| / max, peak={vmax:.2e}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")

                spectrum_ax = axes_flat[5]
                facecolors: str | np.ndarray = "#ffffff"
                if post_rel_norm is not None and post_rel.size == omega.size:
                    log_residual = np.full(omega.size, post_rel_norm.vmax, dtype=float)
                    valid = np.isfinite(post_rel) & (post_rel > 0.0)
                    log_residual[valid] = np.log10(post_rel[valid])
                    facecolors = residual_cmap(post_rel_norm(log_residual))
                spectrum_ax.scatter(
                    np.real(omega),
                    np.imag(omega),
                    facecolors=facecolors,
                    edgecolors="#1f77b4",
                    s=70,
                    linewidths=1.0,
                    alpha=0.95,
                )
                spectrum_ax.scatter(
                    [np.real(omega[mode_index])],
                    [np.imag(omega[mode_index])],
                    facecolors="none",
                    edgecolors="red",
                    s=180,
                    linewidths=1.5,
                )
                if args.show_target:
                    spectrum_ax.scatter(
                        [np.real(target)],
                        [np.imag(target)],
                        color="#d62728",
                        marker="x",
                        s=120,
                        linewidths=2.0,
                    )
                spectrum_ax.set_title("Spectrum")
                spectrum_ax.set_xlabel("omega_r = Re(omega)")
                spectrum_ax.set_ylabel("omega_i = Im(omega) [growth/decay]")
                spectrum_ax.grid(True, alpha=0.25)
                spectrum_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
                spectrum_ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
                if post_rel_norm is not None:
                    fig.colorbar(
                        mpl.cm.ScalarMappable(norm=post_rel_norm, cmap=residual_cmap),
                        ax=spectrum_ax,
                        label="log10(postcheck residual rel)",
                    )

                dist_text = f"{post_dist[mode_index]:.3e}" if mode_index < post_dist.size else "n/a"
                rel_text = f"{post_rel[mode_index]:.3e}" if mode_index < post_rel.size else "n/a"
                fig.suptitle(
                    f"{args.title_prefix}: {label}\n"
                    f"mode={mode_index} omega={omega[mode_index]:.6g} "
                    f"target={target:.6g} target_dist={dist_text} post_rel={rel_text}"
                )

                out_path = case_output_dir / f"mode{mode_index:02d}.png"
                fig.savefig(out_path, dpi=220)
                plt.close(fig)
                saved_paths.append(out_path)

    if not saved_paths:
        raise SystemExit("no mode figures were generated")

    for path in saved_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
