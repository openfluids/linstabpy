#!/usr/bin/env python3
"""Plot saved resolvent response and/or forcing modes from NPZ artifacts (run kit).

Expects the same layout as ``linstabpy.results_runtime`` save: ``responses``,
``forcings``, ``gains``, ``X``, ``Y``, and primitive index maps ``idx_*_j``.

Each primitive panel shows the **real part** with a **coolwarm** colormap.
All five primitives use the **same** fixed symmetric limits **±15** for ``Re(·)``
(coolwarm). Panel titles still report each field's ``|max|``.
The gains bar uses a **y-axis symmetric about zero**.

Example::

    .venv/bin/python results/case2/plot_saved_resolvent_modes.py \\
        results/case2/output/runs/parity_config.npz

    .venv/bin/python results/case2/plot_saved_resolvent_modes.py \\
        --run-dir results/case2/output/runs --pattern 'parity_config.npz'

    .venv/bin/python results/case2/plot_saved_resolvent_modes.py \\
        --kind forcing --mode-index 0 --mode-index 2 \\
        results/case2/output/runs/parity_config.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ROOT / "output" / "runs"
COMPONENT_NAMES = ("rho", "u", "v", "w", "T")
# Symmetric color limits for Re(primitive) panels (coolwarm).
PLOT_RE_SYMMETRIC_LIMIT = 15.0


def _collect_npz_paths(paths: list[Path], run_dir: Path, pattern: str) -> list[Path]:
    if paths:
        resolved = [path.resolve() for path in paths]
    else:
        resolved = sorted(run_dir.glob(pattern))
    return [path for path in resolved if path.suffix == ".npz" and path.exists()]


def _case_label(path: Path) -> str:
    stem = path.stem
    for prefix in ("campaign_", "case2_perf_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
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


def _as_complex_target(raw: np.ndarray) -> complex:
    z = np.asarray(raw).reshape(-1)
    if z.size == 0:
        return complex("nan")
    if np.iscomplexobj(z):
        return complex(z[0])
    if z.size >= 2:
        return complex(float(z[0]), float(z[1]))
    return complex(float(z[0]), 0.0)


def _real_blocks_and_peaks(blocks: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
    """Re parts and per-field max |Re| (for titles)."""
    Rs = [np.real(np.asarray(b, dtype=complex)) for b in blocks]
    peaks = [float(np.nanmax(np.abs(R))) for R in Rs]
    return Rs, peaks


def _plot_resolvent_mode_figure(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    vec: np.ndarray,
    mode_index: int,
    gains: np.ndarray,
    omega_target: complex,
    label: str,
    kind_label: str,
    title_prefix: str,
    out_path: Path,
) -> None:
    blocks = _split_mode_blocks(vec, X.shape)
    Rs, peaks = _real_blocks_and_peaks(blocks)
    lim = float(PLOT_RE_SYMMETRIC_LIMIT)
    sigma_k = float(gains[mode_index]) if mode_index < gains.size else float("nan")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes_flat = list(axes.flat)

    for ax, name, R, peak in zip(axes_flat[:5], COMPONENT_NAMES, Rs, peaks, strict=True):
        im = ax.pcolormesh(X, Y, R, shading="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{name} Re(·), ±{lim:g} (|max|={peak:.2e})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    gains_ax = axes_flat[5]
    n = int(gains.size)
    if n > 0:
        gains_ax.bar(np.arange(n), gains, color="#1f77b4", alpha=0.85)
        gains_ax.axvline(mode_index, color="black", linestyle="--", linewidth=1.2)
        y_max = float(np.nanmax(np.abs(gains)))
        if y_max <= 0.0:
            y_max = 1.0
        pad = 0.05 * y_max
        gains_ax.set_ylim(-y_max - pad, y_max + pad)
        gains_ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.45)
    gains_ax.set_title(r"Gains $\sigma_k$")
    gains_ax.set_xlabel("mode index")
    gains_ax.set_ylabel(r"$\sigma$")
    gains_ax.grid(True, alpha=0.25)

    ot = omega_target
    omega_str = f"{ot.real:.6g}{ot.imag:+.6g}i" if np.isfinite(ot.real) and np.isfinite(ot.imag) else str(ot)
    fig.suptitle(
        f"{title_prefix}: {label}\n"
        f"{kind_label} mode={mode_index}  "
        rf"$\sigma_{{{mode_index}}}={sigma_k:.6g}$  "
        rf"$\omega^* \approx {omega_str}$"
    )

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_paths", nargs="*", type=Path)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument(
        "--kind",
        choices=("response", "forcing", "both"),
        default="response",
        help="Plot resolvent response phi, forcing psi, or both (separate PNGs).",
    )
    parser.add_argument(
        "--mode-index",
        type=int,
        action="append",
        dest="mode_indices",
        default=None,
        help="Mode index to plot. Repeat for multiple; default is all saved modes.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title-prefix", default="Resolvent")
    args = parser.parse_args()

    npz_paths = _collect_npz_paths(args.npz_paths, args.run_dir.resolve(), args.pattern)
    if not npz_paths:
        raise SystemExit("no NPZ files found")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.run_dir.resolve().parent / "plots" / "resolvent_modes"
    output_dir.mkdir(parents=True, exist_ok=True)

    kinds: list[tuple[str, str]] = []
    if args.kind in {"response", "both"}:
        kinds.append(("response", "response"))
    if args.kind in {"forcing", "both"}:
        kinds.append(("forcing", "forcing"))

    saved_paths: list[Path] = []
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as data:
            if "responses" not in data or "gains" not in data:
                print(f"[skip] {path}: missing responses/gains", flush=True)
                continue
            responses = np.asarray(data["responses"])
            gains = np.asarray(data["gains"], dtype=float).reshape(-1)
            forcings = np.asarray(data["forcings"]) if "forcings" in data else None
            n_modes = int(responses.shape[1]) if responses.ndim == 2 else 0
            if n_modes == 0:
                print(f"[skip] {path}: no resolvent modes in responses", flush=True)
                continue

            X = np.asarray(data["X"])
            Y = np.asarray(data["Y"])
            omega_target = _as_complex_target(np.asarray(data["omega_target"]))
            label = _case_label(path)
            case_output_dir = output_dir / path.stem
            case_output_dir.mkdir(parents=True, exist_ok=True)

            n_force = int(forcings.shape[1]) if forcings is not None and forcings.ndim == 2 else 0

            kinds_plot: list[tuple[str, str]] = list(kinds)
            if any(k == "forcing" for k, _ in kinds_plot):
                if forcings is None or n_force == 0:
                    print(f"[skip] {path}: no forcings in NPZ; plotting responses only", flush=True)
                    kinds_plot = [(k, t) for k, t in kinds_plot if k != "forcing"]

            for mode_index in _mode_indices(args.mode_indices, n_modes):
                for kind_key, kind_title in kinds_plot:
                    if kind_key == "forcing":
                        if mode_index >= n_force:
                            print(
                                f"[skip] {path}: forcing mode {mode_index} out of range ({n_force})",
                                flush=True,
                            )
                            continue
                        vec = forcings[:, mode_index]
                        slug = "forcing"
                    else:
                        vec = responses[:, mode_index]
                        slug = "response"
                    out_path = case_output_dir / f"resolvent_{slug}_mode{mode_index:02d}.png"
                    _plot_resolvent_mode_figure(
                        X=X,
                        Y=Y,
                        vec=vec,
                        mode_index=mode_index,
                        gains=gains,
                        omega_target=omega_target,
                        label=label,
                        kind_label=kind_title.capitalize(),
                        title_prefix=args.title_prefix,
                        out_path=out_path,
                    )
                    saved_paths.append(out_path)

    if not saved_paths:
        raise SystemExit("no resolvent mode figures were generated")

    for path in saved_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
