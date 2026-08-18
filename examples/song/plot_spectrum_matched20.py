#!/usr/bin/env python3
"""Song vs LinStabPy spectrum (matched-20 Hungarian assignment, supp Fig S2 style)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.figure_style import apply, check_plot_environment, save
from examples.song.reference_paths import (
    FD4_NPZ,
    FIGURES_VALIDATION_DIR,
    LENGTH_SCALE,
    SPECTRUM_EIGENVALUES_DAT,
    SPECTRUM_MATCH_EPS_DEFAULT,
)
from examples.song.song_plot_utils import hungarian_assignment, load_song_omega


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--song-eigenvalues", type=Path, default=SPECTRUM_EIGENVALUES_DAT)
    p.add_argument("--npz", type=Path, default=FD4_NPZ)
    p.add_argument("--out-dir", type=Path, default=FIGURES_VALIDATION_DIR)
    p.add_argument("--match-eps", type=float, default=SPECTRUM_MATCH_EPS_DEFAULT)
    p.add_argument("--song-scale", type=float, default=LENGTH_SCALE)
    return p.parse_args()


def _load_py_omega(npz: Path) -> np.ndarray:
    with np.load(npz, allow_pickle=True) as z:
        return np.asarray(z["omega"], dtype=complex).reshape(-1)


def _plot_spectrum(
    *,
    omega_song: np.ndarray,
    omega_py_matched: np.ndarray,
    out_path: Path,
) -> None:
    apply()
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.plot(
        np.real(omega_song),
        np.imag(omega_song),
        "o",
        ms=6,
        mfc="none",
        mec="0.1",
        mew=1.0,
        label="Song",
        zorder=3,
    )
    ax.plot(
        np.real(omega_py_matched),
        np.imag(omega_py_matched),
        "+",
        ms=5,
        mew=0.8,
        color="0.15",
        label="linstabpy",
        zorder=2,
    )
    ax.axhline(0.0, color="gray", linewidth=1.0, zorder=0)
    ax.set_xlabel(r"$\omega_r$")
    ax.set_ylabel(r"$\omega_i$")
    ax.tick_params(top=True, right=True)
    ax.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", handletextpad=0.4)
    save(fig, out_path)


def main() -> None:
    check_plot_environment(require_latex=True)
    args = _parse_args()
    npz = args.npz.expanduser().resolve()
    if not npz.is_file():
        raise FileNotFoundError(npz)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    omega_song = load_song_omega(args.song_eigenvalues.expanduser().resolve(), args.song_scale)
    omega_py = _load_py_omega(npz)
    ridx, cidx, dist = hungarian_assignment(omega_song, omega_py)
    n_close = int(np.sum(dist <= args.match_eps))
    out_path = out_dir / "song2023_spectrum_matched20_theta32.png"
    _plot_spectrum(omega_song=omega_song, omega_py_matched=omega_py[cidx], out_path=out_path)
    i_lead = int(np.argmax(np.imag(omega_song)))
    j_lead = int(np.argmin(np.abs(omega_py - omega_song[i_lead])))
    print(
        f"Song {omega_song.size}  LinStabPy {omega_py.size}  "
        f"Hungarian {len(cidx)}  within |Δω|={args.match_eps:g}: {n_close}/{omega_song.size}  "
        f"max|Δω|={float(np.max(dist)):.4g}  mean|Δω|={float(np.mean(dist)):.4g}"
    )
    print(
        f"lead Song {omega_song[i_lead]:.6f}  nearest LinStabPy {omega_py[j_lead]:.6f}  "
        f"|Δ|={abs(omega_py[j_lead] - omega_song[i_lead]):.4g}"
    )
    print(f"wrote {out_path}")
    print(f"npz {npz}")


if __name__ == "__main__":
    main()
