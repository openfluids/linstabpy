#!/usr/bin/env python3
"""Hungarian match Song's 20 reference modes to LinStabPy eigenvalues at β = 19.2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from examples.song.reference_paths import (
    FD4_NPZ,
    LENGTH_SCALE,
    SPECTRUM_EIGENVALUES_DAT,
    SPECTRUM_MATCH_EPS_DEFAULT,
)
from examples.song.song_plot_utils import hungarian_assignment, load_song_omega


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--song-eigenvalues", type=Path, default=SPECTRUM_EIGENVALUES_DAT)
    p.add_argument("--npz", type=Path, default=FD4_NPZ)
    p.add_argument("--song-scale", type=float, default=LENGTH_SCALE)
    p.add_argument("--match-eps", type=float, default=SPECTRUM_MATCH_EPS_DEFAULT)
    return p.parse_args()


def _load_py_omega(npz: Path) -> np.ndarray:
    with np.load(npz, allow_pickle=True) as z:
        return np.asarray(z["omega"], dtype=complex).reshape(-1)


def main() -> None:
    args = _parse_args()
    npz = args.npz.expanduser().resolve()
    if not npz.is_file():
        raise FileNotFoundError(npz)
    omega_song = load_song_omega(args.song_eigenvalues.expanduser().resolve(), args.song_scale)
    omega_py = _load_py_omega(npz)
    ridx, cidx, dist = hungarian_assignment(omega_song, omega_py)
    n_close = int(np.sum(dist <= args.match_eps))
    i_lead = int(np.argmax(np.imag(omega_song)))
    j_lead = int(np.argmin(np.abs(omega_py - omega_song[i_lead])))
    print(
        f"Song {omega_song.size} modes  LinStabPy {omega_py.size} modes  "
        f"Hungarian pairs {len(cidx)}  within |Δω|={args.match_eps:g}: {n_close}/{omega_song.size}  "
        f"max|Δω|={float(np.max(dist)):.4g}  mean|Δω|={float(np.mean(dist)):.4g}"
    )
    print(
        f"lead Song {omega_song[i_lead]:.6f}  nearest LinStabPy {omega_py[j_lead]:.6f}  "
        f"|Δ|={abs(omega_py[j_lead] - omega_song[i_lead]):.4g}"
    )
    print(f"npz {npz}")
    for i, j, d in zip(ridx.tolist(), cidx.tolist(), dist.tolist()):
        print(f"  ref[{i:2d}] {omega_song[i]:.6f}  py[{j:2d}] {omega_py[j]:.6f}  |Δω|={d:.4g}")


if __name__ == "__main__":
    main()
