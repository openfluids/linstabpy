#!/usr/bin/env python3
"""Pack a full Song EVP dump into a spectrum CSV and a few modes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VARS = ("rho", "u", "v", "w", "t")
DEFAULT_DUMP = HERE / "song_32deg_paper.npz"
DEFAULT_SPECTRUM = HERE / "song_32deg_spectrum.csv"
DEFAULT_MODES = HERE / "song_32deg_modes.npz"


def _blocks(vec: np.ndarray, ny: int, nx: int) -> dict[str, np.ndarray]:
    """Split one state vector into real primitive fields on the run grid."""
    n = ny * nx
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(VARS):
        out[name] = np.real(vec[i * n : (i + 1) * n].reshape((ny, nx), order="F"))
    return out


def pack(dump_path: Path, spectrum_path: Path, modes_path: Path, n_modes: int) -> None:
    """Write the eigenvalue table and the least-damped mode fields."""
    dump = np.load(dump_path)
    omega = np.asarray(dump["omega"]).reshape(-1)
    residual = np.asarray(dump["stability_residual_relative"]).reshape(-1)
    if residual.size != omega.size:
        raise ValueError("residual length does not match omega")
    modes = np.asarray(dump["direct_modes"])
    x = np.asarray(dump["X"])
    y = np.asarray(dump["Y"])
    ny, nx = x.shape
    if modes.ndim != 2 or modes.shape[0] != 5 * ny * nx:
        raise ValueError(f"direct_modes shape {modes.shape} does not match grid {x.shape}")
    if modes.shape[1] != omega.size:
        raise ValueError("direct_modes columns do not match omega")

    order = np.argsort(-omega.imag)
    n_keep = min(int(n_modes), int(omega.size))
    keep = order[:n_keep]
    kept = np.zeros(omega.size, dtype=bool)
    kept[keep] = True

    spectrum_path.parent.mkdir(parents=True, exist_ok=True)
    with spectrum_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# linstabpy spectrum at beta=19.2 from the local EVP dump.\n")
        handle.write("# Full state dump is not shipped. kept=1 marks packed modes.\n")
        writer = csv.writer(handle)
        writer.writerow(["index", "omega_r", "omega_i", "residual_relative", "kept"])
        for i, w in enumerate(omega):
            writer.writerow(
                [
                    i,
                    f"{w.real:.16e}",
                    f"{w.imag:.16e}",
                    f"{residual[i]:.16e}",
                    int(kept[i]),
                ]
            )

    packed = np.stack(
        [np.stack([_blocks(modes[:, i], ny, nx)[name] for name in VARS], axis=0) for i in keep],
        axis=0,
    )
    lead = _blocks(modes[:, int(keep[0])], ny, nx)
    payload = {
        "x": x,
        "y": y,
        "omega": omega[keep],
        "index": keep.astype(np.int64),
        "residual_relative": residual[keep],
        "var_names": np.asarray(VARS),
        "modes": packed,
        **lead,
    }
    if "beta" in dump.files:
        payload["beta"] = np.asarray(dump["beta"])
    modes_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(modes_path, **payload)
    print(f"wrote {spectrum_path} ({omega.size} eigenvalues, {n_keep} kept)")
    print(f"wrote {modes_path} ({modes_path.stat().st_size / 1e6:.2f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--spectrum", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--modes", type=Path, default=DEFAULT_MODES)
    parser.add_argument("--n-modes", type=int, default=3)
    args = parser.parse_args()
    if not args.dump.exists():
        raise SystemExit(f"missing dump: {args.dump}")
    if args.n_modes < 1:
        raise SystemExit("--n-modes must be at least 1")
    pack(args.dump, args.spectrum, args.modes, args.n_modes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
