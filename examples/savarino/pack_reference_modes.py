#!/usr/bin/env python3
"""Pack Savarino published response modes into shipped reference NPZs.

Reads Tecplot forcing/response files (Zenodo layout), linearises to primitive
perturbations on the shipped Tecplot base flow (``reference/baseflow/``), and
writes ``reference/savarino_response_<name>.npz`` with the five fields
used by ``compare_modes.py`` (rho, u, v, w, T; complex) plus scalar f, beta.

Usage (from this directory, after placing Tecplot files under reference/zenodo/):
    python pack_reference_modes.py
    python pack_reference_modes.py --base-dat /path/to/state_atcenter_....dat \\
        --modes-dir /path/to/resolvent_angle308_.../
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reference_paths import REFERENCE_DIR, zenodo_baseflow_dat, zenodo_modes_dir
from savarino_mode_to_primitive import CONS, PRIM, linearise_to_primitive
from savarino_to_npy import read_tecplot

CASES = (
    ("mack", "2", "45"),
    ("liftup", "0.0001", "163"),
)


def pack_response(base_dat: Path, modes_dir: Path, out: Path, *, f: str, beta: str) -> None:
    base = read_tecplot(base_dat)
    re_ = read_tecplot(modes_dir / f"response_atcenter_f{f}_w{beta}_n1real.dat")
    im_ = read_tecplot(modes_dir / f"response_atcenter_f{f}_w{beta}_n1imag.dat")
    cons = {k: re_[k] + 1j * im_[k] for k in CONS}
    prim = linearise_to_primitive(cons, base)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        f=float(f),
        beta=float(beta),
        **{v: prim[v] for v in PRIM},
    )
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dat", type=Path, default=None)
    ap.add_argument("--modes-dir", type=Path, default=None)
    args = ap.parse_args()
    base_dat = args.base_dat or zenodo_baseflow_dat()
    modes_dir = args.modes_dir or zenodo_modes_dir()
    for name, f, beta in CASES:
        pack_response(
            base_dat,
            modes_dir,
            REFERENCE_DIR / f"savarino_response_{name}.npz",
            f=f,
            beta=beta,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
