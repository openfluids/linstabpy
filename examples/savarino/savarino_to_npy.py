#!/usr/bin/env python3
"""Convert Savarino's 30.8 deg SWBLI base flow to linstabpy .npy fields.

Produces the base-flow input for the Savarino case-study reproduction
(supplement, resolvent validation section): five primitive fields
(rho, u, v, w, T) plus coordinates and a placeholder sponge array, written
as individual .npy files that ``paths.case_dir`` in the case configs point
at. Required input: the Zenodo base-flow file (see ``fetch_zenodo.py``).

Source: Zenodo 10.5281/zenodo.18854407, file
``state_atcenter_mesh300_y300_xend195_angle308_T157_Mach215.dat``
(Tecplot ASCII, md5 5b3ed38fbe087633e4550b1e1231dc10).

WHY THIS FILE IS SEPARATE FROM A SIMILAR PLOT3D READER. A different base flow
(e.g. from a DNS/LES box) may be Plot3D (grid.grd + mean_flow.dat) with
primitive variables at cell centres plus an edge ghost, needing a
cell-centre -> vertex interpolation. Savarino's is a single Tecplot file
carrying its own coordinates and *conservative* variables, already at the
points we intend to solve on. Two different formats, two readers.

NO INTERPOLATION, BY DESIGN. His 90,000 cell-centre points are taken as the
mesh. Resampling his field onto a mesh of our choosing would mean we
validated our interpolation as much as our operator. The mesh is x-uniform
and y-geometric (growth 1.020/cell), which ``Mesh.create(...).deform(...)``
handles through the metric transform.

CONVENTION. His config states "Pressure, temperature, density and velocities
are referenced to the free-stream static quantities", which is the canonical
convention this code already enforces (freestream rho = u = T = 1). Nothing
is rescaled. The freestream gate below asserts it rather than trusting it.

THE TEMPERATURE CONVERSION IS NOT A GUESS. From ``roe`` we get the internal
energy ``e_int = roe/ro - |vel|^2/2``. In this nondimensionalisation
``e_int = T/(g(g-1)M^2)``, so ``T = e_int * g(g-1)M^2``. The freestream value
predicted by that identity is 1/(g(g-1)M^2) = 0.386309, and the file's
freestream ``e_int`` is 0.386309 exactly. That match is what proves the
conversion, and with it T_inf = 1.

Usage:  python savarino_to_npy.py <src.dat> <out_dir>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

#: md5 of the Zenodo file. Asserted so a wrongly-named or wrongly-sourced
#: file is caught before it silently contaminates a run.
SRC_MD5 = "5b3ed38fbe087633e4550b1e1231dc10"

GAMMA = 1.4
MACH = 2.15

#: Freestream internal energy implied by T = 1, i.e. 1/(gamma*(gamma-1)*M^2).
E_INT_INF = 1.0 / (GAMMA * (GAMMA - 1.0) * MACH**2)

#: How close to 1 the freestream plateau must sit. Loose enough for a
#: cell-centred field, far tighter than any mis-scale would survive (the
#: historical errors were factors of Ma^2 ~ 4.6 and c_inf).
FREESTREAM_TOL = 1e-3


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tecplot(path: Path) -> dict[str, np.ndarray]:
    """Parse the Tecplot ASCII point-format file into (Nx, Ny) arrays.

    The header declares ``ZONE I = 300, J = 300`` and the rows run with i
    fastest, so the flat data reshapes as (I, J) in Fortran order -- axis 0
    streamwise, axis 1 wall-normal, which is the layout linstabpy's loader
    expects.
    """
    ni = nj = None
    rows: list[list[float]] = []
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith(("TITLE", "VARIABLES")):
                continue
            if s.startswith("ZONE"):
                parts = s.replace("ZONE", "").replace(",", " ").split()
                # parts looks like ['I', '=', '300', 'J', '=', '300']
                ni = int(parts[parts.index("I") + 2])
                nj = int(parts[parts.index("J") + 2])
                continue
            rows.append([float(v) for v in s.split()])

    if ni is None or nj is None:
        raise ValueError(f"{path}: no ZONE record with I and J")
    data = np.asarray(rows, dtype=float)
    if data.shape != (ni * nj, 7):
        raise ValueError(f"{path}: expected {(ni * nj, 7)} numbers, got {data.shape}")

    names = ("X", "Y", "ro", "rou", "rov", "row", "roe")
    return {name: data[:, k].reshape((ni, nj), order="F") for k, name in enumerate(names)}


def to_primitive(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rho = raw["ro"]
    u = raw["rou"] / rho
    v = raw["rov"] / rho
    w = raw["row"] / rho
    e_int = raw["roe"] / rho - 0.5 * (u**2 + v**2 + w**2)
    T = e_int * GAMMA * (GAMMA - 1.0) * MACH**2
    return {"x": raw["X"], "y": raw["Y"], "rho": rho, "u": u, "v": v, "w": w, "T": T}


def freestream_gate(f: dict[str, np.ndarray]) -> tuple[int, int]:
    """Assert a contiguous inlet plateau at rho = u = T = 1.

    Taken on the inlet column, above the boundary layer and below the
    incident shock. Returns the (first, last) wall-normal index of the
    plateau so the caller can report how thick it was -- a gate that passes
    on two stray cells is not a gate.
    """
    i = 0  # inlet column
    ok = (
        (np.abs(f["rho"][i] - 1.0) < FREESTREAM_TOL)
        & (np.abs(f["u"][i] - 1.0) < FREESTREAM_TOL)
        & (np.abs(f["T"][i] - 1.0) < FREESTREAM_TOL)
    )
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        raise SystemExit(
            "FREESTREAM GATE FAILED: no cell on the inlet column has rho = u = T = 1 "
            f"to {FREESTREAM_TOL}. The field is mis-scaled; do not run anything on it."
        )
    # longest contiguous run
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    k = int(np.argmax(ends - starts))
    lo, hi = int(starts[k]), int(ends[k])
    if hi - lo < 10:
        raise SystemExit(
            f"FREESTREAM GATE FAILED: plateau is only {hi - lo + 1} cells thick "
            f"(j={lo}..{hi}). Expected a thick inviscid layer between the boundary "
            "layer and the incident shock."
        )
    return lo, hi


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, out = Path(argv[1]), Path(argv[2])

    digest = _md5(src)
    if digest != SRC_MD5:
        raise SystemExit(
            f"md5 mismatch on {src}\n  expected {SRC_MD5}\n  got      {digest}\n"
            "This is not Savarino's 30.8 deg base flow. Refusing to convert."
        )
    print(f"md5 OK: {digest}")

    raw = read_tecplot(src)
    f = to_primitive(raw)
    ni, nj = f["x"].shape
    print(f"grid: {ni} x {nj} (streamwise x wall-normal)")

    lo, hi = freestream_gate(f)
    print(
        f"freestream gate PASSED: inlet plateau j={lo}..{hi} ({hi - lo + 1} cells), "
        f"rho={f['rho'][0, hi]:.6f} u={f['u'][0, hi]:.6f} T={f['T'][0, hi]:.6f}"
    )

    e_inf = raw["roe"][0, hi] / raw["ro"][0, hi] - 0.5 * (
        f["u"][0, hi] ** 2 + f["v"][0, hi] ** 2
    )
    print(f"e_int,inf = {e_inf:.6f}  vs  1/(g(g-1)M^2) = {E_INT_INF:.6f}   (proves T_inf = 1)")

    wmax = float(np.abs(f["w"]).max())
    if wmax != 0.0:
        raise SystemExit(f"expected a 2-D base flow with w == 0 exactly, got max|w| = {wmax}")

    out.mkdir(parents=True, exist_ok=True)
    # The sponge is rebuilt from x, y by the run config (physics.sponge preset), so
    # what is written here is a placeholder of the right shape, never the sponge used.
    f["sponge"] = np.zeros_like(f["x"])
    for name, arr in f.items():
        np.save(out / f"{name}.npy", np.ascontiguousarray(arr, dtype=float))
    print(f"wrote {len(f)} fields to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
