#!/usr/bin/env python3
"""Classify eigenmodes by Chu-energy localisation: wall-local vs global.

Chu (compressible) energy density for a state (ρ', u', v', w', T') is
    e_chu = (T̄/(γ M² ρ̄)) |ρ'|² + ρ̄ |u'|² + ρ̄ |v'|² + ρ̄ |w'|²
            + (ρ̄/(γ(γ−1) M² T̄)) |T'|²
(Chu 1965 / Hanifi et al. 1996 — the exact form used internally by the
solver is the diagonal quadratic weight `W_diag` saved in every npz).

Classifier metrics computed per mode:
  - `E_chu_per_cell = W_diag .* |q|^2`, reshaped to (ny, nx);
  - `wall_frac`: fraction of total E_chu inside `y_index < wall_thr`
    (near-wall strip);
  - `shock_foot_frac`: fraction of total E_chu inside a circle of
    radius `r_shock` centered on the shock-foot location
    (x_sf, y_sf) from the Qraw |∂Ū/∂x|_max (Entry 5);
  - `global_extent`: fraction of `ngp` cells carrying at least 1% of
    the peak Chu-density — "how spread out is the mode".

Heuristic classification:
  - WALL_LOCAL    : wall_frac > 0.30 OR shock_foot_frac > 0.30
  - GLOBAL        : global_extent > 0.20 AND wall_frac < 0.30
  - INTERMEDIATE  : anything else

Usage:
    .venv/bin/python examples/case_studies/case2_lui/phase1p5_classify_modes.py \\
        <npz> [<npz> ...]
Prints a per-mode classification table and writes one summary CSV per
input npz to `phase1p5/<basename>_chu_class.csv`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
OUT_DIR = CASE_DIR / "artifacts" / "phase1p5"
OUT_DIR.mkdir(exist_ok=True)

# Shock-foot location on M21/M2 (Entry 5 gradient analysis).
SHOCK_FOOT = (0.817, -0.099)
R_SHOCK = 0.08

# Near-wall strip: eta index from 0 to WALL_IDX (wall + first ~5% of domain).
WALL_ETA_FRAC = 0.05


def classify_one(d: dict, j: int) -> dict:
    modes = d["direct_modes"]
    W = d["W_diag"]
    X = d["X"]
    Y = d["Y"]
    ny, nx = X.shape
    mode = modes[:, j]
    # Chu energy density per full DOF:
    ec = W * np.abs(mode) ** 2
    # Aggregate across the 5 variables at each grid point:
    # idx_{rho|u|v|w|T}_j give the DOF indices for each variable.
    # Sum per-gridpoint energy.
    ec_per_cell = np.zeros(ny * nx, dtype=float)
    for name in ("rho", "u", "v", "w", "T"):
        idx = d[f"idx_{name}_j"]
        ec_per_cell += ec[idx]
    grid = ec_per_cell.reshape((ny, nx), order="F")
    total = grid.sum()
    if total <= 0:
        return {
            "omega": complex(d["omega"][j]),
            "wall_frac": np.nan,
            "shock_foot_frac": np.nan,
            "global_extent": np.nan,
            "peak_xy": (np.nan, np.nan),
            "klass": "ZERO_ENERGY",
        }

    # Near-wall strip: first WALL_ETA_FRAC of eta-index range (wall is at j=0).
    wall_idx_end = max(1, int(ny * WALL_ETA_FRAC))
    wall_frac = grid[:wall_idx_end, :].sum() / total

    # Shock-foot circle
    xsf, ysf = SHOCK_FOOT
    r2 = (X - xsf) ** 2 + (Y - ysf) ** 2
    sf_mask = r2 < R_SHOCK**2
    sf_frac = grid[sf_mask].sum() / total

    # Spatial extent — fraction of cells with ec > 1% of peak
    peak = grid.max()
    extent = float((grid > 0.01 * peak).mean())

    # Peak location
    pi = np.unravel_index(int(np.argmax(grid)), grid.shape)
    peak_xy = (float(X[pi]), float(Y[pi]))

    if wall_frac > 0.30 or sf_frac > 0.30:
        klass = "WALL_LOCAL"
    elif extent > 0.20 and wall_frac < 0.30:
        klass = "GLOBAL"
    else:
        klass = "INTERMEDIATE"

    return {
        "omega": complex(d["omega"][j]),
        "wall_frac": float(wall_frac),
        "shock_foot_frac": float(sf_frac),
        "global_extent": float(extent),
        "peak_xy": peak_xy,
        "klass": klass,
    }


def classify_file(npz: Path) -> list[dict]:
    d_obj = np.load(npz, allow_pickle=True)
    d = {k: d_obj[k] for k in d_obj.files}
    n = d["direct_modes"].shape[1] if d["direct_modes"].ndim == 2 else 0
    rows = []
    print(f"\n== {npz.name} ==")
    hdr = f"{'j':>3s}  {'ω':>25s}  {'wall%':>6s}  {'shock%':>7s}  {'extent%':>7s}  {'peak (x,y)':>20s}  {'class':>12s}"
    print(hdr)
    for j in range(n):
        r = classify_one(d, j)
        r["j"] = j
        rows.append(r)
        w = r["omega"]
        px, py = r["peak_xy"]
        print(
            f"{j:3d}  {w.real:+.3f} {w.imag:+.3f}j  "
            f"{r['wall_frac'] * 100:6.1f}  {r['shock_foot_frac'] * 100:7.1f}  "
            f"{r['global_extent'] * 100:7.1f}  "
            f"({px:+.3f}, {py:+.3f})  {r['klass']:>12s}"
        )

    # CSV
    out_csv = OUT_DIR / f"{npz.stem}_chu_class.csv"
    with out_csv.open("w") as f:
        f.write("j,omega_re,omega_im,wall_frac,shock_foot_frac,global_extent,peak_x,peak_y,class\n")
        for r in rows:
            w = r["omega"]
            px, py = r["peak_xy"]
            f.write(
                f"{r['j']},{w.real:.6f},{w.imag:.6f},"
                f"{r['wall_frac']:.6f},{r['shock_foot_frac']:.6f},"
                f"{r['global_extent']:.6f},{px:.6f},{py:.6f},"
                f"{r['klass']}\n"
            )
    print(f"wrote {out_csv}")
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: phase1p5_classify_modes.py <npz> [<npz> ...]")
        sys.exit(2)
    for p in sys.argv[1:]:
        classify_file(Path(p))


if __name__ == "__main__":
    main()
