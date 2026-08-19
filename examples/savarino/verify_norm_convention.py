#!/usr/bin/env python3
"""Identify the forcing/response norm convention from the published (f,beta)=(2,45)
pair (supplement, resolvent validation section).

Requires: Savarino's published resolvent modes under ``reference/`` (or
``reference/zenodo/``), plus the Mack-point ``.npz`` from
``run_compare_points.py``.

RESULT (the last section is the decisive one).

  ||q_his||^2_E  in OUR Chu*w_int response norm            = 3.162130e+04.
  ||f_his||^2    mass-weighted L2, CONSERVATIVE momentum   = 0.9960735  (= 1).
  => sigma^2 = E/F                                         = 3.174595e+04.
  his figure at this point (values in units of 1e4)        ~ 3.2e4      (0.4%).

So his stored forcing is unit-norm in F = mass-weighted L2 over the
conservative momentum components (rou, rov, row).
His response norm is the Chu energy WITH integration weights, i.e. exactly
what compute_chu_norm builds.
The colourbar label "sigma^2 x 10^4" means the display is sigma^2/1e4.
Every other candidate weighting misses by orders of magnitude
(95.4, 1.02e8, 5.6e7, 5.99e13, 297.9 in the tables below).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_paths import MACK_OUTPUT_NPZ, zenodo_baseflow_dat, zenodo_modes_dir
from savarino_mode_to_primitive import CONS, PRIM, linearise_to_primitive
from savarino_to_npy import read_tecplot

MODES = zenodo_modes_dir()
BASE = zenodo_baseflow_dat()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--point-npz",
        default=str(MACK_OUTPUT_NPZ),
        help="Output .npz from the single-point (f,beta)=(2,45) run",
    )
    args = ap.parse_args()

    d = np.load(args.point_npz, allow_pickle=True)
    W = d["W_diag"]
    idx = {v + "_j": d[f"idx_{v}_j"] for v in PRIM}
    sig_ours = float(d["gains"][0])

    base = read_tecplot(BASE)

    def mode(kind):
        re_ = read_tecplot(MODES / f"{kind}_atcenter_f2_w45_n1real.dat")
        im_ = read_tecplot(MODES / f"{kind}_atcenter_f2_w45_n1imag.dat")
        return {k: re_[k] + 1j * im_[k] for k in CONS}

    fp = linearise_to_primitive(mode("forcing"), base)
    qp = linearise_to_primitive(mode("response"), base)

    def to_dof(prim):
        v = np.zeros(450000, dtype=complex)
        for var in PRIM:
            v[idx[var + "_j"]] = prim[var].T.ravel(order="F")
        return v

    f = to_dof(fp)
    q = to_dof(qp)

    # integration weights recovered from the u-block: W_u = 0.5*rho_bar*w_int
    rho_b = (base["ro"]).T.ravel(order="F")
    w_int = 2.0 * W[idx["u_j"]] / rho_b
    M = np.zeros(450000)
    for var in PRIM:
        M[idx[var + "_j"]] = w_int

    E_q = float(np.real(np.vdot(q, W * q)))  # our Chu energy, mass-weighted already
    F_l2 = float(np.real(np.vdot(f, f)))  # plain coefficient 2-norm  (what we use)
    F_mass = float(np.real(np.vdot(f, M * f)))  # continuous L2 (mass-weighted)

    print("HIS published pair at (f,beta)=(2,45), evaluated in OUR norms")
    print(f"  ||q_his||^2_E (Chu, mass-weighted) = {E_q:.6e}")
    print(f"  ||f_his||^2  plain 2-norm          = {F_l2:.6e}")
    print(f"  ||f_his||^2  mass-weighted L2      = {F_mass:.6e}")
    print()
    print(
        f"  sigma with plain-2-norm forcing  = {np.sqrt(E_q / F_l2):.6e}   (sigma^2 x1e4 = {E_q / F_l2 * 1e4:.4g})"
    )
    print(
        f"  sigma with mass-weighted forcing = {np.sqrt(E_q / F_mass):.6e}   (sigma^2 x1e4 = {E_q / F_mass * 1e4:.4g})"
    )
    print()
    print(
        f"  our OPTIMAL sigma at this point   = {sig_ours:.6e}   (sigma^2 x1e4 = {sig_ours**2 * 1e4:.4g})"
    )
    print("  his sigma^2 x1e4 from his figure  ~ 3.2")

    print("\n=== systematic: which (response, forcing) weighting reproduces his ~3.2? ===")
    Wc = W.copy()  # Chu factors x integration weights (as built)
    Wc_nomass = np.zeros_like(Wc)  # Chu factors alone, integration weights divided out
    for var in PRIM:
        bl = idx[var + "_j"]
        Wc_nomass[bl] = Wc[bl] / w_int
    combos = {
        "E: Chu*w_int   F: plain 2-norm": (Wc, None),
        "E: Chu*w_int   F: mass": (Wc, M),
        "E: Chu only    F: plain 2-norm": (Wc_nomass, None),
        "E: Chu only    F: mass": (Wc_nomass, M),
    }
    for name, (Wu, Fw) in combos.items():
        E = float(np.real(np.vdot(q, Wu * q)))
        Fn = float(np.real(np.vdot(f, f))) if Fw is None else float(np.real(np.vdot(f, Fw * f)))
        print(f"  {name:<34} sigma^2 x1e4 = {E / Fn * 1e4:.4g}")
    print("\n  his figure ~ 3.2")
    print(
        "\n  ratio (our optimal) / (his own pair, Chu*w_int + plain 2-norm) = %.3f"
        % (sig_ours**2 / (float(np.real(np.vdot(q, Wc * q))) / float(np.real(np.vdot(f, f)))))
    )

    print("\n=== is the offset the CONSERVATIVE-vs-PRIMITIVE forcing norm? ===")
    fc = mode("forcing")  # his RAW conservative forcing: ro, rou, rov, row, roe
    n_cons = sum(float(np.real(np.vdot(fc[k].ravel(), fc[k].ravel()))) for k in CONS)
    n_mom = sum(
        float(np.real(np.vdot(fc[k].ravel(), fc[k].ravel()))) for k in ("rou", "rov", "row")
    )
    print(f"  ||f_his||^2 conservative, all 5 comps  = {n_cons:.6e}")
    print(f"  ||f_his||^2 conservative, momentum only= {n_mom:.6e}")
    print(f"  ||f_his||^2 primitive (what we use)    = {float(np.real(np.vdot(f, f))):.6e}")
    print(
        "  max|f_ro| = {:.3e}   max|f_roe| = {:.3e}   (both should be ~0: momentum-only)".format(
            np.abs(fc["ro"]).max(), np.abs(fc["roe"]).max()
        )
    )
    E = float(np.real(np.vdot(q, Wc * q)))
    for lbl, Fn in [("conservative all-5", n_cons), ("conservative momentum-only", n_mom)]:
        print(f"  sigma^2 x1e4 with E=Chu*w_int, F={lbl:<27} = {E / Fn * 1e4:.4g}")
    print("\n  his figure ~ 3.2")

    print("\n=== which norm of HIS forcing equals 1? (unit-norm hypothesis) ===")
    # candidates over the conservative (as-published) forcing, mass-weighted or not
    n_cons_mass = 0.0
    for k in ("rou", "rov", "row"):
        n_cons_mass += float(np.real(np.vdot(fc[k].ravel(), w_int * fc[k].ravel())))
    print(f"  ||f_his||^2 conservative momentum, MASS-weighted = {n_cons_mass:.6e}")
    print(f"  ||f_his||^2 primitive,             MASS-weighted = {3.098804e00:.6e}")
    print(f"  -> sigma^2 = E/F = {E:.6e} / {n_cons_mass:.6e} = {E / n_cons_mass:.6e}")
    print("  target (figure value x 1e4)                      ~ 3.2e4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
