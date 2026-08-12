#!/usr/bin/env python3
"""Figure S6, the 200-point gain map from results/savarino_map2_200pt.csv.

Render the resolvent gain map styled to match the published reference
figure's layout and color scheme (950x780 px; square 549x549 axes at rect
[0.1516, 0.1769, 0.5779, 0.7038]; colorbar at [0.7495, 0.1769, 0.0284,
0.7038]; 9 contour bands sampled edge-to-edge from the reference colorbar
#311442..#7d0401; colorbar drawn continuous, MATLAB-style, with ticks 0..3
under the title sigma^2 x 10^4). Interpolation is tensor-style (1-D in
log10 f along each swept beta line, then 1-D across beta), which is the
correct rendering for a per-line sweep and introduces no cross-line
Delaunay artifacts.

Requires: the run-record CSV (default: results/savarino_map2_200pt.csv).

Usage: plot_gain_map.py [map.csv] [out.png]
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent

# LaTeX (Computer Modern) typography, matching the reference figure's labels.
matplotlib.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "text.usetex": False,
})

REF_COLORS = ["#311442", "#466de5", "#27bbea", "#31f197", "#a2fc3c",
              "#ebd039", "#fa8021", "#d33204", "#7d0401"]


def tensor_grid(rows):
    pts = {}
    for r in rows:
        b, f, g = float(r["beta_Xsh"]), float(r["f_Xsh_Uinf"]), float(r["gain"])
        if f > 0:
            pts.setdefault(b, []).append((np.log10(f), g * g * 1e-4))
    gx = np.linspace(-4, 1, 501)
    betas = sorted(pts); lines = {}
    for b in betas:
        s = sorted(pts[b]); xs = np.array([p[0] for p in s]); ys = np.array([p[1] for p in s])
        if xs.size == 1:
            col = np.full(gx.size, np.nan); col[np.argmin(np.abs(gx - xs[0]))] = ys[0]
        elif xs.size < 4:
            col = np.interp(gx, xs, ys)
            col[(gx < xs.min()) | (gx > xs.max())] = np.nan
        else:
            # monotone cubic: smooth interpolation, no linear kinks and no
            # cubic overshoot (PCHIP preserves local monotonicity)
            col = PchipInterpolator(xs, ys)(gx)
            col[(gx < xs.min()) | (gx > xs.max())] = np.nan
        lines[b] = col
    gy = np.linspace(0, 200, 501)
    GZ = np.full((gy.size, gx.size), np.nan)
    for j in range(gx.size):
        bb = np.array([b for b in betas if np.isfinite(lines[b][j])])
        if bb.size < 2:
            continue
        vv = np.array([lines[b][j] for b in bb])
        if bb.size < 4:
            col = np.interp(gy, bb, vv)
        else:
            col = PchipInterpolator(bb, vv)(gy)
        col[(gy < bb.min()) | (gy > bb.max())] = np.nan
        GZ[:, j] = col
    return gx, gy, np.ma.masked_invalid(GZ), pts


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "results" / "savarino_map2_200pt.csv")
    dst = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "figures" / "savarino_gain_map.png")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(src)) if r["status"] == "ok"]
    gx, gy, GZ, pts = tensor_grid(rows)
    vmax = float(GZ.max())

    fig = plt.figure(figsize=(9.5, 7.8), dpi=100)
    ax = fig.add_axes([0.1516, 0.1769, 0.5779, 0.7038])
    cax = fig.add_axes([0.7495, 0.1769, 0.0284, 0.7038])

    levels = np.linspace(0, vmax, 10)
    cf = ax.contourf(gx, gy, GZ, levels=levels, colors=REF_COLORS)
    cf.set_edgecolor("face")

    ax.set_xlim(-4, 1); ax.set_ylim(0, 200)
    ax.set_xticks(range(-4, 2))
    ax.set_xticklabels([r"$10^{%d}$" % t for t in range(-4, 2)], fontsize=15)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_yticklabels([0, 50, 100, 150, 200], fontsize=15)
    minor = [np.log10(k * 10.0**e) for e in range(-4, 1) for k in range(2, 10)]
    ax.set_xticks(minor, minor=True)
    ax.tick_params(which="both", direction="in", top=True, right=True, color="k")
    ax.tick_params(which="major", length=6); ax.tick_params(which="minor", length=3)
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    ax.set_xlabel(r"$f\,X_{sh}\,/\,U_{\infty}$", fontsize=17)
    ax.set_ylabel(r"$\beta\,X_{sh}$", fontsize=17)
    ax.set_title(r"$\theta = 30.8^{\circ}$", fontsize=16, pad=10)

    smooth = LinearSegmentedColormap.from_list("ref", REF_COLORS)
    sm = plt.cm.ScalarMappable(cmap=smooth, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 1, 2, 3])
    cb.ax.tick_params(labelsize=15, direction="in")
    cb.ax.set_title(r"$\sigma^2\,{\times}\,10^4$", fontsize=16, pad=12)

    # annotations: square = leading peak, diamond = second peak
    allv = [(v, b, x) for b in pts for x, v in sorted(pts[b])]
    _, bm, xm = max(allv)
    lu = [(v, b, x) for v, b, x in allv if b >= 100 and x <= -3]
    ax.plot(xm, bm, "s", ms=10, mfc="red", mec="black", mew=1.0, zorder=5)
    ax.annotate("Medium-frequency\n1st Mack mode", xy=(xm, bm), xytext=(-2.55, 47),
                fontsize=15, color="white", ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color="white", lw=1.2,
                                shrinkA=8, shrinkB=8))
    if lu:
        vmax_lu = max(lu)[0]
        # among near-max low-f points on the lift-up line, mark the lowest
        # frequency, matching the reference figure's convention
        near = [(x, b) for v, b, x in lu if v >= 0.999 * vmax_lu]
        xl, bl = min(near)
        ax.plot(xl, bl, "D", ms=10, mfc="red", mec="black", mew=1.0,
                zorder=5, clip_on=False)
        ax.annotate("Low-frequency\nlift-up mode", xy=(xl, bl), xytext=(-2.45, 163),
                    fontsize=15, color="black", ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color="black", lw=1.2,
                                    shrinkA=8, shrinkB=8))
    fig.savefig(dst, dpi=100)
    print(f"wrote {dst}  ({len(rows)} pts, vmax={vmax:.3f}, square=({10**xm:.3g},{bm:g}), "
          f"diamond=({10**xl:.3g},{bl:g})" if lu else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
