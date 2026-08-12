#!/usr/bin/env python3
"""Figures S4/S5, response-mode comparison at the lift-up and first-Mack points.

Side-by-side response-mode comparison: Savarino (left) vs linstabpy (right).

One figure per mode family:
  - Mack   : (f, beta) = (2, 45)      -> compare_mode_mack.png
  - Lift-up: (f, beta) = (1e-4, 163)  -> compare_mode_liftup.png

Rows: Re(rho'), Re(u'), Re(v'), Re(w'), Re(T').  Both modes are normalized to
unit Chu norm (our W_diag) and OUR mode is phase-rotated to maximize the real
part of the Chu inner product with HIS mode, so signs/phases are comparable.
Row color limits are shared between the two columns (99.8th pct of |Re|).

Requires: the converted base flow and mode data under ``$SAVARINO_DATA`` (or
``data/`` next to this script), and the ``.npz`` outputs of the (f,beta) =
(2, 45) and (1e-4, 163) points under ``--campaign-dir``.

MANUSCRIPT=1 drops the suptitle (captions carry that information in the paper)
and writes *_ms.png.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from savarino_mode_to_primitive import CONS, PRIM, linearise_to_primitive
from savarino_to_npy import read_tecplot

MANUSCRIPT = os.environ.get("MANUSCRIPT") == "1"
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 9 if MANUSCRIPT else 10,
})

DATA_DIR = Path(os.environ.get("SAVARINO_DATA", HERE / "data"))
MODES = DATA_DIR / "modes"
BASE = DATA_DIR / "state_atcenter_mesh300_y300_xend195_angle308_T157_Mach215.dat"

LABELS = {"rho": r"$\rho'$", "u": r"$u'$", "v": r"$v'$", "w": r"$w'$", "T": r"$T'$"}


def his_mode(base, kind, f, w):
    re_ = read_tecplot(MODES / f"{kind}_atcenter_f{f}_w{w}_n1real.dat")
    im_ = read_tecplot(MODES / f"{kind}_atcenter_f{f}_w{w}_n1imag.dat")
    cons = {k: re_[k] + 1j * im_[k] for k in CONS}
    return linearise_to_primitive(cons, base)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-dir", default="results/campaign",
                     help="Directory containing the per-point sav_map_* subdirectories")
    ap.add_argument("--out-dir", default="figures", help="Output directory for the PNGs")
    args = ap.parse_args()
    campaign_dir = Path(args.campaign_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    CASES = [
        dict(name="mack", title="Medium-frequency 1st Mack mode",
             f="2", w="45", ftex="2", btex="45",
             npz=str(campaign_dir / "sav_map_b45p0_f2" / "sav_map_b45p0_f2.npz")),
        dict(name="liftup", title="Low-frequency lift-up mode",
             f="0.0001", w="163", ftex=r"10^{-4}", btex="163",
             npz=str(campaign_dir / "sav_map_b163p0_f0p0001" / "sav_map_b163p0_f0p0001.npz")),
    ]

    base = read_tecplot(BASE)

    for case in CASES:
        d = np.load(case["npz"], allow_pickle=True)
        X, Y = d["X"], d["Y"]
        W = d["W_diag"]
        idx = {v: d[f"idx_{v}_j"] for v in PRIM}
        sig2 = float(d["gains"][0]) ** 2

        # his response, mapped onto our dof ordering
        hp = his_mode(base, "response", case["f"], case["w"])
        q_his = np.zeros(W.size, dtype=complex)
        for v in PRIM:
            q_his[idx[v]] = hp[v].T.ravel(order="F")
        q_our = d["responses"][:, 0].astype(complex)

        # unit Chu norm for both, then rotate ours onto his phase. His temporal
        # Fourier convention is opposite ours (same source as the opposite
        # stability signs), so the physically matching object can be conj(q_our):
        # take whichever of {q, conj(q)} correlates better and say which was used.
        q_his /= np.sqrt(np.real(np.vdot(q_his, W * q_his)))
        q_our /= np.sqrt(np.real(np.vdot(q_our, W * q_our)))
        c_plain = np.vdot(q_his, W * q_our)
        c_conj = np.vdot(q_his, W * np.conj(q_our))
        conjugated = abs(c_conj) > abs(c_plain)
        if conjugated:
            q_our = np.conj(q_our)
            c = c_conj
        else:
            c = c_plain
        q_our *= np.conj(c) / abs(c)
        align = abs(c)  # Chu-norm correlation between the two unit modes

        fields = {}
        for v in PRIM:
            fields[v] = (np.real(q_his[idx[v]].reshape(X.shape, order="F")),
                         np.real(q_our[idx[v]].reshape(X.shape, order="F")))

        # adaptive y-crop: smallest y containing 99.9% of both modes' Chu energy
        def ycut(q):
            e = np.zeros(X.shape)
            for v in PRIM:
                e += (W[idx[v]] * np.abs(q[idx[v]]) ** 2).reshape(X.shape, order="F")
            prof = np.cumsum(e.sum(axis=1))
            prof /= prof[-1]
            return Y[np.searchsorted(prof, 0.999), 0]
        ymax = 1.15 * max(ycut(q_his), ycut(q_our))

        manuscript = MANUSCRIPT
        figsize = (8.6, 7.6) if manuscript else (9.5, 11.5)
        fig, axes = plt.subplots(len(PRIM), 2, figsize=figsize,
                                 sharex=True, sharey=True, constrained_layout=True)
        if manuscript:
            fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.015,
                                        wspace=0.015)
        for i, v in enumerate(PRIM):
            fh, fo = fields[v]
            m = np.abs(Y) <= ymax
            lim = max(np.percentile(np.abs(fh[m]), 99.8),
                      np.percentile(np.abs(fo[m]), 99.8))
            for j, F in enumerate((fh, fo)):
                ax = axes[i, j]
                pc = ax.pcolormesh(X, Y, F, cmap="RdBu_r", vmin=-lim, vmax=lim,
                                   rasterized=True, shading="auto")
                ax.set_ylim(0, ymax)
                if j == 0:
                    ax.set_ylabel(r"$y/L$")
            axes[i, 0].text(-0.22, 0.5, LABELS[v], transform=axes[i, 0].transAxes,
                            fontsize=13 if manuscript else 15, va="center",
                            ha="center")
            fig.colorbar(pc, ax=axes[i, :], shrink=0.9, pad=0.008, aspect=13)
        axes[0, 0].set_title("Reference" if manuscript else "Savarino (published)",
                             fontsize=12)
        axes[0, 1].set_title("linstabpy", fontsize=12)
        for ax in axes[-1, :]:
            ax.set_xlabel(r"$x/L$")
        if not manuscript:
            fig.suptitle(
                rf"{case['title']} — response mode, $f = {case['ftex']}$, "
                rf"$\beta = {case['btex']}$"
                + "\n"
                + rf"unit Chu norm, phase-aligned"
                + (r" (ours conjugated)" if conjugated else "")
                + rf"; $|\langle q_\mathrm{{ref}}, q_\mathrm{{ours}}\rangle_E| = {align:.4f}$;"
                + rf"  our $\sigma^2\times10^{{-4}} = {sig2/1e4:.3f}$",
                fontsize=12)
        suffix = "_ms" if manuscript else ""
        out = out_dir / f"compare_mode_{case['name']}{suffix}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(case["name"], "->", out, " align=%.4f" % align, " ymax=%.3f" % ymax)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
