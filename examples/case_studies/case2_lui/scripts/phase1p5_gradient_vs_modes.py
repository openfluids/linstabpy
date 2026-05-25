#!/usr/bin/env python3
"""Render |∂Ū/∂x| and |∂Ū/∂y| on the M21 grid with contaminated-mode
peak locations marked.

Strong visual evidence for the "modes peak at mean-flow gradient
maxima" story.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from linstabpy.baseflow.case1 import load_hamada_case  # noqa: E402

CASE_DIR = SCRIPT.parents[1]
OUT_DIR = CASE_DIR / "artifacts" / "phase1p5"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    X, Y, bf, _ = load_hamada_case(SCRIPT.parents[1] / "data" / "baseflow",
                                    stride_ksi=2, stride_eta=1)
    U = bf.U
    dUdx = np.gradient(U, axis=1) / np.gradient(X, axis=1)
    dUdy = np.gradient(U, axis=0) / np.gradient(Y, axis=0)

    # Load the run's eigenmode peaks from npz if available
    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    mode_peaks: list[tuple[str, float, float, complex]] = []
    if npz_path and npz_path.exists():
        d = np.load(npz_path, allow_pickle=True)
        modes = d["direct_modes"]
        idx_u = d["idx_u_j"]
        omegas = d["omega"]
        Xf = X.ravel(order="F")
        Yf = Y.ravel(order="F")
        for j in range(modes.shape[1]):
            amp = np.abs(modes[idx_u, j])
            pi = int(np.argmax(amp))
            mode_peaks.append((f"Mode {j}", float(Xf[pi]), float(Yf[pi]),
                               complex(omegas[j])))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    for ax, field, name in [
        (axes[0], np.log10(np.abs(dUdx) + 1e-12), r"$\log_{10}|\partial \bar U/\partial x|$"),
        (axes[1], np.log10(np.abs(dUdy) + 1e-12), r"$\log_{10}|\partial \bar U/\partial y|$"),
    ]:
        pc = ax.pcolormesh(X, Y, field, shading="auto", cmap="magma",
                            vmin=-2, vmax=3.5)
        fig.colorbar(pc, ax=ax, label=name)
        ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(name)
        for label, xp, yp, w in mode_peaks:
            ax.plot(xp, yp, marker="x", mew=2.5, ms=14, color="cyan")
            ax.annotate(f"{label}\nω={w.imag:+.2f}j",
                        xy=(xp, yp), xytext=(10, 10),
                        textcoords="offset points",
                        color="cyan", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="black", edgecolor="cyan", alpha=0.7))

    title = "Mean-flow gradient magnitudes on M21, with contaminated-mode peaks"
    if npz_path:
        title += f"\n({npz_path.name})"
    fig.suptitle(title, fontsize=11)

    out = OUT_DIR / "qraw_gradient_vs_mode_peaks.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
