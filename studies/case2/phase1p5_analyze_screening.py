#!/usr/bin/env python3
"""Analyze a multi-case screening campaign.

Usage:
    .venv/bin/python studies/case2/phase1p5_analyze_screening.py \\
        <campaign-runs-dir>

Expects the sibling `.config.json` files for each `.npz`.
Outputs:
  - per-case scoring to stdout,
  - summary table,
  - combined spectrum plot (both targets + both sets of returned modes),
  - per-case mode-shape plots.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
OUT_DIR = SCRIPT.parent / "phase1p5"
OUT_DIR.mkdir(exist_ok=True)


def load_run(npz: Path) -> dict:
    cfg = json.loads(npz.with_suffix(".config.json").read_text())
    with np.load(npz, allow_pickle=True) as d:
        omegas = np.asarray(d.get("omega", np.array([], dtype=complex)))
        postcheck = np.atleast_1d(np.asarray(d.get("stability_postcheck_residual_rel", [])))
        tardist = np.atleast_1d(np.asarray(d.get("stability_postcheck_target_distance", [])))
        modes = np.asarray(d.get("direct_modes", np.zeros((0, 0))))
        idx_u = np.asarray(d.get("idx_u_j", []))
        X = np.asarray(d.get("X", []))
        Y = np.asarray(d.get("Y", []))
        nconv = int(d.get("stability_eps_nconv", [0])[()]) if "stability_eps_nconv" in d.files else None
    ct = cfg.get("analysis", {}).get("omega_target")
    target = complex(ct.get("real", 0.0), ct.get("imag", 0.0)) if isinstance(ct, dict) else None
    return {
        "npz": npz,
        "label": cfg.get("_case_id", npz.stem),
        "omegas": omegas,
        "postcheck": postcheck,
        "tardist": tardist,
        "modes": modes,
        "idx_u": idx_u,
        "X": X, "Y": Y,
        "nconv": nconv,
        "target": target,
    }


def score(r: dict, print_fn=print) -> dict:
    tgt = r["target"]
    omegas = r["omegas"]
    print_fn(f"Case {r['label']}   target={tgt}   nconv={r['nconv']}")
    pc_med = float(np.median(r["postcheck"])) if len(r["postcheck"]) else float("nan")
    pc_max = float(np.max(r["postcheck"])) if len(r["postcheck"]) else float("nan")
    td_med = float(np.median(r["tardist"])) if len(r["tardist"]) else float("nan")
    td_min = float(np.min(r["tardist"])) if len(r["tardist"]) else float("nan")
    max_re = float(np.max(np.abs(omegas.real))) if len(omegas) else 0.0
    tgt_mag = abs(tgt) if tgt else float("nan")
    gate_pc = pc_med <= 1e-4
    gate_td = td_med <= 0.1 * tgt_mag
    gate_re = max_re >= 0.2
    print_fn(f"  omegas returned ({len(omegas)}):")
    for i, w in enumerate(omegas):
        pc = r["postcheck"][i] if i < len(r["postcheck"]) else float("nan")
        td = r["tardist"][i] if i < len(r["tardist"]) else float("nan")
        print_fn(f"    [{i:2d}] ω = {w.real:+.4f} {w.imag:+.4f}j  pc={pc:.2e}  td={td:.3e}")
    print_fn(f"  postcheck median={pc_med:.2e}  max={pc_max:.2e}    "
             f"gate (<=1e-4): {'PASS' if gate_pc else 'FAIL'}")
    print_fn(f"  target-dist min={td_min:.3e} median={td_med:.3e}    "
             f"gate (<=0.1*|ω|={0.1*tgt_mag:.3f}): {'PASS' if gate_td else 'FAIL'}")
    print_fn(f"  max |Re ω|={max_re:.3f}    gate (>=0.2): "
             f"{'PASS (osc localised)' if gate_re else 'FAIL (stationary only)'}")
    return {"pc_gate": gate_pc, "td_gate": gate_td, "re_gate": gate_re,
            "pc_med": pc_med, "td_med": td_med, "max_re": max_re}


def plot_combined_spectrum(runs: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.axvline(0, color="k", lw=0.4, alpha=0.4)
    colors = ["C0", "C3"]
    markers = ["o", "s"]
    for run, color, marker in zip(runs, colors, markers):
        w = run["omegas"]
        ax.scatter(w.real, w.imag, marker=marker, color=color, s=70,
                    edgecolor="k", lw=0.6,
                    label=f"{run['label']} ({len(w)} modes)")
        t = run["target"]
        if t is not None:
            ax.scatter(t.real, t.imag, marker="x", color=color, s=200, lw=2.5)
            r = 0.1 * abs(t)
            theta = np.linspace(0, 2 * np.pi, 128)
            ax.plot(t.real + r * np.cos(theta), t.imag + r * np.sin(theta),
                    color=color, lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"Re $\omega$"); ax.set_ylabel(r"Im $\omega$")
    ax.set_title("M21 Qraw screening — both targets, nev=20, tol=5e-3\n"
                 "× = target, dashed = PLAN target-distance gate 0.1|ω|")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_mode_grid(run: dict, out: Path, max_modes: int = 12) -> None:
    modes = run["modes"]; idx_u = run["idx_u"]
    X = run["X"]; Y = run["Y"]
    omegas = run["omegas"]
    ny, nx = X.shape
    n = min(modes.shape[1], max_modes)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              constrained_layout=True, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for j in range(n):
        ax = axes[j // ncols, j % ncols]
        ax.axis("on")
        u_flat = modes[idx_u, j]
        u_grid = u_flat.reshape((ny, nx), order="F")
        mag = np.abs(u_grid)
        mag /= max(mag.max(), 1e-30)
        pc = ax.pcolormesh(X, Y, mag, shading="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_title(f"[{j}] ω = {omegas[j].real:+.3f} {omegas[j].imag:+.3f}j",
                      fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"|u| of returned eigenmodes — {run['label']}", fontsize=11)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: phase1p5_analyze_screening.py <campaign-runs-dir>")
        sys.exit(2)
    runs_dir = Path(sys.argv[1])
    npz_paths = sorted(runs_dir.glob("*.npz"))
    if not npz_paths:
        print(f"No .npz in {runs_dir}")
        sys.exit(1)
    runs = [load_run(p) for p in npz_paths]
    for r in runs:
        score(r)
        print()
    out_spec = OUT_DIR / f"screening_{runs_dir.parent.name}_spectrum.png"
    plot_combined_spectrum(runs, out_spec)
    for r in runs:
        out_modes = OUT_DIR / f"screening_{runs_dir.parent.name}_{r['label']}_modes.png"
        if r["modes"].size:
            plot_mode_grid(r, out_modes)


if __name__ == "__main__":
    main()
