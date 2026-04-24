#!/usr/bin/env python3
"""Phase 0 go/no-go analysis for the fd6 saw-tooth study (v2 rewrite).

Supersedes the prior `phase0_gonogo.py` in the parent repo, which used
cross-grid nearest-omega matching and produced ambiguous results. This
version does per-grid analysis with no cross-matching.

Reads four NPZs from `./output/runs/`:
  fd6_s4 : R3a_baseline.npz      - existing contaminated baseline
  fd4_s4 : B1_fd4_stability.npz  - clean fd4 baseline
  fd4_s2 : R3_stride2_fd4.npz    - Phase 0 refinement
  fd6_s2 : R3_stride2_fd6.npz    - verdict-focus

For each run:
  1. Identify saw-tooth-window + rho-active candidate modes.
  2. Compute tang_adj_step on both near-wall rows, take max.
  3. Also compute tang_adj_step on the BG-S control mode (if captured).

Verdict at fd6 s2 (plan):
  max tang_adj_step < 0.05   -> STOP (refinement fixes the saw-tooth)
  max tang_adj_step >= 0.10  -> PROCEED (closure artefact, do Phase 1)
  in between                  -> AMBIGUOUS

Writes:
  phase0_analysis.md
  phase0_analysis.png
  phase0_analysis.npz
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
NPZ_DIR = HERE / "output" / "runs"
CFG_PATH = HERE / "phase0_analysis.jsonc"

# linstabpy ships a jsonc loader; use it so this script matches the rest
# of the kit's config conventions.
sys.path.insert(0, str(HERE.parent / "src"))
from linstab2d.utils.jsonc import load_jsonc  # noqa: E402

CFG = load_jsonc(CFG_PATH)
OMEGA_WINDOW = CFG["omega_window"]
RHO_ACTIVE_THRESHOLD = float(CFG["rho_active_threshold"])
BGS_TARGET = complex(float(CFG["bgs_target_real"]),
                     float(CFG["bgs_target_imag"]))
BGS_MAX_DIST = float(CFG["bgs_max_dist"])
VERDICT_STOP = float(CFG["verdict_stop"])
VERDICT_PROCEED = float(CFG["verdict_proceed"])
PNG_DPI = int(CFG.get("png_dpi", 600))
RUNS = [(r["tag"], r["npz"], int(r["fd_order"]), int(r["stride"]))
        for r in CFG["runs"]]


def reshape_rho(modes, mode_idx, ngp, Ny, Nx):
    """Extract rho block and reshape to (Ny, Nx) in Fortran order."""
    rho_flat = modes[0 * ngp:1 * ngp, mode_idx]
    return rho_flat.reshape((Ny, Nx), order="F")


def rho_active_fraction(modes, mode_idx, ngp):
    """||rho||_2 / ||q||_2 for mode_idx."""
    q = modes[:, mode_idx]
    rho = q[0 * ngp:1 * ngp]
    return float(np.linalg.norm(rho) / max(np.linalg.norm(q), 1e-300))


def _row_step(row):
    peak = float(np.max(np.abs(row)))
    if peak < 1e-30:
        return float("nan")
    return float(np.max(np.abs(np.diff(row))) / peak)


def tang_adj_step(rho_2d):
    """Step fraction at row 1 (above bottom wall) and row -2 (below top wall)."""
    return _row_step(rho_2d[1, :]), _row_step(rho_2d[-2, :])


def in_sawtooth_window(omega):
    wr, wi = abs(omega.real), abs(omega.imag)
    return (OMEGA_WINDOW["re_abs_min"] <= wr <= OMEGA_WINDOW["re_abs_max"]
            and OMEGA_WINDOW["im_abs_min"] <= wi <= OMEGA_WINDOW["im_abs_max"])


def analyze_run(tag, npz_name):
    npz = NPZ_DIR / npz_name
    print(f"\n=== {tag} ({npz_name}) ===")
    d = np.load(npz)
    omega = d["stability_all_omega"]
    modes = d["direct_modes"]
    X = d["X"]
    Ny, Nx = X.shape
    ngp = Ny * Nx
    postcheck = d["stability_all_postcheck_rel"]

    assert modes.shape[0] == 5 * ngp, (
        f"{tag}: direct_modes.shape[0]={modes.shape[0]} != 5*Ny*Nx={5*ngp}. "
        f"Flattening convention mismatch.")

    # BG-S control
    bgs_idx = int(np.argmin(np.abs(omega - BGS_TARGET)))
    bgs_dist = float(np.abs(omega[bgs_idx] - BGS_TARGET))
    bgs_captured = bgs_dist < BGS_MAX_DIST
    if bgs_captured:
        bgs_rho2d = reshape_rho(modes, bgs_idx, ngp, Ny, Nx)
        bgs_step_bot, bgs_step_top = tang_adj_step(bgs_rho2d)
    else:
        bgs_step_bot = bgs_step_top = float("nan")
    bgs = {
        "captured": bgs_captured,
        "mode_idx": bgs_idx,
        "omega": complex(omega[bgs_idx]),
        "dist_to_target": bgs_dist,
        "rho_frac": rho_active_fraction(modes, bgs_idx, ngp),
        "postcheck_rel": float(postcheck[bgs_idx]),
        "step_bot": bgs_step_bot,
        "step_top": bgs_step_top,
        "step_max": max(bgs_step_bot, bgs_step_top) if bgs_captured else float("nan"),
    }
    if bgs_captured:
        print(f"  BG-S (mode {bgs_idx}): w={omega[bgs_idx]:.4f}, "
              f"dist={bgs_dist:.3f}, step_max={bgs['step_max']:.4f}")
    else:
        print(f"  BG-S: not captured (nearest mode {bgs_idx} at "
              f"dist={bgs_dist:.3f} > {BGS_MAX_DIST})")

    # Saw-tooth-window + rho-active candidates
    candidates = []
    for j in range(len(omega)):
        if not in_sawtooth_window(omega[j]):
            continue
        rho_frac = rho_active_fraction(modes, j, ngp)
        if rho_frac <= RHO_ACTIVE_THRESHOLD:
            continue
        rho2d = reshape_rho(modes, j, ngp, Ny, Nx)
        step_bot, step_top = tang_adj_step(rho2d)
        candidates.append({
            "mode_idx": j,
            "omega": complex(omega[j]),
            "rho_frac": rho_frac,
            "postcheck_rel": float(postcheck[j]),
            "step_bot": step_bot,
            "step_top": step_top,
            "step_max": max(step_bot, step_top),
        })
    candidates.sort(key=lambda r: r["step_max"], reverse=True)
    print(f"  Candidates: {len(candidates)}")
    for c in candidates:
        print(f"    mode {c['mode_idx']}: w={c['omega']:.4f}, "
              f"rho_frac={c['rho_frac']:.3f}, step_max={c['step_max']:.4f}")

    result = {"Ny": Ny, "Nx": Nx, "bgs": bgs, "candidates": candidates}
    d.close()
    return result


def main():
    results = {}
    for tag, npz_name, _, _ in RUNS:
        results[tag] = analyze_run(tag, npz_name)

    # --- Validity gates ---
    print("\n=== Validity gates ===")

    # Gate 1: BG-S step_max < VERDICT_STOP on every captured run
    for tag, r in results.items():
        if not r["bgs"]["captured"]:
            print(f"  {tag}: BG-S skipped (not in returned set)")
            continue
        step = r["bgs"]["step_max"]
        if step < VERDICT_STOP:
            print(f"  {tag}: BG-S step_max = {step:.4f} [OK]")
        else:
            raise AssertionError(
                f"Gate 1 FAIL: {tag} BG-S step_max = {step:.4f} "
                f">= {VERDICT_STOP}. Metric is broken; do not apply verdict.")

    # Gate 2: fd4_s4 (clean baseline) has no candidate >= VERDICT_PROCEED
    fd4_s4_cands = results["fd4_s4"]["candidates"]
    fd4_s4_max = max((c["step_max"] for c in fd4_s4_cands), default=0.0)
    if fd4_s4_max >= VERDICT_PROCEED:
        raise AssertionError(
            f"Gate 2 FAIL: fd4_s4 (clean baseline) has candidate step_max = "
            f"{fd4_s4_max:.4f} >= {VERDICT_PROCEED}. Window/threshold "
            f"miscalibrated.")
    print(f"  fd4_s4 candidate-max = {fd4_s4_max:.4f} [OK, < {VERDICT_PROCEED}]")

    # --- Verdict ---
    fd6_s2_cands = results["fd6_s2"]["candidates"]
    fd6_s2_max = max((c["step_max"] for c in fd6_s2_cands), default=0.0)
    print("\n=== Verdict ===")
    print(f"fd6 stride-2 candidates: {len(fd6_s2_cands)}, "
          f"max step_max = {fd6_s2_max:.4f}")

    if fd6_s2_max < VERDICT_STOP:
        verdict = "STOP"
        verdict_text = (
            f"Max tang_adj_step at fd6 s2 = {fd6_s2_max:.4f} < {VERDICT_STOP}. "
            f"Stride refinement alone resolves the saw-tooth family. "
            f"No Phase 1 operator dissipation needed.")
    elif fd6_s2_max >= VERDICT_PROCEED:
        verdict = "PROCEED"
        verdict_text = (
            f"Max tang_adj_step at fd6 s2 = {fd6_s2_max:.4f} >= {VERDICT_PROCEED}. "
            f"Saw-tooth persists at refined grid as a closure artefact. "
            f"Phase 1 (SBP-compatible operator dissipation) is justified.")
    else:
        verdict = "AMBIGUOUS"
        verdict_text = (
            f"Max tang_adj_step at fd6 s2 = {fd6_s2_max:.4f} in "
            f"[{VERDICT_STOP}, {VERDICT_PROCEED}). Escalate to user.")
    print(f"VERDICT: {verdict}")
    print(verdict_text)

    # --- phase0_analysis.md ---
    md_path = HERE / "phase0_analysis.md"
    lines = [
        "# Phase 0 Go/No-Go Analysis (v2)",
        "",
        f"**Verdict: {verdict}**",
        "",
        verdict_text,
        "",
        "## Methodology",
        "",
        "- Per-grid analysis. No cross-grid omega matching.",
        (f"- Saw-tooth omega-window: |Re(w)| in "
         f"[{OMEGA_WINDOW['re_abs_min']}, {OMEGA_WINDOW['re_abs_max']}], "
         f"|Im(w)| in "
         f"[{OMEGA_WINDOW['im_abs_min']}, {OMEGA_WINDOW['im_abs_max']}]"),
        f"- rho-active threshold: ||rho||_2/||q||_2 > {RHO_ACTIVE_THRESHOLD}",
        "- Near-wall rows sampled: max(row 1, row Ny-2)",
        (f"- Decision: < {VERDICT_STOP} STOP, "
         f">= {VERDICT_PROCEED} PROCEED, else AMBIGUOUS"),
        "",
        "## BG-S control (must all be < 0.05 when captured)",
        "",
        "| Case | captured | mode | w | dist | rho-frac | step_bot | step_top | step_max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, r in results.items():
        b = r["bgs"]
        if b["captured"]:
            lines.append(
                f"| {tag} | yes | {b['mode_idx']} | "
                f"`{b['omega']:.4f}` | {b['dist_to_target']:.3f} | "
                f"{b['rho_frac']:.3f} | {b['step_bot']:.4f} | "
                f"{b['step_top']:.4f} | **{b['step_max']:.4f}** |"
            )
        else:
            lines.append(
                f"| {tag} | no | {b['mode_idx']} | "
                f"`{b['omega']:.4f}` | {b['dist_to_target']:.3f} | "
                f"- | - | - | - |"
            )
    lines.append("")
    for tag, r in results.items():
        lines.append(
            f"## {tag} - sawtooth-window candidates ({len(r['candidates'])})")
        lines.append("")
        if not r["candidates"]:
            lines.append(
                "_No modes satisfy sawtooth-window AND rho-active._")
            lines.append("")
            continue
        lines.append(
            "| mode | w | rho-frac | postcheck | step_bot | step_top | step_max |")
        lines.append(
            "|---|---|---|---|---|---|---|")
        for c in r["candidates"]:
            lines.append(
                f"| {c['mode_idx']} | `{c['omega']:.4f}` | "
                f"{c['rho_frac']:.3f} | {c['postcheck_rel']:.3e} | "
                f"{c['step_bot']:.4f} | {c['step_top']:.4f} | "
                f"**{c['step_max']:.4f}** |")
        lines.append("")
    md_path.write_text("\n".join(lines))
    print(f"\nWrote {md_path}")

    # --- Figure: 2x2, worst-mode near-wall rho line-cut per run ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (tag, r) in zip(axes.flatten(), results.items()):
        if not r["candidates"]:
            ax.text(0.5, 0.5,
                    f"{tag}\nno candidates in sawtooth window",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        worst = r["candidates"][0]
        npz_name = next(n for t, n, _, _ in RUNS if t == tag)
        d = np.load(NPZ_DIR / npz_name)
        modes = d["direct_modes"]
        Ny, Nx = r["Ny"], r["Nx"]
        ngp = Ny * Nx
        rho2d = reshape_rho(modes, worst["mode_idx"], ngp, Ny, Nx)
        d.close()
        row_idx = 1 if worst["step_bot"] >= worst["step_top"] else -2
        row = rho2d[row_idx, :]
        ax.plot(np.real(row), label="Re(rho)", linewidth=0.8)
        ax.plot(np.imag(row), label="Im(rho)", linewidth=0.8, alpha=0.7)
        label_row = "y=1 (above bot wall)" if row_idx == 1 else "y=Ny-2 (below top wall)"
        ax.set_title(
            f"{tag}: mode {worst['mode_idx']}, w={worst['omega']:.3f}\n"
            f"{label_row}, tang_adj_step_max={worst['step_max']:.4f}",
            fontsize=10)
        ax.set_xlabel("x index")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Phase 0: near-wall rho line-cut on worst sawtooth-window mode per run  ->  {verdict}",
        fontsize=12)
    fig.tight_layout()
    png_path = HERE / "phase0_analysis.png"
    fig.savefig(png_path, dpi=PNG_DPI)
    plt.close(fig)
    print(f"Wrote {png_path}")

    # --- Sidecar NPZ ---
    sidecar = {"verdict": np.array(verdict),
               "fd6_s2_max": np.array(fd6_s2_max)}
    for tag, r in results.items():
        b = r["bgs"]
        sidecar[f"{tag}_bgs_captured"] = np.array(b["captured"])
        sidecar[f"{tag}_bgs_step_max"] = np.array(b["step_max"])
        sidecar[f"{tag}_bgs_omega"] = np.array(b["omega"])
        sidecar[f"{tag}_bgs_dist"] = np.array(b["dist_to_target"])
        if r["candidates"]:
            sidecar[f"{tag}_cand_omegas"] = np.array(
                [c["omega"] for c in r["candidates"]])
            sidecar[f"{tag}_cand_step_max"] = np.array(
                [c["step_max"] for c in r["candidates"]])
            sidecar[f"{tag}_cand_rho_frac"] = np.array(
                [c["rho_frac"] for c in r["candidates"]])
        else:
            sidecar[f"{tag}_cand_omegas"] = np.array([], dtype=complex)
            sidecar[f"{tag}_cand_step_max"] = np.array([])
            sidecar[f"{tag}_cand_rho_frac"] = np.array([])
    npz_path = HERE / "phase0_analysis.npz"
    np.savez_compressed(npz_path, **sidecar)
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
