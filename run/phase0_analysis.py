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

    # Full spectrum snapshot (informational)
    print(f"  Returned {len(omega)} modes. All omegas (sorted by |Im(w)|):")
    sort_order = np.argsort(np.abs(omega.imag))
    for j in sort_order:
        in_win = in_sawtooth_window(omega[j])
        rf = rho_active_fraction(modes, int(j), ngp)
        print(f"    mode {int(j):2d}: w={omega[j]:.4f}  "
              f"rho_frac={rf:.3f}  "
              f"{'[window]' if in_win else ''}")

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

    # All modes in saw-tooth omega-window — compute tang_adj_step for every
    # one regardless of rho-active filter. The rho-active flag is used for
    # the verdict but we want the full picture.
    window_modes = []
    for j in range(len(omega)):
        if not in_sawtooth_window(omega[j]):
            continue
        rho_frac = rho_active_fraction(modes, j, ngp)
        rho2d = reshape_rho(modes, j, ngp, Ny, Nx)
        step_bot, step_top = tang_adj_step(rho2d)
        window_modes.append({
            "mode_idx": j,
            "omega": complex(omega[j]),
            "rho_frac": rho_frac,
            "rho_active": rho_frac > RHO_ACTIVE_THRESHOLD,
            "postcheck_rel": float(postcheck[j]),
            "step_bot": step_bot,
            "step_top": step_top,
            "step_max": max(step_bot, step_top),
        })
    window_modes.sort(key=lambda r: r["step_max"], reverse=True)
    candidates = [m for m in window_modes if m["rho_active"]]

    print(f"  Window modes: {len(window_modes)} "
          f"({len(candidates)} rho-active)")
    for m in window_modes:
        flag = "[rho-active]" if m["rho_active"] else "[rho-weak]"
        print(f"    mode {m['mode_idx']}: w={m['omega']:.4f}, "
              f"rho_frac={m['rho_frac']:.3f}, "
              f"step_max={m['step_max']:.4f} {flag}")

    result = {"Ny": Ny, "Nx": Nx, "bgs": bgs,
              "window_modes": window_modes, "candidates": candidates}
    d.close()
    return result


def main():
    results = {}
    for tag, npz_name, _, _ in RUNS:
        results[tag] = analyze_run(tag, npz_name)

    # --- Validity check: structural gate on fd4_s4 baseline ---
    # BG-S step_max is NOT asserted against an absolute threshold: it tracks
    # grid resolution (stride-4 has baseline ~0.14, stride-2 has ~0.025)
    # rather than saw-tooth contamination. We report it alongside the
    # candidates so the reader can see the saw-tooth-to-baseline ratio.
    print("\n=== Grid baseline (BG-S step_max, informational) ===")
    for tag, r in results.items():
        b = r["bgs"]
        if b["captured"]:
            print(f"  {tag}: BG-S step_max = {b['step_max']:.4f} "
                  f"(mode {b['mode_idx']}, w={b['omega']:.4f})")
        else:
            print(f"  {tag}: BG-S not captured "
                  f"(nearest dist={b['dist_to_target']:.3f})")

    # Gate: fd4_s2 (clean refined baseline) must have no window mode
    # >= VERDICT_PROCEED. We compare against the same-stride fd4 run, not
    # fd4_s4: at stride-4 every mode has near-wall step ~0.15 from
    # under-resolved grid noise, so the metric only discriminates at
    # stride-2 where BG-S baseline drops to ~0.025.
    fd4_s2_window = results["fd4_s2"]["window_modes"]
    fd4_s2_max = max((m["step_max"] for m in fd4_s2_window), default=0.0)
    if fd4_s2_max >= VERDICT_PROCEED:
        raise AssertionError(
            f"Gate FAIL: fd4_s2 (clean refined baseline) has window "
            f"step_max = {fd4_s2_max:.4f} >= {VERDICT_PROCEED}. "
            f"Saw-tooth omega-window matches modes fd4 should not produce.")
    print(f"\n=== Window calibration gate ===")
    print(f"  fd4_s2 window-max = {fd4_s2_max:.4f} "
          f"[OK, < {VERDICT_PROCEED}]")

    # --- Verdict ---
    # Use ALL window-matched modes, not just rho-active ones. The saw-tooth
    # signature is high tang_adj_step in the rho component within the
    # known frequency family; rho-fraction varies with grid resolution
    # (stride-4: rho_frac ~0.5; stride-2: same family at ~0.2 because
    # energy redistributes across variables) but the rho-step persists.
    fd6_s2_window = results["fd6_s2"]["window_modes"]
    fd6_s2_max = max((m["step_max"] for m in fd6_s2_window), default=0.0)
    print("\n=== Verdict ===")
    print(f"fd6 stride-2 window modes: {len(fd6_s2_window)}, "
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
        ("## BG-S control (grid-resolution baseline, informational — "
         "no absolute threshold)"),
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
            f"## {tag} - sawtooth-window modes "
            f"({len(r['window_modes'])} total, "
            f"{len(r['candidates'])} rho-active)")
        lines.append("")
        if not r["window_modes"]:
            lines.append("_No modes in sawtooth omega-window._")
            lines.append("")
            continue
        lines.append(
            "| mode | w | rho-frac | rho-active | postcheck | step_bot | step_top | step_max |")
        lines.append(
            "|---|---|---|---|---|---|---|---|")
        for m in r["window_modes"]:
            active = "yes" if m["rho_active"] else "no"
            lines.append(
                f"| {m['mode_idx']} | `{m['omega']:.4f}` | "
                f"{m['rho_frac']:.3f} | {active} | "
                f"{m['postcheck_rel']:.3e} | "
                f"{m['step_bot']:.4f} | {m['step_top']:.4f} | "
                f"**{m['step_max']:.4f}** |")
        lines.append("")
    md_path.write_text("\n".join(lines))
    print(f"\nWrote {md_path}")

    # --- Figure: 2x2, worst-mode near-wall rho line-cut per run ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (tag, r) in zip(axes.flatten(), results.items()):
        if not r["window_modes"]:
            ax.text(0.5, 0.5,
                    f"{tag}\nno modes in sawtooth omega-window",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        # Show worst window mode (sorted by step_max); annotate rho-active status
        worst = r["window_modes"][0]
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
        rho_flag = "rho-active" if worst["rho_active"] else f"rho-weak ({worst['rho_frac']:.2f})"
        ax.set_title(
            f"{tag}: mode {worst['mode_idx']}, w={worst['omega']:.3f} [{rho_flag}]\n"
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
        sidecar[f"{tag}_window_omegas"] = np.array(
            [m["omega"] for m in r["window_modes"]], dtype=complex)
        sidecar[f"{tag}_window_step_max"] = np.array(
            [m["step_max"] for m in r["window_modes"]])
        sidecar[f"{tag}_window_rho_frac"] = np.array(
            [m["rho_frac"] for m in r["window_modes"]])
        sidecar[f"{tag}_window_rho_active"] = np.array(
            [m["rho_active"] for m in r["window_modes"]])
    npz_path = HERE / "phase0_analysis.npz"
    np.savez_compressed(npz_path, **sidecar)
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
