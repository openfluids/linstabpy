#!/usr/bin/env python3
"""Score an EVP run against PLAN.md acceptance criteria.

Usage:
    .venv/bin/python results/case2/phase1p5_score_run.py \
        results/case2/output/<campaign>/runs/<case>.npz

Reads the npz artifact and the sibling .config.json and prints:
    - eigenvalues returned
    - postcheck residuals
    - target-distance values
    - whether each eigenvalue meets PLAN Acceptance Criteria
    - verdict on the oscillatory-branch localisation test (|Re ω| > 0.2)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def score_run(npz_path: Path) -> dict:
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    cfg_path = npz_path.with_suffix(".config.json")
    with np.load(npz_path, allow_pickle=True) as d:
        omegas = d["omega"] if "omega" in d.files else np.array([])
        postcheck = d.get("stability_postcheck_residual_rel", np.array([]))
        tardist = d.get("stability_postcheck_target_distance", np.array([]))
        nconv = int(d.get("stability_eps_nconv", np.array([0]))[()]) if "stability_eps_nconv" in d.files else None
        niter = int(d.get("stability_eps_iterations", np.array([0]))[()]) if "stability_eps_iterations" in d.files else None
    cfg = {}
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
    target = None
    ct = cfg.get("analysis", {}).get("omega_target")
    if isinstance(ct, dict):
        target = complex(ct.get("real", 0.0), ct.get("imag", 0.0))
    return {
        "npz": str(npz_path),
        "omegas": omegas,
        "postcheck": np.atleast_1d(np.asarray(postcheck)),
        "target_distance": np.atleast_1d(np.asarray(tardist)),
        "nconv": nconv,
        "niter": niter,
        "target": target,
        "cfg": cfg,
    }


def pretty_score(r: dict) -> None:
    print(f"Run: {r['npz']}")
    tgt = r["target"]
    print(f"  target omega: {tgt}")
    print(f"  nconv: {r['nconv']}  iterations: {r['niter']}")
    omegas = r["omegas"]
    print(f"  omegas returned ({len(omegas)}):")
    for i, w in enumerate(omegas):
        pc = r["postcheck"][i] if i < len(r["postcheck"]) else float("nan")
        td = r["target_distance"][i] if i < len(r["target_distance"]) else float("nan")
        line = f"    [{i}] ω = {w.real:+.4f} {w.imag:+.4f}j  postcheck_rel={pc:.2e}  target_dist={td:.3e}"
        print(line)

    # PLAN Acceptance Criteria
    print("\n  PLAN acceptance gates:")
    if len(omegas) == 0:
        print("    FAIL — no omegas returned")
        return
    pc_med = float(np.median(r["postcheck"])) if len(r["postcheck"]) else float("nan")
    td_med = float(np.median(r["target_distance"])) if len(r["target_distance"]) else float("nan")
    tgt_mag = abs(tgt) if tgt is not None else float("nan")
    print(f"    postcheck median ({pc_med:.2e})  <= 1e-4 ?   "
          f"{'PASS' if pc_med <= 1e-4 else 'FAIL'}")
    print(f"    target-dist median ({td_med:.3e}) <= 0.1*|ω_target|={0.1*tgt_mag:.3f}?   "
          f"{'PASS' if td_med <= 0.1 * tgt_mag else 'FAIL'}")
    osc_re = float(np.max(np.abs(omegas.real))) if len(omegas) else 0.0
    print(f"    max |Re ω| = {osc_re:.3f}  >= 0.2 ?   "
          f"{'PASS (osc family localised)' if osc_re >= 0.2 else 'FAIL (stationary-only return)'}")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: phase1p5_score_run.py <npz-path> [npz-path ...]")
        sys.exit(2)
    for p in sys.argv[1:]:
        r = score_run(Path(p))
        pretty_score(r)
        print()


if __name__ == "__main__":
    main()
