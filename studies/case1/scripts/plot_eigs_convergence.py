#!/usr/bin/env python3
"""Plot Case 1 eig residual estimates for the native-sponge study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
DEFAULT_REPORT = CASE_DIR / "output" / "case1_perf_memory_latest.json"
OUTPUT_PNG = CASE_DIR / "output" / "case1_eigs_convergence.png"

FD_COLORS = {4: "#f28e2b", 6: "#e15759"}


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _rows(report: dict) -> list[dict]:
    rows: list[dict] = []
    for case in report.get("cases", []):
        if case.get("status") != "ok":
            continue
        resolved = dict(case.get("resolved_config", {}))
        analysis = dict(resolved.get("analysis", {}))
        numerics = dict(resolved.get("numerics", {}))
        physics = dict(resolved.get("physics", {}))
        sponge_cfg = physics.get("sponge", {}) if isinstance(physics.get("sponge"), dict) else {}
        if not analysis.get("run_stability") or analysis.get("run_resolvent"):
            continue
        rows.append(
            {
                "case_id": case.get("case_id"),
                "fd_order": int(numerics.get("fd_order", -1)),
                "sponge_enabled": bool(sponge_cfg.get("enabled", False)),
                "artifact_npz": Path(case["artifact_npz"]),
                "n_modes": int(case.get("output", {}).get("n_modes_stability_requested") or 0),
            }
        )
    rows.sort(key=lambda row: (row["fd_order"], row["sponge_enabled"]))
    return rows


def main() -> int:
    report = _load_report(DEFAULT_REPORT)
    rows = _rows(report)
    if not rows:
        print("no completed Case 1 runs in perf report")
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, row in zip(axes.ravel(), rows, strict=False):
        data = np.load(row["artifact_npz"], allow_pickle=False)
        residuals = np.asarray(data["stability_residual_estimates"], dtype=float).reshape(-1)
        modes = np.arange(1, residuals.size + 1)
        ax.semilogy(
            modes,
            np.maximum(residuals, 1e-18),
            color=FD_COLORS.get(row["fd_order"], "C0"),
            linewidth=1.2,
        )
        ax.set_title(f"fd{row['fd_order']}, sponge {'on' if row['sponge_enabled'] else 'off'}")
        ax.grid(alpha=0.25, which="both")
        ax.text(
            0.03,
            0.04,
            f"nev={row['n_modes']}\nmax={float(np.nanmax(residuals)):.1e}\nmedian={float(np.nanmedian(residuals)):.1e}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("EPS residual estimate")
    for ax in axes[-1, :]:
        ax.set_xlabel("Mode rank")

    fig.suptitle("Case 1 eig residual estimates at nev = 150", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
