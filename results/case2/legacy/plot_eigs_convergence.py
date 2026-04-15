#!/usr/bin/env python3
"""Plot Case 2 stability eigensolver convergence sweeps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parent
DEFAULT_REPORT = CASE_DIR / "output" / "eigs_convergence" / "case2_eigs_convergence_latest.json"
DEFAULT_PNG = CASE_DIR / "output" / "eigs_convergence" / "case2_eigs_convergence_summary.png"


def _load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [case for case in data.get("cases", []) if case.get("status") == "ok"]


def _case_value(case: dict, *keys, default=np.nan):
    node = case
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    return node if node is not None else default


def _series(cases: list[dict], tol: float) -> list[dict]:
    items: list[dict] = []
    for case in cases:
        cfg = case.get("resolved_config", {})
        tol_case = float(_case_value(cfg, "solver", "eigen", "eps_tol", default=np.nan))
        if not np.isfinite(tol_case) or abs(tol_case - tol) > max(1e-12, 1e-12 * abs(tol)):
            continue
        items.append(
            {
                "nev": int(_case_value(cfg, "analysis", "n_modes_stability", default=-1)),
                "ncv": int(_case_value(cfg, "solver", "eigen", "ncv", default=-1)),
                "elapsed": float(_case_value(case, "metrics", "elapsed_seconds", default=np.nan)),
                "nconv": int(_case_value(case, "output", "stability_eps_nconv", default=-1)),
                "eps_le_1e6": int(_case_value(case, "output", "stability_residual_relative_count_le_1e6", default=0)),
                "post_le_1e6": int(_case_value(case, "output", "stability_postcheck_residual_rel_count_le_1e6", default=0)),
                "post_median": float(_case_value(case, "output", "stability_postcheck_residual_rel_median", default=np.nan)),
            }
        )
    return sorted(items, key=lambda item: item["nev"])


def main() -> int:
    report_path = DEFAULT_REPORT
    out_path = DEFAULT_PNG
    cases = _load_cases(report_path)
    if not cases:
        raise SystemExit(f"no successful cases found in {report_path}")

    tolerances = sorted(
        {
            float(_case_value(case, "resolved_config", "solver", "eigen", "eps_tol", default=np.nan))
            for case in cases
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    colors = {1e-4: "#c76b00", 1e-5: "#00798c", 1e-6: "#7a3e9d"}
    labels = {1e-4: r"$10^{-4}$", 1e-5: r"$10^{-5}$", 1e-6: r"$10^{-6}$"}

    for tol in tolerances:
        items = _series(cases, tol)
        if not items:
            continue
        nev = np.array([item["nev"] for item in items], dtype=float)
        nconv = np.array([item["nconv"] for item in items], dtype=float)
        eps_ok = np.array([item["eps_le_1e6"] for item in items], dtype=float)
        post_ok = np.array([item["post_le_1e6"] for item in items], dtype=float)
        elapsed_min = np.array([item["elapsed"] / 60.0 for item in items], dtype=float)
        color = colors.get(tol, None)
        label = labels.get(tol, f"{tol:g}")

        axes[0].plot(nev, nconv, marker="o", linewidth=2, color=color, label=label)
        axes[0].plot(nev, post_ok, marker="s", linewidth=2, linestyle="--", color=color)
        axes[1].plot(nev, elapsed_min, marker="o", linewidth=2, color=color, label=label)
        axes[2].plot(nev, eps_ok, marker="o", linewidth=2, color=color, label=label)
        axes[2].plot(nev, post_ok, marker="s", linewidth=2, linestyle="--", color=color)

    axes[0].set_title("Converged vs Accepted")
    axes[0].set_xlabel("Requested nev")
    axes[0].set_ylabel("Mode count")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(title="EPS tol")
    axes[0].text(
        0.02,
        0.02,
        "solid = EPS nconv\n dashed = postcheck <= 1e-6",
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
    )

    axes[1].set_title("Runtime")
    axes[1].set_xlabel("Requested nev")
    axes[1].set_ylabel("Elapsed minutes")
    axes[1].grid(True, alpha=0.25)

    axes[2].set_title("Residual Gate <= 1e-6")
    axes[2].set_xlabel("Requested nev")
    axes[2].set_ylabel("Mode count")
    axes[2].grid(True, alpha=0.25)
    axes[2].text(
        0.02,
        0.02,
        "solid = EPS relative residual\n dashed = explicit postcheck residual",
        transform=axes[2].transAxes,
        fontsize=9,
        va="bottom",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
