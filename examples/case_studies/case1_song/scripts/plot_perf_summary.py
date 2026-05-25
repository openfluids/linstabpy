#!/usr/bin/env python3
"""Plot Case 1 native-sponge study runtime, memory, and stage breakdown."""

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
DEFAULT_SUMMARY_PNG = CASE_DIR / "output" / "case1_perf_summary.png"
ALIAS_SUMMARY_PNG = CASE_DIR / "output" / "case1_scaling_summary.png"
DEFAULT_BREAKDOWN_PNG = CASE_DIR / "output" / "case1_stage_breakdown.png"

FD_COLORS = {4: "#f28e2b", 6: "#e15759"}
STAGE_KEYS = (
    ("load_case", "load", "#4e79a7"),
    ("build_mesh", "mesh", "#59a14f"),
    ("assemble_operator_bc", "assemble", "#f28e2b"),
    ("compute_chu_norm", "chu", "#76b7b2"),
    ("stability", "stability", "#e15759"),
    ("save_results", "save", "#b07aa1"),
)


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _case_rows(report: dict) -> list[dict]:
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
                "label": case.get("label", case.get("case_id")),
                "fd_order": int(numerics.get("fd_order", -1)),
                "sponge_enabled": bool(sponge_cfg.get("enabled", False)),
                "elapsed_seconds": float(case.get("metrics", {}).get("elapsed_seconds") or 0.0),
                "peak_rss_gib": float(case.get("metrics", {}).get("peak_rss_bytes") or 0.0)
                / 1024**3,
                "stage_timings": dict(case.get("parsed", {}).get("stage_timings_seconds", {})),
                "residual_max": case.get("output", {}).get("stability_residual_max"),
                "n_modes": int(case.get("output", {}).get("n_modes_stability_requested") or 0),
            }
        )
    rows.sort(key=lambda row: (row["fd_order"], row["sponge_enabled"]))
    return rows


def _short_label(row: dict) -> str:
    return f"fd{row['fd_order']} | sponge {'on' if row['sponge_enabled'] else 'off'}"


def _plot_summary(rows: list[dict]) -> None:
    if not rows:
        return
    x = np.arange(len(rows))
    labels = [_short_label(row) for row in rows]
    times = np.array([row["elapsed_seconds"] / 60.0 for row in rows], dtype=float)
    memory = np.array([row["peak_rss_gib"] for row in rows], dtype=float)
    colors = [FD_COLORS.get(row["fd_order"], "#4e79a7") for row in rows]
    hatches = ["//" if row["sponge_enabled"] else "" for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    bars_time = axes[0].bar(x, times, color=colors, alpha=0.9, width=0.72)
    bars_mem = axes[1].bar(x, memory, color=colors, alpha=0.9, width=0.72)
    for bar, hatch in zip(bars_time, hatches, strict=True):
        bar.set_hatch(hatch)
    for bar, hatch in zip(bars_mem, hatches, strict=True):
        bar.set_hatch(hatch)
    for idx, row in enumerate(rows):
        axes[0].annotate(
            f"{times[idx]:.2f} m\nnev={row['n_modes']}",
            (x[idx], times[idx]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
        res = row.get("residual_max")
        axes[1].annotate(
            f"{memory[idx]:.2f} GiB\nmax res {res:.1e}"
            if res is not None
            else f"{memory[idx]:.2f} GiB",
            (x[idx], memory[idx]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    axes[0].set_ylabel("Wall time [min]")
    axes[0].set_title("Case 1 runtime by FD order and native sponge branch")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_ylabel("Peak RSS [GiB]")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x, labels)
    fig.tight_layout()
    DEFAULT_SUMMARY_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEFAULT_SUMMARY_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(ALIAS_SUMMARY_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_breakdown(rows: list[dict]) -> None:
    if not rows:
        return
    labels = [_short_label(row) for row in rows]
    x = np.arange(len(rows))
    bottoms = np.zeros(len(rows), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    for stage_key, stage_label, color in STAGE_KEYS:
        values = (
            np.array(
                [float(row["stage_timings"].get(stage_key, 0.0)) for row in rows],
                dtype=float,
            )
            / 60.0
        )
        if np.allclose(values, 0.0):
            continue
        ax.bar(x, values, bottom=bottoms, width=0.72, color=color, label=stage_label)
        bottoms += values

    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Time [min]")
    ax.set_title("Case 1 stage breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=6, fontsize=8)
    fig.tight_layout()
    DEFAULT_BREAKDOWN_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEFAULT_BREAKDOWN_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    report = _load_report(DEFAULT_REPORT)
    rows = _case_rows(report)
    _plot_summary(rows)
    _plot_breakdown(rows)
    print(f"wrote {DEFAULT_SUMMARY_PNG}")
    print(f"wrote {ALIAS_SUMMARY_PNG}")
    print(f"wrote {DEFAULT_BREAKDOWN_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
