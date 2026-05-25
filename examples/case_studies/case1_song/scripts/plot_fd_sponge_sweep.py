#!/usr/bin/env python3
"""Plot Case 1 fd-order sensitivity for the sigma=0.5 smooth sponge study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
REPORT_JSON = CASE_DIR / "output" / "fd_sponge" / "case1_perf_memory_latest.json"
OUTPUT_PNG = CASE_DIR / "output" / "fd_sponge" / "case1_fd_sponge_sweep.png"

FD_COLORS = {4: "#4e79a7", 6: "#f28e2b", 8: "#e15759"}


def _diag_or_flat(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return np.diag(arr).astype(complex)
    return arr.reshape(-1).astype(complex)


def _load_rows() -> list[dict]:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
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
        if str(sponge_cfg.get("preset")) != "case1_smooth_soft":
            continue
        with np.load(case["artifact_npz"], allow_pickle=False) as data:
            omega = np.asarray(data["omega"]).reshape(-1).astype(complex)
        rows.append(
            {
                "case_id": case["case_id"],
                "fd_order": int(numerics.get("fd_order", -1)),
                "elapsed_seconds": float(case.get("metrics", {}).get("elapsed_seconds") or 0.0),
                "peak_rss_gib": float(case.get("metrics", {}).get("peak_rss_bytes") or 0.0)
                / 1024**3,
                "residual_max": case.get("output", {}).get("stability_residual_max"),
                "omega": omega,
            }
        )
    rows.sort(key=lambda row: row["fd_order"])
    return rows


def main() -> int:
    rows = _load_rows()
    if not rows:
        raise SystemExit("No completed fd-order sponge cases found.")

    matlab = _diag_or_flat(
        np.asarray(
            loadmat(CASE_DIR / "reference" / "matlab" / "MATLAB_exports" / "spectrum.mat")["omega"]
        )
    )
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    ax_spec = fig.add_subplot(gs[0, :])
    ax_cost = fig.add_subplot(gs[1, 0])
    ax_lead = fig.add_subplot(gs[1, 1])

    ax_spec.scatter(
        np.real(matlab),
        np.imag(matlab),
        s=22,
        marker="x",
        linewidths=0.8,
        color="0.5",
        alpha=0.55,
        label="MATLAB fd4 reference",
    )
    for row in rows:
        fd = row["fd_order"]
        omega = row["omega"]
        ax_spec.scatter(
            np.real(omega),
            np.imag(omega),
            s=18,
            facecolors="none",
            edgecolors=FD_COLORS.get(fd, "C0"),
            linewidths=1.0,
            alpha=0.8,
            label=f"fd{fd}",
        )
    ax_spec.axhline(0.0, color="0.7", linewidth=0.8)
    ax_spec.set_xlabel("Re(omega)")
    ax_spec.set_ylabel("Im(omega)")
    ax_spec.set_title(r"Case 1 spectrum with smooth sponge, $\sigma_{max}=0.5$")
    ax_spec.grid(alpha=0.2)
    handles, labels = ax_spec.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_spec.legend(unique.values(), unique.keys(), frameon=False, ncol=4, fontsize=9)

    fd_vals = np.array([row["fd_order"] for row in rows], dtype=int)
    times = np.array([row["elapsed_seconds"] / 60.0 for row in rows], dtype=float)
    memory = np.array([row["peak_rss_gib"] for row in rows], dtype=float)
    residuals = np.array(
        [float(row["residual_max"]) if row["residual_max"] is not None else np.nan for row in rows],
        dtype=float,
    )
    ax_cost.plot(fd_vals, times, marker="o", color="#4e79a7", label="time [min]")
    ax_cost.set_xlabel("FD order")
    ax_cost.set_ylabel("Wall time [min]", color="#4e79a7")
    ax_cost.tick_params(axis="y", labelcolor="#4e79a7")
    ax_cost.grid(alpha=0.25)
    ax_cost2 = ax_cost.twinx()
    ax_cost2.plot(fd_vals, memory, marker="s", color="#e15759", label="peak RSS [GiB]")
    ax_cost2.set_ylabel("Peak RSS [GiB]", color="#e15759")
    ax_cost2.tick_params(axis="y", labelcolor="#e15759")
    for fd, t, res in zip(fd_vals, times, residuals, strict=True):
        if np.isfinite(res):
            ax_cost.annotate(
                f"res {res:.1e}",
                (fd, t),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
    ax_cost.set_title("Cost and residual by FD order")

    lead = np.array([row["omega"][0] for row in rows], dtype=complex)
    ax_lead.plot(np.real(lead), np.imag(lead), color="0.55", linewidth=1.0)
    for row, val in zip(rows, lead, strict=True):
        fd = row["fd_order"]
        ax_lead.scatter(
            np.real(val),
            np.imag(val),
            s=60,
            color=FD_COLORS.get(fd, "C0"),
        )
        ax_lead.annotate(
            f"fd{fd}",
            (np.real(val), np.imag(val)),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax_lead.set_xlabel("Re(omega_1)")
    ax_lead.set_ylabel("Im(omega_1)")
    ax_lead.set_title("Leading mode trajectory")
    ax_lead.grid(alpha=0.25)

    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
