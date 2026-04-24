#!/usr/bin/env python3
"""Plot Case 1 sigma-max sensitivity for the smooth sponge study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parent
REPORT_JSON = CASE_DIR / "output" / "sponge_sigma" / "case1_perf_memory_latest.json"
OUTPUT_PNG = CASE_DIR / "output" / "sponge_sigma" / "case1_sigma_sensitivity.png"

SIGMA_BY_PRESET = {
    "case1_smooth_quarter": 0.25,
    "case1_smooth_half": 0.5,
    "case1_smooth_unit": 1.0,
    "case1_smooth_double": 2.0,
}

COLORS = {
    0.25: "#4e79a7",
    0.5: "#59a14f",
    1.0: "#f28e2b",
    2.0: "#e15759",
}


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
        preset = str(sponge_cfg.get("preset", ""))
        sigma = SIGMA_BY_PRESET.get(preset)
        if sigma is None:
            continue
        with np.load(case["artifact_npz"], allow_pickle=False) as data:
            omega = np.asarray(data["omega"]).reshape(-1).astype(complex)
            direct_modes = np.asarray(data["direct_modes"]).astype(complex)
            sponge = np.asarray(data["sponge"], dtype=float)
            ngp = sponge.size
            mask = sponge.reshape(-1, order="F") > 1e-12
            support_fractions = []
            for k in range(min(10, direct_modes.shape[1])):
                v = direct_modes[:, k]
                cell = np.sqrt(
                    np.abs(v[:ngp]) ** 2
                    + np.abs(v[ngp : 2 * ngp]) ** 2
                    + np.abs(v[2 * ngp : 3 * ngp]) ** 2
                    + np.abs(v[3 * ngp : 4 * ngp]) ** 2
                    + np.abs(v[4 * ngp : 5 * ngp]) ** 2
                )
                total = float(np.linalg.norm(cell))
                sponge_norm = float(np.linalg.norm(cell[mask]))
                support_fractions.append(sponge_norm / total if total else np.nan)
            support_arr = np.asarray(support_fractions, dtype=float)
        rows.append(
            {
                "case_id": case["case_id"],
                "sigma": sigma,
                "fd_order": int(numerics.get("fd_order", -1)),
                "elapsed_seconds": float(case.get("metrics", {}).get("elapsed_seconds") or 0.0),
                "peak_rss_gib": float(case.get("metrics", {}).get("peak_rss_bytes") or 0.0) / 1024**3,
                "residual_max": case.get("output", {}).get("stability_residual_max"),
                "omega": omega,
                "sponge_support_median": float(np.nanmedian(support_arr)),
                "sponge_support_max": float(np.nanmax(support_arr)),
            }
        )
    rows.sort(key=lambda row: row["sigma"])
    return rows


def main() -> int:
    rows = _load_rows()
    if not rows:
        raise SystemExit("No completed sigma-sensitivity cases found.")

    matlab = _diag_or_flat(np.asarray(loadmat(CASE_DIR / "MATLAB_exports" / "spectrum.mat")["omega"]))
    sigmas = np.array([row["sigma"] for row in rows], dtype=float)
    residuals = np.array(
        [float(row["residual_max"]) if row["residual_max"] is not None else np.nan for row in rows],
        dtype=float,
    )
    lead = np.array([row["omega"][0] for row in rows], dtype=complex)
    support_median = np.array([row["sponge_support_median"] for row in rows], dtype=float)
    support_max = np.array([row["sponge_support_max"] for row in rows], dtype=float)

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    ax_spec = fig.add_subplot(gs[0, :])
    ax_lead = fig.add_subplot(gs[1, 0])
    ax_cost = fig.add_subplot(gs[1, 1])

    ax_spec.scatter(
        np.real(matlab),
        np.imag(matlab),
        s=20,
        marker="x",
        linewidths=0.8,
        color="0.5",
        alpha=0.55,
        label="MATLAB fd4 reference",
    )
    for row in rows:
        sigma = row["sigma"]
        omega = row["omega"]
        ax_spec.scatter(
            np.real(omega),
            np.imag(omega),
            s=18,
            facecolors="none",
            edgecolors=COLORS.get(sigma, "C0"),
            linewidths=1.0,
            alpha=0.8,
            label=fr"$\sigma_{{max}}={sigma:g}$",
        )
    ax_spec.axhline(0.0, color="0.7", linewidth=0.8)
    ax_spec.set_xlabel("Re(omega)")
    ax_spec.set_ylabel("Im(omega)")
    ax_spec.set_title("Case 1 eigenspectrum sensitivity to sponge strength")
    ax_spec.grid(alpha=0.2)
    handles, labels = ax_spec.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_spec.legend(unique.values(), unique.keys(), frameon=False, ncol=3, fontsize=9)

    ax_lead.plot(np.real(lead), np.imag(lead), color="0.55", linewidth=1.0, zorder=1)
    for row, val in zip(rows, lead, strict=True):
        sigma = row["sigma"]
        ax_lead.scatter(
            np.real(val),
            np.imag(val),
            s=60,
            color=COLORS.get(sigma, "C0"),
            label=fr"$\sigma_{{max}}={sigma:g}$",
            zorder=2,
        )
        ax_lead.annotate(
            f"{sigma:g}",
            (np.real(val), np.imag(val)),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax_lead.set_xlabel("Re(omega_1)")
    ax_lead.set_ylabel("Im(omega_1)")
    ax_lead.set_title("Leading mode trajectory")
    ax_lead.grid(alpha=0.25)

    ax_cost.plot(sigmas, 100.0 * support_median, marker="o", color="#4e79a7", label="median first-10")
    ax_cost.plot(sigmas, 100.0 * support_max, marker="s", color="#e15759", label="max first-10")
    ax_cost.set_xlabel(r"$\sigma_{max}$")
    ax_cost.set_ylabel("Mode norm in sponge zone [%]")
    ax_cost.grid(alpha=0.25)
    for sigma, res, val in zip(sigmas, residuals, support_median, strict=True):
        if np.isfinite(res):
            ax_cost.annotate(
                f"res {res:.1e}",
                (sigma, 100.0 * val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
    ax_cost.set_title("Leading-branch support inside sponge")
    ax_cost.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
