#!/usr/bin/env python3
"""Plot Case 1 stability/resolvent comparison figures for the sigma=0.5 study."""

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
OUTPUT_DIR = CASE_DIR / "output"

STABILITY_REPORT = OUTPUT_DIR / "stability_socket0" / "case1_perf_memory_latest.json"
RESOLVENT_REPORT = OUTPUT_DIR / "resolvent_socket1" / "case1_perf_memory_latest.json"
MATLAB_SPECTRUM = CASE_DIR / "reference" / "matlab" / "MATLAB_exports" / "spectrum.mat"

STABILITY_PNG = OUTPUT_DIR / "case1_stability_spectra_sigma_0p5.png"
RESOLVENT_PNG = OUTPUT_DIR / "case1_resolvent_gains_sigma_0p5.png"
PERF_PNG = OUTPUT_DIR / "case1_stability_resolvent_perf_sigma_0p5.png"

FD_COLORS = {4: "#4e79a7", 6: "#f28e2b", 8: "#e15759"}


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _diag_or_flat(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return np.diag(arr).astype(complex)
    return arr.reshape(-1).astype(complex)


def _rows_from_report(path: Path, *, stability: bool) -> list[dict]:
    report = _load_report(path)
    rows: list[dict] = []
    for case in report.get("cases", []):
        if case.get("status") != "ok":
            continue
        resolved = dict(case.get("resolved_config", {}))
        analysis = dict(resolved.get("analysis", {}))
        numerics = dict(resolved.get("numerics", {}))
        physics = dict(resolved.get("physics", {}))
        sponge_cfg = physics.get("sponge", {}) if isinstance(physics.get("sponge"), dict) else {}
        if bool(analysis.get("run_stability")) != stability:
            continue
        if bool(analysis.get("run_resolvent")) == stability:
            continue
        fd_order = int(numerics.get("fd_order", -1))
        artifact = Path(case["artifact_npz"])
        with np.load(artifact, allow_pickle=False) as data:
            row = {
                "case_id": case.get("case_id"),
                "fd_order": fd_order,
                "elapsed_seconds": float(case.get("metrics", {}).get("elapsed_seconds") or 0.0),
                "peak_rss_gib": float(case.get("metrics", {}).get("peak_rss_bytes") or 0.0)
                / 1024**3,
                "stage_timings": dict(case.get("parsed", {}).get("stage_timings_seconds", {})),
                "sponge_enabled": bool(sponge_cfg.get("enabled", False)),
                "omega": np.asarray(data["omega"], dtype=complex).reshape(-1),
                "gains": np.asarray(data["gains"], dtype=float).reshape(-1),
            }
            rows.append(row)
    rows.sort(key=lambda row: row["fd_order"])
    return rows


def _plot_stability(rows: list[dict]) -> None:
    matlab = _diag_or_flat(np.asarray(loadmat(MATLAB_SPECTRUM)["omega"]))
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.scatter(
        np.real(matlab),
        np.imag(matlab),
        s=18,
        marker="x",
        linewidths=0.8,
        color="0.45",
        alpha=0.55,
        label="MATLAB fd4 reference",
    )
    for row in rows:
        fd = row["fd_order"]
        omega = row["omega"]
        tmin = row["elapsed_seconds"] / 60.0
        ax.scatter(
            np.real(omega),
            np.imag(omega),
            s=18,
            facecolors="none",
            edgecolors=FD_COLORS.get(fd, "C0"),
            linewidths=1.0,
            alpha=0.85,
            label=f"Python fd{fd} ({tmin:.1f} min)",
        )
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.grid(alpha=0.22)
    ax.set_xlabel("Re(omega)")
    ax.set_ylabel("Im(omega)")
    ax.set_title(r"Case 1 stability spectra, sponge on, $\sigma_{max}=0.5$")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(STABILITY_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_resolvent(rows: list[dict]) -> None:
    fig, (ax_gain, ax_lead) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    for row in rows:
        fd = row["fd_order"]
        gains = row["gains"]
        modes = np.arange(1, gains.size + 1, dtype=int)
        ax_gain.plot(
            modes,
            gains,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            color=FD_COLORS.get(fd, "C0"),
            label=f"fd{fd}",
        )
    ax_gain.set_xlabel("Mode index")
    ax_gain.set_ylabel("Resolvent gain")
    ax_gain.set_title(r"Case 1 resolvent gains, sponge on, $\sigma_{max}=0.5$")
    ax_gain.grid(alpha=0.25)
    ax_gain.legend(frameon=False)

    fd_vals = np.array([row["fd_order"] for row in rows], dtype=int)
    lead = np.array([row["gains"][0] if row["gains"].size else np.nan for row in rows], dtype=float)
    times = np.array([row["elapsed_seconds"] / 60.0 for row in rows], dtype=float)
    ax_lead.plot(fd_vals, lead, color="#4e79a7", marker="o", linewidth=1.3)
    for fd, gain, tmin in zip(fd_vals, lead, times, strict=True):
        ax_lead.annotate(
            f"{gain:.2f}\n{tmin:.1f} min",
            (fd, gain),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax_lead.set_xlabel("FD order")
    ax_lead.set_ylabel("Leading gain")
    ax_lead.set_title("Leading-gain trend")
    ax_lead.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(RESOLVENT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_perf(stability_rows: list[dict], resolvent_rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex="col")

    for col, (title, rows) in enumerate(
        (("Stability", stability_rows), ("Resolvent", resolvent_rows))
    ):
        fd_vals = np.array([row["fd_order"] for row in rows], dtype=int)
        times = np.array([row["elapsed_seconds"] / 60.0 for row in rows], dtype=float)
        memory = np.array([row["peak_rss_gib"] for row in rows], dtype=float)

        ax_t = axes[0, col]
        ax_m = axes[1, col]

        ax_t.plot(fd_vals, times, marker="o", linewidth=1.5, color="#4e79a7")
        ax_t.set_title(title)
        ax_t.set_ylabel("Wall time [min]")
        ax_t.grid(alpha=0.25)
        for fd, tmin in zip(fd_vals, times, strict=True):
            ax_t.annotate(
                f"{tmin:.2f}",
                (fd, tmin),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

        ax_m.plot(fd_vals, memory, marker="s", linewidth=1.5, color="#e15759")
        ax_m.set_xlabel("FD order")
        ax_m.set_ylabel("Peak RSS [GiB]")
        ax_m.grid(alpha=0.25)
        for fd, mem in zip(fd_vals, memory, strict=True):
            ax_m.annotate(
                f"{mem:.2f}",
                (fd, mem),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

    fig.suptitle(r"Case 1 cost summary, sponge on, $\sigma_{max}=0.5$", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PERF_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    stability_rows = _rows_from_report(STABILITY_REPORT, stability=True)
    resolvent_rows = _rows_from_report(RESOLVENT_REPORT, stability=False)
    if not stability_rows:
        raise SystemExit(f"No completed stability cases found in {STABILITY_REPORT}")
    if not resolvent_rows:
        raise SystemExit(f"No completed resolvent cases found in {RESOLVENT_REPORT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_stability(stability_rows)
    _plot_resolvent(resolvent_rows)
    _plot_perf(stability_rows, resolvent_rows)

    print(f"wrote {STABILITY_PNG}")
    print(f"wrote {RESOLVENT_PNG}")
    print(f"wrote {PERF_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
