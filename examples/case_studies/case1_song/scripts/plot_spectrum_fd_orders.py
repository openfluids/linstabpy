#!/usr/bin/env python3
"""Plot Case 1 spectra for the native-sponge study."""

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
DEFAULT_REPORT = CASE_DIR / "output" / "case1_perf_memory_latest.json"
MATLAB_SPECTRUM = CASE_DIR / "reference" / "matlab" / "MATLAB_exports" / "spectrum.mat"
OUTPUT_PNG = CASE_DIR / "output" / "case1_spectrum_fd_orders.png"
ALIAS_PNG = CASE_DIR / "output" / "case1_spectrum_progress.png"

FD_COLORS = {4: "#f28e2b", 6: "#e15759"}


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _diag_or_flat(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return np.diag(arr).astype(complex)
    return arr.reshape(-1).astype(complex)


def _stability_rows(report: dict) -> list[dict]:
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
                "elapsed_seconds": float(case.get("metrics", {}).get("elapsed_seconds") or 0.0),
                "residual_max": case.get("output", {}).get("stability_residual_max"),
            }
        )
    rows.sort(key=lambda row: (row["fd_order"], row["sponge_enabled"]))
    return rows


def _row_label(row: dict) -> str:
    elapsed_min = row["elapsed_seconds"] / 60.0
    residual_max = row.get("residual_max")
    residual_text = f", max res {residual_max:.1e}" if residual_max is not None else ""
    return f"fd{row['fd_order']} ({elapsed_min:.1f} m{residual_text})"


def main() -> int:
    report = _load_report(DEFAULT_REPORT)
    rows = _stability_rows(report)
    if not rows:
        raise SystemExit("No completed Case 1 stability runs in perf report.")

    matlab = _diag_or_flat(np.asarray(loadmat(MATLAB_SPECTRUM)["omega"]))
    panels: dict[tuple[int, bool], dict] = {(row["fd_order"], row["sponge_enabled"]): row for row in rows}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    panel_order = [
        (4, False),
        (4, True),
        (6, False),
        (6, True),
    ]
    for ax, key in zip(axes.ravel(), panel_order, strict=True):
        fd_order, sponge_enabled = key
        row = panels.get(key)
        if row is None:
            ax.text(0.5, 0.5, "missing run", ha="center", va="center", transform=ax.transAxes)
            continue
        if not sponge_enabled:
            ax.scatter(
                np.real(matlab),
                np.imag(matlab),
                s=22,
                marker="x",
                linewidths=0.9,
                color="0.45",
                alpha=0.55,
                label="MATLAB fd4 reference",
            )
        data = np.load(row["artifact_npz"], allow_pickle=False)
        omega = np.asarray(data["omega"], dtype=complex).reshape(-1)
        ax.scatter(
            np.real(omega),
            np.imag(omega),
            s=24,
            marker="o",
            facecolors="none",
            edgecolors=FD_COLORS.get(fd_order, "C0"),
            linewidths=1.2,
            alpha=0.92,
            label=_row_label(row),
        )
        ax.axhline(0.0, color="0.65", linewidth=0.8)
        ax.grid(alpha=0.22)
        ax.set_title(f"fd{fd_order}, sponge {'on' if sponge_enabled else 'off'}")
        ax.legend(frameon=False, fontsize=8)

    for ax in axes[:, 0]:
        ax.set_ylabel("Im(omega)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Re(omega)")

    fig.suptitle("Case 1 spectra for the native sponge study", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(ALIAS_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_PNG}")
    print(f"wrote {ALIAS_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
