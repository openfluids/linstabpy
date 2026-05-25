#!/usr/bin/env python3
"""Plot the configured Case 1 sponge over the streamwise base flow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from linstabpy.baseflow import case1_sponge_from_config
from linstabpy.utils.jsonc import load_jsonc

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
DEFAULT_CASES = CASE_DIR / "configs" / "perf_memory_cases.jsonc"
OUTPUT_PNG = CASE_DIR / "output" / "case1_sponge_field.png"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _current_sponge_cfg() -> tuple[dict, str]:
    cfg = load_jsonc(DEFAULT_CASES)
    default_sponge = (
        cfg.get("defaults", {})
        .get("case_defaults", {})
        .get("config_overrides", {})
        .get("physics", {})
        .get("sponge", {})
    )
    if not isinstance(default_sponge, dict):
        default_sponge = {}

    for case in cfg.get("cases", []):
        if not case.get("enabled", False):
            continue
        override = case.get("config_overrides", {}).get("physics", {}).get("sponge", {})
        if not isinstance(override, dict):
            override = {}
        sponge_cfg = _deep_merge(default_sponge, override)
        if bool(sponge_cfg.get("enabled", False)):
            return sponge_cfg, str(case.get("case_id", "unknown"))

    if bool(default_sponge.get("enabled", False)):
        return dict(default_sponge), "defaults"
    raise SystemExit("No enabled sponge-on Case 1 branch found in perf_memory_cases.jsonc.")


def main() -> int:
    X = np.load(CASE_DIR / "data" / "baseflow" / "x.npy").T
    Y = np.load(CASE_DIR / "data" / "baseflow" / "y.npy").T
    U = np.load(CASE_DIR / "data" / "baseflow" / "u.npy").T
    sponge_cfg, source_case = _current_sponge_cfg()
    sponge = case1_sponge_from_config(X.shape, sponge_cfg, X=X, Y=Y)

    sponge_cmap = LinearSegmentedColormap.from_list(
        "white_orange_red",
        ["#ffffff", "#fee8c8", "#fdbb84", "#e34a33", "#b30000"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    bg = ax.pcolormesh(X, Y, U, shading="auto", cmap="cividis")

    sponge_masked = np.ma.masked_where(sponge <= 1e-12, sponge)
    vmax = float(np.max(sponge)) if np.max(sponge) > 0.0 else 1.0
    overlay = ax.pcolormesh(
        X,
        Y,
        sponge_masked,
        shading="auto",
        cmap=sponge_cmap,
        vmin=0.0,
        vmax=vmax,
        alpha=0.78,
    )
    if np.max(sponge) > 0.0:
        ax.contour(
            X,
            Y,
            sponge,
            levels=[max(0.05 * float(np.max(sponge)), 1e-9)],
            colors="black",
            linewidths=0.9,
        )

    # Highlight the separation / recirculation boundary already used in legacy plots.
    if np.min(U) <= 0.0 <= np.max(U):
        ax.contour(X, Y, U, levels=[0.0], colors="#2b8cbe", linewidths=1.0, alpha=0.9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.12)
    ax.set_title(
        "Case 1 baseflow with sponge overlay\n"
        f"background: U, sponge preset: {sponge_cfg.get('preset', sponge_cfg.get('type', 'custom'))}, "
        f"max={float(np.max(sponge)):.2f}"
    )

    cbar_u = fig.colorbar(bg, ax=ax, shrink=0.92, pad=0.015)
    cbar_u.set_label("Baseflow U")
    cbar_s = fig.colorbar(overlay, ax=ax, shrink=0.92, pad=0.09)
    cbar_s.set_label("Sponge strength")

    ax.text(
        0.015,
        0.02,
        f"source config: {source_case}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.5},
    )

    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
