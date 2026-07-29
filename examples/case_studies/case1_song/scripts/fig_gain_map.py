#!/usr/bin/env python3
"""Render a Case 1 Song/Hamada SWBLI resolvent gain map."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parents[1]
DEFAULT_OUTPUT = CASE_DIR / "output" / "gain_map" / "case1_gain_map.png"


def _json_from_npz(data: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    if key not in data:
        return {}
    raw = np.asarray(data[key]).reshape(-1)
    if raw.size == 0:
        return {}
    value = raw[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return {}


def _title_from_metadata(metadata: dict[str, Any]) -> str:
    angle = str(metadata.get("angle_deg", "<TODO: verify>")).strip()
    if not angle or angle.lower() == "nan":
        angle = "<TODO: verify>"
    return f"SWBLI Resolvent Gain Map (theta={angle} deg)"


def render(input_npz: Path, output_png: Path) -> Path:
    with np.load(input_npz, allow_pickle=False) as data:
        f_nondim = np.asarray(data["f_nondim"], dtype=float)
        beta_nondim = np.asarray(data["beta_nondim"], dtype=float)
        gain = np.asarray(data["gain"], dtype=float)
        metadata = _json_from_npz(data, "metadata")

    if gain.shape != (beta_nondim.size, f_nondim.size):
        raise ValueError(
            "gain shape must be (len(beta_nondim), len(f_nondim)); "
            f"got {gain.shape}, expected {(beta_nondim.size, f_nondim.size)}"
        )

    # Plot the TRUE energy gain sigma^2 (gain is the leading singular value sigma).
    # The map spans many decades (near-pole BG-S resonance vs broadband OM-1/LU-S),
    # so a log colorbar is used so all features are visible; no value rescaling.
    from matplotlib.colors import LogNorm

    sigma_sq = np.square(gain)
    positive = sigma_sq[np.isfinite(sigma_sq) & (sigma_sq > 0)]
    vmin = float(positive.min()) if positive.size else 1e-6
    vmax = float(positive.max()) if positive.size else 1.0
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(
        f_nondim,
        beta_nondim,
        sigma_sq,
        shading="auto",
        cmap="viridis",
        norm=LogNorm(vmin=max(vmin, vmax * 1e-6), vmax=vmax),
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"leading energy gain $\sigma^2$ (log scale)")

    ax.set_xscale("log")
    ax.set_xlim(float(np.min(f_nondim)), float(np.max(f_nondim)))
    ax.set_ylim(0.0, 200.0)
    ax.set_xlabel(r"$f X_{sh}/U_\infty$")
    ax.set_ylabel(r"$\beta X_{sh}$")
    ax.set_title(_title_from_metadata(metadata))

    # NOTE: no OM-1 / LU-S markers are drawn. Structure must be read from the data
    # itself, not asserted by hardcoded annotations. Expected-region guides may be
    # added later only as clearly-labeled references, never as measured peaks.

    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return output_png


def _synthetic_gain(f_nondim: np.ndarray, beta_nondim: np.ndarray) -> np.ndarray:
    F, B = np.meshgrid(f_nondim, beta_nondim)
    om_1 = np.exp(
        -((np.log10(F) - np.log10(2.0)) ** 2) / (2.0 * 0.18**2)
        - ((B - 45.0) ** 2) / (2.0 * 14.0**2)
    )
    lu_s = 0.85 * np.exp(
        -((np.log10(F) - np.log10(1.0e-4)) ** 2) / (2.0 * 0.22**2)
        - ((B - 165.0) ** 2) / (2.0 * 18.0**2)
    )
    return om_1 + lu_s + 0.02


def self_test(output_png: Path) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    f_nondim = np.logspace(-4.0, 1.0, 64)
    beta_nondim = np.linspace(0.0, 200.0, 80)
    gain = _synthetic_gain(f_nondim, beta_nondim)
    metadata = {
        "angle_deg": "<TODO: verify>",
        "angle_source": "synthetic self-test",
    }
    with tempfile.TemporaryDirectory(dir=output_png.parent) as tmp:
        npz = Path(tmp) / "synthetic_gain_map.npz"
        np.savez_compressed(
            npz,
            f_nondim=f_nondim,
            beta_nondim=beta_nondim,
            f=2.0 * np.pi * f_nondim,
            beta=beta_nondim,
            gain=gain,
            regime_label=np.zeros_like(gain, dtype="<U5"),
            reason_code=np.zeros_like(gain, dtype=np.int16),
            cost_metrics=np.array(json.dumps({})),
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )
        render(npz, output_png)
    if not output_png.exists() or output_png.stat().st_size <= 0:
        raise AssertionError(f"self-test did not create a non-empty PNG: {output_png}")
    print(f"self-test wrote {output_png}")
    return output_png


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test(args.output_png)
        return 0
    if args.input_npz is None:
        raise SystemExit("--input-npz is required unless --self-test is set")
    out = render(args.input_npz, args.output_png)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
