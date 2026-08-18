#!/usr/bin/env python3
"""Run all Martini example computes, then all presentation plots."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MARTINI_DIR = Path(__file__).resolve().parent

from linstabpy.utils.yaml_io import load_config

from examples.martini.example_0_curvilinear_validation import (
    OUTPUT_NPZ as EX0_NPZ,
    compute as compute_ex0,
    ensure_artifact,
)
from examples.martini.example_0_plots import plot_example0_figures
from examples.martini.example_1_blasius import OUTPUT_NPZ as EX1_NPZ, compute as compute_ex1
from examples.martini.example_1_plots import plot_example1_figures
from examples.martini.example_2_jet import OUTPUT_NPZ as EX2_NPZ, compute as compute_ex2
from examples.martini.example_2_plots import plot_example2_figures
from examples.martini.example_3_streaky_bl import PAPER_CASES, run_case
from examples.martini.example_3_plots import plot_example3_figures
from examples.martini.example_4_cavity import OUTPUT_NPZ as EX4_NPZ, compute as compute_ex4
from examples.martini.example_4_plots import plot_example4_figures
from examples.martini.example_5_paraboloid import OUTPUT_NPZ as EX5_NPZ, compute as compute_ex5
from examples.martini.example_5_plots import plot_example5_figures


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key] for key in raw.files}


def _banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def _run_step(label: str, fn) -> None:
    t0 = time.perf_counter()
    print(f"[run] {label} ...", flush=True)
    fn()
    print(f"[done] {label} ({time.perf_counter() - t0:.1f}s)", flush=True)


def phase_compute() -> None:
    _banner("Phase 1 — compute all Martini examples")

    cfg0 = load_config(MARTINI_DIR / "example_0_curvilinear_validation.yaml")
    _run_step(
        "Example 0",
        lambda: compute_ex0(cfg0),
    )

    cfg1 = load_config(MARTINI_DIR / "example_1_blasius.yaml")
    _run_step("Example 1", lambda: compute_ex1(cfg1))

    cfg2 = load_config(MARTINI_DIR / "example_2_jet.yaml")
    _run_step("Example 2", lambda: compute_ex2(cfg2))

    for stem in PAPER_CASES:
        _run_step(f"Example 3 campaign {stem}", lambda s=stem: run_case(s))

    cfg4 = load_config(MARTINI_DIR / "example_4_cavity.yaml")
    _run_step("Example 4", lambda: compute_ex4(cfg4))

    cfg5 = load_config(MARTINI_DIR / "example_5_paraboloid.yaml")
    _run_step("Example 5", lambda: compute_ex5(cfg5))


def phase_plot() -> None:
    _banner("Phase 2 — plot all Martini examples")

    cfg0 = load_config(MARTINI_DIR / "example_0_curvilinear_validation.yaml")
    data0 = _load_npz_dict(EX0_NPZ)
    ensure_artifact(data0)

    def plot0() -> None:
        paths = plot_example0_figures(data0, example_dir=MARTINI_DIR, cfg=cfg0)
        for path in paths:
            print(f"  Wrote {path}", flush=True)

    _run_step("Example 0 plots", plot0)

    cfg1 = load_config(MARTINI_DIR / "example_1_blasius.yaml")
    data1 = _load_npz_dict(EX1_NPZ)

    def plot1() -> None:
        paths = plot_example1_figures(data1, example_dir=MARTINI_DIR, cfg=cfg1)
        for path in paths:
            print(f"  Wrote {path}", flush=True)

    _run_step("Example 1 plots", plot1)

    cfg2 = load_config(MARTINI_DIR / "example_2_jet.yaml")
    data2 = _load_npz_dict(EX2_NPZ)
    bc_layout = str(np.asarray(data2.get("bc_mask_layout", "kron")).reshape(-1)[0])

    def plot2() -> None:
        paths = plot_example2_figures(
            data2,
            example_dir=MARTINI_DIR,
            bc_mask_layout=bc_layout,
        )
        for path in paths:
            print(f"  Wrote {path}", flush=True)

    _run_step("Example 2 plots", plot2)

    def plot3() -> None:
        paths = plot_example3_figures(MARTINI_DIR)
        for path in paths:
            print(f"  Wrote {path}", flush=True)

    _run_step("Example 3 plots", plot3)

    def plot4() -> None:
        paths = plot_example4_figures(MARTINI_DIR)
        for path in paths:
            print(f"  Wrote {path}", flush=True)

    _run_step("Example 4 plots", plot4)

    def plot5() -> None:
        paths = plot_example5_figures(MARTINI_DIR)
        for path in paths:
            print(f"  Wrote {path}", flush=True)

    _run_step("Example 5 plots", plot5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip compute phase; regenerate presentation PNGs from existing NPZ.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    try:
        if not args.plots_only:
            phase_compute()
        phase_plot()
    except Exception:
        traceback.print_exc()
        return 1
    _banner(f"All Martini examples finished in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
