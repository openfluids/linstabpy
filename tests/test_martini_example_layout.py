# SPDX-License-Identifier: GPL-3.0-only
"""Check Martini example module paths and the compute/plot split."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARTINI = ROOT / "examples" / "linstab2d"

COMPUTE_SCRIPTS = (
    "example_0_curvilinear_validation.py",
    "example_1_blasius.py",
    "example_2_jet.py",
    "example_3_streaky_bl.py",
    "example_4_cavity.py",
    "example_5_paraboloid.py",
)
PLOT_MODULES = (
    "examples.linstab2d.example_0_plots",
    "examples.linstab2d.example_1_plots",
    "examples.linstab2d.example_2_plots",
    "examples.linstab2d.example_3_plots",
    "examples.linstab2d.example_4_plots",
    "examples.linstab2d.example_5_plots",
)
STALE_MODULE = re.compile(r"examples\.example_[0-9]|examples\.example_1_blasius_smoke")
PUBLIC_PATH_FILES = (
    ROOT / "README.md",
    ROOT / "run_all_examples.sh",
    ROOT / "benchmark.sh",
    ROOT / "docs" / "tutorial" / "tutorial.tex",
)


def test_public_docs_use_martini_example_modules() -> None:
    """Check README, runner, benchmark, and tutorial use examples.linstab2d.*."""
    for path in PUBLIC_PATH_FILES:
        hits = STALE_MODULE.findall(path.read_text(encoding="utf-8"))
        assert hits == [], f"{path.relative_to(ROOT)}: {hits}"


def test_example_0_compute_does_not_write_root_artifacts() -> None:
    text = (MARTINI / "example_0_curvilinear_validation.py").read_text(encoding="utf-8")
    assert "artifacts" not in text
    assert "ensure_artifact" not in text


def test_martini_compute_drivers_do_not_import_plot_modules() -> None:
    for name in COMPUTE_SCRIPTS:
        text = (MARTINI / name).read_text(encoding="utf-8")
        assert "_plots import" not in text, name
        assert "plot_example" not in text, name


def test_martini_plot_modules_expose_main() -> None:
    for module_name in PLOT_MODULES:
        module = importlib.import_module(module_name)
        assert callable(getattr(module, "main", None)), module_name
