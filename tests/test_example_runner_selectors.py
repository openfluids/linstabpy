# SPDX-License-Identifier: GPL-3.0-only
"""Contracts between benchmark.sh and run_all_examples.sh."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_all_examples_list() -> str:
    proc = subprocess.run(
        ["bash", str(ROOT / "run_all_examples.sh"), "--list"],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return proc.stdout


def _benchmark_default_module() -> str:
    text = (ROOT / "benchmark.sh").read_text(encoding="utf-8")
    match = re.search(
        r'EX1_MODULE="\$\{LINSTABPY_BENCHMARK_EX1_MODULE:-([^}]+)\}"',
        text,
    )

    assert match is not None
    return match.group(1)


def _benchmark_example_0_module() -> str:
    text = (ROOT / "benchmark.sh").read_text(encoding="utf-8")
    match = re.search(
        r'run_case "([^"]+)" "example_0_curvilinear_validation"',
        text,
    )

    assert match is not None
    return match.group(1)


def test_benchmark_default_example_is_valid_runner_selector():
    assert _benchmark_default_module() in _run_all_examples_list()


def test_benchmark_example_0_module_is_valid_runner_selector():
    assert _benchmark_example_0_module() in _run_all_examples_list()


def test_benchmark_clears_npz_under_martini():
    text = (ROOT / "benchmark.sh").read_text(encoding="utf-8")
    assert "examples/martini/" in text


def test_runner_accepts_benchmark_default_module_on_mpi_path():
    default_module = _benchmark_default_module()
    true_bin = shutil.which("true")

    assert true_bin is not None

    env = {
        **os.environ,
        "LINSTABPY_PYTHON": true_bin,
        "LINSTABPY_MPIEXEC": true_bin,
    }
    proc = subprocess.run(
        [
            "bash",
            str(ROOT / "run_all_examples.sh"),
            "--mpi",
            "--skip-parity",
            "-n",
            "1",
            "-m",
            default_module,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "unknown example selector" not in proc.stderr
