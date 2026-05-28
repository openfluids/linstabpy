# SPDX-License-Identifier: GPL-3.0-only
"""Unit tests for the Case 1 timing/memory harness."""

from __future__ import annotations

import importlib.util

import pytest

from tests._paths import case_fixture, require_existing

REPO_ROOT = require_existing(case_fixture("case1").parents[2], "repository root")
PERF_MEMORY_PATH = require_existing(
    case_fixture("case1") / "scripts" / "perf_memory.py",
    "Case 1 perf_memory module",
)


def _load_perf_memory_module():
    spec = importlib.util.spec_from_file_location("_case1_perf_memory_for_tests", PERF_MEMORY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


perf_memory = _load_perf_memory_module()
parse_log_metrics = perf_memory._parse_log_metrics
case_row = perf_memory._case_row


def test_harness_resolves_paths_from_repository_root():
    config_dir = case_fixture("case1") / "configs"

    assert perf_memory.ROOT == REPO_ROOT
    assert (
        perf_memory._resolve_executable(
            config_dir,
            "../../../../.venv/bin/python",
            perf_memory.CASE_DIR,
            perf_memory.ROOT,
        )
        == str(REPO_ROOT / ".venv" / "bin" / "python")
    )


def test_parse_log_metrics_extracts_runtime_stage_rss_delta():
    metrics = parse_log_metrics(
        [
            "[case1] stage complete: stage=stability elapsed=8.75s rss_peak_delta_bytes=2048",
            "[case1] stage complete: stage=resolvent elapsed=3.50s rss_peak_delta_bytes=1024",
        ]
    )

    assert metrics["stage_timings_seconds"]["stability"] == pytest.approx(8.75)
    assert metrics["stage_timings_seconds"]["resolvent"] == pytest.approx(3.5)
    assert metrics["stage_rss_delta_bytes"]["stability"] == 2048
    assert metrics["stage_rss_delta_bytes"]["resolvent"] == 1024


def test_case_row_surfaces_runtime_stage_rss_delta():
    row = case_row(
        {
            "case_id": "baseline",
            "status": "ok",
            "metrics": {},
            "parsed": {
                "stage_timings_seconds": {"stability": 2.0, "resolvent_postcheck": 0.25},
                "stage_rss_delta_bytes": {
                    "stability": 4096,
                    "resolvent_postcheck": 512,
                },
            },
            "output": {},
            "effective_case": {},
            "resolved_config": {},
        }
    )

    assert row["stability_rss_delta_bytes"] == 4096
    assert row["resolvent_postcheck_seconds"] == pytest.approx(0.25)
    assert row["resolvent_postcheck_rss_delta_bytes"] == 512
