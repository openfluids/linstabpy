# SPDX-License-Identifier: GPL-3.0-only
"""Unit tests for ``_load_jsonc_with_includes`` in examples/case_studies/case2_lui/perf_memory.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tests._paths import case_fixture, require_existing

PERF_MEMORY_PATH = require_existing(
    case_fixture("case2") / "perf_memory.py", "Case 2 perf_memory module"
)


def _load_perf_memory_module():
    spec = importlib.util.spec_from_file_location("_perf_memory_for_tests", PERF_MEMORY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


perf_memory = _load_perf_memory_module()
load_with_includes = perf_memory._load_jsonc_with_includes
read_assembly_metrics = perf_memory._read_assembly_metrics
case_row = perf_memory._case_row


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_no_include_returns_raw_dict(tmp_path):
    f = tmp_path / "no_include.jsonc"
    _write(f, '{"a": 1, "nested": {"x": 2}}')
    assert load_with_includes(f) == {"a": 1, "nested": {"x": 2}}


def test_single_level_include_deep_merges_with_current_winning(tmp_path):
    _write(tmp_path / "base.jsonc", '{"defaults": {"a": 1, "b": 2}, "cases": [1]}')
    _write(
        tmp_path / "over.jsonc",
        '{"$include": "base.jsonc", "defaults": {"b": 20, "c": 3}}',
    )
    merged = load_with_includes(tmp_path / "over.jsonc")
    assert merged == {"defaults": {"a": 1, "b": 20, "c": 3}, "cases": [1]}
    assert "$include" not in merged


def test_chained_includes_are_supported(tmp_path):
    _write(tmp_path / "c.jsonc", '{"layer": "c", "from_c": 1}')
    _write(tmp_path / "b.jsonc", '{"$include": "c.jsonc", "layer": "b", "from_b": 1}')
    _write(tmp_path / "a.jsonc", '{"$include": "b.jsonc", "layer": "a", "from_a": 1}')
    merged = load_with_includes(tmp_path / "a.jsonc")
    assert merged == {"layer": "a", "from_a": 1, "from_b": 1, "from_c": 1}


def test_cycle_raises_recursion_error(tmp_path):
    _write(tmp_path / "a.jsonc", '{"$include": "b.jsonc"}')
    _write(tmp_path / "b.jsonc", '{"$include": "a.jsonc"}')
    with pytest.raises(RecursionError):
        load_with_includes(tmp_path / "a.jsonc")


def test_missing_include_target_raises_file_not_found(tmp_path):
    _write(tmp_path / "dangling.jsonc", '{"$include": "does_not_exist.jsonc"}')
    with pytest.raises(FileNotFoundError):
        load_with_includes(tmp_path / "dangling.jsonc")


def test_read_assembly_metrics_collects_rank_files(tmp_path):
    template = tmp_path / "case.assembly_metrics_rank{rank}.json"
    (tmp_path / "case.assembly_metrics_rank0.json").write_text(
        json.dumps({"rank": 0, "stages": {"prepare_coefficients": {"seconds": 0.1}}}),
        encoding="utf-8",
    )
    (tmp_path / "case.assembly_metrics_rank1.json").write_text(
        json.dumps({"rank": 1, "stages": {"prepare_coefficients": {"seconds": 0.2}}}),
        encoding="utf-8",
    )

    reports = read_assembly_metrics(template)

    assert [report["rank"] for report in reports] == [0, 1]
    assert reports[0]["path"].endswith("case.assembly_metrics_rank0.json")


def test_read_assembly_metrics_reports_non_object_json(tmp_path):
    template = tmp_path / "case.assembly_metrics_rank{rank}.json"
    (tmp_path / "case.assembly_metrics_rank0.json").write_text("null", encoding="utf-8")

    reports = read_assembly_metrics(template)

    assert reports == [
        {
            "path": str(tmp_path / "case.assembly_metrics_rank0.json"),
            "status": "error",
            "error": "expected JSON object, got NoneType",
        }
    ]


def test_read_assembly_metrics_accepts_padded_rank_template(tmp_path):
    template = tmp_path / "case.assembly_metrics_rank{rank:04d}.json"
    (tmp_path / "case.assembly_metrics_rank0000.json").write_text(
        json.dumps({"rank": 0, "stages": {}}),
        encoding="utf-8",
    )

    reports = read_assembly_metrics(template)

    assert [report["rank"] for report in reports] == [0]


def test_case_row_surfaces_rank0_hot_stage_metrics():
    row = case_row(
        {
            "case_id": "baseline",
            "status": "ok",
            "metrics": {"elapsed_seconds": 1.0, "peak_rss_bytes": 1024},
            "parsed": {"stage_timings_seconds": {"assemble_operator_bc": 0.5}},
            "output": {},
            "effective_case": {},
            "resolved_config": {},
            "assembly_metrics": [
                {
                    "rank": 0,
                    "stages": {
                        "prepare_coefficients": {
                            "seconds": 0.11,
                            "rss_peak_delta_bytes": 2048,
                        },
                        "assemble_raw_blocks": {
                            "seconds": 0.22,
                            "rss_peak_delta_bytes": 4096,
                        },
                        "apply_bc": {"rss_peak_delta_bytes": 512},
                    },
                }
            ],
        }
    )

    assert row["prepare_coefficients_seconds"] == 0.11
    assert row["prepare_coefficients_rss_delta_bytes"] == 2048
    assert row["assemble_raw_blocks_seconds"] == 0.22
    assert row["assemble_raw_blocks_rss_delta_bytes"] == 4096
    assert row["apply_bc_rss_delta_bytes"] == 512


def test_case_row_ignores_bad_rank_values_when_finding_rank0():
    row = case_row(
        {
            "case_id": "baseline",
            "status": "ok",
            "metrics": {},
            "parsed": {},
            "output": {},
            "effective_case": {},
            "resolved_config": {},
            "assembly_metrics": [
                {"rank": None, "stages": {"prepare_coefficients": {"seconds": 99.0}}},
                {"rank": 0, "stages": {"prepare_coefficients": {"seconds": 0.25}}},
            ],
        }
    )

    assert row["prepare_coefficients_seconds"] == 0.25


def test_case_row_does_not_substitute_nonzero_rank_metrics_for_rank0():
    row = case_row(
        {
            "case_id": "baseline",
            "status": "ok",
            "metrics": {},
            "parsed": {},
            "output": {},
            "effective_case": {},
            "resolved_config": {},
            "assembly_metrics": [
                {"rank": 1, "stages": {"prepare_coefficients": {"seconds": 99.0}}},
            ],
        }
    )

    assert row["prepare_coefficients_seconds"] is None
