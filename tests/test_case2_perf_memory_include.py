# SPDX-License-Identifier: GPL-3.0-only
"""Unit tests for ``_load_jsonc_with_includes`` in studies/case2/perf_memory.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests._paths import case_fixture, require_existing


PERF_MEMORY_PATH = require_existing(case_fixture("case2") / "perf_memory.py", "Case 2 perf_memory module")


def _load_perf_memory_module():
    spec = importlib.util.spec_from_file_location("_perf_memory_for_tests", PERF_MEMORY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


perf_memory = _load_perf_memory_module()
load_with_includes = perf_memory._load_jsonc_with_includes


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
