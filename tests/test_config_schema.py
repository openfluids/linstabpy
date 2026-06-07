# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from pathlib import Path

import pytest

from linstabpy.config import ConfigValidationError, validate_config
from linstabpy.utils.jsonc import load_jsonc


def test_valid_config_passes() -> None:
    cfg = {"numerics": {"fd_order": 6}, "Re": 1e5, "n_modes": 10}

    assert validate_config(cfg) is None


def test_aggregates_all_errors_in_one_message() -> None:
    cfg = {"numerics": {"fd_order": 5}, "Rey": 1.0, "n_modes": "ten"}

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(cfg)

    msg = str(exc_info.value)
    assert "fd_order" in msg
    assert "Rey" in msg
    assert "n_modes" in msg


@pytest.mark.parametrize("fd_order", [1, 3, 5, 7])
def test_fd_order_enum_rejects_odd(fd_order: int) -> None:
    with pytest.raises(ConfigValidationError):
        validate_config({"fd_order": fd_order})


@pytest.mark.parametrize("fd_order", [2, 4, 6, 8])
def test_fd_order_enum_accepts_valid(fd_order: int) -> None:
    assert validate_config({"fd_order": fd_order}) is None


def test_fd_order_checked_recursively() -> None:
    cfg = {
        "defaults": {
            "case_defaults": {
                "config_overrides": {
                    "numerics": {
                        "fd_order": 5,
                    },
                },
            },
        },
    }

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(cfg)

    assert "defaults.case_defaults.config_overrides.numerics.fd_order" in str(exc_info.value)


def test_unknown_top_level_key_rejected_with_name() -> None:
    cfg = {"Rey": 1.0, "foo_bar": 2.0}

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(cfg)

    msg = str(exc_info.value)
    assert "Rey" in msg
    assert "foo_bar" in msg


def test_bool_not_accepted_for_int_field() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({"n_modes": True})

    assert "expected int, got bool" in str(exc_info.value)


def test_no_false_positives_on_shipped_configs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_paths = [
        path
        for path in repo_root.rglob("*.jsonc")
        if ".venv" not in path.relative_to(repo_root).parts
    ]

    for path in config_paths:
        validate_config(load_jsonc(path), source=str(path))
