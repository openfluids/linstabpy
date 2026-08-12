# SPDX-License-Identifier: GPL-3.0-only
"""Load case files as YAML. Reject leftover JSONC suffixes."""

from __future__ import annotations

from pathlib import Path

import pytest

from linstabpy.utils.yaml_io import load_config


def test_load_config_reads_yaml_comments(tmp_path: Path) -> None:
    path = tmp_path / "case.yaml"
    path.write_text(
        "# Re based on impingement length\n"
        "Re: 1.0e5\n"
        "penalty: 1e8\n"
        "omega_target: [0.0, 0.125]\n",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg["Re"] == 1.0e5
    assert cfg["penalty"] == 1e8
    assert cfg["omega_target"] == [0.0, 0.125]


def test_load_config_accepts_yml_suffix(tmp_path: Path) -> None:
    path = tmp_path / "case.yml"
    path.write_text("n_modes: 8\n", encoding="utf-8")

    assert load_config(path)["n_modes"] == 8


def test_load_config_rejects_jsonc_suffix(tmp_path: Path) -> None:
    path = tmp_path / "case.jsonc"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="yaml"):
        load_config(path)


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "case.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


