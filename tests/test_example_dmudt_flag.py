# SPDX-License-Identifier: GPL-3.0-only
"""The LinStab2D examples read ``matlab_parity_dmudt`` as a boolean-like value."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import NoReturn

import pytest

from linstabpy.operators import assembly
from linstabpy.utils.yaml_io import load_config

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "linstab2d"


class StopComputeError(Exception):
    """The stub raises this to end ``compute`` after it records the flag."""


@pytest.mark.parametrize("name", ["example_5_paraboloid", "example_2_jet"])
@pytest.mark.parametrize(("text", "expected"), [("false", False), ("true", True)])
def test_example_reads_dmudt_flag_as_boolean(
    monkeypatch: pytest.MonkeyPatch, name: str, text: str, *, expected: bool
) -> None:
    seen: dict[str, object] = {}

    def capture(*_args: object, **kwargs: object) -> NoReturn:
        seen["flag"] = kwargs["matlab_parity_dmudt"]
        raise StopComputeError

    monkeypatch.setattr(assembly, "build_lnse_system_result", capture)
    cfg = load_config(EXAMPLES / f"{name}.yaml")
    cfg["matlab_parity_dmudt"] = text
    module = importlib.import_module(f"examples.linstab2d.{name}")
    with pytest.raises(StopComputeError):
        module.compute(cfg)
    assert seen["flag"] is expected
