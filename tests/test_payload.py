# SPDX-License-Identifier: GPL-3.0-only
"""Runtime payload groups and YAML list parsers."""

from __future__ import annotations

import numpy as np
import pytest

from linstabpy.driver.payload import RUNTIME_NPZ_CONTRACT_KEYS, build_runtime_payload
from linstabpy.utils.yaml_io import parse_complex, parse_complex_list, parse_int_list


def test_build_runtime_payload_merges_groups() -> None:
    payload = build_runtime_payload(
        spectra={"omega": np.array([1.0 + 0.0j]), "beta": 19.2},
        problem={"Re": 1.0e5, "ndof": 10},
        fields={"X": np.zeros((2, 2)), "U": np.ones((2, 2))},
        stability={"stability_eps_nconv": 1},
        dissipation={"numerics_operator_dissipation_enabled": False},
        provenance={"provenance_factor_backend": "mumps"},
    )
    assert payload["beta"] == 19.2
    assert payload["Re"] == 1.0e5
    assert payload["stability_eps_nconv"] == 1
    assert payload["provenance_factor_backend"] == "mumps"
    assert set(payload) >= {
        "omega",
        "beta",
        "Re",
        "ndof",
        "X",
        "U",
        "stability_eps_nconv",
        "numerics_operator_dissipation_enabled",
        "provenance_factor_backend",
    }


def test_runtime_npz_contract_keeps_core_keys() -> None:
    required = {
        "omega",
        "direct_modes",
        "gains",
        "X",
        "Y",
        "U",
        "sponge",
        "stability_all_omega",
        "provenance_factor_backend",
    }
    assert required <= set(RUNTIME_NPZ_CONTRACT_KEYS)


def test_parse_complex_accepts_list_number_and_string() -> None:
    assert parse_complex([0.0, 0.125]) == 0.125j
    assert parse_complex(2.0) == 2.0 + 0.0j
    assert parse_complex("1+2j") == 1.0 + 2.0j


def test_parse_complex_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="length"):
        parse_complex([1.0])
    with pytest.raises(TypeError):
        parse_complex({"real": 1.0})


def test_parse_complex_list_and_int_list() -> None:
    assert parse_complex_list([[0.0, 1.0], 2]) == [1.0j, 2.0 + 0.0j]
    assert parse_int_list([0, 2, 4.0]) == [1, 2, 4]
