# SPDX-License-Identifier: GPL-3.0-only
"""Runtime payload groups and YAML list parsers."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from linstabpy.driver.payload import RUNTIME_NPZ_CONTRACT_KEYS, build_runtime_payload
from linstabpy.utils.provenance import _blas_from_link_line
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


def test_blas_from_link_line_names_known_families() -> None:
    assert _blas_from_link_line("-lmkl_intel_lp64 -lmkl_core") == "mkl"
    assert _blas_from_link_line("-lopenblas") == "openblas"
    assert _blas_from_link_line("-lfblas -lflapack") == "reference"
    assert _blas_from_link_line("-framework Accelerate") == "accelerate"
    assert _blas_from_link_line("  ") == "unavailable"
    assert _blas_from_link_line("-lblas") == "-lblas"


def test_collect_provenance_without_petsc_and_without_config(tmp_path, monkeypatch) -> None:
    from linstabpy.utils import provenance
    from linstabpy.utils.provenance import collect_provenance

    monkeypatch.delenv("LINSTABPY_COMPUTE_MODE", raising=False)
    monkeypatch.delenv("LINSTABPY_HOSTNAME", raising=False)
    monkeypatch.setenv("LINSTABPY_HOSTNAME", "  ")

    # Simulate a host without the PETSc stack. Without this the asserts
    # depend on the machine and fail where petsc4py is installed.
    real_version = provenance.metadata.version

    def _version_without_petsc(name: str) -> str:
        if name in {"petsc4py", "slepc4py"}:
            raise provenance.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(provenance.metadata, "version", _version_without_petsc)
    monkeypatch.setitem(sys.modules, "petsc4py", None)
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    missing = collect_provenance(tmp_path / "no_such.yaml")
    assert missing["config_md5"] == ""
    assert missing["hostname"] == "unknown"
    assert missing["petsc4py_version"] == "unavailable"
    assert missing["petsc_arch"] == "unavailable"
    assert missing["mumps_version"] == "unavailable"
    assert missing["mpi_impl"] == "unavailable"
    monkeypatch.setenv("LINSTABPY_COMPUTE_MODE", "gpu")
    assert collect_provenance(None)["compute_mode"] == "gpu"
    monkeypatch.setenv("LINSTABPY_COMPUTE_MODE", "quantum")
    with pytest.raises(ValueError, match="LINSTABPY_COMPUTE_MODE"):
        collect_provenance(None)


def test_git_head_sha_returns_unknown_when_git_fails(monkeypatch) -> None:
    from linstabpy.utils import provenance

    def _boom(*args, **kwargs):
        raise OSError("git missing")

    monkeypatch.setattr(provenance.subprocess, "run", _boom)
    assert provenance._git_head_sha() == "unknown"
