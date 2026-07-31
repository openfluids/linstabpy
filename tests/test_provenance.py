# SPDX-License-Identifier: GPL-3.0-only

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from linstabpy.driver.io import load_results, save_results
from linstabpy.driver.run import _output_npz_path
from linstabpy.utils.provenance import collect_provenance

PROVENANCE_KEYS = {
    "git_sha",
    "config_md5",
    "hostname",
    "compute_mode",
    "mpi_stack",
    "python_version",
    "numpy_version",
    "scipy_version",
    "petsc4py_version",
    "slepc4py_version",
    "petsc_arch",
    "mumps_version",
    "mpi_impl",
}
REPO_ROOT = Path(__file__).resolve().parents[1]
_CONSTRAINT_LINE = re.compile(
    r"^([A-Za-z0-9_.-]+(==|===|!=|~=|>=|<=|>|<).+|-e\s+.+|[A-Za-z0-9_.-]+\s*@\s*.+)$"
)


def _git_rev_parse_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def test_collect_provenance_has_stable_typed_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("LINSTABPY_COMPUTE_MODE", raising=False)
    monkeypatch.delenv("LINSTABPY_MPI_STACK", raising=False)
    config_path = tmp_path / "case.jsonc"
    first_bytes = b'{"alpha": 1}\n'
    config_path.write_bytes(first_bytes)

    first = collect_provenance(config_path)
    second = collect_provenance(config_path)

    assert set(first) == PROVENANCE_KEYS
    assert all(isinstance(value, str) for value in first.values())
    assert all(value == "unavailable" or value for value in first.values())
    assert first["config_md5"] == second["config_md5"]
    assert first["config_md5"] == hashlib.md5(first_bytes).hexdigest()  # noqa: S324
    assert first["git_sha"] == _git_rev_parse_head()
    assert first["hostname"]
    assert first["compute_mode"] == "cpu"
    assert first["mpi_stack"] == "unknown"

    second_bytes = b'{"alpha": 2}\n'
    config_path.write_bytes(second_bytes)
    changed = collect_provenance(config_path)
    assert changed["config_md5"] == hashlib.md5(second_bytes).hexdigest()  # noqa: S324
    assert changed["config_md5"] != first["config_md5"]


def test_collect_provenance_validates_compute_mode(tmp_path, monkeypatch):
    config_path = tmp_path / "case.jsonc"
    config_path.write_text("{}\n")
    monkeypatch.setenv("LINSTABPY_COMPUTE_MODE", "quantum")

    with pytest.raises(ValueError, match="LINSTABPY_COMPUTE_MODE"):
        collect_provenance(config_path)


@pytest.mark.parametrize("mode", ["cpu", "gpu", "mixed"])
def test_collect_provenance_accepts_supported_compute_modes(tmp_path, monkeypatch, mode):
    config_path = tmp_path / "case.jsonc"
    config_path.write_text("{}\n")
    monkeypatch.setenv("LINSTABPY_COMPUTE_MODE", mode)

    assert collect_provenance(config_path)["compute_mode"] == mode


def test_freeze_env_script_emits_pip_constraints_file(tmp_path):
    script = REPO_ROOT / "scripts" / "tools" / "freeze_env.sh"

    subprocess.run([script, "test"], cwd=tmp_path, check=True, capture_output=True, text=True)

    constraints = tmp_path / "constraints-test.txt"
    assert constraints.is_file()
    lines = [line.strip() for line in constraints.read_text().splitlines()]
    entries = [line for line in lines if line and not line.startswith("#")]
    assert entries
    assert all(_CONSTRAINT_LINE.match(line) for line in entries)


def test_save_results_round_trips_provenance_to_npz_and_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("LINSTAB2D_GPUS", "0")
    config_path = tmp_path / "case.jsonc"
    raw_config = b'{"beta": 3}\n'
    config_path.write_bytes(raw_config)
    provenance = collect_provenance(config_path)
    output_npz = tmp_path / "result.npz"
    payload = {"omega": np.array([1.0 + 0.5j]), "gains": np.array([2.0])}

    save_results(
        output_npz=output_npz,
        heartbeat_seconds=1e9,
        payload=payload,
        peak_rss_bytes=42,
        stage_wall={"stability": 0.25},
        provenance=provenance,
    )

    with load_results(output_npz) as data:
        for key, expected in provenance.items():
            assert data[key].item() == expected
        assert data["config_md5"].item() == hashlib.md5(raw_config).hexdigest()  # noqa: S324
        assert data["git_sha"].item() == _git_rev_parse_head()
        assert data["cost_git_sha"].item() == data["git_sha"].item()

    sidecar = json.loads(output_npz.with_suffix(".metrics.json").read_text())
    for key, expected in provenance.items():
        assert sidecar[key] == expected
    assert sidecar["config_md5"] == hashlib.md5(raw_config).hexdigest()  # noqa: S324
    assert sidecar["git_sha"] == _git_rev_parse_head()
    assert sidecar["cost_git_sha"] == sidecar["git_sha"]


def test_default_output_path_embeds_config_md5_and_git_sha8(tmp_path, monkeypatch):
    config_path = tmp_path / "case.jsonc"
    raw_config = b'{"gamma": 4}\n'
    config_path.write_bytes(raw_config)
    provenance = collect_provenance(config_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LINSTABPY_CASE_LOG_PREFIX", "caseX")
    monkeypatch.delenv("LINSTABPY_CASE_OUTPUT_NPZ", raising=False)
    monkeypatch.delenv("LINSTABPY_CASE2_OUTPUT_NPZ", raising=False)
    monkeypatch.delenv("LINSTABPY_CASE1_OUTPUT_NPZ", raising=False)

    output = _output_npz_path(provenance)

    assert output == (
        tmp_path / f"caseX_{provenance['config_md5']}_{provenance['git_sha'][:8]}.npz"
    )


def test_explicit_output_path_is_honored_unchanged(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit-name.npz"
    monkeypatch.setenv("LINSTABPY_CASE_OUTPUT_NPZ", str(explicit))

    assert _output_npz_path({"config_md5": "ignored", "git_sha": "ignored"}) == explicit.resolve()


def test_provenance_carries_no_machine_identity_by_default(monkeypatch) -> None:
    """No provenance field may leak the host name unless explicitly asked for.

    ``cost_hardware_tag`` has enforced this policy for a long time, guarded by
    ``test_cost_metrics.py::test_hardware_tag_default_is_unknown_not_hostname``.
    ``hostname`` sat in the very same artifact calling ``socket.gethostname()``
    unconditionally, so every sidecar shipped the fully-qualified host name
    beside the field that was carefully withholding it -- one was found in a
    saved gain-map sidecar (``map_s2/pt_b9_f0.metrics.json``).

    The guard was real; it just had a sibling nobody had checked. This test is
    that sibling's guard, and it sweeps EVERY field rather than naming one, so
    the next field added cannot walk around the policy the same way.
    """
    import socket

    monkeypatch.delenv("LINSTABPY_HOSTNAME", raising=False)

    provenance = collect_provenance(None)

    real_host = socket.gethostname()
    short_host = real_host.split(".")[0]

    assert provenance["hostname"] == "unknown"
    for key, value in provenance.items():
        assert real_host not in value, f"provenance[{key!r}] leaks the host name"
        # Guard the short form too: a bare "c9"-style name would otherwise slip
        # through a check written only against the FQDN. Skip absurdly short
        # host names, which would match everything and make the test useless.
        if len(short_host) >= 4:
            assert short_host not in value, f"provenance[{key!r}] leaks the short host name"


def test_provenance_hostname_is_recorded_when_explicitly_requested(monkeypatch) -> None:
    """The opt-in still works: private runs can label which machine ran them.

    Withholding the host name by default must not remove the ability to tell
    c8 from c9 in a cost study -- it just makes recording it a deliberate act.
    """
    monkeypatch.setenv("LINSTABPY_HOSTNAME", "workstation-A")
    assert collect_provenance(None)["hostname"] == "workstation-A"

    # Blank or whitespace-only must fall back, never emit an empty field that
    # would read as "no machine" while actually meaning "misconfigured".
    monkeypatch.setenv("LINSTABPY_HOSTNAME", "   ")
    assert collect_provenance(None)["hostname"] == "unknown"
