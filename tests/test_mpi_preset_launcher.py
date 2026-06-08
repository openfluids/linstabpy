# SPDX-License-Identifier: GPL-3.0-only

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples/case_studies/case2_lui/run_case_once.sh"
CONFIG = REPO_ROOT / "examples/case_studies/case2_lui/configs/parity_config.jsonc"
C9_RANKFILE = REPO_ROOT / "examples/case_studies/case2_lui/c9.rankfile"


def run_dryrun(
    tmp_path: Path, preset=None, rankfile=None, procs=2, extra_env=None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LINSTABPY_MPI_DRYRUN"] = "1"
    env["LINSTABPY_MPI_PROCS"] = str(procs)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    if preset is not None:
        env["LINSTABPY_MPI_PRESET"] = preset
    if rankfile is not None:
        env["LINSTABPY_MPI_RANKFILE"] = rankfile
    if extra_env:
        env.update(extra_env)

    bash_exe = shutil.which("bash") or "bash"
    return subprocess.run(
        [bash_exe, str(SCRIPT), str(CONFIG), str(tmp_path / "dryrun.npz")],
        env=env,
        capture_output=True,
        text=True,
    )


def test_c9_preset_resolves_launcher(tmp_path):
    result = run_dryrun(tmp_path, preset="c9", procs=2)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "LAUNCHER:" in result.stdout
    assert "mpiexec" in result.stdout
    assert "-n 2" in result.stdout


def test_rankfile_override_precedence(tmp_path):
    result = run_dryrun(tmp_path, preset="socket0", rankfile=str(C9_RANKFILE), procs=2)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "LAUNCHER:" in result.stdout
    assert str(C9_RANKFILE) in result.stdout
    assert "socket0.rankfile" not in result.stdout


def test_malformed_rankfile_errors(tmp_path):
    result = run_dryrun(tmp_path, preset="none", rankfile="/nonexistent/nope.rankfile", procs=2)
    assert result.returncode != 0
    assert "not found" in result.stderr


def test_unknown_preset_errors(tmp_path):
    result = run_dryrun(tmp_path, preset="bogus", procs=2)
    assert result.returncode != 0
    assert "c9" in result.stderr, "c9 must be listed in valid preset values"


def test_real_2rank_mpiexec_smoke():
    mpiexec_path = shutil.which("mpiexec")
    if mpiexec_path is None:
        import pytest

        pytest.skip("mpiexec unavailable")

    result = subprocess.run(
        [mpiexec_path, "-n", "2", "echo", "ok"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "ok" in result.stdout
