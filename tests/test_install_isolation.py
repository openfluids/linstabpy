from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "install.sh"


def run_dry_run(*args: str) -> dict[str, str]:
    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--dry-run", *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    plan: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        assert sep, f"dry-run line is not key=value: {line!r}"
        plan[key] = value
    return plan


def write_stub_venv_python(venv_dir: Path, version: str) -> None:
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "{version}"\n')
    python.chmod(python.stat().st_mode | stat.S_IXUSR)


def test_default_dry_run_targets_cpu_venv() -> None:
    plan = run_dry_run()

    assert plan["build_stack"] == "default"
    assert plan["venv_dir"] == str(ROOT / ".venv")
    assert plan["venv_display"] == ".venv"
    assert plan["petsc_configure_branch"] == "cpu-mumps"


def test_gpu_stack_dry_run_targets_isolated_venv() -> None:
    plan = run_dry_run("--stack", "gpu-strumpack")

    assert plan["build_stack"] == "gpu-strumpack"
    assert plan["venv_dir"] == str(ROOT / ".venv-gpu-strumpack")
    assert plan["venv_dir"] != str(ROOT / ".venv")
    assert plan["protected_venv"] == str(ROOT / ".venv")
    assert plan["petsc_configure_branch"] == "cuda-strumpack-design"


def test_gpu_force_dry_run_removes_only_existing_stack_venv() -> None:
    stack = "force-existing-plan-test"
    venv_dir = ROOT / f".venv-{stack}"
    shutil.rmtree(venv_dir, ignore_errors=True)
    venv_dir.mkdir()
    try:
        plan = run_dry_run("--stack", stack, "--force")

        assert plan["force"] == "1"
        assert plan["planned_remove"] == str(venv_dir)
        assert plan["planned_remove"] != str(ROOT / ".venv")
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)


def test_force_dry_run_does_not_plan_absent_stack_venv_removal() -> None:
    stack = "force-absent-plan-test"
    venv_dir = ROOT / f".venv-{stack}"
    shutil.rmtree(venv_dir, ignore_errors=True)
    try:
        plan = run_dry_run("--stack", stack, "--force")

        assert plan["force"] == "1"
        assert plan["planned_remove"] == "none"
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)


def test_mismatched_python_dry_run_plans_stack_venv_recreate() -> None:
    stack = "mismatch-plan-test"
    venv_dir = ROOT / f".venv-{stack}"
    shutil.rmtree(venv_dir, ignore_errors=True)
    try:
        write_stub_venv_python(venv_dir, "3.12")
        plan = run_dry_run("--stack", stack)

        assert plan["force"] == "0"
        assert plan["planned_remove"] == str(venv_dir)
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)


def test_matching_python_dry_run_keeps_existing_stack_venv() -> None:
    stack = "matching-python-plan-test"
    venv_dir = ROOT / f".venv-{stack}"
    shutil.rmtree(venv_dir, ignore_errors=True)
    try:
        write_stub_venv_python(venv_dir, "3.13")
        plan = run_dry_run("--stack", stack)

        assert plan["force"] == "0"
        assert plan["planned_remove"] == "none"
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)


def test_check_mode_dry_run_never_plans_stack_venv_removal() -> None:
    stack = "check-mode-plan-test"
    venv_dir = ROOT / f".venv-{stack}"
    shutil.rmtree(venv_dir, ignore_errors=True)
    try:
        write_stub_venv_python(venv_dir, "3.12")
        plan = run_dry_run("--stack", stack, "--check", "--force")

        assert plan["force"] == "1"
        assert plan["planned_remove"] == "none"
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)


def test_option_like_stack_name_is_rejected() -> None:
    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--dry-run", "--stack", "--force"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Invalid build stack" in result.stderr + result.stdout


def test_symlinked_stack_venv_is_rejected() -> None:
    link = ROOT / ".venv-symlink-guard-test"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(ROOT / ".venv")
    try:
        result = subprocess.run(
            ["bash", str(INSTALL_SH), "--dry-run", "--stack", "symlink-guard-test"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 1
        assert "symlink" in result.stderr + result.stdout
    finally:
        link.unlink()


def test_diagnose_prints_preflight_and_does_not_touch_venv() -> None:
    venv = ROOT / ".venv"
    existed = venv.exists()
    marker = None
    python = venv / "bin" / "python"
    if existed and python.is_file():
        marker = python.stat().st_mtime
    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--diagnose"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    text = result.stdout + result.stderr
    assert "linstabpy install diagnostic" in text
    assert "Installing Linux system dependencies" not in text
    assert existed == venv.exists()
    if marker is not None:
        assert python.stat().st_mtime == marker


def test_dry_run_states_the_blas_choice() -> None:
    """The plan names the BLAS. Default is auto (MKL if MKLROOT, else OpenBLAS)."""
    assert run_dry_run()["blas"] == "auto"
    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "LINSTABPY_BLAS": "mkl"},
        check=True,
        text=True,
        capture_output=True,
    )
    assert "blas=mkl" in result.stdout.splitlines()


def test_reference_blas_is_opt_in_only() -> None:
    """--download-fblaslapack is the reference Fortran BLAS: 0.5 GFlop/s per core
    measured on MUMPS. It must be reachable only through LINSTABPY_BLAS=reference,
    never as the Linux default."""
    src = INSTALL_SH.read_text(encoding="utf-8")
    assert src.count('"--download-fblaslapack"') == 1
    reference_case = src.index("reference)")
    assert src.index('"--download-fblaslapack"') > reference_case
    assert "linux_append_petsc_linalg_options" in src
    assert '"--with-blaslapack-dir=${MKLROOT}"' in src
    assert '"--download-openblas"' in src
