from __future__ import annotations

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


def test_gpu_force_dry_run_removes_only_stack_venv() -> None:
    plan = run_dry_run("--stack", "gpu-strumpack", "--force")

    assert plan["force"] == "1"
    assert plan["planned_remove"] == str(ROOT / ".venv-gpu-strumpack")
    assert plan["planned_remove"] != str(ROOT / ".venv")


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
