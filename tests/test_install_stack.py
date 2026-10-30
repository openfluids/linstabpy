# SPDX-License-Identifier: GPL-3.0-only
"""``linstabpy install-stack`` plans the pip builds and checks the toolchain."""

import pytest

from linstabpy import install_stack
from linstabpy.driver import cli


def test_dry_run_lists_every_build_step_and_the_complex_options(monkeypatch, capsys):
    monkeypatch.setattr(install_stack, "_missing_tools", lambda: [])
    with pytest.raises(SystemExit) as stop:
        cli.main(["install-stack", "--dry-run"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    for step in ("PETSc", "mpi4py", "petsc4py", "SLEPc", "slepc4py"):
        assert step in out
    assert f"petsc=={install_stack.PETSC_VERSION}" in out
    assert "--with-scalar-type=complex" in " ".join(install_stack.CONFIGURE_OPTIONS)
    assert "--download-mumps" in install_stack.CONFIGURE_OPTIONS


def test_missing_toolchain_names_the_missing_tools_and_the_fix(monkeypatch, capsys):
    monkeypatch.setattr(install_stack.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(install_stack, "_missing_tools", lambda: ["Fortran compiler"])
    assert install_stack.main([]) == 1
    err = capsys.readouterr().err
    assert "Fortran compiler" in err
    assert "apt install" in err


def test_existing_petsc4py_is_left_alone_without_force(monkeypatch, capsys):
    monkeypatch.setattr(install_stack, "_missing_tools", lambda: [])
    monkeypatch.setattr(install_stack.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        install_stack.subprocess, "run", lambda *a, **k: pytest.fail("must not build")
    )
    assert install_stack.main([]) == 0
    assert "already installed" in capsys.readouterr().out
