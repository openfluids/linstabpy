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


def test_relocation_replaces_the_build_directory_in_petsc_and_slepc_config(tmp_path):
    build = "/tmp/uv-build/petsc/wheel/petsc"
    petsc = tmp_path / "petsc"
    (petsc / "lib/petsc/conf").mkdir(parents=True)
    (petsc / "include").mkdir()
    (petsc / "lib/petsc/conf/petscvariables").write_text(
        f"PETSC_DIR = {build}\nPETSC_PYTHON_EXE = /tmp/build-env/bin/python\nCC_INCLUDES = -I{build}/include\n"
    )
    (petsc / "include/petscconf.h").write_text(f'#define PETSC_DIR "{build}"\n')
    install_stack._relocate_petsc(str(petsc))
    variables = (petsc / "lib/petsc/conf/petscvariables").read_text()
    assert build not in variables
    assert f"-I{petsc}/include" in variables
    assert "PETSC_PYTHON_EXE = python3" in variables
    assert build not in (petsc / "include/petscconf.h").read_text()

    slepc_build = "/tmp/uv-build/slepc/wheel/slepc"
    slepc = tmp_path / "slepc"
    (slepc / "lib/slepc/conf").mkdir(parents=True)
    (slepc / "lib/slepc/conf/slepc_variables").write_text(f"SLEPC_LIB_DIR = {slepc_build}/lib\n")
    (slepc / "lib/slepc/conf/slepcvariables").write_text(f"SLEPC_INSTALLDIR = {slepc_build}\n")
    install_stack._relocate_slepc(str(slepc))
    assert f"SLEPC_INSTALLDIR = {slepc}" in (slepc / "lib/slepc/conf/slepcvariables").read_text()
    assert slepc_build not in (slepc / "lib/slepc/conf/slepc_variables").read_text()
