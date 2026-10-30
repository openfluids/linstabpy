# SPDX-License-Identifier: GPL-3.0-only
"""A missing PETSc stack fails with the way to install it, not a bare ImportError."""

import sys

import pytest

from linstabpy.utils import petsc_init


@pytest.mark.parametrize("hidden", ["petsc4py", "mpi4py"])
def test_missing_stack_names_the_install_route(monkeypatch, hidden):
    monkeypatch.setattr(petsc_init, "_PETSC_INIT_DONE", False)
    monkeypatch.setitem(sys.modules, hidden, None)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", None)
    with pytest.raises(ImportError, match=r"environment\.yml.*install\.sh"):
        petsc_init.init_petsc_from_argv([])
