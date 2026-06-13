# SPDX-License-Identifier: GPL-3.0-only
"""Driver entrypoints for case-style runtime orchestration.

The heavy PETSc/MPI-backed runtime lives in :mod:`linstabpy.driver.run`.  It is
imported lazily by the CLI (see :func:`linstabpy.driver.cli.main`) so CLI-only
exits such as ``--version`` and ``--help`` do not require the optional PETSc/MPI
stack.  Import the runtime explicitly where it is needed::

    from linstabpy.driver.run import run

This package initializer deliberately imports nothing heavy.
"""

__all__: list[str] = []
