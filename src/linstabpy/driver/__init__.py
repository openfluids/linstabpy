# SPDX-License-Identifier: GPL-3.0-only
"""Driver entrypoints for case-style runtime orchestration.

The heavy PETSc/MPI-backed runtime lives in :mod:`linstabpy.driver.run`.
The CLI (see :func:`linstabpy.driver.cli.main`) imports it lazily.
Then ``--version`` and ``--help`` do not need the optional PETSc/MPI stack.
Import the runtime explicitly on the run path::

    from linstabpy.driver.run import run

PETSc and MPI stay out of this initializer.
"""

__all__: list[str] = []
