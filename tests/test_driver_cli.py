# SPDX-License-Identifier: GPL-3.0-only
"""CLI entry-point tests."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib import metadata


def test_driver_cli_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "linstabpy.driver.cli", "--version"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == metadata.version("linstabpy")
    assert result.stderr == ""


def test_driver_cli_version_and_help_do_not_import_mpi_stack() -> None:
    code = textwrap.dedent(
        """
        import contextlib
        import importlib.abc
        import io
        import sys
        from importlib import metadata

        class BlockMpiStack(importlib.abc.MetaPathFinder):
            blocked = {"mpi4py", "petsc4py"}

            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in self.blocked:
                    raise ImportError(f"blocked optional MPI dependency: {fullname}")
                return None

        sys.meta_path.insert(0, BlockMpiStack())

        from linstabpy.driver.cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["--version"])
        assert out.getvalue().strip() == metadata.version("linstabpy")
        assert "mpi4py" not in sys.modules
        assert "petsc4py" not in sys.modules

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                main(["--help"])
            except SystemExit as exc:
                assert exc.code == 0
            else:
                raise AssertionError("--help should exit via argparse")
        assert "usage: linstabpy" in out.getvalue()
        assert "mpi4py" not in sys.modules
        assert "petsc4py" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
