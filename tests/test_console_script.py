# SPDX-License-Identifier: GPL-3.0-only
"""The installed ``linstabpy`` command must start the driver CLI."""

from importlib.metadata import entry_points

from linstabpy.driver.cli import main


def test_linstabpy_console_script_points_at_the_cli_main() -> None:
    scripts = entry_points(group="console_scripts", name="linstabpy")
    assert len(scripts) == 1
    assert next(iter(scripts)).load() is main
