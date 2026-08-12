# SPDX-License-Identifier: GPL-3.0-only
"""Command-line entry points for shared case drivers."""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

import numpy as np

from linstabpy.utils.yaml_io import load_config


def _package_version() -> str:
    """Return the installed package version, with a source-tree fallback."""
    try:
        return metadata.version("linstabpy")
    except metadata.PackageNotFoundError:
        return "0.0.0+dev"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linstabpy",
        description="Run the shared LinStabPy case driver.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the LinStabPy package version and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Cache-aware entrypoint used when launched as ``python -m linstabpy.driver.cli``."""
    parser = _build_parser()
    args, _unknown = parser.parse_known_args(argv)
    if args.version:
        print(_package_version())
        return

    # Imported lazily so CLI-only exits (--version/--help) never initialize
    # PETSc or require the optional MPI stack.
    runtime = importlib.import_module("linstabpy.driver.run")

    comm = runtime.PETSc.COMM_WORLD
    rank = comm.getRank()
    output_npz = runtime._output_npz_path()
    need_recompute = False

    force_recompute = os.getenv("LINSTABPY_FORCE_RECOMPUTE", "").strip() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if rank == 0:
        if force_recompute:
            print("LINSTABPY_FORCE_RECOMPUTE set: skipping cache check.")
            need_recompute = True
        else:
            try:
                from linstabpy.driver.io import load_results

                data = load_results(output_npz)
            except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
                print(f"Cached results unavailable or invalid ({exc}), recomputing.")
                need_recompute = True
            else:
                try:
                    config_path_raw = runtime._first_env(
                        "LINSTABPY_CASE_CONFIG",
                        "LINSTABPY_CASE2_CONFIG",
                        "LINSTABPY_CASE1_CONFIG",
                    )
                    if config_path_raw is not None:
                        current_config = load_config(Path(config_path_raw))
                        current_hash = runtime._config_hash(current_config)[:16]
                        current_sha = runtime._git_head_sha()
                        cached_hash = (
                            data.get(
                                "provenance_config_hash",
                                np.bytes_(b""),
                            )
                            .item()
                            .decode("utf-8", errors="replace")
                        )
                        cached_sha = (
                            data.get(
                                "provenance_linstabpy_git_sha",
                                np.bytes_(b""),
                            )
                            .item()
                            .decode("utf-8", errors="replace")
                        )
                        if cached_hash != current_hash:
                            print(
                                "Config hash mismatch "
                                f"(cached={cached_hash!r}, current={current_hash!r}), "
                                "recomputing."
                            )
                            need_recompute = True
                        elif (
                            cached_sha != current_sha
                            and cached_sha not in {"", "unknown"}
                            and current_sha != "unknown"
                        ):
                            print(
                                "Git SHA mismatch "
                                f"(cached={cached_sha!r}, current={current_sha!r}), "
                                "recomputing."
                            )
                            need_recompute = True
                        else:
                            print(
                                f"Cache valid (hash={cached_hash!r}, sha={cached_sha!r}), "
                                "using cached results."
                            )
                    else:
                        # No config path available to check; trust cache.
                        pass
                except Exception as exc:
                    print(f"Cache validation failed ({exc}), recomputing.")
                    need_recompute = True
                finally:
                    data.close()
    need_recompute = runtime.MPI.COMM_WORLD.bcast(need_recompute, root=0)
    if need_recompute:
        runtime.run()


if __name__ == "__main__":  # pragma: no cover
    main()
