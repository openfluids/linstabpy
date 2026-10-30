# SPDX-License-Identifier: GPL-3.0-only
"""Command-line entry points for shared case drivers."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import zipfile
from collections.abc import Sequence
from importlib import metadata

import numpy as np

from linstabpy.config.env_registry import find_unknown_env_vars, warn_unknown_env_vars
from linstabpy.utils.yaml_io import load_config


def _package_version() -> str:
    """Return the installed package version, with a source-tree fallback."""
    try:
        return metadata.version("linstabpy")
    except metadata.PackageNotFoundError:
        return "0.0.0+dev"


# Canonical driver env-var contract. Single source of truth: argparse --help
# renders it, tools/generate_wiki_boundaries.py inlines it into
# wiki/boundaries.md, and skills/.../generate_skill_refs.py copies it into the
# agent skill. Each entry: (name, default, purpose).
CLI_ENV_VARS: tuple[tuple[str, str, str], ...] = (
    (
        "LINSTABPY_CASE_CONFIG",
        "none (required)",
        "YAML case file. The CLI stops when it is unset.",
    ),
    (
        "LINSTABPY_CASE_OUTPUT_NPZ",
        "{prefix}_{config_md5}_{git_sha8}.npz in the working directory",
        "Result archive path.",
    ),
    (
        "LINSTABPY_CASE_LOG_PREFIX",
        "linstabpy",
        "Log-line prefix and default output stem.",
    ),
    (
        "LINSTABPY_FORCE_RECOMPUTE",
        "unset (cache check runs)",
        "Set to 1, true, yes, or on to skip the cache check (CLI gate only; library runs always compute).",
    ),
    (
        "LINSTABPY_STAGE_HEARTBEAT_SECONDS",
        "30",
        "Progress heartbeat interval in float seconds.",
    ),
    (
        "LINSTABPY_CASE_ROOT_ASSEMBLY",
        "unset (auto: on for iterative eigen ST solvers under MPI)",
        "Root-only assembly for iterative eigen ST solvers; stability-only.",
    ),
)


def _ignored_cli_words(unknown: Sequence[str]) -> list[str]:
    """Return unknown argv tokens that no downstream consumer will read.

    Mirrors :func:`linstabpy.utils.petsc_init._sanitize_petsc_argv`: single-dash
    tokens (plus one following value) reach the PETSc option database, which
    reads ``sys.argv`` directly at ``init_petsc_from_argv`` time. Everything
    else — bare words and double-dash tokens, which the sanitizer drops — is
    flagged. A bare ``--`` separator is skipped without warning.
    """
    words: list[str] = []
    skip_next = False
    for token in unknown:
        if token == "--":
            skip_next = False
        elif token.startswith("--"):
            skip_next = False
            words.append(token)
        elif token.startswith("-"):
            skip_next = True
        elif skip_next:
            skip_next = False
        else:
            words.append(token)
    return words


def _warn_ignored_cli_words(words: Sequence[str]) -> None:
    """Warn on stderr that bare-word CLI arguments were ignored."""
    print(
        f"warning: ignoring unknown arguments {list(words)}: "
        + "linstabpy takes no case arguments; set LINSTABPY_CASE_CONFIG and "
        + "LINSTABPY_CASE_OUTPUT_NPZ instead (see `linstabpy --help`).",
        file=sys.stderr,
    )


def _env_epilog() -> str:
    """Render the env-var contract for ``--help``."""
    lines = [
        "case configuration comes from environment variables (there is no --config flag):",
    ]
    for name, default, purpose in CLI_ENV_VARS:
        lines.append(f"  {name}: {purpose} Default: {default}.")
    lines.extend(
        [
            "",
            "unknown bare-word and double-dash arguments are ignored with a",
            "warning on stderr; single-dash options (e.g. -ksp_monitor) pass",
            "through to the PETSc option database. Unknown LINSTABPY_*",
            "environment variables likewise warn on stderr and are ignored.",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linstabpy",
        description="Run the shared LinStabPy case driver.",
        epilog=_env_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        "--version",
        action="store_true",
        help="print the LinStabPy package version and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Cache-aware entrypoint used when launched as ``python -m linstabpy.driver.cli``."""
    words = list(sys.argv[1:] if argv is None else argv)
    if words[:1] == ["install-stack"]:
        from linstabpy.install_stack import main as install_stack

        raise SystemExit(install_stack(words[1:]))
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    ignored = _ignored_cli_words(unknown)
    unknown_env = find_unknown_env_vars()
    if args.version:
        if ignored:
            _warn_ignored_cli_words(ignored)
        if unknown_env:
            warn_unknown_env_vars(unknown_env)
        print(_package_version())
        return

    # Imported lazily so CLI-only exits (--version/--help) never initialize
    # PETSc or require the optional MPI stack.
    runtime = importlib.import_module("linstabpy.driver.run")

    comm = runtime.PETSc.COMM_WORLD
    rank = comm.getRank()
    if ignored and rank == 0:
        _warn_ignored_cli_words(ignored)
    if unknown_env and rank == 0:
        warn_unknown_env_vars(unknown_env)
    try:
        _ = runtime._runtime_config_path()
    except runtime.ConfigValidationError as exc:
        raise SystemExit(f"linstabpy: error: {exc}") from None
    output_npz = runtime._output_npz_path()
    need_recompute = False

    force_recompute = os.getenv("LINSTABPY_FORCE_RECOMPUTE", "").strip().lower() in {
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
            except (FileNotFoundError, EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
                print(f"Cached results unavailable or invalid ({exc}), recomputing.")
                need_recompute = True
            else:
                try:
                    # Same path run() would load, including its default.
                    current_config = load_config(runtime._runtime_config_path())
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
