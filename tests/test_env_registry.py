# SPDX-License-Identifier: GPL-3.0-only
"""Registry tests: typo-guard behaviour plus the coverage drift check."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

from linstabpy.config.env_registry import (
    KNOWN_INEFFECTIVE_VARS,
    KNOWN_LINSTABPY_VARS,
    find_unknown_env_vars,
    format_unknown_env_warning,
    suggest_env_var,
    warn_unknown_env_vars,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
_VAR_RE = re.compile(r"(?<![A-Z0-9_])LINSTABPY_[A-Z0-9]+(?:_[A-Z0-9]+)*")
# Bare family prefixes select a YAML subsection in typed.py / factorize.py;
# they are string constants, not env vars (asserted ineffective below).
_PREFIX_CONSTANTS = {"LINSTABPY_EIGEN", "LINSTABPY_RESOLVENT", "LINSTABPY_LINSOLVE"}
# The registry defines itself; its ineffective-name block would self-flag.
_REGISTRY_SELF = REPO_ROOT / "src/linstabpy/config/env_registry.py"


def _iter_scanned_files() -> list[Path]:
    files: list[Path] = [
        REPO_ROOT / "install.sh",
        REPO_ROOT / "benchmark.sh",
        REPO_ROOT / "run_all_examples.sh",
    ]
    for root in (REPO_ROOT / "src", REPO_ROOT / "examples", REPO_ROOT / "tests" / "tools"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix in {".pyc", ".npz", ".h5", ".mat"}:
                continue
            if "__pycache__" in path.parts or path == _REGISTRY_SELF:
                continue
            # Run logs are frozen records of past runs, not sources that run now.
            if path.is_relative_to(REPO_ROOT / "examples" / "logs"):
                continue
            files.append(path)
    return files


def test_registry_covers_all_repo_literals() -> None:
    """Every LINSTABPY_* literal in executable sources must be registered.

    This is the drift check: adding a new variable anywhere in src/,
    install.sh, the runner scripts, examples/ or tests/tools/ without
    adding it to KNOWN_LINSTABPY_VARS fails here.
    """
    missing: dict[str, list[str]] = {}
    for path in _iter_scanned_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        names: set[str] = set(_VAR_RE.findall(text))
        for name in names:
            if name in KNOWN_LINSTABPY_VARS or name in _PREFIX_CONSTANTS:
                continue
            missing.setdefault(name, []).append(str(path.relative_to(REPO_ROOT)))
    assert not missing, (
        "unregistered LINSTABPY_* names found; add them to KNOWN_LINSTABPY_VARS "
        f"in src/linstabpy/config/env_registry.py or fix the typo: {missing}"
    )


def test_registry_covers_prefix_expansion() -> None:
    """LinSolveCfg's f"{prefix}_..." suffixes must exist for both families."""
    typed = (REPO_ROOT / "src/linstabpy/config/typed.py").read_text(encoding="utf-8")
    suffixes: set[str] = set(re.findall(r"\{prefix\}_([A-Z0-9_]+)", typed))
    assert suffixes, "expected f-string prefix expansion in typed.py"
    for suffix in sorted(suffixes):
        for family in ("LINSTABPY_LINSOLVE", "LINSTABPY_RESOLVENT"):
            assert f"{family}_{suffix}" in KNOWN_LINSTABPY_VARS, (
                f"{family}_{suffix} expanded from typed.py but missing from registry"
            )


def test_registry_covers_cli_contract() -> None:
    """Every CLI_ENV_VARS name must be registered."""
    from linstabpy.driver.cli import CLI_ENV_VARS

    for name, _default, _purpose in CLI_ENV_VARS:
        assert name in KNOWN_LINSTABPY_VARS


def test_registry_covers_test_gating_vars() -> None:
    """Vars read by test modules (excluded from the literal scan) stay known."""
    gating = {
        "LINSTABPY_MATLAB_PARITY",
        "LINSTABPY_MATLAB_FIXTURES_NPZ",
        "LINSTABPY_SONG32_CSV",
        "LINSTABPY_TEST_REFERENCE_SPECTRUM",
    }
    assert gating <= KNOWN_LINSTABPY_VARS


def test_ineffective_vars_stay_unknown() -> None:
    """YAML-only names and bare prefixes must keep tripping the guard."""
    assert KNOWN_INEFFECTIVE_VARS.isdisjoint(KNOWN_LINSTABPY_VARS)
    env = {name: "1" for name in KNOWN_INEFFECTIVE_VARS}
    assert set(find_unknown_env_vars(env)) == set(KNOWN_INEFFECTIVE_VARS)


def test_find_unknown_env_vars_ignores_other_prefixes() -> None:
    env = {
        "LINSTABPY_CASE_CONFIG": "a.yaml",
        "LINSTABPY_CASE_CONFG": "typo.yaml",
        "PATH": "/bin",
        "LINSTAB2D_STAGE_HEARTBEAT_SECONDS": "5",
    }
    assert find_unknown_env_vars(env) == ["LINSTABPY_CASE_CONFG"]


def test_removed_case_aliases_are_unknown() -> None:
    """The pre-1.0 CASE1 and CASE2 spellings now trip the unknown-variable guard."""
    removed = [
        f"LINSTABPY_CASE{n}_{suffix}"
        for n in (1, 2)
        for suffix in ("CONFIG", "OUTPUT_NPZ", "ROOT_ASSEMBLY")
    ]
    env = {name: "x" for name in removed}
    assert sorted(find_unknown_env_vars(env)) == sorted(removed)
    assert suggest_env_var("LINSTABPY_CASE1_CONFIG") == "LINSTABPY_CASE_CONFIG"


def test_suggest_env_var() -> None:
    assert suggest_env_var("LINSTABPY_CASE_CONFG") == "LINSTABPY_CASE_CONFIG"
    assert suggest_env_var("LINSTABPY_EIGN_KSP_TYPE") == "LINSTABPY_EIGEN_KSP_TYPE"
    assert suggest_env_var("LINSTABPY_ZZZZ_QQQQ") is None


def test_format_unknown_env_warning_single() -> None:
    text = format_unknown_env_warning(["LINSTABPY_CASE_CONFG"])
    assert text.startswith("warning: unknown LINSTABPY_CASE_CONFG")
    assert "did you mean LINSTABPY_CASE_CONFIG?" in text
    assert "no effect" in text


def test_format_unknown_env_warning_ineffective_hint() -> None:
    text = format_unknown_env_warning(["LINSTABPY_EIGEN_NCV"])
    assert "YAML-only" in text
    assert "did you mean" not in text


def test_format_unknown_env_warning_multiple() -> None:
    text = format_unknown_env_warning(["LINSTABPY_CASE_CONFG", "LINSTABPY_EIGEN_NCV"])
    assert "LINSTABPY_CASE_CONFG" in text and "LINSTABPY_EIGEN_NCV" in text


def test_warn_unknown_env_vars_roundtrip() -> None:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        result = warn_unknown_env_vars(["LINSTABPY_CASE_CONFG"])
    assert result == ["LINSTABPY_CASE_CONFG"]
    assert "LINSTABPY_CASE_CONFG" in err.getvalue()


def test_warn_unknown_env_vars_empty_is_silent() -> None:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert warn_unknown_env_vars([]) == []
    assert err.getvalue() == ""
