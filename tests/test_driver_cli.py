# SPDX-License-Identifier: GPL-3.0-only
"""CLI entry-point tests."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import textwrap
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linstabpy.driver.cli import main


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


def test_package_version_falls_back_when_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import metadata as md

    from linstabpy.driver import cli as cli_mod

    def _boom(_name: str) -> str:
        raise md.PackageNotFoundError("linstabpy")

    monkeypatch.setattr(cli_mod.metadata, "version", _boom)
    assert cli_mod._package_version() == "0.0.0+dev"


def test_driver_cli_version_in_process_does_not_import_mpi_stack() -> None:
    before = set(sys.modules)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main(["--version"])
    assert out.getvalue().strip() == metadata.version("linstabpy")
    added = set(sys.modules) - before
    assert "mpi4py" not in added
    assert "petsc4py" not in added


def test_driver_cli_help_in_process_exits_zero() -> None:
    before = set(sys.modules)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "usage: linstabpy" in out.getvalue()
    added = set(sys.modules) - before
    assert "mpi4py" not in added
    assert "petsc4py" not in added


class _FakeComm:
    def __init__(self, rank: int = 0) -> None:
        self._rank = rank

    def getRank(self) -> int:
        return self._rank

    def bcast(self, value, root: int = 0):
        del root
        return value


class _FakeRuntime:
    def __init__(
        self,
        npz: Path,
        *,
        cfg_hash: str = "a" * 16,
        git_sha: str = "deadbeef",
        config_path: str | None = None,
        rank: int = 0,
    ) -> None:
        self.PETSc = SimpleNamespace(COMM_WORLD=_FakeComm(rank))
        self.MPI = SimpleNamespace(COMM_WORLD=_FakeComm(rank))
        self._npz = npz
        self._cfg_hash = cfg_hash
        self._git_sha = git_sha
        self._config_path = config_path
        self.run_calls = 0

    def _output_npz_path(self) -> Path:
        return self._npz

    def _first_env(self, *names: str) -> str | None:
        if self._config_path is None:
            return None
        for name in names:
            if os.getenv(name):
                return self._config_path
        return self._config_path

    def _config_hash(self, config) -> str:
        del config
        return self._cfg_hash

    def _git_head_sha(self) -> str:
        return self._git_sha

    def run(self) -> None:
        self.run_calls += 1


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, runtime: _FakeRuntime) -> None:
    import importlib

    import linstabpy.driver.cli as cli_mod

    real_import_module = importlib.import_module

    def _import_module(name: str, package: str | None = None):
        if name == "linstabpy.driver.run":
            return runtime
        return real_import_module(name, package)

    monkeypatch.setattr(cli_mod.importlib, "import_module", _import_module)


def _write_cache(path: Path, *, cfg_hash: str, git_sha: str) -> None:
    np.savez(
        path,
        provenance_config_hash=np.bytes_(cfg_hash.encode("utf-8")),
        provenance_linstabpy_git_sha=np.bytes_(git_sha.encode("utf-8")),
    )


def test_driver_cli_force_recompute_skips_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz = tmp_path / "out.npz"
    _write_cache(npz, cfg_hash="a" * 16, git_sha="deadbeef")
    runtime = _FakeRuntime(npz)
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setenv("LINSTABPY_FORCE_RECOMPUTE", "1")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main([])
    assert runtime.run_calls == 1
    assert "LINSTABPY_FORCE_RECOMPUTE" in out.getvalue()


def test_driver_cli_missing_cache_recomputes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _FakeRuntime(tmp_path / "missing.npz")
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.delenv("LINSTABPY_FORCE_RECOMPUTE", raising=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main([])
    assert runtime.run_calls == 1
    assert "Cached results unavailable" in out.getvalue()


def test_driver_cli_valid_cache_skips_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    npz = tmp_path / "out.npz"
    cfg_hash = "b" * 16
    git_sha = "cafef00d"
    _write_cache(npz, cfg_hash=cfg_hash, git_sha=git_sha)
    cfg = tmp_path / "case.yaml"
    cfg.write_text("solver: {}\n")
    runtime = _FakeRuntime(npz, cfg_hash=cfg_hash, git_sha=git_sha, config_path=str(cfg))
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.delenv("LINSTABPY_FORCE_RECOMPUTE", raising=False)
    monkeypatch.setenv("LINSTABPY_CASE_CONFIG", str(cfg))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main([])
    assert runtime.run_calls == 0
    assert "Cache valid" in out.getvalue()


def test_driver_cli_hash_mismatch_recomputes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz = tmp_path / "out.npz"
    _write_cache(npz, cfg_hash="oldhasholdhashxx", git_sha="deadbeef")
    cfg = tmp_path / "case.yaml"
    cfg.write_text("solver: {}\n")
    runtime = _FakeRuntime(
        npz, cfg_hash="newhashnewhashxx", git_sha="deadbeef", config_path=str(cfg)
    )
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.delenv("LINSTABPY_FORCE_RECOMPUTE", raising=False)
    monkeypatch.setenv("LINSTABPY_CASE_CONFIG", str(cfg))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main([])
    assert runtime.run_calls == 1
    assert "Config hash mismatch" in out.getvalue()


def test_driver_cli_git_sha_mismatch_recomputes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz = tmp_path / "out.npz"
    cfg_hash = "d" * 16
    _write_cache(npz, cfg_hash=cfg_hash, git_sha="oldsha01")
    cfg = tmp_path / "case.yaml"
    cfg.write_text("solver: {}\n")
    runtime = _FakeRuntime(npz, cfg_hash=cfg_hash, git_sha="newsha02", config_path=str(cfg))
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.delenv("LINSTABPY_FORCE_RECOMPUTE", raising=False)
    monkeypatch.setenv("LINSTABPY_CASE_CONFIG", str(cfg))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main([])
    assert runtime.run_calls == 1
    assert "Git SHA mismatch" in out.getvalue()


def test_driver_cli_trusts_cache_without_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz = tmp_path / "out.npz"
    _write_cache(npz, cfg_hash="e" * 16, git_sha="abc")
    runtime = _FakeRuntime(npz, config_path=None)
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.delenv("LINSTABPY_FORCE_RECOMPUTE", raising=False)
    monkeypatch.delenv("LINSTABPY_CASE_CONFIG", raising=False)
    monkeypatch.delenv("LINSTABPY_CASE2_CONFIG", raising=False)
    monkeypatch.delenv("LINSTABPY_CASE1_CONFIG", raising=False)
    main([])
    assert runtime.run_calls == 0


def test_driver_cli_corrupt_cache_recomputes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz = tmp_path / "out.npz"
    np.savez(npz, provenance_config_hash=np.array([1.0, 2.0]))
    cfg = tmp_path / "case.yaml"
    cfg.write_text("solver: {}\n")
    runtime = _FakeRuntime(npz, config_path=str(cfg))
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.delenv("LINSTABPY_FORCE_RECOMPUTE", raising=False)
    monkeypatch.setenv("LINSTABPY_CASE_CONFIG", str(cfg))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main([])
    assert runtime.run_calls == 1
    assert "Cache validation failed" in out.getvalue()
