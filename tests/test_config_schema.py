# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from pathlib import Path

import pytest

from linstabpy.config import ConfigValidationError, validate_config
from linstabpy.config.schema import TOP_LEVEL_ALLOWED
from linstabpy.utils.yaml_io import load_config


def test_valid_config_passes() -> None:
    cfg = {"numerics": {"fd_order": 6}, "Re": 1e5, "n_modes": 10}

    assert validate_config(cfg) is None


def test_aggregates_all_errors_in_one_message() -> None:
    cfg = {"numerics": {"fd_order": 5}, "Rey": 1.0, "n_modes": "ten"}

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(cfg)

    msg = str(exc_info.value)
    assert "fd_order" in msg
    assert "Rey" in msg
    assert "n_modes" in msg


@pytest.mark.parametrize("fd_order", [1, 3, 5, 7])
def test_fd_order_enum_rejects_odd(fd_order: int) -> None:
    with pytest.raises(ConfigValidationError):
        validate_config({"fd_order": fd_order})


@pytest.mark.parametrize("fd_order", [2, 4, 6, 8])
def test_fd_order_enum_accepts_valid(fd_order: int) -> None:
    assert validate_config({"fd_order": fd_order}) is None


def test_fd_order_checked_recursively() -> None:
    cfg = {
        "solver": {
            "eigen": {
                "numerics": {
                    "fd_order": 5,
                },
            },
        },
    }

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(cfg)

    assert "solver.eigen.numerics.fd_order" in str(exc_info.value)


def test_unknown_top_level_key_rejected_with_name() -> None:
    cfg = {"Rey": 1.0, "foo_bar": 2.0}

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(cfg)

    msg = str(exc_info.value)
    assert "Rey" in msg
    assert "foo_bar" in msg


def test_bool_not_accepted_for_int_field() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({"n_modes": True})

    assert "expected int, got bool" in str(exc_info.value)


def test_schema_rejects_wrong_python_types_for_known_fields() -> None:
    with pytest.raises(ConfigValidationError, match="expected int or float, got bool"):
        validate_config({"Re": True})
    with pytest.raises(ConfigValidationError, match="expected int or float, got str"):
        validate_config({"Re": "fast"})
    with pytest.raises(ConfigValidationError, match="expected bool, got int"):
        validate_config({"x_periodicity": 1})
    with pytest.raises(ConfigValidationError, match="expected str, got int"):
        validate_config({"model": 2})


def test_driver_rejects_malformed_config_before_case_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    try:
        _DRIVER_RUN_MODULE = importlib.import_module("linstabpy.driver.run")
    except ModuleNotFoundError as exc:
        pytest.skip(f"driver.run needs PETSc/mpi4py: {exc}")

    config_path = tmp_path / "malformed_absurd_grid.yaml"
    config_path.write_text(
        """
        nx: 200000
        ny: 200000
        numerics:
          fd_order: 5
        """,
        encoding="utf-8",
    )
    case_load_calls = 0

    def fail_if_case_load_reached(*args: object, **kwargs: object) -> None:
        nonlocal case_load_calls
        case_load_calls += 1
        raise AssertionError("case loading should not be reached for invalid config")

    monkeypatch.setenv("LINSTABPY_CASE_CONFIG", str(config_path))
    monkeypatch.setattr(_DRIVER_RUN_MODULE, "_stage_load_case", fail_if_case_load_reached)
    monkeypatch.setattr(_DRIVER_RUN_MODULE, "_log", lambda message: None)

    with pytest.raises(ConfigValidationError) as exc_info:
        _DRIVER_RUN_MODULE.run()

    assert str(config_path) in str(exc_info.value)
    assert "numerics.fd_order" in str(exc_info.value)
    assert case_load_calls == 0


def test_campaign_leftover_keys_are_rejected() -> None:
    with pytest.raises(ConfigValidationError, match="defaults"):
        validate_config({"defaults": {}})
    with pytest.raises(ConfigValidationError, match="mumps_ordering"):
        validate_config({"mumps_ordering": "metis"})


def test_no_false_positives_on_shipped_configs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_paths = []
    for path in repo_root.rglob("*.yaml"):
        relative = path.relative_to(repo_root)
        if ".venv" in relative.parts:
            continue
        if relative.parts[0] not in {"examples", "configs"}:
            continue
        config_paths.append(path)
    relative_config_paths = [path.relative_to(repo_root) for path in config_paths]

    assert config_paths, "shipped YAML config sweep found no files"
    assert any(path.parts[0] == "examples" for path in relative_config_paths)

    for path in config_paths:
        validate_config(load_config(path), source=str(path))


def test_unknown_key_suggests_close_match() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({"omega_targt": 1.0})
    assert "did you mean 'omega_target'?" in str(exc_info.value)


def test_unknown_key_without_close_match_has_no_suggestion() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({"zzz_nope_qqq": 1.0})
    msg = str(exc_info.value)
    assert "unknown top-level key 'zzz_nope_qqq' (possible typo)" in msg
    assert "did you mean" not in msg


def test_unknown_non_string_key_does_not_crash_suggester() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({123: 1.0})
    assert "unknown top-level key 123 (possible typo)" in str(exc_info.value)


@pytest.mark.parametrize(
    "mode", ["matlab_parity", "continuum", "matlab", "physics", "matlab-parity"]
)
def test_floquet_shift_accepts_known_modes_nested(mode: str) -> None:
    assert validate_config({"numerics": {"floquet_shift": mode}}) is None


def test_floquet_shift_rejects_unknown_mode_nested() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({"numerics": {"floquet_shift": "random"}})
    msg = str(exc_info.value)
    assert "numerics.floquet_shift" in msg
    assert "matlab_parity" in msg and "continuum" in msg


def test_floquet_shift_rejects_non_string_nested() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({"numerics": {"floquet_shift": 123}})
    assert "numerics.floquet_shift" in str(exc_info.value)
    assert "expected str" in str(exc_info.value)


def test_floquet_shift_top_level_still_validated() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({"floquet_shift": 123})
    assert "floquet_shift" in str(exc_info.value)
    assert validate_config({"floquet_shift": "continuum"}) is None


@pytest.mark.parametrize(
    ("key", "nested"),
    [
        ("beta", "numerics.beta"),
        ("kx", "numerics.beta"),
        ("Re", "physics.Re"),
        ("Ma", "physics.Ma"),
        ("fd_order", "numerics.fd_order"),
        ("penalty", "bc.penalty"),
        ("run_resolvent", "analysis.run_resolvent"),
        ("n_modes_stability", "analysis.n_modes_stability"),
        ("ncv", "solver.eigen.ncv"),
    ],
)
def test_cli_refuses_top_level_key_with_nested_equivalent(key: str, nested: str) -> None:
    """The command reads the nested key only, so the flat key is refused by name."""
    from linstabpy.config.schema import check_cli_ignored_keys

    values = {"fd_order": 4, "ncv": 4, "n_modes_stability": 4, "run_resolvent": False}
    value = values.get(key, 19.2)
    validate_config({key: value})  # the example scripts read the flat key
    with pytest.raises(ConfigValidationError, match=nested.replace(".", r"\.")):
        check_cli_ignored_keys({key: value}, source="case.yaml")


def test_cli_accepts_nested_keys() -> None:
    from linstabpy.config.schema import check_cli_ignored_keys

    cfg = {
        "numerics": {"beta": 19.2, "fd_order": 4},
        "physics": {"Re": 1.0e5},
        "analysis": {"run_resolvent": False},
        "bc": {"penalty": 1.0e8},
    }
    assert check_cli_ignored_keys(cfg) is None


def test_cli_ignored_keys_are_all_allowed_top_level_keys() -> None:
    from linstabpy.config.schema import _SCRIPT_ONLY_KEYS

    assert set(_SCRIPT_ONLY_KEYS) <= TOP_LEVEL_ALLOWED


def test_logging_block_is_accepted_with_its_one_key() -> None:
    assert validate_config({"logging": {"stage_heartbeat_seconds": 5}}) is None
    with pytest.raises(ConfigValidationError, match=r"logging\.chatty"):
        validate_config({"logging": {"chatty": True}})
    with pytest.raises(ConfigValidationError, match="stage_heartbeat_seconds"):
        validate_config({"logging": {"stage_heartbeat_seconds": "often"}})
    with pytest.raises(ConfigValidationError, match="logging"):
        validate_config({"logging": 5})


def test_driver_refuses_flat_re_and_accepts_nested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver stops on a flat Re instead of running with the default Re."""
    pytest.importorskip("mpi4py", reason="SKIPPED: missing mpi4py")
    pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
    from linstabpy.driver.run import _stage_load_runtime_config

    case = tmp_path / "case.yaml"
    monkeypatch.setenv("LINSTABPY_CASE_CONFIG", str(case))
    case.write_text("Re: 1.0e5\n")
    with pytest.raises(ConfigValidationError, match=r"physics\.Re"):
        _stage_load_runtime_config()
    case.write_text("physics:\n  Re: 1.0e5\nlogging:\n  stage_heartbeat_seconds: 5\n")
    assert _stage_load_runtime_config().config["physics"]["Re"] == 1.0e5


def _driver_run_module():
    import importlib

    try:
        return importlib.import_module("linstabpy.driver.run")
    except ModuleNotFoundError as exc:
        pytest.skip(f"driver.run needs PETSc/mpi4py: {exc}")


def test_driver_needs_case_config_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    run_module = _driver_run_module()
    monkeypatch.delenv("LINSTABPY_CASE_CONFIG", raising=False)

    with pytest.raises(ConfigValidationError, match="LINSTABPY_CASE_CONFIG"):
        run_module.run()


def test_driver_needs_physics_re(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_module = _driver_run_module()
    config_path = tmp_path / "no_re.yaml"
    config_path.write_text("numerics:\n  fd_order: 4\n", encoding="utf-8")
    monkeypatch.setenv("LINSTABPY_CASE_CONFIG", str(config_path))
    monkeypatch.setattr(run_module, "_log", lambda message: None)

    with pytest.raises(ConfigValidationError, match=r"physics\.Re is required"):
        run_module.run()


def test_driver_refuses_matlab_sponge_without_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_module = _driver_run_module()
    config_path = tmp_path / "sponge_aux.yaml"
    config_path.write_text(
        "physics:\n  Re: 1000.0\n  use_matlab_sponge_aux: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("LINSTABPY_CASE_CONFIG", str(config_path))
    monkeypatch.setattr(run_module, "_log", lambda message: None)

    with pytest.raises(ConfigValidationError, match=r"use_matlab_sponge_aux is true"):
        run_module.run()
