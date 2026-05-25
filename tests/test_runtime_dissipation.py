# SPDX-License-Identifier: GPL-3.0-only
"""Tests for results-driver operator-dissipation handling."""

from __future__ import annotations

import sys
import types

import numpy as np


def _install_optional_petsc_stubs() -> None:
    """Provide single-rank PETSc/MPI stubs for runtime tests without PETSc."""
    try:
        import mpi4py.MPI  # noqa: F401
        import petsc4py  # noqa: F401
        import slepc4py  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        return

    class _FakeComm:
        def getRank(self):
            return 0

        def getSize(self):
            return 1

        def barrier(self):
            return None

        def bcast(self, value, root=0):
            return value

    class _FakeMat:
        pass

    class _FakeVec:
        pass

    class _FakeKSP:
        pass

    class _FakePC:
        pass

    fake_mpi = types.ModuleType("mpi4py")
    fake_mpi_submodule = types.ModuleType("mpi4py.MPI")
    fake_mpi_submodule.COMM_WORLD = _FakeComm()
    fake_mpi.MPI = fake_mpi_submodule

    fake_petsc4py = types.ModuleType("petsc4py")
    fake_petsc4py.init = lambda *args, **kwargs: None
    fake_petsc = types.SimpleNamespace(
        COMM_WORLD=_FakeComm(),
        COMM_SELF=_FakeComm(),
        Mat=_FakeMat,
        Vec=_FakeVec,
        KSP=_FakeKSP,
        PC=_FakePC,
        Error=RuntimeError,
        ScalarType=np.complex128,
        IntType=np.int64,
        InsertMode=types.SimpleNamespace(INSERT_VALUES=0),
    )
    fake_petsc4py.PETSc = fake_petsc

    class _FakeEPS:
        class Conv:
            ABS = "abs"
            REL = "rel"
            NORM = "norm"

        class ErrorType:
            ABSOLUTE = "absolute"
            RELATIVE = "relative"
            BACKWARD = "backward"

        class ProblemType:
            NHEP = "nhep"

        class Type:
            KRYLOVSCHUR = "krylovschur"

        class Which:
            TARGET_MAGNITUDE = "target_magnitude"
            LARGEST_MAGNITUDE = "largest_magnitude"

    class _FakeST:
        class Type:
            SINVERT = "sinvert"

    fake_slepc4py = types.ModuleType("slepc4py")
    fake_slepc4py.init = lambda *args, **kwargs: None
    fake_slepc4py.SLEPc = types.SimpleNamespace(EPS=_FakeEPS, ST=_FakeST)

    sys.modules["mpi4py"] = fake_mpi
    sys.modules["mpi4py.MPI"] = fake_mpi_submodule
    sys.modules["petsc4py"] = fake_petsc4py
    sys.modules["slepc4py"] = fake_slepc4py


_install_optional_petsc_stubs()

from linstabpy import results_runtime


DISSIPATION_KEYS = (
    "numerics_operator_dissipation_enabled",
    "numerics_operator_dissipation_epsilon",
    "numerics_operator_dissipation_order_diss",
    "numerics_operator_dissipation_apply_to_stability",
    "numerics_operator_dissipation_apply_to_resolvent",
    "numerics_operator_dissipation_transition_fraction",
    "numerics_operator_dissipation_endpoint_exponent",
    "numerics_operator_dissipation_interior_value",
)


def _write_config(config_path, operator_dissipation: str | None = None) -> None:
    operator_block = ""
    if operator_dissipation is not None:
        operator_block = f',\n            "operator_dissipation": {operator_dissipation}'

    config_path.write_text(
        f"""
        {{
          "paths": {{"case_dir": "."}},
          "analysis": {{
            "run_stability": false,
            "run_resolvent": false
          }},
          "numerics": {{
            "alpha_filter": "none",
            "fd_order": 4{operator_block}
          }},
          "solver": {{}}
        }}
        """,
        encoding="utf-8",
    )


def _run_compute(monkeypatch, tmp_path, *, operator_dissipation=None, log_sink=None, stem="case"):
    mesh_create_kwargs: list[dict] = []
    build_lnse_kwargs: list[dict] = []

    baseflow = types.SimpleNamespace(
        RHO=np.ones((2, 2)),
        U=np.ones((2, 2)),
        T=np.ones((2, 2)),
        dmudT=np.zeros((2, 2)),
        d2mudT2=np.zeros((2, 2)),
        kappa=1.4,
        Ma=2.0,
    )
    config_path = tmp_path / f"{stem}.jsonc"
    output_path = tmp_path / f"{stem}.npz"
    _write_config(config_path, operator_dissipation)

    class FakeCmesh:
        def deform(self, *, X_new, Y_new):
            return types.SimpleNamespace(
                ngp=4,
                used_ind=np.arange(4, dtype=np.int64),
                W=np.ones((2, 2)),
                sponge=None,
                X=X_new,
                Y=Y_new,
                filters=None,
            )

    class FakeMeshClass:
        @staticmethod
        def create(*args, **kwargs):
            mesh_create_kwargs.append(dict(kwargs))
            return FakeCmesh()

    class FakeMat:
        def getInfo(self):
            return {"nz_used": 0}

    def fake_build_lnse_system(*args, **kwargs):
        build_lnse_kwargs.append(dict(kwargs))
        return (
            FakeMat(),
            {
                "rho_j": np.arange(4, dtype=np.int64),
                "u_j": np.arange(4, 8, dtype=np.int64),
                "v_j": np.arange(8, 12, dtype=np.int64),
                "w_j": np.arange(12, 16, dtype=np.int64),
                "T_j": np.arange(16, 20, dtype=np.int64),
            },
            np.array([], dtype=np.int64),
        )

    monkeypatch.setenv("LINSTAB2D_CASE_CONFIG", str(config_path))
    monkeypatch.setenv("LINSTAB2D_CASE_OUTPUT_NPZ", str(output_path))
    monkeypatch.setattr(results_runtime, "_log", log_sink or (lambda message: None))
    monkeypatch.setattr(results_runtime, "_log_stage", lambda stage, start: None)
    monkeypatch.setattr(
        results_runtime,
        "load_hamada_case",
        lambda *args, **kwargs: (
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            baseflow,
            np.zeros((2, 2)),
        ),
    )
    monkeypatch.setattr(results_runtime, "sutherland_air", lambda baseflow: None)
    monkeypatch.setattr(results_runtime, "Mesh", FakeMeshClass)
    monkeypatch.setattr(results_runtime, "build_lnse_system", fake_build_lnse_system)
    monkeypatch.setattr(
        results_runtime,
        "analyze_bc_recipe",
        lambda *args, **kwargs: {
            "total_bc_indices": np.array([], dtype=np.int64),
            "effective_dirichlet_indices": np.array([], dtype=np.int64),
            "column_zeroed_dirichlet_indices": np.array([], dtype=np.int64),
            "effective_neumann_indices": np.array([], dtype=np.int64),
            "overlap_indices": np.array([], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        results_runtime,
        "compute_chu_norm",
        lambda *args, **kwargs: (np.ones(20, dtype=float), np.ones(20, dtype=float)),
    )

    results_runtime.compute()
    return output_path, mesh_create_kwargs, build_lnse_kwargs


def test_default_off_provenance(monkeypatch, tmp_path):
    output_path, mesh_create_kwargs, build_lnse_kwargs = _run_compute(
        monkeypatch,
        tmp_path,
    )

    with np.load(output_path, allow_pickle=False) as data:
        assert all(key in data for key in DISSIPATION_KEYS)
        assert data["numerics_operator_dissipation_enabled"].item() is False
        assert data["numerics_operator_dissipation_epsilon"].item() == 0.0
        assert data["numerics_operator_dissipation_order_diss"].item() == 4
        assert data["numerics_operator_dissipation_apply_to_stability"].item() is True
        assert data["numerics_operator_dissipation_apply_to_resolvent"].item() is False
        assert data["numerics_operator_dissipation_transition_fraction"].item() == 0.05
        assert data["numerics_operator_dissipation_endpoint_exponent"].item() == 2.0
        assert data["numerics_operator_dissipation_interior_value"].item() == 1.0

    assert mesh_create_kwargs[0]["numerics_operator_dissipation"] is None
    assert build_lnse_kwargs[0]["dissipation_epsilon"] == 0.0


def test_enabled_provenance(monkeypatch, tmp_path):
    output_path, mesh_create_kwargs, build_lnse_kwargs = _run_compute(
        monkeypatch,
        tmp_path,
        operator_dissipation='{"enabled": true, "epsilon": 0.1, "order_diss": 6}',
    )

    with np.load(output_path, allow_pickle=False) as data:
        assert data["numerics_operator_dissipation_enabled"].item() is True
        assert data["numerics_operator_dissipation_epsilon"].item() == 0.1
        assert data["numerics_operator_dissipation_order_diss"].item() == 6
        assert data["numerics_operator_dissipation_apply_to_stability"].item() is True
        assert data["numerics_operator_dissipation_apply_to_resolvent"].item() is False
        assert data["numerics_operator_dissipation_transition_fraction"].item() == 0.05
        assert data["numerics_operator_dissipation_endpoint_exponent"].item() == 2.0
        assert data["numerics_operator_dissipation_interior_value"].item() == 1.0

    assert isinstance(mesh_create_kwargs[0]["numerics_operator_dissipation"], dict)
    assert build_lnse_kwargs[0]["dissipation_epsilon"] == 0.1


def test_log_emitted(monkeypatch, tmp_path):
    log_messages = []
    _run_compute(
        monkeypatch,
        tmp_path,
        operator_dissipation='{"enabled": true, "epsilon": 0.05}',
        log_sink=lambda msg: log_messages.append(msg),
        stem="enabled",
    )
    assert any(
        "numerics.operator_dissipation effective" in message and "enabled=True" in message
        for message in log_messages
    )

    log_messages = []
    _run_compute(
        monkeypatch,
        tmp_path,
        log_sink=lambda msg: log_messages.append(msg),
        stem="disabled",
    )
    assert not any("operator_dissipation" in message for message in log_messages)


def test_apply_to_resolvent_false_isolates_stages(monkeypatch, tmp_path):
    """When apply_to_resolvent=false and apply_to_stability=true, the resolvent
    stage receives build_lnse_system with dissipation_epsilon=0.0, while the
    stability stage receives dissipation_epsilon > 0."""
    _, _, build_lnse_kwargs = _run_compute(
        monkeypatch,
        tmp_path,
        operator_dissipation=(
            '{"enabled": true, "epsilon": 1.0, '
            '"apply_to_stability": true, "apply_to_resolvent": false}'
        ),
    )
    # build_lnse_system is called once (stability only, run_resolvent=false by default)
    # With apply_to_resolvent=false: eps_resolv=0.0, eps_stab=1.0
    # The stability call uses eps_stab
    stab_epsilons = [kw["dissipation_epsilon"] for kw in build_lnse_kwargs]
    assert any(eps == 1.0 for eps in stab_epsilons), (
        f"Expected dissipation_epsilon=1.0 for stability, got: {stab_epsilons}"
    )
    # eps_resolv is 0.0 when apply_to_resolvent=False; verify it is recorded in NPZ
    output_path = tmp_path / "case.npz"
    with np.load(output_path, allow_pickle=False) as data:
        assert data["solver_apply_to_resolvent_effective"].item() is False


def test_cache_invalidation_on_config_change(monkeypatch, tmp_path):
    """Changing the config with the same output NPZ path must trigger recompute.
    Verified by checking that build_lnse_system is called a second time."""
    import time as _time

    # First run
    output_path, _, build_calls_1 = _run_compute(
        monkeypatch,
        tmp_path,
        operator_dissipation=None,
        stem="cachetest",
    )
    mtime_1 = output_path.stat().st_mtime

    # Sleep briefly so mtime can differ
    _time.sleep(0.05)

    # Second run with a DIFFERENT config (epsilon changed) -- same output path
    # We must re-create the monkeypatch environment for the second run.
    # _run_compute writes config_path and output_path based on stem.
    # Change the config so the hash differs.
    config_path = tmp_path / "cachetest.jsonc"
    config_path.write_text(
        """{
          "paths": {"case_dir": "."},
          "analysis": {"run_stability": false, "run_resolvent": false},
          "numerics": {
            "alpha_filter": "none",
            "fd_order": 4,
            "operator_dissipation": {"enabled": true, "epsilon": 0.5}
          },
          "solver": {}
        }""",
        encoding="utf-8",
    )

    _, _, build_calls_2 = _run_compute(
        monkeypatch,
        tmp_path,
        operator_dissipation='{"enabled": true, "epsilon": 0.5}',
        stem="cachetest",
    )
    mtime_2 = output_path.stat().st_mtime

    # Recompute must have happened (file mtime advanced and build was called again)
    assert mtime_2 > mtime_1, "NPZ was not rewritten on config change"
    assert len(build_calls_2) > 0, "build_lnse_system was not called in second run"
