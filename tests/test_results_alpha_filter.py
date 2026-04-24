"""Tests for results-driver alpha-filter handling."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
from linstab2d import results_runtime


ROOT = Path(__file__).resolve().parent.parent
NCV_PATH = ROOT / "results" / "case2" / "legacy" / "run_ncv_milestones.py"
SHOCK_PATH = ROOT / "results" / "case2" / "legacy" / "run_shock_cleanup_experiment.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_effective_alpha_filter_keeps_filter_for_resolvent():
    assert results_runtime._effective_alpha_filter(0.0, run_resolvent=True) == 0.0
    assert results_runtime._effective_alpha_filter("none", run_resolvent=True) == "none"


def test_effective_alpha_filter_keeps_filter_without_resolvent():
    assert results_runtime._effective_alpha_filter(0.0, run_resolvent=False, context="unit") == 0.0


def test_run_ncv_prepare_case_keeps_mesh_filter_without_resolvent(monkeypatch):
    ncv_runtime = _load_module(NCV_PATH, "case2_ncv_alpha_runtime")
    mesh_create_calls: list[object] = []

    baseflow = types.SimpleNamespace(
        RHO=np.ones((2, 2)),
        T=np.ones((2, 2)),
        dmudT=np.zeros((2, 2)),
        d2mudT2=np.zeros((2, 2)),
        kappa=1.4,
        Ma=2.0,
    )

    class FakeCmesh:
        def deform(self, *, X_new, Y_new):
            return types.SimpleNamespace(
                ngp=4,
                used_ind=np.arange(4, dtype=np.int64),
                W=np.ones((2, 2)),
                sponge=None,
            )

    class FakeMeshClass:
        @staticmethod
        def create(*args, **kwargs):
            mesh_create_calls.append(kwargs["alpha_filter"])
            return FakeCmesh()

    class FakeMat:
        def getInfo(self):
            return {"nz_used": 0}

    monkeypatch.setattr(ncv_runtime, "_log", lambda message: None)
    monkeypatch.setattr(ncv_runtime, "_log_stage", lambda stage, start: None)
    monkeypatch.setattr(
        ncv_runtime,
        "load_hamada_case",
        lambda *args, **kwargs: (
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            baseflow,
            np.zeros((2, 2)),
        ),
    )
    monkeypatch.setattr(ncv_runtime, "sutherland_air", lambda baseflow: None)
    monkeypatch.setattr(ncv_runtime, "Mesh", FakeMeshClass)
    monkeypatch.setattr(
        ncv_runtime,
        "build_lnse_system",
        lambda *args, **kwargs: (FakeMat(), {"u_j": np.arange(4, dtype=np.int64)}, np.array([], dtype=np.int64)),
    )
    monkeypatch.setattr(
        ncv_runtime,
        "analyze_bc_recipe",
        lambda *args, **kwargs: {
            "total_bc_indices": np.array([], dtype=np.int64),
            "effective_dirichlet_indices": np.array([], dtype=np.int64),
            "effective_neumann_indices": np.array([], dtype=np.int64),
            "overlap_indices": np.array([], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        ncv_runtime,
        "compute_chu_norm",
        lambda *args, **kwargs: (np.ones(20, dtype=float), np.ones(20, dtype=float)),
    )

    ncv_runtime._prepare_case(
        {
            "analysis": {"run_resolvent": False},
            "numerics": {"alpha_filter": 0.0},
            "solver": {},
        }
    )

    assert mesh_create_calls == [0.0]


def test_shock_cleanup_branch_keeps_mesh_filter_without_resolvent(monkeypatch, tmp_path):
    shock_runtime = _load_module(SHOCK_PATH, "case2_shock_alpha_runtime")
    mesh_create_calls: list[object] = []

    baseflow = types.SimpleNamespace(
        RHO=np.ones((2, 2)),
        U=np.ones((2, 2)),
        V=np.ones((2, 2)),
        W=np.zeros((2, 2)),
        T=np.ones((2, 2)),
        kappa=1.4,
        Ma=2.0,
    )

    class FakeCmesh:
        def deform(self, *, X_new, Y_new):
            return types.SimpleNamespace(
                ngp=4,
                used_ind=np.arange(4, dtype=np.int64),
                W=np.ones((2, 2)),
                sponge=None,
                filters=None,
            )

    class FakeMeshClass:
        @staticmethod
        def create(*args, **kwargs):
            mesh_create_calls.append(kwargs["alpha_filter"])
            return FakeCmesh()

    class FakeMat:
        shape = (20, 20)

        def getInfo(self):
            return {"nz_used": 0}

    fake_petsc = types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(getRank=lambda: 1))

    monkeypatch.setattr(shock_runtime, "_log", lambda message: None)
    monkeypatch.setattr(shock_runtime, "_log_stage", lambda stage, start: None)
    monkeypatch.setattr(shock_runtime, "Mesh", FakeMeshClass)
    monkeypatch.setattr(
        shock_runtime,
        "build_lnse_system",
        lambda *args, **kwargs: (FakeMat(), {"u_j": np.arange(4, dtype=np.int64)}, np.array([], dtype=np.int64)),
    )
    monkeypatch.setattr(
        shock_runtime,
        "analyze_bc_recipe",
        lambda *args, **kwargs: {
            "effective_dirichlet_indices": np.array([], dtype=np.int64),
            "column_zeroed_dirichlet_indices": np.array([], dtype=np.int64),
            "total_bc_indices": np.array([], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        shock_runtime,
        "compute_chu_norm",
        lambda *args, **kwargs: (np.ones(20, dtype=float), np.ones(20, dtype=float)),
    )
    monkeypatch.setattr(shock_runtime, "PETSc", fake_petsc)

    shock_runtime._run_branch(
        "baseline",
        X=np.zeros((2, 2)),
        Y=np.zeros((2, 2)),
        baseflow=baseflow,
        sponge=np.zeros((2, 2)),
        config={"numerics": {"alpha_filter": 0.0}, "solver": {}},
        run_stability=False,
        run_resolvent=False,
        save_mode_count=1,
        output_dir=tmp_path,
        cleanup_summary="none",
    )

    assert mesh_create_calls == [0.0]
