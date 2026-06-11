# SPDX-License-Identifier: GPL-3.0-only
"""Tests for results-driver alpha-filter handling."""

from __future__ import annotations

import importlib
import types

import numpy as np

results_runtime = importlib.import_module("linstabpy.driver.run")
from linstabpy.operators import LNSESystem


def test_effective_alpha_filter_keeps_filter_for_resolvent():
    assert results_runtime._effective_alpha_filter(0.0, run_resolvent=True) == 0.0
    assert results_runtime._effective_alpha_filter("none", run_resolvent=True) == "none"


def test_effective_alpha_filter_keeps_filter_without_resolvent():
    assert results_runtime._effective_alpha_filter(0.0, run_resolvent=False, context="unit") == 0.0


def test_compute_keeps_mesh_filter_without_resolvent(monkeypatch, tmp_path):
    mesh_create_calls: list[object] = []

    baseflow = types.SimpleNamespace(
        RHO=np.ones((2, 2)),
        U=np.ones((2, 2)),
        T=np.ones((2, 2)),
        dmudT=np.zeros((2, 2)),
        d2mudT2=np.zeros((2, 2)),
        kappa=1.4,
        Ma=2.0,
    )
    config_path = tmp_path / "case.jsonc"
    output_path = tmp_path / "case.npz"
    config_path.write_text(
        """
        {
          "paths": {"case_dir": "."},
          "analysis": {
            "run_stability": false,
            "run_resolvent": false
          },
          "numerics": {
            "alpha_filter": 0.0,
            "fd_order": 4
          },
          "solver": {}
        }
        """,
        encoding="utf-8",
    )

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
            mesh_create_calls.append(kwargs["alpha_filter"])
            return FakeCmesh()

    class FakeMat:
        def getInfo(self):
            return {"nz_used": 0}

    monkeypatch.setenv("LINSTABPY_CASE_CONFIG", str(config_path))
    monkeypatch.setenv("LINSTABPY_CASE_OUTPUT_NPZ", str(output_path))
    monkeypatch.setattr(results_runtime, "_log", lambda message: None)
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
    monkeypatch.setattr(
        results_runtime,
        "build_lnse_system_result",
        lambda *args, **kwargs: LNSESystem(
            operator=FakeMat(),
            idx={
                "rho_j": np.arange(4, dtype=np.int64),
                "u_j": np.arange(4, 8, dtype=np.int64),
                "v_j": np.arange(8, 12, dtype=np.int64),
                "w_j": np.arange(12, 16, dtype=np.int64),
                "T_j": np.arange(16, 20, dtype=np.int64),
            },
            bc_indices=np.array([], dtype=np.int64),
            bc_analysis=None,
            backend_metadata={"matrix_type": "petsc"},
        ),
    )
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

    results_runtime.run()
    assert mesh_create_calls == [0.0]
    with np.load(output_path, allow_pickle=False) as data:
        assert data["alpha_filter"].item() == "0.0"
