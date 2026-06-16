# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import numpy as np
import pytest

from linstabpy.baseflow import BaseFlow, sutherland_air
from linstabpy.mesh import Mesh
from linstabpy.operators.assembly import build_lnse_system
from linstabpy.operators.energy_norm import compute_chu_norm
from linstabpy.solvers.router.band_marking import STABLE


def _tiny_problem() -> tuple[Mesh, BaseFlow]:
    mesh = Mesh.create((0.0, 1.0), (0.0, 1.0), nx=4, ny=4, fd_order=2)
    ones = np.ones_like(mesh.X)
    zeros = np.zeros_like(mesh.X)
    base_flow = BaseFlow(
        Re=1.0e3,
        Ma=0.3,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
        RHO=ones.copy(),
        U=0.2 * ones.copy(),
        V=zeros.copy(),
        W=zeros.copy(),
        T=ones.copy(),
    )
    sutherland_air(base_flow)
    return mesh, base_flow


def _cfg(mesh: Mesh) -> dict[str, object]:
    return {
        "mesh": mesh,
        "model": "2D",
        "bc_recipe": [{"type": "dirichlet", "borders": "lrtb", "variables": "ruvwT"}],
        "bc_penalty": 10.0,
        "assembly_backend": "scipy",
        "mask_bc_forcing": False,
        "n_modes_resolvent": 1,
        "stage_heartbeat_seconds": 0.0,
        "band_marking": {"n_modes": 1, "ncv": 3, "danger_radius": 0.0},
        "solver": {
            "eigen": {"eps_tol": 1.0e-8, "eps_max_it": 1000},
            "resolvent": {"eigen": {"eps_tol": 1.0e-8}},
        },
    }


def _chu_weights(mesh: Mesh, base_flow: BaseFlow) -> np.ndarray:
    p = mesh.used_ind
    W_diag, _ = compute_chu_norm(
        mesh.W.ravel(order="F")[p],
        base_flow.RHO.ravel(order="F")[p],
        base_flow.T.ravel(order="F")[p],
        base_flow.kappa,
        base_flow.Ma,
        mesh.ngp,
    )
    return W_diag


def test_tiny_baseflow_resolvent_map_matches_direct_oracle() -> None:
    pytest.importorskip("petsc4py", reason="SKIPPED: missing PETSc stack")
    pytest.importorskip("slepc4py", reason="SKIPPED: missing SLEPc stack")

    from linstabpy import resolvent_map
    from linstabpy.resolvent.analysis import resolvent_analysis

    mesh, base_flow = _tiny_problem()
    cfg = _cfg(mesh)
    beta = np.array([0.1])
    omega = np.array([0.2, 0.4])

    result = resolvent_map(base_flow, beta, omega, cfg)

    assert result.gain.shape == (beta.size, omega.size)
    assert result.regime_label.shape == result.gain.shape
    assert result.reason_code.shape == result.gain.shape
    assert result.regime_label[0, 0] == STABLE
    assert any(record["engine"] == "regime-a" for record in result.point_records)
    assert result.failure_records == []
    assert result.cost_metrics["cost_mpi_ranks"] >= 1
    for key in ("cost_wall_total_s", "cost_wall_assemble_s", "cost_wall_solve_s", "cost_peak_rss_bytes"):
        assert key in result.cost_metrics

    L0, _idx, _bc = build_lnse_system(
        mesh,
        base_flow,
        kx=float(beta[0]),
        model="2D",
        bc_recipe=cfg["bc_recipe"],
        bc_penalty=float(cfg["bc_penalty"]),
        backend="scipy",
        solver_cfg=cfg,
    )
    direct_gains, *_ = resolvent_analysis(
        L0,
        float(omega[0]),
        1,
        _chu_weights(mesh, base_flow),
        solver_cfg=cfg,
    )

    np.testing.assert_allclose(result.gain[0, 0], direct_gains[0], rtol=1.0e-7, atol=1.0e-7)
