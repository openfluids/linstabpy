# SPDX-License-Identifier: GPL-3.0-only
"""Parametric parity checks for Song operator assembly.

Keep one frozen baseline. Nearby parameter changes must keep a
well-formed operator and must change the assembled matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.baseflow import load_case_fields, sutherland_air
from linstabpy.boundary_conditions.dirichlet import apply_dirichlet
from linstabpy.boundary_conditions.neumann import apply_neumann
from linstabpy.mesh import Mesh
from linstabpy.operators.assembly import (
    _assemble_from_prepared,
    _prepare_coefficients,
    build_lnse_operator,
)
from tests._paths import unpack_song_baseflow


@dataclass(frozen=True)
class _CaseCfg:
    name: str
    re: float
    ma: float
    pr: float
    kappa: float
    t0: float
    beta: float
    bc_penalty: float
    zero_mu_t_derivs: bool


CASES = (
    _CaseCfg(
        name="baseline_parity_mode",
        re=395000.0,
        ma=2.0,
        pr=0.747,
        kappa=1.4,
        t0=1540.0,
        beta=0.0,
        bc_penalty=1e8,
        zero_mu_t_derivs=True,
    ),
    _CaseCfg(
        name="variant_changed_flow_numbers",
        re=420000.0,
        ma=1.8,
        pr=0.72,
        kappa=1.37,
        t0=1500.0,
        beta=0.15,
        bc_penalty=1e8,
        zero_mu_t_derivs=False,
    ),
    _CaseCfg(
        name="variant_stronger_bc_penalty",
        re=395000.0,
        ma=2.2,
        pr=0.747,
        kappa=1.4,
        t0=1540.0,
        beta=0.0,
        bc_penalty=1e10,
        zero_mu_t_derivs=True,
    ),
)


def _case_dir() -> Path:
    return unpack_song_baseflow()


def _apply_song_bcs(
    mat: sp.spmatrix,
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    penalty: float,
) -> tuple[sp.csc_matrix, np.ndarray]:
    out, idx_lt = apply_dirichlet(mat, idx, "lt", "ruvwT", mesh.ngp, penalty=penalty)
    out, idx_b = apply_dirichlet(out, idx, "b", "ruvw", mesh.ngp, penalty=penalty)
    out, idx_nb = apply_neumann(out, mesh, idx, "b", "T", penalty=penalty)
    out, idx_nr = apply_neumann(out, mesh, idx, "r", "ruvwT", penalty=penalty)
    bc_all = np.unique(np.concatenate((idx_lt, idx_b, idx_nb, idx_nr)))
    return out.tocsc(), bc_all


def _assemble_case(cfg: _CaseCfg) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray, Mesh]:
    case_dir = _case_dir()
    X, Y, baseflow, sponge, _info = load_case_fields(
        case_dir,
        Re=cfg.re,
        Ma=cfg.ma,
        Pr=cfg.pr,
        kappa=cfg.kappa,
        T_0=cfg.t0,
        stride_ksi=3,
        stride_eta=3,
        transpose_input=True,
        zero_spanwise=False,
    )
    sutherland_air(baseflow)
    if cfg.zero_mu_t_derivs:
        baseflow.dmudT[:] = 0.0
        baseflow.d2mudT2[:] = 0.0

    ny, nx = X.shape
    cmesh = Mesh.create((0.0, 1.0), (0.0, 0.5), nx=nx, ny=ny, fd_order=4)
    mesh = cmesh.deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge

    l0_pre, idx = build_lnse_operator(mesh, baseflow, kx=cfg.beta, model="2D")
    l0_post, bc_indices = _apply_song_bcs(l0_pre, mesh, idx, cfg.bc_penalty)
    return l0_pre.tocsc(), l0_post, bc_indices, mesh


def _assert_same_sparse_structure_and_values(a: sp.csc_matrix, b: sp.csc_matrix) -> None:
    assert a.shape == b.shape
    assert np.array_equal(a.indptr, b.indptr)
    assert np.array_equal(a.indices, b.indices)
    if a.nnz == 0:
        return
    diff = np.abs(a.data - b.data)
    ref = np.maximum(np.abs(b.data), 1e-30)
    assert float(diff.max()) < 1e-10
    assert float((diff / ref).max()) < 1e-10


@pytest.mark.parametrize("cfg", CASES, ids=[case.name for case in CASES])
def test_song_parametric_operator_is_well_formed(cfg: _CaseCfg) -> None:
    case_dir = _case_dir()
    if not case_dir.exists():
        pytest.skip(f"Missing Song field directory: {case_dir}")

    l0_pre, l0_post, bc_indices, mesh = _assemble_case(cfg)

    assert l0_pre.shape == (mesh.ngp * 5, mesh.ngp * 5)
    assert l0_post.shape == l0_pre.shape
    assert l0_pre.nnz > 0
    assert l0_post.nnz > 0
    assert np.isfinite(l0_pre.data).all()
    assert np.isfinite(l0_post.data).all()
    assert bc_indices.size > 0
    assert np.unique(bc_indices).size == bc_indices.size


@pytest.mark.parametrize("cfg", CASES, ids=[case.name for case in CASES])
def test_song_parametric_build_vs_prepared_parity(cfg: _CaseCfg) -> None:
    case_dir = _case_dir()
    if not case_dir.exists():
        pytest.skip(f"Missing Song field directory: {case_dir}")

    X, Y, baseflow, sponge, _info = load_case_fields(
        case_dir,
        Re=cfg.re,
        Ma=cfg.ma,
        Pr=cfg.pr,
        kappa=cfg.kappa,
        T_0=cfg.t0,
        stride_ksi=3,
        stride_eta=3,
        transpose_input=True,
        zero_spanwise=False,
    )
    sutherland_air(baseflow)
    if cfg.zero_mu_t_derivs:
        baseflow.dmudT[:] = 0.0
        baseflow.d2mudT2[:] = 0.0

    ny, nx = X.shape
    cmesh = Mesh.create((0.0, 1.0), (0.0, 0.5), nx=nx, ny=ny, fd_order=4)
    mesh = cmesh.deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge

    l0_from_build, idx = build_lnse_operator(mesh, baseflow, kx=cfg.beta, model="2D")
    prepared = _prepare_coefficients(mesh, baseflow, kx=cfg.beta, model="2D", m=0.0)
    l0_from_prepared = _assemble_from_prepared(prepared, floquet_exp=0.0)

    l0_build_post, _ = _apply_song_bcs(l0_from_build, mesh, idx, cfg.bc_penalty)
    l0_prepared_post, _ = _apply_song_bcs(l0_from_prepared, mesh, idx, cfg.bc_penalty)

    _assert_same_sparse_structure_and_values(l0_from_build.tocsc(), l0_from_prepared.tocsc())
    _assert_same_sparse_structure_and_values(l0_build_post, l0_prepared_post)


def test_song_parametric_changes_modify_operator() -> None:
    case_dir = _case_dir()
    if not case_dir.exists():
        pytest.skip(f"Missing Song field directory: {case_dir}")

    baseline = CASES[0]
    variant_a = CASES[1]
    variant_b = CASES[2]

    l0_base, _, _, _ = _assemble_case(baseline)
    l0_a, _, _, _ = _assemble_case(variant_a)
    l0_b, _, _, _ = _assemble_case(variant_b)

    rng = np.random.default_rng(1234)
    v = rng.standard_normal(l0_base.shape[1]) + 1j * rng.standard_normal(l0_base.shape[1])

    y_base = l0_base @ v
    y_a = l0_a @ v
    y_b = l0_b @ v

    rel_a = float(np.linalg.norm(y_a - y_base) / max(np.linalg.norm(y_base), 1e-30))
    rel_b = float(np.linalg.norm(y_b - y_base) / max(np.linalg.norm(y_base), 1e-30))

    assert rel_a > 1e-4
    assert rel_b > 1e-4
