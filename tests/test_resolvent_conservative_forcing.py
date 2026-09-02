# SPDX-License-Identifier: GPL-3.0-only
"""Conservative-momentum forcing space (issue resolvent-c9uk).

Kept PETSc-free (only ``linstabpy.resolvent.conservative_forcing`` and the
scipy-path branch of ``resolvent_map._forcing_response_mats``), same
constraint as ``test_forcing_response_mats_loud_fail.py``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.resolvent.conservative_forcing import (
    build_conservative_momentum_matrix,
    conservative_momentum_coefficients,
    dewhiten_forcing,
    validate_conservative_momentum_vars,
)

resolvent_map = importlib.import_module("linstabpy.solvers.router.resolvent_map")

KAPPA = 1.4
MA = 2.0


def _grid_idx() -> tuple[SimpleNamespace, dict[str, np.ndarray], int]:
    # 2x3 grid (Ny=2, Nx=3), Fortran-order point layout -- same convention as
    # tests/test_router_entry.py's _region_mesh_idx.
    X = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    Y = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    ngp = X.size
    idx = {
        f"{var}_j": np.arange(i * ngp, (i + 1) * ngp, dtype=np.int64)
        for i, var in enumerate(("rho", "u", "v", "w", "T"))
    }
    return X, Y, idx, ngp


def _base_flow_arrays(ngp: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "rho_bar": 1.0 + 0.1 * rng.uniform(size=ngp),
        "u_bar": 0.5 + 0.3 * rng.uniform(size=ngp),
        "v_bar": 0.1 * rng.uniform(size=ngp),
        "w_bar": 0.05 * rng.uniform(size=ngp),
        "w_int": 0.2 + 0.05 * rng.uniform(size=ngp),
    }


# ---------------------------------------------------------------------------
# 1. ROUND-TRIP: conservative momentum forcing -> equivalent primitive forcing
# ---------------------------------------------------------------------------


def test_round_trip_maps_conservative_momentum_to_primitive_T() -> None:
    X, Y, idx, ngp = _grid_idx()
    ndof = 5 * ngp
    flow = _base_flow_arrays(ngp)
    point_indices = np.arange(ngp, dtype=np.int64)  # whole domain

    B_tilde, momentum_dewhiten, info = build_conservative_momentum_matrix(
        idx,
        ndof,
        point_indices,
        flow["rho_bar"],
        flow["u_bar"],
        flow["v_bar"],
        flow["w_bar"],
        flow["w_int"],
        KAPPA,
        MA,
    )
    momentum_dofs = info["momentum_dofs"]
    assert info["n_region"] == ngp
    assert momentum_dofs.size == 3 * ngp

    rng = np.random.default_rng(1)
    # Physical conservative momentum forcing f_c = (f_rou, f_rov, f_row) at
    # each point, packed into the u_j/v_j/w_j slots of a full ndof vector.
    f_c = np.zeros(ndof, dtype=complex)
    f_c[idx["u_j"]] = rng.uniform(size=ngp) + 1j * rng.uniform(size=ngp)
    f_c[idx["v_j"]] = rng.uniform(size=ngp) + 1j * rng.uniform(size=ngp)
    f_c[idx["w_j"]] = rng.uniform(size=ngp) + 1j * rng.uniform(size=ngp)

    # Whitened coefficient g = diag(w)^(1/2) f_c (inverse of dewhiten_forcing).
    g = np.zeros(ndof, dtype=complex)
    g[momentum_dofs] = f_c[momentum_dofs] / momentum_dewhiten[momentum_dofs]

    # Invert: de-whitening g must reproduce f_c exactly.
    f_c_recovered = dewhiten_forcing(g, momentum_dewhiten, momentum_dofs)
    np.testing.assert_allclose(f_c_recovered, f_c, atol=1e-14, rtol=1e-14)

    primitive = B_tilde @ g  # == B @ f_c  (B_tilde = B @ diag(w)^(-1/2))

    u_p = f_c[idx["u_j"]] / flow["rho_bar"]
    v_p = f_c[idx["v_j"]] / flow["rho_bar"]
    w_p = f_c[idx["w_j"]] / flow["rho_bar"]
    expected_T = (
        -KAPPA
        * (KAPPA - 1.0)
        * MA**2
        * (flow["u_bar"] * u_p + flow["v_bar"] * v_p + flow["w_bar"] * w_p)
    )

    np.testing.assert_allclose(primitive[idx["rho_j"]], 0.0, atol=1e-14)
    np.testing.assert_allclose(primitive[idx["u_j"]], u_p, rtol=0, atol=1e-13)
    np.testing.assert_allclose(primitive[idx["v_j"]], v_p, rtol=0, atol=1e-13)
    np.testing.assert_allclose(primitive[idx["w_j"]], w_p, rtol=0, atol=1e-13)
    np.testing.assert_allclose(primitive[idx["T_j"]], expected_T, rtol=1e-12, atol=1e-13)


def test_temperature_coeff_scales_with_mach_squared() -> None:
    """T_coeff must go as Ma^2, not Ma. Ma=2 hides **2 vs *2."""
    rho = np.array([1.0, 2.0])
    u = np.array([0.5, 0.25])
    v = np.zeros(2)
    w = np.zeros(2)
    ma_lo, ma_hi = 1.5, 3.0
    t_lo = conservative_momentum_coefficients(rho, u, v, w, KAPPA, ma_lo)["u"][1]
    t_hi = conservative_momentum_coefficients(rho, u, v, w, KAPPA, ma_hi)["u"][1]
    np.testing.assert_allclose(t_hi, t_lo * (ma_hi / ma_lo) ** 2)
    assert not np.allclose(t_hi, t_lo * (ma_hi / ma_lo))


# ---------------------------------------------------------------------------
# 2. NORM: plain L2 of the whitened coefficient == mass-weighted L2 of f_c
# ---------------------------------------------------------------------------


def test_whitened_l2_norm_equals_mass_weighted_conservative_norm() -> None:
    X, Y, idx, ngp = _grid_idx()
    ndof = 5 * ngp
    flow = _base_flow_arrays(ngp, seed=2)
    point_indices = np.arange(ngp, dtype=np.int64)

    _B_tilde, momentum_dewhiten, info = build_conservative_momentum_matrix(
        idx,
        ndof,
        point_indices,
        flow["rho_bar"],
        flow["u_bar"],
        flow["v_bar"],
        flow["w_bar"],
        flow["w_int"],
        KAPPA,
        MA,
    )
    momentum_dofs = info["momentum_dofs"]

    rng = np.random.default_rng(3)
    g = np.zeros(ndof, dtype=complex)
    g[momentum_dofs] = rng.normal(size=momentum_dofs.size) + 1j * rng.normal(
        size=momentum_dofs.size
    )

    plain_l2_sq = float(np.real(np.vdot(g, g)))

    f_c = dewhiten_forcing(g, momentum_dewhiten, momentum_dofs)
    w_int_full = np.zeros(ndof, dtype=float)
    for var in ("u", "v", "w"):
        w_int_full[idx[f"{var}_j"][point_indices]] = flow["w_int"][point_indices]
    mass_weighted_sq = float(np.real(np.vdot(f_c, w_int_full * f_c)))

    assert plain_l2_sq == pytest.approx(mass_weighted_sq, rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# 3. CONFIG: "space": "conservative_momentum" parsing
# ---------------------------------------------------------------------------


def test_validate_conservative_momentum_vars_defaults_to_all_three() -> None:
    assert validate_conservative_momentum_vars(None, name="input_region") == ("u", "v", "w")


def test_validate_conservative_momentum_vars_accepts_subset() -> None:
    assert validate_conservative_momentum_vars(["u", "v"], name="input_region") == ("u", "v")


@pytest.mark.parametrize("bad_vars", [["rho"], ["T"], ["u", "rho"], []])
def test_validate_conservative_momentum_vars_rejects_non_momentum(bad_vars) -> None:
    with pytest.raises(ValueError, match="momentum"):
        validate_conservative_momentum_vars(bad_vars, name="input_region")


def _fake_mesh_and_base_flow():
    X, Y, idx, ngp = _grid_idx()
    flow = _base_flow_arrays(ngp, seed=4)
    mesh = SimpleNamespace(
        X=X,
        Y=Y,
        W=flow["w_int"].reshape(X.shape, order="F"),
        used_ind=np.arange(ngp, dtype=np.int64),
        filters=None,
    )
    base_flow = SimpleNamespace(
        RHO=flow["rho_bar"].reshape(X.shape, order="F"),
        U=flow["u_bar"].reshape(X.shape, order="F"),
        V=flow["v_bar"].reshape(X.shape, order="F"),
        W=flow["w_bar"].reshape(X.shape, order="F"),
        kappa=KAPPA,
        Ma=MA,
    )
    return mesh, idx, ngp, base_flow


def test_forcing_response_mats_conservative_momentum_space_default_stays_primitive() -> None:
    """No `space` key -> the historical diagonal primitive-variable mask."""
    mesh, idx, ngp, base_flow = _fake_mesh_and_base_flow()
    ndof = 5 * ngp
    L0 = sp.eye(ndof, format="csc")
    bc_indices = np.array([], dtype=np.int64)
    cfg = {"input_region": {"vars": ["u"]}, "mask_bc_forcing": False}

    B_mat, C_mat = resolvent_map._forcing_response_mats(
        L0, ndof, bc_indices, cfg, mesh=mesh, idx=idx, base_flow=base_flow
    )
    # Historical behaviour: diagonal mask, so B_mat.diagonal() is a 0/1 vector.
    diag = np.asarray(B_mat.diagonal())
    assert set(np.unique(diag).tolist()) <= {0.0, 1.0}
    assert C_mat is None or C_mat is B_mat or hasattr(C_mat, "diagonal")


def test_forcing_response_mats_conservative_momentum_space_builds_dense_b() -> None:
    mesh, idx, ngp, base_flow = _fake_mesh_and_base_flow()
    ndof = 5 * ngp
    L0 = sp.eye(ndof, format="csc")
    bc_indices = np.array([], dtype=np.int64)
    cfg = {
        "input_region": {"space": "conservative_momentum"},
        "mask_bc_forcing": False,
        "resolvent": {"eigen": {"forcing_norm": "l2"}},
    }

    B_mat, _C_mat = resolvent_map._forcing_response_mats(
        L0, ndof, bc_indices, cfg, mesh=mesh, idx=idx, base_flow=base_flow
    )
    assert B_mat is not None
    # Not a diagonal mask: T_j rows carry nonzero coupling from u_j/v_j/w_j columns.
    B_dense = np.asarray(B_mat.todense())
    T_rows = idx["T_j"]
    u_cols = idx["u_j"]
    assert np.any(B_dense[np.ix_(T_rows, u_cols)] != 0.0)
    # rho_j rows are always zero: conservative-momentum forcing has no density input.
    assert np.all(B_dense[idx["rho_j"], :] == 0.0)


def test_forcing_response_mats_conservative_momentum_invalid_vars_raises() -> None:
    mesh, idx, ngp, base_flow = _fake_mesh_and_base_flow()
    ndof = 5 * ngp
    L0 = sp.eye(ndof, format="csc")
    bc_indices = np.array([], dtype=np.int64)
    cfg = {
        "input_region": {"space": "conservative_momentum", "vars": ["rho"]},
        "mask_bc_forcing": False,
        "resolvent": {"eigen": {"forcing_norm": "l2"}},
    }

    with pytest.raises(ValueError, match="momentum"):
        resolvent_map._forcing_response_mats(
            L0, ndof, bc_indices, cfg, mesh=mesh, idx=idx, base_flow=base_flow
        )


def test_forcing_response_mats_conservative_momentum_requires_base_flow() -> None:
    mesh, idx, ngp, _base_flow = _fake_mesh_and_base_flow()
    ndof = 5 * ngp
    L0 = sp.eye(ndof, format="csc")
    bc_indices = np.array([], dtype=np.int64)
    cfg = {
        "input_region": {"space": "conservative_momentum"},
        "mask_bc_forcing": False,
        "resolvent": {"eigen": {"forcing_norm": "l2"}},
    }

    with pytest.raises(ValueError, match="base flow"):
        resolvent_map._forcing_response_mats(
            L0, ndof, bc_indices, cfg, mesh=mesh, idx=idx, base_flow=None
        )


# ---------------------------------------------------------------------------
# 4. GATE: the map engine is the line walker. The RSVD sweep is a separate
#    1-D tool, so a conservative-momentum B_mat has no sweep gate to trip.
# ---------------------------------------------------------------------------


def test_resolvent_map_walks_the_line_and_never_calls_the_rsvd_sweep() -> None:
    src = (
        Path(__file__).resolve().parents[1] / "src/linstabpy/solvers/router/resolvent_map.py"
    ).read_text(encoding="utf-8")
    assert "sweep_stable_regime" not in src
    assert "LineWalker" in src


def test_conservative_momentum_b_mat_is_built_for_the_walk() -> None:
    mesh, idx, ngp, base_flow = _fake_mesh_and_base_flow()
    ndof = 5 * ngp
    L0 = sp.eye(ndof, format="csc")
    bc_indices = np.array([], dtype=np.int64)
    cfg = {
        "input_region": {"space": "conservative_momentum"},
        "mask_bc_forcing": False,
        "resolvent": {"eigen": {"forcing_norm": "l2"}},
    }

    B_mat, C_mat = resolvent_map._forcing_response_mats(
        L0, ndof, bc_indices, cfg, mesh=mesh, idx=idx, base_flow=base_flow
    )
    filter_fn = None
    filter_ct_fn = None
    # A conservative-momentum region must produce a real forcing matrix for
    # the walk. It must not silently fall back to the unmasked path.
    assert B_mat is not None
    assert B_mat.shape[0] == ndof
    assert B_mat.nnz > 0
    _ = (C_mat, filter_fn, filter_ct_fn)


# ---------------------------------------------------------------------------
# 5. Review fixes: T-row BC exclusion, forcing_norm guard, space-key guards
# ---------------------------------------------------------------------------


def test_excluded_T_row_drops_induced_T_entry_but_keeps_momentum() -> None:
    """A BC row on the T equation must receive no induced-T forcing entry.

    That matches the diagonal-mask convention that forcing never lands on BC
    rows.
    The momentum entry of the same column survives (its row is not a BC row).
    """
    X, Y, idx, ngp = _grid_idx()
    ndof = 5 * ngp
    flow = _base_flow_arrays(ngp, seed=5)
    point_indices = np.arange(ngp, dtype=np.int64)
    bc_T_dof = int(idx["T_j"][0])  # T row of point 0 is a BC row

    B_tilde, _dw, info = build_conservative_momentum_matrix(
        idx,
        ndof,
        point_indices,
        flow["rho_bar"],
        flow["u_bar"],
        flow["v_bar"],
        flow["w_bar"],
        flow["w_int"],
        KAPPA,
        MA,
        excluded_dofs=np.array([bc_T_dof], dtype=np.int64),
    )
    dense = np.asarray(B_tilde.todense())
    # No forcing injects into the excluded T row from any column.
    assert np.all(dense[bc_T_dof, :] == 0.0)
    # The momentum columns of point 0 are still active (their own rows carry 1/rho * w^-1/2).
    for var in ("u", "v", "w"):
        col = int(idx[f"{var}_j"][0])
        assert dense[col, col] != 0.0
        assert col in set(info["momentum_dofs"].tolist())
    # Exclusion must not zero T rows of other points.
    other_T = int(idx["T_j"][1])
    assert np.any(dense[other_T, :] != 0.0)


def test_conservative_momentum_requires_forcing_norm_l2() -> None:
    """The default forcing_norm='chu' would re-weight the whitened coefficients
    and silently produce a mismatched sigma; the builder must refuse to run."""
    mesh, idx, ngp, base_flow = _fake_mesh_and_base_flow()
    ndof = 5 * ngp
    L0 = sp.eye(ndof, format="csc")
    bc_indices = np.array([], dtype=np.int64)
    cfg = {"input_region": {"space": "conservative_momentum"}, "mask_bc_forcing": False}

    with pytest.raises(ValueError, match="forcing_norm"):
        resolvent_map._forcing_response_mats(
            L0, ndof, bc_indices, cfg, mesh=mesh, idx=idx, base_flow=base_flow
        )


def test_output_region_space_is_rejected() -> None:
    mesh, idx, ngp, base_flow = _fake_mesh_and_base_flow()
    ndof = 5 * ngp
    L0 = sp.eye(ndof, format="csc")
    bc_indices = np.array([], dtype=np.int64)
    cfg = {
        "output_region": {"space": "conservative_momentum"},
        "mask_bc_forcing": False,
    }

    with pytest.raises(ValueError, match="output_region"):
        resolvent_map._forcing_response_mats(
            L0, ndof, bc_indices, cfg, mesh=mesh, idx=idx, base_flow=base_flow
        )


def test_unknown_input_space_is_rejected() -> None:
    """A typo like space='conservative' must not fall through to the primitive mask."""
    mesh, idx, ngp, base_flow = _fake_mesh_and_base_flow()
    ndof = 5 * ngp
    L0 = sp.eye(ndof, format="csc")
    bc_indices = np.array([], dtype=np.int64)
    cfg = {"input_region": {"space": "conservative"}, "mask_bc_forcing": False}

    with pytest.raises(ValueError, match="not recognised"):
        resolvent_map._forcing_response_mats(
            L0, ndof, bc_indices, cfg, mesh=mesh, idx=idx, base_flow=base_flow
        )


def test_resolvent_package_lazy_exports_conservative_forcing() -> None:
    import linstabpy.resolvent as resolvent_pkg

    assert "dewhiten_forcing" in resolvent_pkg.__dir__()
    assert resolvent_pkg.dewhiten_forcing is dewhiten_forcing
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(resolvent_pkg, "not_an_export")


def test_validate_conservative_momentum_vars_accepts_a_single_name() -> None:
    assert validate_conservative_momentum_vars("u", name="input_region") == ("u",)
    with pytest.raises(ValueError, match="iterable"):
        validate_conservative_momentum_vars(1, name="input_region")


def test_conservative_coefficients_reject_zero_density_and_unknown_var() -> None:
    ones = np.ones(3)
    with pytest.raises(ValueError, match="rho_bar"):
        conservative_momentum_coefficients(np.array([1.0, 0.0, 1.0]), ones, ones, ones, 1.4, 2.0)
    with pytest.raises(ValueError, match="unknown momentum"):
        conservative_momentum_coefficients(ones, ones, ones, ones, 1.4, 2.0, momentum_vars=("q",))


def test_conservative_matrix_rejects_empty_region_and_length_mismatch() -> None:
    _X, _Y, idx, ngp = _grid_idx()
    ndof = 5 * ngp
    flow = _base_flow_arrays(ngp)
    kwargs = dict(
        idx=idx,
        ndof=ndof,
        rho_bar_active=flow["rho_bar"],
        u_bar_active=flow["u_bar"],
        v_bar_active=flow["v_bar"],
        w_bar_active=flow["w_bar"],
        w_int_active=flow["w_int"],
        kappa=KAPPA,
        ma=MA,
    )
    with pytest.raises(ValueError, match="zero active points"):
        build_conservative_momentum_matrix(point_indices=np.array([], dtype=np.int64), **kwargs)
    with pytest.raises(ValueError, match="u_bar_active"):
        build_conservative_momentum_matrix(
            point_indices=np.arange(ngp, dtype=np.int64),
            **{**kwargs, "u_bar_active": flow["u_bar"][:2]},
        )
    bad_w = flow["w_int"].copy()
    bad_w[0] = 0.0
    with pytest.raises(ValueError, match="w_int"):
        build_conservative_momentum_matrix(
            point_indices=np.arange(ngp, dtype=np.int64),
            **{**kwargs, "w_int_active": bad_w},
        )


def test_conservative_matrix_rejects_missing_index_blocks() -> None:
    _X, _Y, idx, ngp = _grid_idx()
    ndof = 5 * ngp
    flow = _base_flow_arrays(ngp)
    points = np.arange(ngp, dtype=np.int64)
    no_t = {k: v for k, v in idx.items() if k != "T_j"}
    with pytest.raises(ValueError, match="T_j"):
        build_conservative_momentum_matrix(
            no_t,
            ndof,
            points,
            flow["rho_bar"],
            flow["u_bar"],
            flow["v_bar"],
            flow["w_bar"],
            flow["w_int"],
            KAPPA,
            MA,
        )
    short_t = dict(idx)
    short_t["T_j"] = idx["T_j"][:2]
    with pytest.raises(ValueError, match="T_j"):
        build_conservative_momentum_matrix(
            short_t,
            ndof,
            points,
            flow["rho_bar"],
            flow["u_bar"],
            flow["v_bar"],
            flow["w_bar"],
            flow["w_int"],
            KAPPA,
            MA,
        )
    no_u = {k: v for k, v in idx.items() if k != "u_j"}
    with pytest.raises(ValueError, match="u_j"):
        build_conservative_momentum_matrix(
            no_u,
            ndof,
            points,
            flow["rho_bar"],
            flow["u_bar"],
            flow["v_bar"],
            flow["w_bar"],
            flow["w_int"],
            KAPPA,
            MA,
        )
    short_u = dict(idx)
    short_u["u_j"] = idx["u_j"][:2]
    with pytest.raises(ValueError, match="u_j"):
        build_conservative_momentum_matrix(
            short_u,
            ndof,
            points,
            flow["rho_bar"],
            flow["u_bar"],
            flow["v_bar"],
            flow["w_bar"],
            flow["w_int"],
            KAPPA,
            MA,
        )


def test_conservative_matrix_respects_excluded_dofs() -> None:
    _X, _Y, idx, ngp = _grid_idx()
    ndof = 5 * ngp
    flow = _base_flow_arrays(ngp)
    points = np.arange(ngp, dtype=np.int64)
    one_u = int(idx["u_j"][0])
    B, dewhiten, info = build_conservative_momentum_matrix(
        idx,
        ndof,
        points,
        flow["rho_bar"],
        flow["u_bar"],
        flow["v_bar"],
        flow["w_bar"],
        flow["w_int"],
        KAPPA,
        MA,
        excluded_dofs=np.array([one_u], dtype=np.int64),
    )
    assert one_u not in set(info["momentum_dofs"].tolist())
    assert B.shape == (ndof, ndof)
    assert float(dewhiten[one_u]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero forcing dofs"):
        build_conservative_momentum_matrix(
            idx,
            ndof,
            points,
            flow["rho_bar"],
            flow["u_bar"],
            flow["v_bar"],
            flow["w_bar"],
            flow["w_int"],
            KAPPA,
            MA,
            momentum_vars=("u",),
            excluded_dofs=idx["u_j"],
        )
