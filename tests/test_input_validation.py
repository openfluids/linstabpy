# SPDX-License-Identifier: GPL-3.0-only
"""Public-boundary input validation tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from linstabpy.mesh import Mesh
from linstabpy.mesh.sbp import min_points
from linstabpy.operators import build_lnse_operator
from linstabpy.resolvent import analysis as resolvent_module
from linstabpy.stability import adjoint_eigenvalues, eigenvalues, spatial_eigenvalues


def _assert_clean_message(excinfo: pytest.ExceptionInfo[Exception], *names: str) -> None:
    message = str(excinfo.value)
    for name in names:
        assert name in message
    assert "/home/" not in message
    assert "\\" not in message


@pytest.mark.parametrize(
    ("kwargs", "bad_argument"),
    [
        ({"nx": 1}, "nx"),
        ({"ny": 1}, "ny"),
    ],
)
def test_mesh_create_rejects_one_dimensional_grid_at_boundary(kwargs, bad_argument):
    params = dict(xrange=(0.0, 1.0), yrange=(0.0, 1.0), nx=8, ny=8, fd_order=4)
    params.update(kwargs)

    with pytest.raises(ValueError) as excinfo:
        Mesh.create(**params)

    _assert_clean_message(excinfo, bad_argument, "fd_order")


def test_mesh_create_rejects_inconsistent_range_shape_at_boundary():
    with pytest.raises(ValueError) as excinfo:
        Mesh.create(xrange=(0.0, 0.5, 1.0), yrange=(0.0, 1.0), nx=8, ny=8, fd_order=4)

    _assert_clean_message(excinfo, "xrange")


def test_mesh_create_rejects_grid_below_sbp_stencil_minimum():
    fd_order = 6
    required = min_points(fd_order)

    with pytest.raises(ValueError) as excinfo:
        Mesh.create(
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            nx=required - 1,
            ny=required,
            fd_order=fd_order,
        )

    _assert_clean_message(excinfo, "nx", "fd_order", str(required))


def test_unknown_model_string_raises_value_error_listing_allowed_values(
    small_mesh_2d, uniform_baseflow
):
    with pytest.raises(ValueError) as excinfo:
        build_lnse_operator(small_mesh_2d, uniform_baseflow, model="bogus")

    _assert_clean_message(excinfo, "model", "2D", "axy")


@pytest.mark.parametrize(
    ("kwargs", "bad_argument"),
    [
        ({"L0": sp.csr_matrix(np.ones((2, 3), dtype=complex))}, "L0"),
        ({"W_diag": np.ones(3)}, "W_diag"),
        ({"n_modes": 0}, "n_modes"),
        ({"omega": np.inf}, "omega"),
    ],
)
def test_resolvent_analysis_rejects_invalid_inputs_at_boundary(kwargs, bad_argument):
    params = dict(
        L0=sp.eye(2, format="csr", dtype=complex),
        omega=0.1,
        n_modes=1,
        W_diag=np.ones(2),
    )
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)) as excinfo:
        resolvent_module.resolvent_analysis(**params)

    _assert_clean_message(excinfo, bad_argument)


def test_resolvent_analysis_valid_inputs_reach_implementation_unchanged(monkeypatch):
    L0 = sp.eye(2, format="csr", dtype=complex)
    W_diag = np.ones(2)

    def fake_impl(*args, **kwargs):
        assert args[:4] == (L0, 0.1, 1, W_diag)
        return np.ones(1), np.ones((2, 1)), np.ones((2, 1)), None, None

    monkeypatch.setattr(resolvent_module, "_resolvent_analysis_impl", fake_impl)

    gains, responses, forcings = resolvent_module.resolvent_analysis(
        L0=L0,
        omega=0.1,
        n_modes=1,
        W_diag=W_diag,
    )

    assert gains.shape == (1,)
    assert responses.shape == (2, 1)
    assert forcings.shape == (2, 1)


@pytest.mark.parametrize(
    ("kwargs", "bad_argument"),
    [
        ({"L0": sp.csr_matrix(np.ones((2, 3), dtype=complex))}, "L0"),
        ({"W_diag": np.ones(3)}, "W_diag"),
        ({"n_modes": 0}, "n_modes"),
        ({"target": np.nan}, "target"),
    ],
)
def test_eigenvalues_rejects_invalid_inputs_at_boundary(kwargs, bad_argument):
    params = dict(
        L0=sp.eye(2, format="csr", dtype=complex),
        n_modes=1,
        target=0.0,
        W_diag=np.ones(2),
    )
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)) as excinfo:
        eigenvalues(**params)

    _assert_clean_message(excinfo, bad_argument)


def test_eigenvalues_valid_small_input_still_succeeds():
    values, vectors = eigenvalues(
        sp.diags([1.0, 2.0], format="csr", dtype=complex),
        n_modes=1,
        target=1j,
    )

    assert values.shape == (1,)
    assert vectors.shape == (2, 1)


@pytest.mark.parametrize(
    ("kwargs", "bad_argument"),
    [
        ({"W_diag": np.ones(3)}, "W_diag"),
        ({"invW_diag": np.ones(3)}, "invW_diag"),
        ({"n_modes": 0}, "n_modes"),
        ({"target": np.inf}, "target"),
    ],
)
def test_adjoint_eigenvalues_rejects_invalid_inputs_at_boundary(kwargs, bad_argument):
    params = dict(
        L0=sp.eye(2, format="csr", dtype=complex),
        n_modes=1,
        target=0.0,
        W_diag=np.ones(2),
        invW_diag=np.ones(2),
    )
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)) as excinfo:
        adjoint_eigenvalues(**params)

    _assert_clean_message(excinfo, bad_argument)


@pytest.mark.parametrize(
    ("kwargs", "bad_argument"),
    [
        ({"Lc": sp.csr_matrix(np.ones((2, 3), dtype=complex))}, "Lc"),
        ({"R": sp.eye(3, format="csr", dtype=complex)}, "R"),
        ({"n_modes": 0}, "n_modes"),
        ({"omega": np.inf}, "omega"),
        ({"target": np.nan}, "target"),
    ],
)
def test_spatial_eigenvalues_rejects_invalid_inputs_at_boundary(kwargs, bad_argument):
    params = dict(
        Lc=sp.eye(2, format="csr", dtype=complex),
        Lw=sp.eye(2, format="csr", dtype=complex),
        R=sp.eye(2, format="csr", dtype=complex),
        omega=0.1,
        n_modes=1,
        target=0.0,
    )
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)) as excinfo:
        spatial_eigenvalues(**params)

    _assert_clean_message(excinfo, bad_argument)
