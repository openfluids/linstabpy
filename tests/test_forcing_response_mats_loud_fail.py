# SPDX-License-Identifier: GPL-3.0-only
"""`_forcing_response_mats` must fail loudly when it cannot honour a region key.

Kept in its own module (no `linstabpy.driver.run` import) so it collects and
runs without the optional MPI/PETSc stack, unlike
tests/test_driver_resolvent_region.py.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import scipy.sparse as sp

resolvent_map = importlib.import_module("linstabpy.solvers.router.resolvent_map")


def test_forcing_response_mats_raises_loudly_without_mesh_idx() -> None:
    """(c): a region key on a path that cannot honour it (no mesh/idx) raises."""
    ndof = 10
    L0 = sp.eye(ndof, format="csc")
    bc_indices = np.array([], dtype=np.int64)
    cfg = {"input_region": {"vars": ["u"]}}

    with pytest.raises(ValueError, match="input_region/output_region masks require"):
        resolvent_map._forcing_response_mats(L0, ndof, bc_indices, cfg, mesh=None, idx=None)
