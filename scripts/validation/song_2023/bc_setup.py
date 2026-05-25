from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

from linstabpy.mesh import Mesh
from linstabpy.boundary_conditions.dirichlet import apply_dirichlet
from linstabpy.boundary_conditions.neumann import apply_neumann


def my_bc_setup(
    mesh: Mesh,
    idx: Dict[str, np.ndarray],
    L0: sp.csc_matrix,
    penalty: float,
) -> Tuple[sp.csc_matrix, np.ndarray]:
    """
    Boundary-condition recipe matching the Song (2023) MATLAB setup.

    Mesh border convention (see linstabpy.mesh.mesh): ``bi`` = bottom = first
    row (row 0), ``ti`` = top = last row, ``li`` = left = first column (col 0),
    ``ri`` = right = last column. So: inlet = left, outlet = right, wall =
    bottom, free stream = top.

    Recipe:
      - Dirichlet on left+top for r,u,v,w,T (inlet + free stream)
      - Dirichlet on bottom for r,u,v,w (wall)
      - Neumann on bottom for T (adiabatic wall)
      - Neumann on right for r,u,v,w,T (outflow)
    """
    # Dirichlet on left+top ('lt') for rho, u, v, w, T
    L0, idx_dirichlet_lt = apply_dirichlet(
        L0,
        idx,
        borders="lt",
        variables="ruvwT",
        ngp=mesh.ngp,
        penalty=penalty,
    )

    # Dirichlet on bottom ('b') for rho, u, v, w
    L0, idx_dirichlet_b = apply_dirichlet(
        L0,
        idx,
        borders="b",
        variables="ruvw",
        ngp=mesh.ngp,
        penalty=penalty,
    )

    # Neumann on bottom ('b') for temperature only
    L0, idx_neumann_b = apply_neumann(
        L0,
        mesh,
        idx,
        borders="b",
        variables="T",
        penalty=penalty,
    )

    # Neumann on right ('r') for rho, u, v, w, T
    L0, idx_neumann_r = apply_neumann(
        L0,
        mesh,
        idx,
        borders="r",
        variables="ruvwT",
        penalty=penalty,
    )

    bc_all = np.unique(
        np.concatenate((idx_dirichlet_lt, idx_dirichlet_b, idx_neumann_b, idx_neumann_r))
    )
    return L0, bc_all

