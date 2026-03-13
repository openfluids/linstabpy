"""Declarative boundary-condition recipe applicator.

Reads a list of BC rule dicts (typically from a JSONC config) and
applies them sequentially to the LNSE operator, returning the
combined BC index set.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..mesh.mesh import Mesh
from .dirichlet import apply_dirichlet
from .neumann import apply_neumann


def apply_bc_recipe(
    L0: sp.csc_matrix,
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    recipe: list[dict],
    penalty: float = 1e4,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Apply a declarative BC recipe from a JSONC config.

    Parameters
    ----------
    L0 : csc_matrix
        Assembled LNSE operator (modified in place).
    mesh : Mesh
        Computational mesh (needed for Neumann normal-derivative rows).
    idx : dict
        Variable-block index dict from ``build_lnse_operator``.
    recipe : list of dict
        Each entry has keys ``"type"``, ``"borders"``, ``"variables"``.
        Example::

            [
                {"type": "dirichlet", "borders": "lt",  "variables": "ruvwT"},
                {"type": "neumann",   "borders": "b",   "variables": "T"},
            ]

    penalty : float
        Penalty parameter for all BC rows.

    Returns
    -------
    L0 : csc_matrix
        Operator with BCs applied.
    bc_indices : ndarray
        Sorted unique DOF indices affected by the full recipe.
    """
    all_bc: list[np.ndarray] = []

    for rule in recipe:
        bc_type = rule["type"].lower()
        borders = rule["borders"]
        variables = rule["variables"]

        if bc_type == "dirichlet":
            L0, bc_idx = apply_dirichlet(
                L0, idx, borders, variables, mesh.ngp, penalty=penalty,
            )
        elif bc_type == "neumann":
            L0, bc_idx = apply_neumann(
                L0, mesh, idx, borders, variables, penalty=penalty,
            )
        else:
            raise ValueError(f"Unknown BC type {bc_type!r}; expected 'dirichlet' or 'neumann'")

        all_bc.append(bc_idx)

    if all_bc:
        bc_indices = np.unique(np.concatenate(all_bc))
    else:
        bc_indices = np.array([], dtype=np.int64)

    return L0, bc_indices
