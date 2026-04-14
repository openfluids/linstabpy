"""Declarative boundary-condition recipe applicator.

Reads a list of BC rule dicts (typically from a JSONC config) and
applies them sequentially to the LNSE operator, returning the
combined BC index set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from ..mesh.mesh import Mesh
from .dirichlet import (
    _collect_dirichlet_indices,
    apply_dirichlet,
    apply_dirichlet_petsc,
)
from .neumann import _collect_neumann_indices, apply_neumann, apply_neumann_petsc

if TYPE_CHECKING:  # pragma: no cover - typing-only optional dependency
    from petsc4py import PETSc


def _apply_bc_recipe_impl(
    L0,
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    recipe: list[dict],
    *,
    penalty: float,
    dirichlet_fn,
    neumann_fn,
):
    """Apply a declarative BC recipe using backend-specific BC functions."""
    all_bc: list[np.ndarray] = []

    for rule in recipe:
        bc_type = rule["type"].lower()
        borders = rule["borders"]
        variables = rule["variables"]
        exclude_corners = bool(rule.get("exclude_corners", False))

        if bc_type == "dirichlet":
            L0, bc_idx = dirichlet_fn(
                L0, idx, borders, variables, mesh.ngp, penalty=penalty,
            )
        elif bc_type == "neumann":
            L0, bc_idx = neumann_fn(
                L0, mesh, idx, borders, variables, penalty=penalty, exclude_corners=exclude_corners,
            )
        else:
            raise ValueError(f"Unknown BC type {bc_type!r}; expected 'dirichlet' or 'neumann'")

        all_bc.append(bc_idx)

    if all_bc:
        bc_indices = np.unique(np.concatenate(all_bc))
    else:
        bc_indices = np.array([], dtype=np.int64)

    return L0, bc_indices


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
    return _apply_bc_recipe_impl(
        L0,
        mesh,
        idx,
        recipe,
        penalty=penalty,
        dirichlet_fn=apply_dirichlet,
        neumann_fn=apply_neumann,
    )


def apply_bc_recipe_petsc(
    L0: "PETSc.Mat",
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    recipe: list[dict],
    penalty: float = 1e4,
) -> tuple["PETSc.Mat", np.ndarray]:
    """Apply a declarative BC recipe directly to a PETSc LNSE matrix."""
    return _apply_bc_recipe_impl(
        L0,
        mesh,
        idx,
        recipe,
        penalty=penalty,
        dirichlet_fn=apply_dirichlet_petsc,
        neumann_fn=apply_neumann_petsc,
    )


def analyze_bc_recipe(
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    recipe: list[dict],
) -> dict[str, object]:
    """Analyze sequential BC-recipe semantics without modifying a matrix.

    Returns the union of Dirichlet- and Neumann-touched DOFs, plus the
    effective row-owner after sequential application. This is the right view
    for stability EVPs, where later Neumann rules may override earlier
    Dirichlet rows at shared corner DOFs.
    """
    if not recipe:
        empty = np.array([], dtype=np.int64)
        return {
            "rule_reports": [],
            "dirichlet_indices": empty,
            "neumann_indices": empty,
            "effective_dirichlet_indices": empty,
            "effective_neumann_indices": empty,
            "column_zeroed_dirichlet_indices": empty,
            "overlap_indices": empty,
            "total_bc_indices": empty,
        }

    dirichlet_union: set[int] = set()
    neumann_union: set[int] = set()
    row_owner: dict[int, str] = {}
    row_owner_rule: dict[int, int] = {}
    rule_reports: list[dict[str, object]] = []

    for rule_id, rule in enumerate(recipe):
        bc_type = str(rule["type"]).strip().lower()
        borders = str(rule["borders"])
        variables = str(rule["variables"])
        exclude_corners = bool(rule.get("exclude_corners", False))
        if bc_type == "dirichlet":
            bc_idx = _collect_dirichlet_indices(idx, borders, variables, mesh.ngp)
            dirichlet_union.update(int(val) for val in bc_idx)
        elif bc_type == "neumann":
            bc_idx, _, _ = _collect_neumann_indices(
                idx, borders, variables, mesh.ngp, exclude_corners=exclude_corners,
            )
            neumann_union.update(int(val) for val in bc_idx)
        else:
            raise ValueError(f"Unknown BC type {bc_type!r}; expected 'dirichlet' or 'neumann'")

        overlap_prev = np.array(
            [int(val) for val in bc_idx if int(val) in row_owner],
            dtype=np.int64,
        )
        cross_type_overlap = np.array(
            [
                int(val)
                for val in bc_idx
                if int(val) in row_owner and row_owner[int(val)] != bc_type
            ],
            dtype=np.int64,
        )
        for val in bc_idx:
            key = int(val)
            row_owner[key] = bc_type
            row_owner_rule[key] = int(rule_id)
        rule_reports.append(
            {
                "rule_id": int(rule_id),
                "type": bc_type,
                "borders": borders,
                "variables": variables,
                "exclude_corners": bool(exclude_corners),
                "count": int(np.asarray(bc_idx).size),
                "indices": np.asarray(bc_idx, dtype=np.int64),
                "overlap_previous": np.unique(overlap_prev),
                "cross_type_overlap": np.unique(cross_type_overlap),
            }
        )

    total_bc = np.array(sorted(set(dirichlet_union) | set(neumann_union)), dtype=np.int64)
    effective_dirichlet = np.array(
        sorted(dof for dof, owner in row_owner.items() if owner == "dirichlet"),
        dtype=np.int64,
    )
    effective_neumann = np.array(
        sorted(dof for dof, owner in row_owner.items() if owner == "neumann"),
        dtype=np.int64,
    )
    overlap = np.array(sorted(set(dirichlet_union) & set(neumann_union)), dtype=np.int64)

    return {
        "rule_reports": rule_reports,
        "dirichlet_indices": np.array(sorted(dirichlet_union), dtype=np.int64),
        "neumann_indices": np.array(sorted(neumann_union), dtype=np.int64),
        "effective_dirichlet_indices": effective_dirichlet,
        "effective_neumann_indices": effective_neumann,
        "column_zeroed_dirichlet_indices": np.array(sorted(dirichlet_union), dtype=np.int64),
        "overlap_indices": overlap,
        "total_bc_indices": total_bc,
        "row_owner_rule": row_owner_rule,
    }
