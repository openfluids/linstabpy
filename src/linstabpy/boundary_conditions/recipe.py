# SPDX-License-Identifier: GPL-3.0-only
"""Declarative boundary-condition recipe applicator.

Reads a list of BC rule dicts (typically from a JSONC config) and
applies them sequentially to the LNSE operator, returning the
combined BC index set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from ..mesh.mesh import Mesh
from .dirichlet import (
    _BORDER_MAP,
    _VARIABLE_MAP,
    _collect_dirichlet_indices,
    _set_diagonal_entries_csc,
    _zero_cols_csc,
    _zero_rows_csr,
    apply_dirichlet,
    apply_dirichlet_petsc,
)
from .neumann import _collect_neumann_indices, apply_neumann, apply_neumann_petsc

if TYPE_CHECKING:  # pragma: no cover - typing-only optional dependency
    from petsc4py import PETSc


@dataclass(frozen=True)
class BCRule:
    """Normalized boundary-condition rule from a JSONC-compatible recipe entry."""

    type: str
    borders: str
    variables: str
    exclude_corners: bool = False
    curvilinear_normals: bool = False

    @classmethod
    def from_mapping(cls, rule: Mapping[str, Any]) -> "BCRule":
        """Normalize one raw recipe mapping without changing legacy semantics."""
        bc_type = str(rule["type"]).strip().lower()
        if bc_type not in {"dirichlet", "neumann"}:
            raise ValueError(f"Unknown BC type {bc_type!r}; expected 'dirichlet' or 'neumann'")
        borders = str(rule["borders"])
        variables = str(rule["variables"])
        _validate_rule_symbols(bc_type, borders, variables)
        return cls(
            type=bc_type,
            borders=borders,
            variables=variables,
            exclude_corners=bool(rule.get("exclude_corners", False)),
            curvilinear_normals=bool(rule.get("curvilinear_normals", False)),
        )


@dataclass(frozen=True)
class BCRecipe:
    """First-class boundary-condition recipe with legacy list/dict compatibility.

    The rule order is part of the numerical contract: later rows own overlaps
    with earlier rows. Dirichlet column-zeroing still records every Dirichlet
    rule that ran, even if a later Neumann rule replaces the same row.
    """

    rules: tuple[BCRule, ...]

    @classmethod
    def from_config(
        cls,
        recipe: "BCRecipe | Sequence[Mapping[str, Any]]",
    ) -> "BCRecipe":
        """Build a recipe from the current JSONC-compatible list of rule dicts."""
        if isinstance(recipe, BCRecipe):
            return recipe
        return cls(tuple(BCRule.from_mapping(rule) for rule in recipe))

    def apply_to(
        self,
        L0: sp.csc_matrix,
        mesh: Mesh,
        idx: dict[str, np.ndarray],
        *,
        penalty: float = 1e4,
    ) -> tuple[sp.csc_matrix, np.ndarray]:
        """Apply this recipe to a SciPy LNSE operator."""
        return _apply_bc_recipe_impl(
            L0,
            mesh,
            idx,
            self,
            penalty=penalty,
            dirichlet_fn=apply_dirichlet,
            neumann_fn=apply_neumann,
        )

    def apply_to_petsc(
        self,
        L0: "PETSc.Mat",
        mesh: Mesh,
        idx: dict[str, np.ndarray],
        *,
        penalty: float = 1e4,
    ) -> tuple["PETSc.Mat", np.ndarray]:
        """Apply this recipe to a PETSc LNSE matrix."""
        return _apply_bc_recipe_impl(
            L0,
            mesh,
            idx,
            self,
            penalty=penalty,
            dirichlet_fn=apply_dirichlet_petsc,
            neumann_fn=apply_neumann_petsc,
        )

    def apply_to_spatial(
        self,
        Lc: sp.csc_matrix,
        R: sp.csc_matrix,
        mesh: Mesh,
        idx: dict[str, np.ndarray],
        *,
        penalty: float = 1e4,
    ) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray]:
        """Apply Dirichlet recipe rows to doubled SciPy spatial EVP matrices."""
        analysis = self.analyze(mesh, idx)
        _require_spatial_dirichlet_recipe(analysis)
        return _apply_spatial_dirichlet_indices(
            Lc,
            R,
            np.asarray(analysis["column_zeroed_dirichlet_indices"], dtype=np.int64),
            mesh.ngp,
            penalty=penalty,
        )

    def apply_to_spatial_petsc(
        self,
        Lc: "PETSc.Mat",
        R: "PETSc.Mat",
        mesh: Mesh,
        idx: dict[str, np.ndarray],
        *,
        penalty: float = 1e4,
    ) -> tuple["PETSc.Mat", "PETSc.Mat", np.ndarray]:
        """Apply Dirichlet recipe rows to doubled PETSc spatial EVP matrices."""
        analysis = self.analyze(mesh, idx)
        _require_spatial_dirichlet_recipe(analysis)
        return _apply_spatial_dirichlet_indices_petsc(
            Lc,
            R,
            np.asarray(analysis["column_zeroed_dirichlet_indices"], dtype=np.int64),
            mesh.ngp,
            penalty=penalty,
        )

    def analyze(self, mesh: Mesh, idx: dict[str, np.ndarray]) -> dict[str, object]:
        """Analyze effective row owners and Dirichlet column-zeroing sets."""
        return _analyze_bc_recipe_impl(mesh, idx, self)


def _validate_rule_symbols(bc_type: str, borders: str, variables: str) -> None:
    """Fail early on malformed border/variable symbols from config recipes."""
    for border in borders:
        if border not in _BORDER_MAP:
            raise ValueError(
                f"Unknown border character '{border}'; expected one of 'l', 'r', 'b', 't', 'm'"
            )
        if bc_type == "neumann" and border == "m":
            raise ValueError("Neumann B.C. are currently only supported for b, t, l and r.")
    for variable in variables:
        if variable not in _VARIABLE_MAP:
            raise ValueError(
                f"Unknown variable character '{variable}'; expected one of 'r', 'u', 'v', 'w', 'T'"
            )


def _apply_bc_recipe_impl(
    L0,
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    recipe: BCRecipe,
    *,
    penalty: float,
    dirichlet_fn,
    neumann_fn,
):
    """Apply a declarative BC recipe using backend-specific BC functions."""
    all_bc: list[np.ndarray] = []

    for rule in recipe.rules:
        bc_type = rule.type
        borders = rule.borders
        variables = rule.variables

        if bc_type == "dirichlet":
            L0, bc_idx = dirichlet_fn(
                L0,
                idx,
                borders,
                variables,
                mesh.ngp,
                penalty=penalty,
            )
        elif bc_type == "neumann":
            L0, bc_idx = neumann_fn(
                L0,
                mesh,
                idx,
                borders,
                variables,
                penalty=penalty,
                exclude_corners=rule.exclude_corners,
                curvilinear_normals=rule.curvilinear_normals,
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
    recipe: BCRecipe | Sequence[Mapping[str, Any]],
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
    return BCRecipe.from_config(recipe).apply_to(L0, mesh, idx, penalty=penalty)


def apply_bc_recipe_petsc(
    L0: "PETSc.Mat",
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    recipe: BCRecipe | Sequence[Mapping[str, Any]],
    penalty: float = 1e4,
) -> tuple["PETSc.Mat", np.ndarray]:
    """Apply a declarative BC recipe directly to a PETSc LNSE matrix."""
    return BCRecipe.from_config(recipe).apply_to_petsc(L0, mesh, idx, penalty=penalty)


def apply_bc_recipe_spatial(
    Lc: sp.csc_matrix,
    R: sp.csc_matrix,
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    recipe: BCRecipe | Sequence[Mapping[str, Any]],
    penalty: float = 1e4,
) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray]:
    """Apply a declarative Dirichlet recipe to doubled SciPy spatial matrices."""
    return BCRecipe.from_config(recipe).apply_to_spatial(Lc, R, mesh, idx, penalty=penalty)


def apply_bc_recipe_spatial_petsc(
    Lc: "PETSc.Mat",
    R: "PETSc.Mat",
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    recipe: BCRecipe | Sequence[Mapping[str, Any]],
    penalty: float = 1e4,
) -> tuple["PETSc.Mat", "PETSc.Mat", np.ndarray]:
    """Apply a declarative Dirichlet recipe to doubled PETSc spatial matrices."""
    return BCRecipe.from_config(recipe).apply_to_spatial_petsc(Lc, R, mesh, idx, penalty=penalty)


def analyze_bc_recipe(
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    recipe: BCRecipe | Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Analyze sequential BC-recipe semantics without modifying a matrix.

    Returns the union of Dirichlet- and Neumann-touched DOFs, plus the
    effective row-owner after sequential application. This is the right view
    for stability EVPs, where later Neumann rules may override earlier
    Dirichlet rows at shared corner DOFs.
    """
    return BCRecipe.from_config(recipe).analyze(mesh, idx)


def _analyze_bc_recipe_impl(
    mesh: Mesh,
    idx: dict[str, np.ndarray],
    recipe: BCRecipe,
) -> dict[str, object]:
    if not recipe.rules:
        empty = np.array([], dtype=np.int64)
        return {
            "rule_reports": [],
            "dirichlet_indices": empty,
            "neumann_indices": empty,
            "effective_dirichlet_indices": empty,
            "effective_neumann_indices": empty,
            "column_zeroed_dirichlet_indices": empty,
            "constrained_variables": tuple(),
            "constrained_variable_blocks": {},
            "overlap_indices": empty,
            "total_bc_indices": empty,
        }

    dirichlet_union: set[int] = set()
    neumann_union: set[int] = set()
    row_owner: dict[int, str] = {}
    row_owner_rule: dict[int, int] = {}
    constrained_variable_set: set[str] = set()
    rule_reports: list[dict[str, object]] = []

    for rule_id, rule in enumerate(recipe.rules):
        bc_type = rule.type
        borders = rule.borders
        variables = rule.variables
        if bc_type == "dirichlet":
            bc_idx = _collect_dirichlet_indices(idx, borders, variables, mesh.ngp)
            dirichlet_union.update(int(val) for val in bc_idx)
        elif bc_type == "neumann":
            bc_idx, _, _ = _collect_neumann_indices(
                idx,
                borders,
                variables,
                mesh.ngp,
                exclude_corners=rule.exclude_corners,
            )
            neumann_union.update(int(val) for val in bc_idx)
        else:
            raise ValueError(f"Unknown BC type {bc_type!r}; expected 'dirichlet' or 'neumann'")

        if np.asarray(bc_idx).size:
            constrained_variable_set.update(variables)

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
                "exclude_corners": bool(rule.exclude_corners),
                "curvilinear_normals": bool(rule.curvilinear_normals),
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
    constrained_variables = tuple(
        variable for variable in _VARIABLE_MAP if variable in constrained_variable_set
    )

    return {
        "rule_reports": rule_reports,
        "dirichlet_indices": np.array(sorted(dirichlet_union), dtype=np.int64),
        "neumann_indices": np.array(sorted(neumann_union), dtype=np.int64),
        "effective_dirichlet_indices": effective_dirichlet,
        "effective_neumann_indices": effective_neumann,
        "column_zeroed_dirichlet_indices": np.array(sorted(dirichlet_union), dtype=np.int64),
        "constrained_variables": constrained_variables,
        "constrained_variable_blocks": {
            variable: _VARIABLE_MAP[variable] for variable in constrained_variables
        },
        "overlap_indices": overlap,
        "total_bc_indices": total_bc,
        "row_owner_rule": row_owner_rule,
    }


def _require_spatial_dirichlet_recipe(analysis: dict[str, object]) -> None:
    """Reject spatial recipes that need a boundary type not implemented there."""
    if any(report["type"] == "neumann" for report in analysis["rule_reports"]):
        raise ValueError("Spatial BC recipes currently support Dirichlet rules only.")


def _spatial_doubled_indices(base_indices: np.ndarray, ngp: int) -> np.ndarray:
    """Lift single-half boundary indices into both halves of the spatial EVP."""
    base_indices = np.unique(np.asarray(base_indices, dtype=np.int64))
    if base_indices.size == 0:
        return np.array([], dtype=np.int64)
    n = ngp * 5
    return np.sort(np.unique(np.concatenate([base_indices, base_indices + n])))


def _apply_spatial_dirichlet_indices(
    Lc: sp.csc_matrix,
    R: sp.csc_matrix,
    base_indices: np.ndarray,
    ngp: int,
    *,
    penalty: float,
) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray]:
    bc_indices = _spatial_doubled_indices(base_indices, ngp)
    if bc_indices.size == 0:
        return Lc, R, np.array([], dtype=np.int64)

    def _zero_rows_cols(mat: sp.csc_matrix) -> sp.csc_matrix:
        mat = mat.tocsr()
        _zero_rows_csr(mat, bc_indices)
        mat = mat.tocsc()
        _zero_cols_csc(mat, bc_indices)
        mat.eliminate_zeros()
        return mat

    Lc = _zero_rows_cols(Lc)
    Lc = _set_diagonal_entries_csc(Lc, bc_indices, penalty)
    Lc.eliminate_zeros()
    R = _zero_rows_cols(R)
    return Lc, R, bc_indices


def _apply_spatial_dirichlet_indices_petsc(
    Lc: "PETSc.Mat",
    R: "PETSc.Mat",
    base_indices: np.ndarray,
    ngp: int,
    *,
    penalty: float,
) -> tuple["PETSc.Mat", "PETSc.Mat", np.ndarray]:
    from petsc4py import PETSc

    bc_indices = _spatial_doubled_indices(base_indices, ngp)
    if bc_indices.size == 0:
        return Lc, R, np.array([], dtype=np.int64)

    if Lc.getSize()[0] != Lc.getSize()[1] or R.getSize()[0] != R.getSize()[1]:
        raise ValueError("Lc and R must be square")
    if Lc.getSize() != R.getSize():
        raise ValueError("Lc and R must have identical shapes")
    if bc_indices.min() < 0 or bc_indices.max() >= Lc.getSize()[0]:
        raise ValueError("Dirichlet indices are out of bounds for the spatial PETSc matrices")

    bc_idx = np.asarray(bc_indices, dtype=np.int32)
    Lc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    R.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    Lc.zeroRowsColumns(bc_idx, diag=complex(penalty))
    R.zeroRowsColumns(bc_idx, diag=0.0)
    Lc.assemble()
    R.assemble()
    return Lc, R, bc_indices
