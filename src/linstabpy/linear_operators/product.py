# SPDX-License-Identifier: GPL-3.0-only
"""Product (composite) linear operator.

Represents :math:`L = L_r \\cdots L_2 L_1` where each :math:`L_i` acts
via one of {apply, apply_adjoint, solve, solve_adjoint}.

Adapted from resolvent4py's ProductLinearOperator (Padovan et al.).
"""

from __future__ import annotations

from collections.abc import Callable

from ..utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()
from petsc4py import PETSc

from .base import LinearOperator

# Map each forward action to its adjoint counterpart (method names).
_ADJOINT_MAP: dict[str, str] = {
    "apply": "apply_adjoint",
    "apply_adjoint": "apply",
    "solve": "solve_adjoint",
    "solve_adjoint": "solve",
}


def _action_name(linop: LinearOperator, action: Callable) -> str:
    """Return the method name string for *action* on *linop*.

    Uses ``__func__`` comparison because each attribute access creates a new
    bound-method object (so ``op.apply is op.apply`` is False in Python).
    """
    action_func = getattr(action, "__func__", action)
    for name in ("apply", "apply_adjoint", "solve", "solve_adjoint"):
        method = getattr(linop, name)
        if getattr(method, "__func__", method) is action_func:
            return name
    raise ValueError(
        f"Action is not one of apply / apply_adjoint / solve / solve_adjoint for '{linop.name}'"
    )


def _output_dims(linop: LinearOperator, action_name: str) -> tuple[int, int]:
    """Return the output (range) dimension of *action_name* on *linop*."""
    if action_name in ("apply", "solve"):
        return linop.dimensions[0]  # range = rows
    return linop.dimensions[1]  # adjoint: range = cols


def _input_dims(linop: LinearOperator, action_name: str) -> tuple[int, int]:
    """Return the input (domain) dimension of *action_name* on *linop*."""
    if action_name in ("apply", "solve"):
        return linop.dimensions[1]  # domain = cols
    return linop.dimensions[0]  # adjoint: domain = rows


class ProductOperator(LinearOperator):
    r"""Composite linear operator :math:`L = L_r \cdots L_2 L_1`.

    Parameters
    ----------
    operators :
        Sub-operators in application order: ``[L1, L2, ..., Lr]``.
    actions :
        Callables for each sub-operator (same order), each one of
        ``{op.apply, op.apply_adjoint, op.solve, op.solve_adjoint}``.
    nblocks :
        Number of harmonic blocks (None for non-block operators).

    Example
    -------
    To represent :math:`L = L_2^{-1} L_1^H`:

    >>> product = ProductOperator(
    ...     [L1, L2],
    ...     [L1.apply_adjoint, L2.solve],
    ... )
    """

    def __init__(
        self,
        operators: list[LinearOperator],
        actions: list[Callable],
        nblocks: int | None | None = None,
    ) -> None:
        if len(operators) < 2:
            raise ValueError("ProductOperator requires at least 2 sub-operators")
        if len(operators) != len(actions):
            raise ValueError(f"len(operators)={len(operators)} != len(actions)={len(actions)}")

        # Store in application order: operators[0] applied first.
        self._ops = list(operators)
        self._action_names = [_action_name(op, act) for op, act in zip(operators, actions)]
        self._actions = list(actions)

        # Derive adjoint action chain (reversed, each action mapped to adjoint).
        self._adj_action_names = [_ADJOINT_MAP[n] for n in reversed(self._action_names)]
        self._adj_ops = list(reversed(self._ops))
        self._adj_actions = [
            getattr(op, name) for op, name in zip(self._adj_ops, self._adj_action_names)
        ]

        # Compute overall dimensions.
        # Domain = input dims of first operator's action.
        # Range = output dims of last operator's action.
        domain = _input_dims(self._ops[0], self._action_names[0])
        range_ = _output_dims(self._ops[-1], self._action_names[-1])

        # Pre-allocate intermediate vectors.
        self._intermediates = []
        for i in range(len(self._ops) - 1):
            dims = _output_dims(self._ops[i], self._action_names[i])
            v = PETSc.Vec().create(comm=operators[0].comm)
            v.setSizes(dims)
            v.setType("standard")
            self._intermediates.append(v)

        self._adj_intermediates = []
        for i in range(len(self._adj_ops) - 1):
            dims = _output_dims(self._adj_ops[i], self._adj_action_names[i])
            v = PETSc.Vec().create(comm=operators[0].comm)
            v.setSizes(dims)
            v.setType("standard")
            self._adj_intermediates.append(v)

        super().__init__(
            operators[0].comm,
            "ProductOperator",
            (range_, domain),
            nblocks,
        )

    def _chain_apply(self, actions, intermediates, x, y):
        """Apply a chain of actions using pre-allocated intermediates."""
        n = len(actions)
        for j in range(n):
            inp = x if j == 0 else intermediates[j - 1]
            out = y if j == n - 1 else intermediates[j]
            actions[j](inp, out)
        return y

    def apply(self, x, y=None):
        if y is None:
            y = self.create_left_vector()
        return self._chain_apply(self._actions, self._intermediates, x, y)

    def apply_adjoint(self, x, y=None):
        if y is None:
            y = self.create_right_vector()
        return self._chain_apply(self._adj_actions, self._adj_intermediates, x, y)

    def destroy(self):
        for v in self._intermediates:
            v.destroy()
        for v in self._adj_intermediates:
            v.destroy()
        for op in self._ops:
            op.destroy()
