# SPDX-License-Identifier: GPL-3.0-only
"""Linear algebra seams for reusable factorizations."""

from .backends import Backend, available, ensure, select
from .factorize import Factorization, factorize
from .shifted import shifted_operator

__all__ = [
    "Backend",
    "Factorization",
    "available",
    "ensure",
    "factorize",
    "select",
    "shifted_operator",
]
