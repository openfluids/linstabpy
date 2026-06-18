# SPDX-License-Identifier: GPL-3.0-only
"""PETSc-free MUMPS option constants shared by config and PETSc wiring."""

from __future__ import annotations

MUMPS_ORDERING_TO_ICNTL_7 = {
    "amd": 0,
    "pord": 4,
    "metis": 5,
    "qamd": 6,
    "auto": 7,
}

__all__ = ["MUMPS_ORDERING_TO_ICNTL_7"]
