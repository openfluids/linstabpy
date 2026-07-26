# SPDX-License-Identifier: GPL-3.0-only
"""Opt-in gating for parity tests that need MATLAB-derived reference data.

Policy
------
MATLAB reference data is **not committed to this repository** — not the raw
``.mat`` exports, not the derived ``.npz`` bundles. Those files belong to the
upstream MATLAB codes and to the authors who produced them.

The consequence used to be silent: the tests that compare against that data
called ``pytest.skip`` when it was absent, so a default suite run reported them
as skipped. On any machine without the reference data — which is every machine
except the two the data lives on — that is a permanent, invisible hole. A run
saying "1361 passed, 13 skipped" reads as near-total coverage; it does not say
that seven of those skips can never be satisfied by anything the repository
contains.

So these tests are now **opt-in**, and requested explicitly:

    LINSTABPY_MATLAB_PARITY=1 pytest -m matlab_parity

Two properties follow, and both matter:

1. **The default suite makes no claim it cannot back.** MATLAB-parity tests are
   deselected, not skipped, so they do not pad the skip count. Every remaining
   skip in a default run is an absent *optional backend* (STRUMPACK, CUDA,
   superlu_dist), which is a true statement about the machine.
2. **Asking for them means they must run.** When ``LINSTABPY_MATLAB_PARITY=1``
   is set, a missing reference is a **failure**, not a skip. You asked for the
   comparison; silence is the wrong answer. Skipping on request is how a missing
   oracle turns into a green suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Environment switch. Set to "1" to demand the MATLAB comparisons actually run.
ENV_VAR = "LINSTABPY_MATLAB_PARITY"

#: Apply to any test that compares against MATLAB-derived reference data:
#:
#:     from tests._matlab_parity import matlab_parity
#:
#:     @matlab_parity
#:     def test_something_matches_matlab(): ...
matlab_parity = pytest.mark.matlab_parity


def requested() -> bool:
    """True when the caller has explicitly asked for MATLAB parity checks."""
    return os.environ.get(ENV_VAR) == "1"


def require_matlab_reference(*paths: Path, description: str) -> None:
    """Demand the given reference paths exist.

    Fails loudly when the comparison was requested and the data is not there,
    naming every missing path. Falls back to a skip only when the tests were not
    requested at all, which normally cannot happen because the marker is
    deselected by default -- it is a guard for someone running a single test by
    node id.
    """
    missing = [str(path) for path in paths if not Path(path).exists()]
    if not missing:
        return

    detail = "; ".join(missing)
    if requested():
        pytest.fail(
            f"{ENV_VAR}=1 requested the MATLAB parity check, but {description} is "
            f"missing: {detail}. This is a failure rather than a skip on purpose: a "
            f"requested comparison that quietly does not run is indistinguishable "
            f"from one that passed. Supply the reference data or drop {ENV_VAR}."
        )
    pytest.skip(
        f"SKIPPED, {description} absent and {ENV_VAR} is not set -> {detail}. "
        f"Run with {ENV_VAR}=1 -m matlab_parity to require this comparison."
    )
