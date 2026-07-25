# SPDX-License-Identifier: GPL-3.0-only
"""Configuration defaults that more than one layer must agree on.

This module exists so that a default is declared exactly once. The bug it was created
for: `operator_dissipation.apply_to_resolvent` was hardcoded in ``driver/run.py`` as
``False`` while ``apply_to_stability`` was ``True``. A config that enabled dissipation
therefore got a damped eigenvalue problem and a **bare** resolvent -- which runs happily
and returns grid-scale sawtooth (spurious high-beta streaks, gain ~1e6) with nothing in
the log to say so.

These are configuration defaults, not numerics: the shape parameters of the dissipation
operator itself live in ``mesh.sbp.DISSIPATION_DEFAULTS``, which is a frozen math file
under Invariant A and is deliberately not touched from here.
"""

from __future__ import annotations

__all__ = ["DISSIPATION_APPLY_DEFAULTS"]

# Which analysis stages inherit `operator_dissipation` when the config does not say.
# Both are True: enabling dissipation should mean enabling it. A run that wants one stage
# undamped still says so explicitly, and the preflight layer reports the effective value.
DISSIPATION_APPLY_DEFAULTS: dict[str, bool] = {
    "apply_to_stability": True,
    "apply_to_resolvent": True,
}
