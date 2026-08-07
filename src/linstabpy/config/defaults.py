# SPDX-License-Identifier: GPL-3.0-only
"""Configuration defaults that more than one layer must agree on.

This module exists so that a default is declared exactly once. The bug it was created
for: `operator_dissipation.apply_to_resolvent` was hardcoded in ``driver/run.py`` as
``False`` while ``apply_to_stability`` was ``True``. A config that enabled dissipation
therefore got a damped eigenvalue problem and a **bare** resolvent -- two stages
disagreeing about the operator, with nothing in the log to say so.

How much that costs depends on the mesh, and the bug is worth fixing at any size. On a
mesh whose two directions exchange grid-scale content -- which needs a *bent* mesh, since
a rectilinear tensor mesh has identically zero metric cross-coupling -- the bare resolvent
admits an odd-even (sawtooth) mode that the damped eigenvalue problem does not. On a
rectilinear mesh the measured difference is 0.008-1.4% on the gain. Measure with
``driver.preflight.metric_cross_coupling`` rather than assuming either case. An earlier
version of this docstring quoted "spurious high-beta streaks, gain ~1e6"; that signature
was never reproduced and is withdrawn.

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
