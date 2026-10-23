# SPDX-License-Identifier: GPL-3.0-only
"""Configuration defaults that more than one layer must agree on.

Declare each default exactly once.
The mismatch that created this file:
``driver/run.py`` set ``operator_dissipation.apply_to_resolvent`` to
``False`` while ``apply_to_stability`` was ``True``.
A config that enabled dissipation therefore got a damped eigenvalue
problem and a **bare** resolvent.
Two stages disagreed about the operator, with nothing in the log to say so.

How much that costs depends on the mesh, and the mismatch is worth
fixing at any size.
On a mesh whose two directions exchange grid-scale content, the bare
resolvent admits an odd-even (sawtooth) mode.
The damped eigenvalue problem does not.
That needs a *bent* mesh: a rectilinear tensor mesh has identically
zero metric cross-coupling.
On a rectilinear mesh the measured difference is 0.008-1.4% on the gain.
Measure with ``driver.preflight.metric_cross_coupling`` rather than
assuming either case.
An earlier version of this docstring quoted "spurious high-beta streaks,
gain ~1e6"; that signature was never reproduced and is withdrawn.

These are configuration defaults, not numerics.
The shape parameters of the dissipation operator itself live in
``mesh.sbp.DISSIPATION_DEFAULTS``, a frozen math file under Invariant A.
Leave that math file alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

__all__ = [
    "DISSIPATION_APPLY_DEFAULTS",
    "REMOVED_DISSIPATION_KEYS",
    "RESOLVENT_SIGN_CONVENTIONS",
    "stage_dissipation_epsilon",
]

# Which analysis stages inherit `operator_dissipation` when the config does not say.
# Both are True: enabling dissipation should mean enabling it. A run that wants one stage
# undamped still says so explicitly, and the preflight layer reports the effective value.
DISSIPATION_APPLY_DEFAULTS: dict[str, bool] = {
    "apply_to_stability": True,
    "apply_to_resolvent": True,
}

# Top-level spellings of the dissipation strength that the map router used to
# read. The strength now comes only from ``numerics.operator_dissipation``.
REMOVED_DISSIPATION_KEYS: tuple[str, ...] = ("dissipation_epsilon", "eps_resolv")

# Values of ``solver.resolvent_sign_convention``; the first is the default.
# "physical": H = [(gamma - i*omega) I - L0]^-1. "matlab_parity": the LinStab2D
# matrix L0 - (gamma + i*omega) I. See wiki/resolvent-sign-convention.md.
RESOLVENT_SIGN_CONVENTIONS: tuple[str, ...] = ("physical", "matlab_parity")


def stage_dissipation_epsilon(
    operator_dissipation: Mapping[str, Any] | None,
    stage: Literal["stability", "resolvent"],
) -> float:
    """Return the dissipation strength that one analysis stage assembles with.

    Every path that assembles an operator reads the strength here, from the
    ``numerics.operator_dissipation`` block. The strength is ``epsilon`` when
    the block is enabled and applies to the stage, else zero.

    Parameters
    ----------
    operator_dissipation : Mapping or None
        The ``numerics.operator_dissipation`` block. None means no block.
    stage : {"stability", "resolvent"}
        The analysis stage.

    Returns
    -------
    float
        The strength epsilon for the stage.
    """
    od = operator_dissipation if isinstance(operator_dissipation, Mapping) else {}
    if not bool(od.get("enabled", False)):
        return 0.0
    key = f"apply_to_{stage}"
    if not bool(od.get(key, DISSIPATION_APPLY_DEFAULTS[key])):
        return 0.0
    return float(od.get("epsilon", 0.0) or 0.0)
