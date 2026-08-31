# SPDX-License-Identifier: GPL-3.0-only
"""Resolvent sweep routing helpers."""

from . import anchors, band_marking, break_even, certification, recycling
from .anchors import AnchorTriggerRecord, IterProbeResult, choose_anchors_adaptive
from .band_marking import BandMarks, label_band, mark_unstable_band
from .break_even import BreakEvenProbeResult, RoutingDecision, break_even_decision
from .certification import (
    CertificationReport,
    CheckResult,
    certified_rsvd,
    check_adjoint_of_approx_factor,
    check_deterministic_linearity,
    check_forward_adjoint_consistency,
    check_multi_seed_rsvd,
    check_tolerance_tied_to_sigma_accuracy,
    derive_solve_tolerance,
    randomized_resolvent_svd,
)
from .prescribed_map import PrescribedMapResult, prescribed_map
from .ram_autoselect import OperatingPoint, SymbolicProbeResult, ram_autoselect
from .resolvent_map import RouterResult, resolvent_map

_LAZY_EXPORTS = {
    "RouterResult": "linstabpy.solvers.router.resolvent_map",
    "SweepResult": "linstabpy.solvers.router.stable_sweep",
    "resolvent_map": "linstabpy.solvers.router.resolvent_map",
    "stable_sweep": "linstabpy.solvers.router.stable_sweep",
    "sweep_stable_regime": "linstabpy.solvers.router.stable_sweep",
    "PrescribedMapResult": "linstabpy.solvers.router.prescribed_map",
    "prescribed_map": "linstabpy.solvers.router.prescribed_map",
}

__all__ = [
    "AnchorTriggerRecord",
    "BandMarks",
    "BreakEvenProbeResult",
    "CertificationReport",
    "OperatingPoint",
    "CheckResult",
    "IterProbeResult",
    "RoutingDecision",
    "RouterResult",
    "SweepResult",
    "SymbolicProbeResult",
    "anchors",
    "band_marking",
    "break_even",
    "break_even_decision",
    "certification",
    "stable_sweep",
    "certified_rsvd",
    "check_adjoint_of_approx_factor",
    "check_deterministic_linearity",
    "check_forward_adjoint_consistency",
    "check_multi_seed_rsvd",
    "check_tolerance_tied_to_sigma_accuracy",
    "choose_anchors_adaptive",
    "derive_solve_tolerance",
    "label_band",
    "mark_unstable_band",
    "ram_autoselect",
    "randomized_resolvent_svd",
    "recycling",
    "resolvent_map",
    "sweep_stable_regime",
    "PrescribedMapResult",
    "prescribed_map",
]


def __getattr__(name: str):  # noqa: ANN201
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(_LAZY_EXPORTS[name])
        attr = module if name == "stable_sweep" else getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
