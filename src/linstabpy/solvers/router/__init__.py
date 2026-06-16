# SPDX-License-Identifier: GPL-3.0-only
"""Resolvent sweep routing helpers."""

from . import band_marking, certification, recycling
from .band_marking import BandMarks, label_band, mark_unstable_band
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
from .ram_autoselect import OperatingPoint, SymbolicProbeResult, ram_autoselect

_LAZY_EXPORTS = {
    "SweepResult": "linstabpy.solvers.router.stable_sweep",
    "stable_sweep": "linstabpy.solvers.router.stable_sweep",
    "sweep_stable_regime": "linstabpy.solvers.router.stable_sweep",
}

__all__ = [
    "BandMarks",
    "CertificationReport",
    "OperatingPoint",
    "CheckResult",
    "SweepResult",
    "SymbolicProbeResult",
    "band_marking",
    "certification",
    "stable_sweep",
    "certified_rsvd",
    "check_adjoint_of_approx_factor",
    "check_deterministic_linearity",
    "check_forward_adjoint_consistency",
    "check_multi_seed_rsvd",
    "check_tolerance_tied_to_sigma_accuracy",
    "derive_solve_tolerance",
    "label_band",
    "mark_unstable_band",
    "ram_autoselect",
    "randomized_resolvent_svd",
    "recycling",
    "sweep_stable_regime",
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
