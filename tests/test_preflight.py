# SPDX-License-Identifier: GPL-3.0-only
"""Unit tests for the preflight/post-run advisory layer.

These are pure-function tests: no PETSc, no solve, no I/O. That is deliberate -- the
advisory surface exists to be cheap enough that it can be exercised exhaustively on a
laptop, so it cannot rot between cluster runs.

Two properties matter more than any individual check and are tested as invariants:

1. **Advisories never raise.** They run at the top of every job. An advisory that throws
   on an odd config would abort runs that would otherwise have succeeded -- strictly
   worse than the trap it guards.
2. **Every advisory carries a remedy.** The whole point is that a first-time user learns
   what to do, not merely that something is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from linstabpy.driver.preflight import (
    SAWTOOTH_GAIN_SCALE,
    Advisory,
    baseflow_advisories,
    config_advisories,
    format_advisories,
    format_banner,
    format_summary,
    result_advisories,
    translate_solver_error,
)
from linstabpy.config.defaults import DISSIPATION_APPLY_DEFAULTS


def codes(advisories) -> set[str]:
    return {a.code for a in advisories}


def _cfg(**over):
    """A minimal healthy config: resolvent on, dissipation on, stride 1."""
    cfg = {
        "analysis": {"run_stability": False, "run_resolvent": True},
        "numerics": {
            "beta": 19.04,
            "stride_ksi": 1,
            "stride_eta": 1,
            "operator_dissipation": {"enabled": True, "epsilon": 0.02},
        },
    }
    for path, value in over.items():
        node = cfg
        parts = path.split("__")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


# --------------------------------------------------------------------------------------
# The headline trap: dissipation enabled but not reaching the resolvent
# --------------------------------------------------------------------------------------


def test_healthy_config_is_silent():
    """A correct config must produce NO advisories, or the layer becomes noise."""
    assert config_advisories(_cfg()) == []


def test_resolvent_dissipation_off_is_flagged():
    cfg = _cfg(numerics__operator_dissipation__apply_to_resolvent=False)
    assert "DISS-RESOLVENT-OFF" in codes(config_advisories(cfg))


def test_resolvent_dissipation_off_not_flagged_when_resolvent_not_run():
    """No resolvent stage means the switch is irrelevant -- warning here would be noise."""
    cfg = _cfg(
        analysis__run_resolvent=False,
        analysis__run_stability=True,
        numerics__operator_dissipation__apply_to_resolvent=False,
    )
    assert "DISS-RESOLVENT-OFF" not in codes(config_advisories(cfg))


def test_stability_dissipation_off_is_flagged():
    cfg = _cfg(analysis__run_stability=True, numerics__operator_dissipation__apply_to_stability=False)
    assert "DISS-STABILITY-OFF" in codes(config_advisories(cfg))


def test_preflight_uses_the_same_defaults_as_the_driver():
    """Guards the drift that caused the bug: both must read one constant.

    If the driver's default flipped back to False while preflight still assumed True,
    preflight would go quiet exactly when it is most needed.
    """
    assert DISSIPATION_APPLY_DEFAULTS == {
        "apply_to_stability": True,
        "apply_to_resolvent": True,
    }
    # With the switch unset, the shared default applies and nothing is flagged.
    assert "DISS-RESOLVENT-OFF" not in codes(config_advisories(_cfg()))


# --------------------------------------------------------------------------------------
# Remaining config checks
# --------------------------------------------------------------------------------------


def test_missing_dissipation_is_flagged():
    cfg = _cfg(numerics__operator_dissipation={"enabled": False})
    assert "DISS-ABSENT" in codes(config_advisories(cfg))


def test_enabled_with_zero_epsilon_is_flagged():
    cfg = _cfg(numerics__operator_dissipation={"enabled": True, "epsilon": 0.0})
    assert "DISS-EPS-ZERO" in codes(config_advisories(cfg))


def test_sawtooth_beta_band_flagged_only_without_dissipation():
    hot = _cfg(numerics__beta=110.0, numerics__operator_dissipation={"enabled": False})
    assert "BETA-SAWTOOTH-BAND" in codes(config_advisories(hot))
    # Damped run in the same band: the band is not itself an error.
    damped = _cfg(numerics__beta=110.0)
    assert "BETA-SAWTOOTH-BAND" not in codes(config_advisories(damped))
    # Outside the band, undamped: DISS-ABSENT fires but not the band warning.
    outside = _cfg(numerics__beta=19.04, numerics__operator_dissipation={"enabled": False})
    assert "BETA-SAWTOOTH-BAND" not in codes(config_advisories(outside))


@pytest.mark.parametrize("sk,se,expected", [(1, 1, False), (2, 1, True), (1, 3, True), (4, 4, True)])
def test_stride_note(sk, se, expected):
    cfg = _cfg(numerics__stride_ksi=sk, numerics__stride_eta=se)
    assert ("STRIDE-NOT-FINAL" in codes(config_advisories(cfg))) is expected


def test_backend_shadowing_note():
    cfg = _cfg()
    cfg["solver"] = {"factor_backend": "superlu_dist", "linsolve": {"factor_backend": "mumps"}}
    assert "BACKEND-SHADOWED" in codes(config_advisories(cfg))
    # Solver-level alone now resolves, so it is not worth a note.
    cfg2 = _cfg()
    cfg2["solver"] = {"factor_backend": "superlu_dist"}
    assert "BACKEND-SHADOWED" not in codes(config_advisories(cfg2))


# --------------------------------------------------------------------------------------
# Base flow normalisation
# --------------------------------------------------------------------------------------


def test_canonical_baseflow_is_silent():
    ones = np.ones((4, 5))
    assert baseflow_advisories(ones, ones, ones) == []


def test_mis_scaled_baseflow_is_flagged():
    """The convention error that already voided one campaign session."""
    ones = np.ones((4, 5))
    assert "BASEFLOW-NORMALISATION" in codes(baseflow_advisories(ones, ones, ones * 4.6))


def test_baseflow_tolerates_missing_and_degenerate_fields():
    assert baseflow_advisories(None, None, None) == []
    assert baseflow_advisories(np.array([]), None, None) == []
    assert baseflow_advisories(np.array([np.nan, np.nan]), None, None) == []


# --------------------------------------------------------------------------------------
# Post-run sanity
# --------------------------------------------------------------------------------------


def test_off_scale_gain_is_flagged():
    adv = result_advisories(gains=[SAWTOOTH_GAIN_SCALE * 10])
    assert "GAIN-OFF-SCALE" in codes(adv)


def test_physical_gain_is_silent():
    assert result_advisories(gains=[642.4], residuals=[3.5e-9]) == []


def test_off_scale_gain_names_the_bare_operator_when_dissipation_was_off():
    adv = result_advisories(gains=[1.0e6], dissipation_effective=False)
    assert "dissipation was NOT applied" in adv[0].message


def test_high_residual_is_flagged():
    assert "RESIDUAL-HIGH" in codes(result_advisories(residuals=[1.0e-3]))


def test_result_advisories_ignore_non_finite():
    assert result_advisories(gains=[np.nan, np.inf], residuals=[np.nan]) == []


# --------------------------------------------------------------------------------------
# Opaque solver errors
# --------------------------------------------------------------------------------------


def test_mumps_workspace_error_is_translated_from_mapping():
    adv = translate_solver_error({"mumps_infog1": -9, "mumps_infog2": 19859})
    assert adv is not None
    assert "ICNTL(14)" in adv.remedy
    assert "19859" in adv.message


def test_mumps_workspace_error_is_translated_from_raw_text():
    """The real failure arrives as a long context string, not a tidy mapping."""
    text = (
        "SLEPc EPS.setUp failed ... context: ndof=10240, nev=5, factor_solver=mumps, "
        "mumps_icntl7=7, mumps_infog1=-9, mumps_infog2=19859"
    )
    adv = translate_solver_error(text)
    assert adv is not None and adv.code == "MUMPS-INFOG1-9"


def test_unknown_or_successful_codes_translate_to_nothing():
    assert translate_solver_error(None) is None
    assert translate_solver_error({"mumps_infog1": 0}) is None
    assert translate_solver_error({"mumps_infog1": -999}) is None
    assert translate_solver_error("no codes in this string at all") is None


# --------------------------------------------------------------------------------------
# Invariants -- these matter more than any single check
# --------------------------------------------------------------------------------------


MALFORMED = [
    None,
    {},
    {"analysis": None},
    {"numerics": {"operator_dissipation": None}},
    {"numerics": {"beta": "not-a-number", "stride_ksi": None}},
    {"numerics": {"operator_dissipation": {"enabled": "yes", "epsilon": None}}},
    {"solver": "not-a-mapping"},
    {"analysis": {"run_resolvent": True}, "numerics": []},
]


@pytest.mark.parametrize("cfg", MALFORMED)
def test_config_advisories_never_raise(cfg):
    """Preflight runs on every job; it must never be the reason a run dies."""
    result = config_advisories(cfg)
    assert isinstance(result, list)


def test_every_advisory_carries_a_remedy():
    """A warning without a remedy is the thing this layer exists to replace."""
    produced: list[Advisory] = []
    produced += config_advisories(
        _cfg(
            numerics__beta=110.0,
            numerics__stride_ksi=3,
            numerics__operator_dissipation={"enabled": False},
        )
    )
    produced += config_advisories(_cfg(numerics__operator_dissipation__apply_to_resolvent=False))
    produced += baseflow_advisories(np.ones((2, 2)), np.ones((2, 2)) * 3.0, None)
    produced += result_advisories(gains=[1e7], residuals=[1e-2])
    extra = translate_solver_error({"mumps_infog1": -9})
    if extra is not None:
        produced.append(extra)

    assert len(produced) >= 6
    for a in produced:
        assert a.code and a.code.strip(), f"empty code: {a}"
        assert a.message and a.message.strip(), f"empty message: {a.code}"
        assert a.remedy and a.remedy.strip(), f"advisory without a remedy: {a.code}"
        assert a.severity in {"warning", "note"}, f"bad severity: {a.severity}"


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def test_format_advisories_states_none_explicitly():
    """Silence must be positively reported, so an empty section is not read as 'not run'."""
    assert format_advisories([], title="preflight") == "preflight: none"


def test_format_advisories_lists_each():
    rendered = format_advisories(config_advisories(_cfg(numerics__stride_ksi=2)), title="preflight")
    assert "STRIDE-NOT-FINAL" in rendered and "->" in rendered


def test_banner_and_summary_render_all_fields():
    banner = format_banner({"stride": 2, "ndof": 151500, "epsilon (effective)": 0.02})
    for token in ("stride", "ndof", "151500", "epsilon (effective)"):
        assert token in banner
    summary = format_summary({"gain": 642.4}, result_advisories(gains=[1e7]))
    assert "gain" in summary and "GAIN-OFF-SCALE" in summary


def test_banner_handles_empty_fields():
    assert "linstabpy run configuration" in format_banner({})
