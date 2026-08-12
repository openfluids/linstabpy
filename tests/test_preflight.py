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

from linstabpy.config.defaults import DISSIPATION_APPLY_DEFAULTS
from linstabpy.driver.preflight import (
    METRIC_COUPLING_GATE,
    SAWTOOTH_GAIN_SCALE,
    Advisory,
    baseflow_advisories,
    config_advisories,
    format_advisories,
    format_banner,
    format_summary,
    metric_cross_coupling,
    result_advisories,
    translate_solver_error,
)


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
    cfg = _cfg(
        analysis__run_stability=True, numerics__operator_dissipation__apply_to_stability=False
    )
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


def test_no_advisory_depends_on_beta():
    """beta must not trigger a sawtooth warning at any value.

    A BETA-SAWTOOTH-BAND advisory used to fire for beta in 90-135 without effective
    dissipation. It was withdrawn 2026-08-07: the band had never actually been run
    with the bare operator, and when it was (beta = 110, two resolutions) the gain
    came out 1.3098e+02 against 1.3099e+02 damped -- 0.008%. Whether the odd-even
    mode exists is decided by the MESH, not by the spanwise wavenumber, so no
    advisory here may key on beta.

    This is the inverse of the old test on purpose. If someone reintroduces a
    beta-keyed warning, this fails.
    """
    inside = _cfg(numerics__beta=110.0, numerics__operator_dissipation={"enabled": False})
    outside = _cfg(numerics__beta=19.04, numerics__operator_dissipation={"enabled": False})
    assert codes(config_advisories(inside)) == codes(config_advisories(outside))
    assert not any("BETA" in c for c in codes(config_advisories(inside)))
    # Damped, inside the old band: still nothing beta-specific.
    assert not any("BETA" in c for c in codes(config_advisories(_cfg(numerics__beta=110.0))))


def test_metric_cross_coupling_separates_rectilinear_from_curvilinear():
    """Zero for a tensor mesh -- exactly, not approximately -- and O(1) once bent.

    The rectilinear case is the load-bearing one: x depends only on xi and y only on
    eta, so the cross-metrics are identically zero and the odd-even mode has nowhere
    to live. Stretching must not change that, which is why the tensor mesh below is
    strongly graded in y and still expected to read zero.
    """
    xi = np.linspace(0.0, 2.0, 40)
    eta = np.geomspace(1.0e-4, 1.0, 30)  # 4 decades of stretching
    X_rect, Y_rect = np.meshgrid(xi, eta, indexing="ij")
    assert metric_cross_coupling(X_rect, Y_rect) == pytest.approx(0.0, abs=1e-14)
    assert metric_cross_coupling(X_rect, Y_rect) < METRIC_COUPLING_GATE

    # Bend it: the wall rises with x, so x-lines are no longer straight in y.
    Y_curv = Y_rect + 0.3 * np.sin(np.pi * X_rect / 2.0)
    assert metric_cross_coupling(X_rect, Y_curv) > METRIC_COUPLING_GATE

    # A shear that tilts x-lines is caught through the other ratio.
    X_shear = X_rect + 0.5 * Y_rect
    assert metric_cross_coupling(X_shear, Y_rect) > METRIC_COUPLING_GATE


def test_metric_cross_coupling_rejects_a_degenerate_mesh():
    flat = np.zeros((5, 5))
    with pytest.raises(ValueError, match="degenerate"):
        metric_cross_coupling(flat, flat)


@pytest.mark.parametrize(
    "sk,se,expected", [(1, 1, False), (2, 1, True), (1, 3, True), (4, 4, True)]
)
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


def test_band_marking_backend_trap():
    """The full-resolution 'stall': resolvent scaled up, band-marking EVP left behind."""
    cfg = _cfg()
    cfg["solver"] = {"resolvent": {"factor_backend": "strumpack"}}
    assert "BAND-MARKING-BACKEND" in codes(config_advisories(cfg))


@pytest.mark.parametrize(
    "solver,extra,reason",
    [
        (
            {"resolvent": {"factor_backend": "strumpack"}, "eigen": {"factor_backend": "mumps"}},
            {},
            "eigen backend set explicitly",
        ),
        (
            {"factor_backend": "strumpack", "resolvent": {"factor_backend": "strumpack"}},
            {},
            "solver-level backend covers eigen",
        ),
        (
            {"resolvent": {"factor_backend": "strumpack"}},
            {"analysis": {"skip_band_marking": True}},
            "band marking skipped outright",
        ),
        ({}, {}, "no backend set at all -- nothing to mismatch"),
    ],
    ids=["eigen-explicit", "solver-level", "skipped", "no-backend"],
)
def test_band_marking_backend_not_flagged_when_covered(solver, extra, reason):
    cfg = _cfg()
    cfg["solver"] = solver
    for key, value in extra.items():
        cfg.setdefault(key, {}).update(value)
    assert "BAND-MARKING-BACKEND" not in codes(config_advisories(cfg)), reason


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


ADVISORY_PRODUCERS = [
    "config_advisories",
    "baseflow_advisories",
    "result_advisories",
    "translate_solver_error",
]


def test_every_advisory_producer_is_called_from_the_runtime():
    """A guard nothing calls is worse than no guard: it reads as coverage.

    Every test above this line passes whether or not the driver ever invokes these
    functions. That gap is not hypothetical -- `baseflow_advisories` and
    `translate_solver_error` shipped with 37 green tests and zero call sites, so the
    freestream-normalisation gate and the MUMPS error translation never ran in
    production. This test asserts reachability, which unit tests structurally cannot.

    A bare import does not count; the name must appear applied to arguments.
    """
    import pathlib
    import re

    from linstabpy.driver import preflight as _pf

    pkg = pathlib.Path(_pf.__file__).parent.parent
    sources = {
        path: path.read_text(encoding="utf-8")
        for path in pkg.rglob("*.py")
        if path.name != "preflight.py"
    }
    unwired = [
        fn
        for fn in ADVISORY_PRODUCERS
        if not any(re.search(rf"\b{fn}\s*\(", text) for text in sources.values())
    ]
    assert not unwired, (
        "advisory producers defined but never called from the runtime: "
        f"{unwired}. Wire them into driver/run.py (or the relevant stage) or delete "
        "them -- an unreachable guard is dead weight that looks like safety."
    )


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
# The advisory layer against reality: every config this repository ships
# --------------------------------------------------------------------------------------


def _shipped_configs():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    found = sorted(set(root.glob("examples/**/*.yaml")) | set(root.glob("configs/**/*.yaml")))
    return found


def test_advisories_survive_every_shipped_config():
    """Synthetic dicts prove logic; only the real corpus proves it does not explode."""
    from linstabpy.utils.jsonc import load_config

    configs = _shipped_configs()
    assert configs, "SKIPPED-BY-EMPTINESS GUARD: no shipped .yaml found -- wrong root?"
    for path in configs:
        cfg = load_config(str(path))
        result = config_advisories(cfg)
        assert isinstance(result, list), path


def test_no_shipped_config_depends_on_the_old_resolvent_dissipation_default():
    """Pins the blast radius of flipping apply_to_resolvent to True.

    The claim when that default changed was 'no shipped config changes behaviour'.
    That was an analysis; this makes it a test. A config that trips
    DISS-RESOLVENT-OFF is one that asks for dissipation and then withholds it from
    the resolvent -- exactly what the old default did silently.
    """
    from linstabpy.utils.jsonc import load_config

    offenders = [
        path.name
        for path in _shipped_configs()
        if "DISS-RESOLVENT-OFF" in codes(config_advisories(load_config(str(path))))
    ]
    assert not offenders, f"configs running the resolvent on the bare operator: {offenders}"


def test_the_production_template_does_not_ship_a_trap():
    """A file named 'production template' is what a new user copies first.

    It shipped with no dissipation block at all, so the very first production run a
    newcomer made would be sawtooth-contaminated with nothing in the log to say so.
    """
    from linstabpy.utils.jsonc import load_config

    templates = [p for p in _shipped_configs() if "production_case_template" in p.name]
    if not templates:
        pytest.skip("no production_case_template ships in this tree")
    for path in templates:
        found = codes(config_advisories(load_config(str(path))))
        assert "DISS-ABSENT" not in found, f"{path.name} still ships without dissipation"


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
