# SPDX-License-Identifier: GPL-3.0-only
"""Preflight and post-run advisories: catch the configurations that run but lie.

Every check here exists because the configuration in question *runs to completion and
produces plausible numbers*. Nothing below is a schema error -- ``config.validate_config``
already rejects unknown keys and bad types. These are the semantic traps: settings that
are individually legal, jointly meaningless, and silent.

Design rules, deliberately narrow:

* **Advisories never block.** A false positive that aborts a twelve-hour job is worse
  than the trap it guards. The driver logs these and continues; only pre-existing hard
  errors stay hard.
* **Pure functions over plain data.** Every check takes a config dict or a few arrays and
  returns advisories. No PETSc, no solve, no I/O -- so the whole surface is unit-testable
  in milliseconds on a laptop.
* **Each advisory carries a remedy.** "Warning: dissipation disabled" helps nobody; the
  message must say what to set and where.

The intent is that a first-time user reading the log learns what an experienced user of
this code knows, without having to be told.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from linstabpy.config.defaults import DISSIPATION_APPLY_DEFAULTS

__all__ = [
    "Advisory",
    "baseflow_advisories",
    "config_advisories",
    "format_advisories",
    "format_banner",
    "format_summary",
    "result_advisories",
    "translate_solver_error",
]

# The spanwise band where grid-scale sawtooth dominates on this operator family.
# Documented from repeated SWBLI runs: inside it, an under-damped operator produces
# near-pole streaks that look like a physical high-beta response.
SAWTOOTH_BETA_BAND = (90.0, 135.0)
# A leading Chu-norm gain above this is not a physical amplification on the cases this
# code targets; it is the signature of an under-damped odd-even mode.
SAWTOOTH_GAIN_SCALE = 1.0e5
# Freestream normalisation: rho = u = T = 1. A base flow that misses this by more than
# the tolerance has been scaled with the wrong convention, which silently invalidates
# every eigenvalue and gain computed from it.
FREESTREAM_TOL = 1.0e-2


@dataclass(frozen=True)
class Advisory:
    """One actionable finding. ``code`` is stable and greppable; ``remedy`` is required."""

    code: str
    severity: str  # "warning" | "note"
    message: str
    remedy: str

    def render(self) -> str:
        tag = "WARNING" if self.severity == "warning" else "note"
        return f"  [{tag}] {self.code}: {self.message}\n           -> {self.remedy}"


def _get(cfg: Mapping[str, Any] | None, path: str, default: Any = None) -> Any:
    """Fetch a dotted path from a nested mapping, tolerating missing intermediate dicts."""
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def config_advisories(config: Mapping[str, Any] | None) -> list[Advisory]:
    """Semantic checks on a loaded config, before anything is assembled."""
    out: list[Advisory] = []
    if not isinstance(config, Mapping):
        return out

    run_stab = bool(_get(config, "analysis.run_stability", False))
    run_res = bool(_get(config, "analysis.run_resolvent", False))
    od = _get(config, "numerics.operator_dissipation", {})
    od = od if isinstance(od, Mapping) else {}
    od_enabled = bool(od.get("enabled", False))
    od_eps = float(od.get("epsilon", 0.0) or 0.0)
    # Read the SAME constant the driver reads, so this can never advise about a default
    # the driver no longer uses.
    a2s = bool(od.get("apply_to_stability", DISSIPATION_APPLY_DEFAULTS["apply_to_stability"]))
    a2r = bool(od.get("apply_to_resolvent", DISSIPATION_APPLY_DEFAULTS["apply_to_resolvent"]))

    if run_res and od_enabled and not a2r:
        out.append(
            Advisory(
                "DISS-RESOLVENT-OFF",
                "warning",
                "dissipation is enabled but apply_to_resolvent is false, so the resolvent "
                "runs on the BARE operator while the eigenvalue problem does not",
                "set numerics.operator_dissipation.apply_to_resolvent = true, or accept "
                "grid-scale sawtooth in the gain (spurious high-beta streaks, gain ~1e6)",
            )
        )
    if run_stab and od_enabled and not a2s:
        out.append(
            Advisory(
                "DISS-STABILITY-OFF",
                "warning",
                "dissipation is enabled but apply_to_stability is false, so the eigenvalue "
                "problem runs on the BARE operator",
                "set numerics.operator_dissipation.apply_to_stability = true unless you are "
                "deliberately measuring the undamped spectrum",
            )
        )
    if (run_stab or run_res) and not od_enabled:
        out.append(
            Advisory(
                "DISS-ABSENT",
                "warning",
                "no artificial dissipation: the bare finite-difference operator admits an "
                "odd-even (sawtooth) mode that contaminates eigenvalues and gains",
                "set numerics.operator_dissipation.enabled = true and tune epsilon AT THIS "
                "RESOLUTION; a value tuned on a coarser grid does not transfer",
            )
        )
    if od_enabled and od_eps == 0.0:
        out.append(
            Advisory(
                "DISS-EPS-ZERO",
                "warning",
                "dissipation is enabled but epsilon = 0, which assembles the damping "
                "operator and then applies none of it",
                "set a nonzero numerics.operator_dissipation.epsilon, or set enabled = false "
                "so the artifact records honestly that the run was undamped",
            )
        )

    beta = _get(config, "numerics.beta", None)
    if isinstance(beta, (int, float)) and not isinstance(beta, bool):
        lo, hi = SAWTOOTH_BETA_BAND
        if lo <= abs(float(beta)) <= hi and (not od_enabled or od_eps <= 0.0):
            out.append(
                Advisory(
                    "BETA-SAWTOOTH-BAND",
                    "warning",
                    f"beta = {beta} lies in the {lo:g}-{hi:g} band where grid-scale sawtooth "
                    "dominates, and this run has no effective dissipation",
                    "enable dissipation and re-tune epsilon before interpreting any structure "
                    "found at this spanwise wavenumber",
                )
            )

    sk = _get(config, "numerics.stride_ksi", 1)
    se = _get(config, "numerics.stride_eta", 1)
    try:
        strided = int(sk) > 1 or int(se) > 1
    except (TypeError, ValueError):
        strided = False
    if strided:
        out.append(
            Advisory(
                "STRIDE-NOT-FINAL",
                "note",
                f"strided mesh (stride_ksi={sk}, stride_eta={se}): this is a reduced-resolution "
                "run, valid as ladder/parity/cost evidence but not as a final result",
                "state in the run record why this resolution was chosen, and produce the "
                "publishable number at stride 1",
            )
        )

    solver = _get(config, "solver", {})
    if isinstance(solver, Mapping) and "factor_backend" in solver:
        shadowing = [
            name
            for name in ("linsolve", "eigen", "resolvent")
            if isinstance(solver.get(name), Mapping) and "factor_backend" in solver[name]
        ]
        if shadowing:
            out.append(
                Advisory(
                    "BACKEND-SHADOWED",
                    "note",
                    f"solver.factor_backend is set but overridden for: {', '.join(shadowing)}",
                    "remove the per-stage factor_backend, or drop the solver-level one, so the "
                    "run has a single obvious backend",
                )
            )

    return out


def baseflow_advisories(
    rho: np.ndarray | None,
    u: np.ndarray | None,
    temperature: np.ndarray | None,
    *,
    tol: float = FREESTREAM_TOL,
) -> list[Advisory]:
    """Check the freestream normalisation convention: rho = u = T = 1.

    A base flow scaled with a different convention runs perfectly and produces entirely
    wrong physics -- this has already invalidated a full campaign session once. The check
    uses the field maximum rather than a corner value so it does not depend on knowing
    where the freestream sits in the array.
    """
    out: list[Advisory] = []
    fields = (("rho", rho), ("u", u), ("T", temperature))
    offenders = []
    for name, arr in fields:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        if a.size == 0 or not np.isfinite(a).any():
            continue
        peak = float(np.nanmax(np.abs(a)))
        if abs(peak - 1.0) > tol:
            offenders.append(f"max|{name}| = {peak:.6g}")
    if offenders:
        out.append(
            Advisory(
                "BASEFLOW-NORMALISATION",
                "warning",
                "base flow does not look normalised to the canonical freestream "
                f"(rho = u = T = 1): {'; '.join(offenders)}",
                "rescale to velocity/U_inf and T/T_inf (NOT /c_inf, NOT T*Ma^2); a "
                "mis-scaled base flow invalidates every eigenvalue and gain from this run",
            )
        )
    return out


def result_advisories(
    *,
    gains: Sequence[float] | np.ndarray | None = None,
    residuals: Sequence[float] | np.ndarray | None = None,
    residual_tol: float = 1.0e-6,
    dissipation_effective: bool | None = None,
) -> list[Advisory]:
    """Checks on what the run actually produced."""
    out: list[Advisory] = []

    if gains is not None:
        g = np.asarray(gains, dtype=float).ravel()
        g = g[np.isfinite(g)]
        if g.size and float(np.max(g)) > SAWTOOTH_GAIN_SCALE:
            peak = float(np.max(g))
            out.append(
                Advisory(
                    "GAIN-OFF-SCALE",
                    "warning",
                    f"leading gain {peak:.4g} exceeds {SAWTOOTH_GAIN_SCALE:g}, which on these "
                    "cases indicates an under-damped sawtooth mode rather than amplification",
                    "re-tune the dissipation epsilon AT THIS RESOLUTION and re-run before "
                    "interpreting the result as physics",
                )
            )
            if dissipation_effective is False:
                out[-1] = Advisory(
                    out[-1].code,
                    "warning",
                    out[-1].message + " -- and dissipation was NOT applied to this stage",
                    "this is almost certainly the bare-operator trap: enable dissipation for "
                    "the stage that produced this number",
                )

    if residuals is not None:
        r = np.asarray(residuals, dtype=float).ravel()
        r = r[np.isfinite(r)]
        if r.size and float(np.max(r)) > residual_tol:
            out.append(
                Advisory(
                    "RESIDUAL-HIGH",
                    "warning",
                    f"worst postcheck residual {float(np.max(r)):.3e} exceeds {residual_tol:.0e}",
                    "the returned mode does not satisfy the operator to tolerance; treat the "
                    "value as unconverged rather than as a result",
                )
            )

    return out


# MUMPS INFOG(1) codes seen in practice on this workload. The raw SLEPc failure is an
# opaque "EPS.setUp failed", which sent one triage down the wrong path entirely.
_MUMPS_INFOG1 = {
    -9: (
        "MUMPS internal work array too small",
        "raise the MUMPS workspace: set ICNTL(14) higher (e.g. -mat_mumps_icntl_14 40 or "
        "more). INFOG(2) reports how much was needed.",
    ),
    -8: (
        "MUMPS integer work array too small",
        "raise ICNTL(14); if it persists, try a different ordering (ICNTL(7)).",
    ),
    -13: (
        "MUMPS allocation failure (out of memory)",
        "reduce the problem size, add ranks, or lower ICNTL(14); check peak RSS against "
        "available memory.",
    ),
    -10: (
        "matrix is numerically singular",
        "check the boundary-condition penalty and the shift: a shift exactly on an "
        "eigenvalue makes the shifted operator singular.",
    ),
}


def translate_solver_error(context: Mapping[str, Any] | str | None) -> Advisory | None:
    """Turn an opaque factorization failure into an actionable message.

    Accepts either the driver's context mapping or the raw error text, because the useful
    code (``mumps_infog1``) appears in both.
    """
    if context is None:
        return None
    if isinstance(context, Mapping):
        code = context.get("mumps_infog1")
        need = context.get("mumps_infog2")
    else:
        text = str(context)
        code = need = None
        for key, target in (("mumps_infog1=", "code"), ("mumps_infog2=", "need")):
            if key in text:
                frag = text.split(key, 1)[1].split(",")[0].split()[0].strip()
                try:
                    value = int(frag)
                except ValueError:
                    continue
                if target == "code":
                    code = value
                else:
                    need = value
    try:
        code = int(code) if code is not None else None
    except (TypeError, ValueError):
        return None
    if code is None or code >= 0 or code not in _MUMPS_INFOG1:
        return None
    what, remedy = _MUMPS_INFOG1[code]
    extra = f" (INFOG(2) = {need})" if need not in (None, "") else ""
    return Advisory(
        f"MUMPS-INFOG1-{abs(code)}",
        "warning",
        f"factorization failed: {what}{extra}",
        remedy,
    )


def format_advisories(advisories: Sequence[Advisory], *, title: str) -> str:
    """Render advisories as a block, or a single clean line when there are none."""
    if not advisories:
        return f"{title}: none"
    lines = [f"{title}: {len(advisories)}"]
    lines.extend(a.render() for a in advisories)
    return "\n".join(lines)


def format_banner(fields: Mapping[str, Any]) -> str:
    """One block naming what is ACTUALLY about to run.

    Reports effective values, not requested ones: the recurring failure mode is a config
    that asks for something the driver then declines to do.
    """
    width = max((len(str(k)) for k in fields), default=0)
    lines = ["=" * 72, "linstabpy run configuration (effective)", "-" * 72]
    lines.extend(f"  {str(k).ljust(width)} : {v}" for k, v in fields.items())
    lines.append("=" * 72)
    return "\n".join(lines)


def format_summary(fields: Mapping[str, Any], advisories: Sequence[Advisory] = ()) -> str:
    """Closing block: what came out, and whether to trust it."""
    width = max((len(str(k)) for k in fields), default=0)
    lines = ["=" * 72, "linstabpy run summary", "-" * 72]
    lines.extend(f"  {str(k).ljust(width)} : {v}" for k, v in fields.items())
    lines.append("-" * 72)
    lines.append(format_advisories(advisories, title="  post-run advisories"))
    lines.append("=" * 72)
    return "\n".join(lines)
