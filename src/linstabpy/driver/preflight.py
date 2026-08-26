# SPDX-License-Identifier: GPL-3.0-only
"""Preflight and post-run advisories: catch the configurations that run but lie.

Every check here exists because the configuration *runs to completion and
produces plausible numbers*. Nothing below is a schema error.
``config.validate_config`` already rejects unknown keys and bad types.
These are the semantic traps: settings that are individually legal,
jointly meaningless, and silent.

Design rules, deliberately narrow.

* Advisories never block. A false positive that aborts a twelve-hour job
  is worse than the trap it guards. The driver logs these and continues.
  Only pre-existing hard errors stay hard.
* Pure functions over plain data. Every check takes a config dict or a
  few arrays and returns advisories. No PETSc, no solve, no I/O.
  The unit tests run in milliseconds on a laptop.
* Each advisory carries a remedy. "Warning: dissipation disabled" helps
  nobody. The message must say what to set and where.

A first-time user who reads the log learns what an experienced user of
this code already knows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from linstabpy.config.defaults import DISSIPATION_APPLY_DEFAULTS
from linstabpy.linalg.backends import _normalize_backend

__all__ = [
    "Advisory",
    "baseflow_advisories",
    "config_advisories",
    "format_advisories",
    "format_banner",
    "format_summary",
    "metric_cross_coupling",
    "result_advisories",
    "translate_solver_error",
]

# WITHDRAWN 2026-08-07 -- kept only so an old import does not break.
#
# This encoded a claim that no measurement supported: that grid-scale sawtooth
# "dominates" for beta in 90-135. A direct run of the bare operator at
# beta = 110 at two resolutions (no undamped run entered the band) gave
# gain 1.3098e+02 against 1.3099e+02 damped, a difference of 0.008%.
# Every gain map in the source project stops at beta = 80 with a maximum
# of 689.9, and a controlled bare-vs-damped comparison at a single point
# measured 1.2%. There is no band.
#
# What actually decides whether the odd-even mode exists is the MESH -- see
# `metric_cross_coupling` below. Nothing in this module uses this constant any more.
SAWTOOTH_BETA_BAND = (90.0, 135.0)

# A leading Chu-norm gain above this is far outside the O(1e2-1e3) these cases
# produce, so it is worth a warning -- but the warning no longer asserts a cause.
# The historical "gain ~1e6 means sawtooth" reading was a conflation: on a globally
# UNSTABLE base flow the plain resolvent has poles on the real omega axis and the
# gain diverges for an honest reason (a missing discount, not missing dissipation).
SAWTOOTH_GAIN_SCALE = 1.0e5

#: Metric cross-coupling above which the mesh mixes grid-scale content between its
#: two directions, so the operator admits an odd-even mode that dissipation or a
#: filter must control. Measured on five meshes: 7e-18, 2e-17 and 2e-18 on
#: rectilinear ones (Savarino 300x300, Song 600x200 box, MATLAB box) against 2.6e-01
#: on curvilinear ones (curved-wall LES, and the same field on a coarser mesh).
#: Sixteen orders of magnitude, so any threshold in 1e-10..1e-3 separates them
#: identically. We pick 1e-8 only because it is far from both.
METRIC_COUPLING_GATE = 1.0e-8
# Freestream normalisation: rho = u = T = 1. A base flow that misses this by more than
# the tolerance used a different scaling convention, which silently invalidates
# every eigenvalue and gain computed from it.
FREESTREAM_TOL = 1.0e-2


@dataclass(frozen=True)
class Advisory:
    """One actionable finding. ``code`` is stable and greppable. ``remedy`` names the fix."""

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
    """Semantic checks on a loaded config, before operator assembly."""
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
                "set numerics.operator_dissipation.apply_to_resolvent = true so the two "
                "stages agree, or set enabled = false so the artifact records that BOTH "
                "ran bare; a silent difference between the two stages is a defect "
                "whatever its size, and on a mesh with nonzero metric cross-coupling it "
                "is not small",
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
        # Deliberately phrased as "measure it", not a verdict. Whether the bare
        # operator admits the odd-even mode at all depends on the MESH: a rectilinear
        # tensor mesh has zero metric cross-coupling and cannot support it, and on
        # such a mesh dissipation changed the gain by 0.008%-1.4% for +15% nnz
        # and roughly triple the wall time. Warning unconditionally taught people to
        # scroll past this, which is how a real case gets waved through.
        out.append(
            Advisory(
                "DISS-ABSENT",
                "warning",
                "no artificial dissipation: on a mesh whose two directions exchange "
                "grid-scale content, the bare operator admits an odd-even (sawtooth) mode "
                "that contaminates eigenvalues and gains",
                "check the mesh with metric_cross_coupling(): above "
                f"{METRIC_COUPLING_GATE:.0e} enable numerics.operator_dissipation and tune "
                "epsilon AT THIS MESH (a value tuned on another grid does not transfer); "
                "at or below it, record the coupling and run undamped on purpose",
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

    # We deleted the BETA-SAWTOOTH-BAND advisory that used to sit here. We did not gate it off.
    # It fired on beta in 90-135 with no effective dissipation, on the strength of a
    # claim that no run supported; when the band was finally entered with the bare
    # operator the effect measured 0.008%. A warning that fires on correct runs is
    # worse than no warning, because it trains the reader to ignore the channel.

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

    # Band-marking backend trap. resolvent_map runs a mandatory band-marking EVP, and
    # that EVP reads solver.eigen -- NOT solver.resolvent. A config that scales up only
    # the resolvent backend leaves the EVP on the default direct solver, which at full
    # resolution does not appear to fail: it simply never finishes. The run looks like a
    # slow resolvent sweep for hours before anyone suspects the EVP.
    if run_res and isinstance(solver, Mapping):
        skip_band = bool(
            _get(config, "analysis.skip_band_marking", _get(config, "skip_band_marking", False))
        )
        res_sub = solver.get("resolvent")
        if not isinstance(res_sub, Mapping):
            res_sub = solver.get("linsolve")
        # Normalize with the resolver's own rule so sentinel strings such as
        # "none" or "default" do not count as a configured backend.
        res_backend = _normalize_backend(
            res_sub.get("factor_backend") if isinstance(res_sub, Mapping) else None
        )
        eigen_sub = solver.get("eigen")
        eigen_backend = _normalize_backend(
            eigen_sub.get("factor_backend") if isinstance(eigen_sub, Mapping) else None
        )
        solver_backend = _normalize_backend(solver.get("factor_backend"))
        eigen_resolved = eigen_backend or solver_backend
        if not skip_band and res_backend and not eigen_resolved:
            out.append(
                Advisory(
                    "BAND-MARKING-BACKEND",
                    "warning",
                    f"the resolvent backend is set to '{res_backend}' but the band-marking EVP "
                    "reads solver.eigen, which is unset and will fall back to the default "
                    "direct solver; at full resolution this stalls rather than errors",
                    "set solver.factor_backend (applies to every stage), or set "
                    "solver.eigen.factor_backend explicitly, or set analysis.skip_band_marking "
                    "= true if the stable-regime assumption genuinely holds for this sweep",
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

    A base flow scaled with a different convention runs to completion.
    The numbers look plausible but the physics is not the intended one.
    This already invalidated one full campaign session.
    The check uses the field maximum rather than a corner value.
    It does not need to know where the freestream sits in the array.
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


def metric_cross_coupling(X: np.ndarray, Y: np.ndarray) -> float:
    """How strongly does this mesh mix the two directions at the grid scale?

    Return ``max(|dx/deta|)/max(|dx/dxi|)`` and its transpose, whichever is larger.
    The number is exactly 0 for a rectilinear tensor mesh.
    It is O(1) for a body-fitted curvilinear mesh.

    Why this quantity matters. The 2-D metric transform is

        Dx_new = Ji11 * Dx + Ji12 * Dy.     Ji12 = -(Dx @ y) / detJ.
        Dy_new = Ji21 * Dx + Ji22 * Dy.     Ji21 = -(Dy @ x) / detJ.

    On a rectilinear mesh ``x = x(xi)`` and ``y = y(eta)``.
    Then ``Dx @ y`` and ``Dy @ x`` vanish identically.
    Each derivative is a pure rescaling of its own one-dimensional stencil.
    The two directions' grid-scale null spaces never meet.
    The odd-even mode has nowhere to live.
    Bend the mesh and those terms become O(1).
    The x-derivative now contains the y-stencil.
    The second derivatives acquire the cross terms ``Dxy = Dx @ Dy`` and
    ``Dyx = Dy @ Dx``.
    The operator then supports a mode that neither one-dimensional stencil
    admits alone.

    Stretching is not what does it. Bending is.
    A mesh may stretch by two orders of magnitude in the wall-normal
    direction and still return exactly 0 here.

    Five meshes measured: 7e-18, 2e-17, 2e-18 (rectilinear) against
    2.6e-01 twice (curvilinear).
    The same at 2nd, 4th and 6th order.
    This is a property of the mapping, not of the stencil.
    The caller can compute it before any operator assembly.

    Parameters
    ----------
    X, Y : np.ndarray
        Physical coordinates on the computational grid, same 2-D shape.

    Returns
    -------
    float
        Cross-coupling. Compare against :data:`METRIC_COUPLING_GATE`.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape or X.ndim != 2:
        raise ValueError(f"X and Y must be the same 2-D shape, got {X.shape} and {Y.shape}")

    dx_dxi = np.abs(np.diff(X, axis=0)).max()
    dx_deta = np.abs(np.diff(X, axis=1)).max()
    dy_dxi = np.abs(np.diff(Y, axis=0)).max()
    dy_deta = np.abs(np.diff(Y, axis=1)).max()

    # Guard a degenerate direction rather than dividing by zero: a mesh with no extent
    # along one axis is not curvilinear. That missing extent is a different error.
    if dx_dxi <= 0.0 or dy_deta <= 0.0:
        raise ValueError("degenerate mesh: no extent along one computational direction")

    return float(max(dx_deta / dx_dxi, dy_dxi / dy_deta))


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
# opaque "EPS.setUp failed", which sent one triage away from the INFOG(1) code.
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

    Report effective values, not requested ones.
    A common failure is a config that asks for a path the driver then declines.
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
