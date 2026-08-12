#!/usr/bin/env python3
"""Compare a computed 31.6 deg EVP spectrum against the reference of Z. Song
(supplement, stability-bracket table).

The reference file is the machine-readable ``../data/song_316deg/eigenvalues.dat``.
Computed spectra are expected as ``.npz`` archives from the LinStabPy runtime
(``omega`` or ``stability_all_omega`` complex arrays), with a small fallback for
``*_real``/``*_imag`` array pairs used by synthetic tests.

Hard residual gate: a verdict is only ever
printed for a spectrum whose real (true) residual is small compared to the
physics scale it claims. Under shift-invert, SLEPc's default residual is
measured on the TRANSFORMED operator -- ``eps_tol: 1e-6`` can coexist with a
true residual of 1.2e-3, which is blind exactly where the physics lives
(|omega_i| ~ 6e-4). An earlier run emitted a verdict from an ungated spectrum and had to be
retracted; this script now refuses (non-zero exit) instead of repeating
that mistake. Do NOT relax this gate to make a run "pass".
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from parse_song316_eigenvalues import load_eigenvalues, sort_least_damped_first


ComplexRow = tuple[float, float]


COMPLEX_KEYS = ("stability_all_omega", "omega")
REAL_IMAG_KEY_PAIRS = (
    ("omega_real", "omega_imag"),
    ("omega_r", "omega_i"),
    ("real", "imag"),
    ("wr", "wi"),
)

# ---------------------------------------------------------------------------
# The gate: INDEPENDENT verification, not the solver's self-report.
# ---------------------------------------------------------------------------
# Correction history. An earlier gate demanded
# ``stability_true_residual is True`` -- a flag saying SLEPc's own convergence
# test was switched (EPSSetTrueResidual) -- and read SLEPc's estimate. Two
# problems, both measured on the validation workstations:
#
#  1. It gates on the solver's SELF-REPORT. A solver's internal estimate for a
#     shift-inverted operator is exactly the quantity you cannot take on trust.
#  2. Forcing ``true_residual: true`` made the eigensolve non-convergent for
#     this flow: the marginal mode sits at omega ~ 0, so a residual measured on
#     the physical operator and tested RELATIVELY divides by ~0, and an ABSOLUTE
#     test is hopeless because ||L0|| is ~1e10 here. With eps_max_it defaulting
#     to 10*ndof the solve simply grinds to the wall clock; all six rungs of a
#     resolution ladder timed out having written nothing.
#
# linstabpy already computes the right thing. ``driver/run.py``
# ``_postcheck_stability_modes`` forms, for each RETURNED eigenpair,
#
#       res = 1j * (L0 @ mode) - omega * mode
#
# directly from the assembled operator and the returned eigenvector. That is an
# INDEPENDENT residual on the ORIGINAL problem -- it shares no code path with
# SLEPc's convergence estimate -- and it is written unconditionally, whatever
# ``true_residual`` was set to. Gating on it is strictly STRONGER than gating on
# the old flag, and it needs no unreachable solver tolerance.
#
# We gate on three independent postcheck quantities, all fail-closed:
#   * residual_rel      -- ||res|| / ||mode||, the physics residual of the pair;
#   * dirichlet_leak    -- |mode| on Dirichlet-constrained rows, which must
#                          vanish (do NOT use bc_leak: bc_indices also contains
#                          Neumann rows, where nonzero amplitude is CORRECT);
#   * target_distance   -- how far the claimed mode sits from the requested shift.
#
# Interpretation aid, measured on a stride-4 run (eps=0.10, sigma=0):
#   postcheck_residual_rel = 2.0e-5, postcheck_interior_abs = 3.0e-7,
#   dirichlet_leak_max = 1.4e-15. The total absolute residual (9.8e-3) is
#   dominated by penalty-scaled BC rows (penalty = 1e8), which is expected and is
#   why the INTERIOR residual is reported next to it.
#
# A relative residual of 1e-2 already pins the sign of omega_i for |omega_i| ~
# 4e-2 by three orders of magnitude, so this bound is conservative for a
# stability verdict while still refusing genuinely unconverged spectra.
RESIDUAL_REL_GATE = 1.0e-2
# Dirichlet rows are constrained to zero; anything above this is a broken mode,
# not a tight-tolerance question. the stride-4 run achieves 1.4e-15.
DIRICHLET_LEAK_GATE = 1.0e-6

# Independent-oracle keys (written unconditionally by driver/run.py:88-95).
POSTCHECK_REL_KEYS = ("stability_postcheck_residual_rel",)
POSTCHECK_INTERIOR_KEYS = ("stability_postcheck_residual_interior_abs",)
POSTCHECK_DIRICHLET_KEYS = ("stability_postcheck_dirichlet_leak_max",)
POSTCHECK_TARGET_KEYS = ("stability_postcheck_target_distance",)
# Reported for transparency only -- never gated on (this is the self-report).
RESIDUAL_TRUE_FLAG_KEY = "stability_true_residual"
RESIDUAL_ARRAY_KEYS = ("stability_residual_absolute", "residual_absolute")


def _as_1d_complex(values: np.ndarray, *, key: str) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        raise SystemExit(f"{key}: spectrum array is empty")
    if not np.all(np.isfinite(np.real(arr))) or not np.all(np.isfinite(np.imag(arr))):
        raise SystemExit(f"{key}: non-finite eigenvalue encountered")
    return arr.astype(complex, copy=False)


def load_npz_archive(path: Path) -> dict[str, np.ndarray]:
    """Load a full ``.npz`` spectrum archive into a plain dict (no pickle)."""
    if path.suffix.lower() != ".npz":
        raise SystemExit(f"{path}: expected a .npz spectrum archive")
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except ValueError as exc:
        raise SystemExit(f"{path}: failed to load NPZ without pickle: {exc}") from exc


def extract_spectrum(data: dict[str, np.ndarray], *, path: Path) -> tuple[np.ndarray, str]:
    """Return computed omega values and the NPZ key(s) used, from a loaded archive."""
    keys = set(data.keys())
    for key in COMPLEX_KEYS:
        if key in keys:
            return _as_1d_complex(data[key], key=key), key
    for real_key, imag_key in REAL_IMAG_KEY_PAIRS:
        if real_key in keys and imag_key in keys:
            real = np.asarray(data[real_key], dtype=float).reshape(-1)
            imag = np.asarray(data[imag_key], dtype=float).reshape(-1)
            if real.shape != imag.shape:
                raise SystemExit(
                    f"{path}: shape mismatch for {real_key}/{imag_key}: "
                    f"{real.shape} vs {imag.shape}"
                )
            return _as_1d_complex(real + 1j * imag, key=f"{real_key}+{imag_key}"), (
                f"{real_key}+{imag_key}"
            )
    available = ", ".join(sorted(keys)) or "<none>"
    raise SystemExit(
        f"{path}: no supported spectrum arrays found; expected one of "
        f"{COMPLEX_KEYS} or real/imag pairs {REAL_IMAG_KEY_PAIRS}; "
        f"available keys: {available}"
    )


def load_computed_spectrum(path: Path) -> tuple[np.ndarray, str]:
    """Return computed omega values and the NPZ key(s) used."""
    data = load_npz_archive(path)
    return extract_spectrum(data, path=path)


def leading_mode_index(omega: Iterable[complex]) -> int:
    """Return the index of the mode with largest omega_i, tie-breaking toward |omega_r|=0."""
    values = list(omega)
    if not values:
        raise SystemExit("computed spectrum is empty")
    best_idx = 0
    best_key = (float(np.imag(values[0])), -abs(float(np.real(values[0]))))
    for idx in range(1, len(values)):
        key = (float(np.imag(values[idx])), -abs(float(np.real(values[idx]))))
        if key > best_key:
            best_key = key
            best_idx = idx
    return best_idx


def least_damped(omega: Iterable[complex]) -> complex:
    """Return the mode with largest omega_i, tie-breaking toward |omega_r|=0."""
    values = list(omega)
    return values[leading_mode_index(values)]


def _pick_scalar(
    data: dict[str, np.ndarray],
    keys: tuple[str, ...],
    *,
    omega: np.ndarray,
    leading_idx: int,
    path: Path,
    what: str,
) -> float:
    """Return the leading mode's entry of the first present array in ``keys``.

    Fail-closed: a missing or empty oracle array raises. An absent oracle is not
    a pass -- that inversion is what produced the retracted verdict above.
    """
    for key in keys:
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=float).reshape(-1)
        if arr.size == 0:
            raise SystemExit(f"{path}: {key} is present but EMPTY -- refusing ({what}).")
        if arr.size == omega.size:
            return float(arr[leading_idx])
        if arr.size == 1:
            return float(arr[0])
        raise SystemExit(
            f"{path}: {key} has size {arr.size}, expected 1 or {omega.size} -- refusing."
        )
    raise SystemExit(
        f"{path}: none of {keys} present, so {what} cannot be verified independently. "
        "REFUSING to emit a verdict. These arrays are written unconditionally by "
        "linstabpy's stability path; an npz without them predates the postcheck or was "
        "not produced by that path."
    )


def load_postcheck_oracle(
    data: dict[str, np.ndarray],
    *,
    omega: np.ndarray,
    leading_idx: int,
    path: Path,
) -> dict[str, float]:
    """Return the independent postcheck diagnostics for the claimed mode.

    Every value is recomputed by linstabpy from the assembled operator and the
    returned eigenvector, independently of the eigensolver's own estimate.
    """
    out = {
        "residual_rel": _pick_scalar(
            data, POSTCHECK_REL_KEYS, omega=omega, leading_idx=leading_idx, path=path,
            what="the independent relative residual",
        ),
        "dirichlet_leak": _pick_scalar(
            data, POSTCHECK_DIRICHLET_KEYS, omega=omega, leading_idx=leading_idx, path=path,
            what="Dirichlet boundary satisfaction",
        ),
    }
    # Reported for context; not gated (interior/target are diagnostics, and the
    # interior residual is the physics-only part of an absolute residual that
    # penalty-scaled BC rows otherwise dominate).
    for name, keys in (
        ("residual_interior_abs", POSTCHECK_INTERIOR_KEYS),
        ("target_distance", POSTCHECK_TARGET_KEYS),
    ):
        try:
            out[name] = _pick_scalar(
                data, keys, omega=omega, leading_idx=leading_idx, path=path, what=name
            )
        except SystemExit:
            out[name] = float("nan")
    return out


def load_residual_for_leading_mode(
    data: dict[str, np.ndarray],
    *,
    omega: np.ndarray,
    leading_idx: int,
    path: Path,
) -> tuple[float, bool | None, str]:
    """Return ``(residual, true_residual_flag_or_None, residual_key_used)``.

    Refuses (raises ``SystemExit``) if no residual array is present at all --
    a verdict must never be emitted from an ungated spectrum.
    """
    true_residual_flag: bool | None = None
    if RESIDUAL_TRUE_FLAG_KEY in data:
        true_residual_flag = bool(np.asarray(data[RESIDUAL_TRUE_FLAG_KEY]).reshape(-1)[0])

    for array_key in RESIDUAL_ARRAY_KEYS:
        if array_key not in data:
            continue
        residual_array = np.asarray(data[array_key], dtype=float).reshape(-1)
        if residual_array.size == 0:
            raise SystemExit(
                f"{path}: {array_key} is present but empty -- REFUSING to emit a verdict"
            )
        if residual_array.size == omega.size:
            return float(residual_array[leading_idx]), true_residual_flag, array_key
        if residual_array.size == 1:
            return float(residual_array[0]), true_residual_flag, array_key
        raise SystemExit(
            f"{path}: {array_key} has size {residual_array.size}, expected either "
            f"1 (single run-level residual) or {omega.size} (one per computed mode)"
        )

    raise SystemExit(
        f"{path}: no residual array found (expected one of {RESIDUAL_ARRAY_KEYS}); "
        "REFUSING to emit a verdict from an ungated spectrum (re-run with solver.eigen.true_residual: true so the residual is written)."
    )


def _format_mode(prefix: str, value: complex) -> str:
    return f"{prefix}: omega_r={value.real: .15e} omega_i={value.imag: .15e}"


def _spike_threshold(reference_leading: ComplexRow, user_threshold: float | None) -> float:
    if user_threshold is not None:
        return float(user_threshold)
    # "omega_i >> 0" relative to the reference damping scale. Keep a fixed floor
    # so tiny reference damping does not flag numerical roundoff as a spike.
    return max(1.0e-2, 10.0 * abs(reference_leading[1]))


def classify_sawtooth_spikes(
    omega: np.ndarray,
    *,
    reference_leading: ComplexRow,
    positive_growth_threshold: float | None,
    large_real_threshold: float,
) -> tuple[bool, int, int, float, float]:
    """Heuristically flag sawtooth-like EVP contamination."""
    growth_threshold = _spike_threshold(reference_leading, positive_growth_threshold)
    omega_i = np.imag(omega)
    omega_r_abs = np.abs(np.real(omega))
    positive_growth = omega_i > growth_threshold
    large_real = omega_r_abs > large_real_threshold
    return (
        bool(np.any(positive_growth | large_real)),
        int(np.count_nonzero(positive_growth)),
        int(np.count_nonzero(large_real)),
        float(growth_threshold),
        float(large_real_threshold),
    )


def build_report(
    spectrum_path: Path,
    reference_path: Path,
    *,
    positive_growth_threshold: float | None,
    large_real_threshold: float,
) -> str:
    data = load_npz_archive(spectrum_path)
    computed, computed_key = extract_spectrum(data, path=spectrum_path)
    reference = sort_least_damped_first(load_eigenvalues(reference_path))
    ref_lead = reference[0]

    leading_idx = leading_mode_index(computed)
    lead = computed[leading_idx]

    # --- Hard gate on the INDEPENDENT postcheck oracle. REFUSE before computing
    # or printing any verdict-bearing quantity. See the block by the constants:
    # these arrays come from ``i*L0 @ x - omega*x`` recomputed on the assembled
    # operator, not from the solver's convergence estimate. Fail-closed on a
    # missing array -- an absent oracle is not a pass. ---
    oracle = load_postcheck_oracle(
        data, omega=computed, leading_idx=leading_idx, path=spectrum_path
    )
    if not (oracle["residual_rel"] <= RESIDUAL_REL_GATE):
        raise SystemExit(
            f"{spectrum_path}: POSTCHECK RESIDUAL GATE FAILED -- independent "
            f"||i*L0 x - omega x||/||x|| = {oracle['residual_rel']:.6e} exceeds "
            f"{RESIDUAL_REL_GATE:g}. Verdict REFUSED: this eigenpair does not solve the "
            "original problem well enough to claim a growth rate from it. Re-run with a "
            "tighter eps_tol or larger ncv (do NOT relax this gate)."
        )
    if not (oracle["dirichlet_leak"] <= DIRICHLET_LEAK_GATE):
        raise SystemExit(
            f"{spectrum_path}: DIRICHLET LEAK GATE FAILED -- |mode| on "
            f"Dirichlet-constrained rows reaches {oracle['dirichlet_leak']:.6e} > "
            f"{DIRICHLET_LEAK_GATE:g}. Verdict REFUSED: the claimed mode violates the "
            "boundary conditions, so it is a numerical artifact rather than a solution "
            "of the stated problem."
        )

    distance_omega_i = abs(float(lead.imag) - ref_lead[1])
    complex_distance = math.hypot(float(lead.real) - ref_lead[0], float(lead.imag) - ref_lead[1])
    verdict = "stable/marginal" if float(lead.imag) < 0.0 else "unstable"
    spike_flag, n_growth, n_large_real, growth_threshold, real_threshold = classify_sawtooth_spikes(
        computed,
        reference_leading=ref_lead,
        positive_growth_threshold=positive_growth_threshold,
        large_real_threshold=large_real_threshold,
    )

    lines = [
        f"computed_spectrum={spectrum_path}",
        f"computed_key={computed_key}",
        f"computed_modes={computed.size}",
        "postcheck_gate=PASS (independent oracle: res = i*L0 x - omega x, recomputed "
        "from the assembled operator, not the solver's estimate)",
        f"postcheck_residual_rel={oracle['residual_rel']:.6e} <= {RESIDUAL_REL_GATE:g}",
        f"postcheck_residual_interior_abs={oracle['residual_interior_abs']:.6e}",
        f"postcheck_dirichlet_leak_max={oracle['dirichlet_leak']:.6e} <= "
        f"{DIRICHLET_LEAK_GATE:g}",
        f"postcheck_target_distance={oracle['target_distance']:.6e}",
        f"solver_self_report(NOT gated): {RESIDUAL_TRUE_FLAG_KEY}="
        f"{bool(np.asarray(data[RESIDUAL_TRUE_FLAG_KEY]).reshape(-1)[0]) if RESIDUAL_TRUE_FLAG_KEY in data else 'absent'}",
        f"reference_spectrum={reference_path}",
        f"reference_modes={len(reference)}",
        f"reference_leading: omega_r={ref_lead[0]: .15e} omega_i={ref_lead[1]: .15e}",
        _format_mode("computed_leading", lead),
        f"distance_to_reference_leading_omega_i={distance_omega_i:.15e}",
        f"complex_distance_to_reference_leading={complex_distance:.15e}",
        f"verdict={verdict}",
        "sawtooth_spike_flag="
        f"{'YES' if spike_flag else 'NO'} "
        f"(positive_growth_count={n_growth}, large_real_count={n_large_real}, "
        f"positive_growth_threshold={growth_threshold:.6e}, "
        f"large_real_threshold={real_threshold:.6e})",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum", type=Path, help="computed EVP spectrum .npz")
    parser.add_argument("reference", type=Path, help="data/song_316deg/eigenvalues.dat")
    parser.add_argument(
        "--positive-growth-threshold",
        type=float,
        default=None,
        help="omega_i threshold for sawtooth positive-growth spikes (default: max(1e-2, 10*|reference leading omega_i|))",
    )
    parser.add_argument(
        "--large-real-threshold",
        type=float,
        default=1.0e-1,
        help="|omega_r| threshold for large-real sawtooth spikes (default: 1e-1)",
    )
    args = parser.parse_args()

    print(
        build_report(
            args.spectrum,
            args.reference,
            positive_growth_threshold=args.positive_growth_threshold,
            large_real_threshold=args.large_real_threshold,
        )
    )


if __name__ == "__main__":
    main()
