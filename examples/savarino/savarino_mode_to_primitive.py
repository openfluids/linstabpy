#!/usr/bin/env python3
"""Linearised conservative -> primitive transform for Savarino's published modes.

Every other script in this directory uses this transform.
It converts his published forcing and response modes, and his base flow,
from conservative perturbation variables to linstabpy primitives.
It has no inputs of its own.
``main()`` below runs four falsifying checks on a synthetic state and
needs nothing external.

Why this is not ``savarino_to_npy.py``.
That file converts a STATE: it divides, ``u = rou/ro``.
That division is exact for a base flow and mismatched for a perturbation.
His published forcing and response modes are perturbations in conservative
variables.
Every mode header reads ``"X" "Y" "ro" "rou" "rov" "row" "roe"``.
Our state vector is primitive ``(rho, u, v, w, T)``.
Feeding a nonlinearly-converted mode into the operator produces a field
that looks like a mode and correlates badly.
That is indistinguishable from a mismatched operator.
So we derive the transform, and gate it, before it touches anything.

The derivation.
Write the conservative state ``q_c = (ro, rou, rov, row, roe)``.
Write the primitive state ``q_p = (rho, u, v, w, T)``.
The base-flow identities are as follows.

    e_int = roe/ro - |vel|^2 / 2                    (specific internal energy).
    T     = e_int * gamma (gamma - 1) M^2.

Linearising about a base state (barred) gives ``q_p' = Minv q_c'`` with
the following.

    rho' = ro'.
    u'   = (rou)'/rho_bar - u_bar rho'/rho_bar          (v, w alike).
    T'   = gamma (gamma - 1) M^2 times the internal-energy perturbation.
    That perturbation is roe'/rho_bar - e_tot_bar rho'/rho_bar
    - (u_bar u' + v_bar v' + w_bar w').

``e_tot_bar = roe_bar / rho_bar`` is the specific TOTAL energy, not the
internal energy.
Dropping that distinction is the easiest way to mismatch this map.

The same map serves the forcing.
Force the conservative system as ``dq_c/dt = R_c + f_c``.
The identical primitive system is ``dq_p/dt = R_p + f_p``.
With ``q_c = Phi(q_p)`` and ``M = dPhi/dq_p`` we get ``M dq_p/dt = R_c + f_c``.
Hence ``f_p = Minv f_c``.
One transform, used for both his response mode and his forcing mode.

Run this file directly to execute the four falsifying checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from savarino_to_npy import GAMMA, MACH, read_tecplot  # noqa: E402

#: gamma (gamma - 1) M^2 -- the coefficient turning specific internal energy into T.
T_COEFF = GAMMA * (GAMMA - 1.0) * MACH**2

CONS = ("ro", "rou", "rov", "row", "roe")
PRIM = ("rho", "u", "v", "w", "T")


def primitive_base(cons: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Exact (nonlinear) conservative -> primitive map, for a STATE."""
    rho = cons["ro"]
    u = cons["rou"] / rho
    v = cons["rov"] / rho
    w = cons["row"] / rho
    e_int = cons["roe"] / rho - 0.5 * (u**2 + v**2 + w**2)
    return {"rho": rho, "u": u, "v": v, "w": w, "T": e_int * T_COEFF}


def conservative_state(prim: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Exact (nonlinear) primitive -> conservative map, for a STATE."""
    rho, u, v, w = prim["rho"], prim["u"], prim["v"], prim["w"]
    e_int = prim["T"] / T_COEFF
    return {
        "ro": rho,
        "rou": rho * u,
        "rov": rho * v,
        "row": rho * w,
        "roe": rho * (e_int + 0.5 * (u**2 + v**2 + w**2)),
    }


def linearise_to_primitive(
    pert: dict[str, np.ndarray], base_cons: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Apply ``Minv`` to a conservative PERTURBATION about ``base_cons``.

    Works for real or complex perturbation arrays.
    The base flow is real and the map is linear, so a complex mode
    transforms component-wise.
    """
    rho_b = base_cons["ro"]
    u_b = base_cons["rou"] / rho_b
    v_b = base_cons["rov"] / rho_b
    w_b = base_cons["row"] / rho_b
    e_tot_b = base_cons["roe"] / rho_b  # specific TOTAL energy, not internal

    d_rho = pert["ro"]
    d_u = (pert["rou"] - u_b * d_rho) / rho_b
    d_v = (pert["rov"] - v_b * d_rho) / rho_b
    d_w = (pert["row"] - w_b * d_rho) / rho_b
    d_e_int = (
        pert["roe"] / rho_b - e_tot_b * d_rho / rho_b - (u_b * d_u + v_b * d_v + w_b * d_w)
    )
    return {"rho": d_rho, "u": d_u, "v": d_v, "w": d_w, "T": d_e_int * T_COEFF}


# ---------------------------------------------------------------- falsifying checks


def _synthetic_base(n: int = 40, seed: int = 0) -> dict[str, np.ndarray]:
    """Return a smooth, strictly positive synthetic state.

    This is not his data, so the checks run anywhere.
    They cannot be accidentally satisfied by a special feature of his field.
    """
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.3 * rng.random((n, n))
    u = 0.5 + 0.4 * rng.random((n, n))
    v = 0.1 * rng.standard_normal((n, n))
    w = 0.05 * rng.standard_normal((n, n))
    T = 1.0 + 0.5 * rng.random((n, n))
    return conservative_state({"rho": rho, "u": u, "v": v, "w": w, "T": T})


def check_1_zero_maps_to_zero(base: dict[str, np.ndarray]) -> float:
    zero = {k: np.zeros_like(base[k]) for k in CONS}
    out = linearise_to_primitive(zero, base)
    worst = max(float(np.abs(out[k]).max()) for k in PRIM)
    print(f"  check 1  zero -> zero                         worst |.| = {worst:.3e}")
    return worst


def check_2_second_order_convergence(base: dict[str, np.ndarray]) -> float:
    """(exact(q+eps*d) - exact(q)) / eps  ->  Minv d  with an O(eps) error.

    Halving eps must halve the error.
    That is what proves the map is the derivative and not merely something
    with the right units.
    """
    rng = np.random.default_rng(7)
    d = {k: rng.standard_normal(base[k].shape) * np.abs(base[k]).mean() for k in CONS}
    lin = linearise_to_primitive(d, base)
    p0 = primitive_base(base)

    errs = []
    eps_list = [1e-3, 5e-4, 2.5e-4]
    for eps in eps_list:
        pert = {k: base[k] + eps * d[k] for k in CONS}
        p1 = primitive_base(pert)
        fd = {k: (p1[k] - p0[k]) / eps for k in PRIM}
        errs.append(max(float(np.abs(fd[k] - lin[k]).max()) for k in PRIM))

    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print(
        f"  check 2  finite-difference vs Minv            errors = "
        f"{', '.join(f'{e:.3e}' for e in errs)}"
    )
    print(
        f"           halving eps must halve the error     ratios = "
        f"{', '.join(f'{r:.2f}' for r in ratios)}  (want ~2)"
    )
    return max(abs(r - 2.0) for r in ratios)


def check_3_round_trip(base: dict[str, np.ndarray]) -> float:
    p = primitive_base(base)
    c = conservative_state(p)
    worst = max(float(np.abs(c[k] - base[k]).max()) for k in CONS)
    print(f"  check 3  primitive -> conservative round trip worst |.| = {worst:.3e}")
    return worst


def check_4_freestream_coefficient() -> float:
    """The identity that pinned the base-flow conversion must reappear here."""
    expected = 1.0 / (GAMMA * (GAMMA - 1.0) * MACH**2)
    err = abs(1.0 / T_COEFF - expected)
    print(
        f"  check 4  1/T_COEFF = {1.0 / T_COEFF:.9f}  vs  "
        f"1/(g(g-1)M^2) = {expected:.9f}   diff = {err:.3e}"
    )
    return err


def main() -> int:
    print("Falsifying checks for the linearised conservative -> primitive transform")
    print("(synthetic base state, so nothing here can pass by a quirk of his field)")
    base = _synthetic_base()
    c1 = check_1_zero_maps_to_zero(base)
    c2 = check_2_second_order_convergence(base)
    c3 = check_3_round_trip(base)
    c4 = check_4_freestream_coefficient()

    failures = []
    if c1 != 0.0:
        failures.append("check 1: zero perturbation did not map to exactly zero")
    if c2 > 0.15:
        failures.append(f"check 2: convergence ratio off 2.0 by {c2:.3f}")
    if c3 > 1e-12:
        failures.append(f"check 3: round trip error {c3:.3e} above 1e-12")
    if c4 > 1e-15:
        failures.append(f"check 4: coefficient mismatch {c4:.3e}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall four checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
