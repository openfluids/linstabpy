#!/usr/bin/env python3
"""Forced-response oracle: injects the published forcing and compares the response.

He publishes no gain values, so a sigma-vs-sigma table against him is impossible.
This is the comparison that is available, and it is stronger anyway: a gain is one
number and can agree by accident, a field cannot.

Requires: the converted base flow and mode data under ``$SAVARINO_DATA`` (or
``data/`` next to this script), and a case configuration reachable from the
current working directory (see ``README.md``).

WHAT IT TESTS, precisely.  The forward map is ``H v = C inv(L_sigma) B v`` and the Chu
weight W appears nowhere in it (see ``linstabpy/resolvent/operator.py``) -- W enters
only the optimisation, i.e. *which* forcing is best.  Handing the operator a forcing
therefore tests ``inv(L_sigma)`` -- the assembled operator, the boundary conditions,
the base-flow ingest, the metric transform, beta and omega -- and C.  It does NOT
test the Chu weight, and it barely tests P, because his forcing is already exactly
zero outside his own restriction.  That independence is the point: the oracle can
run before the norm and P questions are settled.

HOW IT RUNS.  It does not rebuild the operator -- duplicating the driver's assembly
would risk testing a subtly different operator than the one that produces our gains.
Instead it monkeypatches ``driver.run._stage_run_resolvent``, which receives the
assembled ``L0_resolv`` together with ``mesh`` and ``idx``, and does the single solve
there.  Everything upstream is the production path, unmodified.

TWO CONVENTIONS ARE TESTED, NOT ASSUMED.  Our shift is ``L_sigma = L0 - (gamma +
i*omega) I``.  His ansatz is ``exp(i beta z - lambda t)`` with ``lambda = sigma + i
omega``, and the relative sign of the time convention is not stated anywhere we can
read.  Both ``+omega`` and ``-omega`` are solved and both correlations reported.  This is
a binary convention with a right answer, not a tunable parameter: if the operator is
right exactly one should correlate well and the other should not.  If NEITHER does, that
is the finding, and no amount of sign-flipping fixes it.

Usage (inside the case directory, via the normal run environment):
    LINSTABPY_CASE1_CONFIG=<oracle>.yaml python mode_oracle.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from savarino_mode_to_primitive import CONS, PRIM, linearise_to_primitive
from savarino_to_npy import read_tecplot

DATA_DIR = Path(os.environ.get("SAVARINO_DATA", HERE / "data"))

BASE_DAT = Path(
    os.environ.get(
        "SAVARINO_BASE_DAT",
        str(DATA_DIR / "state_atcenter_mesh300_y300_xend195_angle308_T157_Mach215.dat"),
    )
)
MODES = Path(os.environ.get("SAVARINO_MODES", str(DATA_DIR / "modes")))
POINT = os.environ.get("SAVARINO_POINT", "f2_w45")
OUT = Path(os.environ.get("SAVARINO_ORACLE_OUT", str(HERE / "out" / "oracle")))


def _complex_mode(kind: str) -> dict[str, np.ndarray]:
    re_ = read_tecplot(MODES / f"{kind}_atcenter_{POINT}_n1real.dat")
    im_ = read_tecplot(MODES / f"{kind}_atcenter_{POINT}_n1imag.dat")
    return {k: re_[k] + 1j * im_[k] for k in CONS}


def _to_dof(prim: dict[str, np.ndarray], mesh, idx, ndof: int) -> np.ndarray:
    """Scatter primitive (streamwise, wall-normal) fields onto the state vector.

    read_tecplot returns arrays with axis 0 streamwise and axis 1 wall-normal.  The
    case loader reads the .npy with ``transpose_input=True``, so the in-memory X/Y --
    and therefore ``mesh.used_ind`` and every idx block -- are (wall-normal,
    streamwise).  The transpose below is what keeps the two consistent; without it the
    error is invisible, because his grid is 300x300 and a transposed field has exactly
    the right shape.
    """
    used = np.asarray(mesh.used_ind)
    vec = np.zeros(ndof, dtype=complex)
    for var in PRIM:
        block = np.asarray(idx[f"{var}_j"], dtype=np.int64).reshape(-1)
        flat = prim[var].T.ravel(order="F")[used]
        vec[block] = flat
    return vec


def _compare(ours: np.ndarray, his: np.ndarray) -> dict[str, float]:
    """One global complex normalisation, then how much of his field we reproduce."""
    denom = float(np.vdot(ours, ours).real)
    alpha = complex(np.vdot(ours, his) / denom) if denom > 0 else 0.0
    resid = np.linalg.norm(alpha * ours - his) / np.linalg.norm(his)
    corr = abs(complex(np.vdot(his, ours))) / (np.linalg.norm(his) * np.linalg.norm(ours))
    return {"alpha_abs": abs(alpha), "rel_error": float(resid), "correlation": float(corr)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import linstabpy.driver.run as run_module
    from linstabpy.linalg import shifted_operator
    from linstabpy.linalg.factorize import factorize
    from linstabpy.resolvent.operator import ResolventOperator

    base_cons = read_tecplot(BASE_DAT)
    forcing_prim = linearise_to_primitive(_complex_mode("forcing"), base_cons)
    response_prim = linearise_to_primitive(_complex_mode("response"), base_cons)

    original = run_module._stage_run_resolvent
    results: dict[str, dict[str, float]] = {}

    def oracle_stage(*, L0_resolv, mesh, idx, config, W_diag, omega_target,
                     total_bc_indices, solver_cfg, cfg, **kw):
        ndof = mesh.ngp * 5
        f_vec = _to_dof(forcing_prim, mesh, idx, ndof)
        q_his = _to_dof(response_prim, mesh, idx, ndof)
        # Forcing a constrained row is meaningless; the production path masks them too.
        f_vec[np.asarray(total_bc_indices, dtype=np.int64)] = 0.0

        for label, omega in (("+omega", omega_target), ("-omega", -omega_target)):
            L_sigma = shifted_operator(L0_resolv, omega, gamma=0.0)
            lu = factorize(
                L_sigma, env_prefix="LINSTABPY_RESOLVENT", solver_cfg=solver_cfg, cfg=cfg
            )
            op = ResolventOperator(
                lu, None, None,
                mesh.filters.filter if mesh.filters is not None else None,
                mesh.filters.filter_ct if mesh.filters is not None else None,
            )
            q_ours = op.apply(f_vec)
            m = _compare(q_ours, q_his)
            results[label] = m
            print(
                f"[oracle] {label}: correlation = {m['correlation']:.6f}   "
                f"relative error after one global scale = {m['rel_error']:.6f}   "
                f"|alpha| = {m['alpha_abs']:.6e}",
                flush=True,
            )
            np.savez_compressed(
                OUT / f"oracle_{POINT}_{label.replace('+','p').replace('-','m')}.npz",
                q_ours=q_ours, q_his=q_his, f_vec=f_vec, **m,
            )

        best = min(results, key=lambda k: results[k]["rel_error"])
        print(f"[oracle] best convention: {best}", flush=True)
        # Let the driver finish tidily; the numbers above are the deliverable.
        return original(
            L0_resolv=L0_resolv, mesh=mesh, idx=idx, config=config, W_diag=W_diag,
            omega_target=omega_target, total_bc_indices=total_bc_indices,
            solver_cfg=solver_cfg, cfg=cfg, run_resolvent=False,
            n_modes_resolvent=kw.get("n_modes_resolvent", 1),
            heartbeat_seconds=kw.get("heartbeat_seconds", 0.0),
        )

    run_module._stage_run_resolvent = oracle_stage
    os.environ.setdefault(
        "LINSTABPY_CASE1_CONFIG", str(HERE / "savarino_point_f2_b45.yaml")
    )
    from linstabpy.driver.run import run as compute

    compute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
