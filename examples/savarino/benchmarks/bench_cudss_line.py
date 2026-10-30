#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-only
# Benchmark tool for the linstabpy backend measurements.
"""cuDSS along an omega line, timed properly.

The map walks a beta line: same L0, omega changes. So K(f) = K(f0) - 2*pi*i*(f-f0)*I
shares one sparsity pattern. This benchmark measures what that pattern reuse is worth:

1. plan once (cuDSS threading layer, 10 host threads) and factor at f0;
2. per next f: update the values, then ``reset_operands(a=...)`` + ``factorize()``.
   That refactorization time sets the line-mode cost estimate;
3. per f: 10 forward + 10 adjoint FGMRES solves, chained like the SVD iteration.
   Each RHS is the normalized previous solution. Vectors stay in the
   production subspace;
4. verify everything. Each refactored factor must solve the SHIFTED operator
   (true residual in double). The ladder's K(f_verify) must equal the second
   dump bit-for-bit. One MUMPS cross-check runs at f_verify.

Arms: double (no matching, no equilibration) and single (matching
MAX_DIAG_PRODUCT + equilibration -- the uniform FP32 opt-in candidate).

Run on one rank in the cudss venv. Submit on c9 through sbatch. Example::

    mpiexec -n 1 python bench_cudss_line.py --npz dumps/K_b40_f1.npz
        --verify-npz dumps/K_b40_f1p8.npz --out results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from bench_cudss import (
    MAX_IT_DEFAULT,
    RESTART_DEFAULT,
    RTOL_DEFAULT,
    CudssFactor,
    ScaledFactorPC,
    csr_int32,
    device_memory_used_mib,
    equilibrate,
    ksp_solve,
    load_dump,
    make_fgmres,
    petsc_mat,
)


def load_b_mask(path: Path) -> sp.csr_matrix:
    """Load the forcing matrix B from a dump (the production RHS mask)."""
    z = np.load(path, allow_pickle=False)
    shape = tuple(int(v) for v in z["B_shape"])
    return sp.csr_matrix((z["B_data"], z["B_indices"], z["B_indptr"]), shape=shape)


def diag_positions(a: sp.csr_matrix) -> np.ndarray:
    """Return the position of each diagonal entry inside ``a.data`` (sorted CSR)."""
    n = a.shape[0]
    rows = np.repeat(np.arange(n), np.diff(a.indptr))
    pos = np.flatnonzero(a.indices == rows)
    if pos.size != n or not np.array_equal(rows[pos], np.arange(n)):
        msg = "matrix has a missing diagonal entry; the omega shift needs a full diagonal"
        raise ValueError(msg)
    return pos


def build_kf(dump: dict[str, Any], f: float, f0: float, pos: np.ndarray) -> sp.csr_matrix:
    """Return K(f) = K(f0) - 2*pi*i*(f-f0)*I in double, same pattern as the dump."""
    k0: sp.csr_matrix = dump["K"]
    data = k0.data.copy()
    data[pos] = data[pos] - 2j * np.pi * (f - f0)
    return sp.csr_matrix((data, k0.indices, k0.indptr), shape=k0.shape)


def arm_data(kf: sp.csr_matrix, dtype: Any, *, eq: bool) -> dict[str, Any]:  # noqa: ANN401
    """Cast K(f) (equilibrated or not) and its adjoint to the solver dtype.

    The two data arrays keep the entry order the solvers were built with:
    ``csr_int32`` of the matrix and of its conjugate transpose. The record
    also carries the scalings for the PC wrappers.
    """
    n = kf.shape[0]
    t = time.perf_counter()
    if eq:
        ks, r_sc, c_sc = equilibrate(kf)
    else:
        ks, r_sc, c_sc = kf, np.ones(n), np.ones(n)
    fwd = csr_int32(ks.astype(dtype))
    adj = csr_int32(ks.conj().T.tocsr().astype(dtype))
    return {
        "fwd_data": fwd.data,
        "adj_data": adj.data,
        "r": r_sc,
        "c": c_sc,
        "host_prep_s": time.perf_counter() - t,
        "ks": ks,
    }


def refactor(factor: CudssFactor, new_data: np.ndarray) -> float:
    """Load new values into the held pattern and refactorize; return seconds.

    The call must use ``reset_operands``: nvmath copies a host operand to the
    device when the caller sets it, not on each call (the job-2695 lesson).
    """
    factor.a.data[:] = new_data
    t = time.perf_counter()
    factor.solver.reset_operands(a=factor.a)
    factor.solver.factorize()
    return time.perf_counter() - t


def solve_chain(
    ksp: Any,  # noqa: ANN401  (PETSc KSP)
    amat: Any,  # noqa: ANN401  (PETSc Mat)
    kmat: sp.csr_matrix,
    b0: np.ndarray,
    nsolves: int,
    mask: sp.csr_matrix | None = None,
) -> dict[str, Any]:
    """Run ``nsolves`` FGMRES solves, each RHS the normalized previous solution.

    With ``mask`` (the dump's B), each next RHS is ``B @ x`` normalized — the
    production family; without it the chain feeds general vectors.
    """
    its: list[int] = []
    worst = 0.0
    b = b0 / np.linalg.norm(b0)
    t = time.perf_counter()
    for _ in range(nsolves):
        r = ksp_solve(ksp, amat, b, kmat, None)
        its.append(r["its"])
        worst = max(worst, r["true_residual"])
        x = r["x"] if mask is None else np.asarray(mask @ r["x"]).reshape(-1)
        b = x / np.linalg.norm(x)
    return {"wall_s": time.perf_counter() - t, "its": its, "worst_true_residual": float(worst)}


def run_arm(  # noqa: PLR0915  (one arm is one measured protocol; splitting it would scatter the timeline)
    dump: dict[str, Any],
    ladder: list[float],
    cfg: dict[str, Any],
    mods: dict[str, Any],
) -> dict[str, Any]:
    """Measure one precision arm along the f ladder; return the full record."""
    dtype, eq, args = cfg["dtype"], cfg["eq"], cfg["args"]
    petsc = mods["petsc"]
    f0 = float(dump["meta"]["f"])
    pos = diag_positions(dump["K"])
    rec: dict[str, Any] = {"dtype": np.dtype(dtype).name, "equilibrated": eq, "points": {}}
    mem0 = device_memory_used_mib()

    kf0 = build_kf(dump, f0, f0, pos)
    d0 = arm_data(kf0, dtype, eq=eq)
    fwd = CudssFactor(d0["ks"], dtype, mods["nvmath"], args)
    adj = CudssFactor(d0["ks"].conj().T.tocsr(), dtype, mods["nvmath"], args)
    fwd.plan()
    adj.plan()
    fwd.factorize()
    adj.factorize()
    rec["plan_s"] = fwd.plan_s
    rec["adj_plan_s"] = adj.plan_s
    rec["factor_s"] = fwd.factor_s
    rec["adj_factor_s"] = adj.factor_s
    rec["lu_nnz"] = fwd.lu_nnz
    # Warm repeats on unchanged values: how much of "factor" is one-time setup.
    rec["refactor_same_values_s"] = [refactor(fwd, d0["fwd_data"]) for _ in range(2)]

    for f in ladder:
        pt: dict[str, Any] = {}
        kf = build_kf(dump, f, f0, pos)
        d = arm_data(kf, dtype, eq=eq)
        pt["host_prep_s"] = d["host_prep_s"]
        if f != f0:
            pt["refactor_s"] = refactor(fwd, d["fwd_data"])
            pt["adj_refactor_s"] = refactor(adj, d["adj_data"])
        t = time.perf_counter()
        amat = petsc_mat(kf, petsc)
        kmat_h = kf.conj().T.tocsr()
        amat_h = petsc_mat(kmat_h, petsc)
        pt["petsc_setup_s"] = time.perf_counter() - t
        pc_f = ScaledFactorPC(lambda v, _fw=fwd: _fw.solve(v), d["r"], d["c"])
        pc_a = ScaledFactorPC(lambda v, _ad=adj: _ad.solve(v), d["c"], d["r"])
        ksp = make_fgmres(amat, petsc, pc_f, args)
        ksp_h = make_fgmres(amat_h, petsc, pc_a, args)
        b0 = dump["rhs"]["forcing"]
        mask = cfg.get("mask")
        pt["forward"] = solve_chain(ksp, amat, kf, b0, args.nsolves, mask)
        pt["adjoint"] = solve_chain(ksp_h, amat_h, kmat_h, np.conj(b0), args.nsolves, mask)
        pt["line_point_s"] = (
            pt.get("refactor_s", 0.0)
            + pt.get("adj_refactor_s", 0.0)
            + pt["forward"]["wall_s"]
            + pt["adjoint"]["wall_s"]
        )
        rec["points"][f"{f:g}"] = pt
        worst_res = max(pt["forward"]["worst_true_residual"], pt["adjoint"]["worst_true_residual"])
        ksp.destroy()
        ksp_h.destroy()
        amat.destroy()
        amat_h.destroy()
        sys.stdout.write(
            f"{rec['dtype']} eq={eq} f={f:g}: refac={pt.get('refactor_s', float('nan')):.2f}s "
            f"adj_refac={pt.get('adj_refactor_s', float('nan')):.2f}s "
            f"fwd_its={pt['forward']['its']} adj_its={pt['adjoint']['its']} "
            f"worst_res={worst_res:.1e} "
            f"line_point={pt['line_point_s']:.1f}s\n"
        )
        sys.stdout.flush()
    rec["device_mib_two_factors"] = device_memory_used_mib() - mem0
    fwd.free()
    adj.free()
    return rec


def verify_ladder(dump: dict[str, Any], verify: dict[str, Any], pos: np.ndarray) -> dict[str, Any]:
    """Check K(f_verify) built from the f0 dump against the independent dump."""
    fv = float(verify["meta"]["f"])
    f0 = float(dump["meta"]["f"])
    kv = build_kf(dump, fv, f0, pos)
    kd: sp.csr_matrix = verify["K"]
    same_pattern = np.array_equal(kv.indptr, kd.indptr) and np.array_equal(kv.indices, kd.indices)
    diff = float(np.abs(kv.data - kd.data).max()) if same_pattern else float("inf")
    return {"f_verify": fv, "same_pattern": bool(same_pattern), "max_abs_data_diff": diff}


def mumps_crosscheck(verify: dict[str, Any], petsc: Any) -> dict[str, Any]:  # noqa: ANN401  (petsc4py module)
    """MUMPS double solution at f_verify, for the accuracy comparison."""
    amat = petsc_mat(verify["K"], petsc)
    ksp = petsc.KSP().create(comm=amat.getComm())
    ksp.setOperators(amat)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    t = time.perf_counter()
    ksp.setUp()
    factor_s = time.perf_counter() - t
    b = verify["rhs"]["forcing"]
    r = ksp_solve(ksp, amat, b / np.linalg.norm(b), verify["K"], None)
    ksp.destroy()
    amat.destroy()
    return {"factor_s": factor_s, "x": r.pop("x"), "true_residual": r["true_residual"]}


def parse_args() -> argparse.Namespace:
    """Build the parser and parse ``sys.argv``."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--npz", type=Path, required=True, help="the f0 dump (beta line anchor)")
    ap.add_argument("--verify-npz", type=Path, default=None, help="independent dump on the ladder")
    ap.add_argument("--f-ladder", type=float, nargs="+", default=[1.0, 1.2, 1.4, 1.6, 1.8])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nsolves", type=int, default=10, help="forward and adjoint solves per f")
    ap.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    ap.add_argument("--max-it", type=int, default=MAX_IT_DEFAULT)
    ap.add_argument("--restart", type=int, default=RESTART_DEFAULT)
    ap.add_argument("--mt-lib", default=None, help="path to libcudss_mtlayer_gomp.so.0")
    ap.add_argument("--host-threads", type=int, default=10)
    ap.add_argument("--matching-single", default="MAX_DIAG_PRODUCT")
    ap.add_argument("--matching", default=None, help="set per arm; CLI value used for double")
    ap.add_argument("--mask-chain", action="store_true", help="apply the dump B to each next RHS")
    ap.add_argument("--arms", nargs="+", default=["double", "single"], choices=("double", "single"))
    ap.add_argument("--tag", default="")
    return ap.parse_args()


def main() -> int:
    """Run the double and single arms along the ladder; write JSON and a summary."""
    args = parse_args()
    # Import mpi4py before petsc4py: mpi4py requests the MPI thread level the GPU venv needs.
    from mpi4py import MPI  # noqa: PLC0415

    if MPI.COMM_WORLD.Get_size() != 1:
        msg = "bench_cudss_line.py runs on one rank"
        raise SystemExit(msg)
    import petsc4py  # noqa: PLC0415

    petsc4py.init(sys.argv[:1])
    import nvmath  # noqa: PLC0415
    from petsc4py import PETSc  # noqa: PLC0415

    mods = {"petsc": PETSc, "nvmath": nvmath}
    args.out.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    dump = load_dump(args.npz)
    pos = diag_positions(dump["K"])
    ladder = [float(f) for f in args.f_ladder]
    rec: dict[str, Any] = {
        "meta": dump["meta"],
        "f_ladder": ladder,
        "nsolves": args.nsolves,
        "mt_lib": args.mt_lib,
        "host_threads": args.host_threads,
    }
    if args.verify_npz:
        verify = load_dump(args.verify_npz)
        rec["ladder_check"] = verify_ladder(dump, verify, pos)
        sys.stdout.write(f"ladder check: {json.dumps(rec['ladder_check'])}\n")
        mc = mumps_crosscheck(verify, PETSc)
        x_mumps = mc.pop("x")
        rec["mumps_crosscheck"] = mc

    mask = load_b_mask(args.npz) if args.mask_chain else None
    rec["mask_chain"] = bool(args.mask_chain)
    arms: dict[str, dict[str, Any]] = {
        "double": {"dtype": np.complex128, "eq": False, "args": args, "mask": mask},
        "single": {"dtype": np.complex64, "eq": True, "args": args, "mask": mask},
    }
    arms = {k: v for k, v in arms.items() if k in args.arms}
    for name, cfg in arms.items():
        args.matching = args.matching_single if name == "single" else None
        rec[name] = run_arm(dump, ladder, cfg, mods)

    if args.verify_npz:
        # Accuracy of the double arm at f_verify against MUMPS: rebuild and solve once.
        fv = rec["ladder_check"]["f_verify"]
        kv = build_kf(dump, fv, float(dump["meta"]["f"]), pos)
        d = arm_data(kv, np.complex128, eq=False)
        fwd = CudssFactor(d["ks"], np.complex128, nvmath, args)
        fwd.plan()
        fwd.factorize()
        amat = petsc_mat(kv, PETSc)
        ksp = make_fgmres(amat, PETSc, ScaledFactorPC(fwd.solve, d["r"], d["c"]), args)
        b = verify["rhs"]["forcing"]
        r = ksp_solve(ksp, amat, b / np.linalg.norm(b), kv, x_mumps)
        rec["accuracy_vs_mumps_at_f_verify"] = {
            "rel_err": r.get("rel_err_vs_ref"),
            "its": r["its"],
            "true_residual": r["true_residual"],
        }
        ksp.destroy()
        amat.destroy()
        fwd.free()

    (args.out / f"cudss_line_bench{args.tag}.job{job}.json").write_text(
        json.dumps(rec, indent=1, default=str)
    )
    sys.stdout.write(
        f"accuracy vs MUMPS at f_verify: {json.dumps(rec.get('accuracy_vs_mumps_at_f_verify'))}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
