#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-only
# Benchmark tool for the linstabpy backend measurements.
"""STRUMPACK and MUMPS preconditioner variants: levers that need no new production code.

Three questions, on the dumped operators of ``dump_operator.py``:

1. STRUMPACK on the GPU with loose block low-rank (BLR) compression as the
   right preconditioner of FGMRES. The iteration owns the accuracy. We record
   factor time, memory and iteration count against the exact factor.
2. MUMPS on the CPU with loose BLR (``icntl_35 = 2``, ``cntl_7 = tol``) in the
   same role.
3. What ``MatSolveTranspose`` does on a STRUMPACK factor. PETSc registers no
   transposed solve for that backend. When the factor's ``symmetric`` flag is
   "unknown", ``MatSolveTranspose`` (matrix.c) runs the forward solve instead.
   The probe compares the "transposed" solution with the forward one.

Run on one rank in the GPU (STRUMPACK) venv, on c9 through sbatch::

    mpiexec -n 1 python bench_strumpack.py --npz dumps/K_*.npz --out results
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
    device_memory_used_mib,
    ksp_solve,
    load_dump,
    petsc_mat,
)

N_APPLY_REPEATS = 3


def lu_ksp(
    amat: Any,  # noqa: ANN401  (PETSc Mat)
    petsc: Any,  # noqa: ANN401  (petsc4py module)
    *,
    prefix: str,
    solver: str,
    options: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, float]:
    """FGMRES, right-preconditioned by an LU of ``solver``; return (ksp, factor seconds).

    ``options`` are written to the PETSc options database before ``setUp``.
    MUMPS reads its options at symbolic factorization, after PETSc has given
    the factor the PC prefix, so ``<prefix>mat_mumps_*`` works. STRUMPACK reads
    its options inside ``MatGetFactor_aij_strumpack`` (PETSc 3.24.2), BEFORE
    the prefix exists, so its keys must be global: every STRUMPACK variant
    writes the full key set unprefixed and no key leaks between variants.
    """
    opts = petsc.Options()
    for key, val in options.items():
        opts[f"{prefix}{key}"] = val
        if key.startswith("mat_strumpack_"):
            opts[key] = val
    ksp = petsc.KSP().create(comm=amat.getComm())
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(amat)
    ksp.setType("fgmres")
    ksp.setPCSide(petsc.PC.Side.RIGHT)
    ksp.setGMRESRestart(args.restart)
    ksp.setTolerances(rtol=args.rtol, atol=1.0e-300, divtol=1.0e12, max_it=args.max_it)
    ksp.setConvergenceHistory(length=args.max_it + 2)
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType(solver)
    ksp.setFromOptions()
    t = time.perf_counter()
    ksp.setUp()
    pc.setUp()
    return ksp, time.perf_counter() - t


def pc_apply_timing(ksp: Any, amat: Any, b_np: np.ndarray, kmat: sp.csr_matrix) -> dict[str, Any]:  # noqa: ANN401
    """Time one preconditioner application (a plain LU solve) and its true residual."""
    pc = ksp.getPC()
    b = amat.createVecRight()
    b.setArray(b_np)
    x = amat.createVecRight()
    pc.apply(b, x)  # warm-up (first call may allocate)
    t = time.perf_counter()
    for _ in range(N_APPLY_REPEATS):
        pc.apply(b, x)
    wall = (time.perf_counter() - t) / N_APPLY_REPEATS
    xa = np.array(x.getArray(readonly=True), copy=True)
    res = float(np.linalg.norm(kmat @ xa - b_np) / np.linalg.norm(b_np))
    return {"pc_apply_s": wall, "plain_factor_true_residual": res}


def transpose_probe(ksp: Any, amat: Any, b_np: np.ndarray, kmat: sp.csr_matrix) -> dict[str, Any]:  # noqa: ANN401
    """Apply the factor's transposed solve and say which system it actually solved.

    The record holds residuals against ``K^T x = b`` and ``K x = b``. A small
    second residual means ``MatSolveTranspose`` ran the forward solve.
    """
    pc = ksp.getPC()
    b = amat.createVecRight()
    b.setArray(b_np)
    x = amat.createVecRight()
    try:
        pc.applyTranspose(b, x)
    except Exception as exc:  # noqa: BLE001  (the exception IS the measurement)
        return {"transpose_apply_error": repr(exc)}
    xa = np.array(x.getArray(readonly=True), copy=True)
    nb = np.linalg.norm(b_np)
    return {
        "res_vs_KT": float(np.linalg.norm(kmat.T @ xa - b_np) / nb),
        "res_vs_KH": float(np.linalg.norm(kmat.conj().T @ xa - b_np) / nb),
        "res_vs_K": float(np.linalg.norm(kmat @ xa - b_np) / nb),
    }


def mumps_stats(ksp: Any) -> dict[str, Any]:  # noqa: ANN401
    """MUMPS flop count (RINFOG 3) and factor entries (INFOG 29) if the API exposes them."""
    out: dict[str, Any] = {}
    try:
        fmat = ksp.getPC().getFactorMatrix()
        out["mumps_rinfog3_flops"] = float(fmat.getMumpsRinfog(3))
        out["mumps_infog29_factor_entries"] = int(fmat.getMumpsInfog(29))
    except Exception as exc:  # noqa: BLE001  (optional)
        out["mumps_stats_error"] = repr(exc)
    return out


def run_variant(
    amat: Any,  # noqa: ANN401  (PETSc Mat)
    petsc: Any,  # noqa: ANN401  (petsc4py module)
    dump: dict[str, Any],
    variant: tuple[str, str, dict[str, Any], bool],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Factor, time one LU apply, run FGMRES on the forcing RHS, optionally probe the transpose."""
    prefix, solver, options, probe_transpose = variant
    mem0 = device_memory_used_mib()
    ksp, t_fact = lu_ksp(amat, petsc, prefix=prefix, solver=solver, options=options, args=args)
    mem1 = device_memory_used_mib()
    kmat = dump["K"]
    b_np = dump["rhs"]["forcing"]
    rec: dict[str, Any] = {
        "solver": solver,
        "options": options,
        "factor_s": t_fact,
        "device_mib_factor": mem1 - mem0,
    }
    rec.update(pc_apply_timing(ksp, amat, b_np, kmat))
    if solver == "mumps":
        rec.update(mumps_stats(ksp))
    sol = ksp_solve(ksp, amat, b_np, kmat, None)
    sol.pop("x")
    rec["fgmres"] = sol
    if probe_transpose:
        rec["transpose_probe"] = transpose_probe(ksp, amat, b_np, kmat)
    ksp.destroy()
    return rec


def summarize(rec: dict[str, Any]) -> str:
    """One line per variant for the log."""
    m = rec["meta"]
    head = f"beta={m['beta']:g} f={m['f']:g}"
    lines = []
    for name, v in rec["variants"].items():
        if "error" in v:
            lines.append(f"{head} {name}: ERROR {v['error']}")
            continue
        fg = v["fgmres"]
        line = (
            f"{head} {name}: factor={v['factor_s']:.1f}s apply={v['pc_apply_s']:.3f}s "
            f"plain_res={v['plain_factor_true_residual']:.1e} "
            f"FGMRES its={fg['its']} reason={fg['reason']} "
            f"res={fg['true_residual']:.1e} wall={fg['wall_s']:.1f}s "
            f"mem={v['device_mib_factor']:.0f}MiB"
        )
        tp = v.get("transpose_probe")
        if tp:
            line += (
                f" | transpose probe: res_vs_KT={tp.get('res_vs_KT', float('nan')):.1e} "
                f"res_vs_KH={tp.get('res_vs_KH', float('nan')):.1e} "
                f"res_vs_K={tp.get('res_vs_K', float('nan')):.1e} "
                f"err={tp.get('transpose_apply_error', '')}"
            )
        lines.append(line)
    return "\n".join(lines)


def variants(args: argparse.Namespace) -> list[tuple[str, str, dict[str, Any], bool]]:
    """Return the (name, solver, options, probe_transpose) list this run measures."""

    def sp_opts(*, gpu: bool, compression: str, tol: float) -> dict[str, Any]:
        # Full key set every time: STRUMPACK keys are global (see lu_ksp), so a
        # variant must overwrite everything the previous one set.
        return {
            "mat_strumpack_gpu": gpu,
            "mat_strumpack_verbose": True,
            "mat_strumpack_compression": compression,
            "mat_strumpack_compression_rel_tol": tol,
            "mat_strumpack_compression_min_sep_size": args.min_sep_size,
            "mat_strumpack_compression_leaf_size": args.leaf_size,
        }

    out: list[tuple[str, str, dict[str, Any], bool]] = [
        ("strumpack_gpu_exact", "strumpack", sp_opts(gpu=True, compression="NONE", tol=0.0), True),
    ]
    for i, tol in enumerate(args.blr_tols):
        out.append(
            (
                f"strumpack_gpu_blr{i}_{tol:g}",
                "strumpack",
                sp_opts(gpu=True, compression="BLR", tol=tol),
                False,
            )
        )
    if args.cpu_blr_tol is not None:
        out.append(
            (
                f"strumpack_cpu_blr_{args.cpu_blr_tol:g}",
                "strumpack",
                sp_opts(gpu=False, compression="BLR", tol=args.cpu_blr_tol),
                False,
            )
        )
    if not args.skip_mumps:
        out.append(("mumps_exact", "mumps", {}, True))
        for i, tol in enumerate(args.mumps_blr_tols):
            out.append(
                (
                    f"mumps_blr{i}_{tol:g}",
                    "mumps",
                    {"mat_mumps_icntl_35": 2, "mat_mumps_cntl_7": tol},
                    False,
                )
            )
    return out


def parse_args() -> argparse.Namespace:
    """Build the parser and parse ``sys.argv``."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--npz", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    ap.add_argument("--max-it", type=int, default=MAX_IT_DEFAULT)
    ap.add_argument("--restart", type=int, default=RESTART_DEFAULT)
    ap.add_argument("--blr-tols", type=float, nargs="*", default=[1.0e-2, 1.0e-4])
    ap.add_argument("--cpu-blr-tol", type=float, default=None)
    ap.add_argument("--min-sep-size", type=int, default=256)
    ap.add_argument("--leaf-size", type=int, default=128)
    ap.add_argument("--mumps-blr-tols", type=float, nargs="*", default=[1.0e-3, 1.0e-5])
    ap.add_argument("--skip-mumps", action="store_true")
    ap.add_argument("--tag", default="")
    return ap.parse_args()


def main() -> int:
    """Run every dump through every variant; write JSON and print the summary."""
    args = parse_args()
    # Import mpi4py before petsc4py: mpi4py requests the MPI thread level the GPU venv needs.
    from mpi4py import MPI  # noqa: PLC0415

    if MPI.COMM_WORLD.Get_size() != 1:
        msg = "bench_strumpack.py runs on one rank"
        raise SystemExit(msg)
    import petsc4py  # noqa: PLC0415

    petsc4py.init(sys.argv[:1])
    from petsc4py import PETSc  # noqa: PLC0415

    args.out.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    results: list[dict[str, Any]] = []
    for path in args.npz:
        dump = load_dump(path)
        amat = petsc_mat(dump["K"], PETSc)
        rec: dict[str, Any] = {"meta": dump["meta"], "n": dump["n"], "variants": {}}
        for k, (name, solver, options, probe) in enumerate(variants(args)):
            try:
                rec["variants"][name] = run_variant(
                    amat, PETSc, dump, (f"v{k}_", solver, options, probe), args
                )
            except Exception as exc:  # noqa: BLE001  (one failing variant must not end the run)
                rec["variants"][name] = {"error": repr(exc), "solver": solver, "options": options}
            sys.stdout.write(
                summarize({"meta": rec["meta"], "variants": {name: rec["variants"][name]}}) + "\n"
            )
            sys.stdout.flush()
        results.append(rec)
        stem = path.stem.replace("K_", "")
        (args.out / f"strumpack_bench_{stem}{args.tag}.job{job}.json").write_text(
            json.dumps(rec, indent=1, default=str)
        )
        amat.destroy()
    (args.out / f"strumpack_bench_summary{args.tag}.job{job}.txt").write_text(
        "\n".join(summarize(r) for r in results) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
