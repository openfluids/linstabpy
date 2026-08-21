#!/usr/bin/env python
# Copyright (c) 2026 Ricardo Frantz. Benchmark tool for the linstabpy backend measurements.
"""cuDSS timings in FP64 and FP32 on one GPU.

For each dumped operator K (see ``dump_operator.py``) the script measures,
through ``nvmath-python`` (cuDSS):

1. analysis (reordering), factorization and one-RHS solve time, in complex128
   and in complex64;
2. device memory after one factor and after two (K and K^H). The adjoint
   needs its own factor: cuDSS has no transpose solve;
3. PETSc FGMRES in double, right-preconditioned by the cuDSS factor, for
   both precisions. It records iterations to ``rtol`` and the true residual.
   The PC structure is the CPU rehearsal's;
4. the adjoint identity ``<K^-1 a, b> = <a, K^-H b>`` through the two factors.

The MUMPS double solve through PETSc is the reference. Output: one JSON per
dump and one figure. Run through sbatch on c9 (one rank, one GPU)::

    python bench_cudss.py --npz DIR/K_*.npz --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

RTOL_DEFAULT = 1.0e-10
MAX_IT_DEFAULT = 200
RESTART_DEFAULT = 30
N_SOLVE_REPEATS = 5
PROBES = 3


def load_dump(path: Path) -> dict[str, Any]:
    """Load one ``.npz`` dump into a dict with the CSR operator and its vectors."""
    z = np.load(path, allow_pickle=False)
    n = int(z["K_indptr"].shape[0] - 1)
    K = sp.csr_matrix((z["K_data"], z["K_indices"], z["K_indptr"]), shape=(n, n))  # noqa: N806
    return {
        "K": K,
        "rhs": {"random": z["rhs_random"], "forcing": z["rhs_forcing"]},
        "meta": json.loads(str(z["meta"])),
        "n": n,
    }


def equilibrate(K: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:  # noqa: N803
    """Return ``D_r K D_c`` and the two scalings (inf-norm rows, then columns), in double."""
    r = 1.0 / np.maximum(np.asarray(abs(K).max(axis=1).todense()).ravel(), 1e-300)
    Ks = sp.diags(r) @ K  # noqa: N806
    c = 1.0 / np.maximum(np.asarray(abs(Ks).max(axis=0).todense()).ravel(), 1e-300)
    Ks = (Ks @ sp.diags(c)).tocsr()  # noqa: N806
    return Ks, r, c


def device_memory_used_mib() -> float:
    """Device memory in use (MiB) on the visible GPU, from ``nvidia-smi``; NaN if unavailable."""
    smi = shutil.which("nvidia-smi")
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if smi is None:
        return float("nan")
    try:
        out = subprocess.check_output(  # noqa: S603
            [smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", dev], text=True
        )
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return float("nan")


def csr_int32(kmat: sp.csr_matrix) -> sp.csr_matrix:
    """Sorted CSR with int32 index arrays (one shared entry order for value updates)."""
    out = sp.csr_matrix(
        (kmat.data, kmat.indices.astype(np.int32), kmat.indptr.astype(np.int32)),
        shape=kmat.shape,
    )
    out.sort_indices()
    return out


class CudssFactor:
    """One cuDSS factorization of a CSR matrix, in the dtype asked for.

    Holds a persistent host right-hand-side buffer; ``solve`` writes into it
    and calls cuDSS. Host operands: nvmath moves them to the device.
    """

    def __init__(
        self,
        kmat: sp.csr_matrix,
        dtype: Any,  # noqa: ANN401  (numpy dtype)
        nvmath: Any,  # noqa: ANN401  (the nvmath module)
        args: argparse.Namespace,
    ) -> None:
        """Create the solver object; nothing runs until ``plan``/``factorize``."""
        self.dtype = dtype
        self.a = csr_int32(kmat.astype(dtype))
        self.b = np.zeros(kmat.shape[0], dtype=dtype)
        adv = nvmath.sparse.advanced
        execution = adv.ExecutionCUDA(device_id=0)
        options = None
        mt_lib = getattr(args, "mt_lib", None)
        if mt_lib:
            options = adv.DirectSolverOptions(multithreading_lib=str(mt_lib))
        self.solver = adv.DirectSolver(self.a, self.b, options=options, execution=execution)
        pc = self.solver.plan_config
        # Optional knobs come through getattr: not every caller's CLI defines them.
        reordering = getattr(args, "reordering", None)
        if reordering:
            pc.reordering_algorithm = getattr(adv.DirectSolverReorderingAlg, reordering)
        matching = getattr(args, "matching", None)
        if matching:
            pc.matching_algorithm = getattr(nvmath.bindings.cudss.MatchingAlg, matching)
        host_threads = getattr(args, "host_threads", None)
        if host_threads:
            pc.host_nthreads = int(host_threads)
        self.solver.solution_config.ir_num_steps = 0
        self.plan_s = float("nan")
        self.factor_s = float("nan")
        self.lu_nnz: int | None = None
        self.memory_estimates: Any = None
        self.calls = 0
        self.seconds = 0.0

    def plan(self) -> None:
        """Run the analysis (reordering) phase once per sparsity pattern."""
        t = time.perf_counter()
        info = self.solver.plan()
        self.plan_s = time.perf_counter() - t
        try:
            self.memory_estimates = [int(v) for v in np.asarray(info.memory_estimates).ravel()]
        except (AttributeError, TypeError, ValueError):
            self.memory_estimates = None

    def factorize(self) -> None:
        """Run the numerical factorization."""
        t = time.perf_counter()
        info = self.solver.factorize()
        self.factor_s = time.perf_counter() - t
        try:
            self.lu_nnz = int(info.lu_nnz)
        except (AttributeError, TypeError, ValueError):
            self.lu_nnz = None

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve for one right-hand side (cast in, cuDSS, cast out to complex128)."""
        t = time.perf_counter()
        self.b[:] = rhs
        # A host operand is copied to the device when it is (re)set, not read
        # per call: without reset_operands every solve reuses the FIRST copy
        # (measured: all-zero solutions, job 2695). See nvmath example05.
        self.solver.reset_operands(b=self.b)
        x = self.solver.solve()
        out = np.asarray(x).reshape(-1).astype(np.complex128)
        self.calls += 1
        self.seconds += time.perf_counter() - t
        return out

    def free(self) -> None:
        """Release the device resources."""
        self.solver.free()


class ScaledFactorPC:
    """petsc4py Python PC: ``y = D_c * F^{-1} (D_r * x)``."""

    def __init__(self, solve: Any, r: np.ndarray, c: np.ndarray) -> None:  # noqa: ANN401  (callable)
        """Keep the solve callable and the scalings."""
        self.solve = solve
        self.r = r
        self.c = c

    def apply(self, pc: Any, x: Any, y: Any) -> None:  # noqa: ANN401, ARG002  (PETSc objects; pc unused by design)
        """Apply the preconditioner once."""
        z = self.solve(self.r * x.getArray(readonly=True))
        y.setArray(self.c * z)


def petsc_mat(kmat: sp.csr_matrix, petsc: Any) -> Any:  # noqa: ANN401
    """Build a one-rank PETSc AIJ matrix from a scipy CSR matrix."""
    n = kmat.shape[0]
    amat = petsc.Mat().createAIJ(
        size=(n, n), csr=(kmat.indptr.astype(np.int32), kmat.indices.astype(np.int32), kmat.data)
    )
    amat.assemble()
    return amat


def make_fgmres(amat: Any, petsc: Any, pc_ctx: ScaledFactorPC, args: argparse.Namespace) -> Any:  # noqa: ANN401
    """FGMRES, right-preconditioned by a Python PC, with the rehearsal's tolerances."""
    ksp = petsc.KSP().create(comm=amat.getComm())
    ksp.setOperators(amat)
    ksp.setType("fgmres")
    ksp.setPCSide(petsc.PC.Side.RIGHT)
    ksp.setGMRESRestart(args.restart)
    ksp.setTolerances(rtol=args.rtol, atol=1.0e-300, divtol=1.0e12, max_it=args.max_it)
    ksp.setConvergenceHistory(length=args.max_it + 2)
    ksp.getPC().setType(petsc.PC.Type.PYTHON)
    ksp.getPC().setPythonContext(pc_ctx)
    ksp.setUp()
    return ksp


def ksp_solve(
    ksp: Any,  # noqa: ANN401  (PETSc KSP; no stubs)
    amat: Any,  # noqa: ANN401  (PETSc Mat)
    b_np: np.ndarray,
    kmat: sp.csr_matrix,
    x_ref: np.ndarray | None,
) -> dict[str, Any]:
    """Solve with ``ksp``; record iterations, true residual, error vs reference, wall time."""
    b = amat.createVecRight()
    b.setArray(b_np)
    x = amat.createVecRight()
    t = time.perf_counter()
    ksp.solve(b, x)
    wall = time.perf_counter() - t
    xa = np.array(x.getArray(readonly=True), copy=True)
    out: dict[str, Any] = {
        "its": int(ksp.getIterationNumber()),
        "reason": int(ksp.getConvergedReason()),
        "true_residual": float(np.linalg.norm(kmat @ xa - b_np) / np.linalg.norm(b_np)),
        "wall_s": wall,
    }
    if ksp.getType() != "preonly":
        hist = np.asarray(ksp.getConvergenceHistory(), dtype=float)
        out["residual_history_rel"] = (hist / max(hist[0], 1e-300)).tolist() if hist.size else []
    if x_ref is not None:
        out["rel_err_vs_ref"] = float(np.linalg.norm(xa - x_ref) / np.linalg.norm(x_ref))
    out["x"] = xa
    return out


def mumps_reference(
    amat: Any,  # noqa: ANN401  (PETSc Mat)
    petsc: Any,  # noqa: ANN401  (petsc4py module)
    rhs: dict[str, np.ndarray],
    kmat: sp.csr_matrix,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """MUMPS in double through PETSc: factor time, per-RHS solve and residual, solutions."""
    t = time.perf_counter()
    ksp = petsc.KSP().create(comm=amat.getComm())
    ksp.setOperators(amat)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setUp()
    rec: dict[str, Any] = {"factor_s": time.perf_counter() - t, "rhs": {}}
    refs: dict[str, np.ndarray] = {}
    for name, b in rhs.items():
        r = ksp_solve(ksp, amat, b, kmat, None)
        refs[name] = r.pop("x")
        rec["rhs"][name] = r
    ksp.destroy()
    return rec, refs


def adjoint_defect(fwd: Any, adj: Any, n: int) -> float:  # noqa: ANN401
    """Max relative defect of ``<fwd(a), b> - <a, adj(b)>`` over random probe pairs."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(PROBES):
        a = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        lhs = np.vdot(b, fwd(a))
        rhs = np.vdot(adj(b), a)
        worst = max(worst, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300))
    return float(worst)


def run_precision(  # noqa: PLR0913  (the inputs are the point, its solver, the settings and the references)
    dump: dict[str, Any],
    amat: Any,  # noqa: ANN401  (PETSc Mat)
    refs: dict[str, np.ndarray],
    *,
    dtype: Any,  # noqa: ANN401  (numpy dtype)
    eq: bool,
    args: argparse.Namespace,
    mods: dict[str, Any],
) -> dict[str, Any]:
    """Factor K (and K^H) with cuDSS in ``dtype``; time it; run FGMRES and the adjoint check."""
    K: sp.csr_matrix = dump["K"]  # noqa: N806
    n = dump["n"]
    petsc, nvmath = mods["petsc"], mods["nvmath"]
    if eq:
        Ks, r_sc, c_sc = equilibrate(K)  # noqa: N806
    else:
        Ks, r_sc, c_sc = K, np.ones(n), np.ones(n)  # noqa: N806
    rec: dict[str, Any] = {"dtype": np.dtype(dtype).name, "equilibrated": eq, "rhs": {}}
    mem0 = device_memory_used_mib()

    fwd = CudssFactor(Ks, dtype, nvmath, args)
    fwd.plan()
    fwd.factorize()
    mem1 = device_memory_used_mib()
    rec.update(
        plan_s=fwd.plan_s,
        factor_s=fwd.factor_s,
        lu_nnz=fwd.lu_nnz,
        memory_estimates=fwd.memory_estimates,
    )
    rec["device_mib_one_factor"] = mem1 - mem0

    # one-RHS solve timing (repeat; the first call may include lazy setup)
    b0 = dump["rhs"]["forcing"]
    times = []
    for _ in range(N_SOLVE_REPEATS):
        t = time.perf_counter()
        fwd.solve((r_sc * b0).astype(dtype))
        times.append(time.perf_counter() - t)
    rec["solve_s_first"] = times[0]
    rec["solve_s_mean_after_first"] = float(np.mean(times[1:])) if len(times) > 1 else float("nan")

    # FGMRES in double with the cuDSS factor as right preconditioner
    pc_fwd = ScaledFactorPC(lambda v: fwd.solve(v.astype(dtype)), r_sc, c_sc)
    ksp = make_fgmres(amat, petsc, pc_fwd, args)
    for name, b in dump["rhs"].items():
        x32 = c_sc * fwd.solve((r_sc * b).astype(dtype))
        plain = float(np.linalg.norm(K @ x32 - b) / np.linalg.norm(b))
        fwd.calls, fwd.seconds = 0, 0.0
        r = ksp_solve(ksp, amat, b, K, refs[name])
        r.pop("x")
        r["plain_factor_true_residual"] = plain
        r["pc_applies"] = fwd.calls
        r["pc_apply_s_mean"] = fwd.seconds / max(fwd.calls, 1)
        rec["rhs"][name] = r

    rec.update(
        adjoint_check(K, Ks, (r_sc, c_sc), dtype=dtype, args=args, mods=mods, fwd_ksp=(ksp, amat))
    )
    rec["device_mib_two_factors"] = rec.pop("_mem_two") - mem0
    ksp.destroy()
    fwd.free()
    return rec


def adjoint_check(  # noqa: PLR0913  (the inputs are the operator, its scaled form, the settings and the forward solver)
    K: sp.csr_matrix,  # noqa: N803  (paper name)
    Ks: sp.csr_matrix,  # noqa: N803
    scalings: tuple[np.ndarray, np.ndarray],
    *,
    dtype: Any,  # noqa: ANN401  (numpy dtype)
    args: argparse.Namespace,
    mods: dict[str, Any],
    fwd_ksp: tuple[Any, Any],
) -> dict[str, Any]:
    """Factor K^H with cuDSS (second factor) and check the adjoint identity through FGMRES.

    With ``Ks = D_r K D_c`` and real diagonal scalings, ``K^H x = b`` is
    ``x = D_r Ks^H^{-1} (D_c b)``; the adjoint PC applies exactly that.
    """
    r_sc, c_sc = scalings
    petsc, nvmath = mods["petsc"], mods["nvmath"]
    ksp, amat = fwd_ksp
    n = K.shape[0]
    adj = CudssFactor(Ks.conj().T.tocsr(), dtype, nvmath, args)
    adj.plan()
    adj.factorize()
    rec: dict[str, Any] = {
        "adj_plan_s": adj.plan_s,
        "adj_factor_s": adj.factor_s,
        "adj_lu_nnz": adj.lu_nnz,
        "_mem_two": device_memory_used_mib(),
    }
    kmat_h = K.conj().T.tocsr()
    amat_h = petsc_mat(kmat_h, petsc)
    pc_adj = ScaledFactorPC(lambda v: adj.solve(v.astype(dtype)), c_sc, r_sc)
    ksp_h = make_fgmres(amat_h, petsc, pc_adj, args)

    def solve_fwd(v: np.ndarray) -> np.ndarray:
        return ksp_solve(ksp, amat, v, K, None)["x"]

    def solve_adj(v: np.ndarray) -> np.ndarray:
        return ksp_solve(ksp_h, amat_h, v, kmat_h, None)["x"]

    rec["adjoint_identity_defect"] = adjoint_defect(solve_fwd, solve_adj, n)
    rec["adjoint_fgmres_its_last"] = int(ksp_h.getIterationNumber())
    ksp_h.destroy()
    amat_h.destroy()
    adj.free()
    return rec


def summarize(rec: dict[str, Any]) -> str:
    """Lines a reader can compare against the acceptance criteria."""
    m = rec["meta"]
    head = f"beta={m['beta']:g} f={m['f']:g} n={rec['n']}"
    lines = [f"{head} MUMPS64 factor={rec['mumps']['factor_s']:.1f}s"]
    for label, p in rec["cudss"].items():
        d = p["rhs"]["forcing"]
        lines.append(
            f"{head} cuDSS {label}: plan={p['plan_s']:.1f}s factor={p['factor_s']:.1f}s "
            f"solve={p['solve_s_mean_after_first']:.3f}s lu_nnz={p['lu_nnz']} "
            f"mem1={p['device_mib_one_factor']:.0f}MiB mem2={p['device_mib_two_factors']:.0f}MiB "
            f"FGMRES(forcing) its={d['its']} res={d['true_residual']:.1e} "
            f"err={d.get('rel_err_vs_ref', float('nan')):.1e} "
            f"plain={d['plain_factor_true_residual']:.1e} "
            f"adjoint_defect={p['adjoint_identity_defect']:.1e}"
        )
    return "\n".join(lines)


def plot(results: list[dict[str, Any]], out: Path) -> None:
    """Plot factor times and FGMRES iterations per point and variant (measured data)."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = [f"β={r['meta']['beta']:g}\nf={r['meta']['f']:g}" for r in results]
    variants = sorted({k for r in results for k in r["cudss"]})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    xi = np.arange(len(labels))
    w = 0.8 / (len(variants) + 1)
    axes[0].bar(
        xi - 0.4 + w / 2, [r["mumps"]["factor_s"] for r in results], w, label="MUMPS FP64 (1 rank)"
    )
    for j, v in enumerate(variants, start=1):
        axes[0].bar(
            xi - 0.4 + w * (j + 0.5),
            [r["cudss"][v]["factor_s"] for r in results],
            w,
            label=f"cuDSS {v}",
        )
        axes[1].bar(
            xi - 0.4 + w * (j + 0.5),
            [r["cudss"][v]["rhs"]["forcing"]["its"] for r in results],
            w,
            label=f"cuDSS {v}",
        )
        axes[2].bar(
            xi - 0.4 + w * (j + 0.5),
            [r["cudss"][v]["device_mib_two_factors"] / 1024 for r in results],
            w,
            label=f"cuDSS {v}",
        )
    axes[0].set_ylabel("factorization time [s]")
    axes[1].set_ylabel("FGMRES iterations to rtol 1e-10 (forcing RHS)")
    axes[1].axhline(8, color="k", ls="--", lw=0.8)
    axes[2].set_ylabel("device memory, two factors resident [GiB]")
    axes[2].axhline(16, color="k", ls="--", lw=0.8, label="16 GiB card")
    for ax in axes:
        ax.set_xticks(xi, labels)
        ax.legend(fontsize=7)
    fig.suptitle("cuDSS on one RTX A6000, Savarino 300x300 (measured)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)


def parse_args() -> argparse.Namespace:
    """Build the command-line parser and parse ``sys.argv``."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--npz", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    ap.add_argument("--max-it", type=int, default=MAX_IT_DEFAULT)
    ap.add_argument("--restart", type=int, default=RESTART_DEFAULT)
    ap.add_argument(
        "--precisions", nargs="+", default=["double", "single"], choices=("double", "single")
    )
    ap.add_argument(
        "--equilibrate", choices=("both", "on", "off"), default="both", help="for single only"
    )
    ap.add_argument(
        "--reordering",
        default=None,
        help="DirectSolverReorderingAlg member name, e.g. NESTED_DISSECTION",
    )
    ap.add_argument(
        "--matching", default=None, help="cudss.MatchingAlg member name (cuDSS scaling/matching)"
    )
    ap.add_argument("--host-threads", type=int, default=None)
    ap.add_argument("--mt-lib", default=None, help="path to libcudss_mtlayer_gomp.so.0")
    ap.add_argument("--tag", default="")
    return ap.parse_args()


def main() -> int:
    """Run every dump, write JSON + figure, print the summary."""
    args = parse_args()

    # Import mpi4py before petsc4py: mpi4py requests the MPI thread level the GPU venv needs.
    from mpi4py import MPI  # noqa: PLC0415

    if MPI.COMM_WORLD.Get_size() != 1:
        msg = "bench_cudss.py runs on one rank"
        raise SystemExit(msg)
    import petsc4py  # noqa: PLC0415

    petsc4py.init(sys.argv[:1])
    import nvmath  # noqa: PLC0415
    from petsc4py import PETSc  # noqa: PLC0415

    mods = {"petsc": PETSc, "nvmath": nvmath}
    args.out.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    results: list[dict[str, Any]] = []
    for path in args.npz:
        dump = load_dump(path)
        amat = petsc_mat(dump["K"], PETSc)
        mumps, refs = mumps_reference(amat, PETSc, dump["rhs"], dump["K"])
        rec: dict[str, Any] = {
            "meta": dump["meta"],
            "n": dump["n"],
            "nnz": int(dump["K"].nnz),
            "mumps": mumps,
            "cudss": {},
        }
        for prec in args.precisions:
            dtype = np.complex128 if prec == "double" else np.complex64
            eqs = (
                [False]
                if prec == "double"
                else ({"both": [False, True], "on": [True], "off": [False]}[args.equilibrate])
            )
            for eq in eqs:
                label = f"{prec}" + ("_eq" if eq else "")
                rec["cudss"][label] = run_precision(
                    dump, amat, refs, dtype=dtype, eq=eq, args=args, mods=mods
                )
                sys.stdout.write(f"done {path.name} {label}\n")
                sys.stdout.flush()
        amat.destroy()
        rec["dump"] = str(path)
        rec["job"] = job
        rec["device"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        rec["nvmath_version"] = getattr(nvmath, "__version__", "?")
        rec["settings"] = {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "npz"
        }
        results.append(rec)
        stem = path.stem.replace("K_", "")
        (args.out / f"cudss_bench_{stem}{args.tag}.job{job}.json").write_text(
            json.dumps(rec, indent=1, default=str)
        )
        sys.stdout.write(summarize(rec) + "\n")
        sys.stdout.flush()
    (args.out / f"cudss_bench_summary{args.tag}.job{job}.txt").write_text(
        "\n".join(summarize(r) for r in results) + "\n"
    )
    plot(results, args.out / f"cudss_bench{args.tag}.job{job}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
