#!/usr/bin/env python
# Copyright (c) 2026 Ricardo Frantz. Benchmark tool for the linstabpy backend measurements.
"""FP32 factor inside FP64 FGMRES, on the CPU.

For each dumped operator K (see ``dump_operator.py``) the script:

1. solves ``K x = b`` with MUMPS in double through PETSc (the reference);
2. runs PETSc FGMRES, right-preconditioned by that MUMPS factor (the CPU
   production path; expect 1-2 iterations);
3. factorizes ``K`` in complex64 with SuperLU (``scipy.sparse.linalg.splu``).
   That factor becomes a petsc4py Python PC inside the same FGMRES. The run
   repeats with and without double-precision equilibration ``D_r K D_c``.

Per right-hand side it records: iterations to ``rtol``; the true residual
in double; the error against the MUMPS solution; the wall times. It needs
no GPU and no new dependency. Output: one JSON per dump and one figure.

Run through sbatch on c9 (one rank, MKL venv)::

    python bench_fp32.py --npz DIR/K_*.npz --out DIR
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
import scipy.sparse.linalg as spla

RTOL_DEFAULT = 1.0e-10
MAX_IT_DEFAULT = 200
RESTART_DEFAULT = 30
PERMC_DEFAULT = "COLAMD"


def load_dump(path: Path) -> dict[str, Any]:
    """Load one ``.npz`` dump into a dict with the CSR operator and its vectors."""
    z = np.load(path, allow_pickle=False)
    n = int(z["K_indptr"].shape[0] - 1)
    K = sp.csr_matrix((z["K_data"], z["K_indices"], z["K_indptr"]), shape=(n, n))  # noqa: N806
    meta = json.loads(str(z["meta"]))
    return {
        "K": K,
        "rhs": {"random": z["rhs_random"], "forcing": z["rhs_forcing"]},
        "meta": meta,
        "n": n,
    }


def equilibrate(K: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:  # noqa: N803
    """Return ``D_r K D_c`` and the two scalings (inf-norm rows, then columns), in double."""
    r = 1.0 / np.maximum(np.asarray(abs(K).max(axis=1).todense()).ravel(), 1e-300)
    Ks = sp.diags(r) @ K  # noqa: N806
    c = 1.0 / np.maximum(np.asarray(abs(Ks).max(axis=0).todense()).ravel(), 1e-300)
    Ks = (Ks @ sp.diags(c)).tocsr()  # noqa: N806
    return Ks, r, c


def row_spread(K: sp.csr_matrix) -> dict[str, float]:  # noqa: N803
    """Inf-norm row spread of a matrix: max, min, decades."""
    row = np.asarray(abs(K).max(axis=1).todense()).ravel()
    pos = row[row > 0]
    return {
        "max": float(pos.max()),
        "min": float(pos.min()),
        "decades": float(np.log10(pos.max() / pos.min())),
    }


class ScaledFactorPC:
    """petsc4py Python PC: ``y = D_c * F^{-1} (D_r * x)`` with ``F`` an LU in another precision."""

    def __init__(self, solve: Any, dtype: Any, r: np.ndarray, c: np.ndarray) -> None:  # noqa: ANN401  (callable, numpy dtype)
        """Keep the solve callable, its working dtype and the scalings."""
        self.solve = solve
        self.dtype = dtype
        self.r = r
        self.c = c
        self.calls = 0
        self.seconds = 0.0

    def apply(self, pc: Any, x: Any, y: Any) -> None:  # noqa: ANN401, ARG002  (PETSc objects; pc unused by design)
        """Apply the preconditioner once."""
        t = time.perf_counter()
        xa = x.getArray(readonly=True)
        z = self.solve((self.r * xa).astype(self.dtype))
        y.setArray(self.c * np.asarray(z).astype(np.complex128))
        self.calls += 1
        self.seconds += time.perf_counter() - t


def kappa1_estimates(
    K: sp.csr_matrix,  # noqa: N803  (paper name)
    ref: Any,  # noqa: ANN401  (PETSc KSP with a MUMPS LU; MUMPS has a true transposed solve)
    amat: Any,  # noqa: ANN401  (PETSc Mat)
) -> dict[str, float]:
    """Estimate the 1-norm condition number of ``K`` and of its equilibrated form.

    ``onenormest`` needs ``K^-1 v`` and ``K^-H v``; both come from the MUMPS
    factor (``solve`` and the conjugated ``solveTranspose``). The equilibrated
    matrix ``Ks = D_r K D_c`` reuses the same factor through the scalings.
    """
    n = K.shape[0]
    b = amat.createVecRight()
    x = amat.createVecRight()

    def inv(v: np.ndarray) -> np.ndarray:
        b.setArray(v)
        ref.solve(b, x)
        return np.array(x.getArray(readonly=True), copy=True)

    def inv_h(v: np.ndarray) -> np.ndarray:
        b.setArray(np.conj(v))
        ref.solveTranspose(b, x)
        return np.conj(np.array(x.getArray(readonly=True), copy=True))

    ks, r, c = equilibrate(K)
    op = spla.LinearOperator((n, n), matvec=inv, rmatvec=inv_h, dtype=np.complex128)
    op_s = spla.LinearOperator(
        (n, n),
        matvec=lambda v: inv(v / r) / c,
        rmatvec=lambda v: inv_h(v / c) / r,
        dtype=np.complex128,
    )
    out = {
        "kappa1_K": float(spla.onenormest(K) * spla.onenormest(op)),
        "kappa1_K_equilibrated": float(spla.onenormest(ks) * spla.onenormest(op_s)),
    }
    b.destroy()
    x.destroy()
    return out


def petsc_mat(K: sp.csr_matrix, petsc: Any) -> Any:  # noqa: ANN401, N803
    """Build a one-rank PETSc AIJ matrix from a scipy CSR matrix."""
    n = K.shape[0]
    amat = petsc.Mat().createAIJ(
        size=(n, n), csr=(K.indptr.astype(np.int32), K.indices.astype(np.int32), K.data)
    )
    amat.assemble()
    return amat


def make_ksp(A: Any, petsc: Any, ksp_type: str, rtol: float, max_it: int, restart: int) -> Any:  # noqa: ANN401, N803
    """Create a right-preconditioned KSP of the given type on ``A``."""
    ksp = petsc.KSP().create(comm=A.getComm())
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    if ksp_type != "preonly":
        ksp.setPCSide(petsc.PC.Side.RIGHT)
        ksp.setGMRESRestart(restart)
        ksp.setTolerances(rtol=rtol, atol=1.0e-300, divtol=1.0e12, max_it=max_it)
        ksp.setConvergenceHistory(length=max_it + 2)
    return ksp


def solve_and_record(
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
    res = float(np.linalg.norm(kmat @ xa - b_np) / np.linalg.norm(b_np))
    out: dict[str, Any] = {
        "its": int(ksp.getIterationNumber()),
        "reason": int(ksp.getConvergedReason()),
        "true_residual": res,
        "wall_s": wall,
    }
    if ksp.getType() != "preonly":
        hist = np.asarray(ksp.getConvergenceHistory(), dtype=float)
        out["residual_history_rel"] = (hist / max(hist[0], 1e-300)).tolist() if hist.size else []
    if x_ref is not None:
        out["rel_err_vs_ref"] = float(np.linalg.norm(xa - x_ref) / np.linalg.norm(x_ref))
    out["x"] = xa
    return out


def fp32_variant(  # noqa: PLR0913  (the inputs are the point, its solver and the settings)
    K: sp.csr_matrix,  # noqa: N803  (paper name)
    amat: Any,  # noqa: ANN401  (PETSc Mat)
    rhs: dict[str, np.ndarray],
    refs: dict[str, np.ndarray],
    *,
    eq: bool,
    args: argparse.Namespace,
    petsc: Any,  # noqa: ANN401  (petsc4py module)
) -> dict[str, Any]:
    """Factor ``K`` in complex64 (SuperLU) and run it as the FGMRES preconditioner."""
    n = K.shape[0]
    if eq:
        Ks, r_sc, c_sc = equilibrate(K)  # noqa: N806
    else:
        Ks, r_sc, c_sc = K, np.ones(n), np.ones(n)  # noqa: N806
    spread_eq = row_spread(Ks)
    t = time.perf_counter()
    lu32 = spla.splu(
        Ks.astype(np.complex64).tocsc(), permc_spec=args.permc, diag_pivot_thresh=args.pivot_thresh
    )
    t_fact = time.perf_counter() - t
    nnz_lu = int(lu32.L.nnz + lu32.U.nnz)
    v: dict[str, Any] = {
        "equilibrated": eq,
        "row_spread_after": spread_eq,
        "permc_spec": args.permc,
        "diag_pivot_thresh": args.pivot_thresh,
        "factor32_s": t_fact,
        "nnz_LU": nnz_lu,
        "LU_bytes_complex64": nnz_lu * 8,
        "rhs": {},
    }
    pc_ctx = ScaledFactorPC(lu32.solve, np.complex64, r_sc, c_sc)
    ksp = make_ksp(amat, petsc, "fgmres", args.rtol, args.max_it, args.restart)
    ksp.getPC().setType(petsc.PC.Type.PYTHON)
    ksp.getPC().setPythonContext(pc_ctx)
    ksp.setUp()
    for name, b in rhs.items():
        # plain FP32 solve, no refinement: what the factor alone gives
        x32 = c_sc * lu32.solve((r_sc * b).astype(np.complex64)).astype(np.complex128)
        plain = float(np.linalg.norm(K @ x32 - b) / np.linalg.norm(b))
        pc_ctx.calls, pc_ctx.seconds = 0, 0.0
        r = solve_and_record(ksp, amat, b, K, refs[name])
        r.pop("x")
        r["plain_fp32_true_residual"] = plain
        r["pc_applies"] = pc_ctx.calls
        r["pc_apply_s_mean"] = pc_ctx.seconds / max(pc_ctx.calls, 1)
        v["rhs"][name] = r
    ksp.destroy()
    del lu32
    return v


def run_point(dump: dict[str, Any], args: argparse.Namespace, petsc: Any) -> dict[str, Any]:  # noqa: ANN401
    """Run the reference, the FP64 baseline and the FP32 variants for one dump."""
    K: sp.csr_matrix = dump["K"]  # noqa: N806
    meta = dump["meta"]
    n = dump["n"]
    rec: dict[str, Any] = {
        "meta": meta,
        "n": n,
        "nnz": int(K.nnz),
        "row_spread": row_spread(K),
        "rhs": {},
    }
    A = petsc_mat(K, petsc)  # noqa: N806

    # Reference: MUMPS in double, one rank (the CPU arm's factorization).
    t = time.perf_counter()
    ref = make_ksp(A, petsc, "preonly", args.rtol, args.max_it, args.restart)
    ref.getPC().setType("lu")
    ref.getPC().setFactorSolverType("mumps")
    ref.setUp()
    rec["mumps_factor_s"] = time.perf_counter() - t
    refs: dict[str, np.ndarray] = {}
    for name, b in dump["rhs"].items():
        r = solve_and_record(ref, A, b, K, None)
        refs[name] = r.pop("x")
        rec["rhs"].setdefault(name, {})["mumps64_preonly"] = r
    t = time.perf_counter()
    rec["condition"] = kappa1_estimates(K, ref, A)
    rec["condition"]["estimate_s"] = time.perf_counter() - t

    # FP64 baseline: FGMRES right-preconditioned by the MUMPS factor (production CPU path).
    base = make_ksp(A, petsc, "fgmres", args.rtol, args.max_it, args.restart)
    base.getPC().setType("lu")
    base.getPC().setFactorSolverType("mumps")
    base.setUp()
    for name, b in dump["rhs"].items():
        r = solve_and_record(base, A, b, K, refs[name])
        r.pop("x")
        rec["rhs"][name]["fgmres_mumps64"] = r
    base.destroy()
    ref.destroy()

    # FP32 variants: SuperLU complex64, with and without double-precision equilibration.
    if args.equilibrate == "both":
        variants = {"off": False, "on": True}
    else:
        variants = {args.equilibrate: args.equilibrate == "on"}
    rec["fp32"] = {}
    for label, eq in variants.items():
        try:
            rec["fp32"][label] = fp32_variant(
                K, A, dump["rhs"], refs, eq=eq, args=args, petsc=petsc
            )
        except (SystemError, MemoryError, RuntimeError) as exc:
            # SuperLU reports a failed workspace expansion as SystemError; keep the point.
            rec["fp32"][label] = {"error": repr(exc), "permc_spec": args.permc}
    A.destroy()
    return rec


def summarize(results: list[dict[str, Any]]) -> str:
    """One line per (point, variant, rhs): the numbers the acceptance check reads."""
    lines = []
    for rec in results:
        m = rec["meta"]
        dec = rec["row_spread"]["decades"]
        cond = rec.get("condition", {})
        head = (
            f"beta={m['beta']:g} f={m['f']:g} n={rec['n']} rowspread={dec:.1f}dec "
            f"kappa1={cond.get('kappa1_K', float('nan')):.1e} "
            f"kappa1_eq={cond.get('kappa1_K_equilibrated', float('nan')):.1e}"
        )
        for name, d in rec["rhs"].items():
            fb = d["fgmres_mumps64"]
            lines.append(
                f"{head} rhs={name} FGMRES+MUMPS64: its={fb['its']} res={fb['true_residual']:.1e}"
            )
        for label, v in rec["fp32"].items():
            if "error" in v:
                lines.append(f"{head} FP32 eq={label}: ERROR {v['error']}")
                continue
            for name, d in v["rhs"].items():
                err = d.get("rel_err_vs_ref", float("nan"))
                lines.append(
                    f"{head} rhs={name} FP32 eq={label}: its={d['its']} "
                    f"res={d['true_residual']:.1e} err={err:.1e} "
                    f"plain32={d['plain_fp32_true_residual']:.1e} "
                    f"factor32={v['factor32_s']:.1f}s nnzLU={v['nnz_LU']:.2e} reason={d['reason']}"
                )
    return "\n".join(lines)


def plot(results: list[dict[str, Any]], out: Path) -> None:
    """Plot iterations per point and variant, and the residual histories (measured data)."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels, its_off, its_on, its_64 = [], [], [], []
    for rec in results:
        m = rec["meta"]
        labels.append(f"β={m['beta']:g}\nf={m['f']:g}")
        its_64.append(rec["rhs"]["forcing"]["fgmres_mumps64"]["its"])
        its_off.append(
            rec["fp32"].get("off", {}).get("rhs", {}).get("forcing", {}).get("its", np.nan)
        )
        its_on.append(
            rec["fp32"].get("on", {}).get("rhs", {}).get("forcing", {}).get("its", np.nan)
        )
    xi = np.arange(len(labels))
    w = 0.27
    axes[0].bar(xi - w, its_64, w, label="FP64 MUMPS factor")
    axes[0].bar(xi, its_off, w, label="FP32 SuperLU, no equilibration")
    axes[0].bar(xi + w, its_on, w, label="FP32 SuperLU, equilibrated")
    axes[0].axhline(8, color="k", ls="--", lw=0.8, label="8 iterations")
    axes[0].set_xticks(xi, labels)
    axes[0].set_ylabel("FGMRES iterations to rtol 1e-10 (forcing RHS)")
    axes[0].legend(fontsize=8)
    for rec in results:
        m = rec["meta"]
        for label, v in rec["fp32"].items():
            h = v["rhs"]["forcing"].get("residual_history_rel", [])
            if h:
                axes[1].semilogy(h, label=f"β={m['beta']:g} f={m['f']:g} eq={label}")
    axes[1].axhline(1e-10, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("FGMRES iteration")
    axes[1].set_ylabel("true residual / initial (forcing RHS)")
    axes[1].legend(fontsize=7)
    fig.suptitle("FP32 factor as FGMRES preconditioner, Savarino 300x300 (measured)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)


def main() -> int:
    """Run every dump, write JSON + figure, print the summary."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--npz", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    ap.add_argument("--max-it", type=int, default=MAX_IT_DEFAULT)
    ap.add_argument("--restart", type=int, default=RESTART_DEFAULT)
    ap.add_argument("--equilibrate", choices=("both", "on", "off"), default="both")
    ap.add_argument("--permc", default=PERMC_DEFAULT, help="SuperLU column ordering")
    ap.add_argument(
        "--pivot-thresh",
        type=float,
        default=1.0,
        help="SuperLU diag_pivot_thresh (1 = partial pivoting)",
    )
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # Import mpi4py before petsc4py: mpi4py requests the MPI thread level the venvs need.
    from mpi4py import MPI  # noqa: PLC0415

    if MPI.COMM_WORLD.Get_size() != 1:
        msg = "bench_fp32.py runs on one rank"
        raise SystemExit(msg)
    import petsc4py  # noqa: PLC0415

    petsc4py.init(sys.argv[:1])
    from petsc4py import PETSc  # noqa: PLC0415

    args.out.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    results: list[dict[str, Any]] = []
    for path in args.npz:
        dump = load_dump(path)
        rec = run_point(dump, args, PETSc)
        rec["dump"] = str(path)
        rec["job"] = job
        rec["settings"] = {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "npz"
        }
        results.append(rec)
        stem = path.stem.replace("K_", "")
        (args.out / f"fp32_bench_{stem}{args.tag}.job{job}.json").write_text(
            json.dumps(rec, indent=1)
        )
        sys.stdout.write(summarize([rec]) + "\n")
        sys.stdout.flush()
    (args.out / f"fp32_bench_summary{args.tag}.job{job}.txt").write_text(summarize(results) + "\n")
    plot(results, args.out / f"fp32_bench{args.tag}.job{job}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
