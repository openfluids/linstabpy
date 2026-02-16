#!/usr/bin/env python3
"""Benchmark sparse direct solves and resolvent assembly scaling.

Benchmarks:

- LU factorization time (SuperLU + optional scikit-umfpack path)
- forward + adjoint LU solve times
- resolvent timing (LU + ARPACK setup stage)
- per-case memory footprint

Solver policy:
    - SuperLU is the default baseline and remains enabled even when UMFPACK is
      available.
    - UMFPACK is retained as an optional path for targeted performance checks
      when SciPy has scikit-umfpack detection available.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg._dsolve import linsolve

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linstab2d.baseflow import load_blasius, sutherland_air
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.operators.energy_norm import compute_chu_norm
Backend = Literal["superlu", "umfpack"]

HAS_UMFPACK = not getattr(linsolve, "noScikit", True)

OUTPUT_PNG = SCRIPT.with_suffix(".png")
OUTPUT_CSV = SCRIPT.with_suffix(".csv")

DEFAULT_GRIDS = ((50, 50), (100, 100), (200, 100), (300, 100))
DEFAULT_OMEGA = 60.0


@dataclass(frozen=True)
class BenchmarkSpec:
    nx: int
    ny: int
    omega: float = DEFAULT_OMEGA


@dataclass
class BenchmarkResult:
    backend: Backend
    grid: str
    ndof: int
    nnz: int
    lu_factorization_ms: float
    lu_forward_solve_ms: float
    lu_adjoint_solve_ms: float
    resolvent_ms: float | None
    memory_mb: float


def _process_memory_mb() -> float:
    """Return current RSS memory in MB."""
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        return process.memory_info().rss / (1024**2)
    except Exception:
        import resource
        import sys

        ru = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return ru.ru_maxrss / (1024**2)
        return ru.ru_maxrss / 1024


def _set_backend(backend: Backend) -> None:
    """Force SciPy direct-solve backend selection."""
    # Keep SuperLU as default for robust runs; UMFPACK remains a controlled
    # optional performance path when requested by this benchmark flow.
    if backend == "superlu":
        if hasattr(linsolve, "useUmfpack"):
            linsolve.useUmfpack.u = False
        return

    if not HAS_UMFPACK:
        raise RuntimeError("scikit-umfpack is not installed")
    if hasattr(linsolve, "useUmfpack"):
        linsolve.useUmfpack.u = True


def _build_case(nx: int, ny: int):
    """Build one benchmark resolvent operator from the Blasius baseline."""
    Re = 6e5
    Ma = 0.3
    Pr = 0.7
    kappa = 1.4
    T_0 = 273.15

    fd_order = 4
    xrange = (5e-2, 1.25 + 5e-2)
    yrange = (-1.0, 1.0)

    cmesh = Mesh.create(xrange, yrange, int(nx), int(ny), fd_order)

    # Hanifi-like map used in example 1.
    a = 0.01 * 0.05 / (0.05 - 2 * 0.01)
    b = 1 + 2 * a / 0.05
    Y_phys = a * (1 + cmesh.Y) / (b - cmesh.Y)
    mesh = cmesh.deform(Y_new=Y_phys)

    baseflow = load_blasius(mesh.X, Y_phys, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0)
    sutherland_air(baseflow)

    L0, idx = build_lnse_operator(mesh, baseflow, kx=0.0)
    L0, bc_indices = apply_dirichlet(L0, idx, "lrbt", "uvwT", mesh.ngp)

    used_ind = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[used_ind]
    rho = baseflow.RHO.ravel(order="F")[used_ind]
    temp = baseflow.T.ravel(order="F")[used_ind]
    W_diag, _ = compute_chu_norm(W_int, rho, temp, kappa, Ma, mesh.ngp)

    B_diag = np.ones(5 * mesh.ngp)
    B_diag[idx["T_j"]] = 0.0
    B_diag[idx["rho_j"]] = 0.0
    B_diag[idx["w_j"]] = 0.0
    B_diag[bc_indices] = 0.0
    B_mat = sp.diags(B_diag).tocsc()
    C_mat = B_mat.copy()

    return L0, W_diag, B_mat, C_mat


def _bench_lu_stage(matrix: sp.spmatrix, backend: Backend, repeats: int = 1) -> tuple[float, float, float, float]:
    """Benchmark LU factorization and forward/adjoint solves."""
    if repeats < 1:
        repeats = 1

    mat = matrix.tocsc()
    rng = np.random.default_rng(12345)
    rhs = rng.standard_normal(mat.shape[0]) + 1j * rng.standard_normal(mat.shape[0])
    rhs2 = rng.standard_normal(mat.shape[0]) + 1j * rng.standard_normal(mat.shape[0])

    factor_sum = 0.0
    fwd_sum = 0.0
    adj_sum = 0.0
    _set_backend(backend)

    for _ in range(repeats):
        t0 = time.perf_counter()
        if backend == "superlu":
            lu = spla.splu(mat)
            t1 = time.perf_counter()
            _ = lu.solve(rhs)
            t2 = time.perf_counter()
            _ = lu.solve(rhs2, trans="H")
            t3 = time.perf_counter()
        else:
            solve = spla.factorized(mat)
            t1 = time.perf_counter()
            _ = solve(rhs)
            t2 = time.perf_counter()
            solve_adj = spla.factorized(mat.conj().T)
            t3 = time.perf_counter()
            _ = solve_adj(rhs2)
            t4 = time.perf_counter()

            fwd_sum += (t2 - t1) * 1_000
            adj_sum += (t4 - t3) * 1_000
            factor_sum += (t1 - t0) * 1_000
            continue

        factor_sum += (t1 - t0) * 1_000
        fwd_sum += (t2 - t1) * 1_000
        adj_sum += (t3 - t2) * 1_000

    factor_ms = factor_sum / repeats
    forward_ms = fwd_sum / repeats
    adjoint_ms = adj_sum / repeats
    mem_mb = _process_memory_mb()
    return factor_ms, forward_ms, adjoint_ms, mem_mb


def _bench_resolvent(
    matrix: sp.spmatrix,
    W_diag: np.ndarray,
    B_mat: sp.spmatrix,
    C_mat: sp.spmatrix,
    omega: float,
    backend: Backend,
) -> float | None:
    """Benchmark one-mode resolvent timing with the requested backend."""
    matrix = matrix.tocsc()
    n = matrix.shape[0]
    shifted = matrix - 1j * float(omega) * sp.eye(n, format="csc")
    Wd = np.asarray(W_diag)
    inv_Wd = 1.0 / Wd

    _set_backend(backend)
    t0 = time.perf_counter()

    if backend == "superlu":
        lu = spla.splu(shifted)

        def apply_H(v: np.ndarray) -> np.ndarray:
            out = B_mat @ v
            out = lu.solve(out)
            out = C_mat @ out
            return out

        def apply_Hadj(v: np.ndarray) -> np.ndarray:
            out = C_mat.conj().T @ v
            out = lu.solve(out, trans="H")
            out = B_mat.conj().T @ out
            return out

    else:
        solve = spla.factorized(shifted)
        solve_adj = spla.factorized(shifted.conj().T)

        def apply_H(v: np.ndarray) -> np.ndarray:
            out = B_mat @ v
            out = solve(out)
            out = C_mat @ out
            return out

        def apply_Hadj(v: np.ndarray) -> np.ndarray:
            out = C_mat.conj().T @ v
            out = solve_adj(out)
            out = B_mat.conj().T @ out
            return out

    def _matvec(v: np.ndarray) -> np.ndarray:
        out = apply_H(v)
        out = Wd * out
        out = apply_Hadj(out)
        return inv_Wd * out

    operator = LinearOperator((n, n), matvec=_matvec, dtype=complex)

    try:
        _ = spla.eigs(operator, k=1, which="LM", tol=np.finfo(float).eps)
    except Exception:
        # Keep benchmark running even if ARPACK fails on a challenging grid.
        return None

    t1 = time.perf_counter()
    return (t1 - t0) * 1_000


def run_benchmarks(specs: list[BenchmarkSpec], repeats: int, run_resolvent: bool) -> list[BenchmarkResult]:
    rows: list[BenchmarkResult] = []
    backends: list[Backend] = ["superlu"]
    if HAS_UMFPACK:
        # SuperLU first for baseline comparability; append UMFPACK only if present.
        backends.append("umfpack")
    else:
        print("scikit-umfpack not available; skipping UMFPACK path")

    for spec in specs:
        nx = int(spec.nx)
        ny = int(spec.ny)
        print(f"\nGrid {nx:4d}x{ny:4d} ...")

        L0, W_diag, B_mat, C_mat = _build_case(nx, ny)

        for backend in backends:
            print(f"  backend: {backend}")
            factor_ms, forward_ms, adj_ms, mem_mb = _bench_lu_stage(L0, backend, repeats=repeats)
            resolvent_ms = None
            if run_resolvent:
                resolvent_ms = _bench_resolvent(L0, W_diag, B_mat, C_mat, omega=spec.omega, backend=backend)

            rows.append(
                BenchmarkResult(
                    backend=backend,
                    grid=f"{nx}x{ny}",
                    ndof=L0.shape[0],
                    nnz=L0.nnz,
                    lu_factorization_ms=factor_ms,
                    lu_forward_solve_ms=forward_ms,
                    lu_adjoint_solve_ms=adj_ms,
                    resolvent_ms=resolvent_ms,
                    memory_mb=mem_mb,
                )
            )

    return rows


def _write_csv(rows: list[BenchmarkResult], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "grid",
                "ndof",
                "nnz",
                "lu_factorization_ms",
                "lu_forward_solve_ms",
                "lu_adjoint_solve_ms",
                "resolvent_ms",
                "memory_mb",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _plot(rows: list[BenchmarkResult], path: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return

    grids = sorted({row.grid for row in rows}, key=lambda s: tuple(int(x) for x in s.split("x")))
    backends = sorted({row.backend for row in rows})

    metrics = [
        ("LU factorization", "lu_factorization_ms"),
        ("LU forward solve", "lu_forward_solve_ms"),
        ("LU adjoint solve", "lu_adjoint_solve_ms"),
    ]

    fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(14, 4), sharex=True)
    fig.suptitle("linstab2d direct-solver benchmark")

    x = np.arange(len(grids))
    width = 0.35
    colors = {"superlu": "tab:blue", "umfpack": "tab:orange"}

    for ax, (label, field) in zip(axes[:-1], metrics):
        for i, backend in enumerate(backends):
            vals = []
            for grid in grids:
                match = [r for r in rows if r.grid == grid and r.backend == backend]
                if not match:
                    vals.append(float("nan"))
                else:
                    vals.append(float(getattr(match[0], field)))
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=backend, color=colors.get(backend))

        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(grids, rotation=20)
        ax.set_ylabel("milliseconds")
        ax.grid(True, alpha=0.3)

    mem_axis = axes[-1]
    for i, backend in enumerate(backends):
        vals = []
        for grid in grids:
            match = [r for r in rows if r.grid == grid and r.backend == backend]
            if not match:
                vals.append(float("nan"))
            else:
                vals.append(float(match[0].memory_mb))
        mem_axis.plot(x + (i - 0.5) * width, vals, marker="o", label=backend)

    mem_axis.set_title("Process RSS (snapshot)")
    mem_axis.set_xticks(x)
    mem_axis.set_xticklabels(grids, rotation=20)
    mem_axis.set_ylabel("MB")
    mem_axis.grid(True, alpha=0.3)
    mem_axis.legend()

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_grid_list(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"invalid grid '{token}', expected Nx x Ny")
        nx_str, ny_str = token.lower().split("x", 1)
        out.append((int(nx_str), int(ny_str)))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sparse LU/UMFPACK solve stages and resolvent")
    parser.add_argument(
        "--grids",
        default=",".join(f"{nx}x{ny}" for nx, ny in DEFAULT_GRIDS),
        help="Comma-separated list of Nx x Ny grids.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repetitions for LU timing.")
    parser.add_argument("--no-resolvent", action="store_true", help="Skip resolvent stage benchmark.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip writing the output plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_pairs = parse_grid_list(args.grids)
    if not grid_pairs:
        raise ValueError("No grids supplied; use --grids like '50x50,100x100'.")

    specs = [BenchmarkSpec(nx=nx, ny=ny, omega=DEFAULT_OMEGA) for nx, ny in grid_pairs]
    results = run_benchmarks(specs, repeats=args.repeats, run_resolvent=not args.no_resolvent)

    _write_csv(results, OUTPUT_CSV)
    print(f"Wrote table: {OUTPUT_CSV}")

    for row in results:
        res = "nan" if row.resolvent_ms is None else f"{row.resolvent_ms:.3f}"
        print(
            f"{row.grid} | {row.backend} | ndof={row.ndof} | "
            f"factor={row.lu_factorization_ms:.3f} ms | "
            f"solve_f={row.lu_forward_solve_ms:.3f} ms | "
            f"solve_a={row.lu_adjoint_solve_ms:.3f} ms | "
            f"res={res} ms | "
            f"rss={row.memory_mb:.1f} MB"
        )

    if not args.skip_plot:
        _plot(results, OUTPUT_PNG)
        print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
