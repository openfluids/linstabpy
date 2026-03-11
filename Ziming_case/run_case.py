from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from linstab2d.utils.petsc_init import init_petsc_from_argv

init_petsc_from_argv()

from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

from linstab2d.mesh import Mesh  # noqa: E402
from linstab2d.operators.assembly import build_lnse_operator  # noqa: E402
from linstab2d.operators.energy_norm import compute_chu_norm  # noqa: E402
from linstab2d.solvers.eigenvalue import temporal_eigenvalues, _eigs_near_target  # noqa: E402
from linstab2d.solvers.resolvent import resolvent_analysis  # noqa: E402

from .baseflow_loader import load_my_case  # noqa: E402
from .bc_setup import my_bc_setup  # noqa: E402


def _get_memory_mb() -> float:
    """Return current process memory (approximate, in MiB)."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux, ru_maxrss is in kilobytes.
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


def run_case(
    case_dir: str | Path,
    *,
    # Defaults chosen to mirror replace_original_parameters.sh
    Re: float = 1.0e5,
    Ma: float = 2.15,
    Pr: float = 0.72,
    kappa: float = 1.4,
    T_0: float = 153.33,
    # Defaults chosen to mirror macro_single.sh
    beta: float = 19.2,
    St: float = 0.0,
    n_modes_stability: int = 20,
    n_modes_resolvent: int = 10,
    run_stability: bool = True,
    run_resolvent: bool = False,
    fd_order: int = 2,
    stride_ksi: int = 1,
    stride_eta: int = 1,
    penalty: float = 1e8,
    alpha_filter: float | str = "none",
    output_npz: str = "template_case_output.npz",
    use_raw_eigs: bool = False,
    log_path: str | Path | None = None,
) -> None:
    """Minimal driver to run LNSE resolvent/stability on an arbitrary case."""
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    t0 = time.perf_counter()
    mem0 = _get_memory_mb()
    step_stats: List[Tuple[str, float, float]] = []

    # 1) Load baseflow and sponge from .npy files
    t_start = time.perf_counter()
    mem_start = _get_memory_mb()
    X, Y, baseflow, sponge = load_my_case(
        case_dir,
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        T_0=T_0,
        stride_ksi=stride_ksi,
        stride_eta=stride_eta,
    )
    t_end = time.perf_counter()
    mem_end = _get_memory_mb()
    step_stats.append(("load_baseflow", t_end - t_start, mem_end - mem_start))

    ny, nx = X.shape

    # 2) Build a simple computational mesh and deform to physical coordinates
    t_start = time.perf_counter()
    mem_start = _get_memory_mb()
    cmesh = Mesh.create(
        (0.0, 2.0),
        (0.0, 1.0),
        nx=nx,
        ny=ny,
        fd_order=fd_order,
        alpha_filter=alpha_filter,
    )
    mesh = cmesh.deform(X_new=X, Y_new=Y)
    mesh.sponge = sponge
    t_end = time.perf_counter()
    mem_end = _get_memory_mb()
    step_stats.append(("build_mesh", t_end - t_start, mem_end - mem_start))

    # 3) Assemble LNSE operator and apply BCs
    t_start = time.perf_counter()
    mem_start = _get_memory_mb()
    L0, idx = build_lnse_operator(mesh, baseflow, kx=beta, model="2D")
    L0, bc_indices = my_bc_setup(mesh, idx, L0, penalty)
    t_end = time.perf_counter()
    mem_end = _get_memory_mb()
    step_stats.append(("assemble_operator_bc", t_end - t_start, mem_end - mem_start))

    # 4) Chu energy norm
    t_start = time.perf_counter()
    mem_start = _get_memory_mb()
    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO_p = baseflow.RHO.ravel(order="F")[p]
    T_p = baseflow.T.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(W_int, RHO_p, T_p, baseflow.kappa, baseflow.Ma, mesh.ngp)
    t_end = time.perf_counter()
    mem_end = _get_memory_mb()
    step_stats.append(("chu_norm", t_end - t_start, mem_end - mem_start))

    omega_target = 2.0 * np.pi * St

    gains = np.array([], dtype=float)
    responses = np.empty((mesh.ngp * 5, 0), dtype=complex)
    forcings = np.empty((mesh.ngp * 5, 0), dtype=complex)

    if run_resolvent:
        t_start = time.perf_counter()
        mem_start = _get_memory_mb()
        B_diag = np.ones(mesh.ngp * 5, dtype=float)
        B_diag[bc_indices] = 0.0
        B_mat = sp.diags(B_diag, format="csc")
        C_mat = B_mat.copy()
        gains, responses, forcings = resolvent_analysis(
            L0,
            omega_target,
            n_modes_resolvent,
            W_diag,
            B_mat=B_mat,
            C_mat=C_mat,
            filter_fn=mesh.filters.filter if mesh.filters is not None else None,
            filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
        )
        t_end = time.perf_counter()
        mem_end = _get_memory_mb()
        step_stats.append(("resolvent", t_end - t_start, mem_end - mem_start))

    omega = np.array([], dtype=complex)
    direct_modes = np.empty((mesh.ngp * 5, 0), dtype=complex)

    if run_stability:
        t_start = time.perf_counter()
        mem_start = _get_memory_mb()
        if use_raw_eigs:
            omega, direct_modes = _eigs_near_target(
                L0.tocsc(),
                n_modes=n_modes_stability,
                target=omega_target,
            )
            order = np.argsort(np.abs(omega - omega_target))
            omega = omega[order]
            direct_modes = direct_modes[:, order]
        else:
            omega, direct_modes = temporal_eigenvalues(
                L0,
                n_modes=n_modes_stability,
                target=omega_target,
                W_diag=W_diag,
            )
        t_end = time.perf_counter()
        mem_end = _get_memory_mb()
        step_stats.append(("stability", t_end - t_start, mem_end - mem_start))

    # 5) Save results on rank 0
    if rank == 0:
        t_start = time.perf_counter()
        mem_start = _get_memory_mb()
        out_path = Path(output_npz)
        np.savez_compressed(
            out_path,
            omega=omega,
            direct_modes=direct_modes,
            gains=gains,
            responses=responses,
            forcings=forcings,
            omega_target=omega_target,
            beta=beta,
            St=St,
            Re=Re,
            Ma=Ma,
            Pr=Pr,
            kappa=kappa,
            T_0=T_0,
            X=mesh.X,
            Y=mesh.Y,
            U=baseflow.U,
            sponge=mesh.sponge,
            W_diag=W_diag,
            invW_diag=invW_diag,
        )
        t_end = time.perf_counter()
        mem_end = _get_memory_mb()
        step_stats.append(("save_output", t_end - t_start, mem_end - mem_start))

        elapsed = time.perf_counter() - t0
        mem_total = _get_memory_mb()

        print(f"[template] run finished in {elapsed:.2f}s, output={out_path}")
        print("[template] step timings (s) and memory deltas (MiB):")
        for name, dt, dmem in step_stats:
            print(f"  - {name:20s}: {dt:8.3f} s, Δmem={dmem:8.2f} MiB")
        if not np.isnan(mem_total):
            print(f"[template] approximate current/peak memory: {mem_total:.2f} MiB")
        if not np.isnan(mem0):
            print(f"[template] overall Δmem since start: {mem_total - mem0:.2f} MiB")

        # Write timing and memory stats to .log for comparison with other codes
        log_file = Path(log_path) if log_path is not None else out_path.with_suffix(".log")
        with open(log_file, "w") as f:
            f.write(f"# linstab2d template run log\n")
            f.write(f"# generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} UTC\n")
            f.write(f"case_dir={case_dir}\n")
            f.write(f"output_npz={out_path}\n")
            f.write(f"Re={Re} Ma={Ma} Pr={Pr} kappa={kappa} T_0={T_0}\n")
            f.write(f"beta={beta} St={St} fd_order={fd_order} stride_ksi={stride_ksi} stride_eta={stride_eta}\n")
            f.write(f"nx={nx} ny={ny} ngp={mesh.ngp}\n")
            f.write(f"n_modes_stability={n_modes_stability} n_modes_resolvent={n_modes_resolvent}\n")
            f.write(f"run_stability={run_stability} run_resolvent={run_resolvent} use_raw_eigs={use_raw_eigs}\n")
            f.write(f"total_time_s={elapsed:.6f}\n")
            f.write(f"peak_memory_mib={mem_total:.4f}\n")
            if not np.isnan(mem0):
                f.write(f"memory_delta_mib={mem_total - mem0:.4f}\n")
            f.write("# step name | time_s | memory_delta_mib\n")
            for name, dt, dmem in step_stats:
                f.write(f"step {name} {dt:.6f} {dmem:.4f}\n")
        print(f"[template] stats written to {log_file}")

    MPI.COMM_WORLD.Barrier()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal LNSE/resolvent/stability template case.")
    parser.add_argument("--case-dir", type=str, required=True, help="Directory with x/y/rho/u/v/w/T/sponge .npy files.")
    parser.add_argument("--Re", type=float, default=1.0e5)
    parser.add_argument("--Ma", type=float, default=2.15)
    parser.add_argument("--Pr", type=float, default=0.72)
    parser.add_argument("--kappa", type=float, default=1.4)
    parser.add_argument("--T0", type=float, default=153.33)
    parser.add_argument("--beta", type=float, default=19.2)
    parser.add_argument("--St", type=float, default=0.0)
    parser.add_argument("--n-modes-stability", type=int, default=20)
    parser.add_argument("--n-modes-resolvent", type=int, default=10)
    parser.add_argument("--no-stability", action="store_true", help="Disable stability solve.")
    parser.add_argument("--no-resolvent", action="store_true", help="Disable resolvent solve.")
    parser.add_argument("--fd-order", type=int, default=2)
    parser.add_argument("--stride-ksi", type=int, default=1)
    parser.add_argument("--stride-eta", type=int, default=1)
    parser.add_argument("--penalty", type=float, default=1e8)
    parser.add_argument("--alpha-filter", type=str, default="none")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz path (default: <case_dir>_St<St>_beta<beta>.npz, e.g. Ziming_case_St0.0_beta19.2.npz)",
    )
    parser.add_argument("--log", type=str, default=None, help="Path for .log file (default: same stem as --output with .log)")
    parser.add_argument("--raw-eigs", action="store_true", help="Use raw eigensolve on L0 instead of temporal.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Default output basename from case name and key params (e.g. Ziming_case_St0.0_beta19.2.npz)
    if args.output is None:
        case_name = Path(args.case_dir).name
        args.output = f"{case_name}_St{args.St}_beta{args.beta}.npz"
    run_case(
        case_dir=args.case_dir,
        Re=args.Re,
        Ma=args.Ma,
        Pr=args.Pr,
        kappa=args.kappa,
        T_0=args.T0,
        beta=args.beta,
        St=args.St,
        n_modes_stability=args.n_modes_stability,
        n_modes_resolvent=args.n_modes_resolvent,
        run_stability=not args.no_stability,
        run_resolvent=not args.no_resolvent,
        fd_order=args.fd_order,
        stride_ksi=args.stride_ksi,
        stride_eta=args.stride_eta,
        penalty=args.penalty,
        alpha_filter=args.alpha_filter,
        output_npz=args.output,
        use_raw_eigs=args.raw_eigs,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()

