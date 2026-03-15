"""Song & Hao (2023) / Robinet (2007) oblique SWBLI validation case.

Usage (from repo root):
    .venv/bin/python -m validation.song_2023.song_2023
"""
from __future__ import annotations

import argparse
import math
import os
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
from linstab2d.operators.assembly import build_lnse_system  # noqa: E402
from linstab2d.operators.energy_norm import compute_chu_norm  # noqa: E402
from linstab2d.solvers.eigenvalue import temporal_eigenvalues  # noqa: E402
from linstab2d.solvers.petsc_utils import scipy_to_petsc  # noqa: E402
from linstab2d.solvers.resolvent import resolvent_analysis  # noqa: E402
from linstab2d.utils.jsonc import load_jsonc, parse_complex  # noqa: E402
from linstab2d.utils.petsc_core import _petsc_matrix_mode, _use_distributed_matrix  # noqa: E402

from examples._helpers import (  # noqa: E402
    format_complex,
    plot_gain_bars,
    plot_scalar_panel,
    plot_signed_panel,
    save_figure,
)

from .baseflow_loader import load_my_case  # noqa: E402


SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")
OUTPUT_BASEFLOW = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_baseflow.png")
OUTPUT_SUPPORT = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_support_direct_modes.png")
OUTPUT_RESOLVENT = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_resolvent.png")
DEFAULT_CONFIG_PATH = SCRIPT.with_suffix(".jsonc")
MATLAB_REF_DIR = SCRIPT.parent / "matlab_reference"
MATLAB_EIG_DIR = MATLAB_REF_DIR / "EigModes"
MATLAB_SPLIT_DIR = MATLAB_REF_DIR / "ziming_split"
SONG_REF_DIR = SCRIPT.parent / "ref_results"


def _get_memory_mb() -> float:
    """Return current process memory (approximate, in MiB)."""
    try:
        import resource
        import sys

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS: ru_maxrss is in bytes; Linux: in kilobytes.
        if sys.platform == "darwin":
            return float(usage.ru_maxrss) / (1024.0 * 1024.0)
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


def _resolve_frequency_target(cfg: dict) -> tuple[complex, str]:
    """Resolve the spectral target frequency from JSONC configuration.

    Supported forms:
    - explicit ``omega_target`` (preferred, direct solver input)
    - legacy scalar ``St`` interpreted as ``omega_target = 2π St``
    - structured ``frequency_target`` block with either:
      - ``{"omega": ...}``
      - ``{"St": ..., "omega_scale": ...}``
    """
    if "omega_target" in cfg:
        omega_target = parse_complex(cfg["omega_target"])
        return omega_target, "explicit omega_target"

    freq_cfg = cfg.get("frequency_target")
    if freq_cfg is not None:
        if not isinstance(freq_cfg, dict):
            raise TypeError("frequency_target must be a JSON object when provided")
        if "omega" in freq_cfg:
            omega_target = parse_complex(freq_cfg["omega"])
            return omega_target, "frequency_target.omega"
        if "St" in freq_cfg:
            St = float(freq_cfg["St"])
            omega_scale = parse_complex(freq_cfg.get("omega_scale", 2.0 * np.pi))
            omega_target = omega_scale * St
            return omega_target, f"frequency_target.St with omega_scale={omega_scale}"
        raise ValueError("frequency_target must define either 'omega' or 'St'")

    if "St" in cfg:
        St = float(cfg["St"])
        omega_target = 2.0 * np.pi * St
        return omega_target, "legacy St with omega_scale=2π"

    return 0.0 + 0.0j, "default omega_target=0"


def _resolve_baseflow_input_options(cfg: dict) -> dict[str, bool]:
    """Resolve Song baseflow-loader layout options from JSONC.

    Song's saved `.npy` fields follow the original MATLAB export layout, not
    the repo-wide `(ny, nx)` mesh convention. Expose that explicitly in JSONC
    so the validation case does not silently assemble the mesh with swapped
    physical edges.
    """
    input_cfg = cfg.get("baseflow_input", {})
    if input_cfg is None:
        input_cfg = {}
    if not isinstance(input_cfg, dict):
        raise TypeError("baseflow_input must be a JSON object when provided")
    return {
        "transpose_input": bool(input_cfg.get("transpose_input", False)),
        "zero_spanwise": bool(input_cfg.get("zero_spanwise", False)),
    }


def _resolve_stability_request_count(cfg: dict) -> int:
    """Return how many temporal modes to extract before optional truncation."""
    n_modes = int(cfg["n_modes_stability"])
    extra = int(cfg.get("stability_search_extra_modes", 0))
    if extra < 0:
        raise ValueError("stability_search_extra_modes must be >= 0")
    return n_modes + extra


def _safe_float(value: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _summarize_rank_metrics(
    step_stats_by_rank: list[list[tuple[str, float, float]]],
    elapsed_by_rank: list[float],
    memory_by_rank: list[float],
) -> dict[str, object]:
    """Aggregate per-rank Song metrics into min/mean/max summaries."""
    elapsed = np.asarray([_safe_float(v) for v in elapsed_by_rank], dtype=float)
    memory = np.asarray([_safe_float(v) for v in memory_by_rank], dtype=float)

    step_order: list[str] = []
    for stats in step_stats_by_rank:
        for name, _, _ in stats:
            if name not in step_order:
                step_order.append(name)

    step_summary: dict[str, dict[str, float | int]] = {}
    for name in step_order:
        times = []
        mems = []
        for stats in step_stats_by_rank:
            for step_name, dt, dmem in stats:
                if step_name == name:
                    times.append(_safe_float(dt))
                    mems.append(_safe_float(dmem))
        t = np.asarray(times, dtype=float)
        m = np.asarray(mems, dtype=float)
        step_summary[name] = {
            "count": int(t.size),
            "time_min": float(np.nanmin(t)) if t.size else float("nan"),
            "time_mean": float(np.nanmean(t)) if t.size else float("nan"),
            "time_max": float(np.nanmax(t)) if t.size else float("nan"),
            "memory_min": float(np.nanmin(m)) if m.size else float("nan"),
            "memory_mean": float(np.nanmean(m)) if m.size else float("nan"),
            "memory_max": float(np.nanmax(m)) if m.size else float("nan"),
        }

    return {
        "total_time_min": float(np.nanmin(elapsed)) if elapsed.size else float("nan"),
        "total_time_mean": float(np.nanmean(elapsed)) if elapsed.size else float("nan"),
        "total_time_max": float(np.nanmax(elapsed)) if elapsed.size else float("nan"),
        "peak_memory_rank0": float(memory[0]) if memory.size else float("nan"),
        "peak_memory_max": float(np.nanmax(memory)) if memory.size else float("nan"),
        "peak_memory_sum": float(np.nansum(memory)) if memory.size else float("nan"),
        "step_summary": step_summary,
    }


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
    omega_target: complex = 0.0 + 0.0j,
    n_modes_stability: int = 20,
    n_modes_stability_request: int | None = None,
    n_modes_resolvent: int = 10,
    run_stability: bool = True,
    run_resolvent: bool = False,
    fd_order: int = 2,
    stride_ksi: int = 1,
    stride_eta: int = 1,
    penalty: float = 1e8,
    alpha_filter: float | str = "none",
    assembly_backend: str | None = None,
    output_npz: str = "song_2023.npz",
    bc_recipe: list[dict] | None = None,
    solver_cfg: dict | None = None,
    preprocess_cfg: dict | None = None,
    transpose_input: bool = False,
    zero_spanwise: bool = False,
) -> None:
    """Driver for the Song & Hao (2023) / Robinet (2007) oblique SWBLI validation case."""
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()

    t0 = time.perf_counter()
    mem0 = _get_memory_mb()
    step_stats: List[Tuple[str, float, float]] = []
    matrix_mode = _petsc_matrix_mode(solver_cfg)
    distributed_matrix = _use_distributed_matrix(solver_cfg)
    matrix_type = "scipy"
    ownership_ranges: list[tuple[int, int]] | None = None
    operator_shape: tuple[int, int] | None = None
    operator_nnz: int | None = None

    def _status(message: str) -> None:
        if rank == 0:
            elapsed = time.perf_counter() - t0
            print(f"[song_2023] +{elapsed:8.2f}s {message}", flush=True)

    _status(
        "starting run "
        f"(beta={beta:.3f}, St={St:.3f}, omega_target={omega_target.real:+.6e}{omega_target.imag:+.6e}j, "
        f"fd_order={fd_order}, stride=({stride_ksi},{stride_eta}), "
        f"stability={run_stability}, resolvent={run_resolvent})"
    )
    _status(
        "runtime mode "
        f"(mpi_ranks={size}, petsc_matrix_mode={matrix_mode}, "
        f"distributed_matrix={distributed_matrix})"
    )

    # 1) Load baseflow and sponge from .npy files
    _status("loading baseflow and sponge fields")
    t_start = time.perf_counter()
    mem_start = _get_memory_mb()
    X, Y, baseflow, sponge, preprocess_info = load_my_case(
        case_dir,
        Re=Re,
        Ma=Ma,
        Pr=Pr,
        kappa=kappa,
        T_0=T_0,
        stride_ksi=stride_ksi,
        stride_eta=stride_eta,
        transpose_input=transpose_input,
        zero_spanwise=zero_spanwise,
        preprocess_cfg=preprocess_cfg,
    )
    t_end = time.perf_counter()
    mem_end = _get_memory_mb()
    step_stats.append(("load_baseflow", t_end - t_start, mem_end - mem_start))

    ny, nx = X.shape
    _status(f"loaded baseflow on grid ny={ny}, nx={nx}")
    _status(f"baseflow preprocessing: {preprocess_info.get('summary', 'n/a')}")

    # 2) Build a simple computational mesh and deform to physical coordinates
    _status("building and deforming mesh")
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
    _status(f"mesh built with ngp={mesh.ngp}")

    # 3) Assemble LNSE operator and apply BCs
    _status("assembling LNSE operator and boundary conditions")
    t_start = time.perf_counter()
    mem_start = _get_memory_mb()
    if bc_recipe is None:
        bc_recipe = [
            {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
            {"type": "dirichlet", "borders": "b", "variables": "ruvw"},
            {"type": "neumann", "borders": "b", "variables": "T"},
            {"type": "neumann", "borders": "r", "variables": "ruvwT"},
        ]
    backend = assembly_backend
    L0, idx, bc_indices = build_lnse_system(
        mesh,
        baseflow,
        kx=beta,
        model="2D",
        bc_recipe=bc_recipe,
        bc_penalty=penalty,
        backend=backend,
        solver_cfg=solver_cfg,
    )
    t_end = time.perf_counter()
    mem_end = _get_memory_mb()
    step_stats.append(("assemble_operator_bc", t_end - t_start, mem_end - mem_start))
    nnz = int(L0.getInfo()["nz_used"]) if hasattr(L0, "getInfo") else int(L0.nnz)
    operator_shape = (mesh.ngp * 5, mesh.ngp * 5)
    operator_nnz = nnz
    if hasattr(L0, "getOwnershipRange"):
        matrix_type = str(L0.getType())
        ownership_ranges = list(MPI.COMM_WORLD.allgather(tuple(int(v) for v in L0.getOwnershipRange())))
        if rank == 0:
            _status(
                "PETSc operator "
                f"(type={matrix_type}, ownership_ranges={ownership_ranges})"
            )
    _status(f"assembled operator shape={(mesh.ngp * 5, mesh.ngp * 5)}, nnz={nnz}")

    # 4) Chu energy norm
    _status("building Chu energy norm")
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
    _status(f"Chu norm built with {W_diag.size} weights")

    gains = np.array([], dtype=float)
    responses = np.empty((mesh.ngp * 5, 0), dtype=complex)
    forcings = np.empty((mesh.ngp * 5, 0), dtype=complex)

    if run_resolvent:
        _status(f"starting resolvent solve for {n_modes_resolvent} modes at omega_target={omega_target:.6g}")
        t_start = time.perf_counter()
        mem_start = _get_memory_mb()
        B_diag = np.ones(mesh.ngp * 5, dtype=float)
        B_diag[bc_indices] = 0.0
        B_mat_sp = sp.diags(B_diag, format="csc")
        B_mat = scipy_to_petsc(B_mat_sp, solver_cfg) if hasattr(L0, "getInfo") else B_mat_sp
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
            solver_cfg=solver_cfg,
        )
        t_end = time.perf_counter()
        mem_end = _get_memory_mb()
        step_stats.append(("resolvent", t_end - t_start, mem_end - mem_start))
        _status(f"finished resolvent solve with {gains.size} gains")

    omega = np.array([], dtype=complex)
    direct_modes = np.empty((mesh.ngp * 5, 0), dtype=complex)

    if run_stability:
        _status(f"starting stability eigensolve for {n_modes_stability} modes at omega_target={omega_target:.6g}")
        t_start = time.perf_counter()
        mem_start = _get_memory_mb()
        requested_modes = (
            n_modes_stability
            if n_modes_stability_request is None
            else max(int(n_modes_stability_request), int(n_modes_stability))
        )
        omega, direct_modes = temporal_eigenvalues(
            L0,
            n_modes=requested_modes,
            target=omega_target,
            W_diag=W_diag,
            solver_cfg=solver_cfg,
        )
        if requested_modes > n_modes_stability:
            omega = omega[:n_modes_stability]
            direct_modes = direct_modes[:, :n_modes_stability]
        t_end = time.perf_counter()
        mem_end = _get_memory_mb()
        step_stats.append(("stability", t_end - t_start, mem_end - mem_start))
        if rank == 0 and omega.size:
            lead = omega[np.argmin(np.abs(omega - omega_target))]
            _status(f"finished stability eigensolve; nearest eigenvalue to target is {lead.real:+.6e}{lead.imag:+.6e}j")
        else:
            _status("finished stability eigensolve")

    # 5) Save results on rank 0
    if rank == 0:
        _status(f"saving output to {output_npz}")
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
            preprocess_enabled=bool(preprocess_info.get("enabled", False)),
            preprocess_summary=np.asarray(preprocess_info.get("summary", "")),
            transpose_input=bool(transpose_input),
            zero_spanwise=bool(zero_spanwise),
            n_modes_stability_requested=(
                int(n_modes_stability)
                if n_modes_stability_request is None
                else max(int(n_modes_stability_request), int(n_modes_stability))
            ),
            mpi_ranks=np.int32(size),
            petsc_matrix_mode=np.asarray(matrix_mode),
            distributed_matrix=bool(distributed_matrix),
            matrix_type=np.asarray(matrix_type),
        )
        t_end = time.perf_counter()
        mem_end = _get_memory_mb()
        step_stats.append(("save_output", t_end - t_start, mem_end - mem_start))

    elapsed_local = time.perf_counter() - t0
    mem_total_local = _get_memory_mb()
    gathered_steps = MPI.COMM_WORLD.gather(step_stats, root=0)
    gathered_elapsed = MPI.COMM_WORLD.gather(elapsed_local, root=0)
    gathered_memory = MPI.COMM_WORLD.gather(mem_total_local, root=0)

    if rank == 0:
        summary = _summarize_rank_metrics(gathered_steps, gathered_elapsed, gathered_memory)
        total_time_s = float(summary["total_time_max"])
        peak_memory_mib = float(summary["peak_memory_max"])
        peak_memory_sum_mib = float(summary["peak_memory_sum"])
        peak_memory_rank0_mib = float(summary["peak_memory_rank0"])

        print(f"[song_2023] run finished in {total_time_s:.2f}s, output={out_path}")
        print("[song_2023] step timings (s) and memory deltas (MiB):")
        for name, metrics in summary["step_summary"].items():
            print(
                "  - "
                f"{name:20s}: "
                f"t[min/mean/max]=({metrics['time_min']:.3f}, {metrics['time_mean']:.3f}, {metrics['time_max']:.3f}) s, "
                f"Δmem[min/mean/max]=({metrics['memory_min']:.2f}, {metrics['memory_mean']:.2f}, {metrics['memory_max']:.2f}) MiB"
            )
        if not np.isnan(peak_memory_mib):
            print(
                "[song_2023] peak memory summary: "
                f"rank0={peak_memory_rank0_mib:.2f} MiB, "
                f"max-rank={peak_memory_mib:.2f} MiB, "
                f"sum-ranks={peak_memory_sum_mib:.2f} MiB"
            )
        if not np.isnan(mem0):
            print(
                "[song_2023] overall Δmem since start (rank 0): "
                f"{peak_memory_rank0_mib - mem0:.2f} MiB"
            )

        # Write timing and memory stats to .log for comparison with other codes
        log_file = out_path.with_suffix(".log")
        with open(log_file, "w") as f:
            f.write(f"# linstab2d template run log\n")
            f.write(f"# generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} UTC\n")
            f.write(f"case_dir={case_dir}\n")
            f.write(f"output_npz={out_path}\n")
            f.write(f"Re={Re} Ma={Ma} Pr={Pr} kappa={kappa} T_0={T_0}\n")
            f.write(
                f"beta={beta} St={St} omega_target={omega_target.real:+.16e}{omega_target.imag:+.16e}j "
                f"fd_order={fd_order} stride_ksi={stride_ksi} stride_eta={stride_eta}\n"
            )
            f.write(
                f"baseflow_input.transpose_input={bool(transpose_input)} "
                f"baseflow_input.zero_spanwise={bool(zero_spanwise)}\n"
            )
            f.write(
                f"mpi_ranks={size} petsc_matrix_mode={matrix_mode} "
                f"distributed_matrix={bool(distributed_matrix)} matrix_type={matrix_type}\n"
            )
            if ownership_ranges is not None:
                f.write(f"ownership_ranges={ownership_ranges}\n")
            if operator_shape is not None:
                f.write(f"operator_shape={operator_shape}\n")
            if operator_nnz is not None:
                f.write(f"operator_nnz={operator_nnz}\n")
            f.write(f"baseflow_preprocess={preprocess_info.get('summary', '')}\n")
            f.write(f"nx={nx} ny={ny} ngp={mesh.ngp}\n")
            f.write(
                f"n_modes_stability={n_modes_stability} "
                f"n_modes_stability_requested="
                f"{(int(n_modes_stability) if n_modes_stability_request is None else max(int(n_modes_stability_request), int(n_modes_stability)))} "
                f"n_modes_resolvent={n_modes_resolvent}\n"
            )
            f.write(f"run_stability={run_stability} run_resolvent={run_resolvent}\n")
            f.write(f"total_time_s={total_time_s:.6f}\n")
            f.write(f"total_time_s_min_rank={float(summary['total_time_min']):.6f}\n")
            f.write(f"total_time_s_mean_rank={float(summary['total_time_mean']):.6f}\n")
            f.write(f"peak_memory_mib={peak_memory_mib:.4f}\n")
            f.write(f"peak_memory_mib_rank0={peak_memory_rank0_mib:.4f}\n")
            f.write(f"peak_memory_mib_sum_ranks={peak_memory_sum_mib:.4f}\n")
            if not np.isnan(mem0):
                f.write(f"memory_delta_mib_rank0={peak_memory_rank0_mib - mem0:.4f}\n")
            f.write("# step name | time_min time_mean time_max mem_min mem_mean mem_max count\n")
            for name, metrics in summary["step_summary"].items():
                f.write(
                    "step_summary "
                    f"{name} "
                    f"{metrics['time_min']:.6f} {metrics['time_mean']:.6f} {metrics['time_max']:.6f} "
                    f"{metrics['memory_min']:.4f} {metrics['memory_mean']:.4f} {metrics['memory_max']:.4f} "
                    f"{metrics['count']}\n"
                )
        print(f"[song_2023] stats written to {log_file}")

    MPI.COMM_WORLD.Barrier()


def _load_existing_npz(path: Path) -> dict[str, np.ndarray]:
    """Load an existing Song 2023 NPZ artifact if present and valid."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (FileNotFoundError, EOFError, OSError, ValueError):
        return {}


def _component(matrix: np.ndarray, ngp: int, component: int, mode_index: int, shape: tuple[int, int]) -> np.ndarray:
    """Extract one primitive-variable block from a flattened mode matrix."""
    start = component * ngp
    stop = (component + 1) * ngp
    return np.asarray(matrix[start:stop, mode_index], dtype=complex).reshape(shape, order="F")


def _load_matlab_omega() -> np.ndarray | None:
    """Load MATLAB direct-spectrum eigenvalues if available."""
    spectrum_path = MATLAB_EIG_DIR / "spectrum.mat"
    if not spectrum_path.exists():
        return None
    from scipy.io import loadmat

    omega = np.asarray(loadmat(spectrum_path)["omega"], dtype=complex)
    if omega.ndim == 2 and omega.shape[0] == omega.shape[1]:
        return np.diag(omega)
    return omega.reshape(-1)


def _load_song_original_omega() -> np.ndarray | None:
    """Load the original Song spectrum from the text export."""
    eig_path = SONG_REF_DIR / "eigenvalues_Ziming.dat"
    if not eig_path.exists():
        return None
    values = np.loadtxt(eig_path)
    if values.ndim != 2 or values.shape[1] != 2:
        return None
    length_scale = 80.0
    return length_scale * (values[:, 0] + 1j * values[:, 1])


def _load_spectrum_overlays(overlay_cfg: dict | None) -> list[dict[str, object]]:
    """Load optional extra Python spectra for the Song summary spectrum panel."""
    if not overlay_cfg or not bool(overlay_cfg.get("enabled", False)):
        return []

    variants = overlay_cfg.get("variants", [])
    if not isinstance(variants, list):
        raise TypeError("spectrum_overlay.variants must be a list when provided")

    overlays: list[dict[str, object]] = []
    for item in variants:
        if not isinstance(item, dict):
            raise TypeError("Each spectrum_overlay variant must be a JSON object")
        raw_path = item.get("npz")
        if raw_path is None:
            raise ValueError("Each spectrum_overlay variant must define an 'npz' path")
        npz_path = Path(str(raw_path))
        if not npz_path.is_absolute():
            npz_path = (SCRIPT.parent / npz_path).resolve()
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            if "omega" not in data:
                continue
            overlays.append(
                {
                    "label": str(item.get("label", npz_path.parent.name)),
                    "marker": str(item.get("marker", "^")),
                    "color": str(item.get("color", "#c26d00")),
                    "omega": np.asarray(data["omega"], dtype=complex).reshape(-1),
                }
            )
    return overlays


def _load_song_reference_baseflow() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load Song original baseflow arrays if available."""
    paths = [SONG_REF_DIR / f"{name}.npy" for name in ("x", "y", "u")]
    if not all(path.exists() for path in paths):
        return None
    X_song = np.asarray(np.load(paths[0]), dtype=float)
    Y_song = np.asarray(np.load(paths[1]), dtype=float)
    U_song = np.asarray(np.load(paths[2]), dtype=float)
    return X_song, Y_song, U_song


def _load_song_mode1_u() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load the original Song leading-mode streamwise velocity from the Tecplot DAT export."""
    dat_path = SONG_REF_DIR / "eigenmode1.dat"
    if not dat_path.exists():
        return None

    total_points = 601 * 301
    values = np.empty((total_points, 8), dtype=float)
    with dat_path.open("r", encoding="utf-8") as handle:
        handle.readline()
        handle.readline()
        for i in range(total_points):
            line1 = handle.readline().split()
            line2 = handle.readline().split()
            line3 = handle.readline().split()
            block = line1 + line2 + line3
            if len(block) != 8:
                raise ValueError(f"Unexpected Song DAT block length at point {i}: {len(block)}")
            values[i, :] = np.asarray(block, dtype=float)

    X_song = values[:, 0].reshape(301, 601) / 80.0
    Y_song = values[:, 1].reshape(301, 601) / 80.0
    U_song = values[:, 3].reshape(301, 601)
    return X_song, Y_song, U_song


def _load_matlab_u_mode(index_1based: int, shape: tuple[int, int]) -> np.ndarray | None:
    """Load one MATLAB direct-mode streamwise-velocity field."""
    eig_path = MATLAB_EIG_DIR / f"eigenvector_{index_1based}.mat"
    if eig_path.exists():
        from scipy.io import loadmat

        data = loadmat(eig_path)
        if "u" in data:
            arr = np.asarray(data["u"], dtype=complex).reshape(-1)
            if arr.size == int(np.prod(shape)):
                return arr.reshape(shape, order="F")
            return None

    split_path = MATLAB_SPLIT_DIR / f"direct_mode_{index_1based - 1:03d}.npy"
    if split_path.exists():
        ngp = int(np.prod(shape))
        direct_mode = np.asarray(np.load(split_path), dtype=complex).reshape(-1)
        if direct_mode.size == 5 * ngp:
            return direct_mode[ngp:2 * ngp].reshape(shape, order="F")
    return None


def _select_reference_mode_indices(omega_ref: np.ndarray, target: complex = 0.0 + 0.0j) -> list[int]:
    """Pick the steady and oscillatory reference modes for summary plots.

    Mode 1:
        the reference eigenvalue closest to the target (for Song, the
        nearly-steady branch near positive growth).
    Mode 2:
        the closest reference eigenvalue with non-zero real part, preferring
        the positive-real member when the oscillatory pair is symmetric.
    """
    omega_ref = np.asarray(omega_ref, dtype=complex).reshape(-1)
    if omega_ref.size == 0:
        return []

    distances = np.abs(omega_ref - target)
    steady_idx = int(np.argmin(distances))
    selected = [steady_idx]

    nonzero_real = np.flatnonzero(np.abs(np.real(omega_ref)) > 1e-8)
    if nonzero_real.size:
        osc_dist = distances[nonzero_real]
        best = float(np.min(osc_dist))
        near = nonzero_real[np.abs(osc_dist - best) <= 1e-10]
        positive = near[np.real(omega_ref[near]) > 0.0]
        if positive.size:
            oscillatory_idx = int(positive[np.argmin(np.abs(omega_ref[positive] - target))])
        else:
            oscillatory_idx = int(near[np.argmin(np.abs(omega_ref[near] - target))])
        if oscillatory_idx not in selected:
            selected.append(oscillatory_idx)

    if len(selected) < 2 and omega_ref.size > 1:
        for idx in np.argsort(distances):
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
                break

    return selected[:2]


def _match_selected_modes(
    omega_ref: np.ndarray,
    omega_py: np.ndarray,
    *,
    target: complex = 0.0 + 0.0j,
) -> list[tuple[int, int]]:
    """Match selected MATLAB reference modes to unique Python modes."""
    omega_ref = np.asarray(omega_ref, dtype=complex).reshape(-1)
    omega_py = np.asarray(omega_py, dtype=complex).reshape(-1)

    pairs: list[tuple[int, int]] = []
    used_py: set[int] = set()
    for ref_i in _select_reference_mode_indices(omega_ref, target=target):
        distances = np.abs(omega_py - omega_ref[ref_i])
        for py_i in np.argsort(distances):
            py_i = int(py_i)
            if py_i not in used_py:
                pairs.append((int(ref_i), py_i))
                used_py.add(py_i)
                break
    return pairs


def _align_mode_to_reference(candidate: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    """Return ``candidate`` aligned to ``reference`` by complex least-squares phase."""
    candidate = np.asarray(candidate, dtype=complex)
    if reference is None:
        return candidate
    reference = np.asarray(reference, dtype=complex)
    denom = np.vdot(candidate.reshape(-1), candidate.reshape(-1))
    if abs(denom) == 0.0:
        return candidate
    phase = np.vdot(candidate.reshape(-1), reference.reshape(-1)) / denom
    return candidate * phase


def _plot(data: dict[str, np.ndarray], *, overlay_cfg: dict | None = None) -> None:
    """Generate baseflow, stability, and optional resolvent figures for Song 2023."""
    import matplotlib.pyplot as plt

    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    U = np.asarray(data["U"], dtype=float)
    sponge = np.asarray(data["sponge"], dtype=float)
    shape = X.shape
    ngp = int(np.prod(shape))
    xlim = (float(np.min(X)), float(np.max(X)))
    ylim = (float(np.min(Y)), float(np.max(Y)))
    song_baseflow = _load_song_reference_baseflow()

    fig_base, ax_base = plt.subplots(1, 1, figsize=(11.8, 4.2))
    plot_scalar_panel(
        ax_base,
        X,
        Y,
        U,
        title="Song 2023 / Robinet 2007 baseflow",
        cmap="turbo",
        xlim=xlim,
        ylim=ylim,
        aspect="auto",
        levels=101,
    )
    if float(np.nanmax(sponge)) > 0.0:
        ax_base.contour(X, Y, sponge, levels=[1e-8], colors="black", linewidths=1.0, alpha=0.8)
    ax_base.contour(X, Y, U, levels=[0.0], colors=["white"], linewidths=[1.1], alpha=0.9)
    if song_baseflow is not None:
        X_song, Y_song, U_song = song_baseflow
        ax_base.contour(X_song, Y_song, U_song, levels=[0.0], colors=["black"], linewidths=[0.9], linestyles=["--"], alpha=0.85)
    ax_base.set_xlabel("x")
    ax_base.set_ylabel("y")
    save_figure(fig_base, OUTPUT_BASEFLOW)

    omega = np.asarray(data["omega"], dtype=complex).reshape(-1)
    direct_modes = np.asarray(data["direct_modes"], dtype=complex)
    matlab_omega = _load_matlab_omega()
    song_omega = _load_song_original_omega()
    overlay_spectra = _load_spectrum_overlays(overlay_cfg)
    if omega.size:
        target = complex(np.asarray(data.get("omega_target", 0.0 + 0.0j)).reshape(-1)[0])
        if matlab_omega is not None and matlab_omega.size:
            matched = _match_selected_modes(
                np.asarray(matlab_omega, dtype=complex).reshape(-1),
                omega,
                target=target,
            )
        else:
            order = np.argsort(np.abs(omega - target))[:2]
            matched = [(i, int(py_i)) for i, py_i in enumerate(order)]
        n_show = len(matched)

        fig = plt.figure(figsize=(15.4, 7.8))
        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=(1.0, 1.15, 1.15),
            hspace=0.24,
            wspace=0.18,
        )
        ax_spec = fig.add_subplot(grid[:, 0])
        if song_omega is not None and song_omega.size:
            ax_spec.plot(
                np.real(song_omega),
                np.imag(song_omega),
                "s",
                mfc="white",
                mec="#6a3d9a",
                mew=1.0,
                ms=6.0,
                label="Song",
                zorder=1,
            )
        if matlab_omega is not None and matlab_omega.size:
            ax_spec.plot(
                np.real(matlab_omega),
                np.imag(matlab_omega),
                "o",
                mfc="white",
                mec="black",
                mew=1.0,
                ms=6.5,
                label="MATLAB",
                zorder=2,
            )
        ax_spec.plot(
            np.real(omega),
            np.imag(omega),
            "o",
            color="#2a5caa",
            ms=5,
            label="Python 2nd",
            zorder=3,
        )
        for overlay in overlay_spectra:
            omega_extra = np.asarray(overlay["omega"], dtype=complex).reshape(-1)
            if omega_extra.size == 0:
                continue
            ax_spec.plot(
                np.real(omega_extra),
                np.imag(omega_extra),
                linestyle="none",
                marker=str(overlay["marker"]),
                color=str(overlay["color"]),
                ms=5,
                label=str(overlay["label"]),
                zorder=2.5,
            )
        for row, (mat_i, py_i) in enumerate(matched, start=1):
            ax_spec.plot(
                np.real(omega[py_i]),
                np.imag(omega[py_i]),
                "D",
                color="#d99100",
                mec="black",
                mew=0.5,
                ms=8,
            )
            if matlab_omega is not None and matlab_omega.size and mat_i < matlab_omega.size:
                ax_spec.text(
                    np.real(omega[py_i]) + 0.01,
                    np.imag(omega[py_i]) + 0.01,
                    f"M{row}",
                    fontsize=9,
                )
        ax_spec.set_xlabel(r"$\Re(\omega)$")
        ax_spec.set_ylabel(r"$\Im(\omega)$")
        ax_spec.set_title(
            rf"(a) Temporal spectrum at $\beta={float(np.asarray(data['beta']).reshape(-1)[0]):.2f}$",
            loc="left",
        )
        ax_spec.grid(True, alpha=0.3)
        if (
            (matlab_omega is not None and matlab_omega.size)
            or (song_omega is not None and song_omega.size)
        ):
            ax_spec.legend(loc="lower right", frameon=False, fontsize=9)

        mode_labels = ["Mode 1 (steady)", "Mode 2 (oscillatory)"]
        subplot_labels = [("b", "d"), ("c", "e")]
        for col, (mat_i, py_i) in enumerate(matched):
            ax_mat = fig.add_subplot(grid[0, col + 1])
            ax_py = fig.add_subplot(grid[1, col + 1])
            py_field = _component(direct_modes, ngp, 1, py_i, shape)
            mat_field = _load_matlab_u_mode(mat_i + 1, shape)
            py_aligned = _align_mode_to_reference(py_field, mat_field)

            ax_mat.text(
                0.0,
                1.08,
                mode_labels[col] if col < len(mode_labels) else f"Mode {col + 1}",
                transform=ax_mat.transAxes,
                ha="left",
                va="bottom",
                fontsize=12,
            )
            if mat_field is not None:
                plot_signed_panel(
                    ax_mat,
                    X,
                    Y,
                    np.real(mat_field),
                    title="",
                    xlim=xlim,
                    ylim=ylim,
                    aspect="auto",
                    colorbar=False,
                    robust_percentile=99.0,
                )
                if float(np.nanmax(sponge)) > 0.0:
                    ax_mat.contour(X, Y, sponge, levels=[1e-8], colors="black", linewidths=0.8, alpha=0.7)
                ax_mat.set_title(rf"({subplot_labels[col][0]}) MATLAB $u$", loc="left")
            else:
                ax_mat.set_title(rf"({subplot_labels[col][0]}) MATLAB mode unavailable", loc="left")
                ax_mat.text(
                    0.5,
                    0.5,
                    "Reference field unavailable\nfor this grid shape or mode",
                    ha="center",
                    va="center",
                    transform=ax_mat.transAxes,
                    fontsize=10,
                )
            if matlab_omega is not None and matlab_omega.size and mat_i < matlab_omega.size:
                ax_mat.text(
                    0.02,
                    0.92,
                    rf"$\omega={format_complex(matlab_omega[mat_i], real_precision=4)}$",
                    transform=ax_mat.transAxes,
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )

            plot_signed_panel(
                ax_py,
                X,
                Y,
                np.real(py_aligned),
                title="",
                xlim=xlim,
                ylim=ylim,
                aspect="auto",
                colorbar=False,
                robust_percentile=99.0,
            )
            if float(np.nanmax(sponge)) > 0.0:
                ax_py.contour(X, Y, sponge, levels=[1e-8], colors="black", linewidths=0.8, alpha=0.7)
            ax_py.set_title(rf"({subplot_labels[col][1]}) Python $u$", loc="left")
            ax_py.text(
                0.02,
                0.92,
                rf"$\omega={format_complex(omega[py_i], real_precision=4)}$",
                transform=ax_py.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )

            ax_py.set_xlabel("x")
            ax_py.set_ylabel("y")
            ax_mat.set_xlabel("")
            ax_mat.set_ylabel("y")

        fig.tight_layout()
        save_figure(fig, OUTPUT_PNG)

        song_mode1 = _load_song_mode1_u()
        fig_support, axes = plt.subplots(n_show, 4, figsize=(18.0, 3.8 * n_show))
        if n_show == 1:
            axes = np.asarray([axes])
        for row, (mat_i, py_i) in enumerate(matched):
            py_field = _component(direct_modes, ngp, 1, py_i, shape)
            mat_field = _load_matlab_u_mode(mat_i + 1, shape)
            py_aligned = _align_mode_to_reference(py_field, mat_field)
            diff = np.real(py_aligned - mat_field) if mat_field is not None else None

            panels = [
                (np.real(py_aligned), rf"Python $u$, mode {py_i + 1}"),
                (np.real(mat_field) if mat_field is not None else None, rf"MATLAB $u$, mode {mat_i + 1}"),
                (
                    np.asarray(song_mode1[2], dtype=float) if (song_mode1 is not None and row == 0) else None,
                    "Song original $u$, mode 1" if row == 0 else "Song original",
                ),
                (diff, r"$u_{\mathrm{py}}-u_{\mathrm{mat}}$"),
            ]
            for col, (field, title) in enumerate(panels):
                if field is None:
                    axes[row, col].set_title(title)
                    axes[row, col].text(
                        0.5,
                        0.5,
                        "Reference field unavailable\nfor this grid shape or mode",
                        ha="center",
                        va="center",
                        transform=axes[row, col].transAxes,
                        fontsize=10,
                    )
                else:
                    X_plot = X if not (song_mode1 is not None and row == 0 and col == 2) else np.asarray(song_mode1[0], dtype=float)
                    Y_plot = Y if not (song_mode1 is not None and row == 0 and col == 2) else np.asarray(song_mode1[1], dtype=float)
                    plot_signed_panel(
                        axes[row, col],
                        X_plot,
                        Y_plot,
                        field,
                        title=title,
                        xlim=xlim,
                        ylim=ylim,
                        aspect="auto",
                        colorbar=False,
                        robust_percentile=99.0 if col < 2 else 99.5,
                    )
                axes[row, col].set_xlabel("x")
                axes[row, col].set_ylabel("y")
        fig_support.suptitle("Song 2023 direct-mode comparison: Python vs MATLAB vs Song", fontsize=14)
        fig_support.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        save_figure(fig_support, OUTPUT_SUPPORT)

    gains = np.asarray(data["gains"], dtype=float).reshape(-1)
    responses = np.asarray(data["responses"], dtype=complex)
    forcings = np.asarray(data["forcings"], dtype=complex)
    if gains.size:
        n_plot = min(3, gains.size)
        fig_res = plt.figure(figsize=(12.8, 7.8))
        grid = fig_res.add_gridspec(n_plot, 3, width_ratios=(1.0, 1.4, 1.4), hspace=0.22, wspace=0.18)
        ax_gain = fig_res.add_subplot(grid[:, 0])
        plot_gain_bars(ax_gain, gains[:n_plot])

        for i in range(n_plot):
            ax_f = fig_res.add_subplot(grid[i, 1])
            ax_r = fig_res.add_subplot(grid[i, 2])
            f_u = _component(forcings, ngp, 1, i, shape)
            r_u = _component(responses, ngp, 1, i, shape)
            plot_signed_panel(
                ax_f,
                X,
                Y,
                np.real(f_u),
                title=rf"$f_u^{{({i + 1})}}$",
                xlim=xlim,
                ylim=ylim,
                aspect="auto",
                colorbar=False,
                robust_percentile=99.0,
            )
            plot_signed_panel(
                ax_r,
                X,
                Y,
                np.real(r_u),
                title=rf"$u^{{({i + 1})}}$",
                xlim=xlim,
                ylim=ylim,
                aspect="auto",
                colorbar=False,
                robust_percentile=99.0,
            )
            ax_f.set_xlabel("x")
            ax_r.set_xlabel("x")
            ax_f.set_ylabel("y")
        fig_res.suptitle("Song 2023 resolvent summary", fontsize=14)
        fig_res.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        save_figure(fig_res, OUTPUT_RESOLVENT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Song2023 / Robinet validation case.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSONC config file. Defaults to song_2023.jsonc next to this module.",
    )
    return parser.parse_args()


def _resolve_config_path(cli_config_path: Path | None = None) -> Path:
    """Return the Song configuration path.

    Preferred selection is the explicit ``--config`` CLI argument so all run
    settings live in JSONC files rather than environment variables.
    """
    if cli_config_path is not None:
        return cli_config_path.expanduser().resolve()
    raw = os.getenv("LINSTAB2D_SONG2023_CONFIG")
    if raw is None or raw.strip() == "":
        return DEFAULT_CONFIG_PATH
    return Path(raw).expanduser().resolve()


def main() -> None:
    """Load parameters from song_2023.jsonc, run the case if needed, then plot."""
    case_dir = Path(__file__).resolve().parent
    args = _parse_args()
    config_path = _resolve_config_path(args.config)
    cfg = load_jsonc(config_path)
    baseflow_input = _resolve_baseflow_input_options(cfg)
    n_modes_stability_request = _resolve_stability_request_count(cfg)
    omega_target, frequency_source = _resolve_frequency_target(cfg)
    if "St" in cfg:
        St = float(cfg["St"])
    elif isinstance(cfg.get("frequency_target"), dict) and "St" in cfg["frequency_target"]:
        St = float(cfg["frequency_target"]["St"])
    else:
        St = 0.0
    data = _load_existing_npz(OUTPUT_NPZ)
    required = {"X", "Y", "U", "omega", "direct_modes", "gains", "responses", "forcings", "beta"}
    if not required.issubset(data):
        bc_recipe = cfg.get("bc", {}).get("recipe")
        print(
            "[song_2023] configuration frequency target: "
            f"source={frequency_source}, St={St:.6g}, "
            f"omega_target={omega_target.real:+.6e}{omega_target.imag:+.6e}j",
            flush=True,
        )
        run_case(
            case_dir=str(case_dir),
            Re=cfg["Re"],
            Ma=cfg["Ma"],
            Pr=cfg["Pr"],
            kappa=cfg["kappa"],
            T_0=cfg["T_0"],
            beta=cfg["beta"],
            St=St,
            omega_target=omega_target,
            n_modes_stability=cfg["n_modes_stability"],
            n_modes_stability_request=n_modes_stability_request,
            n_modes_resolvent=cfg["n_modes_resolvent"],
            run_stability=cfg["run_stability"],
            run_resolvent=cfg["run_resolvent"],
            fd_order=cfg["fd_order"],
            stride_ksi=cfg.get("stride_ksi", 1),
            stride_eta=cfg.get("stride_eta", 1),
            penalty=cfg["penalty"],
            alpha_filter=cfg.get("alpha_filter", "none"),
            assembly_backend=cfg.get("assembly_backend"),
            output_npz=str(OUTPUT_NPZ),
            bc_recipe=bc_recipe,
            solver_cfg=cfg.get("solver"),
            preprocess_cfg=cfg.get("baseflow_preprocess"),
            transpose_input=baseflow_input["transpose_input"],
            zero_spanwise=baseflow_input["zero_spanwise"],
        )
        data = _load_existing_npz(OUTPUT_NPZ)
    if not required.issubset(data):
        raise RuntimeError(f"Failed to generate {OUTPUT_NPZ}")
    _plot(data, overlay_cfg=cfg.get("spectrum_overlay"))


if __name__ == "__main__":
    main()
