"""Song & Hao (2023) / Robinet (2007) oblique SWBLI validation case.

Usage (from repo root):
    .venv/bin/python -m validation.song_2023.song_2023
"""
from __future__ import annotations

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
) -> None:
    """Driver for the Song & Hao (2023) / Robinet (2007) oblique SWBLI validation case."""
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    t0 = time.perf_counter()
    mem0 = _get_memory_mb()
    step_stats: List[Tuple[str, float, float]] = []

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
        omega, direct_modes = temporal_eigenvalues(
            L0,
            n_modes=n_modes_stability,
            target=omega_target,
            W_diag=W_diag,
            solver_cfg=solver_cfg,
        )
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
            preprocess_info=np.asarray([preprocess_info], dtype=object),
        )
        t_end = time.perf_counter()
        mem_end = _get_memory_mb()
        step_stats.append(("save_output", t_end - t_start, mem_end - mem_start))

        elapsed = time.perf_counter() - t0
        mem_total = _get_memory_mb()

        print(f"[song_2023] run finished in {elapsed:.2f}s, output={out_path}")
        print("[song_2023] step timings (s) and memory deltas (MiB):")
        for name, dt, dmem in step_stats:
            print(f"  - {name:20s}: {dt:8.3f} s, Δmem={dmem:8.2f} MiB")
        if not np.isnan(mem_total):
            print(f"[song_2023] approximate current/peak memory: {mem_total:.2f} MiB")
        if not np.isnan(mem0):
            print(f"[song_2023] overall Δmem since start: {mem_total - mem0:.2f} MiB")

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
            f.write(f"baseflow_preprocess={preprocess_info.get('summary', '')}\n")
            f.write(f"nx={nx} ny={ny} ngp={mesh.ngp}\n")
            f.write(f"n_modes_stability={n_modes_stability} n_modes_resolvent={n_modes_resolvent}\n")
            f.write(f"run_stability={run_stability} run_resolvent={run_resolvent}\n")
            f.write(f"total_time_s={elapsed:.6f}\n")
            f.write(f"peak_memory_mib={mem_total:.4f}\n")
            if not np.isnan(mem0):
                f.write(f"memory_delta_mib={mem_total - mem0:.4f}\n")
            f.write("# step name | time_s | memory_delta_mib\n")
            for name, dt, dmem in step_stats:
                f.write(f"step {name} {dt:.6f} {dmem:.4f}\n")
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


def _best_mode_pairs(omega_mat: np.ndarray, omega_py: np.ndarray, n_pairs: int) -> list[tuple[int, int]]:
    """Match MATLAB and Python eigenvalues by nearest distance in the complex plane."""
    pairs: list[tuple[int, int]] = []
    used_py: set[int] = set()
    for mat_i in np.argsort(np.imag(omega_mat)):
        distances = np.abs(omega_py - omega_mat[mat_i])
        py_order = np.argsort(distances)
        for py_i in py_order:
            if int(py_i) not in used_py:
                pairs.append((int(mat_i), int(py_i)))
                used_py.add(int(py_i))
                break
        if len(pairs) >= n_pairs:
            break
    pairs.sort(key=lambda pair: np.imag(omega_py[pair[1]]))
    return pairs


def _plot(data: dict[str, np.ndarray]) -> None:
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
    if omega.size:
        n_show = min(2, omega.size)
        if matlab_omega is not None and matlab_omega.size:
            matched = _best_mode_pairs(np.asarray(matlab_omega, dtype=complex).reshape(-1), omega, n_show)
        else:
            order = np.argsort(np.imag(omega))[:n_show]
            matched = [(i, int(py_i)) for i, py_i in enumerate(order)]

        fig = plt.figure(figsize=(12.8, 7.4))
        grid = fig.add_gridspec(n_show, 2, width_ratios=(1.0, 1.5), hspace=0.24, wspace=0.18)
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
            label="Python",
            zorder=3,
        )
        for _, py_i in matched:
            ax_spec.plot(np.real(omega[py_i]), np.imag(omega[py_i]), "D", color="#d99100", mec="black", mew=0.5, ms=8)
        ax_spec.set_xlabel(r"$\Re(\omega)$")
        ax_spec.set_ylabel(r"$\Im(\omega)$")
        ax_spec.set_title(rf"Temporal spectrum at $\beta={float(np.asarray(data['beta']).reshape(-1)[0]):.2f}$")
        ax_spec.grid(True, alpha=0.3)
        if (
            (matlab_omega is not None and matlab_omega.size)
            or (song_omega is not None and song_omega.size)
        ):
            ax_spec.legend(loc="lower right", frameon=False, fontsize=9)

        for row, (mat_i, py_i) in enumerate(matched):
            ax_mode = fig.add_subplot(grid[row, 1])
            field = _component(direct_modes, ngp, 1, py_i, shape)
            plot_signed_panel(
                ax_mode,
                X,
                Y,
                np.real(field),
                title=rf"Direct $u$, $\omega={format_complex(omega[py_i], real_precision=4)}$",
                xlim=xlim,
                ylim=ylim,
                aspect="auto",
                colorbar=False,
                robust_percentile=99.0,
            )
            if float(np.nanmax(sponge)) > 0.0:
                ax_mode.contour(X, Y, sponge, levels=[1e-8], colors="black", linewidths=0.8, alpha=0.7)
            ax_mode.set_xlabel("x")
            ax_mode.set_ylabel("y")
            if matlab_omega is not None and matlab_omega.size and mat_i < matlab_omega.size:
                ax_mode.text(
                    0.02,
                    0.90,
                    rf"MATLAB #{mat_i + 1}: $\omega={format_complex(matlab_omega[mat_i], real_precision=4)}$",
                    transform=ax_mode.transAxes,
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )

        fig.suptitle("Song 2023 stability summary", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        save_figure(fig, OUTPUT_PNG)

        song_mode1 = _load_song_mode1_u()
        fig_support, axes = plt.subplots(n_show, 4, figsize=(18.0, 3.8 * n_show))
        if n_show == 1:
            axes = np.asarray([axes])
        for row, (mat_i, py_i) in enumerate(matched):
            py_field = _component(direct_modes, ngp, 1, py_i, shape)
            mat_field = _load_matlab_u_mode(mat_i + 1, shape)
            phase = 1.0 + 0.0j
            denom = np.vdot(py_field.reshape(-1), py_field.reshape(-1))
            if mat_field is not None and abs(denom) > 0:
                phase = np.vdot(py_field.reshape(-1), mat_field.reshape(-1)) / denom
            py_aligned = py_field * phase
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


def _resolve_config_path() -> Path:
    """Return the Song configuration path, optionally overridden by env var."""
    raw = os.getenv("LINSTAB2D_SONG2023_CONFIG")
    if raw is None or raw.strip() == "":
        return DEFAULT_CONFIG_PATH
    return Path(raw).expanduser().resolve()


def main() -> None:
    """Load parameters from song_2023.jsonc, run the case if needed, then plot."""
    case_dir = Path(__file__).resolve().parent
    config_path = _resolve_config_path()
    cfg = load_jsonc(config_path)
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
        )
        data = _load_existing_npz(OUTPUT_NPZ)
    if not required.issubset(data):
        raise RuntimeError(f"Failed to generate {OUTPUT_NPZ}")
    _plot(data)


if __name__ == "__main__":
    main()
