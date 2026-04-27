#!/usr/bin/env python3
"""
Example 3: Streaky boundary-layer temporal and spatial analysis.

Translates example_3_Streaky_BL.m with:
- periodic x-domain and Hanifi-like y stretch
- temporal eigenvalues
- spatial eigenvalues in generalized (alpha) form

OUTPUTS:
  - example_3_streaky_bl.npz
  - example_3_streaky_bl_baseflow.png
  - example_3_streaky_bl.png
  - example_3_streaky_bl_spatial.png
  - example_3_streaky_bl_spectra.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from linstab2d.utils.jsonc import load_jsonc, parse_complex_list, parse_int_list

from ._helpers import (
    apply_figure_layout,
    format_complex,
    format_cache_recompute_message,
    format_scientific_ticks,
    label_axes,
    leading_mode_real_frequency,
    load_parity_npz,
    parity_npz_get,
    plot_scalar_panel,
    plot_signed_panel,
    save_figure,
    smallest_abs_indices,
    set_suptitle,
    top_imag_indices,
)

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")
OUTPUT_BASEFLOW = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_baseflow.png")
OUTPUT_SPATIAL = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_spatial.png")
OUTPUT_SPECTRA = OUTPUT_PNG.with_name(f"{OUTPUT_PNG.stem}_spectra.png")
MATLAB_TEMPORAL_STEM = "example_3_Streaky_BL_results_temporal"
MATLAB_SPATIAL_STEM = "example_3_Streaky_BL_results_spatial"


def _load_matlab_temporal_omega() -> np.ndarray | None:
    """Load the extracted MATLAB temporal spectrum when available."""
    try:
        bundle = load_parity_npz(MATLAB_TEMPORAL_STEM)
    except FileNotFoundError:
        return None
    return np.asarray(parity_npz_get(bundle, "results_StreakyBL_temporal/omega"), dtype=complex).reshape(-1)


def _load_matlab_spatial_alpha() -> np.ndarray | None:
    """Load the extracted MATLAB spatial spectrum when available."""
    try:
        bundle = load_parity_npz(MATLAB_SPATIAL_STEM)
    except FileNotFoundError:
        return None
    return np.asarray(parity_npz_get(bundle, "results_StreakyBL_spatial/alphas"), dtype=complex).reshape(-1)


def _load_existing_npz() -> dict[str, np.ndarray]:
    """Load an existing NPZ artifact if present and valid."""
    try:
        with np.load(OUTPUT_NPZ, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (FileNotFoundError, EOFError, OSError, ValueError):
        return {}


def _save_npz(data: dict[str, np.ndarray | float | complex | int]) -> None:
    """Save a mixed metadata/results dict to the example NPZ path."""
    payload: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            payload[key] = value
        else:
            payload[key] = np.asarray(value)
    np.savez_compressed(OUTPUT_NPZ, **payload)
    print(f"Results saved to {OUTPUT_NPZ}")


def _merge_temporal_target_batches(
    batches: list[tuple[complex, np.ndarray, np.ndarray]],
    quotas: list[int],
    *,
    dedupe_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge target-window solves with fixed quotas per target."""
    if len(batches) != len(quotas):
        raise ValueError("batches and quotas must have the same length")

    picked_omega: list[complex] = []
    picked_modes: list[np.ndarray] = []
    for (target, omegas, modes), quota in zip(batches, quotas):
        order = np.argsort(np.abs(np.asarray(omegas, dtype=complex) - target))
        added = 0
        for idx in order:
            omega = complex(omegas[idx])
            if any(abs(omega - existing) <= dedupe_tol for existing in picked_omega):
                continue
            picked_omega.append(omega)
            picked_modes.append(np.asarray(modes[:, idx], dtype=complex))
            added += 1
            if added >= quota:
                break
        if added < quota:
            raise RuntimeError(
                f"Target {target} yielded only {added} unique modes, expected {quota}."
            )

    return np.asarray(picked_omega, dtype=complex), np.column_stack(picked_modes)


def _build_streaky_mesh(cfg: dict) -> tuple[Mesh, float, float]:
    """Build the Example 3 mesh and mapped physical coordinates."""
    from linstab2d.mesh import Mesh

    nx = cfg["nx"]
    ny = cfg["ny"]
    Lx = cfg["Lx"]
    Ly = cfg["Ly"]
    cmesh = Mesh.create(
        (0.0, Lx),
        (-1.0, 1.0),
        nx,
        ny,
        cfg["fd_order"],
        y_symmetry=False,
        x_periodicity=cfg["x_periodicity"],
        alpha_filter=cfg["alpha_filter"],
    )

    yi = cfg["y_stretch_yi"]
    a = yi * Ly / (Ly - 2 * yi)
    b = 1.0 + 2.0 * a / Ly
    Y = a * (1.0 + cmesh.Y) / (b - cmesh.Y)
    mesh = cmesh.deform(Y_new=Y)
    return mesh, Lx, Ly


def _apply_streaky_sponge(mesh: Mesh, cfg: dict) -> None:
    """Apply the MATLAB-like top sponge used in Example 3."""
    sponge = cfg["sponge"]
    Ly = cfg["Ly"]
    ys = Ly - sponge["ys_offset"]
    ys_trans = sponge["ys_trans"]
    sponge_amp = sponge["amplitude"]
    mesh.sponge = sponge_amp / 2 * (np.tanh((mesh.Y - ys) / ys_trans) + 1.0)


def _compute_spatial_residuals(
    Lc,
    Lw,
    R,
    omega_spatial: complex,
    spatial_alpha: np.ndarray,
    spatial_modes: np.ndarray,
) -> np.ndarray:
    """Compute residuals for the spatial generalized eigenproblem."""
    A = Lc + omega_spatial * Lw
    residuals = []
    for alpha, q in zip(
        np.asarray(spatial_alpha, dtype=complex).reshape(-1),
        np.asarray(spatial_modes, dtype=complex).T,
    ):
        numer = np.linalg.norm(A @ q - alpha * (R @ q))
        denom = max(abs(alpha) * float(np.linalg.norm(q)), 1e-300)
        residuals.append(float(numer / denom))
    return np.asarray(residuals, dtype=float)


def _augment_cached_results(results: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Backfill fields needed by the revised plotting/parity flow."""
    if not results:
        return results

    changed = False
    mesh = None
    baseflow = None

    # Minimal cfg for mesh/sponge reconstruction from cached data shape
    mini_cfg: dict = {
        "nx": 0, "ny": 0, "Lx": np.pi, "Ly": 5.0,
        "fd_order": 4, "alpha_filter": 0.0,
        "x_periodicity": True, "y_stretch_yi": 2.0,
        "sponge": {"amplitude": 0.1, "ys_offset": 1.0, "ys_trans": 0.5},
    }

    def ensure_problem() -> tuple[Mesh, object]:
        nonlocal mesh, baseflow
        if mesh is None or baseflow is None:
            from linstab2d.baseflow import load_streaky_bl

            ny, nx = np.asarray(results["X"]).shape
            mini_cfg["nx"] = nx
            mini_cfg["ny"] = ny
            mesh, _, _ = _build_streaky_mesh(mini_cfg)
            baseflow = load_streaky_bl(mesh)
        return mesh, baseflow

    if "baseflow_u" not in results:
        _, baseflow = ensure_problem()
        results["baseflow_u"] = np.asarray(baseflow.U, dtype=float)
        changed = True
    if "baseflow_w" not in results:
        _, baseflow = ensure_problem()
        results["baseflow_w"] = np.asarray(baseflow.W, dtype=float)
        changed = True

    if (
        "spatial_alpha" in results
        and "spatial_modes" in results
        and "omega_spatial" in results
        and "spatial_residuals" not in results
    ):
        from linstab2d.boundary_conditions.dirichlet import apply_dirichlet_spatial
        from linstab2d.operators.spatial import build_spatial_operator

        mesh, baseflow = ensure_problem()
        _apply_streaky_sponge(mesh, mini_cfg)
        Lc, R, Lw, idx = build_spatial_operator(
            mesh,
            baseflow,
            model="2D",
            floquet_exp=float(results["floquet_exp"]),
        )
        Lc, R, _ = apply_dirichlet_spatial(Lc, R, idx, "tb", "uvwT", mesh.ngp)
        omega_spatial = complex(np.asarray(results["omega_spatial"]).reshape(-1)[0])
        results["spatial_residuals"] = _compute_spatial_residuals(
            Lc,
            Lw,
            R,
            omega_spatial,
            np.asarray(results["spatial_alpha"]),
            np.asarray(results["spatial_modes"]),
        )
        changed = True

    if changed:
        _save_npz(results)
    return results


def _reconstruct_mode_field(
    X: np.ndarray,
    Y: np.ndarray,
    mode: np.ndarray,
    idx_block: np.ndarray,
    used_ind: np.ndarray,
    floquet_exp: float,
) -> np.ndarray:
    """Reconstruct one Floquet mode component on the physical mesh."""
    from linstab2d.utils.floquet import reconstruct_floquet

    return reconstruct_floquet(
        X,
        mode[np.asarray(idx_block)],
        used_ind=used_ind,
        floquet_exp=floquet_exp,
    ).reshape(X.shape, order="F")


def _tile_periodic_field(
    X: np.ndarray,
    Y: np.ndarray,
    field: np.ndarray,
    *,
    n_tiles: int,
    phase_shift: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile a periodic or Floquet field for multi-period display."""
    width = float(np.max(X) - np.min(X))
    x_tiles = []
    y_tiles = []
    field_tiles = []
    for k in range(n_tiles):
        x_tiles.append(np.asarray(X, dtype=float) + k * width)
        y_tiles.append(np.asarray(Y, dtype=float))
        if phase_shift is None:
            field_tiles.append(np.asarray(field))
        else:
            field_tiles.append(np.asarray(field) * np.exp(1j * phase_shift * k))
    return (
        np.concatenate(x_tiles, axis=1),
        np.concatenate(y_tiles, axis=1),
        np.concatenate(field_tiles, axis=1),
    )


def _resolve_baseflow_plot_field(data: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    """Prefer the streamwise W baseflow for plotting when it is available."""
    if "baseflow_w" in data:
        return np.asarray(data["baseflow_w"], dtype=float), r"Streaky boundary-layer baseflow $W$"
    if "baseflow_u" in data:
        return np.asarray(data["baseflow_u"], dtype=float), r"Streaky boundary-layer baseflow $U$"
    raise KeyError("Expected either 'baseflow_w' or 'baseflow_u' in Example 3 data")


def _compute(cfg: dict) -> None:
    """Run temporal and spatial eigensolutions."""
    from linstab2d.baseflow import load_streaky_bl
    from linstab2d.boundary_conditions.dirichlet import (
        apply_dirichlet,
        apply_dirichlet_petsc,
        apply_dirichlet_spatial,
        apply_dirichlet_spatial_petsc,
    )
    from linstab2d.operators.assembly import build_lnse_operator, build_lnse_operator_petsc
    from linstab2d.operators.energy_norm import compute_chu_norm
    from linstab2d.operators.spatial import build_spatial_operator, build_spatial_operator_petsc
    from linstab2d.solvers.eigenvalue import temporal_eigenvalues
    from linstab2d.solvers.petsc_utils import petsc_to_scipy
    from linstab2d.solvers.spatial_eig import spatial_eigenvalues

    solver_cfg = cfg.get("solver")
    assembly_backend = str(cfg.get("assembly_backend", "scipy")).lower()
    use_petsc_assembly = assembly_backend == "petsc"
    if assembly_backend not in {"scipy", "petsc"}:
        raise ValueError(
            f"Invalid assembly_backend={assembly_backend!r}; expected 'scipy' or 'petsc'"
        )

    print(f"Streaky BL grid points: Nx={cfg['nx']}, Ny={cfg['ny']}")
    mesh, Lx, Ly = _build_streaky_mesh(cfg)

    n_periods = cfg["n_periods"]
    floquet_exp = 0.0 if n_periods == 0 else 2 * np.pi / (Lx * n_periods)
    alpha_temp = cfg["alpha_temp"]
    temporal_targets = parse_complex_list(cfg["temporal_targets"])
    temporal_quotas = parse_int_list(cfg["temporal_quotas"])
    if len(temporal_targets) != len(temporal_quotas):
        raise ValueError(
            "temporal_targets and temporal_quotas must have the same length."
        )
    omega_tar = temporal_targets[0]
    alpha_target = float(alpha_temp)
    n_modes = cfg["n_modes"]
    arnoldi_ncv = cfg["ncv"]
    spatial_omega_imag_shift = cfg["spatial_omega_imag_shift"]
    run_temporal = cfg["run_temporal"]
    run_spatial = cfg["run_spatial"]
    dedupe_tol = cfg.get("dedupe_tol", 1e-4)
    if not run_temporal and not run_spatial:
        raise ValueError(
            "At least one of run_temporal or run_spatial must be true."
        )

    baseflow = load_streaky_bl(mesh)
    _apply_streaky_sponge(mesh, cfg)

    borders = "tb"
    variables = "uvwT"
    results = _load_existing_npz()
    used_ind = mesh.used_ind

    def _to_scipy_matrix(mat):
        return petsc_to_scipy(mat, format="csc") if use_petsc_assembly else mat

    common = {
        "X": mesh.X,
        "Y": mesh.Y,
        "used_ind": used_ind,
        "floquet_exp": floquet_exp,
        "n_modes": n_modes,
        "alpha_temp": alpha_temp,
        "omega_tar_real": float(np.real(omega_tar)),
        "omega_tar_imag": float(np.imag(omega_tar)),
        "alpha_target": float(alpha_target),
        "baseflow_u": baseflow.U,
        "baseflow_w": baseflow.W,
    }

    temporal_omega: np.ndarray | None = None
    if run_temporal:
        if not run_spatial:
            for key in ("spatial_alpha", "spatial_modes", "omega_spatial", "spatial_residuals"):
                results.pop(key, None)
        if use_petsc_assembly:
            L0, idx = build_lnse_operator_petsc(
                mesh,
                baseflow,
                kx=alpha_temp,
                model="2D",
                floquet_exp=floquet_exp,
                solver_cfg=solver_cfg,
            )
            L0, _ = apply_dirichlet_petsc(L0, idx, borders, variables, mesh.ngp)
        else:
            L0, idx = build_lnse_operator(
                mesh,
                baseflow,
                kx=alpha_temp,
                model="2D",
                floquet_exp=floquet_exp,
            )
            L0, _ = apply_dirichlet(L0, idx, borders, variables, mesh.ngp)

        W_int = mesh.W.ravel(order="F")[used_ind]
        rho_p = baseflow.RHO.ravel(order="F")[used_ind]
        T_p = baseflow.T.ravel(order="F")[used_ind]
        W_diag, _ = compute_chu_norm(W_int, rho_p, T_p, baseflow.kappa, baseflow.Ma, mesh.ngp)

        temporal_batches: list[tuple[complex, np.ndarray, np.ndarray]] = []
        temporal_nev = max(max(temporal_quotas), 20)
        for temporal_target in temporal_targets:
            omegas_i, modes_i = temporal_eigenvalues(
                L0,
                temporal_nev,
                temporal_target,
                W_diag=W_diag,
                ncv=arnoldi_ncv,
                solver_cfg=solver_cfg,
            )
            temporal_batches.append((temporal_target, omegas_i, modes_i))

        temporal_omega, temporal_modes = _merge_temporal_target_batches(
            temporal_batches,
            temporal_quotas,
            dedupe_tol=dedupe_tol,
        )

        iL0 = 1j * _to_scipy_matrix(L0)
        temporal_residuals = np.array(
            [
                np.linalg.norm(iL0 @ q - omega * q)
                / max(abs(omega) * float(np.linalg.norm(q)), 1e-300)
                for omega, q in zip(temporal_omega, temporal_modes.T)
            ],
            dtype=float,
        )
        print("Temporal eigenvalue residuals ||iL0 q - w q|| / (|w| ||q||):")
        for i, (omega, res) in enumerate(zip(temporal_omega, temporal_residuals)):
            print(f"  [{i}]  w={omega.real:.5f}{omega.imag:+.5f}j   res={res:.3e}")

        results.update(common)
        results.update(
            {
                "temporal_omega": temporal_omega,
                "temporal_modes": temporal_modes,
                "temporal_residuals": temporal_residuals,
                "idx_rho_j": idx["rho_j"],
                "idx_u_j": idx["u_j"],
                "idx_v_j": idx["v_j"],
                "idx_w_j": idx["w_j"],
                "idx_T_j": idx["T_j"],
                "omega_target_real": float(np.real(omega_tar)),
                "omega_target_imag": float(np.imag(omega_tar)),
            }
        )
        _save_npz(results)

    if run_spatial:
        if temporal_omega is None:
            if "temporal_omega" not in results:
                raise ValueError(
                    "Spatial Example 3 run requires temporal_omega in the existing "
                    "NPZ or run_temporal=true in the JSONC config."
                )
            temporal_omega = np.asarray(results["temporal_omega"], dtype=complex)

        omega_spatial = leading_mode_real_frequency(
            temporal_omega,
            imag_shift=spatial_omega_imag_shift,
        )

        if use_petsc_assembly:
            Lc, R, Lw, idx = build_spatial_operator_petsc(
                mesh,
                baseflow,
                model="2D",
                floquet_exp=floquet_exp,
                solver_cfg=solver_cfg,
            )
            Lc, R, _ = apply_dirichlet_spatial_petsc(Lc, R, idx, borders, variables, mesh.ngp)
        else:
            Lc, R, Lw, idx = build_spatial_operator(
                mesh,
                baseflow,
                model="2D",
                floquet_exp=floquet_exp,
            )
            Lc, R, _ = apply_dirichlet_spatial(Lc, R, idx, borders, variables, mesh.ngp)
        spatial_alpha, spatial_modes = spatial_eigenvalues(
            Lc,
            Lw,
            R,
            omega_spatial,
            n_modes,
            alpha_target,
            ncv=arnoldi_ncv,
            solver_cfg=solver_cfg,
        )
        spatial_residuals = _compute_spatial_residuals(
            _to_scipy_matrix(Lc),
            _to_scipy_matrix(Lw),
            _to_scipy_matrix(R),
            omega_spatial,
            spatial_alpha,
            spatial_modes,
        )
        print("Spatial eigenvalue residuals ||A q - a B q|| / (|a| ||q||):")
        for i, (alpha, res) in enumerate(zip(spatial_alpha[:5], spatial_residuals[:5])):
            print(f"  [{i}]  a={alpha.real:.5f}{alpha.imag:+.5f}j   res={res:.3e}")

        results.update(common)
        results.update(
            {
                "spatial_alpha": spatial_alpha,
                "spatial_modes": spatial_modes,
                "omega_spatial": omega_spatial,
                "spatial_residuals": spatial_residuals,
                "idx_rho_j": idx["rho_j"],
                "idx_u_j": idx["u_j"],
                "idx_v_j": idx["v_j"],
                "idx_w_j": idx["w_j"],
                "idx_T_j": idx["T_j"],
            }
        )
        _save_npz(results)


def _plot(data: dict[str, np.ndarray]) -> None:
    """Plot publication-style baseflow, temporal, spatial, and diagnostic figures."""
    import matplotlib.pyplot as plt

    has_temporal = "temporal_omega" in data and "temporal_modes" in data
    has_spatial = "spatial_alpha" in data and "spatial_modes" in data
    if not has_temporal and not has_spatial:
        raise ValueError("Example 3 NPZ does not contain temporal or spatial results to plot.")

    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    used_ind = np.asarray(data["used_ind"], dtype=int)
    floquet_exp = float(np.asarray(data["floquet_exp"]).reshape(-1)[0])
    idx_rho = np.asarray(data["idx_rho_j"], dtype=int)
    idx_u = np.asarray(data["idx_u_j"], dtype=int)
    idx_v = np.asarray(data["idx_v_j"], dtype=int)
    idx_w = np.asarray(data["idx_w_j"], dtype=int)
    idx_T = np.asarray(data["idx_T_j"], dtype=int)
    n_modes = int(np.asarray(data["n_modes"]).reshape(-1)[0])
    baseflow_field, baseflow_title = _resolve_baseflow_plot_field(data)
    phase_step = floquet_exp * (float(np.max(X)) - float(np.min(X)))

    X_base, Y_base, U_base = _tile_periodic_field(X, Y, baseflow_field, n_tiles=4)
    fig_base, ax_base = plt.subplots(1, 1, figsize=(11.5, 3.0))
    plot_scalar_panel(
        ax_base,
        X_base,
        Y_base,
        U_base,
        title=baseflow_title,
        cmap="turbo",
        xlim=(0.0, 4.0 * np.pi),
        ylim=(0.0, 5.0),
        aspect="auto",
        levels=101,
        colorbar=False,
        title_pad=4.0,
    )
    for xpos, width, color in [
        (np.pi, np.pi, "#b94f4f"),
        (2.0 * np.pi, 2.0 * np.pi, "#4a76b8"),
    ]:
        ax_base.add_patch(
            plt.Rectangle(
                (xpos, 0.0),
                width,
                5.0,
                fill=False,
                lw=1.5,
                ls="--",
                edgecolor=color,
            )
        )
    label_axes(ax_base)
    save_figure(fig_base, OUTPUT_BASEFLOW)

    if has_temporal:
        temporal_omega = np.asarray(data["temporal_omega"], dtype=complex)
        temporal_modes = np.asarray(data["temporal_modes"], dtype=complex)
        temporal_residuals = np.asarray(data["temporal_residuals"], dtype=float)
        temporal_order = top_imag_indices(temporal_omega, 2)
        matlab_temporal = _load_matlab_temporal_omega()

        fig_temporal = plt.figure(figsize=(12.5, 7.0))
        grid = fig_temporal.add_gridspec(
            max(1, temporal_order.size),
            2,
            width_ratios=(1.1, 1.7),
            hspace=0.26,
            wspace=0.24,
        )
        ax_spec = fig_temporal.add_subplot(grid[:, 0])
        if matlab_temporal is not None:
            ax_spec.plot(
                np.real(matlab_temporal),
                np.imag(matlab_temporal),
                "o",
                mfc="white",
                mec="black",
                mew=1.0,
                ms=7,
                label="MATLAB",
                zorder=2,
            )
        ax_spec.plot(
            np.real(temporal_omega),
            np.imag(temporal_omega),
            "o",
            color="#2a5caa",
            ms=5,
            alpha=0.9,
            label="Python",
            zorder=3,
        )
        for idx in temporal_order:
            ax_spec.plot(
                np.real(temporal_omega[idx]),
                np.imag(temporal_omega[idx]),
                "D",
                color="#d99100",
                mec="black",
                mew=0.5,
                ms=8,
            )
        ax_spec.set_title(f"Temporal spectrum, max r = {np.max(temporal_residuals):.1e}", pad=4.0)
        ax_spec.set_xlabel(r"$\Re(\omega)$")
        ax_spec.set_ylabel(r"$\Im(\omega)$")
        ax_spec.grid(True, alpha=0.3)
        format_scientific_ticks(ax_spec)
        if matlab_temporal is not None:
            ax_spec.legend(loc="upper right", frameon=False, fontsize=9)

        for row, idx in enumerate(temporal_order):
            field = _reconstruct_mode_field(
                X,
                Y,
                temporal_modes[:, int(idx)],
                idx_w,
                used_ind,
                floquet_exp,
            )
            Xt, Yt, Ft = _tile_periodic_field(
                X,
                Y,
                field,
                n_tiles=2,
                phase_shift=phase_step,
            )
            ax_mode = fig_temporal.add_subplot(grid[row, 1])
            plot_signed_panel(
                ax_mode,
                Xt,
                Yt,
                np.real(Ft),
                title=(
                    rf"$w^{{({row + 1})}}$, "
                    rf"$\omega = {format_complex(temporal_omega[idx], real_precision=4)}$"
                ),
                xlim=(0.0, 2.0 * np.pi),
                ylim=(0.0, 4.0),
                aspect="auto",
                colorbar=False,
                title_pad=4.0,
            )
            ax_mode.set_ylabel(r"$y$")
            if row == temporal_order.size - 1:
                ax_mode.set_xlabel(r"$x$")
            else:
                ax_mode.set_xticklabels([])
        set_suptitle(fig_temporal, r"Temporal streaky-boundary-layer summary at $\beta = 0.6283$", y=0.972)
        apply_figure_layout(fig_temporal, rect=(0.0, 0.0, 1.0, 0.94), pad=0.7)
        save_figure(fig_temporal, OUTPUT_PNG)

    if has_spatial:
        spatial_alpha = np.asarray(data["spatial_alpha"], dtype=complex)
        spatial_modes = np.asarray(data["spatial_modes"], dtype=complex)
        spatial_residuals = np.asarray(data["spatial_residuals"], dtype=float)
        spatial_order = smallest_abs_indices(spatial_alpha, 2)
        n_states = int(np.max(np.hstack([idx_u, idx_v, idx_w, idx_rho, idx_T])) + 1)
        matlab_spatial = _load_matlab_spatial_alpha()

        fig_spatial = plt.figure(figsize=(12.0, 5.8))
        grid = fig_spatial.add_gridspec(1, 2, width_ratios=(1.15, 1.4), wspace=0.24)
        ax_spec = fig_spatial.add_subplot(grid[0, 0])
        if matlab_spatial is not None:
            ax_spec.plot(
                np.real(matlab_spatial),
                np.imag(matlab_spatial),
                "o",
                mfc="white",
                mec="black",
                mew=1.0,
                ms=7,
                label="MATLAB subset",
                zorder=2,
            )
        ax_spec.plot(
            np.real(spatial_alpha),
            np.imag(spatial_alpha),
            "o",
            color="#2a5caa",
            ms=5,
            alpha=0.9,
            label="Python",
            zorder=3,
        )
        for idx in spatial_order:
            ax_spec.plot(
                np.real(spatial_alpha[idx]),
                np.imag(spatial_alpha[idx]),
                "D",
                color="#d99100",
                mec="black",
                mew=0.5,
                ms=8,
            )
        ax_spec.set_title(f"Spatial spectrum, max r = {np.max(spatial_residuals):.1e}", pad=4.0)
        ax_spec.set_xlabel(r"$\Re(\alpha)$")
        ax_spec.set_ylabel(r"$\Im(\alpha)$")
        ax_spec.grid(True, alpha=0.3)
        format_scientific_ticks(ax_spec)
        if matlab_spatial is not None:
            ax_spec.legend(loc="upper left", frameon=False, fontsize=9)

        spatial_mode = spatial_modes[:n_states, int(spatial_order[0])]
        field = _reconstruct_mode_field(
            X,
            Y,
            spatial_mode,
            idx_w,
            used_ind,
            floquet_exp,
        )
        Xt, Yt, Ft = _tile_periodic_field(
            X,
            Y,
            field,
            n_tiles=2,
            phase_shift=phase_step,
        )
        ax_mode = fig_spatial.add_subplot(grid[0, 1])
        plot_signed_panel(
            ax_mode,
            Xt,
            Yt,
            np.real(Ft),
            title=(
                rf"Leading spatial $w$, "
                rf"$\alpha = {format_complex(spatial_alpha[int(spatial_order[0])])}$"
            ),
            xlim=(0.0, 2.0 * np.pi),
            ylim=(0.0, 4.0),
            aspect="auto",
            colorbar=False,
            title_pad=4.0,
        )
        label_axes(ax_mode)
        set_suptitle(fig_spatial, "Spatial streaky-boundary-layer summary", y=0.972)
        apply_figure_layout(fig_spatial, rect=(0.0, 0.0, 1.0, 0.94), pad=0.7)
        save_figure(fig_spatial, OUTPUT_SPATIAL)

    fig_support, axes_support = plt.subplots(1, 2, figsize=(12, 4.8))
    if has_temporal:
        temporal_omega = np.asarray(data["temporal_omega"], dtype=complex)
        temporal_residuals = np.asarray(data["temporal_residuals"], dtype=float)
        matlab_temporal = _load_matlab_temporal_omega()
        if matlab_temporal is not None:
            axes_support[0].plot(
                np.real(matlab_temporal),
                np.imag(matlab_temporal),
                "o",
                mfc="white",
                mec="black",
                mew=1.0,
                ms=7,
                label="MATLAB",
            )
        axes_support[0].plot(
            np.real(temporal_omega),
            np.imag(temporal_omega),
            "o",
            color="#2a5caa",
            ms=5,
            label="Python",
        )
        axes_support[0].set_title(f"Temporal spectrum, max r = {np.max(temporal_residuals):.1e}", pad=4.0)
        axes_support[0].set_xlabel(r"$\Re(\omega)$")
        axes_support[0].set_ylabel(r"$\Im(\omega)$")
        axes_support[0].grid(True, alpha=0.3)
        format_scientific_ticks(axes_support[0])
        if matlab_temporal is not None:
            axes_support[0].legend(loc="upper right", frameon=False, fontsize=9)
    else:
        axes_support[0].set_visible(False)

    if has_spatial:
        spatial_alpha = np.asarray(data["spatial_alpha"], dtype=complex)
        spatial_residuals = np.asarray(data["spatial_residuals"], dtype=float)
        matlab_spatial = _load_matlab_spatial_alpha()
        if matlab_spatial is not None:
            axes_support[1].plot(
                np.real(matlab_spatial),
                np.imag(matlab_spatial),
                "o",
                mfc="white",
                mec="black",
                mew=1.0,
                ms=7,
                label="MATLAB subset",
            )
        axes_support[1].plot(
            np.real(spatial_alpha),
            np.imag(spatial_alpha),
            "o",
            color="#2a5caa",
            ms=5,
            label="Python",
        )
        axes_support[1].set_title(f"Spatial spectrum, max r = {np.max(spatial_residuals):.1e}", pad=4.0)
        axes_support[1].set_xlabel(r"$\Re(\alpha)$")
        axes_support[1].set_ylabel(r"$\Im(\alpha)$")
        axes_support[1].grid(True, alpha=0.3)
        format_scientific_ticks(axes_support[1])
        if matlab_spatial is not None:
            axes_support[1].legend(loc="upper left", frameon=False, fontsize=9)
    else:
        axes_support[1].set_visible(False)
    set_suptitle(fig_support, "Streaky boundary-layer diagnostics", y=0.972)
    apply_figure_layout(fig_support, rect=(0.0, 0.0, 1.0, 0.94), pad=0.7)
    save_figure(fig_support, OUTPUT_SPECTRA)


def main() -> None:
    cfg = load_jsonc(SCRIPT.with_suffix(".jsonc"))
    data = _load_existing_npz()
    if not data:
        print(format_cache_recompute_message(OUTPUT_NPZ, None))
        _compute(cfg)
        data = _load_existing_npz()
    if not data:
        raise RuntimeError(f"Failed to generate {OUTPUT_NPZ}")

    data = _augment_cached_results(data)
    _plot(data)


if __name__ == "__main__":
    main()
