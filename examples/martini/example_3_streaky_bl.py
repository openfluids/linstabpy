#!/usr/bin/env python3
"""
Example 3: Streaky boundary-layer temporal and spatial analysis.

Translates example_3_Streaky_BL.m and the four paper campaigns
(``example_3_paper_*.yaml``) with periodic x-domain, Hanifi-like y stretch,
temporal eigenvalues, and spatial eigenvalues in generalized (alpha) form.

OUTPUTS (four paper NPZ bundles + campaign figures):
  - example_3_paper_temporal_Lx2pi_nP2.npz
  - example_3_paper_temporal_Lxpi_nfloq0.npz
  - example_3_paper_temporal_Lxpi_nfloq2.npz
  - example_3_paper_spatial_Lxpi_nP2.npz
  - example_3_streaky_bl_baseflow.png
  - example_3_streaky_bl_spectrum_*.png
  - example_3_streaky_bl_modes_temporal_* / example_3_streaky_bl_modes_spatial_* mosaics

USAGE:   python example_3_streaky_bl.py
         rm example_3_paper_*.npz && python example_3_streaky_bl.py  # recompute all

Single-case override (batch / debugging):
  LINSTABPY_EXAMPLE_STEM=example_3_paper_spatial_Lxpi_nP2 \\
  LINSTABPY_EXAMPLE_CONFIG=examples/martini/example_3_paper_spatial_Lxpi_nP2.yaml \\
  python examples/martini/example_3_streaky_bl.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from linstabpy.utils.yaml_io import load_config, parse_complex_list, parse_int_list

from examples._helpers import (
    format_cache_recompute_message,
    leading_mode_real_frequency,
    load_or_compute,
)
from examples.martini.example_3_plots import plot_example3_figures

SCRIPT = Path(__file__).resolve()
EXAMPLE_DIR = SCRIPT.parent

PAPER_CASES = (
    "example_3_paper_temporal_Lx2pi_nP2",
    "example_3_paper_temporal_Lxpi_nfloq0",
    "example_3_paper_temporal_Lxpi_nfloq2",
    "example_3_paper_spatial_Lxpi_nP2",
)


def _resolve_cases() -> tuple[str, ...]:
    stem = os.environ.get("LINSTABPY_EXAMPLE_STEM", "").strip()
    if stem:
        return (stem,)
    return PAPER_CASES


def _config_path(stem: str) -> Path:
    env_cfg = os.environ.get("LINSTABPY_EXAMPLE_CONFIG", "").strip()
    if env_cfg and len(_resolve_cases()) == 1:
        return Path(env_cfg)
    return EXAMPLE_DIR / f"{stem}.yaml"


def _output_npz(stem: str) -> Path:
    return EXAMPLE_DIR / f"{stem}.npz"


def _configure_floquet_shift(cfg: dict) -> str:
    """Apply Floquet FD mode for this run (Example 3 pins LinStab2D / Martini parity)."""
    mode = str(cfg.get("floquet_shift", "matlab_parity"))
    os.environ["LINSTABPY_FLOQUET_SHIFT"] = mode
    return mode


def _load_existing_npz(npz_path: Path) -> dict[str, np.ndarray]:
    """Load an existing NPZ artifact if present and valid."""
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (FileNotFoundError, EOFError, OSError, ValueError):
        return {}


def _save_npz(npz_path: Path, data: dict[str, np.ndarray | float | complex | int]) -> None:
    """Save a mixed metadata/results dict to the example NPZ path."""
    payload: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            payload[key] = value
        else:
            payload[key] = np.asarray(value)
    np.savez_compressed(npz_path, **payload)
    print(f"Results saved to {npz_path}")


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


def _build_streaky_mesh(cfg: dict) -> tuple[object, float, float]:
    """Build the Example 3 mesh and mapped physical coordinates."""
    from linstabpy.mesh import Mesh

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


def _apply_streaky_sponge(mesh, cfg: dict) -> None:
    """Apply the MATLAB-like top sponge used in Example 3."""
    sponge = cfg["sponge"]
    Ly = cfg["Ly"]
    ys = Ly - sponge["ys_offset"]
    ys_trans = sponge["ys_trans"]
    sponge_amp = sponge["amplitude"]
    mesh.sponge = sponge_amp / 2 * (np.tanh((mesh.Y - ys) / ys_trans) + 1.0)


def _compute_spatial_residuals(
    L,
    R,
    spatial_alpha: np.ndarray,
    spatial_modes: np.ndarray,
) -> np.ndarray:
    """Compute residuals for ``L q = alpha R q`` after Dirichlet rows."""
    residuals = []
    for alpha, q in zip(
        np.asarray(spatial_alpha, dtype=complex).reshape(-1),
        np.asarray(spatial_modes, dtype=complex).T,
    ):
        numer = np.linalg.norm(L @ q - alpha * (R @ q))
        denom = max(abs(alpha) * float(np.linalg.norm(q)), 1e-300)
        residuals.append(float(numer / denom))
    return np.asarray(residuals, dtype=float)


def _resolve_baseflow_plot_field(data: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    """Prefer the streamwise W baseflow for plotting when it is available."""
    if "baseflow_w" in data:
        return np.asarray(data["baseflow_w"], dtype=float), r"Streaky boundary-layer baseflow $W$"
    if "baseflow_u" in data:
        return np.asarray(data["baseflow_u"], dtype=float), r"Streaky boundary-layer baseflow $U$"
    raise KeyError("Expected either 'baseflow_w' or 'baseflow_u' in Example 3 data")


def _augment_cached_results(
    results: dict[str, np.ndarray],
    cfg: dict,
    npz_path: Path,
) -> dict[str, np.ndarray]:
    """Backfill fields needed by the revised plotting/parity flow."""
    if not results:
        return results

    changed = False
    mesh = None
    baseflow = None

    mini_cfg: dict = {
        "nx": 0,
        "ny": 0,
        "Lx": cfg.get("Lx", np.pi),
        "Ly": cfg.get("Ly", 5.0),
        "fd_order": cfg.get("fd_order", 4),
        "alpha_filter": cfg.get("alpha_filter", 0.0),
        "x_periodicity": cfg.get("x_periodicity", True),
        "y_stretch_yi": cfg.get("y_stretch_yi", 2.0),
        "sponge": cfg.get(
            "sponge",
            {"amplitude": 0.1, "ys_offset": 1.0, "ys_trans": 0.5},
        ),
    }

    def ensure_problem() -> tuple[object, object]:
        nonlocal mesh, baseflow
        if mesh is None or baseflow is None:
            from examples.martini.matlab_baseflow import load_streaky_bl

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
        _configure_floquet_shift(cfg)
        from linstabpy.operators.spatial import (
            build_spatial_operator,
            spatial_lhs_with_dirichlet,
        )

        mesh, baseflow = ensure_problem()
        _apply_streaky_sponge(mesh, mini_cfg)
        Lc, R, Lw, idx = build_spatial_operator(
            mesh,
            baseflow,
            model="2D",
            floquet_exp=float(results["floquet_exp"]),
        )
        omega_spatial = complex(np.asarray(results["omega_spatial"]).reshape(-1)[0])
        L, R, _ = spatial_lhs_with_dirichlet(
            Lc,
            Lw,
            R,
            omega_spatial,
            idx,
            "tb",
            "uvwT",
            mesh.ngp,
            use_petsc=False,
        )
        results["spatial_residuals"] = _compute_spatial_residuals(
            L,
            R,
            np.asarray(results["spatial_alpha"]),
            np.asarray(results["spatial_modes"]),
        )
        changed = True

    if changed:
        _save_npz(npz_path, results)
    return results


def compute(cfg: dict, npz_path: Path) -> None:
    """Run temporal and/or spatial eigensolutions for one paper campaign."""
    from examples.martini.matlab_baseflow import load_streaky_bl
    from linstabpy.boundary_conditions.dirichlet import (
        apply_dirichlet,
        apply_dirichlet_petsc,
    )
    from linstabpy.operators.assembly import build_lnse_operator, build_lnse_operator_petsc
    from linstabpy.operators.energy_norm import compute_chu_norm
    from linstabpy.operators.spatial import (
        build_spatial_operator,
        build_spatial_operator_petsc,
        spatial_lhs_with_dirichlet,
    )
    from linstabpy.solvers.petsc_utils import petsc_to_scipy
    from linstabpy.stability import eigenvalues as temporal_eigenvalues
    from linstabpy.stability import spatial_eigenvalues

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
        raise ValueError("temporal_targets and temporal_quotas must have the same length.")
    omega_tar = temporal_targets[0]
    alpha_target = float(alpha_temp)
    n_modes = cfg["n_modes"]
    arnoldi_ncv = cfg["ncv"]
    spatial_omega_imag_shift = cfg["spatial_omega_imag_shift"]
    run_temporal = cfg["run_temporal"]
    run_spatial = cfg["run_spatial"]
    dedupe_tol = cfg.get("dedupe_tol", 1e-4)
    if not run_temporal and not run_spatial:
        raise ValueError("At least one of run_temporal or run_spatial must be true.")

    floquet_shift = _configure_floquet_shift(cfg)
    print(f"Floquet FD shift: {floquet_shift}")

    baseflow = load_streaky_bl(mesh)
    _apply_streaky_sponge(mesh, cfg)

    borders = "tb"
    variables = "uvwT"
    results = _load_existing_npz(npz_path)
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
        "floquet_shift": floquet_shift,
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
        _save_npz(npz_path, results)

    if run_spatial:
        if temporal_omega is None:
            if "temporal_omega" not in results:
                raise ValueError(
                    "Spatial Example 3 run requires temporal_omega in the existing "
                    "NPZ or run_temporal=true in the YAML config."
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
        else:
            Lc, R, Lw, idx = build_spatial_operator(
                mesh,
                baseflow,
                model="2D",
                floquet_exp=floquet_exp,
            )
        L, R, _ = spatial_lhs_with_dirichlet(
            Lc,
            Lw,
            R,
            omega_spatial,
            idx,
            borders,
            variables,
            mesh.ngp,
            use_petsc=use_petsc_assembly,
        )
        spatial_alpha, spatial_modes = spatial_eigenvalues(
            L,
            Lw,
            R,
            0.0,
            n_modes,
            alpha_target,
            ncv=arnoldi_ncv,
            solver_cfg=solver_cfg,
        )
        spatial_residuals = _compute_spatial_residuals(
            _to_scipy_matrix(L),
            _to_scipy_matrix(R),
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
        _save_npz(npz_path, results)


def run_case(stem: str) -> dict[str, np.ndarray]:
    """Load or compute one paper campaign NPZ."""
    cfg_path = _config_path(stem)
    npz_path = _output_npz(stem)
    cfg = load_config(cfg_path)
    if not npz_path.is_file():
        print(format_cache_recompute_message(npz_path, None))
    data = load_or_compute(npz_path, lambda: compute(cfg, npz_path))
    return _augment_cached_results(dict(data), cfg, npz_path)


def plot_all(example_dir: Path = EXAMPLE_DIR) -> list[Path]:
    """Plot all four paper campaigns using LinStab2D fixtures when present."""
    return plot_example3_figures(example_dir)


def main() -> None:
    cases = _resolve_cases()
    for stem in cases:
        print(f"\n=== Example 3 campaign: {stem} ===")
        run_case(stem)

    if len(cases) == len(PAPER_CASES):
        print("\n=== Example 3 presentation figures ===")
        for path in plot_all():
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
