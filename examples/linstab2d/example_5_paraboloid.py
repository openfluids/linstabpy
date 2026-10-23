#!/usr/bin/env python3
"""
Example 5: Paraboloid resolvent analysis.

Translates example_5_Paraboloid.m using a curvilinear transformation to a
parabolic physical grid and an axisymmetric model.

OUTPUTS.
  - example_5_paraboloid.npz.
  - example_5_paraboloid.log.

Mesh parity (see ``example_5_D2y_asymm_fix.md``).
  - ``Mesh.deform`` metric terms in ``_deform_full_2d`` use full ``Dy``/``D2y``
    when building ``dJ``/``Q`` for symm/asymm families (fixes ``D2y_asymm``).

USAGE:   python example_5_paraboloid.py
         rm example_5_paraboloid.npz && python example_5_paraboloid.py
         python example_5_plots.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from examples._helpers import load_or_compute
from linstabpy.utils._solver_cfg import as_bool
from linstabpy.utils.yaml_io import load_config

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
LOG_PATH = SCRIPT.with_suffix(".log")


def compute(cfg: dict) -> None:
    """Run resolvent analysis on the paraboloid-mapped cavity.

    Uses m = 5 azimuthal Fourier mode and axy model.
    """
    import scipy.sparse as sp

    from examples.linstab2d.matlab_baseflow import load_paraboloid
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.operators.assembly import build_lnse_system_result
    from linstabpy.operators.energy_norm import compute_chu_norm
    from linstabpy.resolvent.analysis import resolvent_analysis
    from linstabpy.solvers.petsc_utils import scipy_to_petsc

    solver_cfg = cfg.get("solver")

    omega = cfg["omega"]
    omega_shift_fallback = cfg["omega_shift_fallback"]
    use_shift_fallback = cfg["use_omega_shift_fallback"]
    n_modes = cfg["n_modes"]
    m = cfg["m"]
    fd_order = cfg["fd_order"]

    Nx = cfg["nx"]
    Ny = cfg["ny"]
    Lx = cfg["Lx"]
    Ly = cfg["Ly"]

    xrange = tuple(cfg["xrange"])
    yrange = tuple(cfg["yrange"])
    cmesh = Mesh.create(
        xrange,
        yrange,
        Nx,
        Ny,
        fd_order,
        y_symmetry=cfg["y_symmetry"],
        x_periodicity=cfg["x_periodicity"],
        alpha_filter=cfg["alpha_filter"],
    )

    X = cmesh.X.copy()
    x_nonzero = X != 0.0
    X_mapped = np.zeros_like(X)
    X_mapped[x_nonzero] = 1 / (1 / (X[x_nonzero] ** 4) * np.sign(X[x_nonzero]) + 1 / X[x_nonzero])
    X = X_mapped
    X[np.isnan(X)] = 0.0
    X = X / np.max(np.abs(X))
    Y = cmesh.Y.copy()
    Y = 1 - (1 - Y) ** 1.4
    Y = 1 - (1 - Y) ** 1.4
    Y = np.maximum(Y, 1e-10)

    # Scale to physical domain before Z² conformal mapping (MATLAB convention)
    X = X * Lx - 0.5
    Y = Y * Ly

    Z = X + 1j * Y
    Xp = -np.real(Z**2)
    Yp = -np.imag(Z**2)
    mesh = cmesh.deform(X_new=Xp, Y_new=Yp)

    baseflow = load_paraboloid(mesh)
    sutherland_air(baseflow)

    xs = -0.5 - Lx + cfg["sponge"]["xs_offset"]
    xs_trans = Lx * cfg["sponge"]["xs_trans_fraction"]
    ys = Ly - cfg["sponge"]["ys_offset"]
    ys_trans = cfg["sponge"]["ys_trans"]
    sponge_amp = cfg["sponge"]["amplitude"]
    # MATLAB defines the sponge on the pre-conformal (x, y) grid, before the
    # final Z^2 mapping to the physical mesh.
    sponge_x = np.tanh(-(X - xs) / xs_trans) + 1
    sponge_y = np.tanh((Y - ys) / ys_trans) + 1
    mesh.sponge = sponge_amp / 2 * np.maximum(sponge_x, sponge_y)

    assembly_backend = cfg.get("assembly_backend")
    bc_penalty = cfg["bc"].get("penalty", 1e4)
    lnse_system = build_lnse_system_result(
        mesh,
        baseflow,
        model=cfg["model"],
        m=m,
        bc_recipe=cfg["bc"]["recipe"],
        bc_penalty=bc_penalty,
        backend=assembly_backend,
        solver_cfg=solver_cfg,
        matlab_parity_dmudt=as_bool(cfg.get("matlab_parity_dmudt", False)),
    )
    L0 = lnse_system.operator
    idx = lnse_system.idx
    bc_indices = lnse_system.bc_indices

    used_ind = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[used_ind]
    rho = baseflow.RHO.ravel(order="F")[used_ind]
    T = baseflow.T.ravel(order="F")[used_ind]
    R = mesh.Y.ravel(order="F")[used_ind]
    W_diag, _ = compute_chu_norm(
        W_int,
        rho,
        T,
        baseflow.kappa,
        baseflow.Ma,
        mesh.ngp,
        model=cfg["model"],
        R=R,
    )

    B_diag = np.ones(5 * mesh.ngp)
    var_map = {"rho": "rho_j", "u": "u_j", "v": "v_j", "w": "w_j", "T": "T_j"}
    for var_name in cfg.get("forcing_zero_vars", []):
        B_diag[idx[var_map[var_name]]] = 0.0
    B_diag[bc_indices] = 0.0
    B_mat_sp = sp.diags(B_diag).tocsc()
    B_mat = scipy_to_petsc(B_mat_sp, solver_cfg) if hasattr(L0, "getInfo") else B_mat_sp
    C_mat = B_mat.copy()

    # Paraboloid at ω=0 can be singular depending on discretization and
    # boundary conditions; a small imaginary shift is a conservative fallback
    # to keep this example numerically stable in reduced-memory settings.
    solve_freqs = [complex(omega)]
    if use_shift_fallback and abs(omega) < 1e-12 and omega_shift_fallback > 0.0:
        solve_freqs.append(1j * omega_shift_fallback)
    last_error: Exception | None = None
    gains: np.ndarray
    responses: np.ndarray
    forcings: np.ndarray
    omega_used = solve_freqs[0]

    for candidate in solve_freqs:
        try:
            (
                gains,
                responses,
                forcings,
                svd_residuals_fwd,
                svd_residuals_adj,
            ) = resolvent_analysis(
                L0,
                candidate,
                n_modes,
                W_diag,
                B_mat=B_mat,
                C_mat=C_mat,
                filter_fn=mesh.filters.filter if mesh.filters is not None else None,
                filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
                solver_cfg=solver_cfg,
                residuals=True,
            )
            omega_used = candidate
            if candidate != solve_freqs[0]:
                print(
                    f"Using fallback frequency ω={candidate:.3g} to avoid singular/ill-conditioned solve at ω={solve_freqs[0]:.3g}."
                )
            break
        except RuntimeError as exc:
            last_error = exc
            if candidate == solve_freqs[0] and len(solve_freqs) > 1:
                print(
                    "Paraboloid resolvent failed at ω=0; retrying with small imaginary"
                    f" shift ω={solve_freqs[1]:.3g}."
                )
                continue
            raise

    if "gains" not in locals():
        raise RuntimeError(
            "Paraboloid resolvent failed for all frequency attempts."
        ) from last_error

    np.savez_compressed(
        OUTPUT_NPZ,
        gains=gains,
        responses=responses,
        forcings=forcings,
        svd_residuals_fwd=svd_residuals_fwd,
        svd_residuals_adj=svd_residuals_adj,
        idx_rho_j=idx["rho_j"],
        idx_u_j=idx["u_j"],
        idx_v_j=idx["v_j"],
        idx_w_j=idx["w_j"],
        idx_T_j=idx["T_j"],
        used_ind=used_ind,
        omega=omega,
        omega_used_real=float(np.real(omega_used)),
        omega_used_imag=float(np.imag(omega_used)),
        m=m,
        X=mesh.X,
        Y=mesh.Y,
        n_modes=n_modes,
        baseflow_u=baseflow.U,
        baseflow_v=baseflow.V,
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def _log(line: str) -> None:
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    cfg = load_config(SCRIPT.with_suffix(".yaml"))
    t0 = time.perf_counter()
    load_or_compute(OUTPUT_NPZ, lambda: compute(cfg))
    _log(f"example_5_paraboloid  {time.perf_counter() - t0:.3f} s")


if __name__ == "__main__":
    main()
