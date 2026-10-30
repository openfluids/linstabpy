#!/usr/bin/env python3
"""
Example 4: Cavity global/adjoint stability analysis.

Translates example_4_Cavity.m using a masked cavity geometry, deformed mesh,
and direct + adjoint temporal eigenvalue problems.

OUTPUTS.
  - example_4_cavity.npz.
  - example_4_cavity.log.

Martini parity settings:
  - ``Pr: 0.7`` (MATLAB driver; ``Step_BF.mat`` stores 0.71).
  - ``zero_dmudT: true`` (LinStab2D ``readbaseflow`` uses ``dMUdT=0``).

USAGE:   python example_4_cavity.py
         rm example_4_cavity.npz && python example_4_cavity.py
         python example_4_plots.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from examples._helpers import load_or_compute
from linstabpy.utils.yaml_io import load_config, parse_complex

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
LOG_PATH = SCRIPT.with_suffix(".log")


def compute(cfg: dict) -> None:
    """Run cavity direct/adjoint temporal eigenproblems."""
    from examples.linstab2d.matlab_baseflow import load_cavity
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.operators.assembly import build_lnse_system_result
    from linstabpy.operators.energy_norm import compute_chu_norm
    from linstabpy.stability import adjoint_eigenvalues
    from linstabpy.stability import eigenvalues as temporal_eigenvalues

    solver_cfg = cfg.get("solver")
    if solver_cfg is None:
        solver_cfg = {}
    assembly_backend = cfg.get("assembly_backend", "scipy")
    if assembly_backend != "scipy":
        raise NotImplementedError(
            "example_4_cavity currently supports only assembly_backend='scipy' "
            "because the adjoint workflow remains SciPy-based"
        )
    bc_penalty = cfg["bc"].get("penalty", 1e4)

    omega_target = parse_complex(cfg["omega_target"])
    n_modes = cfg["n_modes"]
    n_plot = min(3, n_modes)
    beta = cfg["beta"]
    arnoldi_ncv = cfg["ncv"]
    zero_dmudT = bool(cfg.get("zero_dmudT", False))
    pr_value = float(cfg.get("Pr", 0.7))

    nx = cfg["nx"]
    ny = cfg["ny"]
    fd_order = cfg["fd_order"]

    x_cavity_start = cfg["x_cavity_start"]
    L_cavity = cfg["L_cavity"]
    xrange = (
        x_cavity_start - L_cavity,
        x_cavity_start + 2 * L_cavity,
    )
    yrange = (-1.0, 1.0)

    cmesh = Mesh.create(
        xrange,
        yrange,
        nx,
        ny,
        fd_order,
        y_symmetry=cfg["y_symmetry"],
        x_periodicity=cfg["x_periodicity"],
        alpha_filter=cfg["alpha_filter"],
    )
    mask_x_left = cfg["mask_x_left"]
    mask_x_right = cfg["mask_x_right"]
    remove = (cmesh.Y < 0.0) & ((cmesh.X < mask_x_left) | (cmesh.X > mask_x_right))
    mesh = cmesh.mask(~remove)

    deform = cfg["deform"]
    Y = mesh.Y.copy()
    a = deform["y_cubic_a"]
    Y = (Y**3 + a * Y) / (1 + a)
    stretch_n = deform["stretch_n"]
    y_threshold = deform["y_threshold"]
    y_amplitude = deform["y_amplitude"]
    p = Y > y_threshold
    Y[p] = Y[p] + y_amplitude * ((Y[p] - y_threshold) / (np.max(Y[p]) - y_threshold)) ** stretch_n
    mesh = mesh.deform(Y_new=Y)

    x_threshold = deform["x_threshold"]
    x_amplitude = deform["x_amplitude"]
    X = mesh.X.copy()
    p = X > x_threshold
    X[p] = X[p] + x_amplitude * ((X[p] - x_threshold) / (np.max(X[p]) - x_threshold)) ** stretch_n
    mesh = mesh.deform(X_new=X)

    baseflow = load_cavity(mesh)
    baseflow.Pr = pr_value
    sutherland_air(baseflow)
    if zero_dmudT:
        baseflow.dmudT[:] = 0.0
        baseflow.d2mudT2[:] = 0.0

    sp_cfg = cfg["sponge"]
    xs_trans = sp_cfg["xs_trans"]
    xs_offset = sp_cfg["xs_offset"]
    ys_trans = sp_cfg["ys_trans"]
    ys_offset = sp_cfg["ys_offset"]
    sponge_amp = sp_cfg["amplitude"]
    sponge = np.zeros(mesh.X.shape, dtype=float)
    ind = mesh.used_ind
    x = mesh.X.ravel(order="F")[ind]
    y = mesh.Y.ravel(order="F")[ind]
    x_min = float(np.min(mesh.X))
    x_max = float(np.max(mesh.X))
    y_max = float(np.max(mesh.Y))
    term_x = np.tanh(-(x - x_min - xs_offset) / xs_trans) + 1
    term_y = np.tanh((x - x_max + xs_offset) / xs_trans) + 1
    term_t = np.tanh((y - y_max + ys_offset) / ys_trans) + 1
    sponge_vals = sponge_amp / 2 * np.maximum(term_t, term_x + term_y)
    sponge_flat = sponge.ravel(order="F")
    sponge_flat[ind] = sponge_vals
    sponge = sponge_flat.reshape(mesh.X.shape, order="F")
    mesh.sponge = sponge

    lnse_system = build_lnse_system_result(
        mesh,
        baseflow,
        model=cfg.get("model", "2D"),
        kx=beta,
        bc_recipe=cfg["bc"]["recipe"],
        bc_penalty=bc_penalty,
        backend=assembly_backend,
        solver_cfg=solver_cfg,
    )
    L0 = lnse_system.operator
    idx = lnse_system.idx
    bc_indices = lnse_system.bc_indices

    p = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[p]
    RHO = baseflow.RHO.ravel(order="F")[p]
    T = baseflow.T.ravel(order="F")[p]
    W_diag, invW_diag = compute_chu_norm(
        W_int,
        RHO,
        T,
        baseflow.kappa,
        baseflow.Ma,
        mesh.ngp,
        model="2D",
    )

    omega, direct_modes = temporal_eigenvalues(
        L0,
        n_modes,
        omega_target,
        W_diag=W_diag,
        ncv=arnoldi_ncv,
        solver_cfg=solver_cfg,
    )
    omega_adj_target = 1j * np.conj(omega_target)
    omega_adj, adjoint_modes = adjoint_eigenvalues(
        L0,
        n_modes,
        omega_adj_target,
        W_diag,
        invW_diag,
        ncv=arnoldi_ncv,
        solver_cfg=solver_cfg,
    )

    iL0 = 1j * L0
    direct_residuals = np.array(
        [
            np.linalg.norm(iL0 @ q - om * q) / (abs(om) * max(float(np.linalg.norm(q)), 1e-300))
            for om, q in zip(omega, direct_modes.T)
        ]
    )
    adjoint_residuals = np.array(
        [
            np.linalg.norm(L0.conj().T @ (W_diag * q) - om_adj * (W_diag * q))
            / (abs(om_adj) * max(float(np.linalg.norm(W_diag * q)), 1e-300))
            for om_adj, q in zip(omega_adj, adjoint_modes.T)
        ]
    )
    print("Direct eigenvalue residuals ||iL0 q - omega q|| / (|omega| ||q||):")
    for i, (om, res) in enumerate(zip(omega, direct_residuals)):
        print(f"  [{i}]  omega={om.real:.5f}{om.imag:+.5f}j   res={res:.3e}")
    print("Adjoint eigenvalue residuals ||L0^H (Wq) - omega_adj (Wq)|| / (|omega_adj| ||Wq||):")
    for i, (om, res) in enumerate(zip(omega_adj, adjoint_residuals)):
        print(f"  [{i}]  omega_adj={om.real:.5f}{om.imag:+.5f}j   res={res:.3e}")

    np.savez_compressed(
        OUTPUT_NPZ,
        omega=omega,
        omega_adj=omega_adj,
        direct_modes=direct_modes,
        adjoint_modes=adjoint_modes,
        direct_residuals=direct_residuals,
        adjoint_residuals=adjoint_residuals,
        X=mesh.X,
        Y=mesh.Y,
        idx_rho_j=idx["rho_j"],
        idx_u_j=idx["u_j"],
        idx_v_j=idx["v_j"],
        idx_w_j=idx["w_j"],
        idx_T_j=idx["T_j"],
        used_ind=mesh.used_ind,
        bc_indices=bc_indices,
        n_plot=n_plot,
        baseflow_u=baseflow.U,
        Pr=pr_value,
        zero_dmudT=zero_dmudT,
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
    _log(f"example_4_cavity  {time.perf_counter() - t0:.3f} s")


if __name__ == "__main__":
    main()
