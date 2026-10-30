#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""
Example 2: Jet resolvent analysis.

Translates example_2_Jet.m for an axisymmetric (m=0) jet base flow.

OUTPUTS.
  - example_2_jet.npz.
  - example_2_jet.log.

USAGE:   python example_2_jet.py
         rm example_2_jet.npz && python example_2_jet.py
         python example_2_plots.py
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
REFERENCE_MAT = SCRIPT.with_name("example_2_reference.mat")


def _as_grid(values: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel(order="F")
    if arr.size != np.prod(shape):
        raise ValueError(f"{name} has size {arr.size}, expected {shape[0] * shape[1]}")
    return arr.reshape(shape, order="F")


def _reference_or_warn(label: str, actual: np.ndarray, key: str) -> None:
    from scipy.io import loadmat

    if not REFERENCE_MAT.exists():
        return
    try:
        reference = loadmat(REFERENCE_MAT)
    except Exception as exc:
        print(f"Reference comparison skipped ({REFERENCE_MAT.name} load failed): {exc}")
        return
    if key not in reference:
        print(f"Reference comparison skipped ({REFERENCE_MAT.name} has no '{key}' field)")
        return

    expected = np.asarray(reference[key]).reshape(-1)
    try:
        np.testing.assert_allclose(
            actual,
            expected[: actual.size],
            rtol=1e-6,
            atol=1e-6,
        )
    except AssertionError as exc:
        raise RuntimeError(f"MATLAB reference comparison failed for {label}") from exc
    print(f"Reference comparison passed for {label} ({REFERENCE_MAT.name})")


def _load_sponge(mesh, r_1D: np.ndarray, z_1D: np.ndarray) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator
    from scipy.io import loadmat

    from examples.linstab2d.matlab_baseflow import _BASEFLOW_DATA_DIR

    mat = loadmat(_BASEFLOW_DATA_DIR / "M09jet.mat")
    source_shape = (int(r_1D.size), int(z_1D.size))
    sponge = _as_grid(mat["spongeFun"], source_shape, "spongeFun")
    if tuple(mesh.X.shape) == source_shape:
        return sponge

    interpolator = RegularGridInterpolator(
        (r_1D.astype(float), z_1D.astype(float)),
        sponge,
        bounds_error=False,
        fill_value=np.nan,
    )
    query = np.column_stack(
        (
            mesh.Y.ravel(order="F").astype(float),
            mesh.X.ravel(order="F").astype(float),
        )
    )
    return interpolator(query).reshape(mesh.X.shape, order="F")


def _print_velocity_scales(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> None:
    print(
        "Jet velocity scales: "
        f"|U|_max={np.max(np.abs(U)):.3e}, "
        f"|V|_max={np.max(np.abs(V)):.3e}, "
        f"|W|_max={np.max(np.abs(W)):.3e}"
    )


def _bc_mask_layout(cfg: dict) -> str:
    return str(cfg.get("bc_mask_layout", "kron")).strip().lower()


def _build_bc_diag(sponge0: np.ndarray, layout: str) -> np.ndarray:
    from examples.linstab2d.plot_compare_utils import jet_bc_mask_diag

    return jet_bc_mask_diag(sponge0, layout=layout)


def compute(cfg: dict):
    """Run jet resolvent analysis."""
    import scipy.sparse as sp
    from scipy.io import loadmat

    from examples.linstab2d.matlab_baseflow import _BASEFLOW_DATA_DIR, load_jet
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.operators.assembly import build_lnse_system_result
    from linstabpy.operators.energy_norm import compute_chu_norm
    from linstabpy.resolvent.analysis import resolvent_analysis
    from linstabpy.solvers.petsc_utils import scipy_to_petsc

    solver_cfg = cfg.get("solver")
    bc_layout = _bc_mask_layout(cfg)

    St = cfg["St"]
    omega = 2 * np.pi * St
    coarsen = max(1, cfg["coarsen"])
    n_modes = cfg["n_modes"]
    m = cfg["m"]
    fd_order = cfg["fd_order"]
    model = cfg["model"]

    data = loadmat(_BASEFLOW_DATA_DIR / "M09jet.mat")
    r_1D = np.asarray(data["r_1D"], dtype=float).ravel()
    z_1D = np.asarray(data["z_1D"], dtype=float).ravel()
    nr = int(r_1D[::coarsen].size)
    nz = int(z_1D[::coarsen].size)
    print(f"Jet mesh grid points: nz={nz}, nr={nr}, coarsen={coarsen}")
    r_1D_coarse = r_1D[::coarsen]
    z_1D_coarse = z_1D[::coarsen]

    cmesh = Mesh.create(
        (0.0, 1.0),
        (0.0, 1.0),
        nz,
        nr,
        fd_order,
        y_symmetry=cfg["y_symmetry"],
        x_periodicity=cfg["x_periodicity"],
        alpha_filter=cfg["alpha_filter"],
    )
    X = np.meshgrid(z_1D_coarse, r_1D_coarse, indexing="xy")[0]
    Y = np.meshgrid(z_1D_coarse, r_1D_coarse, indexing="xy")[1]
    mesh = cmesh.deform(X_new=X, Y_new=Y)

    baseflow = load_jet(mesh)
    sutherland_air(baseflow)
    _print_velocity_scales(baseflow.U, baseflow.V, baseflow.W)
    mesh.sponge = _load_sponge(mesh, r_1D, z_1D)

    t_lnse = time.perf_counter()
    lnse_system = build_lnse_system_result(
        mesh,
        baseflow,
        model=model,
        m=m,
        bc_recipe=cfg["bc"]["recipe"],
        bc_penalty=cfg["bc"].get("penalty", 1e4),
        backend=cfg.get("assembly_backend"),
        solver_cfg=solver_cfg,
        matlab_parity_dmudt=as_bool(cfg.get("matlab_parity_dmudt", False)),
    )
    L0 = lnse_system.operator
    idx = lnse_system.idx
    t_lnse = time.perf_counter() - t_lnse

    used_ind = mesh.used_ind
    W_int = mesh.W.ravel(order="F")[used_ind]
    rho_p = baseflow.RHO.ravel(order="F")[used_ind]
    T_p = baseflow.T.ravel(order="F")[used_ind]
    R = mesh.Y.ravel(order="F")[used_ind]
    W_diag, _ = compute_chu_norm(
        W_int,
        rho_p,
        T_p,
        baseflow.kappa,
        baseflow.Ma,
        mesh.ngp,
        model=model,
        R=R,
    )

    sponge0 = (mesh.sponge.ravel(order="F")[used_ind] == 0.0).astype(float)
    # Published MATLAB parity: kron(B(:), ones(5,1)) == np.repeat(sponge0, 5).
    # Block-correct repmat/tile is available via bc_mask_layout: tile in yaml.
    B_diag = _build_bc_diag(sponge0, bc_layout)
    B_mat_sp = sp.diags(B_diag).tocsc()
    B_mat = scipy_to_petsc(B_mat_sp, solver_cfg) if hasattr(L0, "getInfo") else B_mat_sp
    C_mat = B_mat.copy()

    t_resolvent = time.perf_counter()
    (
        gains,
        responses,
        forcings,
        svd_residuals_fwd,
        svd_residuals_adj,
    ) = resolvent_analysis(
        L0,
        omega,
        n_modes,
        W_diag,
        B_mat=B_mat,
        C_mat=C_mat,
        filter_fn=mesh.filters.filter if mesh.filters is not None else None,
        filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
        solver_cfg=solver_cfg,
        residuals=True,
    )
    t_resolvent = time.perf_counter() - t_resolvent
    print(f"Resolvent gains: {gains}")
    print(
        f"Stage 1C timing: build_lnse_operator={t_lnse:.2f}s, resolvent_analysis={t_resolvent:.2f}s"
    )

    if REFERENCE_MAT.exists():
        _reference_or_warn("resolvent gains", gains, "gains")

    np.savez_compressed(
        OUTPUT_NPZ,
        gains=gains,
        responses=responses,
        forcings=forcings,
        svd_residuals_fwd=svd_residuals_fwd,
        svd_residuals_adj=svd_residuals_adj,
        omega=omega,
        n_modes=n_modes,
        X=mesh.X,
        Y=mesh.Y,
        used_ind=used_ind,
        idx_rho_j=idx["rho_j"],
        idx_u_j=idx["u_j"],
        idx_v_j=idx["v_j"],
        idx_w_j=idx["w_j"],
        idx_T_j=idx["T_j"],
        ngp=mesh.ngp,
        baseflow_u=baseflow.U,
        bc_mask_layout=bc_layout,
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def _log(line: str) -> None:
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main():
    cfg = load_config(SCRIPT.with_suffix(".yaml"))
    t0 = time.perf_counter()
    load_or_compute(OUTPUT_NPZ, lambda: compute(cfg))
    _log(f"example_2_jet  {time.perf_counter() - t0:.3f} s")


if __name__ == "__main__":
    main()
