#!/usr/bin/env python3
"""
Example 2: Jet resolvent analysis.

Translates example_2_Jet.m for an axisymmetric (m=0) jet base flow.

OUTPUTS: example_2_jet.npz, example_2_jet.png
USAGE:   python example_2_jet.py                       # .npz exists → plot only
         rm example_2_jet.npz && python example_2_jet.py  # recompute
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from linstab2d.baseflow import load_jet, sutherland_air
from linstab2d.boundary_conditions.dirichlet import apply_dirichlet
from linstab2d.mesh import Mesh
from linstab2d.operators.assembly import build_lnse_operator
from linstab2d.operators.energy_norm import compute_chu_norm
from linstab2d.solvers.resolvent import resolvent_analysis
from linstab2d.utils.plotting import plot_flow

from ._helpers import env_int

SCRIPT = Path(__file__).resolve()
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")
REFERENCE_MAT = SCRIPT.with_name("example_2_reference.mat")


from linstab2d.baseflow.loaders import _BASEFLOW_DATA_DIR

_BASEFLOW_MAT = _BASEFLOW_DATA_DIR / "M09jet.mat"


def _as_grid(values: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel(order="F")
    if arr.size != np.prod(shape):
        raise ValueError(f"{name} has size {arr.size}, expected {shape[0] * shape[1]}")
    return arr.reshape(shape, order="F")


def _reference_or_warn(label: str, actual: np.ndarray, key: str) -> None:
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
    mat = loadmat(_BASEFLOW_MAT)
    source_shape = (int(r_1D.size), int(z_1D.size))
    sponge = _as_grid(mat["spongeFun"], source_shape, "spongeFun")
    if tuple(mesh.X.shape) == source_shape:
        return sponge

    # Mesh grid is built as np.meshgrid(z_1D, r_1D, indexing="xy"), so axis 0 is r and axis 1 is z.
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


def _dominant_component(fields: dict[str, np.ndarray]) -> str:
    norms = {name: np.linalg.norm(np.abs(field)) for name, field in fields.items()}
    return max(norms, key=norms.get)


def compute():
    """Run jet resolvent analysis."""
    St = 0.39
    omega = 2 * np.pi * St
    coarsen = env_int("LINSTAB2D_JET_COARSEN", 1)
    coarsen = max(1, coarsen)
    n_modes = 3
    m = 0
    fd_order = 4

    data = loadmat(_BASEFLOW_MAT)
    r_1D = np.asarray(data["r_1D"], dtype=float).ravel()
    z_1D = np.asarray(data["z_1D"], dtype=float).ravel()
    nr = int(r_1D[::coarsen].size)
    nz = int(z_1D[::coarsen].size)
    print(f"Jet mesh grid points: nz={nz}, nr={nr}, coarsen={coarsen}")
    r_1D_coarse = r_1D[::coarsen]
    z_1D_coarse = z_1D[::coarsen]

    xrange = (0.0, 1.0)
    yrange = (0.0, 1.0)

    cmesh = Mesh.create(
        xrange,
        yrange,
        nz,
        nr,
        fd_order,
        y_symmetry=True,
        x_periodicity=False,
        alpha_filter=0.0,
    )
    X = np.meshgrid(z_1D_coarse, r_1D_coarse, indexing="xy")[0]
    Y = np.meshgrid(z_1D_coarse, r_1D_coarse, indexing="xy")[1]
    mesh = cmesh.deform(X_new=X, Y_new=Y)

    baseflow = load_jet(mesh)
    sutherland_air(baseflow)
    _print_velocity_scales(
        baseflow.U,
        baseflow.V,
        baseflow.W,
    )
    mesh.sponge = _load_sponge(mesh, r_1D, z_1D)

    t_lnse = time.perf_counter()
    L0, idx = build_lnse_operator(mesh, baseflow, model="axy", m=m)
    t_lnse = time.perf_counter() - t_lnse

    borders = "ltr"
    vars_ = "uvwT"
    L0, _ = apply_dirichlet(L0, idx, borders, vars_, mesh.ngp)

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
        model="axy",
        R=R,
    )

    sponge0 = (mesh.sponge.ravel(order="F")[used_ind] == 0.0).astype(float)
    # Match the MATLAB example literally: kron(B(:), ones(5,1)).
    # This repeats each grid-point mask entry across the flattened 5-field
    # vector in the same order as the reference workflow.
    B_diag = np.repeat(sponge0, 5)
    B_mat = sp.diags(B_diag).tocsc()

    t_resolvent = time.perf_counter()
    gains, responses, forcings = resolvent_analysis(
        L0,
        omega,
        n_modes,
        W_diag,
        B_mat=B_mat,
        C_mat=B_mat,
        filter_fn=mesh.filters.filter if mesh.filters is not None else None,
        filter_ct_fn=mesh.filters.filter_ct if mesh.filters is not None else None,
    )
    t_resolvent = time.perf_counter() - t_resolvent
    print(f"Resolvent gains: {gains}")
    print(f"Stage 1C timing: build_lnse_operator={t_lnse:.2f}s, resolvent_analysis={t_resolvent:.2f}s")

    if REFERENCE_MAT.exists():
        _reference_or_warn("resolvent gains", gains, "gains")

    np.savez_compressed(
        OUTPUT_NPZ,
        gains=gains,
        responses=responses,
        forcings=forcings,
        omega=omega,
        n_modes=n_modes,
        X=mesh.X,
        Y=mesh.Y,
        idx_rho_j=idx["rho_j"],
        idx_u_j=idx["u_j"],
        idx_v_j=idx["v_j"],
        idx_w_j=idx["w_j"],
        idx_T_j=idx["T_j"],
        ngp=mesh.ngp,
    )
    print(f"Results saved to {OUTPUT_NPZ}")


def _plot_field(ax, x, y, field, title: str) -> None:
    contour = plot_flow(x, y, field, ax=ax, levels=51, cmap="RdBu_r")
    ax.set_title(title)


def _load_index_vector(
    data: np.lib.npyio.NpzFile,
    key: str,
    ngp: int,
    start: int,
) -> np.ndarray:
    if key in data:
        idx = np.asarray(data[key], dtype=int)
    else:
        idx = np.arange(start, start + ngp, dtype=int)
    return idx


def _read_field_components(
    field_data: np.ndarray,
    idx: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    return np.asarray(field_data[idx, :], dtype=complex).reshape((shape[0], shape[1], -1), order="F")


def plot(data: np.lib.npyio.NpzFile) -> None:
    """Plot forcing and response dominant components for the computed modes."""
    import matplotlib.pyplot as plt

    gains = data["gains"]
    responses = data["responses"]
    forcings = data["forcings"]
    X = data["X"]
    Y = data["Y"]
    shape = X.shape
    ngp = int(data["ngp"])
    idx_rho = _load_index_vector(data, "idx_rho_j", ngp, 0)
    idx_u = _load_index_vector(data, "idx_u_j", ngp, ngp)
    idx_v = _load_index_vector(data, "idx_v_j", ngp, 2 * ngp)
    idx_w = _load_index_vector(data, "idx_w_j", ngp, 3 * ngp)
    idx_T = _load_index_vector(data, "idx_T_j", ngp, 4 * ngp)

    f_rho = _read_field_components(forcings, idx_rho, shape)
    f_u = _read_field_components(forcings, idx_u, shape)
    f_v = _read_field_components(forcings, idx_v, shape)
    f_w = _read_field_components(forcings, idx_w, shape)
    f_T = _read_field_components(forcings, idx_T, shape)

    r_rho = _read_field_components(responses, idx_rho, shape)
    r_u = _read_field_components(responses, idx_u, shape)
    r_v = _read_field_components(responses, idx_v, shape)
    r_w = _read_field_components(responses, idx_w, shape)
    r_T = _read_field_components(responses, idx_T, shape)

    n_modes = min(int(data["n_modes"]), len(gains))
    fig, axes = plt.subplots(max(1, n_modes), 2, figsize=(14, 6 * max(1, n_modes)))
    if n_modes == 1:
        axes = np.array([axes])

    for i in range(n_modes):
        forcing_fields = {
            "rho": f_rho,
            "u": f_u,
            "v": f_v,
            "w": f_w,
            "T": f_T,
        }
        response_fields = {
            "rho": r_rho,
            "u": r_u,
            "v": r_v,
            "w": r_w,
            "T": r_T,
        }
        f_name = _dominant_component(forcing_fields)
        r_name = _dominant_component(response_fields)

        _plot_field(
            axes[i, 0],
            X,
            Y,
            np.real(forcing_fields[f_name][:, :, i]),
            f"f_{f_name}^({i + 1}), sigma = {gains[i]:.2f}",
        )
        _plot_field(
            axes[i, 1],
            X,
            Y,
            np.real(response_fields[r_name][:, :, i]),
            f"{r_name}^({i + 1})",
        )

    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("x")
    fig.suptitle("Jet Resolvent Analysis (m=0, axy)")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {OUTPUT_PNG}")


def main():
    try:
        data = np.load(OUTPUT_NPZ)
    except (FileNotFoundError, EOFError, OSError, ValueError) as exc:
        print(f"Cached results unavailable or invalid ({exc}), recomputing.")
        compute()
        data = np.load(OUTPUT_NPZ)

    plot(data)


if __name__ == "__main__":
    main()
