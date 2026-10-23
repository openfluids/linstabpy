# SPDX-License-Identifier: GPL-3.0-only
"""Odd-even decoupling of the leading resolvent response on a wavy-wall mesh.

We compute the leading resolvent gain of a boundary-layer-like base flow on
two meshes of the same box. One mesh is Cartesian. The other mesh is
curvilinear, with a wavy lower wall. We repeat the solve at two resolutions
and for a ladder of operator-dissipation coefficients epsilon.

For each case we record the gain and the fraction of the energy of the u
component of the response at the grid scale (r_n). The other components are
not used. One (1/4, 1/2, 1/4) smoothing pass along each mesh direction removes
the grid-scale content; r_n is the removed energy fraction.

All lengths are nondimensional. We use the SciPy path only (no PETSc).

Run from the repository root::

    PYTHONPATH=src python -m examples.odd_even.odd_even

Output: ``odd_even.csv`` next to this file and ``output/odd_even.npz``.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from linstabpy.baseflow import BaseFlow, sutherland_air
from linstabpy.boundary_conditions import apply_bc_recipe
from linstabpy.mesh.mesh import Mesh
from linstabpy.operators import build_lnse_operator

LOG = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "odd_even.csv"
NPZ_PATH = HERE / "output" / "odd_even.npz"

# Box size and wall shape (nondimensional).
LX, LY = 4.0, 1.0
WALL_AMPLITUDE = 0.15
# Spanwise wavenumber and forcing frequency.
BETA = 1.0
OMEGA = 1.0
FD_ORDER = 4
# Outlet sponge: quadratic ramp over the last 15 % of the box, peak value 2.
SPONGE_FRACTION = 0.15
SPONGE_AMPLITUDE = 2.0

RESOLUTIONS = ((81, 41), (161, 81))
EPS_LADDER = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)

# Relative tolerance of the ARPACK singular-value iteration.
SVD_TOL = 1e-10

BC_RECIPE = [
    {"type": "dirichlet", "borders": "lt", "variables": "ruvwT"},
    {"type": "dirichlet", "borders": "b", "variables": "uvwT"},
    {"type": "neumann", "borders": "r", "variables": "ruvwT"},
]

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CaseResult:
    """One (mesh, resolution, epsilon) point of the ladder."""

    mesh_kind: str
    nx: int
    ny: int
    fd_order: int
    epsilon: float
    leak_into_dx: float
    gain_l2: float
    r_n: float
    nnz: int
    wall_factor_s: float
    wall_total_s: float


def make_mesh(nx: int, ny: int, epsilon: float, *, curvilinear: bool) -> Mesh:
    """Make the Cartesian mesh or the wavy-wall mesh, with the outlet sponge.

    Parameters
    ----------
    nx, ny : int
        Number of nodes along x and along y.
    epsilon : float
        Operator-dissipation coefficient. Zero turns the dissipation off.
    curvilinear : bool
        True gives the wavy lower wall. False gives the Cartesian mesh.

    Returns
    -------
    Mesh
        The mesh with its sponge field set.
    """
    dissipation = {"enabled": epsilon > 0.0, "epsilon": epsilon, "order_diss": FD_ORDER}
    mesh = Mesh.create(
        (0.0, LX), (0.0, LY), nx, ny, FD_ORDER, numerics_operator_dissipation=dissipation
    )
    if curvilinear:
        X, Y = mesh.X, mesh.Y
        # The wall displacement decays to zero at the flat top boundary.
        Y_new = Y + WALL_AMPLITUDE * np.sin(2.0 * np.pi * X / LX) * (1.0 - Y / LY) ** 2
        mesh = mesh.deform(Y_new=Y_new, X_new=X.copy())
    x0 = (1.0 - SPONGE_FRACTION) * LX
    ramp = np.clip((mesh.X - x0) / (LX - x0), 0.0, 1.0)
    mesh.sponge = SPONGE_AMPLITUDE * ramp**2
    return mesh


def make_baseflow(nx: int, ny: int) -> BaseFlow:
    """Make a boundary-layer-like base flow in computational coordinates.

    Parameters
    ----------
    nx, ny : int
        Number of nodes along x and along y.

    Returns
    -------
    BaseFlow
        Uniform density and temperature, U = 1 - exp(-(eta / 0.15)^2), V = W = 0.
    """
    eta = np.linspace(0.0, LY, ny)[:, None] * np.ones((1, nx))
    ones = np.ones_like(eta)
    zeros = np.zeros_like(eta)
    baseflow = BaseFlow(
        Re=1.0e4,
        Ma=0.5,
        Pr=0.7,
        kappa=1.4,
        T_0=288.15,
        RHO=ones.copy(),
        U=1.0 - np.exp(-((eta / 0.15) ** 2)),
        V=zeros.copy(),
        W=zeros.copy(),
        T=ones.copy(),
    )
    sutherland_air(baseflow)
    return baseflow


def grid_scale_leak(mesh: Mesh, nx: int, ny: int) -> float:
    """Measure how much of an eta-sawtooth goes into the x-derivative.

    Parameters
    ----------
    mesh : Mesh
        The mesh with its derivative matrices.
    nx, ny : int
        Number of nodes along x and along y.

    Returns
    -------
    float
        The ratio ||Dx s|| / ||Dy s|| for the field s = (-1)^j.
    """
    # Column-major node order: each block of ny entries is one line of constant x.
    saw_eta: FloatArray = np.tile((-1.0) ** np.arange(ny, dtype=np.float64), nx)
    return float(np.linalg.norm(mesh.Dx @ saw_eta) / np.linalg.norm(mesh.Dy @ saw_eta))


def grid_scale_fraction(u: ComplexArray) -> float:
    """Return the fraction of the energy of u at the grid scale.

    One pass of the (1/4, 1/2, 1/4) filter along each direction removes a field
    (-1)^i or (-1)^j. Smooth content changes by O((kh)^2). The filter acts on
    interior nodes only, so the boundary rows and columns add zero.

    Parameters
    ----------
    u : ndarray of complex, shape (ny, nx)
        The field. The result does not depend on its phase.

    Returns
    -------
    float
        ||u - S_eta S_xi u||^2 / ||u||^2, or 0 for a zero field.
    """
    total = float(np.sum(np.abs(u) ** 2))
    if total <= 0.0:
        return 0.0
    s = u.copy()
    s[:, 1:-1] = 0.25 * u[:, :-2] + 0.5 * u[:, 1:-1] + 0.25 * u[:, 2:]
    t = s.copy()
    t[1:-1, :] = 0.25 * s[:-2, :] + 0.5 * s[1:-1, :] + 0.25 * s[2:, :]
    return float(np.sum(np.abs(u - t) ** 2) / total)


def leading_resolvent(L: sp.csc_matrix, omega: float) -> tuple[float, ComplexArray, float]:
    """Compute the leading singular value of R = (-i omega I - L)^-1.

    Parameters
    ----------
    L : scipy.sparse.csc_matrix
        The linear operator with the boundary conditions applied.
    omega : float
        Forcing frequency.

    Returns
    -------
    gain : float
        The leading singular value of R in the L2 norm.
    response : ndarray of complex
        The leading left singular vector. Its largest entry is real and positive.
    factor_time : float
        Wall time of the LU factorization, in seconds.
    """
    n = L.shape[0]
    A = (
        (-1j * omega) * sp.identity(n, dtype=np.complex128, format="csc") - L.astype(np.complex128)
    ).tocsc()
    t0 = time.perf_counter()
    lu = spla.splu(A)
    factor_time = time.perf_counter() - t0

    def matvec(v: ComplexArray) -> ComplexArray:
        return lu.solve(v)

    def rmatvec(v: ComplexArray) -> ComplexArray:
        return lu.solve(v, trans="H")

    resolvent = spla.LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec, dtype=np.complex128)
    u, s, _ = spla.svds(resolvent, k=1, tol=SVD_TOL)
    response: ComplexArray = u[:, 0]
    k = int(np.argmax(np.abs(response)))
    response *= np.conj(response[k]) / np.abs(response[k])
    return float(s[0]), response, factor_time


def run_case(
    nx: int, ny: int, epsilon: float, *, curvilinear: bool
) -> tuple[CaseResult, ComplexArray, Mesh]:
    """Assemble, solve and measure one case.

    Parameters
    ----------
    nx, ny : int
        Number of nodes along x and along y.
    epsilon : float
        Operator-dissipation coefficient.
    curvilinear : bool
        True for the wavy-wall mesh.

    Returns
    -------
    result : CaseResult
        The row of the run record.
    u : ndarray of complex, shape (ny, nx)
        The streamwise velocity of the leading response.
    mesh : Mesh
        The mesh of the case.
    """
    t_start = time.perf_counter()
    mesh = make_mesh(nx, ny, epsilon, curvilinear=curvilinear)
    idx: dict[str, NDArray[np.intp]]
    L0, idx = build_lnse_operator(mesh, make_baseflow(nx, ny), kx=BETA, dissipation_epsilon=epsilon)
    L, _ = apply_bc_recipe(L0, mesh, idx, BC_RECIPE)
    gain, q, factor_time = leading_resolvent(L.tocsc(), OMEGA)
    u = q[idx["u_j"]].reshape((ny, nx), order="F")
    result = CaseResult(
        mesh_kind="curvilinear" if curvilinear else "rectilinear",
        nx=nx,
        ny=ny,
        fd_order=FD_ORDER,
        epsilon=epsilon,
        leak_into_dx=grid_scale_leak(mesh, nx, ny),
        gain_l2=gain,
        r_n=grid_scale_fraction(u),
        nnz=int(L.nnz),
        wall_factor_s=factor_time,
        wall_total_s=time.perf_counter() - t_start,
    )
    return result, u, mesh


def write_csv(results: list[CaseResult]) -> None:
    """Write the run record. Call it after each case, so a crash keeps the data.

    Parameters
    ----------
    results : list of CaseResult
        The rows computed so far.
    """
    names = [f.name for f in fields(CaseResult)]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


def main() -> int:
    """Run the resolution and epsilon ladder and write the CSV and the NPZ.

    Returns
    -------
    int
        Exit status 0.

    Raises
    ------
    FloatingPointError
        If a response or a gain is not finite.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results: list[CaseResult] = []
    arrays: dict[str, FloatArray | ComplexArray] = {}
    for nx, ny in RESOLUTIONS:
        for curvilinear in (False, True):
            tag = "curvilinear" if curvilinear else "rect"
            for epsilon in EPS_LADDER:
                result, u, mesh = run_case(nx, ny, epsilon, curvilinear=curvilinear)
                if not np.all(np.isfinite(u)) or not np.isfinite(result.gain_l2):
                    msg = f"non-finite response for {tag} {nx}x{ny} eps={epsilon:g}"
                    raise FloatingPointError(msg)
                results.append(result)
                write_csv(results)
                arrays[f"u_{tag}_{nx}x{ny}_{epsilon:g}"] = u
                arrays[f"X_{tag}_{nx}x{ny}"] = mesh.X
                arrays[f"Y_{tag}_{nx}x{ny}"] = mesh.Y
                LOG.info(
                    "%-11s %dx%d eps=%-6g gain=%.6e r_n=%.4f factor=%.1f s",
                    result.mesh_kind,
                    nx,
                    ny,
                    epsilon,
                    result.gain_l2,
                    result.r_n,
                    result.wall_factor_s,
                )
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ_PATH, allow_pickle=False, **arrays)
    LOG.info("wrote %s and %s", CSV_PATH, NPZ_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
