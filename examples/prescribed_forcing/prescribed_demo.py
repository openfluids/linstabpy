#!/usr/bin/env python3
"""
Example: prescribed forcing resolvent on Blasius.

Manufactures a localized forcing field, saves it to NPZ, loads it into the
prescribed_map solver, and reports the gain.

OUTPUTS:
  - prescribed_demo_forcing.npz (the manufactured forcing field).
  - prescribed_demo_gains.csv (one row per solved point).
  - prescribed_demo.log (solver output).

USAGE:   python prescribed_demo.py
         rm prescribed_demo_forcing.npz && python prescribed_demo.py
"""

from __future__ import annotations

import logging
import time
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from examples._helpers import load_or_compute

SCRIPT = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT.parent
OUTPUT_NPZ = SCRIPT_DIR / "prescribed_demo_forcing.npz"
OUTPUT_CSV = SCRIPT_DIR / "prescribed_demo_gains.csv"
OUTPUT_LOG = SCRIPT_DIR / "prescribed_demo.log"

# The single demo point. The forcing npz and the config row must carry
# the same pair, so both read these constants.
BETA = 0.0
OMEGA = 60.0


def _setup_logging() -> logging.Logger:
    """Configure logging to both file and console.

    Returns
    -------
    logging.Logger
        The module logger.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_LOG, mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def _manufacture_forcing_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Create a synthetic localized forcing field in all five state fields.

    The state vector holds five field blocks. Each block lists the grid
    points in the solver's column-major order, which *x* and *y* already
    follow. The same Gaussian bump goes into every block, centered in the
    streamwise direction and near the boundary layer. The result is
    unit-normalized.

    Parameters
    ----------
    x, y : ndarray
        Physical grid-point coordinates in the solver's grid ordering,
        one entry per point.

    Returns
    -------
    f_hat : ndarray
        Complex forcing vector, shape (5 * x.size,).
    """
    x_min = float(x.min())
    x_max = float(x.max())
    x_c = 0.5 * (x_min + x_max)
    # 0.01 is the mapping parameter y_i of example 1: half of the
    # wall-normal points sit below this height, so the grid resolves
    # the bump.
    y_c = 0.01
    s_x = 0.1 * (x_max - x_min)
    s_y = 0.01

    weight = np.exp(-(((x - x_c) / s_x) ** 2 + ((y - y_c) / s_y) ** 2))
    f_hat = np.tile(weight.astype(complex), 5)

    norm = np.linalg.norm(f_hat)
    if norm > 0.0:
        f_hat /= norm

    return f_hat


def compute() -> None:
    """Run the prescribed-forcing example.

    Loads Blasius base flow, manufactures a forcing field,
    saves it to npz, and calls prescribed_map to solve the resolvent.

    Raises
    ------
    RuntimeError
        If more than one MPI rank runs the demo.
    """
    logger = logging.getLogger(__name__)
    logger.info("prescribed_demo compute starting")

    # The chokepoint imports mpi4py.MPI before petsc4py and initializes
    # PETSc exactly once (utils/petsc_init.py). Never init MPI directly.
    from linstabpy.utils.petsc_init import init_petsc_from_argv

    init_petsc_from_argv()

    from petsc4py import PETSc

    # prescribed_map is serial. Under mpiexec every rank would write the
    # same npz and append to the same CSV, so refuse more than one rank.
    if PETSc.COMM_WORLD.getSize() > 1:
        message = "prescribed_demo is serial; run it without mpiexec"
        raise RuntimeError(message)

    from examples.linstab2d.matlab_baseflow import load_blasius
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.solvers.router.prescribed_map import prescribed_map

    # Physical parameters follow example_1_blasius.yaml. kappa is the
    # ratio of specific heats: sutherland_air divides by kappa*(kappa-1),
    # and T_0 [K] makes the Sutherland temperature dimensional.
    Re = 6.0e5
    Ma = 0.3
    Pr = 0.7
    kappa = 1.4
    T_0 = 273.15

    # Domain and mapping also follow example 1: x starts away from the
    # leading edge because the Blasius solution is singular at x = 0.
    nx = 40
    ny = 25
    dx = 0.01
    xrange = (5 * dx, 1.25 + dx)
    yrange = (-1.0, 1.0)
    fd_order = 4

    logger.info("Building mesh: nx=%d ny=%d", nx, ny)
    # alpha_filter="none" disables the stencil filter. Example 1 instead
    # runs alpha_filter 0.0, which builds the maximum-strength filter
    # (the MATLAB convention; see mesh/filters.py).
    cmesh = Mesh.create(xrange, yrange, nx, ny, fd_order, alpha_filter="none")

    Y = cmesh.Y.copy()
    y_i = 0.01
    y_max = 0.05
    a = y_i * y_max / (y_max - 2 * y_i)
    b = 1 + 2 * a / y_max
    Y_phys = a * (1 + Y) / (b - Y)
    mesh = cmesh.deform(Y_new=Y_phys)

    logger.info("Loading Blasius base flow")
    bf = load_blasius(mesh.X, Y_phys, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0)
    sutherland_air(bf)

    # Sponge follows example 1: an absorbing layer at the outlet and the
    # top boundary keeps reflections out of the response.
    sponge_amp = 5.0
    xs_trans = 0.02
    xs_offset = 0.02
    ys_trans = 0.01
    ys_offset = 0.01
    x_grid = mesh.X
    y_grid = mesh.Y
    mesh.sponge = (
        sponge_amp
        / 2
        * np.maximum(
            np.tanh((y_grid - y_grid.max() + ys_offset) / ys_trans) + 1,
            np.tanh(-(x_grid - x_grid.min() - xs_offset) / xs_trans)
            + 1
            + np.tanh((x_grid - x_grid.max() + xs_offset) / xs_trans)
            + 1,
        )
    )

    logger.info("Manufacturing forcing field")
    # The solver flattens each field block in column-major order and then
    # keeps the points in mesh.used_ind (operators/assembly.py). Build the
    # bump on the same point sequence so it lands where the comment says.
    x_pts = np.asarray(mesh.X).ravel(order="F")[mesh.used_ind]
    y_pts = np.asarray(mesh.Y).ravel(order="F")[mesh.used_ind]
    f_hat = _manufacture_forcing_field(x_pts, y_pts)
    np.savez(OUTPUT_NPZ, f=f_hat, beta=BETA, omega=OMEGA)
    logger.info("Saved forcing to %s", OUTPUT_NPZ)

    cfg = {
        "mesh": mesh,
        "Re": Re,
        "Ma": Ma,
        "Pr": Pr,
        "kappa": kappa,
        "T_0": T_0,
        "fd_order": fd_order,
        "model": "2D",
        # Dirichlet walls on all four borders follow example 1. Without a
        # recipe the assembly applies no boundary conditions at all.
        "bc_recipe": [{"type": "dirichlet", "borders": "lrbt", "variables": "uvwT"}],
        "bc_penalty": 1.0e4,
        "prescribed_forcing": [
            {
                "beta": BETA,
                "omega": OMEGA,
                "forcing_path": str(OUTPUT_NPZ),
            },
        ],
        "prescribed_output_csv": str(OUTPUT_CSV),
        "forcing_norm": "l2",
    }

    # The CSV writer appends rows. Remove a previous file so a rerun
    # cannot keep old rows next to the new one.
    OUTPUT_CSV.unlink(missing_ok=True)

    logger.info("Running prescribed_map")
    t0 = time.perf_counter()
    result = prescribed_map(bf, cfg=cfg)
    elapsed = time.perf_counter() - t0
    logger.info("prescribed_map complete: %d points in %.3f s", len(result.point_records), elapsed)

    for record in result.point_records:
        status = record.get("status")
        gain = record.get("gain")
        logger.info(
            "  row=%d beta=%.4g omega=%.4g gain=%s status=%s",
            record.get("row"),
            record.get("beta"),
            record.get("omega"),
            f"{gain:.6g}" if gain is not None else "none",
            status,
        )

    # Add the gain to the npz only after a successful solve. main() requires
    # the key, so a crash between the first savez and here cannot leave a
    # cache that reads as complete. A failed point stores gain as None, so
    # the float check is the success gate.
    records = result.point_records
    gain_value = records[0].get("gain") if records else None
    if isinstance(gain_value, float):
        np.savez(OUTPUT_NPZ, f=f_hat, beta=BETA, omega=OMEGA, gain=gain_value)


def main() -> None:
    """Check for PETSc, then load or compute and cache results."""
    logger = _setup_logging()
    logger.info("prescribed_demo starting")

    # Availability check only; the real init happens in compute() through
    # the petsc_init chokepoint. find_spec has no import side effects.
    petsc_available = find_spec("petsc4py") is not None and find_spec("mpi4py") is not None

    if not petsc_available:
        logger.info("SKIPPED: requires petsc4py/mpi4py (not installed)")
        return

    t0 = time.perf_counter()
    load_or_compute(OUTPUT_NPZ, compute, required_keys={"f", "beta", "omega", "gain"})
    elapsed = time.perf_counter() - t0
    logger.info("prescribed_demo complete (%.3f s)", elapsed)


if __name__ == "__main__":
    main()
