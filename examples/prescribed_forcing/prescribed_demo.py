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
    """
    logger = logging.getLogger(__name__)
    logger.info("prescribed_demo compute starting")

    # The chokepoint imports mpi4py.MPI before petsc4py and initializes
    # PETSc exactly once (utils/petsc_init.py). Never init MPI directly.
    from linstabpy.utils.petsc_init import init_petsc_from_argv

    init_petsc_from_argv()

    from examples.martini.matlab_baseflow import load_blasius
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

    logger.info("Manufacturing forcing field")
    # The solver flattens each field block in column-major order and then
    # keeps the points in mesh.used_ind (operators/assembly.py). Build the
    # bump on the same point sequence so it lands where the comment says.
    x_pts = np.asarray(mesh.X).ravel(order="F")[mesh.used_ind]
    y_pts = np.asarray(mesh.Y).ravel(order="F")[mesh.used_ind]
    f_hat = _manufacture_forcing_field(x_pts, y_pts)
    np.savez(OUTPUT_NPZ, f=f_hat, beta=0.0, omega=60.0)
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
        "prescribed_forcing": [
            {
                "beta": 0.0,
                "omega": 60.0,
                "forcing_path": str(OUTPUT_NPZ),
            },
        ],
        "prescribed_output_csv": str(OUTPUT_CSV),
        "forcing_norm": "l2",
    }

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
            f"{gain:.6g}" if gain is not None else "FAILED",
            status,
        )


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
    load_or_compute(OUTPUT_NPZ, compute, required_keys={"f", "beta", "omega"})
    elapsed = time.perf_counter() - t0
    logger.info("prescribed_demo complete (%.3f s)", elapsed)


if __name__ == "__main__":
    main()
