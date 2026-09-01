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


def _manufacture_forcing_field(ny: int, nx: int) -> np.ndarray:
    """Create a synthetic localized forcing field in all five state fields.

    Returns a complex vector with a Gaussian bump at the domain center.
    The forcing is unit-normalized before returning.

    Parameters
    ----------
    ny, nx : int
        Grid dimensions.

    Returns
    -------
    f_hat : ndarray
        Complex forcing vector, shape (5*ny*nx,).
    """
    ndof = 5 * ny * nx
    f_hat = np.zeros(ndof, dtype=complex)

    x_min, x_max = 0.0, 1.0
    y_min, y_max = -1.0, 1.0
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)

    for j in range(ny):
        for i in range(nx):
            xi = x_min + i * dx
            yi = y_min + j * dy
            dist = np.sqrt((xi - x_center) ** 2 + (yi - y_center) ** 2)
            weight = np.exp(-(dist**2) / 0.01)

            flat_idx = j * nx + i
            for field_offset in range(5):
                f_hat[field_offset * ny * nx + flat_idx] = weight * (1.0 + 0.1j)

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

    logger.info("Loading Blasius base flow")
    Re = 420.0
    Ma = 0.3
    Pr = 0.72
    kappa = 0.0
    T_0 = 1.0

    nx = 40
    ny = 25
    xrange = (0.0, 1.0)
    yrange = (-1.0, 1.0)
    fd_order = 4

    logger.info("Building mesh: nx=%d ny=%d", nx, ny)
    cmesh = Mesh.create(xrange, yrange, nx, ny, fd_order, alpha_filter="none")

    Y = cmesh.Y.copy()
    y_i = 0.2
    y_max = 6.0
    a = y_i * y_max / (y_max - 2 * y_i)
    b = 1 + 2 * a / y_max
    Y_phys = a * (1 + Y) / (b - Y)
    mesh = cmesh.deform(Y_new=Y_phys)

    logger.info("Loading Blasius base flow")
    bf = load_blasius(mesh.X, Y_phys, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa, T_0=T_0)
    sutherland_air(bf)

    logger.info("Manufacturing forcing field")
    f_hat = _manufacture_forcing_field(ny, nx)
    np.savez(OUTPUT_NPZ, f=f_hat, beta=0.0, omega=0.5)
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
                "omega": 0.5,
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
