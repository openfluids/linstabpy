#!/usr/bin/env python3
"""
Example: prescribed forcing resolvent on Blasius.

Manufactures a localized forcing field in the u-component, loads it into
the prescribed_forcing_response solver, and reports the gain.

OUTPUTS:
  - prescribed_demo_forcing.npz (the manufactured forcing field).
  - prescribed_demo_gains.csv (one row per solved point).
  - prescribed_demo.log (solver output).

USAGE:   python prescribed_demo.py
         python prescribed_demo_fig1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT.parent
OUTPUT_FORCING_NPZ = SCRIPT_DIR / "prescribed_demo_forcing.npz"
OUTPUT_CONFIG = SCRIPT_DIR / "prescribed_demo_config.yaml"
OUTPUT_LOG = SCRIPT_DIR / "prescribed_demo.log"


def manufacture_forcing(
    mesh,
    base_flow,
    cfg,
    beta=0.1,
    omega=0.5,
    ndof=None,
):
    """Create a synthetic localized forcing field in the u-component.

    Parameters
    ----------
    mesh
        Mesh object with x, y coordinates.
    base_flow
        Base-flow dict (unused here, but part of the API).
    cfg
        Config dict (unused).
    beta, omega
        Wavenumber and frequency for the npz file.
    ndof
        If provided, the forcing will be this size; otherwise inferred from mesh.

    Returns
    -------
    f_hat : ndarray
        Complex forcing vector, shape (ndof,).
    """
    from linstabpy.operators.assembly import build_lnse_system_result

    # Build the operator to determine ndof and the DOF layout.
    result = build_lnse_system_result(mesh, base_flow, cfg, beta=beta)
    L0 = result.L0
    idx = result.idx
    ndof_actual = L0.shape[0]
    if ndof is None:
        ndof = ndof_actual

    f_hat = np.zeros(ndof, dtype=complex)

    # Insert a Gaussian bump in the u-component at the center of the domain.
    if hasattr(idx, "u"):
        u_indices = idx.u
        x_center = (mesh.x.min() + mesh.x.max()) / 2.0
        y_center = (mesh.y.min() + mesh.y.max()) / 2.0
        for i, (xi, yi) in enumerate(zip(mesh.x.ravel(), mesh.y.ravel())):
            dist = np.sqrt((xi - x_center) ** 2 + (yi - y_center) ** 2)
            weight = np.exp(-(dist**2) / 0.01)
            if i < len(u_indices):
                f_hat[u_indices[i]] = weight * (1.0 + 0.1j)

    # Normalize to unit magnitude.
    norm = np.linalg.norm(f_hat)
    if norm > 0:
        f_hat = f_hat / norm

    return f_hat


def generate_config_yaml(forcing_npz_path, output_csv_path):
    """Generate a YAML config for the prescribed_forcing demo.

    Parameters
    ----------
    forcing_npz_path : Path
        Path to the manufactured forcing npz file.
    output_csv_path : Path
        Path where the CSV output should go.

    Returns
    -------
    str
        YAML config text.
    """
    yaml_text = f"""
# Prescribed-forcing demo config (generated)
schema_version: 1

physics:
  Re: 420.0
  Ma: 0.3
  Pr: 0.72
  kappa: 0.0  # constant viscosity
  T_0: 1.0

numerics:
  fd_order: 4
  operator_dissipation:
    enabled: false

# Prescribed-forcing table: one row per (beta, omega, forcing_path).
prescribed_forcing:
  - beta: 0.1
    omega: 0.5
    forcing_path: {str(forcing_npz_path)}
  - beta: 0.1
    omega: 1.0
    forcing_path: {str(forcing_npz_path)}

# CSV output path.
prescribed_output_csv: {str(output_csv_path)}
"""
    return yaml_text


def main():
    """Run the prescribed-forcing example."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_LOG, mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("prescribed_demo starting")

    try:
        import petsc4py

        petsc4py.init([])
        from linstabpy.utils.petsc_init import init_petsc_from_argv

        init_petsc_from_argv()
        petsc_available = True
    except ImportError:
        logger.warning("petsc4py not available; manufacturing forcing only")
        petsc_available = False

    if not petsc_available:
        logger.info("Manufacturing forcing without solve (no petsc4py)")
        # Generate a dummy forcing field for the npz file.
        f_hat = np.ones(100, dtype=complex) * 0.01
        np.savez(
            OUTPUT_FORCING_NPZ,
            f=f_hat,
            beta=0.1,
            omega=0.5,
        )
        logger.info("Created %s (dummy, no solve)", OUTPUT_FORCING_NPZ)
        return

    # With petsc4py: load Blasius and run the prescribed map.
    from examples.martini.matlab_baseflow import load_blasius
    from linstabpy.baseflow import sutherland_air
    from linstabpy.mesh import Mesh
    from linstabpy.solvers.router import prescribed_map

    logger.info("Loading Blasius base flow")
    base_flow = load_blasius()
    Re = 420.0
    Ma = 0.3
    Pr = 0.72
    T_0 = 1.0
    kappa = 0.0
    cfg_dict = {
        "Re": Re,
        "Ma": Ma,
        "Pr": Pr,
        "kappa": kappa,
        "T_0": T_0,
        "fd_order": 4,
        "numerics": {"operator_dissipation": {"enabled": False}},
    }
    air = sutherland_air(T_0=T_0, Re=Re, Ma=Ma, Pr=Pr, kappa=kappa)
    cfg_dict.update(air)

    logger.info("Building mesh from base flow")
    X, Y = base_flow["x"], base_flow["y"]
    mesh = Mesh.from_arrays(X, Y)

    logger.info("Manufacturing forcing field")
    f_hat = manufacture_forcing(mesh, base_flow, cfg_dict, beta=0.1, omega=0.5)
    np.savez(
        OUTPUT_FORCING_NPZ,
        f=f_hat,
        beta=0.1,
        omega=0.5,
    )
    logger.info("Saved forcing to %s", OUTPUT_FORCING_NPZ)

    # Generate and load the config.
    yaml_text = generate_config_yaml(OUTPUT_FORCING_NPZ, SCRIPT_DIR / "prescribed_demo_gains.csv")
    config_path = OUTPUT_CONFIG
    with open(config_path, "w") as f:
        f.write(yaml_text)
    logger.info("Generated config at %s", config_path)

    # Run prescribed_map.
    logger.info("Running prescribed_map")
    result = prescribed_map(base_flow, cfg=config_path)
    logger.info("prescribed_map complete: %d points", len(result.point_records))
    for record in result.point_records:
        logger.info(
            "  row=%d beta=%.4g omega=%.4g gain=%s status=%s",
            record.get("row"),
            record.get("beta"),
            record.get("omega"),
            record.get("gain"),
            record.get("status"),
        )


if __name__ == "__main__":
    main()
