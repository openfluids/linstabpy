#!/usr/bin/env python3
"""Map Song (2023) finite-volume cell-centre mean flow to nodal baseflow arrays.

Cell-centre primitive fields sit on a stretched tensor-product Cartesian
grid.
The mapper reconstructs them at vertices by separable one-dimensional
linear interpolation.
It interpolates in physical wall-normal then streamwise coordinates
(equivalent to bilinear reconstruction on the full vertex grid).

Place Song's Plot3D export under ``inputs/raw/`` (``grid.grd``, ``mean_flow.dat``).
Default output: ``inputs/baseflow/*.npy`` (599 × 301 after streamwise trim).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.song.reference_paths import BASEFLOW_NPZ, INPUTS_BASEFLOW_DIR, INPUTS_RAW_DIR

SCRIPT_DIR = Path(__file__).resolve().parent

# Freestream / geometry reference values (Song & Hao 2023; Robinet 2007; Degrez et al. 1987).
# Used only to non-dimensionalise the finite-volume export (grid length, rho, u, v, p, T).
SONG2023_REFERENCE = {
    "mach": 2.15,
    "rho_inf": 0.0246,  # kg/m^3
    "p_inf": 1082.0,  # Pa
    "t_inf": 153.33,  # K
    "gamma": 1.4,
    "reference_length_mm": 80.0,  # leading edge to inviscid shock impingement
    "reynolds": 1.0e5,  # Re based on reference_length (not used in scaling here)
    "prandtl": 0.72,  # used downstream in LSA, not in this file
    "t_wall": 273.6,  # K; encoded in mean_flow.dat, not in scaling here
}


def linear_interpolate(
    x_values: np.ndarray, y_values: np.ndarray, x_queries: np.ndarray
) -> np.ndarray:
    """Vectorised 1D linear interpolation (physical abscissae, ascending)."""
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    x_queries = np.asarray(x_queries)

    if x_values.ndim != 1 or y_values.ndim != 1 or x_queries.ndim != 1:
        raise ValueError("All inputs must be 1D arrays.")
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    if np.any(x_queries < x_values[0]) or np.any(x_queries > x_values[-1]):
        raise ValueError("Some query points are outside the interpolation range.")

    indices = np.searchsorted(x_values, x_queries) - 1
    indices = np.clip(indices, 0, len(x_values) - 2)

    x0 = x_values[indices]
    x1 = x_values[indices + 1]
    y0 = y_values[indices]
    y1 = y_values[indices + 1]

    slopes = (y1 - y0) / (x1 - x0)
    return y0 + slopes * (x_queries - x0)


def parse_number(val: str) -> int | float | None:
    val = val.strip().replace("\u00a0", "").replace("\ufeff", "")
    if not val:
        return None
    num = float(val)
    return num if "." in val else int(num)


def read_numbers_with_filter(
    filename: Path | str,
    *,
    delimiter: str | None = None,
    exclude_condition: Callable[[str, int], bool] | None = None,
    save_excluded: bool = False,
) -> list[float | int] | tuple[list[float | int], list[float | int]]:
    included_values: list[float | int] = []
    excluded_values: list[float | int] = []

    with open(filename, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                tokens = stripped_line.split(delimiter)
                parsed_values = [parse_number(val) for val in tokens if val.strip()]
                parsed_values = [v for v in parsed_values if v is not None]
            except ValueError:
                print(f"Warning: line {line_number} contains invalid number(s): {stripped_line!r}")
                continue

            if exclude_condition and exclude_condition(stripped_line, line_number):
                if save_excluded:
                    excluded_values.extend(parsed_values)
                continue

            included_values.extend(parsed_values)

    return (included_values, excluded_values) if save_excluded else included_values


def build_cell_centre_coordinates(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_cc, y_cc, y_on_cc_rows)`` for FV ghost-inclusive CC storage."""
    nx, ny = x.shape

    y_cc = np.zeros((nx + 1, ny + 1))
    y_cc[:-1, 1:-1] = (y[:, :-1] + y[:, 1:]) / 2
    y_cc[:-1, 0] = y[:, 0] - (y_cc[:-1, 1] - y[:, 0])
    y_cc[:-1, -1] = y[:, -1] + (y[:, -1] - y_cc[:-1, -2])
    y_cc[-1, :] = y_cc[-2, :]

    y_on_cc_rows = np.zeros((nx + 1, ny))
    y_on_cc_rows[:-1, :] = y
    y_on_cc_rows[-1, :] = y[-1, :]

    x_cc = np.zeros((nx + 1, ny))
    x_cc[1:-1, :] = (x[:-1, :] + x[1:, :]) / 2
    x_cc[0, :] = x[0, :] - (x_cc[1, :] - x[0, :])
    x_cc[-1, :] = x[-1, :] + (x[-1, :] - x_cc[-2, :])

    return x_cc, y_cc, y_on_cc_rows


def cc_to_vertex_separable(
    field_cc: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_cc: np.ndarray,
    y_cc: np.ndarray,
    y_on_cc_rows: np.ndarray,
) -> np.ndarray:
    """Cell-centre field -> vertex field via physical y-then-x linear interpolation."""
    nx, ny = x.shape
    if field_cc.shape != (nx + 1, ny + 1):
        raise ValueError(f"field_cc shape {field_cc.shape} != ({nx + 1}, {ny + 1})")

    interm = np.zeros((nx + 1, ny))
    for i in range(nx + 1):
        interm[i, :] = linear_interpolate(y_cc[i, :], field_cc[i, :], y_on_cc_rows[i, :])

    out = np.zeros((nx, ny))
    for j in range(ny):
        out[:, j] = linear_interpolate(x_cc[:, j], interm[:, j], x[:, j])

    return out


def load_song2023_fv_export(
    grid_path: Path,
    mean_flow_path: Path,
    *,
    reference_length_mm: float,
    rho_inf: float,
    p_inf: float,
    t_inf: float,
    mach: float,
    gamma: float = 1.4,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read ``grid.grd`` + ``mean_flow.dat``; return vertex coords and CC fields."""
    grid, idx_grid = read_numbers_with_filter(
        grid_path,
        exclude_condition=lambda _line, line_number: line_number == 2,
        save_excluded=True,
    )
    nx = int(idx_grid[0])
    ny = int(idx_grid[1])

    grid_arr = np.asarray(grid, dtype=float) / reference_length_mm
    grid_arr = grid_arr.reshape((nx, ny, 2), order="F")
    x = grid_arr[:, :, 0]
    y = grid_arr[:, :, 1]

    npts_data = (nx + 1) * (ny + 1)
    data = np.asarray(read_numbers_with_filter(mean_flow_path), dtype=float)

    c_inf = np.sqrt(gamma * p_inf / rho_inf)
    u_inf = mach * c_inf

    def _reshape_block(offset: int, scale: float) -> np.ndarray:
        block = data[offset * npts_data : (offset + 1) * npts_data] / scale
        return block.reshape((nx + 1, ny + 1), order="F")

    fields_cc = {
        "rho": _reshape_block(0, rho_inf),
        "u": _reshape_block(1, u_inf),
        "v": _reshape_block(2, u_inf),
        "p": _reshape_block(3, p_inf),
        "T": _reshape_block(4, t_inf),
        "mu": _reshape_block(5, 1.0),
    }
    return x, y, fields_cc


def apply_streamwise_trim(
    arrays: dict[str, np.ndarray],
    *,
    mode: str,
) -> dict[str, np.ndarray]:
    if mode == "none":
        return arrays
    if mode == "both":
        sl = slice(1, -1)
    elif mode == "upstream":
        sl = slice(1, None)
    elif mode == "downstream":
        sl = slice(None, -1)
    else:
        raise ValueError(f"unknown trim mode {mode!r}")

    out = {}
    for key, arr in arrays.items():
        if arr.shape == arrays["x"].shape:
            out[key] = arr[sl, :]
        else:
            out[key] = arr
    return out


def interpolate_baseflow(
    grid_path: Path,
    mean_flow_path: Path,
    *,
    reference_length_mm: float = SONG2023_REFERENCE["reference_length_mm"],
    rho_inf: float = SONG2023_REFERENCE["rho_inf"],
    p_inf: float = SONG2023_REFERENCE["p_inf"],
    t_inf: float = SONG2023_REFERENCE["t_inf"],
    mach: float = SONG2023_REFERENCE["mach"],
    gamma: float = SONG2023_REFERENCE["gamma"],
    trim_streamwise: str = "both",
) -> dict[str, np.ndarray]:
    x, y, fields_cc = load_song2023_fv_export(
        grid_path,
        mean_flow_path,
        reference_length_mm=reference_length_mm,
        rho_inf=rho_inf,
        p_inf=p_inf,
        t_inf=t_inf,
        mach=mach,
        gamma=gamma,
    )

    x_cc, y_cc, y_on_cc_rows = build_cell_centre_coordinates(x, y)

    nodal: dict[str, np.ndarray] = {"x": x, "y": y}
    for name in ("rho", "u", "v", "p", "T", "mu"):
        nodal[name] = cc_to_vertex_separable(fields_cc[name], x, y, x_cc, y_cc, y_on_cc_rows)
    nodal["w"] = np.zeros_like(x)

    return apply_streamwise_trim(nodal, mode=trim_streamwise)


def write_baseflow_npy(output_dir: Path, fields: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("x", "y", "rho", "u", "v", "w", "p", "T"):
        np.save(output_dir / f"{name}.npy", fields[name])
    np.save(output_dir / "sponge.npy", np.zeros_like(fields["x"]))


def read_reference_baseflow(path: Path = BASEFLOW_NPZ) -> dict[str, np.ndarray]:
    """Read the shipped nodal base flow and scale it for the solver.

    The npz stores velocity / c_inf, so its u_max is Ma = 2.15. The solver
    reads velocity / U_inf, so u, v and w are divided by the stored Ma.
    The other fields are already in solver scaling.
    """
    with np.load(path, allow_pickle=False) as packed:
        mach = float(packed["Ma"])
        fields = {
            name: np.asarray(packed[name], dtype=float)
            for name in ("x", "y", "rho", "u", "v", "w", "T", "sponge")
        }
    for name in ("u", "v", "w"):
        fields[name] = fields[name] / mach
    return fields


def plot_fields(output_dir: Path, fields: dict[str, np.ndarray]) -> None:
    x, y = fields["x"], fields["y"]
    for name in ("rho", "u", "v", "p", "T", "mu"):
        fig, ax = plt.subplots()
        ax.contourf(x, y, fields[name], cmap="coolwarm")
        ax.set_aspect("equal")
        fig.savefig(output_dir / f"{name}.png")
        plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=INPUTS_RAW_DIR / "grid.grd")
    parser.add_argument("--mean-flow", type=Path, default=INPUTS_RAW_DIR / "mean_flow.dat")
    parser.add_argument("--output-dir", type=Path, default=INPUTS_BASEFLOW_DIR)
    parser.add_argument(
        "--reference-length-mm", type=float, default=SONG2023_REFERENCE["reference_length_mm"]
    )
    parser.add_argument("--rho-inf", type=float, default=SONG2023_REFERENCE["rho_inf"])
    parser.add_argument("--p-inf", type=float, default=SONG2023_REFERENCE["p_inf"])
    parser.add_argument("--t-inf", type=float, default=SONG2023_REFERENCE["t_inf"])
    parser.add_argument("--mach", type=float, default=SONG2023_REFERENCE["mach"])
    parser.add_argument("--gamma", type=float, default=SONG2023_REFERENCE["gamma"])
    parser.add_argument(
        "--trim-streamwise",
        choices=("none", "both", "upstream", "downstream"),
        default="both",
        help="Discard boundary streamwise rows after interpolation (default: both -> 599x301).",
    )
    parser.add_argument(
        "--from-reference",
        action="store_true",
        help=f"Start from the shipped {BASEFLOW_NPZ.name} instead of the raw Plot3D export.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip contour PNGs of nodal fields (default: write plots).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.from_reference:
        fields = read_reference_baseflow()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in fields.items():
            np.save(args.output_dir / f"{name}.npy", arr)
        print(f"Wrote baseflow from {BASEFLOW_NPZ.name} to {args.output_dir}")
        return 0
    ref = SONG2023_REFERENCE
    c_inf = np.sqrt(ref["gamma"] * ref["p_inf"] / ref["rho_inf"])
    u_inf = ref["mach"] * c_inf
    print(
        "Reference: "
        f"M={ref['mach']}, rho_inf={ref['rho_inf']} kg/m^3, "
        f"p_inf={ref['p_inf']} Pa, T_inf={ref['t_inf']} K, "
        f"gamma={ref['gamma']}, L={ref['reference_length_mm']} mm, "
        f"U_inf={u_inf:.4f} m/s (from M and gamma,p,rho)"
    )
    fields = interpolate_baseflow(
        args.grid,
        args.mean_flow,
        reference_length_mm=args.reference_length_mm,
        rho_inf=args.rho_inf,
        p_inf=args.p_inf,
        t_inf=args.t_inf,
        mach=args.mach,
        gamma=args.gamma,
        trim_streamwise=args.trim_streamwise,
    )
    write_baseflow_npy(args.output_dir, fields)
    if not args.no_plot:
        plot_fields(args.output_dir, fields)
    shape = fields["rho"].shape
    print(f"Wrote baseflow to {args.output_dir} (shape {shape[0]} x {shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
