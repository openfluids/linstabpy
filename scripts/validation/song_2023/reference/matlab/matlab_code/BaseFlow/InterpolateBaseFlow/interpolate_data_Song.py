# from scipy.interpolate import griddata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def linear_interpolate(x_values, y_values, x_queries):
    """
    Vectorized linear interpolation for multiple query points using NumPy.

    Parameters:
        x_values (1D array): Known x values (must be sorted in ascending order).
        y_values (1D array): Known y values corresponding to x_values.
        x_queries (1D array): Query x values for which to interpolate.

    Returns:
        1D array: Interpolated y values at x_queries.
    """
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    x_queries = np.asarray(x_queries)

    if x_values.ndim != 1 or y_values.ndim != 1 or x_queries.ndim != 1:
        raise ValueError("All inputs must be 1D arrays.")
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    if np.any(x_queries < x_values[0]) or np.any(x_queries > x_values[-1]):
        raise ValueError("Some query points are outside the interpolation range.")

    # Find indices of the intervals
    indices = np.searchsorted(x_values, x_queries) - 1
    indices = np.clip(indices, 0, len(x_values) - 2)

    x0 = x_values[indices]
    x1 = x_values[indices + 1]
    y0 = y_values[indices]
    y1 = y_values[indices + 1]

    # Vectorized linear interpolation
    slopes = (y1 - y0) / (x1 - x0)
    y_queries = y0 + slopes * (x_queries - x0)

    return y_queries


def parse_number(val):
    """
    Parses a number from a string. Converts it to an integer if it has no decimal part,
    otherwise, keeps it as a float (even if the float has many trailing zeros).

    Parameters:
    val (str): The value to be parsed from the line.

    Returns:
    int or float: The parsed number, either an int (if no decimals) or a float.
    """
    # Remove any non-breaking spaces or BOM characters
    val = val.strip().replace("\u00a0", "").replace("\ufeff", "")
    if not val:
        return None  # Return None if the string is empty after cleanup

    # Convert the string to a float first
    num = float(val)

    # Check if the number has a decimal point (i.e., it's a float)
    if "." in val:
        return num  # Return as float (even if it's a "whole" float like 80.00000000)

    return int(num)  # If there's no decimal part, convert it to an int


def read_numbers_with_filter(
    filename, delimiter=None, exclude_condition=None, save_excluded=False, excluded_output_file=None
):
    """
    Reads a text file, parses its values, and stores them as either int or float based on
    whether the number has a decimal part. Optionally excludes certain lines and saves
    the excluded lines separately.

    Parameters:
    filename (str): Path to the .txt file to be read.
    delimiter (str or None): The delimiter to split values in a line. Default is None (any whitespace).
    exclude_condition (callable or None): A function that returns True for lines that should be excluded.
    save_excluded (bool): Whether to save the excluded lines.
    excluded_output_file (str or None): If specified, saves excluded lines to this file.

    Returns:
    list of int/float: The parsed numbers from the included lines.
    list of int/float (optional): The parsed numbers from the excluded lines (if save_excluded=True).
    """
    included_values = []  # List to store parsed values from non-excluded lines
    excluded_values = []  # List to store parsed values from excluded lines
    raw_excluded_lines = []  # Raw excluded lines for writing to file later

    with open(filename, "r") as file:
        # Iterate through each line in the file
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()  # Strip any leading/trailing whitespace
            if not stripped_line:
                continue  # Skip empty lines

            try:
                # Split the line by the delimiter (e.g., spaces, commas)
                tokens = stripped_line.split(delimiter)

                # Parse the tokens into numbers (either int or float)
                parsed_values = [
                    parse_number(val)  # Use the parse_number function to process each token
                    for val in tokens
                    if val.strip()  # Only process non-empty tokens
                ]
            except ValueError:
                print(f"Warning: Line {line_number} contains invalid number(s): '{stripped_line}'")
                continue  # Skip lines with invalid values

            # Apply the exclusion condition (if any)
            if exclude_condition and exclude_condition(stripped_line, line_number):
                if save_excluded:
                    excluded_values.extend(parsed_values)  # Add to excluded values list
                    raw_excluded_lines.append(
                        stripped_line
                    )  # Keep the raw excluded line for later use
                continue  # Skip the line and move to the next

            # Add parsed values from non-excluded lines to the included values list
            included_values.extend(parsed_values)

    # Optionally write the raw excluded lines to a file if requested
    if excluded_output_file and raw_excluded_lines:
        with open(excluded_output_file, "w") as out_file:
            for line in raw_excluded_lines:
                out_file.write(line + "\n")

    # Return both included and excluded parsed values if requested
    return (included_values, excluded_values) if save_excluded else included_values


savepath = "./original_mesh/"
Path(savepath).mkdir(parents=True, exist_ok=True)

rho_inf = 0.0246  # kg/m3
P_inf = 1082  # Pa
T_inf = 153.33  # K
L = 80  # mm
Mach = 2.15
Reynolds = 100000
gamma = 1.4

c_inf = np.sqrt(gamma * P_inf / rho_inf)
U_inf = Mach * c_inf


plot_fields = True


grid, idx_grid = read_numbers_with_filter(
    "grid.grd", exclude_condition=lambda line, line_number: line_number == 2, save_excluded=True
)

### the grid is obtained at cell vertice, lower left vertice
npts = idx_grid[0] * idx_grid[1]
grid = np.array(grid) / L
x = grid[:npts]
y = grid[npts:]

#### interpolate the grid to obtain the positions at cell center
grid = grid.reshape((idx_grid[0], idx_grid[1], 2), order="F")
x = grid[:, :, 0]
y = grid[:, :, 1]

y_cc = np.zeros((idx_grid[0] + 1, idx_grid[1] + 1))
y_cc[:-1, 1:-1] = (y[:, :-1] + y[:, 1:]) / 2
y_cc[:-1, 0] = y[:, 0] - (y_cc[:-1, 1] - y[:, 0])
y_cc[:-1, -1] = y[:, -1] + (y[:, -1] - y_cc[:-1, -2])
y_cc[-1, :] = y_cc[-2, :]

y_interm = np.zeros((idx_grid[0] + 1, idx_grid[1]))
y_interm[:-1, :] = y
y_interm[-1, :] = y[-1, :]

x_cc = np.zeros((idx_grid[0] + 1, idx_grid[1]))
x_cc[1:-1, :] = (x[:-1, :] + x[1:, :]) / 2
x_cc[0, :] = x[0, :] - (x_cc[1, :] - x[0, :])
x_cc[-1, :] = x[-1, :] + (x[-1, :] - x_cc[-2, :])


### data is obtained at the cell center (and has a ghost cell)
npts_data = (idx_grid[0] + 1) * (idx_grid[1] + 1)
data = read_numbers_with_filter("mean_flow.dat")
data = np.array(data)

rho = data[:npts_data] / rho_inf
u = data[npts_data : 2 * npts_data] / U_inf
v = data[2 * npts_data : 3 * npts_data] / U_inf
p = data[3 * npts_data : 4 * npts_data] / P_inf
T = data[4 * npts_data : 5 * npts_data] / T_inf
mu = data[5 * npts_data :]

rho = rho.reshape((idx_grid[0] + 1, idx_grid[1] + 1), order="F")
u = u.reshape((idx_grid[0] + 1, idx_grid[1] + 1), order="F")
v = v.reshape((idx_grid[0] + 1, idx_grid[1] + 1), order="F")
p = p.reshape((idx_grid[0] + 1, idx_grid[1] + 1), order="F")
T = T.reshape((idx_grid[0] + 1, idx_grid[1] + 1), order="F")
mu = mu.reshape((idx_grid[0] + 1, idx_grid[1] + 1), order="F")

### y interpolation
rho_interm = np.zeros((idx_grid[0] + 1, idx_grid[1]))
u_interm = np.zeros((idx_grid[0] + 1, idx_grid[1]))
v_interm = np.zeros((idx_grid[0] + 1, idx_grid[1]))
p_interm = np.zeros((idx_grid[0] + 1, idx_grid[1]))
T_interm = np.zeros((idx_grid[0] + 1, idx_grid[1]))
mu_interm = np.zeros((idx_grid[0] + 1, idx_grid[1]))
for i in range(idx_grid[0]):
    rho_interm[i, :] = linear_interpolate(y_cc[i, :], rho[i, :], y_interm[i, :])
    u_interm[i, :] = linear_interpolate(y_cc[i, :], u[i, :], y_interm[i, :])
    v_interm[i, :] = linear_interpolate(y_cc[i, :], v[i, :], y_interm[i, :])
    p_interm[i, :] = linear_interpolate(y_cc[i, :], p[i, :], y_interm[i, :])
    T_interm[i, :] = linear_interpolate(y_cc[i, :], T[i, :], y_interm[i, :])
    mu_interm[i, :] = linear_interpolate(y_cc[i, :], mu[i, :], y_interm[i, :])


rho_int = np.zeros((idx_grid[0], idx_grid[1]))
u_int = np.zeros((idx_grid[0], idx_grid[1]))
v_int = np.zeros((idx_grid[0], idx_grid[1]))
p_int = np.zeros((idx_grid[0], idx_grid[1]))
T_int = np.zeros((idx_grid[0], idx_grid[1]))
mu_int = np.zeros((idx_grid[0], idx_grid[1]))
### x interpolation
for j in range(idx_grid[1]):
    rho_int[:, j] = linear_interpolate(x_cc[:, j], rho_interm[:, j], x[:, j])
    u_int[:, j] = linear_interpolate(x_cc[:, j], u_interm[:, j], x[:, j])
    v_int[:, j] = linear_interpolate(x_cc[:, j], v_interm[:, j], x[:, j])
    p_int[:, j] = linear_interpolate(x_cc[:, j], p_interm[:, j], x[:, j])
    T_int[:, j] = linear_interpolate(x_cc[:, j], T_interm[:, j], x[:, j])
    mu_int[:, j] = linear_interpolate(x_cc[:, j], mu_interm[:, j], x[:, j])

#### create w
w = np.zeros(np.shape(x))

#### kill the first and last point that has errors
x = x[1:-1, :]
y = y[1:-1, :]
rho_int = rho_int[1:-1, :]
u_int = u_int[1:-1, :]
v_int = v_int[1:-1, :]
w = w[1:-1, :]
T_int = T_int[1:-1, :]
mu_int = mu_int[1:-1, :]
p_int = p_int[1:-1, :]

if plot_fields == True:
    fig, ax = plt.subplots()
    ax.contourf(x, y, rho_int, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"{savepath}/rho.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x, y, u_int, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"{savepath}u.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x, y, v_int, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"{savepath}v.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x, y, p_int, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"{savepath}p.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x, y, T_int, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"{savepath}T.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x, y, mu_int, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"{savepath}mu.png")
    plt.close()

np.save(f"{savepath}/rho.npy", rho_int)
np.save(f"{savepath}/p.npy", p_int)
np.save(f"{savepath}/u.npy", u_int)
np.save(f"{savepath}/v.npy", v_int)
np.save(f"{savepath}/w.npy", w)
np.save(f"{savepath}/T.npy", T_int)
np.save(f"{savepath}/x.npy", x)
np.save(f"{savepath}/y.npy", y)
