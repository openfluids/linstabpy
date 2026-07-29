import matplotlib
import matplotlib.pyplot as plt
import numpy as np

cmap = plt.get_cmap("coolwarm")
matplotlib.rcParams.update(
    {
        "font.size": 22,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "text.usetex": True,
    }
)


def read_dat_file(filename):
    """
    Reads a .dat file with specific format:
    - First line: VARIABLES = X Y RHO U V W P T
    - Second line: ZONE I=601 J=301 F=POINT
    - Subsequent lines: Data in blocks of 3 lines with 3,3,2 values per line

    Returns:
        dict: A dictionary with keys as variable names and values as numpy arrays
    """
    # Initialize variables
    variables = ["X", "Y", "RHO", "U", "V", "W", "P", "T"]
    data = {var: [] for var in variables}
    total_points = 601 * 301
    points_read = 0

    with open(filename, "r") as file:
        # Skip the first two header lines
        file.readline()  # VARIABLES line
        file.readline()  # ZONE line

        # Read data lines
        while points_read < total_points:
            # Each block has 3 lines with 3, 3, 2 values respectively (total 8 values per block)
            line1 = file.readline().strip().split()
            line2 = file.readline().strip().split()
            line3 = file.readline().strip().split()

            # Combine all values from the block
            block_values = line1 + line2 + line3

            # There should be 8 values per block (one for each variable)
            if len(block_values) != 8:
                raise ValueError(f"Expected 8 values in block, got {len(block_values)}")

            # Append each value to its corresponding variable
            for i, var in enumerate(variables):
                data[var].append(float(block_values[i]))

            points_read += 1

    # Convert lists to numpy arrays and reshape to 601x301
    for var in variables:
        data[var] = np.array(data[var]).reshape(
            301, 601
        )  # Note: J (301) is rows, I (601) is columns

    return data


def read_complex_numbers(file_path):
    """Reads a .txt file with two columns (real and imaginary parts) and returns a list of complex numbers."""
    complex_numbers = []
    with open(file_path, "r") as file:
        for line in file:
            # Skip empty lines
            if line.strip() == "":
                continue
            # Split the line into parts
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Each line must contain exactly two numbers. Found: {line}")
            real_part = float(parts[0])
            imag_part = float(parts[1])
            complex_num = complex(real_part, imag_part)
            complex_numbers.append(complex_num)
    return complex_numbers


#### import x,y,u for plotting the separation bubble
x_bubble = np.load("x.npy")
y_bubble = np.load("y.npy")
u_bubble = np.load("u.npy")

title = "Song (2023)"
####define the variables
rho_inf = 0.0246  # kg/m3
P_inf = 1082  # Pa
T_inf = 153.33  # K
L = 80  # mm
Mach = 2.15
Reynolds = 100000
gamma = 1.4
c_inf = np.sqrt(gamma * P_inf / rho_inf)
U_inf = Mach * c_inf

beta = 19.2
St = 0.0

##### read the eigvector
mode = read_dat_file("eigenmode1.dat")
nx, ny = np.shape(mode["X"])
x = mode["X"] / L
y = mode["Y"] / L


# import the spectrum
omega = np.array(read_complex_numbers("./eigenvalues_Ziming.dat"))
omega *= L

levels = np.linspace(-1, 1, 11) * 5e-1
levels = np.delete(levels, np.where(levels == 0)[:])
rho = mode["RHO"]
u = mode["U"]
v = mode["V"]
w = mode["W"]
T = mode["T"]

max_rho = np.max(np.abs(rho.real))
max_u = np.max(np.abs(u.real))
max_v = np.max(np.abs(v.real))
max_T = np.max(np.abs(T.real))

rho /= max_rho
u /= max_u
v /= max_v
T /= max_T

if beta > 1e-8:
    max_w = np.max(np.abs(w.real))
    w /= max_w


######### plot balance
fig = plt.figure(figsize=(16, 10))
(ax1, ax11), (ax2, ax21), (ax3, ax31) = fig.subplots(3, 2)

axes_list = [ax1, ax11, ax2, ax21, ax3, ax31]

ax1.scatter(omega.real, omega.imag, color="black", s=30)
ax1.scatter(omega.real[0], omega.imag[0], color="blue", s=60)
ax1.axhline(y=0, color="gray", linewidth=1)
ax1.set_xlabel("$\\omega_r$", labelpad=-10)
ax1.set_ylabel("$\\omega_i$")
ax1.set_title(f"{omega[0]}")

CSf = ax11.contourf(x, y, rho.real, cmap="coolwarm", levels=levels, extend="both")
ax11.contour(x_bubble, y_bubble, u_bubble, levels=[-1e-6], colors=["blue"], linewidths=[1])
ax11.set_aspect("equal")
ax11.set_ylabel(r"$R(\rho/\rho_{max})$")

CSf = ax2.contourf(x, y, u.real, cmap="coolwarm", levels=levels, extend="both")
ax2.contour(x_bubble, y_bubble, u_bubble, levels=[-1e-6], colors=["blue"], linewidths=[1])
ax2.set_aspect("equal")
ax2.set_ylabel(r"$R(u/u_{max})$")

CSf = ax21.contourf(x, y, v.real, cmap="coolwarm", levels=levels, extend="both")
ax21.contour(x_bubble, y_bubble, u_bubble, levels=[-1e-6], colors=["blue"], linewidths=[1])
ax21.set_aspect("equal")
ax21.set_ylabel(r"$R(v/v_{max})$")

CSf = ax3.contourf(x, y, w.real, cmap="coolwarm", levels=levels, extend="both")
ax3.contour(x_bubble, y_bubble, u_bubble, levels=[-1e-6], colors=["blue"], linewidths=[1])
ax3.set_aspect("equal")
ax3.set_ylabel(r"$R(w/w_{max})$")

CSf = ax31.contourf(x, y, T.real, cmap="coolwarm", levels=levels, extend="both")
ax31.contour(x_bubble, y_bubble, u_bubble, levels=[-1e-6], colors=["blue"], linewidths=[1])
ax31.set_aspect("equal")
ax31.set_ylabel(r"$R(T/T_{max})$")

fig.suptitle(f"$\\beta = {beta}, St_L = {St}$ - {title}")
cbar = fig.colorbar(
    CSf,
    ax=axes_list,
    orientation="vertical",
    location="right",
    shrink=0.6,  # scale down colorbar height
    pad=0.02,
    aspect=30,
    anchor=(0.1, 0.5),
)
cbar.set_label("Scale - real part, normalized by max(forcing,response)")

# plt.subplots_adjust(hspace=0.25,wspace=0.1)
plt.savefig("eigmode1.png", dpi=600, bbox_inches="tight")
plt.close()
