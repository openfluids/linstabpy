# from scipy.interpolate import griddata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

savepath = "./doubleY_mesh/"
Path(savepath).mkdir(parents=True, exist_ok=True)

original_path = "./original_mesh/"
plot_fields = True

x = np.load(f"{original_path}x.npy")
y = np.load(f"{original_path}y.npy")

nxi, nyi = np.shape(x)
x_double = np.zeros((nxi, 2 * nyi - 1))
y_double = np.zeros((nxi, 2 * nyi - 1))

xy_int = (x[:, :-1] + x[:, 1:]) / 2
yy_int = (y[:, :-1] + y[:, 1:]) / 2

print(np.shape(x), np.shape(x_double), np.shape(xy_int))
for j in range(nyi):
    ## put the original points
    x_double[:, 2 * j] = x[:, j]
    y_double[:, 2 * j] = y[:, j]

for j in range(nyi - 1):
    # ## add the interpolated points in y direction
    x_double[:, 2 * j + 1] = xy_int[:, j]
    y_double[:, 2 * j + 1] = yy_int[:, j]


rho = np.load(f"{original_path}rho.npy")
p = np.load(f"{original_path}p.npy")
u = np.load(f"{original_path}u.npy")
v = np.load(f"{original_path}v.npy")
w = np.load(f"{original_path}w.npy")
T = np.load(f"{original_path}T.npy")


rho_double = np.zeros((nxi, 2 * nyi - 1))
p_double = np.zeros((nxi, 2 * nyi - 1))
u_double = np.zeros((nxi, 2 * nyi - 1))
v_double = np.zeros((nxi, 2 * nyi - 1))
w_double = np.zeros((nxi, 2 * nyi - 1))
T_double = np.zeros((nxi, 2 * nyi - 1))


rhoy_int = (rho[:, :-1] + rho[:, 1:]) / 2
py_int = (p[:, :-1] + p[:, 1:]) / 2
uy_int = (u[:, :-1] + u[:, 1:]) / 2
vy_int = (v[:, :-1] + v[:, 1:]) / 2
wy_int = (w[:, :-1] + w[:, 1:]) / 2
Ty_int = (T[:, :-1] + T[:, 1:]) / 2

for j in range(nyi):
    ## put the original points
    rho_double[:, 2 * j] = rho[:, j]
    p_double[:, 2 * j] = p[:, j]
    u_double[:, 2 * j] = u[:, j]
    v_double[:, 2 * j] = v[:, j]
    w_double[:, 2 * j] = w[:, j]
    T_double[:, 2 * j] = T[:, j]
for j in range(nyi - 1):
    # ## add the interpolated points in y direction
    rho_double[:, 2 * j + 1] = rhoy_int[:, j]
    p_double[:, 2 * j + 1] = py_int[:, j]
    u_double[:, 2 * j + 1] = uy_int[:, j]
    v_double[:, 2 * j + 1] = vy_int[:, j]
    w_double[:, 2 * j + 1] = wy_int[:, j]
    T_double[:, 2 * j + 1] = Ty_int[:, j]


if plot_fields:
    fig, ax = plt.subplots()
    ax.contourf(x_double, y_double, rho_double, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"{savepath}rho.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_double, y_double, u_double, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"{savepath}u.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_double, y_double, v_double, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"{savepath}v.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_double, y_double, p_double, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"{savepath}p.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_double, y_double, T_double, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"{savepath}T.png")
    plt.close()


np.save(f"{savepath}/rho.npy", rho_double)
np.save(f"{savepath}/p.npy", p_double)
np.save(f"{savepath}/u.npy", u_double)
np.save(f"{savepath}/v.npy", v_double)
np.save(f"{savepath}/w.npy", w_double)
np.save(f"{savepath}/T.npy", T_double)
np.save(f"{savepath}/x.npy", x_double)
np.save(f"{savepath}/y.npy", y_double)
