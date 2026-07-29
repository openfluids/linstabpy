# from scipy.interpolate import griddata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

savepath = "./doubleX_mesh/"
Path(savepath).mkdir(parents=True, exist_ok=True)

original_path = "./original_mesh/"
plot_fields = True


x = np.load(f"{original_path}x.npy")
y = np.load(f"{original_path}y.npy")

nxi, nyi = np.shape(x)
x_double = np.zeros((2 * nxi - 1, nyi))
y_double = np.zeros((2 * nxi - 1, nyi))

xx_int = (x[:-1, :] + x[1:, :]) / 2
yx_int = (y[:-1, :] + y[1:, :]) / 2

print(np.shape(x), np.shape(x_double), np.shape(xx_int))
for i in range(nxi):
    ## put the original points
    x_double[2 * i, :] = x[i, :]
    y_double[2 * i, :] = y[i, :]
for i in range(nxi - 1):
    ## add the interpolated points in x direction
    x_double[2 * i + 1, :] = xx_int[i, :]
    y_double[2 * i + 1, :] = yx_int[i, :]

rho = np.load(f"{original_path}rho.npy")
p = np.load(f"{original_path}p.npy")
u = np.load(f"{original_path}u.npy")
v = np.load(f"{original_path}v.npy")
w = np.load(f"{original_path}w.npy")
T = np.load(f"{original_path}T.npy")


rho_double = np.zeros((2 * nxi - 1, nyi))
p_double = np.zeros((2 * nxi - 1, nyi))
u_double = np.zeros((2 * nxi - 1, nyi))
v_double = np.zeros((2 * nxi - 1, nyi))
w_double = np.zeros((2 * nxi - 1, nyi))
T_double = np.zeros((2 * nxi - 1, nyi))


rhox_int = (rho[:-1, :] + rho[1:, :]) / 2
px_int = (p[:-1, :] + p[1:, :]) / 2
ux_int = (u[:-1, :] + u[1:, :]) / 2
vx_int = (v[:-1, :] + v[1:, :]) / 2
wx_int = (w[:-1, :] + w[1:, :]) / 2
Tx_int = (T[:-1, :] + T[1:, :]) / 2


for i in range(nxi):
    ## put the original points
    rho_double[2 * i, :] = rho[i, :]
    p_double[2 * i, :] = p[i, :]
    u_double[2 * i, :] = u[i, :]
    v_double[2 * i, :] = v[i, :]
    w_double[2 * i, :] = w[i, :]
    T_double[2 * i, :] = T[i, :]
for i in range(nxi - 1):
    ## add the interpolated points in x direction
    rho_double[2 * i + 1, :] = rhox_int[i, :]
    p_double[2 * i + 1, :] = px_int[i, :]
    u_double[2 * i + 1, :] = ux_int[i, :]
    v_double[2 * i + 1, :] = vx_int[i, :]
    w_double[2 * i + 1, :] = wx_int[i, :]
    T_double[2 * i + 1, :] = Tx_int[i, :]

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
