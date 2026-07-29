from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from explicit_filter_4thOrder_Lele import fourth_order_filter
from GridMetrics2D.fd_compact_10th_nonperiodic import compact_10th_nonperiodic
from GridMetrics2D.MetricClass2D import MetricClass
from Heaviside_velocity_divergence import heaviside_compression_waves
from scipy.io import loadmat, savemat
from sponge_generator import generate_sponge

print("Creating base flow ...")

plot_fields = True

#### these variables will be modified by the bash script
savepath = "Results_St0.0_beta19.2_NeumannOutBotTemp/NoSponge_NoShock_NoFilter/"
casepath = "../InterpolateBaseFlow/"
casename = "original_mesh"
matlabpath = "SolverMartini"
### define sponge parameters
#### top, right, bot, left direction
sponge_flag = False
sponge_pts = [0, 0, 0, 0]
sponge_amp = [0, 0, 0, 0]


## flag to see if heaviside or specific shock sponge is used
shock_flag = False
threshold_heaviside = -1.0
amplitude_shock = 0

### filter parameters
filter_flag = False
# how many times the filter will pass in x and y directions
nfilter = 0
# minimum point in y direction which the filter will pass
nyminfilter = 0


################### The code
### create the saving paths
Path(f"../../{savepath}/EigModes/").mkdir(parents=True, exist_ok=True)
Path(f"../../{savepath}/FigEigModes/").mkdir(parents=True, exist_ok=True)
Path(f"../../{savepath}/ResolventModes/").mkdir(parents=True, exist_ok=True)
Path(f"../../{savepath}/FigResolventModes/").mkdir(parents=True, exist_ok=True)


### read the flow and mesh variables
x = np.load(f"{casepath}{casename}/x.npy")
y = np.load(f"{casepath}{casename}/y.npy")
# for i in range(50):
#     x[:,:] = fourth_order_filter(x[:,:],'x')
#     x[:,:] = fourth_order_filter(x[:,:],'y')
#     y[:,:] = fourth_order_filter(y[:,:],'x')
#     y[:,:] = fourth_order_filter(y[:,:],'y')

# dy = y[:,1:] -y[:,:-1]
# print(dy[15,:])

rho = np.load(f"{casepath}{casename}/rho.npy")
u = np.load(f"{casepath}{casename}/u.npy")
v = np.load(f"{casepath}{casename}/v.npy")
w = np.load(f"{casepath}{casename}/w.npy")
T = np.load(f"{casepath}{casename}/T.npy")

##create metrics object
metrics = MetricClass(x=x, y=y, scheme_metrics=compact_10th_nonperiodic)

### apply Ducros
if shock_flag:
    Dx = loadmat(f"{casepath}{casename}/Dx.mat")["Dx"]
    Dy = loadmat(f"{casepath}{casename}/Dy.mat")["Dy"]
    sponge = heaviside_compression_waves(u, v, Dx, Dy, threshold_heaviside) * amplitude_shock
else:
    sponge = np.zeros_like(x)

if sponge_flag:
    print("entrou")
    sponge = generate_sponge(sponge, sponge_pts, sponge_amp)

#### filter
if filter_flag:
    # evaluate metric terms
    metrics.eval_metric_terms()
    detJ = metrics.det_jacobian
    dmdx = metrics.comp_to_phys
    dxdm = metrics.phys_to_comp

    rho_cont = detJ * rho
    T_cont = detJ * T

    u_cont = (u * dmdx[:, :, 0, 0] + v * dmdx[:, :, 0, 1]) * detJ
    v_cont = (u * dmdx[:, :, 1, 0] + v * dmdx[:, :, 1, 1]) * detJ

    for i in range(nfilter):
        rho_cont[:, nyminfilter:] = fourth_order_filter(rho_cont[:, nyminfilter:], "x")
        u_cont[:, nyminfilter:] = fourth_order_filter(u_cont[:, nyminfilter:], "x")
        v_cont[:, nyminfilter:] = fourth_order_filter(v_cont[:, nyminfilter:], "x")
        T_cont[:, nyminfilter:] = fourth_order_filter(T_cont[:, nyminfilter:], "x")

        rho_cont[:, nyminfilter:] = fourth_order_filter(rho_cont[:, nyminfilter:], "y")
        u_cont[:, nyminfilter:] = fourth_order_filter(u_cont[:, nyminfilter:], "y")
        v_cont[:, nyminfilter:] = fourth_order_filter(v_cont[:, nyminfilter:], "y")
        T_cont[:, nyminfilter:] = fourth_order_filter(T_cont[:, nyminfilter:], "y")

    rho = rho_cont / detJ
    T = T_cont / detJ
    u_cont = u_cont / detJ
    v_cont = v_cont / detJ
    u = u_cont * dxdm[:, :, 0, 0] + v_cont * dxdm[:, :, 0, 1]
    v = u_cont * dxdm[:, :, 1, 0] + v_cont * dxdm[:, :, 1, 1]

    # for i in range(nfilter):
    #     rho[:,nyminfilter:] = fourth_order_filter(rho[:,nyminfilter:],'x')
    #     u[:,nyminfilter:] = fourth_order_filter(u[:,nyminfilter:],'x')
    #     v[:,nyminfilter:] = fourth_order_filter(v[:,nyminfilter:],'x')
    #     T[:,nyminfilter:] = fourth_order_filter(T[:,nyminfilter:],'x')

    #     rho[:,nyminfilter:] = fourth_order_filter(rho[:,nyminfilter:],'y')
    #     u[:,nyminfilter:] = fourth_order_filter(u[:,nyminfilter:],'y')
    #     v[:,nyminfilter:] = fourth_order_filter(v[:,nyminfilter:],'y')
    #     T[:,nyminfilter:] = fourth_order_filter(T[:,nyminfilter:],'y')


savemat(f"../../{matlabpath}/BaseFlows/base_flow.mat", {"rho": rho, "u": u, "v": v, "w": w, "T": T})
savemat(f"../../{matlabpath}/mesh.mat", {"x": x, "y": y})
savemat(f"../../{matlabpath}/sponge.mat", {"sponge_aux": sponge})

np.save(f"../../{savepath}/rho.npy", rho)
np.save(f"../../{savepath}/u.npy", u)
np.save(f"../../{savepath}/v.npy", v)
np.save(f"../../{savepath}/w.npy", w)
np.save(f"../../{savepath}/T.npy", T)
np.save(f"../../{savepath}/x.npy", x)
np.save(f"../../{savepath}/y.npy", y)
np.save(f"../../{savepath}/sponge.npy", sponge)

if plot_fields:
    fig, ax = plt.subplots()
    ax.contourf(x, y, rho, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"../../{savepath}rho.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x, y, u, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"../../{savepath}u.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x, y, v, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"../../{savepath}v.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x, y, T, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"../../{savepath}T.png")
    plt.close()

    # fig, ax = plt.subplots()
    # ax.contourf(x,y,detJ, cmap = 'coolwarm')
    # ax.set_aspect('equal')
    # plt.savefig(f'../../{savepath}detJ.png')
    # plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x, y, sponge, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"../../{savepath}sponge.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x, y, x, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"../../{savepath}x.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x, y, y, cmap="coolwarm")
    ax.set_aspect("equal")
    plt.savefig(f"../../{savepath}y.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(x[::10], y[::10], "-k", linewidth=1)
    ax.plot(x[::10].T, y[::10].T, "-k", linewidth=1)
    ax.set_aspect("equal")
    plt.savefig(f"../../{savepath}grid.png")
    plt.close()
