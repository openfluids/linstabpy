from scipy.io import loadmat, savemat
# from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from pathlib import Path

savepath = "./half_mesh/"
Path(savepath).mkdir(parents=True,exist_ok=True)

original_path = "./original_mesh/"
plot_fields = True

# if skip points in mesh
eta_skip = 2
ksi_skip = 2


x = np.load(f"{original_path}x.npy")
y = np.load(f"{original_path}y.npy")

rho = np.load(f'{original_path}rho.npy')
p = np.load(f'{original_path}p.npy')
u = np.load(f'{original_path}u.npy')
v = np.load(f'{original_path}v.npy')
w = np.load(f'{original_path}w.npy')
T = np.load(f'{original_path}T.npy')


#### skip the points if necessary
x = x[::ksi_skip,::eta_skip]
y = y[::ksi_skip,::eta_skip]
rho = rho[::ksi_skip,::eta_skip]
u = u[::ksi_skip,::eta_skip]
v = v[::ksi_skip,::eta_skip]
w = w[::ksi_skip,::eta_skip]
T = T[::ksi_skip,::eta_skip]
p = p[::ksi_skip,::eta_skip]


if plot_fields:
    fig, ax = plt.subplots()
    ax.contourf(x,y,rho, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}rho.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x,y,u, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}u.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x,y,v, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}v.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x,y,p, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}p.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x,y,T, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}T.png')
    plt.close()

np.save(f'{savepath}/rho.npy',rho)
np.save(f'{savepath}/p.npy',p)
np.save(f'{savepath}/u.npy',u)
np.save(f'{savepath}/v.npy',v)
np.save(f'{savepath}/w.npy',w)
np.save(f'{savepath}/T.npy',T)
np.save(f'{savepath}/x.npy',x)
np.save(f'{savepath}/y.npy',y)

