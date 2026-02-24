#from scipy.io import loadmat, savemat
# from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter
import numpy as np
from pathlib import Path
matplotlib.use('Agg')  # Use non-interactive backend

original_path = "./half_mesh/"
plot_fields = True

x = np.load(f"{original_path}x.npy")
y = np.load(f"{original_path}y.npy")

rho = np.load(f'{original_path}rho.npy')
p = np.load(f'{original_path}p.npy')
u = np.load(f'{original_path}u.npy')
v = np.load(f'{original_path}v.npy')
w = np.load(f'{original_path}w.npy')
T = np.load(f'{original_path}T.npy')


# ##################3 create new domain based on mesh parameters
savepath = "./suction_side/"
Path(savepath).mkdir(parents=True,exist_ok=True)

n_init = 100
n_end = 250  #310

##################3 create new domain based on mesh parameters
# savepath = "./pressure_side/"
# Path(savepath).mkdir(parents=True,exist_ok=True)

# n_init = 910
# n_end = 1010

#### skip the points if necessary
x = x[n_init:n_end,:]
y = y[n_init:n_end,:]
rho = rho[n_init:n_end,:]
u = u[n_init:n_end,:]
v = v[n_init:n_end,:]
w = w[n_init:n_end,:]
T = T[n_init:n_end,:]
p = p[n_init:n_end,:]


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

