from scipy.io import loadmat, savemat
# from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from pathlib import Path

nmax = 200
cut_half = 2

savepath = f"./meshY_nmax{nmax}_cut{cut_half}/"
Path(savepath).mkdir(parents=True,exist_ok=True)

original_path = "./original_mesh/"
plot_fields = True


x = np.load(f"{original_path}x.npy")
y = np.load(f"{original_path}y.npy")

nxi, nyi = np.shape(x)

x_interp = x[:,nmax:]
y_interp = y[:,nmax:]

for nt in range(cut_half):
    x_new = []
    y_new = []
    x_half = (x_interp[:,:-1]+x_interp[:,1:])/2
    y_half = (y_interp[:,:-1]+y_interp[:,1:])/2
    
    for i in range(np.shape(x_half)[1]):
        x_new.append(x_interp[:,i])
        x_new.append(x_half[:,i])
        y_new.append(y_interp[:,i])
        y_new.append(y_half[:,i])
    
    x_new.append(x_interp[:,-1])
    y_new.append(y_interp[:,-1])

    x_interp = np.array(x_new).T
    y_interp = np.array(y_new).T

x_new = np.array(x_new).T
y_new = np.array(y_new).T
x_new = np.concatenate([x[:,:nmax],x_new],axis=-1)
y_new = np.concatenate([y[:,:nmax],y_new],axis=-1)

print(np.shape(x_new))


rho = np.load(f'{original_path}rho.npy')
p = np.load(f'{original_path}p.npy')
u = np.load(f'{original_path}u.npy')
v = np.load(f'{original_path}v.npy')
w = np.load(f'{original_path}w.npy')
T = np.load(f'{original_path}T.npy')

rho_interp = rho[:,nmax:]
p_interp = p[:,nmax:]
u_interp = u[:,nmax:]
v_interp = v[:,nmax:]
w_interp = w[:,nmax:]
T_interp = T[:,nmax:]

for nt in range(cut_half):
    rho_new = []
    p_new = []
    u_new = []
    v_new = []
    w_new = []
    T_new = []

    rho_half = (rho_interp[:,:-1]+rho_interp[:,1:])/2
    p_half = (p_interp[:,:-1]+p_interp[:,1:])/2
    u_half = (u_interp[:,:-1]+u_interp[:,1:])/2
    v_half = (v_interp[:,:-1]+v_interp[:,1:])/2
    w_half = (w_interp[:,:-1]+w_interp[:,1:])/2
    T_half = (T_interp[:,:-1]+T_interp[:,1:])/2
    
    for i in range(np.shape(rho_half)[1]):
        rho_new.append(rho_interp[:,i])
        rho_new.append(rho_half[:,i])
        p_new.append(p_interp[:,i])
        p_new.append(p_half[:,i])
        u_new.append(u_interp[:,i])
        u_new.append(u_half[:,i])
        v_new.append(v_interp[:,i])
        v_new.append(v_half[:,i])
        w_new.append(w_interp[:,i])
        w_new.append(w_half[:,i])
        T_new.append(T_interp[:,i])
        T_new.append(T_half[:,i])
    
    rho_new.append(rho_interp[:,-1])
    p_new.append(p_interp[:,-1])
    u_new.append(u_interp[:,-1])
    v_new.append(v_interp[:,-1])
    w_new.append(w_interp[:,-1])
    T_new.append(T_interp[:,-1])

    rho_interp = np.array(rho_new).T
    p_interp = np.array(p_new).T
    u_interp = np.array(u_new).T
    v_interp = np.array(v_new).T
    w_interp = np.array(w_new).T
    T_interp = np.array(T_new).T

rho_new = np.array(rho_new).T
p_new = np.array(p_new).T
u_new = np.array(u_new).T
v_new = np.array(v_new).T
w_new = np.array(w_new).T
T_new = np.array(T_new).T
rho_new = np.concatenate([rho[:,:nmax],rho_new],axis=-1)
p_new = np.concatenate([p[:,:nmax],p_new],axis=-1)
u_new = np.concatenate([u[:,:nmax],u_new],axis=-1)
v_new = np.concatenate([v[:,:nmax],v_new],axis=-1)
w_new = np.concatenate([w[:,:nmax],w_new],axis=-1)
T_new = np.concatenate([T[:,:nmax],T_new],axis=-1)





if plot_fields:
    fig, ax = plt.subplots()
    ax.contourf(x_new,y_new,rho_new, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}rho.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_new,y_new,u_new, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}u.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_new,y_new,v_new, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}v.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_new,y_new,p_new, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}p.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_new,y_new,T_new, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}T.png')
    plt.close()



np.save(f'{savepath}/rho.npy',rho_new)
np.save(f'{savepath}/p.npy',p_new)
np.save(f'{savepath}/u.npy',u_new)
np.save(f'{savepath}/v.npy',v_new)
np.save(f'{savepath}/w.npy',w_new)
np.save(f'{savepath}/T.npy',T_new)
np.save(f'{savepath}/x.npy',x_new)
np.save(f'{savepath}/y.npy',y_new)

