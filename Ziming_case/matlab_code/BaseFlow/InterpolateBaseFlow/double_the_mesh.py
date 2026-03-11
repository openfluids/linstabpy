from scipy.io import loadmat, savemat
# from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from pathlib import Path

savepath = "./double_mesh/"
Path(savepath).mkdir(parents=True,exist_ok=True)

original_path = "./original_mesh/"
plot_fields = False


x = np.load(f"{original_path}x.npy")
y = np.load(f"{original_path}y.npy")

nxi, nyi = np.shape(x)
x_double = np.zeros((2*nxi-1,2*nyi-1))
y_double = np.zeros((2*nxi-1,2*nyi-1))

xx_int = (x[:-1,:]+x[1:,:])/2
xy_int = (x[:,:-1]+x[:,1:])/2
yx_int = (y[:-1,:]+y[1:,:])/2
yy_int = (y[:,:-1]+y[:,1:])/2
xxy_int = (xx_int[:,:-1]+xx_int[:,1:])/2
yxy_int = (yx_int[:,:-1]+yx_int[:,1:])/2

print(np.shape(x),np.shape(x_double),np.shape(xx_int),np.shape(xy_int),np.shape(xxy_int))
for i in range(nxi):
    for j in range(nyi):
        ## put the original points 
        x_double[2*i,2*j] = x[i,j]  
        y_double[2*i,2*j] = y[i,j]  
for i in range(nxi-1):
    for j in range(nyi):
        ## add the interpolated points in x direction       
        x_double[2*i+1,2*j] = xx_int[i,j]  
        y_double[2*i+1,2*j] = yx_int[i,j]  
for i in range(nxi):
    for j in range(nyi-1):
        # ## add the interpolated points in y direction
        x_double[2*i,2*j+1] = xy_int[i,j]  
        y_double[2*i,2*j+1] = yy_int[i,j] 
for i in range(nxi-1):
    for j in range(nyi-1): 
        ## add the points in the middle (x and y interpolation)
        x_double[2*i+1,2*j+1] = xxy_int[i,j]  
        y_double[2*i+1,2*j+1] = yxy_int[i,j]  


rho = np.load(f'{original_path}rho.npy')
p = np.load(f'{original_path}p.npy')
u = np.load(f'{original_path}u.npy')
v = np.load(f'{original_path}v.npy')
w = np.load(f'{original_path}w.npy')
T = np.load(f'{original_path}T.npy')


rho_double = np.zeros((2*nxi-1,2*nyi-1))
p_double = np.zeros((2*nxi-1,2*nyi-1))
u_double = np.zeros((2*nxi-1,2*nyi-1))
v_double = np.zeros((2*nxi-1,2*nyi-1))
w_double = np.zeros((2*nxi-1,2*nyi-1))
T_double = np.zeros((2*nxi-1,2*nyi-1))


rhox_int = (rho[:-1,:]+rho[1:,:])/2
rhoy_int = (rho[:,:-1]+rho[:,1:])/2
rhoxy_int = (rhox_int[:,:-1]+rhox_int[:,1:])/2

px_int = (p[:-1,:]+p[1:,:])/2
py_int = (p[:,:-1]+p[:,1:])/2
pxy_int = (px_int[:,:-1]+px_int[:,1:])/2

ux_int = (u[:-1,:]+u[1:,:])/2
uy_int = (u[:,:-1]+u[:,1:])/2
uxy_int = (ux_int[:,:-1]+ux_int[:,1:])/2

vx_int = (v[:-1,:]+v[1:,:])/2
vy_int = (v[:,:-1]+v[:,1:])/2
vxy_int = (vx_int[:,:-1]+vx_int[:,1:])/2

wx_int = (w[:-1,:]+w[1:,:])/2
wy_int = (w[:,:-1]+w[:,1:])/2
wxy_int = (wx_int[:,:-1]+wx_int[:,1:])/2

Tx_int = (T[:-1,:]+T[1:,:])/2
Ty_int = (T[:,:-1]+T[:,1:])/2
Txy_int = (Tx_int[:,:-1]+Tx_int[:,1:])/2


for i in range(nxi):
    for j in range(nyi):
        ## put the original points 
        rho_double[2*i,2*j] = rho[i,j]  
        p_double[2*i,2*j] = p[i,j]  
        u_double[2*i,2*j] = u[i,j]  
        v_double[2*i,2*j] = v[i,j]  
        w_double[2*i,2*j] = w[i,j]  
        T_double[2*i,2*j] = T[i,j]  
for i in range(nxi-1):
    for j in range(nyi):
        ## add the interpolated points in x direction       
        rho_double[2*i+1,2*j] = rhox_int[i,j]  
        p_double[2*i+1,2*j] = px_int[i,j]  
        u_double[2*i+1,2*j] = ux_int[i,j]  
        v_double[2*i+1,2*j] = vx_int[i,j]  
        w_double[2*i+1,2*j] = wx_int[i,j]  
        T_double[2*i+1,2*j] = Tx_int[i,j]  
for i in range(nxi):
    for j in range(nyi-1):
        # ## add the interpolated points in y direction
        rho_double[2*i,2*j+1] = rhoy_int[i,j]  
        p_double[2*i,2*j+1] = py_int[i,j] 
        u_double[2*i,2*j+1] = uy_int[i,j] 
        v_double[2*i,2*j+1] = vy_int[i,j] 
        w_double[2*i,2*j+1] = wy_int[i,j] 
        T_double[2*i,2*j+1] = Ty_int[i,j] 
for i in range(nxi-1):
    for j in range(nyi-1): 
        ## add the points in the middle (x and y interpolation)
        rho_double[2*i+1,2*j+1] = rhoxy_int[i,j]  
        p_double[2*i+1,2*j+1] = pxy_int[i,j]  
        u_double[2*i+1,2*j+1] = uxy_int[i,j]  
        v_double[2*i+1,2*j+1] = vxy_int[i,j]  
        w_double[2*i+1,2*j+1] = wxy_int[i,j]  
        T_double[2*i+1,2*j+1] = Txy_int[i,j]  


if plot_fields:
    fig, ax = plt.subplots()
    ax.contourf(x_double,y_double,rho_double, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}rho.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_double,y_double,u_double, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}u.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_double,y_double,v_double, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}v.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_double,y_double,p_double, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}p.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x_double,y_double,T_double, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'{savepath}T.png')
    plt.close()

np.save(f'{savepath}/rho.npy',rho_double)
np.save(f'{savepath}/p.npy',p_double)
np.save(f'{savepath}/u.npy',u_double)
np.save(f'{savepath}/v.npy',v_double)
np.save(f'{savepath}/w.npy',w_double)
np.save(f'{savepath}/T.npy',T_double)
np.save(f'{savepath}/x.npy',x_double)
np.save(f'{savepath}/y.npy',y_double)

