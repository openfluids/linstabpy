from scipy.io import loadmat, savemat
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter
import numpy as np
cmap = plt.get_cmap('coolwarm') 
matplotlib.rcParams.update({'font.size': 22, 'font.family': 'serif',"font.serif": ["Computer Modern"], 'text.usetex': True})


print("Writing eigenvalue images ...")

savepath = "Results_St0.0_beta19.2_NeumannOutBotTemp/NoSponge_NoShock_NoFilter/"
title = "NoSponge_NoShock_NoFilter"

beta = 19.2
St = 0.0

x = np.load(f'./{savepath}/x.npy')
y = np.load(f'./{savepath}/y.npy')
u_velocity = np.load(f'./{savepath}/u.npy')
sponge = np.load(f'./{savepath}/sponge.npy')
sponge_contour = 0.1
sponge_flag = True
if np.max(sponge) >= 0.1:
    sponge_flag = True


#import the spectrum
omega =  np.diag(np.array(loadmat(f'./{savepath}/EigModes/spectrum')['omega']))
npts = x.shape[0]*x.shape[1]


levels = np.linspace(-1,1,11)*5e-1
levels = np.delete(levels,np.where(levels == 0)[:])

for i in range(1,len(omega)+1):

    rho = np.array(loadmat(f'./{savepath}/EigModes/eigenvector_{i}.mat')['rho'])
    u = np.array(loadmat(f'./{savepath}/EigModes/eigenvector_{i}.mat')['u'])
    v = np.array(loadmat(f'./{savepath}/EigModes/eigenvector_{i}.mat')['v'])
    w = np.array(loadmat(f'./{savepath}/EigModes/eigenvector_{i}.mat')['w'])
    T = np.array(loadmat(f'./{savepath}/EigModes/eigenvector_{i}.mat')['T'])

    max_rho = np.max(np.abs(rho.real))
    max_u = np.max(np.abs(u.real))
    max_v = np.max(np.abs(v.real))
    max_T = np.max(np.abs(T.real))

    rho /=max_rho
    u /=max_u
    v /=max_v
    T /=max_T

    if beta > 1e-8:
        max_w = np.max(np.abs(w.real))
        w /=max_w

    rho = rho.reshape((x.shape[0],x.shape[1]),order="C")
    u = u.reshape((x.shape[0],x.shape[1]),order="C")
    v = v.reshape((x.shape[0],x.shape[1]),order="C")
    w = w.reshape((x.shape[0],x.shape[1]),order="C")
    T = T.reshape((x.shape[0],x.shape[1]),order="C")

    ######### plot balance
    fig = plt.figure(figsize=(16,10))
    (ax1,ax11), (ax2,ax21), (ax3,ax31) = fig.subplots(3,2)

    axes_list = [ax1,ax11, ax2, ax21, ax3, ax31]

    ax1.scatter(omega.real,omega.imag,color='black',s=30)
    ax1.scatter(omega.real[i-1],omega.imag[i-1],color='blue',s=60)
    ax1.axhline(y=0,color='gray',linewidth=1)
    ax1.set_xlabel('$\\omega_r$',labelpad=-10)
    ax1.set_ylabel('$\\omega_i$')
    ax1.set_title(f"{omega[i-1]}")

    CSf = ax11.contourf(x,y,rho.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax11.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax11.set_aspect('equal')
    ax11.set_ylabel(r"$R(\rho/\rho_{max})$")

    CSf = ax2.contourf(x,y,u.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax2.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax2.set_aspect('equal')
    ax2.set_ylabel(r"$R(u/u_{max})$")

    CSf = ax21.contourf(x,y,v.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax21.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax21.set_aspect('equal')
    ax21.set_ylabel(r"$R(v/v_{max})$")

    CSf = ax3.contourf(x,y,w.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax3.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax3.set_aspect('equal')
    ax3.set_ylabel(r"$R(w/w_{max})$")

    CSf = ax31.contourf(x,y,T.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax31.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax31.set_aspect('equal')
    ax31.set_ylabel(r"$R(T/T_{max})$")

    if sponge_flag == True:
        ax11.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax2.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax21.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax3.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax31.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])


    fig.suptitle(f"$\\beta = {beta}, St_L = {St}$ - {title}")
    cbar = fig.colorbar(CSf, ax=axes_list, 
                        orientation='vertical',
                        location='right',
                        shrink=0.6,  # scale down colorbar height
                        pad=0.02,
                        aspect=30,
                        anchor=(0.1, 0.5))
    cbar.set_label("Scale - real part, normalized by max(forcing,response)")

    # plt.subplots_adjust(hspace=0.25,wspace=0.1)
    plt.savefig(f'./{savepath}/FigEigModes/eig_{i}.png',dpi=600, bbox_inches='tight')
    plt.close()
    
    print(i)
