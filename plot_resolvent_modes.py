from scipy.io import loadmat, savemat
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter
import numpy as np
cmap = plt.get_cmap('coolwarm') 
matplotlib.rcParams.update({'font.size': 22, 'font.family': 'serif',"font.serif": ["Computer Modern"], 'text.usetex': True})

print("Writing resolvent images ...")

savepath = "./resolvent"

data = np.load("example_6_hamada.npz")

beta = data["beta"]
St = data["St"]

x = data["X"]
y = data["Y"]
u_velocity = data["U"]
sponge = data["sponge"]
sponge_contour = 1e-6
sponge_flag = False
if np.max(sponge) >= 1e-6:
    sponge_flag = True


#import the resolvent gains
gains =  np.ndarray.flatten(data["gains"])
gains_orig = np.ndarray.flatten(data["gains"])
npts = x.shape[0]*x.shape[1]

gain_total = np.sum(gains)

gain_sum = np.zeros(np.shape(gains))
for i in range(len(gains)):
    if i == 0:
        gain_sum[i] = gains[i]
    else:
        gain_sum[i] = gains[i] +gain_sum[i-1]

gain_sum /= gain_total
gains /= gain_total

fig, ax = plt.subplots()
ax.bar(range(1,len(gains)+1,1),gains,color='blue')
ax.plot(range(1,len(gains)+1,1),gain_sum,'-k')
ax.set_xlabel('Mode')
ax.set_ylabel('Normalized Gain')
plt.savefig(f'./{savepath}/gains.png',dpi=600, bbox_inches='tight')
plt.close()

levels = np.linspace(-1,1,11)*5e-1
levels = np.delete(levels,np.where(levels == 0)[:])
print(levels)


forcing = data["forcings"].T
response = data["responses"].T

idx_rho = data["idx_rho_j"]
idx_u = data["idx_u_j"]
idx_v = data["idx_v_j"]
idx_w = data["idx_w_j"]
idx_T = data["idx_T_j"]

for i in range(len(gains)):    
    
    rho_f = forcing[i,idx_rho]
    u_f = forcing[i,idx_u]
    v_f = forcing[i,idx_v]
    w_f = forcing[i,idx_w]
    T_f = forcing[i,idx_T]

    rho_r = response[i,idx_rho]
    u_r = response[i,idx_u]
    v_r = response[i,idx_v]
    w_r = response[i,idx_w]
    T_r = response[i,idx_T]

    max_rho = np.max([np.max(np.abs(rho_f.real)),np.max(np.abs(rho_r.real))])
    max_u = np.max([np.max(np.abs(u_f.real)),np.max(np.abs(u_r.real))])
    max_v = np.max([np.max(np.abs(v_f.real)),np.max(np.abs(v_r.real))])
    max_T = np.max([np.max(np.abs(T_f.real)),np.max(np.abs(T_r.real))])

    rho_f /=max_rho
    rho_r /=max_rho

    u_f /=max_u
    u_r /=max_u

    v_f /=max_v
    v_r /=max_v

    T_f /=max_T
    T_r /=max_T

    if beta > 1e-8:
        max_w = np.max([np.max(np.abs(w_f.real)),np.max(np.abs(w_r.real))])
        w_f /=max_w
        w_r /=max_w


    rho_f = rho_f.reshape((x.shape[0],x.shape[1]),order="F")
    u_f = u_f.reshape((x.shape[0],x.shape[1]),order="F")
    v_f = v_f.reshape((x.shape[0],x.shape[1]),order="F")
    w_f = w_f.reshape((x.shape[0],x.shape[1]),order="F")
    T_f = T_f.reshape((x.shape[0],x.shape[1]),order="F")

    rho_r = rho_r.reshape((x.shape[0],x.shape[1]),order="F")
    u_r = u_r.reshape((x.shape[0],x.shape[1]),order="F")
    v_r = v_r.reshape((x.shape[0],x.shape[1]),order="F")
    w_r = w_r.reshape((x.shape[0],x.shape[1]),order="F")
    T_r = T_r.reshape((x.shape[0],x.shape[1]),order="F")
    

    
    ######### plot balance
    fig = plt.figure(figsize=(12,16))
    (ax1,ax11), (ax2,ax21), (ax3,ax31), (ax4,ax41), (ax5,ax51) = fig.subplots(5,2,sharey=True)

    axes_list = [ax1, ax11, ax2, ax21, ax3, ax31, ax4, ax41, ax5, ax51]

    ax1.contourf(x,y,rho_f.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax1.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax1.set_aspect('equal')
    ax1.set_ylabel("$\\rho$")

    ax11.contourf(x,y,rho_r.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax11.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax11.set_aspect('equal')


    ax2.contourf(x,y,u_f.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax2.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax2.set_aspect('equal')
    ax2.set_ylabel("$u$")

    ax21.contourf(x,y,u_r.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax21.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax21.set_aspect('equal')


    ax3.contourf(x,y,v_f.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax3.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax3.set_aspect('equal')
    ax3.set_ylabel("$v$")

    ax31.contourf(x,y,v_r.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax31.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax31.set_aspect('equal')


    ax4.contourf(x,y,w_f.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax4.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax4.set_aspect('equal')
    ax4.set_ylabel("$w$")

    ax41.contourf(x,y,w_r.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax41.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax41.set_aspect('equal')


    ax5.contourf(x,y,T_f.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax5.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax5.set_aspect('equal')
    ax5.set_ylabel("$T$")
    ax5.set_xlabel("Forcing")

    CSf = ax51.contourf(x,y,T_r.real, cmap = 'coolwarm',levels=levels,extend='both')
    ax51.contour(x,y,u_velocity,levels=[-1e-6],colors=['blue'],linewidths=[1])
    ax51.set_aspect('equal')
    ax51.set_xlabel("Response")

    if sponge_flag == True:
        ax1.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax11.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax2.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax21.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax3.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax31.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax4.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax41.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax5.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])
        ax51.contour(x,y,sponge,levels=[sponge_contour],colors=['black'],linewidths=[1])


    fig.suptitle(f"""$\\beta = {beta}, St_L = {St}$ \n {gains[i]*100:.3f}$\\%$ of the total gain, {gains_orig[i]:.7e}""",y=0.925)
    cbar = fig.colorbar(CSf, ax=axes_list, 
                        orientation='vertical',
                        location='right',
                        shrink=0.6,  # scale down colorbar height
                        pad=0.02,
                        aspect=30,
                        anchor=(1.5, 0.5))
    cbar.set_label("Scale - real part, normalized by max(forcing,response)")

    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    plt.savefig(f'./{savepath}/mode_{i}.png',dpi=600, bbox_inches='tight')
    plt.close()

    print(i)
