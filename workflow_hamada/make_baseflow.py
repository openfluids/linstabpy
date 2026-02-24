from scipy.io import loadmat, savemat
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from pathlib import Path
#from Heaviside_velocity_divergence import heaviside_compression_waves
from sponge_generator import generate_sponge
#from explicit_filter_4thOrder_Lele import fourth_order_filter
#from GridMetrics2D.MetricClass2D import MetricClass
#from GridMetrics2D.fd_compact_10th_nonperiodic import compact_10th_nonperiodic


print("Creating base flow ...")

plot_fields = True

#### these variables will be modified by the bash script
savepath = "Results_St0.1_beta0.0_NeumannOutBotTemp_SS_200-500_100eig_4thOrder_LargeMesh_Half/NoSponge_NoShock_NoFilter/"
casepath = "../InterpolateBaseFlow/"
casename = "suction_side"
matlabpath = "SolverMartini"
### define sponge parameters
#### top, right, bot, left direction 
sponge_flag = False
sponge_type = "constant"
sponge_pts = [0,0,0,0]
sponge_amp = [0,0,0,0]


## flag to see if heaviside or specific shock sponge is used
shock_flag = False
threshold_heaviside = -8.0
amplitude_shock = 0.01

### filter parameters
filter_flag = False
#how many times the filter will pass in x and y directions
nfilter = 20
#minimum point in y direction which the filter will pass
nyminfilter = 0


################### The code
### create the saving paths
Path(f"../../{savepath}/EigModes/").mkdir(parents=True,exist_ok=True)
Path(f"../../{savepath}/FigEigModes/").mkdir(parents=True,exist_ok=True)
Path(f"../../{savepath}/ResolventModes/").mkdir(parents=True,exist_ok=True)
Path(f"../../{savepath}/FigResolventModes/").mkdir(parents=True,exist_ok=True)


### read the flow and mesh variables
x = np.load(f'{casepath}{casename}/x.npy')
y = np.load(f'{casepath}{casename}/y.npy')

rho = np.load(f'{casepath}{casename}/rho.npy')
u = np.load(f'{casepath}{casename}/u.npy')
v = np.load(f'{casepath}{casename}/v.npy')
w = np.load(f'{casepath}{casename}/w.npy')
T = np.load(f'{casepath}{casename}/T.npy')

##create metrics object
#metrics = MetricClass(x=x,
#                      y=y,
#                      scheme_metrics=compact_10th_nonperiodic)

# evaluate metric terms
#metrics.eval_metric_terms()
#detJ = metrics.det_jacobian
#dmdx = metrics.comp_to_phys
#dxdm = metrics.phys_to_comp

#ut_u = dmdx[:,:,1,1]/np.sqrt(dmdx[:,:,0,0]**2+dmdx[:,:,1,1]**2)
#ut_v = -dmdx[:,:,1,0]/np.sqrt(dmdx[:,:,0,0]**2+dmdx[:,:,1,1]**2)

#un_u = dmdx[:,:,1,0]/np.sqrt(dmdx[:,:,0,0]**2+dmdx[:,:,1,1]**2)
#un_v = dmdx[:,:,1,1]/np.sqrt(dmdx[:,:,0,0]**2+dmdx[:,:,1,1]**2)

#np.save(f'../../{savepath}/ut_u.npy',ut_u)
#np.save(f'../../{savepath}/ut_v.npy',ut_v)
#np.save(f'../../{savepath}/un_u.npy',un_u)
#np.save(f'../../{savepath}/un_v.npy',un_v)

#ut_u = np.ravel(ut_u,order='F')
#ut_v = np.ravel(ut_v,order='F')
#un_u = np.ravel(un_u,order='F')
#un_v = np.ravel(un_v,order='F')


### apply Ducros
if shock_flag:
    Dx = loadmat(f'{casepath}{casename}/Dx.mat')['Dx']
    Dy = loadmat(f'{casepath}{casename}/Dy.mat')['Dy']
    sponge = heaviside_compression_waves(u,v,Dx,Dy,threshold_heaviside)*amplitude_shock
else:
    sponge = np.zeros_like(x)

if sponge_flag:
    sponge = generate_sponge(sponge,sponge_type,sponge_pts,sponge_amp)

# print(sponge[0,-sponge_pts[0]:])
# print(sponge[-sponge_pts[1]:,0])
# print(sponge[0,:sponge_pts[2]])
# print(sponge[:sponge_pts[3],0])

#### filter 
if filter_flag:

    rho_cont = detJ*rho
    T_cont = detJ*T

    u_cont = (u*dmdx[:,:,0,0] +v*dmdx[:,:,0,1])*detJ
    v_cont = (u*dmdx[:,:,1,0] +v*dmdx[:,:,1,1])*detJ

    for i in range(nfilter):
        rho_cont[:,nyminfilter:] = fourth_order_filter(rho_cont[:,nyminfilter:],'x')
        u_cont[:,nyminfilter:] = fourth_order_filter(u_cont[:,nyminfilter:],'x')
        v_cont[:,nyminfilter:] = fourth_order_filter(v_cont[:,nyminfilter:],'x')
        T_cont[:,nyminfilter:] = fourth_order_filter(T_cont[:,nyminfilter:],'x')
        
        rho_cont[:,nyminfilter:] = fourth_order_filter(rho_cont[:,nyminfilter:],'y')
        u_cont[:,nyminfilter:] = fourth_order_filter(u_cont[:,nyminfilter:],'y')
        v_cont[:,nyminfilter:] = fourth_order_filter(v_cont[:,nyminfilter:],'y')
        T_cont[:,nyminfilter:] = fourth_order_filter(T_cont[:,nyminfilter:],'y')

    rho = rho_cont/detJ
    T = T_cont/detJ
    u_cont = u_cont/detJ
    v_cont = v_cont/detJ
    u = u_cont*dxdm[:,:,0,0] +v_cont*dxdm[:,:,0,1]
    v = u_cont*dxdm[:,:,1,0] +v_cont*dxdm[:,:,1,1]


    # for i in range(nfilter):
    #     rho[:,nyminfilter:] = fourth_order_filter(rho[:,nyminfilter:],'x')
    #     u[:,nyminfilter:] = fourth_order_filter(u[:,nyminfilter:],'x')
    #     v[:,nyminfilter:] = fourth_order_filter(v[:,nyminfilter:],'x')
    #     T[:,nyminfilter:] = fourth_order_filter(T[:,nyminfilter:],'x')
        
    #     rho[:,nyminfilter:] = fourth_order_filter(rho[:,nyminfilter:],'y')
    #     u[:,nyminfilter:] = fourth_order_filter(u[:,nyminfilter:],'y')
    #     v[:,nyminfilter:] = fourth_order_filter(v[:,nyminfilter:],'y')
    #     T[:,nyminfilter:] = fourth_order_filter(T[:,nyminfilter:],'y')


savemat(f'../../{matlabpath}/BaseFlows/base_flow.mat', {'rho' : rho, 'u' : u, 'v' :v , 'w' : w, 'T' : T})
savemat(f'../../{matlabpath}/mesh.mat', {'x' : x, 'y' : y})
savemat(f'../../{matlabpath}/sponge.mat', {'sponge_aux' : sponge})
#savemat(f'../../{matlabpath}/idx_ut_un.mat', {'ut_u' : ut_u, 'ut_v' : ut_v, 'un_u' : un_u, 'un_v' : un_v})

np.save(f'../../{savepath}/rho.npy',rho)
np.save(f'../../{savepath}/u.npy',u)
np.save(f'../../{savepath}/v.npy',v)
np.save(f'../../{savepath}/w.npy',w)
np.save(f'../../{savepath}/T.npy',T)
np.save(f'../../{savepath}/x.npy',x)
np.save(f'../../{savepath}/y.npy',y)
np.save(f'../../{savepath}/sponge.npy',sponge)

if plot_fields:
    fig, ax = plt.subplots()
    ax.contourf(x,y,rho, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'../../{savepath}rho.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x,y,u, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'../../{savepath}u.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x,y,v, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'../../{savepath}v.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x,y,T, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'../../{savepath}T.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.contourf(x,y,sponge, cmap = 'coolwarm')
    ax.set_aspect('equal')
    plt.savefig(f'../../{savepath}sponge.png')
    plt.close()
