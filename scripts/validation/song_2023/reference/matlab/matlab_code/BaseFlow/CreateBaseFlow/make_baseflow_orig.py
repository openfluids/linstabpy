from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from pathlib import Path
from Heaviside_velocity_divergence import heaviside_compression_waves
from sponge_generator import generate_sponge
from explicit_filter_4thOrder_Lele import fourth_order_filter


print("Creating base flow ...")
#### these variables will be modified by the bash script
savepath = "Results_St0.0_beta19.2_NeumannOut/NoSponge_NoShock_NoFilter/"
casepath = "../InterpolateBaseFlow/"
casename = "original_mesh"
matlabpath = "SolverMartini"
### define sponge parameters
#### top, right, bot, left direction 
sponge_flag = False
sponge_pts = [0,0,0,0]
sponge_amp = [0,0,0,0]


## flag to see if heaviside or specific shock sponge is used
shock_flag = False
threshold_heaviside = -1.0
amplitude_shock = 0

### filter parameters
filter_flag = False
#how many times the filter will pass in x and y directions
nfilter = 0
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


### apply Ducros
if shock_flag:
    Dx = loadmat(f'{casepath}{casename}/Dx.mat')['Dx']
    Dy = loadmat(f'{casepath}{casename}/Dy.mat')['Dy']
    sponge = heaviside_compression_waves(u,v,Dx,Dy,threshold_heaviside)*amplitude_shock
else:
    sponge = np.zeros_like(x)

if sponge_flag:
    sponge = generate_sponge(sponge,sponge_pts,sponge_amp)


#### filter 
if filter_flag:
    for i in range(nfilter):
        rho[:,nyminfilter:] = fourth_order_filter(rho[:,nyminfilter:],'x')
        u[:,nyminfilter:] = fourth_order_filter(u[:,nyminfilter:],'x')
        v[:,nyminfilter:] = fourth_order_filter(v[:,nyminfilter:],'x')
        T[:,nyminfilter:] = fourth_order_filter(T[:,nyminfilter:],'x')
        
        rho[:,nyminfilter:] = fourth_order_filter(rho[:,nyminfilter:],'y')
        u[:,nyminfilter:] = fourth_order_filter(u[:,nyminfilter:],'y')
        v[:,nyminfilter:] = fourth_order_filter(v[:,nyminfilter:],'y')
        T[:,nyminfilter:] = fourth_order_filter(T[:,nyminfilter:],'y')


savemat(f'../../{matlabpath}/BaseFlows/base_flow.mat', {'rho' : rho, 'u' : u, 'v' :v , 'w' : w, 'T' : T})
savemat(f'../../{matlabpath}/mesh.mat', {'x' : x, 'y' : y})
savemat(f'../../{matlabpath}/sponge.mat', {'sponge_aux' : sponge})

np.save(f'../../{savepath}/rho.npy',rho)
np.save(f'../../{savepath}/u.npy',u)
np.save(f'../../{savepath}/v.npy',v)
np.save(f'../../{savepath}/w.npy',w)
np.save(f'../../{savepath}/T.npy',T)
np.save(f'../../{savepath}/x.npy',x)
np.save(f'../../{savepath}/y.npy',y)
np.save(f'../../{savepath}/sponge.npy',sponge)
