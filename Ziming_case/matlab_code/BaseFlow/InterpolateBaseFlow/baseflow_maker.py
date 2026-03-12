from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from pathlib import Path


print("Creating base flow ...")
#### these variables will be modified by the bash script
savepath = "../../SolverMartini"
casepath = "./half_mesh"

### read the flow and mesh variables
x = np.load(f"{casepath}/x.npy")
y = np.load(f"{casepath}/y.npy")

rho = np.load(f"{casepath}/rho.npy")
u = np.load(f"{casepath}/u.npy")
v = np.load(f"{casepath}/v.npy")
w = np.load(f"{casepath}/w.npy")
T = np.load(f"{casepath}/T.npy")
sponge = np.zeros(np.shape(x))


savemat(f'{savepath}/BaseFlows/base_flow.mat', {'rho' : rho, 'u' : u, 'v' :v , 'w' : w, 'T' : T})
savemat(f'{savepath}/mesh.mat', {'x' : x, 'y' : y})
savemat(f'{savepath}/sponge.mat', {'sponge_aux' : sponge})

