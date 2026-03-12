import numpy as np

def eval_velocity_divergence(u:np.ndarray, v:np.ndarray, Dx: np.ndarray, Dy: np.ndarray) -> np.ndarray:

    u_flat = np.ndarray.flatten(u,order="C")
    v_flat = np.ndarray.flatten(v,order="C")

    div_u = Dx*u_flat +Dy*v_flat 
    div_u = div_u.reshape((u.shape[0],u.shape[1]),order="C")

    return div_u

def heaviside_compression_waves(u:np.ndarray, v:np.ndarray, Dx: np.ndarray, Dy: np.ndarray, threshold: float) -> np.ndarray:

    div_u = eval_velocity_divergence(u,v,Dx,Dy)
    nx, ny = np.shape(u)[:]

    heaviside = np.zeros(np.shape(u))
    
    for i in range(nx):
        for j in range(ny):
            if div_u[i,j] <= threshold:
                heaviside[i,j] = 1.0
    
    return heaviside

def heaviside_magnitude(u:np.ndarray, v:np.ndarray, Dx: np.ndarray, Dy: np.ndarray, threshold: float) -> np.ndarray:

    div_u = np.abs(eval_velocity_divergence(u,v,Dx,Dy))
    nx, ny = np.shape(u)[:]

    heaviside = np.zeros(np.shape(u))
    
    for i in range(nx):
        for j in range(ny):
            if div_u[i,j] >= threshold:
                heaviside[i,j] = 1.0
    
    return heaviside

