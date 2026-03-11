import numpy as np

def eval_velocity_divergence(u:np.ndarray, v:np.ndarray, Dx: np.ndarray, Dy: np.ndarray) -> np.ndarray:

    u_flat = np.ndarray.flatten(u,order="C")
    v_flat = np.ndarray.flatten(v,order="C")

    div_u = Dx*u_flat +Dy*v_flat 
    div_u = div_u.reshape((u.shape[0],u.shape[1]),order="C")

    return div_u

def eval_velocity_rotational(u:np.ndarray, v:np.ndarray, Dx: np.ndarray, Dy: np.ndarray) -> np.ndarray:

    u_flat = np.ndarray.flatten(u,order="C")
    v_flat = np.ndarray.flatten(v,order="C")

    rot_u = np.sqrt((Dx*v_flat -Dy*u_flat)**2)
    rot_u = rot_u.reshape((u.shape[0],u.shape[1]),order="C")

    return rot_u

#### function to obtain the Ducros sensor
def ducros_sensor(u:np.ndarray, v:np.ndarray, Dx: np.ndarray, Dy: np.ndarray, threshold: float) -> np.ndarray:

    div_u = eval_velocity_divergence(u,v,Dx,Dy)
    rot_u = eval_velocity_rotational(u,v,Dx,Dy)

    aux = np.abs(div_u**2/(div_u**2 +rot_u**2 +1e-14))

    nx, ny = np.shape(aux)[:]

    ducros = np.zeros(np.shape(aux))

    for i in range(nx):
        for j in range(ny):
            if aux[i,j] > threshold:
                ducros[i,j] = 1.0

    return ducros


def ducros_sensor_modified(x:np.ndarray, y:np.ndarray, u:np.ndarray, v:np.ndarray, Dx: np.ndarray, Dy: np.ndarray, scale: float, threshold: float) -> np.ndarray:

    mag_u = np.sqrt(u**2+v**2)

    div_u = eval_velocity_divergence(u,v,Dx,Dy)
    
    dx0 = np.abs(np.diff(x,axis=0))
    dy0 = np.abs(np.diff(y,axis=0))
    dx1 = np.abs(np.diff(x,axis=1))
    dy1 = np.abs(np.diff(y,axis=1))

    dx = np.zeros(np.shape(u))
    dy = np.zeros(np.shape(u))
    dx[:-1,:] = np.sqrt(dx0**2+dy0**2)
    dy[:,:-1] = np.sqrt(dx1**2+dy1**2)
    dx[-1,:] = dx[-2,:]
    # dx[:,-1] = dx[:,-2]
    # dy[-1,:] = dy[-2,:]
    dy[:,-1] = dy[:,-2]

    freq_x = scale*(mag_u/dx)
    freq_y = scale*(mag_u/dy)
                               
    nx, ny = np.shape(u)[:]

    ducros = np.zeros(np.shape(u))

    for i in range(nx):
        for j in range(ny):
            aux = np.abs(div_u[i,j]**2/(div_u[i,j]**2 +np.max([freq_x[i,j],freq_y[i,j]])**2 + 1e-14))
            if aux > threshold:
                
                ducros[i,j] = 1.0

    return ducros


def ducros_sensor_kawai(u:np.ndarray, v:np.ndarray, Dx: np.ndarray, Dy: np.ndarray, threshold_heaviside: float, threshold_ducros) -> np.ndarray:


    div_u = eval_velocity_divergence(u,v,Dx,Dy)
    rot_u = eval_velocity_rotational(u,v,Dx,Dy)

    nx, ny = np.shape(u)[:]

    heaviside = np.zeros(np.shape(u))
    
    for i in range(nx):
        for j in range(ny):
            if div_u[i,j] <= threshold_heaviside:
                heaviside[i,j] = 1.0
    
    aux = heaviside*div_u**2/(div_u**2 +np.abs(rot_u)**2 +1e-14)
    ducros = np.zeros(np.shape(aux))

    for i in range(nx):
        for j in range(ny):
            if aux[i,j] > threshold_ducros:
                ducros[i,j] = 1.0

    return ducros