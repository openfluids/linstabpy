import numpy as np

def generate_sponge(sponge: np.ndarray, pts_dir: np.ndarray, amp_dir: np.ndarray) -> np.ndarray:

    ### pts_dir, amp_dir: top, right, bot, left direction 
    ## top sponge
    sponge[:,-pts_dir[0]:] = amp_dir[0]
    ##right sponge
    sponge[-pts_dir[1]:,:] = amp_dir[1]
    ##bottom sponge
    sponge[:,:pts_dir[2]] = amp_dir[2]
    ##left sponge
    sponge[:pts_dir[3],:] = amp_dir[3]

    return sponge

