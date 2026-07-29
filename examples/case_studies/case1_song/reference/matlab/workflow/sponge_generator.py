import numpy as np


def generate_sponge(
    sponge: np.ndarray, sponge_type: str, pts_dir: np.ndarray, amp_dir: np.ndarray
) -> np.ndarray:

    if sponge_type == "constant":
        sponge = constant_sponge(sponge, pts_dir, amp_dir)

    if sponge_type == "parabolic":
        sponge = parabolic_sponge(sponge, pts_dir, amp_dir)

    if sponge_type == "localized":
        sponge = localized_sponge(sponge, pts_dir, amp_dir)

    return sponge


def constant_sponge(sponge: np.ndarray, pts_dir: np.ndarray, amp_dir: np.ndarray) -> np.ndarray:

    ### pts_dir, amp_dir: top, right, bot, left direction
    ## top sponge
    if pts_dir[0] != 0:
        sponge[:, -pts_dir[0] :] = amp_dir[0]
    ##right sponge
    if pts_dir[1] != 0:
        sponge[-pts_dir[1] :, :] = amp_dir[1]
    ##bottom sponge
    if pts_dir[2] != 0:
        sponge[:, : pts_dir[2]] = amp_dir[2]
    ##left sponge
    if pts_dir[3] != 0:
        sponge[: pts_dir[3], :] = amp_dir[3]

    return sponge


def parabolic_sponge(sponge: np.ndarray, pts_dir: np.ndarray, amp_dir: np.ndarray) -> np.ndarray:

    nx, ny = np.shape(sponge)

    ### pts_dir, amp_dir: top, right, bot, left direction
    ## top sponge
    if pts_dir[0] != 0:
        for i in range(-pts_dir[0], 0):
            sponge[:, i] = amp_dir[0] * (pts_dir[0] + 1 + i) ** 2 / pts_dir[0] ** 2
    ##right sponge
    if pts_dir[1] != 0:
        for i in range(-pts_dir[1], 0):
            sponge[i, :] = amp_dir[1] * (pts_dir[1] + 1 + i) ** 2 / pts_dir[1] ** 2
    ##bottom sponge
    if pts_dir[2] != 0:
        for i in range(0, pts_dir[2] + 1):
            sponge[:, i] = amp_dir[2] * (pts_dir[2] - (i)) ** 2 / pts_dir[2] ** 2
    ##left sponge
    if pts_dir[3] != 0:
        for i in range(0, pts_dir[3] + 1):
            sponge[i, :] = amp_dir[3] * (pts_dir[3] - i) ** 2 / pts_dir[3] ** 2
    return sponge


def localized_sponge(sponge: np.ndarray, pts_dir: np.ndarray, amp_dir: np.ndarray) -> np.ndarray:

    nx, ny = np.shape(sponge)

    sponge = np.ones(np.shape(sponge))

    ### pts_dir, amp_dir: top, right, bot, left direction
    ## top sponge
    if pts_dir[0] != 0:
        sponge[:, : -pts_dir[0]] *= 0
    ##right sponge
    if pts_dir[1] != 0:
        sponge[: -pts_dir[1], :] *= 0
    ##bottom sponge
    if pts_dir[2] != 0:
        sponge[:, pts_dir[2] :] *= 0
    ##left sponge
    if pts_dir[3] != 0:
        sponge[pts_dir[3] :, :] *= 0

    sponge *= amp_dir[0]

    return sponge
