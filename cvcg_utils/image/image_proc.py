import os
import numpy as np
import cv2
from scipy.sparse import csc_array

def get_laplacian(H: int, W: int):
    """
    **Modified from pytorch3d.ops.laplacian**
    
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        n_verts
        faces: tensor of shape (Nf, 2)
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """

    pix_ind_img = np.arange(H*W, dtype=int).reshape(H, W)
    hori_edges = np.stack([pix_ind_img[:, :-1], pix_ind_img[:, 1:]], axis=-1)   # [H, W-1, 2]
    vert_edges = np.stack([pix_ind_img[:-1, :], pix_ind_img[1:, :]], axis=-1)   # [H-1, W, 2]
    all_edges = np.concatenate([hori_edges.reshape(-1, 2), vert_edges.reshape(-1, 2)], axis=0)  # [N_edges, 2]
    # this is already unique given the construction

    # get adjacency
    e0, e1 = all_edges[:, 0], all_edges[:, 1]
    idx01 = np.stack([e0, e1], axis=1)  # (E, 2)
    idx10 = np.stack([e1, e0], axis=1)  # (E, 2)
    idx_edges = np.concatenate([idx01, idx10], axis=0)  # (2*E, 2)
    ones_edges = np.ones(idx_edges.shape[0], dtype=float)  # (2*E,)
    adjacency = csc_array((ones_edges, (idx_edges[:, 0], idx_edges[:, 1])), shape=(H*W, H*W))   # for counting degrees

    # get degrees and non diag part
    deg = adjacency.sum(1)
    deg0 = deg[e0]
    deg0[deg0 > 0] = 1 / deg0[deg0 > 0]
    deg1 = deg[e1]
    deg1[deg1 > 0] = 1 / deg1[deg1 > 0]
    L_nondiag_vals = np.concatenate([deg0, deg1], 0)
    L_nondiag = csc_array((L_nondiag_vals, ((idx_edges[:, 0], idx_edges[:, 1]))), shape=(H*W, H*W))

    # get diag part
    idx_diag = np.arange(H*W, dtype=int)
    L_diag_vals = np.ones(H*W, dtype=int)
    L_diag = csc_array((L_diag_vals, (idx_diag, idx_diag)), shape=(H*W, H*W))

    L = L_nondiag - L_diag

    return L

def get_value_and_laplacian(H: int, W: int, mask: np.ndarray, value_scale: float=1.0, return_separate: bool=False):
    """
    **Modified from pytorch3d.ops.laplacian**
    
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        n_verts
        faces: tensor of shape (Nf, 2)
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """

    assert (H, W) == mask.shape

    pix_ind_img = np.arange(H*W, dtype=int).reshape(H, W)
    hori_edges = np.stack([pix_ind_img[:, :-1], pix_ind_img[:, 1:]], axis=-1)   # [H, W-1, 2]
    vert_edges = np.stack([pix_ind_img[:-1, :], pix_ind_img[1:, :]], axis=-1)   # [H-1, W, 2]
    all_edges = np.concatenate([hori_edges.reshape(-1, 2), vert_edges.reshape(-1, 2)], axis=0)  # [N_edges, 2]
    # this is already unique given the construction

    # get adjacency
    e0, e1 = all_edges[:, 0], all_edges[:, 1]
    idx01 = np.stack([e0, e1], axis=1)  # (E, 2)
    idx10 = np.stack([e1, e0], axis=1)  # (E, 2)
    idx_edges = np.concatenate([idx01, idx10], axis=0)  # (2*E, 2)
    ones_edges = np.ones(idx_edges.shape[0], dtype=float)  # (2*E,)
    adjacency = csc_array((ones_edges, (idx_edges[:, 0], idx_edges[:, 1])), shape=(H*W, H*W))   # for counting degrees

    # get degrees and non diag part
    deg = adjacency.sum(1)
    deg0 = deg[e0]
    deg0[deg0 > 0] = 1 / deg0[deg0 > 0]
    deg1 = deg[e1]
    deg1[deg1 > 0] = 1 / deg1[deg1 > 0]
    L_nondiag_vals = np.concatenate([deg0, deg1], 0)
    # L_nondiag = csc_array((L_nondiag_vals, ((idx_edges[:, 0], idx_edges[:, 1]))), shape=(H*W, H*W))
    # NOTE this is how you would normally construct L, but we will directly compute concatenated indices below

    # get diag part
    idx_diag = np.arange(H*W, dtype=int)
    L_diag_vals = -1 * np.ones(H*W, dtype=int)
    # L_diag = csc_array((L_diag_vals, (idx_diag, idx_diag)), shape=(H*W, H*W))
    # NOTE this is how you would normally construct L, but we will directly compute concatenated indices below

    # get L
    if return_separate:
        L_vals = np.concatenate([L_nondiag_vals, L_diag_vals])
        L_rowinds = np.concatenate([idx_edges[:, 0], idx_diag])
        L_colinds = np.concatenate([idx_edges[:, 1], idx_diag])
        L = csc_array((L_vals, (L_rowinds, L_colinds)), shape=(H*W, H*W))
        # NOTE this is how you would normally construct L, but we will directly compute concatenated indices below
    else:
        L = None

    # get value part
    in_mask_pix_inds = pix_ind_img[mask]  # [N_validpix]
    N_validpix = mask.sum()
    V_vals = value_scale * np.ones(N_validpix)
    V_rowinds = np.arange(N_validpix)
    V_colinds = in_mask_pix_inds
    if return_separate:
        V = csc_array((V_vals, (V_rowinds, V_colinds)), shape=(N_validpix, H*W))
        # NOTE this is how you would normally construct V, but we will directly compute concatenated indices below
    else:
        V = None

    all_vals    = np.concatenate([V_vals,    L_diag_vals,           L_nondiag_vals              ])
    all_rowinds = np.concatenate([V_rowinds, idx_diag + N_validpix, idx_edges[:, 0] + N_validpix])
    all_colinds = np.concatenate([V_colinds, idx_diag,              idx_edges[:, 1]             ])
    all_shape = (N_validpix + H*W, H*W)

    VL = csc_array((all_vals, (all_rowinds, all_colinds)), shape=all_shape)
    
    return VL, V, L



def get_value_and_laplacian_masked(H: int, W: int, vmask: np.ndarray, dmask: np.ndarray, value_scale: float=1.0):
    """
    **Modified from pytorch3d.ops.laplacian**
    
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        n_verts
        faces: tensor of shape (Nf, 2)
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """

    assert (H, W) == vmask.shape    # value part mask
    assert (H, W) == dmask.shape    # domain mask
    vmask = dmask & vmask

    vmask_flat = vmask.reshape(-1)
    N_validv = vmask.sum()

    dmask_flat = dmask.reshape(-1)
    N_validd = dmask.sum()

    pix_ind_img = np.arange(H*W, dtype=int).reshape(H, W)
    ind_map_full2validd = -1 * np.ones((H, W), dtype=int)
    ind_map_full2validd[dmask] = np.arange(N_validd, dtype=int)
    ind_map_full2validd_flat = ind_map_full2validd.reshape(-1)

    hori_edges = np.stack([pix_ind_img[:, :-1], pix_ind_img[:, 1:]], axis=-1)   # [H, W-1, 2]
    vert_edges = np.stack([pix_ind_img[:-1, :], pix_ind_img[1:, :]], axis=-1)   # [H-1, W, 2]
    all_edges = np.concatenate([hori_edges.reshape(-1, 2), vert_edges.reshape(-1, 2)], axis=0)  # [N_edges, 2]
    edge_mask = dmask_flat[all_edges].all(axis=-1)    # [N_edges]
    valid_edges = all_edges[edge_mask]           # [N_validedges, 2]
    # this is already unique given the construction

    # get adjacency
    e0, e1 = valid_edges[:, 0], valid_edges[:, 1]
    idx01 = np.stack([e0, e1], axis=1)  # (E, 2)
    idx10 = np.stack([e1, e0], axis=1)  # (E, 2)
    idx_edges = np.concatenate([idx01, idx10], axis=0)  # (2*E, 2)
    ones_edges = np.ones(idx_edges.shape[0], dtype=float)  # (2*E,)
    adjacency_full = csc_array((ones_edges, (idx_edges[:, 0], idx_edges[:, 1])), shape=(H*W, H*W))   # for counting degrees

    # get degrees and non diag part
    deg_full = adjacency_full.sum(1)
    deg0_full = deg_full[e0]
    deg0_full[deg0_full > 0] = 1 / deg0_full[deg0_full > 0]
    deg1_full = deg_full[e1]
    deg1_full[deg1_full > 0] = 1 / deg1_full[deg1_full > 0]
    L_nondiag_vals = np.concatenate([deg0_full, deg1_full], 0)
    # L_nondiag = csc_array((L_nondiag_vals, ((idx_edges[:, 0], idx_edges[:, 1]))), shape=(H*W, H*W))
    # NOTE this is how you would normally construct L, but we will directly compute concatenated indices below

    # get diag part
    idx_diag = np.arange(H*W, dtype=int)[dmask_flat]
    L_diag_vals = -1 * np.ones(H*W, dtype=int)[dmask_flat]
    # L_diag = csc_array((L_diag_vals, (idx_diag, idx_diag)), shape=(H*W, H*W))
    # NOTE this is how you would normally construct L, but we will directly compute concatenated indices below

    # get L
    L_vals = np.concatenate([L_nondiag_vals, L_diag_vals])
    L_rowinds = np.concatenate([idx_edges[:, 0], idx_diag])
    L_colinds = np.concatenate([idx_edges[:, 1], idx_diag])
    

    L_rowinds = ind_map_full2validd_flat[L_rowinds]
    L_colinds = ind_map_full2validd_flat[L_colinds]
    L = csc_array((L_vals, (L_rowinds, L_colinds)), shape=(N_validd, N_validd))

    # get value part
    in_mask_pix_inds = pix_ind_img[vmask]  # [N_validpix]
    V_vals = value_scale * np.ones(N_validv)
    V_rowinds = np.arange(N_validv)
    V_colinds = ind_map_full2validd_flat[in_mask_pix_inds]
    V = csc_array((V_vals, (V_rowinds, V_colinds)), shape=(N_validv, N_validd))

    all_vals    = np.concatenate([V_vals,    L_vals])
    all_rowinds = np.concatenate([V_rowinds, L_rowinds + N_validv])
    all_colinds = np.concatenate([V_colinds, L_colinds])
    all_shape = (N_validv + N_validd, N_validd)

    VL = csc_array((all_vals, (all_rowinds, all_colinds)), shape=all_shape)
    
    return VL, V, L, ind_map_full2validd_flat

def get_value_and_uv_laplacian_masked(H: int, W: int, vmask: np.ndarray, dmask: np.ndarray, value_scale: float=1.0):
    """
    **Modified from pytorch3d.ops.laplacian**
    
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        n_verts
        faces: tensor of shape (Nf, 2)
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """

    assert (H, W) == vmask.shape    # value part mask
    assert (H, W) == dmask.shape    # domain mask
    vmask = dmask & vmask

    vmask_flat = vmask.reshape(-1)
    N_validv = vmask.sum()

    dmask_flat = dmask.reshape(-1)
    N_validd = dmask.sum()

    pix_ind_img = np.arange(H*W, dtype=int).reshape(H, W)
    ind_map_full2validd = -1 * np.ones((H, W), dtype=int)
    ind_map_full2validd[dmask] = np.arange(N_validd, dtype=int)
    ind_map_full2validd_flat = ind_map_full2validd.reshape(-1)

    ###### du ######
    du_edges = np.stack([pix_ind_img[:, :-1], pix_ind_img[:, 1:]], axis=-1).reshape(-1, 2)   # [H*(W-1), 2]
    du_edge_mask = dmask_flat[du_edges].all(axis=-1)    # [N_edges]
    du_valid_edges = du_edges[du_edge_mask]             # [N_validedges, 2]
    
    # get adjacency (du)
    du_e0, du_e1 = du_valid_edges[:, 0], du_valid_edges[:, 1]
    du_idx01 = np.stack([du_e0, du_e1], axis=1)  # (E, 2)
    du_idx10 = np.stack([du_e1, du_e0], axis=1)  # (E, 2)
    du_idx_edges = np.concatenate([du_idx01, du_idx10], axis=0)  # (2*E, 2)
    du_ones_edges = np.ones(du_idx_edges.shape[0], dtype=float)  # (2*E,)
    du_adjacency_full = csc_array((du_ones_edges, (du_idx_edges[:, 0], du_idx_edges[:, 1])), shape=(H*W, H*W))   # for counting degrees

    # get degrees and non diag part
    du_deg_full = du_adjacency_full.sum(1)
    du_deg0_full = du_deg_full[du_e0]
    du_deg0_full[du_deg0_full > 0] = 1 / du_deg0_full[du_deg0_full > 0]
    du_deg1_full = du_deg_full[du_e1]
    du_deg1_full[du_deg1_full > 0] = 1 / du_deg1_full[du_deg1_full > 0]
    du_L_nondiag_vals = np.concatenate([du_deg0_full, du_deg1_full], 0)
    # du_L_nondiag = csc_array((du_L_nondiag_vals, ((du_idx_edges[:, 0], du_idx_edges[:, 1]))), shape=(H*W, H*W))
    # NOTE this is how you would normally construct L, but we will directly compute concatenated indices below

    ###### dv ######
    # get adjacency (dv)
    dv_edges = np.stack([pix_ind_img[:-1, :], pix_ind_img[1:, :]], axis=-1).reshape(-1, 2)   # [(H-1)*W, 2]
    dv_edge_mask = dmask_flat[dv_edges].all(axis=-1)    # [N_edges]
    dv_valid_edges = dv_edges[dv_edge_mask]             # [N_validedges, 2]
    # this is already unique given the construction

    dv_e0, dv_e1 = dv_valid_edges[:, 0], dv_valid_edges[:, 1]
    dv_idx01 = np.stack([dv_e0, dv_e1], axis=1)  # (E, 2)
    dv_idx10 = np.stack([dv_e1, dv_e0], axis=1)  # (E, 2)
    dv_idx_edges = np.concatenate([dv_idx01, dv_idx10], axis=0)  # (2*E, 2)
    dv_ones_edges = np.ones(dv_idx_edges.shape[0], dtype=float)  # (2*E,)
    dv_adjacency_full = csc_array((dv_ones_edges, (dv_idx_edges[:, 0], dv_idx_edges[:, 1])), shape=(H*W, H*W))   # for counting degrees

    # get degrees and non diag part
    dv_deg_full = dv_adjacency_full.sum(1)
    dv_deg0_full = dv_deg_full[dv_e0]
    dv_deg0_full[dv_deg0_full > 0] = 1 / dv_deg0_full[dv_deg0_full > 0]
    dv_deg1_full = dv_deg_full[dv_e1]
    dv_deg1_full[dv_deg1_full > 0] = 1 / dv_deg1_full[dv_deg1_full > 0]
    dv_L_nondiag_vals = np.concatenate([dv_deg0_full, dv_deg1_full], 0)
    # dv_L_nondiag = csc_array((dv_L_nondiag_vals, ((dv_idx_edges[:, 0], dv_idx_edges[:, 1]))), shape=(H*W, H*W))
    # NOTE this is how you would normally construct L, but we will directly compute concatenated indices below

    # get diag part
    du_idx_diag = np.arange(H*W, dtype=int)[dmask_flat]
    du_L_diag_vals = -1 * np.ones(H*W, dtype=int)[dmask_flat] * (du_deg_full[dmask_flat] > 1).astype(float) # NOTE only compute laplacian for verts that have neighbors

    dv_idx_diag = np.arange(H*W, dtype=int)[dmask_flat]
    dv_L_diag_vals = -1 * np.ones(H*W, dtype=int)[dmask_flat] * (dv_deg_full[dmask_flat] > 1).astype(float) # NOTE only compute laplacian for verts that have neighbors
    # L_diag = csc_array((L_diag_vals, (idx_diag, idx_diag)), shape=(H*W, H*W))
    # NOTE this is how you would normally construct L, but we will directly compute concatenated indices below

    # get L
    du_L_vals = np.concatenate([du_L_nondiag_vals, du_L_diag_vals])
    du_L_rowinds = np.concatenate([du_idx_edges[:, 0], du_idx_diag])
    du_L_colinds = np.concatenate([du_idx_edges[:, 1], du_idx_diag])
    du_L_rowinds = ind_map_full2validd_flat[du_L_rowinds]
    du_L_colinds = ind_map_full2validd_flat[du_L_colinds]
    du_L = csc_array((du_L_vals, (du_L_rowinds, du_L_colinds)), shape=(N_validd, N_validd))

    dv_L_vals = np.concatenate([dv_L_nondiag_vals, dv_L_diag_vals])
    dv_L_rowinds = np.concatenate([dv_idx_edges[:, 0], dv_idx_diag])
    dv_L_colinds = np.concatenate([dv_idx_edges[:, 1], dv_idx_diag])
    dv_L_rowinds = ind_map_full2validd_flat[dv_L_rowinds]
    dv_L_colinds = ind_map_full2validd_flat[dv_L_colinds]
    dv_L = csc_array((dv_L_vals, (dv_L_rowinds, dv_L_colinds)), shape=(N_validd, N_validd))

    # get value part
    in_mask_pix_inds = pix_ind_img[vmask]  # [N_validpix]
    V_vals = value_scale * np.ones(N_validv)
    V_rowinds = np.arange(N_validv)
    V_colinds = ind_map_full2validd_flat[in_mask_pix_inds]
    V = csc_array((V_vals, (V_rowinds, V_colinds)), shape=(N_validv, N_validd))

    all_vals    = np.concatenate([V_vals,    du_L_vals,               dv_L_vals])
    all_rowinds = np.concatenate([V_rowinds, du_L_rowinds + N_validv, dv_L_rowinds + N_validv + N_validd])
    all_colinds = np.concatenate([V_colinds, du_L_colinds,            dv_L_colinds])
    all_shape = (N_validv + N_validd + N_validd, N_validd)

    VL = csc_array((all_vals, (all_rowinds, all_colinds)), shape=all_shape)
    
    return VL, V, du_L, dv_L, ind_map_full2validd_flat
