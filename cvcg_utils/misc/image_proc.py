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

