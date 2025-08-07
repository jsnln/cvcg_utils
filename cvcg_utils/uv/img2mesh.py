import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy.spatial import Delaunay

def make_pixel_uvs(H, W, homogeneous):
    """
    generates pixel uv coord image
    """
    u_axis = np.linspace(0, W-1, W) + 0.5
    v_axis = np.linspace(0, H-1, H) + 0.5
    uv_img = np.stack([np.broadcast_to(u_axis[None, :], (H, W)),
                        np.broadcast_to(v_axis[:, None], (H, W))], axis=-1)
    
    if homogeneous:
        uv_img = np.pad(uv_img, ((0,0), (0,0), (0,1)), constant_values=1.)
    
    return uv_img


def make_texture_uvs(H, W):
    """
    generates texture uv coord image
    """
    u_axis = np.linspace(0, 1, W)
    v_axis = np.linspace(1, 0, H)
    uv_img = np.stack([np.broadcast_to(u_axis[None, :], (H, W)),
                       np.broadcast_to(v_axis[:, None], (H, W))], axis=-1)
    
    return uv_img

def make_pixel_indices(H, W):
    return np.linspace(0, H*W-1, H*W, dtype=int).reshape(H, W)

def make_pixel_triangles(H, W):
    """
    generates triangles for an image of size (H, W):

    pixel indices:
    [[0, 1, 2, ..., W-1],
     [W, W+1, ..., 2W-1],
     ...
     [...,         HW-1]]
    """
    pixel_ind_img = make_pixel_indices(H, W)

    triangles_top_left  = np.stack([pixel_ind_img[0:H-1, 0:W-1],
                                    pixel_ind_img[1:H  , 0:W-1],
                                    pixel_ind_img[0:H-1, 1:W  ]], axis=-1)  # [H, W, 3]
    triangles_bot_right = np.stack([pixel_ind_img[1:H  , 0:W-1],
                                    pixel_ind_img[1:H  , 1:W  ],
                                    pixel_ind_img[0:H-1, 1:W  ]], axis=-1)  # [H, W, 3]

    triangles_all = np.stack([triangles_top_left, triangles_bot_right], axis=-2)    # [H, W, 2, 3]
    return triangles_all.reshape(-1, 3)

def make_masked_delaunay(mask):
    """
    mask: [H, W], bool
    """
    H, W = mask.shape
    uv_img = make_pixel_uvs(H, W, homogeneous=False)
    # pix_ind_img = make_pixel_indices(H, W)
    
    valid_uv_coords = uv_img[mask]
    # valid_pix_index = 

    delaunay_tris = Delaunay(valid_uv_coords).simplices
    # somehow delaunay has wrong orientation, reverse it
    delaunay_tris = np.stack([delaunay_tris[:, 0], delaunay_tris[:, 2], delaunay_tris[:, 1]], axis=-1)
    return delaunay_tris


if __name__ == '__main__':
    H = 64
    W = 128
    depth_img = np.random.randn(H, W, 1)
    uv_img = make_pixel_uvs(H, W, homogeneous=False)
    faces = make_pixel_triangles(H, W)
    coords = np.concatenate([uv_img, depth_img], axis=-1)

    from cvcg_utils.mesh import write_ply
    write_ply('debug.ply', coords.reshape(-1, 3), faces)

    mask = (depth_img > 0.5)[..., 0]
    delaunay_tris = make_masked_delaunay(mask)
    masked_coords = coords[mask]
    write_ply('delaunay.ply', masked_coords, delaunay_tris)



