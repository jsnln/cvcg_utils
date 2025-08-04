from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt

def gen_line_visualization(start: np.ndarray, end: np.ndarray):
    """
    3d vectors
    """
    eps = 1e-4
    v1 = np.array(start)
    v2 = np.array(end)
    v1_e = v1.copy()
    normal = np.linalg.cross(v2 - v1, np.random.randn(3))
    normal /= np.linalg.norm(normal).clip(1e-6)
    v1_e += normal * eps

    verts = np.stack([v1, v1_e, v2], 0)

    faces = np.array([
        [0, 1, 2],
    ])
    
    return verts, faces

def gen_camera_visualization(K: np.ndarray, c2w: np.ndarray, H: int, W: int, cone_height: float):
    """
    all in opencv format
    K: 3x3
    c2w: 4x4
    """
    uv_corners_homog = np.array([
        [0, 0, 1.],
        [W, 0, 1.],
        [W, H, 1.],
        [0, H, 1.],
    ])
    corners_cam = np.einsum('ij,nj->ni', np.linalg.inv(K), uv_corners_homog) * cone_height
    points_cam = np.pad(corners_cam, ((1,0), (0,0)), constant_values=0)   # [5, 3]
    points_world = np.einsum('ij,nj->ni', c2w[:3, :3], points_cam) + c2w[:3, 3]

    faces = np.array([
        [0, 2, 1],
        [0, 3, 2],
        [0, 4, 3],
        [0, 1, 4],
        [2, 4, 1],
        [2, 3, 4],
    ])
    
    return points_world, faces

def gen_camera_set_visualization(K_list: List[np.ndarray], c2w_list: List[np.ndarray], H: Union[int, List[int]], W: Union[int, List[int]], cone_height: float):
    """
    all in opencv format
    K: 3x3
    c2w: 4x4
    """
    def merge_mesh(verts_1, faces_1, verts_2, faces_2):
        verts = np.concatenate([verts_1, verts_2], axis=0)
        faces = np.concatenate([faces_1, faces_2 + len(verts_1)], axis=0)
        return verts, faces

    if isinstance(H, int):
        H = [H] * len(K_list)
    if isinstance(W, int):
        W = [W] * len(K_list)

    verts_all = np.zeros((0, 3))
    faces_all = np.zeros((0, 3), dtype=int)
    for K, c2w, curH, curW in zip(K_list, c2w_list, H, W):
        points_world, faces = gen_camera_visualization(K, c2w, curH, curW, cone_height)
        verts_all, faces_all = merge_mesh(verts_all, faces_all, points_world, faces)

    cmap = plt.colormaps['rainbow']
    v_colors = cmap(np.linspace(0, 1, len(K_list)))[:, :3] # [N, 3]
    v_colors = np.broadcast_to(v_colors[:, None], (len(K_list), 5, 3)).reshape(-1, 3)
    
    return verts_all, faces_all, v_colors
