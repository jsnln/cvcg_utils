import numpy as np


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
        [H, 0, 1.],
        [H, W, 1.],
        [0, W, 1.],
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