import numpy as np

def gen_camera_visualization(K: np.ndarray, c2w: np.ndarray, H: int, W: int, cone_height: float, mode: str):
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

    if mode == 'line':
        faces = np.array([
            [0, 1, 0],
            [0, 2, 0],
            [0, 3, 0],
            [0, 4, 0],
            [1, 2, 1],
            [2, 3, 2],
            [3, 4, 3],
            [4, 1, 4],
        ])
    else:
        raise NotImplementedError(f"mode {mode} is not supported")
    
    return points_world, faces