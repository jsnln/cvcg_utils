import os
import pickle
import torch

from cvcg_utils import UnifiedCamera, render_gs, make_gs_rasterizer

def load_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

in_path = '/home/trisst/data/cvdata_archive/4D-Dress/4D-DRESS_00122/Outer/Take9'
frame_id = 11

### camera
# basic_info_pkl = os.path.join(in_path, f'basic_info.pkl')
# basic_info_data = load_pickle(basic_info_pkl)

cam_pkl = os.path.join(in_path, 'Capture', f'cameras.pkl')
cam_data = load_pickle(cam_pkl)
camera = UnifiedCamera.from_4d_dress(cam_data['0004']['intrinsics'], cam_data['0004']['extrinsics'], 1280, 940)

### geometry
mesh_pkl = os.path.join(in_path, 'Meshes_pkl', f'mesh-f{frame_id:05d}.pkl')
mesh_data = load_pickle(mesh_pkl)   # vertices, normals, uvs, faces, colors, uv_path

xyz = torch.from_numpy(mesh_data['vertices']).float().cuda()
opacity = torch.ones(xyz.shape[0]).cuda() * 0.99
cov3D_precomp = torch.zeros(xyz.shape[0], 3, 3).cuda()
cov3D_precomp[:, 0, 0] = 0.00005
cov3D_precomp[:, 1, 1] = 0.00005
cov3D_precomp[:, 2, 2] = 0.00005
override_color = torch.ones(xyz.shape[0], 3).cuda()

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

cov3D_precomp = strip_symmetric(cov3D_precomp)

# rendered = render_gs(camera, xyz, opacity, scales=None, rotations=None, cov3D_precomp=cov3D_precomp, override_color=override_color, clip_value=False)
# img = rendered['render'].detach().permute(1,2,0).cpu().numpy()

xyz_camera_space = torch.from_numpy(camera.proj_world2camera_opencv(xyz.cpu().numpy())).float().cuda()
rendered_xyz = render_gs(camera, xyz, opacity, scales=None, rotations=None, cov3D_precomp=cov3D_precomp, override_color=xyz_camera_space, clip_value=False)
img = rendered_xyz['render'].detach().permute(1,2,0).cpu().numpy()

import matplotlib.pyplot as plt
plt.imshow(img[..., 2])
plt.show()

import cv2
import numpy as np
cv2.imwrite('test_gs_render.png', (img*255).astype(np.uint8))

