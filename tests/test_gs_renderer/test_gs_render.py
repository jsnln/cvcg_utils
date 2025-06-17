import os
import pickle
import torch
import matplotlib.pyplot as plt

import numpy as np
from cvcg_utils.render.camera import UnifiedCamera
from cvcg_utils.render.gs_renderer import render_gs, strip_symmetric
from cvcg_utils.misc.image import write_rgb

def load_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

camera = UnifiedCamera.from_lookat(
    center=np.array([0., 0., 1.]),
    lookat=np.array([0., 0., 0.]),
    up=np.array([0., 1., 0.]),
    fov_x=60, fov_y=60, fov_mode='deg',
    K=None, H=1024, W=1024).to_3dgs_format()

# camera = camera
verts = np.array([
    [-.1, 0., -.001],   # red
    [0., .2, 0.001],    # green
])

xyz = torch.from_numpy(verts).float().cuda()
opacity = torch.ones(xyz.shape[0]).cuda() * 0.99
cov3D_precomp = torch.zeros(xyz.shape[0], 3, 3).cuda()
cov3D_precomp[:, 0, 0] = 0.005
cov3D_precomp[:, 1, 1] = 0.005
cov3D_precomp[:, 2, 2] = 0.005

override_color = torch.ones(xyz.shape[0], 3).cuda()

override_color[0, :] = 0
override_color[0, 0] = 1    # red

override_color[1, :] = 0
override_color[1, 1] = 1    # green    

cov3D_precomp = strip_symmetric(cov3D_precomp)
# rendered = render_gs(camera, xyz, opacity, scales=None, rotations=None, cov3D_precomp=cov3D_precomp, override_color=override_color, clip_value=False)
# img = rendered['render'].detach().permute(1,2,0).cpu().numpy()

# xyz_camera_space = torch.from_numpy(camera.proj_world2camera_opencv(xyz.cpu().numpy())).float().cuda()
termi_depth_img = torch.ones(camera.H, camera.W).cuda() * 2
termi_depth_img[:500,:500] = 1.0
# rendered_xyz = render_gs(camera, xyz, opacity, scales=None, rotations=None, cov3D_precomp=cov3D_precomp, override_color=override_color, clip_value=False, termi_depth_img=termi_depth_img)
rendered_xyz = render_gs(camera, xyz, opacity, scales=None, rotations=None, cov3D_precomp=cov3D_precomp, override_color=override_color, clip_value=False, termi_depth_img=None)
img = rendered_xyz['render'].detach().permute(1,2,0).cpu().numpy()
depth = rendered_xyz['depth'].detach()[0].cpu().numpy()

write_rgb('debug.png', (img * 255).astype(np.uint8))

# plt.imshow(img[...,])
# plt.show()

# plt.imshow(depth[...,])
# plt.show()

# import cv2
# import numpy as np
# cv2.imwrite('test_gs_render.png', (img*255).astype(np.uint8))

