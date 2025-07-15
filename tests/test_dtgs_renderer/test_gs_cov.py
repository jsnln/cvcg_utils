import os
import pickle
import torch
import cv2
import numpy as np
import math
from tqdm import trange

from cvcg_utils.render.camera import UnifiedCamera
from cvcg_utils.render.gs_renderer import render_gs, quat_from_axis_angle
from cvcg_utils.misc.video_io import write_video


camera1 = UnifiedCamera.from_lookat(
    center=np.array([0., 0., 5.0]),
    lookat=np.array([0., 0., 0.]),
    up=np.array([0., 1., 0.]),
    fov_x=60,
    fov_y=60,
    fov_mode='deg',
    K=None,
    H=1024,
    W=648,
).to_3dgs_format()

camera2 = UnifiedCamera.from_lookat(
    center=np.array([5., 2., 5.0]),
    lookat=np.array([0., 0., 0.]),
    up=np.array([0., 1., 0.]),
    fov_x=60,
    fov_y=60,
    fov_mode='deg',
    K=None,
    H=1024,
    W=648,
).to_3dgs_format()

xyz = torch.from_numpy(np.array([[0., 0., 0.]])).float().cuda()
opacity = torch.ones(xyz.shape[0]).cuda() * 0.99
scales = torch.ones(xyz.shape[0], 3).cuda()
rotations = torch.zeros(xyz.shape[0], 4).cuda()
override_color = torch.ones(xyz.shape[0], 3).cuda()

# no transf
# scales[:] = scales * 1.0
# rotations[:, 0] = 1

# scale only
# scales[:, 0] = 2.0
# rotations[:, 0] = 1

# rot
scales[:, 0] = 2.0

# conclusion:
# scales and rotations in 3DGS are forward transformations
# with scales first and rotations later

img_seq = []
for i in trange(360):
    rotations = quat_from_axis_angle(1, torch.tensor([0.,0,1]), i/180*math.pi).cuda()

    rendered1 = render_gs(camera1, xyz, opacity, scales=scales, rotations=rotations, cov3D_precomp=None, override_color=override_color, clip_value=False)
    img1 = rendered1['render'].detach().permute(1,2,0).cpu().numpy()

    rendered2 = render_gs(camera2, xyz, opacity, scales=scales, rotations=rotations, cov3D_precomp=None, override_color=override_color, clip_value=False)
    img2 = rendered2['render'].detach().permute(1,2,0).cpu().numpy()

    img = np.concatenate([
        np.pad(img1, ((0,0), (0,100), (0,0)), constant_values=1),
        img2
    ], axis=1)
    img = (img*255).astype(np.uint8)[..., ::-1]
    # cv2.imwrite('test_gs_cov.png', (img*255).astype(np.uint8))
    img_seq.append(img)

write_video('test_gs_cov.mp4', img_seq)
