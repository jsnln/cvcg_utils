import os
import pickle
import torch
import cv2
import numpy as np
import math
from tqdm import trange

from geometry_utils import UnifiedCamera, render_gs, write_video

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

def quat_from_axis_angle(npts, axis, angle):
    axis = torch.nn.functional.normalize(axis, dim=-1)
    cos_angleo2 = math.cos(angle/2)
    sin_angleo2 = math.sin(angle/2)

    quats = torch.zeros(npts, 4)
    quats[:, 0] = cos_angleo2
    quats[:, 1] = sin_angleo2 * axis[0]
    quats[:, 2] = sin_angleo2 * axis[1]
    quats[:, 3] = sin_angleo2 * axis[2]

    return quats

camera1 = UnifiedCamera.from_lookat(
    center=np.array([0., 0., 5.0]),
    lookat=np.array([0., 0., 0.]),
    up=np.array([0., 1., 0.]),
    fov_x=60/180*np.pi,
    fov_y=None,
    H=1024,
    W=648,
)

camera2 = UnifiedCamera.from_lookat(
    center=np.array([5., 2., 5.0]),
    lookat=np.array([0., 0., 0.]),
    up=np.array([0., 1., 0.]),
    fov_x=60/180*np.pi,
    fov_y=None,
    H=1024,
    W=648,
)

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
