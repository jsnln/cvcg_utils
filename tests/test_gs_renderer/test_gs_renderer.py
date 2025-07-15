import numpy as np
import torch
from cvcg_utils.mesh.gaussian_io import load_ply
from cvcg_utils.render.camera import UnifiedCamera
from cvcg_utils.render.gs_renderer import render_gs
from cvcg_utils.misc.image import write_rgb

uni_cam = UnifiedCamera.from_lookat(
    center=np.array([0, 0, 0]),
    lookat=np.array([0, 0, -1.]),
    up=np.array([0, 1, 0]),
    fov_mode='deg',
    fov_x=60,
    fov_y=60,
    K=None,
    H=1024,
    W=1024,
)
gs_cam = uni_cam.to_3dgs_format()

gs_np = load_ply('point_cloud1.ply')
gs_th = gs_np.activate()

with torch.no_grad():
    render = render_gs(gs_cam, gs_th.xyz, gs_th.opacity, gs_th.scaling, gs_th.rotation, gs_th.features, clip_value=True)

write_rgb('debug_gs_render.png', (render['render'].permute(1,2,0) * 255).byte().cpu().numpy())

