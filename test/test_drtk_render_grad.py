import numpy as np
import os
import pickle
import igl
from geometry_utils import write_ply
from geometry_utils import UnifiedCamera, DRTKCamera, render_drtk_face_attr, render_drtk_uv_textured
import drtk
import torch as th
from torchvision.utils import save_image

v = th.as_tensor(
    [[70, 110, 10], [400, 60, 10], [300, 400, 10]], dtype=th.float32
).cuda()[None]
vi = th.as_tensor([[0, 1, 2]], dtype=th.int32).cuda()

index_img = drtk.rasterize(v, vi, width=512, height=512)
image_gt = (index_img != -1).float()

uni_camera = UnifiedCamera.from_lookat(
    np.array([256., 256, 90]),
    np.array([256., 256, 10]),
    np.array([0., -1., 0]),
    60, 60, None, 512, 512)
drtk_camera = uni_camera.to_drtk_format().float().cuda()


v = th.as_tensor(
    [[80, 110, 10], [390, 70, 10], [300, 290, 10]], dtype=th.float32
).cuda()[None]

uv_v = th.as_tensor(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=th.float32
).cuda()
texture = th.ones(1, 1, 2, 2).float().cuda()
# v = th.as_tensor(
#     [[167.5732, 328.4392,   0.9995],
#     [322.4442, 348.4226,   0.9995],
#     [277.4817, 238.5141,   0.9995]], dtype=th.float32
# ).cuda()[None]

v_param = th.nn.Parameter(v)

opt = th.optim.SGD([v_param], lr=10000.0)
loss_list = []

os.makedirs('test_drtk_render_grad', exist_ok=True)
from tqdm import tqdm
for iter in tqdm(range(1000)):

    # v_proj = drtk_camera.proj_points_to_drtk_screen(v_param, detach_z=True)
    # v_proj = drtk_camera.proj_points_to_drtk_screen(v_param, detach_z=False)
    # print(v_proj)
    # breakpoint()
    # v_proj = v_param

    # index_img = drtk.rasterize(v_proj, vi, width=512, height=512)
    # depth_img, bary_img = drtk.render(v_proj, vi, index_img)

    # image = (index_img != -1).float()

    # # Make `image` differentiable
    # image_differentiable = drtk.edge_grad_estimator(
    #     v_proj, vi, bary_img, image[:, None], index_img
    # )

    image_differentiable, mask_differentiable = render_drtk_uv_textured(drtk_camera, v_param, vi, uv_v, vi, texture, True, True)

    # Compute loss and backpropagate
    l2_loss = th.nn.functional.mse_loss(mask_differentiable, image_gt[:, None])
    l2_loss.backward()
    opt.step()
    opt.zero_grad()

    loss_list.append(l2_loss.item())
    
    with th.no_grad():
        save_image((image_differentiable - image_gt[:, None]).abs(), os.path.join('test_drtk_render_grad', f"img{iter:05d}.png"))

