import drtk
import torch as th
import torch.nn.functional as thf
from tqdm import tqdm, trange
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt

v = th.as_tensor(
    [[70, 110, 10], [400, 60, 10], [300, 400, 10]], dtype=th.float32
).cuda()[None]
vi = th.as_tensor([[0, 1, 2]], dtype=th.int32).cuda()
index_img = drtk.rasterize(v, vi, width=512, height=512)
image_gt = (index_img != -1).float()
save_image(image_gt, "img.png")


v = th.as_tensor(
    [[120, 60, 10], [400, 200, 10], [100, 300, 10]], dtype=th.float32
).cuda()[None]
index_img = drtk.rasterize(v, vi, width=512, height=512)
depth_img, bary_img = drtk.render(v, vi, index_img)
image = (index_img != -1).float()
save_image(image, "img.png")


l2_loss = thf.mse_loss(image, image_gt, reduction="none")


# Need to make vertex positions differentiable, otherwise the gradient will not be computed.
v = th.as_tensor(
    [[120, 60, 10], [400, 200, 10], [100, 300, 10]], dtype=th.float32
).cuda()[None]
v_param = th.nn.Parameter(v)

opt = th.optim.SGD([v_param], lr=1000.0)
loss_list = []


for iter in tqdm(range(1000)):
    index_img = drtk.rasterize(v_param, vi, width=512, height=512)
    depth_img, bary_img = drtk.render(v_param, vi, index_img)
    image = (index_img != -1).float()

    # Make `image` differentiable
    image_differentiable = drtk.edge_grad_estimator(
        v_param,    # verts for rasterization
        vi,         # faces
        bary_img,   # barys
        image[:, None], # rendered image
        index_img)  # face indices

    # Compute loss and backpropagate
    l2_loss = thf.mse_loss(image_differentiable, image_gt[:, None])
    l2_loss.backward()
    opt.step()
    opt.zero_grad()

    loss_list.append(l2_loss.item())

plt.plot(loss_list)
plt.show()


