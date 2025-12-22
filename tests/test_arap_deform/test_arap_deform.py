import torch
from tqdm import tqdm
from cvcg_utils.mesh.mesh_loss import arap_loss, cotangent_weights
from cvcg_utils.mesh import read_ply, write_ply

mesh = read_ply('box_deform_test.ply')

verts_rest = torch.from_numpy(mesh.verts).cuda().float()
verts_deformed = verts_rest.clone()
faces = torch.from_numpy(mesh.faces).cuda().long()
edge_ij, w_ij = cotangent_weights(verts_rest, faces, eps=1e-9)  # undirected i<j
# 0, 1, 4, 5 invariant, 2 -> 6 -> 7 -> 3 -> 2
corner_targets = torch.stack([
    verts_rest[0],
    verts_rest[1],
    verts_rest[6],
    verts_rest[2],
    verts_rest[4],
    verts_rest[5],
    verts_rest[7],
    verts_rest[3],
], dim=0)
corner_targets[2, 1] += 0.2
corner_targets[3, 1] += 0.2
corner_targets[6, 1] += 0.2
corner_targets[7, 1] += 0.2

# corner_targets = torch.stack([
#     verts_rest[0],
#     verts_rest[1],
#     verts_rest[2],
#     verts_rest[3],
#     verts_rest[4],
#     verts_rest[5],
#     verts_rest[6],
#     verts_rest[7],
# ], dim=0)
# corner_targets[2, 1] += 2
# corner_targets[3, 1] += 2
# corner_targets[6, 1] += 2
# corner_targets[7, 1] += 2

verts_deformed.requires_grad_(True)
optimizer = torch.optim.Adam([verts_deformed], lr=1e-3)

bar = tqdm(range(5000))
for iter in bar:

    loss_arap, R = arap_loss(verts_deformed, verts_rest, edge_ij, w_ij, return_rotations=True)
    loss_tgt = torch.nn.functional.mse_loss(verts_deformed[:8], corner_targets)
    loss = loss_arap * 0.1 + loss_tgt
    # loss = loss_arap

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    bar.set_postfix({'arap': loss_arap.item(), 'tgt': loss_tgt.item()})
    # bar.set_postfix({'arap': loss_arap.item()})

write_ply('arap_debug.ply', verts_deformed.detach().cpu().numpy(), faces.cpu().numpy())