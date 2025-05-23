import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class MaskedPosmap2Normal(nn.Module):
    def __init__(self, mask, flip_mask=None):
        """
        mask: [h, w], bool
        flip_mask: [h, w], bool
        """
        super(MaskedPosmap2Normal, self).__init__()

        mask = mask.cpu()   # always init on cpu

        diff_kernel = torch.tensor([
            [[0, 0, 0],
            [0, -1, 1],
            [0, 0, 0]],    # dx+
            [[0., 1, 0],
            [0, -1, 0],
            [0, 0, 0]],    # dy+
            [[0, 0, 0],
            [1, -1, 0],
            [0, 0, 0]],    # dx-
            [[0, 0, 0],
            [0, -1, 0],
            [0, 1, 0]],    # dy-
        ])[:, None]  # [4, 1, 3, 3]
        self.register_buffer('diff_kernel', diff_kernel)

        valid_kernel = torch.tensor([
            [[0.,1, 0],
            [0, 1, 1],
            [0, 0, 0]],    # dx+, dy+
            [[0, 1, 0],
            [1, 1, 0],
            [0, 0, 0]],    # dy+, dx-
            [[0, 0, 0],
            [1, 1, 0],
            [0, 1, 0]],    # dx-, dy-
            [[0, 0, 0],
            [0, 1, 1],
            [0, 1, 0]],    # dy-, dx+
        ])[:, None]  # [4, 1, 3, 3]

        valid_map = F.conv2d(mask[None, None].float(), valid_kernel, padding=1) # [1, 4, h, w]
        valid_map[valid_map != 3] = 0.
        valid_map[valid_map == 3] = 1.
        if flip_mask is not None:
            valid_map[:, :, flip_mask] *= -1
        self.register_buffer('valid_map', valid_map)

    def forward(self, posmap: torch.Tensor, normalize=True) -> torch.Tensor:
        """
        posmap: [b, 3, h, w]
        normal_map: [b, 3, h, w]
        """
        assert len(posmap.shape) == 4
        assert posmap.shape[1] == 3

        b, _, h, w = posmap.shape

        posmap = posmap.view(-1, h, w)   # [b*3, h, w]

        diff_map = F.conv2d(posmap[:, None], self.diff_kernel, padding=1)    # [b*3, 4 (dx+, dy+, dx-, dy-), h, w]
        diff_map2 = torch.roll(diff_map, shifts=-1, dims=1)              # [b*3, 4 (dy+, dx-, dy-, dx+), h, w]
        
        # change diff_maps from [b*3, 4, h, w] to [b, 3, 4, h, w]
        diff_map = diff_map.view(b, 3, 4, h, w)
        diff_map2 = diff_map2.view(b, 3, 4, h, w)
        cross_map = torch.cross(diff_map, diff_map2, dim=1) # [b, 3, 4, h, w]

        normal_map = (cross_map * self.valid_map[None]).sum(2)    # [b, 3, 4, h, w] * [1, 1, 4, h, w] -> [b, 3, h, w]
        if normalize:
            normal_map = F.normalize(normal_map, dim=1) # [b, 3, h, w]
        
        return normal_map


if __name__ == '__main__':
    posmap = cv2.imread('cano_smpl_pos_map.exr', cv2.IMREAD_UNCHANGED)
    posmap = torch.from_numpy(posmap).permute(2, 0, 1)
    mask = (torch.norm(posmap, dim=0) > 0.)
    flip_mask = mask.clone()
    flip_mask[:] = False
    flip_mask[:, 1024:] = True

    posmap_to_normal = MaskedPosmap2Normal(mask, flip_mask)

    normal_map = posmap_to_normal(posmap)
    cv2.imwrite('normal.png', (((normal_map.permute(1, 2, 0) + 1) / 2).cpu().numpy() * 255).astype(np.uint8))

    pts = posmap[:, mask]
    normals = normal_map[:, mask]

    pts_normals = torch.cat([pts, normals], dim=0).transpose(0, 1)
    np.savetxt('pcd.xyz', pts_normals.numpy())