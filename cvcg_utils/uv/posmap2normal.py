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



class Posmap2Normal(nn.Module):
    def __init__(self):
        super(Posmap2Normal, self).__init__()

        # mask = mask.cpu()   # always init on cpu

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
        self.register_buffer('valid_kernel', valid_kernel)
        

    def forward(self, posmap: torch.Tensor, mask: torch.Tensor, normalize=True) -> torch.Tensor:
        """
        posmap: [b, 3, h, w]
        normal_map: [b, 3, h, w]
        mask: [b, h, w], bool
        flip_mask: Not supported yet
        """
        assert len(posmap.shape) == 4
        assert posmap.shape[1] == 3
        
        if mask is not None: assert len(mask) == 3
        # assert flip_mask is None

        b, _, h, w = posmap.shape

        posmap = posmap.view(-1, h, w)   # [b*3, h, w]

        diff_map = F.conv2d(posmap[:, None], self.diff_kernel, padding=1)    # [b*3, 4 (dx+, dy+, dx-, dy-), h, w]
        diff_map2 = torch.roll(diff_map, shifts=-1, dims=1)              # [b*3, 4 (dy+, dx-, dy-, dx+), h, w]
        
        # change diff_maps from [b*3, 4, h, w] to [b, 3, 4, h, w]
        diff_map = diff_map.view(b, 3, 4, h, w)
        diff_map2 = diff_map2.view(b, 3, 4, h, w)
        cross_map = torch.cross(diff_map, diff_map2, dim=1) # [b, 3, 4, h, w]

        # if there's mask
        with torch.no_grad():
            if mask is None:
                mask = torch.ones(b, h, w, dtype=torch.bool, device=posmap.device)
            valid_map = F.conv2d(mask[:, None].float(), self.valid_kernel, padding=1) # [b, 4, h, w]
            valid_map[valid_map != 3] = 0.
            valid_map[valid_map == 3] = 1.
        
        # if (valid_map is not None) and (flip_mask is not None):
        #     # valid_map[:, :, flip_mask] *= -1 # TODO batch dim hanlding is wrong accoding to the new formulation
        #     pass

        normal_map = (cross_map * valid_map[:, None]).sum(2)    # [b, 3, 4, h, w] * [b, 1, 4, h, w] -> [b, 3, h, w]
        if normalize:
            normal_map = F.normalize(normal_map, dim=1) # [b, 3, h, w]
        
        return normal_map


class Img2Grad(nn.Module):
    def __init__(self):
        super(Img2Grad, self).__init__()

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
            [[0.,0, 0],
            [0, 1, 1],
            [0, 0, 0]],    # dx+
            [[0, 1, 0],
            [0, 1, 0],
            [0, 0, 0]],    # dy+
            [[0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]],    # dx-
            [[0, 0, 0],
            [0, 1, 0],
            [0, 1, 0]],    # dy-
        ])[:, None]  # [4, 1, 3, 3]
        self.register_buffer('valid_kernel', valid_kernel)
        

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        img: [b, c, h, w]
        mask: [b, h, w], bool
        """
        assert len(img.shape) == 4
        assert img.shape[1] == 3
        if mask is not None: assert len(mask) == 3

        B, C, H, W = img.shape

        posmap = posmap.view(-1, H, W)   # [B*C, H, W]

        diff_map = F.conv2d(posmap[:, None], self.diff_kernel, padding=1)    # [B*C, 4 (dx+, dy+, dx-, dy-), H, W]
        
        # change diff_maps from [B*C, 4, H, W] to [B, C, 4, H, W]
        diff_map = diff_map.view(B, C, 4, H, W)

        # if there's mask
        with torch.no_grad():
            if mask is None:
                mask = torch.ones(B, H, W, dtype=torch.bool, device=img.device)
            valid_map = F.conv2d(mask[:, None].float(), self.valid_kernel, padding=1) # [B, 4, H, W]
            valid_map[valid_map != 2] = 0.
            valid_map[valid_map == 2] = 1.
        
        grad_map = (diff_map * valid_map[:, None]).sum(2)    # [B, C, 4, H, W] * [B, 1, 4, H, W] -> [B, C, 4, H, W]
        
        return grad_map, valid_map


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