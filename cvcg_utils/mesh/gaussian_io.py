#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import torch
import numpy as np
# from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
# from utils.system_utils import mkdir_p
from dataclasses import dataclass
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

@dataclass
class ActGaussianModelTorch:
    xyz: torch.Tensor
    features: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    max_sh_degree: int

    def rigid_transform(self, transf_mat: torch.Tensor=None, scale: float=None):
        """
        - returns a copy of self after a rigid transformation and isotropic scaling
        - if both are None, simply returns a copy
        - scale is applied after rigid transform

        transf_mat: [4, 4], must be rigid transform
        scale: float, must be positive

        """

        if transf_mat is not None:
            xyz_new = torch.einsum('ij,nj->ni', transf_mat[:3, :3], self.xyz) + transf_mat[:3, 3]
        else:
            xyz_new = self.xyz.clone()
        
        if scale is not None:
            assert scale > 0
            xyz_new = xyz_new * scale


        # TODO add feature transform: https://github.com/graphdeco-inria/gaussian-splatting/issues/176
        features_new = self.features.clone()

        opacity_new = self.opacity.clone()

        scaling_new = self.scaling.clone()
        if scale is not None:
            scaling_new = scaling_new * scale

        if transf_mat is not None:
            quats = self.rotation                   # [N, 4] in normalized quaternions
            rotmats = quaternion_to_matrix(quats)   # [N, 3, 3]
            rotmats_new = torch.einsum('ij,njk->nik', transf_mat[:3, :3], rotmats)
            quats_new = matrix_to_quaternion(rotmats_new)
        else:
            quats_new = self.rotation.clone()

        # TODO add feature transform
        return ActGaussianModelTorch(xyz_new, features_new, opacity_new, scaling_new, quats_new, self.max_sh_degree)


@dataclass
class PreactGaussianModelNp:
    xyz: np.ndarray
    features_dc: np.ndarray
    features_rest: np.ndarray
    opacity: np.ndarray
    scaling: np.ndarray
    rotation: np.ndarray
    max_sh_degree: int

    def activate(self) -> ActGaussianModelTorch:
        xyz = torch.from_numpy(self.xyz).float().contiguous().cuda()
        features_dc = torch.from_numpy(self.features_dc).float().contiguous().cuda()
        features_rest = torch.from_numpy(self.features_rest).float().contiguous().cuda()
        features = torch.cat((features_dc, features_rest), dim=1)
        opacity = torch.sigmoid(torch.from_numpy(self.opacity).float().contiguous()).cuda()
        scaling = torch.exp(torch.from_numpy(self.scaling).float().contiguous()).cuda()
        rotation = torch.nn.functional.normalize(torch.from_numpy(self.rotation).float().contiguous()).cuda()
        max_sh_degree = self.max_sh_degree
        
        return ActGaussianModelTorch(
            xyz=xyz,
            features=features,
            opacity=opacity,
            scaling=scaling,
            rotation=rotation,
            max_sh_degree=max_sh_degree,
        )
        
def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    max_sh_degree = round(math.sqrt((len(extra_f_names) + 3) / 3) - 1)
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    
    
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    # self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    # self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    # self.active_sh_degree = self.max_sh_degree

    return PreactGaussianModelNp(xyz, features_dc.transpose(0,2,1), features_extra.transpose(0,2,1), opacities, scales, rots, max_sh_degree)
