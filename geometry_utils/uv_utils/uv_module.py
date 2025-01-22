from typing_extensions import Optional, Tuple, Union

import torch
from drtk.rasterize import rasterize

def convert_edge_uv_to_uv_mesh(edge_uv: torch.Tensor):
    """
    edge_uv: [N_faces, 3, 2] tensor storing edge uv coords, compatible with PLY formats.
    """
    n_tris = edge_uv.shape[0]
    uv_verts = edge_uv.reshape(n_tris*3, 2)  # [N_f * 3, 2]
    uv_faces = torch.linspace(0, n_tris*3-1,
                              n_tris*3,
                              device=edge_uv.device,
                              dtype=torch.int32).reshape(n_tris, 3)
    return uv_verts, uv_faces


def get_uv_face_index_map(
    uv_verts: torch.Tensor,
    uv_faces: torch.Tensor,
    uv_size: Union[Tuple[int, int], int],
    flip_v: bool = True,
) -> torch.Tensor:
    """
    uv_verts: [Nv_uv, 2], uv-space vertices
    uv_faces: [Nf, 3] triangles
    uv_size: (H, W) or just int

    out: [H, W], with -1 denoting no triangles
    """

    if isinstance(uv_size, int):
        uv_size = (uv_size, uv_size)

    if flip_v:
        uv_verts = torch.stack([uv_verts[..., 0], 1 - uv_verts[..., 1]], dim=-1)

    # scale uv coords from range [0, 1] to pixel space [-0.5, uv_size-0.5] (see rasterize)
    u_pixel_space = uv_verts[..., 0] * uv_size[1] - 0.5
    v_pixel_space = uv_verts[..., 1] * uv_size[0] - 0.5
    z_pixel_space = torch.ones_like(u_pixel_space)  # pad with ones for z (see rasterize)
    uvz_pixel_space = torch.stack([u_pixel_space,
                                   v_pixel_space,
                                   z_pixel_space], dim=-1)     # [N_f*3, 3]

    index_map = rasterize(uvz_pixel_space[None], uv_faces, uv_size[0], uv_size[1])
    return index_map.squeeze(0)


def get_bary_coords_2d(query_coords: torch.Tensor, target_tris: torch.Tensor):
    """
    Assuming we already know the triangles of each query point, i.e.,
    dim 0 of query_coords and target_triangles are in correspondence
    
    query_coords: [Nv, 2 (uv coords)]
    target_tris: [Nv, 3 (vertices of the tri), 2 (uv coords)]
    """
    # a, b, c <=> 0, 1, 2 (at dim 1)
    dx_ab = target_tris[:, 1, 0] - target_tris[:, 0, 0]    # [Nv,]
    dx_ac = target_tris[:, 2, 0] - target_tris[:, 0, 0]    # [Nv,]
    dy_ab = target_tris[:, 1, 1] - target_tris[:, 0, 1]    # [Nv,]
    dy_ac = target_tris[:, 2, 1] - target_tris[:, 0, 1]    # [Nv,]
    # q <=> query point
    dx_aq = query_coords[:, 0] - target_tris[:, 0, 0]    # [Nv,]
    dy_aq = query_coords[:, 1] - target_tris[:, 0, 1]    # [Nv,]

    # twice the area (darea, d for double)
    darea_abc = dx_ab * dy_ac - dx_ac * dy_ab
    darea_abq = dx_ab * dy_aq - dx_aq * dy_ab
    darea_aqc = dx_aq * dy_ac - dx_ac * dy_aq

    eps = 1e-8
    darea_abc_sign = (darea_abc >= 0).float() * 2 - 1
    darea_abc = darea_abc_sign * darea_abc.abs().clamp(min=eps)

    bary_c = darea_abq / darea_abc
    bary_b = darea_aqc / darea_abc
    bary_a = 1 - bary_b - bary_c

    bary_coords = torch.stack([bary_a, bary_b, bary_c], dim=-1)
    return bary_coords


def get_uv_assets(
    faces: torch.Tensor,
    uv_verts: torch.Tensor,
    uv_faces: torch.Tensor,
    uv_size: Union[Tuple[int, int], int],
    flip_v: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    get the following assets for uv-based utils:

    vert_index_map: [H, W, 3]
    face_index_map: [H, W]
    bary_coords_map: [H, W, 3]
    uv_mask: [H, W]
    """

    assert faces.shape[0] == uv_faces.shape[0]

    if isinstance(uv_size, int):
        uv_size = (uv_size, uv_size)

    if flip_v:
        uv_verts = torch.stack([uv_verts[..., 0], 1-uv_verts[..., 1]], dim=-1)

    # get face index map
    face_index_map = get_uv_face_index_map(uv_verts, uv_faces, uv_size, flip_v=False)   # [h, w]
    uv_mask = (face_index_map >= 0)

    # get uv index map
    vert_index_map = faces[face_index_map.clamp(min=0)]           # [h, w, 3]
    vert_index_map[~uv_mask] = -1   # reset invalid
    
    # get uv coords map
    # NOTE This is NOT THE REAL vert_index_map because it is generated from `uv_faces`
    uv_vert_index_map = uv_faces[face_index_map.clamp(min=0)]           # [h, w, 3]
    uv_vert_index_map[~uv_mask] = -1   # reset invalid
    uv_coords_of_faces_map = uv_verts[uv_vert_index_map.clamp(min=0)]    # [h, w, 3, 2]
    # NOTE each pixel in `uv_coords_of_faces_map` contains
    # the uv coords (dim == -1) of the three verts (dim == -2) of the triangle at that pixel

    # NOTE below is the original uv grid definition
    # u_pixel_coord_grid = torch.linspace(0, uv_size[1]-1, uv_size[1])  # pixel space: [0, W-1]
    # u_coord_grid = (u_coord_grid + 0.5) / uv_size[1]    # pixel space => uv space
    # NOTE but can be simplified as
    u_coord_grid = torch.linspace(0.5, uv_size[1]-0.5, uv_size[1], device=uv_verts.device) / uv_size[1]
    v_coord_grid = torch.linspace(0.5, uv_size[0]-0.5, uv_size[0], device=uv_verts.device) / uv_size[0]
    uv_coord_grid = torch.stack([u_coord_grid[None, :].expand(uv_size[0], -1),
                                 v_coord_grid[:, None].expand(-1, uv_size[1])], dim=-1) # [h, w, 2]

    # get bary coords
    maskindexed_uv_coords_of_faces_map = uv_coords_of_faces_map[uv_mask]  # [N, 3, 2]
    maskindexed_uv_coord_grid = uv_coord_grid[uv_mask]                    # [N, 2]
    maskindexed_bary_coords = get_bary_coords_2d(maskindexed_uv_coord_grid, maskindexed_uv_coords_of_faces_map) # [N, 3]
    # bary_recon = torch.einsum('nb,nbc->nc', maskindexed_bary_coords, maskindexed_uv_coords_of_faces_map)  # for debug only (success)

    bary_coords_map = torch.zeros(uv_size + (3,), dtype=uv_coords_of_faces_map.dtype, device=uv_coords_of_faces_map.device)
    bary_coords_map[uv_mask] = maskindexed_bary_coords

    return vert_index_map, face_index_map, bary_coords_map, uv_mask

def vert_attr_to_uv(vert_attrs, vert_index_map, bary_coords_map, uv_mask):
    """
    vert_attrs: [B, N, C]
    vert_index_map: [H, W, 3]
    bary_coords_map: [H, W, 3]
    uv_mask: [H, W]
    """
    B, _, C = vert_attrs.shape
    H, W = uv_mask.shape

    maskindexed_vert_index_map = vert_index_map[uv_mask]    # [Np, 3]
    maskindexed_bary_coords_map = bary_coords_map[uv_mask]  # [Np, 3]
    maskindexed_vert_attrs = vert_attrs[:, maskindexed_vert_index_map]  # [B, Np, 3, C]
    maskindexed_vert_attrs_barysummed = torch.einsum('kpbc,pb->kpc',
                                                     maskindexed_vert_attrs,
                                                     maskindexed_bary_coords_map
                                                     )  # [B, Np, C]
    vert_attr_map = torch.zeros((B, H, W, C), dtype=vert_attrs.dtype, device=vert_attrs.device)
    vert_attr_map[:, uv_mask] = maskindexed_vert_attrs_barysummed
    return vert_attr_map

def face_attr_to_uv(face_attrs, face_index_map, uv_mask):
    """
    face_attrs: [B, N, C]
    face_index_map: [H, W, 3]
    uv_mask: [H, W]
    """
    B, _, C = face_attrs.shape
    H, W = uv_mask.shape

    maskindexed_face_index_map = face_index_map[uv_mask]    # [Np,]
    maskindexed_face_attrs = face_attrs[:, maskindexed_face_index_map]  # [B, Np, C]

    face_attr_map = torch.zeros((B, H, W, C), dtype=face_attrs.dtype, device=face_attrs.device)
    face_attr_map[:, uv_mask] = maskindexed_face_attrs
    return face_attr_map    # [H, W, C]
