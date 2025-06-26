import os
import igl
import numpy as np
import drtk
import drtk.interpolate_ext
import torch
from typing import Callable, Union
from torchvision.utils import save_image  # to save images
from ..mesh.mesh_proc import get_vert_normals, get_face_normals
from .camera import DRTKCamera, BatchDRTKCamera

def render_drtk_face_attr(
        camera: Union[DRTKCamera, BatchDRTKCamera],
        verts: torch.Tensor,
        faces: torch.Tensor,
        face_attrs: torch.Tensor,
        make_differentiable=False):
    
    """
    supports two batching modes: camera batching and vert batching

    the function uses `isinstance(camera, DRTKCamera)` to determine which batching type to use

    only one of `camera` and `verts` is allowed to be batched

    camera: unbatched type DRTKCamera or batched type BatchDRTKCamera
    verts: unbatched [Nv, 3], or batched [B, Nv, 3]
    faces: unbatched [Nf, 3], shared by items in the vertex batch
    face_attrs: unbatched [Nf, C], shared by items in the vertex batch
    bg_attr: [C], shared by items in the vertex batch
    """
    # determine batching mode
    batched = True      # whether at least one of `camera` or `verts` is batched
    if isinstance(camera, BatchDRTKCamera):     # if `camera` is batched, then either `verts` is unbatched or has the same batch size
        assert len(verts.shape) == 2 or \
              (len(verts.shape) == 3 and verts.shape[0] == camera.batch_size)
    elif isinstance(camera, DRTKCamera):        # if `camera` is unbatched, then `verts` can be batched
        if len(verts.shape) == 2:
            batched = False
            verts = verts[None]

    assert len(faces.shape) == 2
    assert len(face_attrs.shape) == 2
    assert faces.shape[0] == face_attrs.shape[0]

    pts_screen = camera.proj_points_to_drtk_screen(verts, detach_z=False)  # [B, N, 3]

    face_index_img = drtk.rasterize(pts_screen, faces, height=camera.H, width=camera.W)   # [B, H, W]
    mask = (face_index_img > -1)  # [B, H, W]
    # mask_float = mask.float()  # [B, H, W]
    
    depth_img, bary_img = drtk.render(pts_screen, faces, face_index_img)    # [B, 3, H, W]
    
    # face_index_img[~mask] = 0     # NOTE you can't do this!!! otherwise there's no edge grad
    face_attr_img = face_attrs[face_index_img].permute(0, 3, 1, 2)        # [Nf, C] indexed by [B, H, W] => [B, H, W, C] => [B, C, H, W], to be consistent with other APIs
    face_attr_img = face_attr_img * mask[:, None]   # re-mask

    if make_differentiable:
        face_attr_img = drtk.edge_grad_estimator(
            pts_screen,     # verts for rasterization
            faces,          # faces
            bary_img,       # barys
            face_attr_img,  # rendered image
            face_index_img)  # face indices
        
        mask = drtk.edge_grad_estimator(
            pts_screen,     # verts for rasterization
            faces,          # faces
            bary_img,       # barys
            mask.float()[:, None],  # rendered image
            face_index_img, # face indices
            ).squeeze(1)    # [B, H, W]

    if not batched:
        face_attr_img = face_attr_img.squeeze(0)
        depth_img = depth_img.squeeze(0)
        mask = mask.squeeze(0)
        face_index_img = face_index_img.squeeze(0)

    return face_attr_img, depth_img, mask, face_index_img


def render_drtk_vert_attr(
        camera: Union[DRTKCamera, BatchDRTKCamera],
        verts: torch.Tensor,
        faces: torch.Tensor,
        vert_attrs: torch.Tensor,
        attr_faces: torch.Tensor,   # this is to support a different attr buffer, e.g., uvs
        make_differentiable: bool = False,
    ):
    
    """
    similar to `render_drtk_face_attr`, supports two batching modes

    whether `vert_attrs` is batched should be consistent with `camera` and `verts`

    attr_faces: unbatched [Nf, 3]

    if verts and vert_attrs align, then attr_faces == faces,
    but distinguishing them allows other cases, e.g., uvs
    """
    assert len(verts.shape) == len(vert_attrs.shape)
    if len(vert_attrs.shape) == 3 and len(verts.shape) == 3:
            assert verts.shape[0] == vert_attrs.shape[0]
    
    # determine batching mode
    batched = True      # whether at least one of `camera` or `verts` is batched
    if isinstance(camera, BatchDRTKCamera):     # if `camera` is batched, then either `verts` is unbatched or has the same batch size
        assert len(verts.shape) == 2 or \
              (len(verts.shape) == 3 and verts.shape[0] == camera.batch_size)

        # additionally check whether `vert_attrs` is batched
        if len(vert_attrs.shape) == 2:
            vert_attrs = vert_attrs[None].expand(camera.batch_size, -1, -1)

    elif isinstance(camera, DRTKCamera):        # if `camera` is unbatched, then `verts` can be batched
        if len(verts.shape) == 2:
            batched = False
            verts = verts[None]
            vert_attrs = vert_attrs[None]


    assert len(faces.shape) == 2
    assert len(attr_faces.shape) == 2
    assert faces.shape[0] == attr_faces.shape[0]

    pts_screen = camera.proj_points_to_drtk_screen(verts, detach_z=False)  # [B, N, 3]

    face_index_img = drtk.rasterize(pts_screen, faces, height=camera.H, width=camera.W)   # [B, H, W]
    mask = (face_index_img > -1)  # [B, H, W]
    # face_index_img[~mask] = 0 # NOTE you can't do this!!! otherwise there's no edge grad
    
    depth_img, bary_img = drtk.render(pts_screen, faces, face_index_img)    # [B, 3, H, W]
    vert_attr_img = drtk.interpolate(vert_attrs, attr_faces, face_index_img, bary_img)   # [B, 3, H, W]

    vert_attr_img = vert_attr_img * mask[:, None]

    if make_differentiable:
        vert_attr_img = drtk.edge_grad_estimator(
            pts_screen,     # verts for rasterization
            faces,          # faces
            bary_img,       # barys
            vert_attr_img, # rendered image
            face_index_img)  # face indices
        
        mask = drtk.edge_grad_estimator(
            pts_screen,     # verts for rasterization
            faces,          # faces
            bary_img,       # barys
            mask.float()[:, None],  # rendered image
            face_index_img, # face indices
            ).squeeze(1)    # [B, H, W]
        
    if not batched:
        vert_attr_img = vert_attr_img.squeeze(0)
        depth_img = depth_img.squeeze(0)
        mask = mask.squeeze(0)
        face_index_img = face_index_img.squeeze(0)

    return vert_attr_img, depth_img, mask, face_index_img


def render_drtk_shaded(
        camera: Union[DRTKCamera, BatchDRTKCamera],
        verts: torch.Tensor,
        faces: torch.Tensor,
        shading_mode: str,  # either 'vert' or 'face'
        shading_func: Callable,
        make_differentiable: bool = False,
    ):
    
    """
    verts: batched [B, Nv, 3]
    faces: unbatched [Nf, 3], shared by items in the vertex batch

    shading_func: a callable that turns normals [..., 3] to rgb [..., 3]
    
    face_attrs: unbatched [Nf, C], shared by items in the vertex batch
    bg_attr: [C], shared by items in the vertex batch
    """
    if isinstance(camera, DRTKCamera):
        assert len(verts.shape) == 3
    elif isinstance(camera, BatchDRTKCamera):
        assert len(verts.shape) == 2
    else:
        raise NotImplementedError

    assert len(faces.shape) == 2

    pts_screen = camera.proj_points_to_drtk_screen(verts, detach_z=False)  # [B, N, 3]

    face_index_img = drtk.rasterize(pts_screen, faces, height=camera.H, width=camera.W)   # [B, H, W]
    mask = (face_index_img > -1)  # [B, H, W]
    # face_index_img[~mask] = 0 # NOTE you can't do this!!! otherwise there's no edge grad
    
    _, bary_img = drtk.render(pts_screen, faces, face_index_img)    # [B, 3, H, W]

    if shading_mode == 'face':
        face_normals = get_face_normals(verts, faces, normalize=True)   # [B, Nf, 3]
        if len(face_normals.shape) == 2:
            face_normals = face_normals[None].expand(pts_screen.shape[0], -1, -1)
        b, h, w = face_index_img.shape
        face_index_img_flat = face_index_img.reshape(b, -1, 1).expand(-1, -1, 3).clone()    # [B, H*W, 3]
        face_index_img_flat[face_index_img_flat == -1] = 0
        face_normal_img_flat = torch.gather(input=face_normals, dim=-2, index=face_index_img_flat.long())  # [B, Nf, 3] indexed by [B, H*W, 3] => [B, H*W, 3]
        face_normal_img = face_normal_img_flat.reshape(b, h, w, 3)  # [B, H, W, 3]

        shaded_img = shading_func(face_normal_img, face_index_img)  # [B, H, W, 3]
        shaded_img = shaded_img.permute(0,3,1,2)    # [B, 3, H, W]

    elif shading_mode == 'vert':
        vert_normals = get_vert_normals(verts, faces, normalize=True)   # [B, Nv, 3]
        if len(vert_normals.shape) == 2:
            vert_normals = vert_normals[None].expand(pts_screen.shape[0], -1, -1)

        vert_normal_img = drtk.interpolate(vert_normals, faces, face_index_img, bary_img).permute(0,2,3,1)   # [B, H, W, 3]

        shaded_img = shading_func(vert_normal_img, face_index_img).permute(0,3,1,2)        # [B, 3, H, W]

    shaded_img = shaded_img * mask[:, None]

    if make_differentiable:
        shaded_img = drtk.edge_grad_estimator(
            pts_screen,     # verts for rasterization
            faces,          # faces
            bary_img,       # barys
            shaded_img, # rendered image
            face_index_img)  # face indices

    return shaded_img

def render_drtk_uv_textured(
        camera: Union[DRTKCamera, BatchDRTKCamera],
        verts: torch.Tensor,
        faces: torch.Tensor,
        uv_verts: torch.Tensor,
        uv_faces: torch.Tensor,
        texture_img: torch.Tensor,
        make_differentiable: bool = False,
        flip_v: bool = True,
    ):
    
    """
    verts: batched [B, Nv, 3]
    faces: unbatched [Nf, 3], shared by items in the vertex batch
    uv_verts: unbatched [Nv, 3]
    uv_faces: unbatched [Nf, 3], shared by items in the vertex batch
    texture_img: [B, C, H, W]
    
    out:
    textured_render_img: [B, C, H, W]
    mask:   [B, H, W]
    face_index_img: ...
    """
    batched = True      # whether at least one of `camera` or `verts` is batched
    if isinstance(camera, BatchDRTKCamera):     # if `camera` is batched, then either `verts` is unbatched or has the same batch size
        assert len(verts.shape) == 2 or \
              (len(verts.shape) == 3 and verts.shape[0] == camera.batch_size)
    elif isinstance(camera, DRTKCamera):        # if `camera` is unbatched, then `verts` can be batched
        if len(verts.shape) == 2:
            batched = False
            verts = verts[None]

    assert len(faces.shape) == 2
    assert len(uv_verts.shape) == 2

    if not batched:
        assert len(texture_img.shape) == 3, "since both cameras and verts are unbatched, texture_img must also be unbatched"
        texture_img = texture_img[None]
    else:
        assert len(texture_img.shape) == 4, "since cameras and/or verts are unbatched, texture_img must also be batched"
        assert texture_img.shape[0] == camera.batch_size, "texture_img must have the same batch size as cameras/verts"



    if flip_v:
        uv_verts = torch.stack([uv_verts[..., 0], 1 - uv_verts[..., 1]], dim=-1)

    pts_screen = camera.proj_points_to_drtk_screen(verts, detach_z=False)  # [B, N, 3]

    face_index_img = drtk.rasterize(pts_screen, faces, height=camera.H, width=camera.W)   # [B, H, W]
    mask = (face_index_img > -1)  # [B, H, W]
    # face_index_img[~mask] = 0 # NOTE you can't do this!!! otherwise there's no edge grad
    
    depth_img, bary_img = drtk.render(pts_screen, faces, face_index_img)    # [B, 3, H, W]    
    uv_img = drtk.interpolate(uv_verts[None].expand(pts_screen.shape[0], -1, -1), uv_faces, face_index_img, bary_img)    # [B, 2, H, W]
    
    textured_render_img = torch.nn.functional.grid_sample(
        texture_img,
        (uv_img * 2 - 1).permute(0,2,3,1),
        align_corners=False,
        padding_mode='border',
    )   # [B, C, H, W]
    textured_render_img = textured_render_img * mask[:, None]

    if make_differentiable:
        textured_render_img = drtk.edge_grad_estimator(
            pts_screen,     # verts for rasterization
            faces,          # faces
            bary_img,       # barys
            textured_render_img, # rendered image
            face_index_img)  # face indices
        
        mask = drtk.edge_grad_estimator(
            pts_screen,     # verts for rasterization
            faces,          # faces
            bary_img,       # barys
            mask.float()[:, None],  # rendered image
            face_index_img, # face indices
            ).squeeze(1)    # [B, H, W]
        
    if not batched:
        textured_render_img = textured_render_img.squeeze(0)
        mask = mask.squeeze(0)
        depth_img = depth_img.squeeze(0)
        face_index_img = face_index_img.squeeze(0)

    return textured_render_img, depth_img, mask, face_index_img


# def render_drtk_uv_textured_batched_cams(
#         batched_cams: BatchDRTKCamera,
#         verts: torch.Tensor,
#         faces: torch.Tensor,
#         uv_verts: torch.Tensor,
#         uv_faces: torch.Tensor,
#         texture_img: torch.Tensor,
#         make_differentiable: bool = False,
#         flip_v: bool = True,
#     ):
    
#     """
#     verts: unbatched [Nv, 3]
#     faces: unbatched [Nf, 3], shared by items in the vertex batch
#     uv_verts: unbatched [Nv, 3]
#     uv_faces: unbatched [Nf, 3], shared by items in the vertex batch
#     texture_img: [C, H, W]
    
#     face_attrs: unbatched [Nf, C], shared by items in the vertex batch
#     bg_attr: [C], shared by items in the vertex batch
#     """
#     assert len(verts.shape) == 2
#     assert len(faces.shape) == 2
#     # assert len(face_attrs.shape) == 2
#     # assert faces.shape[0] == face_attrs.shape[0]
#     # assert face_attrs.shape[1] == bg_attr.shape[0]

#     if flip_v:
#         uv_verts = torch.stack([uv_verts[..., 0], 1 - uv_verts[..., 1]], dim=-1)

#     pts_screen = batched_cams.proj_points_to_drtk_screen(verts, detach_z=False)  # [B, N, 3]
#     batch_size = pts_screen.shape[0]

#     face_index_img = drtk.rasterize(pts_screen, faces, height=batched_cams.H, width=batched_cams.W)   # [B, H, W]
#     mask = (face_index_img > -1)  # [B, H, W]
#     # face_index_img[~mask] = 0 # NOTE you can't do this!!! otherwise there's no edge grad
    
#     _, bary_img = drtk.render(pts_screen, faces, face_index_img)    # [B, 3, H, W]    
#     uv_img = drtk.interpolate(uv_verts[None].expand(batch_size, -1, -1), uv_faces, face_index_img, bary_img)    # [B, 2, H, W]
    
#     textured_render_img = torch.nn.functional.grid_sample(
#         texture_img.unsqueeze(0).expand(batch_size, -1, -1, -1),
#         (uv_img * 2 - 1).permute(0,2,3,1),
#         align_corners=False,
#         padding_mode='border',
#     )   # [B, C, H, W]
#     textured_render_img = textured_render_img * mask[:, None]

#     if make_differentiable:
#         textured_render_img = drtk.edge_grad_estimator(
#             pts_screen,     # verts for rasterization
#             faces,          # faces
#             bary_img,       # barys
#             textured_render_img, # rendered image
#             face_index_img)  # face indices
        
#         mask = drtk.edge_grad_estimator(
#             pts_screen,     # verts for rasterization
#             faces,          # faces
#             bary_img,       # barys
#             mask.float()[:, None],  # rendered image
#             face_index_img, # face indices
#             ).squeeze(1)    # [B, H, W]

#     return textured_render_img, mask, face_index_img

if __name__ == '__main__':


    # v = torch.as_tensor([[[0, 511, 1], [255, 0, 1], [511, 511, 1]]]).float().cuda()
    # vi = torch.as_tensor([[0, 1, 2]]).int().cuda()
    # index_img = drtk.rasterize(v, vi, height=512, width=512)
    # _, bary = drtk.render(v, vi, index_img)
    # img = bary * (index_img != -1)

    # save_image(img, "render.png")
    pass
