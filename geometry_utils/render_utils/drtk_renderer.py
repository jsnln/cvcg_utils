import os
import igl
import numpy as np
import drtk
import drtk.interpolate_ext
import torch
from typing import Callable
from torchvision.utils import save_image  # to save images
from ..mesh_proc.mesh_proc import get_vert_normals, get_face_normals
from .camera import DRTKCamera

def render_drtk_face_attr(
        camera: DRTKCamera,
        verts: torch.Tensor,
        faces: torch.Tensor,
        face_attrs: torch.Tensor,
        bg_attr: torch.Tensor):
    
    """
    verts: batched [B, Nv, 3]
    faces: unbatched [Nf, 3], shared by items in the vertex batch
    face_attrs: unbatched [Nf, C], shared by items in the vertex batch
    bg_attr: [C], shared by items in the vertex batch
    """
    assert len(verts.shape) == 3
    assert len(faces.shape) == 2
    assert len(face_attrs.shape) == 2
    assert faces.shape[0] == face_attrs.shape[0]
    assert face_attrs.shape[1] == bg_attr.shape[0]

    pts_screen = camera.proj_points_to_drtk_screen(verts, detach_z=True)  # [B, N, 3]

    face_index_img = drtk.rasterize(pts_screen, faces, height=camera.H, width=camera.W)   # [B, H, W]
    mask = (face_index_img > -1)  # [B, H, W]
    # mask_float = mask.float()  # [B, H, W]
    
    # face_index_img[~mask] = 0
    face_attr_img = face_attrs[face_index_img]        # [Nf, C] indexed by [B, H, W] => [B, H, W, C]
    face_attr_img = face_attr_img * mask[..., None]   # re-mask
    bg_img = bg_attr[None, None, None] * (~mask)[..., None]    # [1, 1, 1, C] * [B, H, W, 1] => [B, H, W, C]

    out_img = face_attr_img + bg_img

    return out_img

def render_drtk_shaded(
        camera: DRTKCamera,
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
    assert len(verts.shape) == 3
    assert len(faces.shape) == 2

    pts_screen = camera.proj_points_to_drtk_screen(verts, detach_z=True)  # [B, N, 3]

    face_index_img = drtk.rasterize(pts_screen, faces, height=camera.H, width=camera.W)   # [B, H, W]
    mask = (face_index_img > -1)  # [B, H, W]
    # face_index_img[~mask] = 0 # NOTE you can't do this!!! otherwise there's no edge grad
    
    _, bary_img = drtk.render(pts_screen, faces, face_index_img)    # [B, 3, H, W]   


    if shading_mode == 'face':
        face_normals = get_face_normals(verts, faces, normalize=True)   # [B, Nf, 3]
        b, h, w = face_index_img.shape
        face_index_img_flat = face_index_img.reshape(b, -1, 1).expand(-1, -1, 3).clone()    # [B, H*W, 3]
        face_index_img_flat[face_index_img_flat == -1] = 0
        face_normal_img_flat = torch.gather(input=face_normals, dim=-2, index=face_index_img_flat.long())  # [B, Nf, 3] indexed by [B, H*W, 3] => [B, H*W, 3]
        face_normal_img = face_normal_img_flat.reshape(b, h, w, 3)  # [B, H, W, 3]

        shaded_img = shading_func(face_normal_img, face_index_img)  # [B, H, W, 3]
        shaded_img = shaded_img.permute(0,3,1,2)    # [B, 3, H, W]

    elif shading_mode == 'vert':
        vert_normals = get_vert_normals(verts, faces, normalize=True)   # [B, Nv, 3]
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

def get_batched_ao(verts: torch.Tensor, faces: torch.Tensor):
    vert_normals = get_vert_normals(verts, faces, normalize=True)   # [B, Nv, 3]
    # Compute ambient occlusion factor using embree
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    vert_normals_np = vert_normals.detach().cpu().numpy()

    import time
    time_s = time.time()
    ao_list = []
    for i in range(verts.shape[0]):
        ao = 1 - igl.ambient_occlusion(verts_np[i], faces_np, verts_np[i], vert_normals_np[i], 500) # shape = [Nv,]
        ao_list.append(ao)
    time_e = time.time()
    print(f'[LOG] ao: {time_e - time_s:.4e}')

    batched_ao = np.stack(ao_list, axis=0)
    batched_ao = torch.from_numpy(batched_ao).to(verts.dtype).to(verts.device)
    return batched_ao


def render_drtk_ao(
        camera: DRTKCamera,
        verts: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
    
    """
    verts: batched [B, Nv, 3]
    faces: unbatched [Nf, 3], shared by items in the vertex batch

    shading_func: a callable that turns normals [..., 3] to rgb [..., 3]
    
    face_attrs: unbatched [Nf, C], shared by items in the vertex batch
    bg_attr: [C], shared by items in the vertex batch
    """
    assert len(verts.shape) == 3
    assert len(faces.shape) == 2

    pts_screen = camera.proj_points_to_drtk_screen(verts, detach_z=True)  # [B, N, 3]

    face_index_img = drtk.rasterize(pts_screen, faces, height=camera.H, width=camera.W)   # [B, H, W]
    mask = (face_index_img > -1)  # [B, H, W]
    # face_index_img[~mask] = 0 # NOTE you can't do this!!! otherwise there's no edge grad
    
    _, bary_img = drtk.render(pts_screen, faces, face_index_img)    # [B, 3, H, W]   


    vert_ao = get_batched_ao(verts, faces)   # [B, Nv]
    vert_ao_img = drtk.interpolate(vert_ao[..., None], faces, face_index_img, bary_img)   # [B, 1, H, W]
    vert_ao_img = vert_ao_img * mask[:, None]

    # if make_differentiable:
    #     shaded_img = drtk.edge_grad_estimator(
    #         pts_screen,     # verts for rasterization
    #         faces,          # faces
    #         bary_img,       # barys
    #         shaded_img, # rendered image
    #         face_index_img)  # face indices

    return vert_ao_img

def render_drtk_uv_textured(
        camera: DRTKCamera,
        verts: torch.Tensor,
        faces: torch.Tensor,
        uv_verts: torch.Tensor,
        uv_faces: torch.Tensor,
        texture_img: torch.Tensor,
        vert_ao: torch.Tensor,
        make_differentiable: bool = False,
        flip_v: bool = True,
    ):
    
    """
    verts: batched [B, Nv, 3]
    faces: unbatched [Nf, 3], shared by items in the vertex batch
    uv_verts: unbatched [Nv, 3]
    uv_faces: unbatched [Nf, 3], shared by items in the vertex batch
    texture_img: [B, C, H, W]
    
    face_attrs: unbatched [Nf, C], shared by items in the vertex batch
    bg_attr: [C], shared by items in the vertex batch
    """
    assert len(verts.shape) == 3
    assert len(faces.shape) == 2
    # assert len(face_attrs.shape) == 2
    # assert faces.shape[0] == face_attrs.shape[0]
    # assert face_attrs.shape[1] == bg_attr.shape[0]

    if flip_v:
        uv_verts = torch.stack([uv_verts[..., 0], 1 - uv_verts[..., 1]], dim=-1)

    pts_screen = camera.proj_points_to_drtk_screen(verts, detach_z=True)  # [B, N, 3]

    face_index_img = drtk.rasterize(pts_screen, faces, height=camera.H, width=camera.W)   # [B, H, W]
    mask = (face_index_img > -1)  # [B, H, W]
    # face_index_img[~mask] = 0 # NOTE you can't do this!!! otherwise there's no edge grad
    
    _, bary_img = drtk.render(pts_screen, faces, face_index_img)    # [B, 3, H, W]    
    uv_img = drtk.interpolate(uv_verts[None].expand(verts.shape[0], -1, -1), uv_faces, face_index_img, bary_img)    # [B, 2, H, W]
    
    # with torch.no_grad():
    #     if vert_ao is not None:
    #         ao_img = drtk.interpolate(vert_ao, faces, face_index_img, bary_img)    # [B, 1, H, W]

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

    if vert_ao is not None:
        raise NotImplementedError
    #     textured_rgb_image = textured_render_img[:, :3]
    #     textured_rgb_image_wao = textured_rgb_image * ao_img
    #     textured_seg_image = textured_render_img[:, 3:]
        

    #     textured_render_img = torch.cat([textured_rgb_image_wao, textured_seg_image], dim=1)
    #     return textured_render_img, textured_rgb_image, mask, ao_img
    else:
        return textured_render_img, mask, face_index_img



if __name__ == '__main__':


    # v = torch.as_tensor([[[0, 511, 1], [255, 0, 1], [511, 511, 1]]]).float().cuda()
    # vi = torch.as_tensor([[0, 1, 2]]).int().cuda()
    # index_img = drtk.rasterize(v, vi, height=512, width=512)
    # _, bary = drtk.render(v, vi, index_img)
    # img = bary * (index_img != -1)

    # save_image(img, "render.png")
    pass
