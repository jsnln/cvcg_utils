import os
import drtk
import drtk.interpolate_ext
import torch
import torch.nn.functional as F

from torchvision.utils import save_image  # to save images
from .camera import UnifiedCamera, DRTKCamera
from .gs_renderer import render_gs

def render_gs_and_drtk_uv_textured(
        uni_camera: UnifiedCamera,  # for gs
        gs_xyz: torch.Tensor,
        gs_opa: torch.Tensor,
        gs_cov: torch.Tensor,   # [N, 6]
        gs_rgb: torch.Tensor,
        gs_lab: torch.Tensor,   # [N, 3], onehot labels
        m_camera: DRTKCamera,
        m_verts: torch.Tensor,
        m_faces: torch.Tensor,
        m_uv_verts: torch.Tensor,
        m_uv_faces: torch.Tensor,
        m_texture_img: torch.Tensor,
        m_label_img: torch.Tensor,
        m_make_differentiable: bool = False,
        m_detach_z: bool = False,
        m_flip_v: bool = True,
    ):
    
    """
    for compatibility, both GS and DRTK use unbatched data here

    verts: batched [Nv, 3]
    faces: unbatched [Nf, 3], shared by items in the vertex batch
    uv_verts: unbatched [Nv, 3]
    uv_faces: unbatched [Nf, 3], shared by items in the vertex batch
    texture_img: [C, H, W]
    
    face_attrs: unbatched [Nf, C], shared by items in the vertex batch
    bg_attr: [C], shared by items in the vertex batch
    """
    assert len(m_verts.shape) == 2
    assert len(m_faces.shape) == 2
    assert len(m_texture_img.shape) == 3
    assert len(m_label_img.shape) == 3
    # assert len(face_attrs.shape) == 2
    # assert faces.shape[0] == face_attrs.shape[0]
    # assert face_attrs.shape[1] == bg_attr.shape[0]

    if m_flip_v:
        m_uv_verts = torch.stack([m_uv_verts[..., 0], 1 - m_uv_verts[..., 1]], dim=-1)

    # Step 1: rasterize to depth
    # intermediate data below are batched just for convenience
    m_pts_screen = m_camera.proj_points_to_drtk_screen(m_verts[None], detach_z=m_detach_z)  # [B=1, N, 3]
    m_face_index_img = drtk.rasterize(m_pts_screen, m_faces, height=m_camera.H, width=m_camera.W)   # [B=1, H, W]
    m_depth_img, m_bary_img = drtk.render(m_pts_screen, m_faces, m_face_index_img)    # [B=1, 3, H, W]

    m_mask = (m_face_index_img > -1)  # [B, H, W]
    m_depth_img[~m_mask] = 1.   # farthest z coord in clip space (infinitely far)

    # Step 2: render mesh
    m_uv_img = drtk.interpolate(m_uv_verts[None], m_uv_faces, m_face_index_img, m_bary_img)    # [B, 2, H, W]
    m_texture_label_img = torch.cat([m_texture_img, m_label_img], dim=0)
    m_render_img = torch.nn.functional.grid_sample(
        m_texture_label_img[None],
        (m_uv_img * 2 - 1).permute(0,2,3,1),
        align_corners=False
    )   # [B, C, H, W]
    m_render_img = m_render_img * m_mask[:, None]   # [B, C, H, W]

    if m_make_differentiable:
        m_render_img = drtk.edge_grad_estimator(
            m_pts_screen,     # verts for rasterization
            m_faces,          # faces
            m_bary_img,       # barys
            m_render_img, # rendered image
            m_face_index_img)  # face indices
        m_render_img = m_render_img.squeeze(0)  # [C, H, W]

        m_mask = drtk.edge_grad_estimator(
            m_pts_screen,     # verts for rasterization
            m_faces,          # faces
            m_bary_img,       # barys
            m_mask.float()[:, None], # rendered image
            m_face_index_img)  # face indices
        m_mask = m_mask.squeeze(0).squeeze(0)  # [H, W]
    
        
    # Step 3: gs to screen space
    with torch.no_grad():
        gs_pts_screen = m_camera.proj_points_to_grid_sample(gs_xyz[None], False)       # [B=1, N, 3]
        m_depth_at_gs_proj_loc = F.grid_sample(m_depth_img[:, None],                     # [B, 1, H, W]
                                            gs_pts_screen[..., None, :2],    # [B, N, 1 (aux W), 2]
                                            align_corners=False,
                                            padding_mode='border')             # [B, 1 (channel), N, 1 (aux W)]
        m_depth_at_gs_proj_loc = m_depth_at_gs_proj_loc.squeeze(1).squeeze(-1)  # [B, N]

        gs_visible_mask = (gs_pts_screen[..., 2] < m_depth_at_gs_proj_loc).squeeze(0)  # [N]

    if gs_visible_mask.sum() > 0:
        gs_visible_xyz = gs_xyz[gs_visible_mask]
        gs_visible_opa = gs_opa[gs_visible_mask]
        gs_visible_cov = gs_cov[gs_visible_mask]
        gs_visible_rgb = gs_rgb[gs_visible_mask]
        gs_visible_lab = gs_lab[gs_visible_mask]

        gs_rendered_rgb = render_gs(uni_camera, gs_visible_xyz, gs_visible_opa, scales=None, rotations=None, cov3D_precomp=gs_visible_cov, override_color=gs_visible_rgb, clip_value=False)['render']
        gs_rendered_lab = render_gs(uni_camera, gs_visible_xyz, gs_visible_opa, scales=None, rotations=None, cov3D_precomp=gs_visible_cov, override_color=gs_visible_lab, clip_value=False)['render']
        gs_rendered_opa = gs_rendered_lab[[0]]  # [1, H, W]

        final_render_rgb = gs_rendered_opa * gs_rendered_rgb + (1 - gs_rendered_opa) * m_render_img[:3]  # [C, H, W]
        final_render_msk = (gs_rendered_opa + (1 - gs_rendered_opa) * m_mask[None]).squeeze(0)  # [H, W]
        final_render_lab = gs_rendered_opa * gs_rendered_lab[1:] + (1 - gs_rendered_opa) * m_render_img[4:]  # [C, H, W]

    else:
        final_render_rgb = m_render_img[:3]  # [C, H, W]
        final_render_msk = m_mask  # [H, W]
        final_render_lab = m_render_img[4:]  # [C, H, W]

    return final_render_rgb, final_render_msk, final_render_lab


if __name__ == '__main__':

    # v = torch.as_tensor([[[0, 511, 1], [255, 0, 1], [511, 511, 1]]]).float().cuda()
    # vi = torch.as_tensor([[0, 1, 2]]).int().cuda()
    # index_img = drtk.rasterize(v, vi, height=512, width=512)
    # _, bary = drtk.render(v, vi, index_img)
    # img = bary * (index_img != -1)

    # save_image(img, "render.png")
    pass
