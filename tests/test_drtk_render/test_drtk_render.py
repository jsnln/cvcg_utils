import torch
import numpy as np
from cvcg_utils.render.camera import UnifiedCamera, DRTKCamera
from cvcg_utils.render.drtk_renderer import render_drtk_face_attr, render_drtk_vert_attr, render_drtk_uv_textured, render_drtk_point_sprites
from cvcg_utils.mesh import read_obj, get_vert_normals, get_face_normals
from cvcg_utils.misc.image import read_rgb, write_rgb
from cvcg_utils.misc.tensor_utils import np2cuda, th2np

mesh = read_obj('../test_mesh_io/bunny_with_texture.obj')
mesh_verts = np2cuda(mesh.verts).float()
mesh_faces = np2cuda(mesh.faces).int()
mesh_uv_verts = np2cuda(mesh.uv_verts).float()
mesh_uv_faces = np2cuda(mesh.uv_faces).int()
 
# ====== EXAMPLE 1: texture render ======
texture_img = np2cuda(read_rgb('../test_mesh_io/bunny_with_texture.obj.png')).float().permute(2,0,1)
uni_camera = UnifiedCamera.from_lookat(
    np.array([0., 2., 2.]),
    np.array([0., 0,  0.]),
    np.array([0., 1., 0]),
    60, 60, 'deg', None, 512, 512)
drtk_camera = uni_camera.to_drtk_format().float().cuda()

textured_render_img, depth, mask, face_index_img = render_drtk_uv_textured(drtk_camera, mesh_verts, mesh_faces, mesh_uv_verts, mesh_uv_faces, texture_img, make_differentiable=False, flip_v=True)
textured_render_img = textured_render_img.permute(1,2,0)
write_rgb('test_drtk_texture_render.png', (textured_render_img.clamp(0,255)).byte().cpu().numpy())

# ====== EXAMPLE 2: vert attr render ======
vert_normals = get_vert_normals(mesh_verts, mesh_faces, normalize=True)
vert_normal_img, depth, mask, face_index_img = render_drtk_vert_attr(drtk_camera, mesh_verts, mesh_faces, vert_normals, mesh_faces, make_differentiable=False)
vert_normal_img = vert_normal_img.permute(1,2,0)
vert_normal_img = (((vert_normal_img + 1) / 2).clamp(0,1) * 255)
vert_normal_img[~mask] = 0
write_rgb('test_drtk_vert_attr_render.png', vert_normal_img.byte().cpu().numpy())

# ====== EXAMPLE 3: face attr render ======
face_normals = get_face_normals(mesh_verts, mesh_faces, normalize=True)
face_normal_img, depth, mask, face_index_img = render_drtk_face_attr(drtk_camera, mesh_verts, mesh_faces, face_normals, make_differentiable=False)
face_normal_img = face_normal_img.permute(1,2,0)
face_normal_img = (((face_normal_img + 1) / 2).clamp(0,1) * 255)
face_normal_img[~mask] = 0
write_rgb('test_drtk_face_attr_render.png', face_normal_img.byte().cpu().numpy())

# ====== EXAMPLE 4: point sprite render ======
point_img, depth, mask, face_index_img = render_drtk_point_sprites(drtk_camera, mesh_verts, vert_normals, point_size=0.01)
point_img = point_img.permute(1,2,0)
point_img = (((point_img + 1) / 2).clamp(0,1) * 255)
point_img[~mask] = 0
point_mask = (mask * 255).float()[..., None].expand(-1, -1, 3)
point_test_render = torch.cat([point_img, point_mask], dim=1)

write_rgb('test_drtk_point_attr_render.png', point_test_render.byte().cpu().numpy())

