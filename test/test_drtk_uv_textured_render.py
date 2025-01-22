import os
import cv2
import numpy as np
import torch
from geometry_utils import UnifiedCamera
from geometry_utils import render_drtk_uv_textured

def load_pickle(path):
    import pickle
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data    

in_path = '/home/trisst/data/cvdata_archive/4D-Dress/4D-DRESS_00122/Outer/Take9'
frame_id = 68
out_name = f'00122_mesh_{frame_id:05d}.ply'

atlas_pkl = os.path.join(in_path, 'Meshes_pkl', f'atlas-f{frame_id:05d}.pkl')
mesh_pkl = os.path.join(in_path, 'Meshes_pkl', f'mesh-f{frame_id:05d}.pkl')
cam_pkl = os.path.join(in_path, 'Capture', f'cameras.pkl')

atlas_data = load_pickle(atlas_pkl) # [1024, 1024, 3], don't know what it is
mesh_data = load_pickle(mesh_pkl)   # vertices, normals, uvs, faces, colors, uv_path
cam_data = load_pickle(cam_pkl)

intr = cam_data['0076']['intrinsics']

camera = UnifiedCamera.from_4d_dress(
    intr,
    cam_data['0076']['extrinsics'],
    h=1280,
    w=940,
).to_drtk_format().float().cuda()

verts = torch.from_numpy(mesh_data['vertices']).float().cuda()
faces = torch.from_numpy(mesh_data['faces']).int().cuda()
uv_verts = torch.from_numpy(mesh_data['uvs']).float().cuda()
uv_faces = faces.clone()
texture_img = torch.from_numpy(atlas_data).float().cuda().permute(2,0,1)[None]

rendered_img = render_drtk_uv_textured(camera, verts[None], faces, uv_verts, uv_faces, texture_img)
cv2.imwrite('test_drtk_uv_textured_render.png', rendered_img[0].permute(1,2,0).detach().clamp(min=0,max=255).cpu().numpy().astype(np.uint8)[..., ::-1])