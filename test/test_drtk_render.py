import numpy as np
import os
import pickle
import igl
import torch
from geometry_utils import write_ply
from geometry_utils import UnifiedCamera, DRTKCamera, render_drtk_face_attr

def load_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def convert_vert_labels_to_faces(faces, vert_labels):
    vert_labels = vert_labels.astype(np.int64)
    num_cats = vert_labels.max() + 1

    face_labels_0 = vert_labels[faces[:, 0]]
    face_labels_1 = vert_labels[faces[:, 1]]
    face_labels_2 = vert_labels[faces[:, 2]]

    face_onehot_labels = np.zeros((faces.shape[0], num_cats), dtype=np.int64)
    fid_range = np.arange(faces.shape[0])
    face_onehot_labels[fid_range, face_labels_0] += 1
    face_onehot_labels[fid_range, face_labels_1] += 1
    face_onehot_labels[fid_range, face_labels_2] += 1

    return face_onehot_labels

in_path = '/home/trisst/data/cvdata_archive/4D-Dress/4D-DRESS_00122/Outer/Take9'
frame_id = 11
out_name_vq = f'00122_mesh_{frame_id:05d}.ply'
out_name_fq = f'00122_mesh_{frame_id:05d}_facelabels.ply'


atlas_pkl = os.path.join(in_path, 'Meshes_pkl', f'atlas-f{frame_id:05d}.pkl')
mesh_pkl = os.path.join(in_path, 'Meshes_pkl', f'mesh-f{frame_id:05d}.pkl')
sema_pkl = os.path.join(in_path, 'Semantic', 'labels', f'label-f{frame_id:05d}.pkl')
smpl_pkl = os.path.join(in_path, 'SMPL', f'mesh-f{frame_id:05d}_smpl.pkl')
smplx_pkl = os.path.join(in_path, 'SMPLX', f'mesh-f{frame_id:05d}_smplx.pkl')
cam_pkl = os.path.join(in_path, 'Capture', f'cameras.pkl')

atlas_data = load_pickle(atlas_pkl) # [1024, 1024, 3], don't know what it is
mesh_data = load_pickle(mesh_pkl)   # vertices, normals, uvs, faces, colors, uv_path
sema_data = load_pickle(sema_pkl)
cam_data = load_pickle(cam_pkl)
smplx_data = load_pickle(smplx_pkl)


face_onehot_labels = convert_vert_labels_to_faces(mesh_data['faces'], sema_data['scan_labels'])
face_labels = np.argmax(face_onehot_labels, axis=1)

write_ply(out_name_vq, mesh_data['vertices'], mesh_data['faces'], v_color=mesh_data['colors'][:, :3], v_quality=sema_data['scan_labels'])
write_ply(out_name_fq, mesh_data['vertices'], mesh_data['faces'], f_quality=face_labels)


cam_pkl = os.path.join(in_path, 'Capture', f'cameras.pkl')
cam_data = load_pickle(cam_pkl)
camera = UnifiedCamera.from_4d_dress(cam_data['0004']['intrinsics'], cam_data['0004']['extrinsics'], 1280, 940)

drtk_camera = camera.to_drtk_format().float().cuda()
verts = torch.from_numpy(mesh_data['vertices']).float().cuda()
faces = torch.from_numpy(mesh_data['faces']).int().cuda()
face_labels = torch.from_numpy(face_labels).float().cuda()

out_img = render_drtk_face_attr(drtk_camera, verts[None], faces, face_labels[:, None]+1, bg_attr=torch.Tensor([8]).cuda())
out_img = out_img[0, ..., 0].detach().cpu().numpy()

import cv2
import numpy as np
cv2.imwrite('test_drtk_render.png', (out_img*(255/8)).astype(np.uint8))
gt_img = cv2.imread(os.path.join(in_path, 'Capture', '0004', 'images/capture-f00011.png'))
cv2.imwrite('test_drtk_render_gt.png', gt_img)

# write_ply('debug.ply', mesh_camera_space[0].cpu().numpy(), mesh_data['faces'], v_color=mesh_data['colors'][:, :3], v_quality=sema_data['scan_labels'])
