import numpy as np
import os
import pickle
import igl
import torch
from cvcg_utils import write_ply
from cvcg_utils import UnifiedCamera, DRTKCamera, render_drtk_face_attr

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


camera = UnifiedCamera.from_local_frame(
    np.array([1.,0,0]),
    np.array([0.,1,0]),
    np.array([0,0,0]),
    60 / 180 * np.pi,
    60 / 180 * np.pi,
    1024, 512,
)

drtk_camera = camera.to_drtk_format_debug().float().cuda()
verts = torch.tensor([[1.0,-0.2,-3.0],
                      [0.5,1.0,-3.0],
                      [-1.0,0.0,-3.0],
                      [1.5,-0.2,-2.9],
                      [1.0,1.0,-2.9],
                      [-0.5,0.0,-2.9]]).float().cuda()
faces = torch.tensor([[0,1,2],[3,4,5]]).int().cuda()
face_labels = torch.tensor([1.0, 2.0]).float().cuda()

out_img = render_drtk_face_attr(drtk_camera, verts[None], faces, face_labels[:, None])
out_img = out_img[0, ..., 0].detach().cpu().numpy()

import cv2
import numpy as np
cv2.imwrite('test_drtk_render_debug.png', (out_img*255/2).astype(np.uint8))
# gt_img = cv2.imread(os.path.join(in_path, 'Capture', '0004', 'images/capture-f00011.png'))
# cv2.imwrite('test_drtk_render_gt.png', gt_img)
