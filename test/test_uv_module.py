import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'

from geometry_utils import read_ply, convert_edge_uv_to_uv_mesh, get_uv_assets, vert_attr_to_uv
import numpy as np
import torch
import cv2
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    verts, vert_uv, faces, edge_uv, plydata = read_ply('/home/trisst/workplace/cloth_tracking/dataproc/00122_mesh_00011_template_process/untitled3.ply')
    verts = torch.from_numpy(verts).float().cuda()
    faces = torch.from_numpy(faces).int().cuda()
    edge_uv = torch.from_numpy(edge_uv).float().cuda()

    uv_verts, uv_faces = convert_edge_uv_to_uv_mesh(edge_uv)
    vert_index_map, face_index_map, bary_coords_map, uv_mask = get_uv_assets(faces, uv_verts, uv_faces, uv_size=128)

    uv_vert_coords = vert_attr_to_uv(verts[None], vert_index_map, bary_coords_map, uv_mask)
    # face_index_map = face_index_map_from_edge_uv(edge_uv, 128)    # [H, W]
    # vert_index_map = vert_index_map_from_edge_uv(faces, edge_uv, 128)

    # plt.imshow(index_img.cpu().float().numpy())
    # plt.show()

    # plt.imshow(vert_index_map.cpu().float().numpy())
    # plt.show()

    # plt.imshow(bary_coords_map.cpu().float().numpy())
    # plt.show()

    uv_pcd = uv_vert_coords[0, uv_mask].cpu().float().numpy()
    np.savetxt('debug.xyz', uv_pcd)

    # plt.imshow(uv_vert_coords[0, ..., 0].cpu().float().numpy())
    # plt.show()

    # cv2.imwrite('test.exr', index_img.cpu().float().numpy())
