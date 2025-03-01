import numpy as np
from cvcg_utils import write_ply, read_ply

if __name__ == '__main__':
    write_test = True
    read_test  = True
    
    if write_test:
        vertex = np.array([[0,0,0],
                        [0,1,1],
                        [1,0,1],
                        [1,1,0]])
        vertex_colors = np.array([[0,0,0],
                                [0,255,255],
                                [255,0,255],
                                [255,255,0]])
        vertex_quality = np.array([2, 0, 1, 3])
        face = np.array([[0,1,2],
                        [0,2,3],
                        [0,1,3],
                        [1,2,3]])
        face_colors = np.array([[0,255,255],
                        [255,0,255],
                        [255,255,0],
                        [255,255,255]])

        write_ply('test.ply', vertex, face, vertex_colors, face_colors, vertex_quality)

    if read_test:
        verts, vert_uv, faces, edge_uv, plydata = read_ply('/home/trisst/workplace/cloth_tracking/dataproc/00122_mesh_00011_template_process/untitled3.ply')