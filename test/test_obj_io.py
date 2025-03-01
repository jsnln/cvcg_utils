import numpy as np
from cvcg_utils import read_obj

if __name__ == '__main__':
    read_test  = True
    
    if read_test:
        verts, uv_verts, vert_normals, faces, uv_faces, face_normals = \
            read_obj('/home/trisst/humanreconlab/smpl_assets/uv/smplx_uv/smplx_uv.obj')