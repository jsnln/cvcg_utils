import os
import igl
import numpy as np
from ..image_io.image_io import write_rgb

def read_obj(fn: str):
    assert fn.endswith('.obj')

    verts, uv_verts, vert_normals, faces, uv_faces, face_normals = \
        igl.readOBJ(fn)
    return verts, uv_verts, vert_normals, faces, uv_faces, face_normals

def write_obj(fn: str,
              verts: np.ndarray,
              faces: np.ndarray,
              uv_verts: np.ndarray = None,
              uv_faces: np.ndarray = None,
              texture_img: np.ndarray = None):
    """
    texture_img should be uint8, [H, W, C], rgb
    """
    
    assert fn.endswith('.obj')
    assert len(verts.shape) == 2
    assert verts.shape[1] == 3
    assert len(faces.shape) == 2
    assert faces.shape[1] == 3

    with open(fn, 'w') as file:
        if texture_img is not None:
            mtl_fn = os.path.basename(fn + ".mtl")
            tex_fn = os.path.basename(fn + ".png")
            file.write(f'mtllib {mtl_fn}\n')
        
        for v in verts:
            file.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')

        if uv_verts is not None:
            for uv_v in uv_verts:
                file.write(f'vt {uv_v[0]:.6f} {uv_v[1]:.6f}\n')
        
        if uv_faces is not None:
            for f, uv_f in zip(faces + 1, uv_faces + 1):
                file.write(f'f {f[0]:d}/{uv_f[0]:d} {f[1]:d}/{uv_f[1]:d} {f[2]:d}/{uv_f[2]:d}\n')
        else:
            for f in faces + 1:
                file.write(f'f {f[0]:d} {f[1]:d} {f[2]:d}\n')

    if texture_img is not None:
        with open(os.path.join(os.path.dirname(fn), mtl_fn), 'w') as mtl_file:
            mtl_file.write("""
newmtl material_0
Ka 0.200000 0.200000 0.200000
Kd 0.752941 0.752941 0.752941
Ks 1.000000 1.000000 1.000000
Tr 0.000000
illum 2
Ns 0.000000
""")    # no idea why these params
            mtl_file.write(f'map_Kd {tex_fn}')

        write_rgb(os.path.join(os.path.dirname(fn), tex_fn), texture_img)
    


