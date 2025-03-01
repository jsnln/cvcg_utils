import numpy as np
import cv2
from cvcg_utils.mesh.mesh_io import read_obj, write_obj, read_ply, write_ply

if __name__ == '__main__':
    mesh = read_obj('bunny_with_uv.obj')

    rand_texture = np.random.rand(16, 16, 3)
    rand_texture = (rand_texture / np.linalg.norm(rand_texture, axis=-1, keepdims=True).clip(min=1e-5)).clip(min=0., max=1.)
    rand_texture = (rand_texture * 255).astype(np.uint8)
    rand_texture = cv2.resize(rand_texture, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    
    write_obj('bunny_with_texture.obj', mesh.verts, mesh.faces, mesh.uv_verts, mesh.uv_faces, rand_texture)
    