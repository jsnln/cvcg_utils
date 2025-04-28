import os
import igl
import plyfile
import numpy as np
from dataclasses import dataclass
from ..misc.image import write_rgb

@dataclass
class MeshData:
    verts: np.ndarray = None
    faces: np.ndarray = None
    uv_verts: np.ndarray = None     # compatible with vt coords in obj
    uv_faces: np.ndarray = None     # compatible with obj
    v_normals: np.ndarray = None
    f_normals: np.ndarray = None
    v_colors: np.ndarray = None
    f_colors: np.ndarray = None
    edge_uv: np.ndarray = None      # compatible with the `texcoords` field in ply

####################
##### obj file #####
####################

def read_obj(fn: str):
    assert fn.endswith('.obj')

    verts, uv_verts, v_normals, faces, uv_faces, f_normals = \
        igl.readOBJ(fn)
    mesh = MeshData(verts, faces, uv_verts, uv_faces, v_normals, f_normals)
    return mesh

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
""")    # no idea why these params, probably some default
            mtl_file.write(f'map_Kd {tex_fn}')

        write_rgb(os.path.join(os.path.dirname(fn), tex_fn), texture_img)
    

####################
##### ply file #####
####################

def read_ply(fn):
    with open(fn, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
    
    verts = np.stack([plydata['vertex']['x'],
                      plydata['vertex']['y'],
                      plydata['vertex']['z']], axis=-1)
    try:
        vert_uv = np.stack([plydata['vertex']['texture_u'],
                            plydata['vertex']['texture_v']], axis=-1)
    except:
        vert_uv = None
    
    try:
        v_colors = np.stack([plydata['vertex']['red'],
                             plydata['vertex']['green'],
                             plydata['vertex']['blue']], axis=-1)
    except:
        v_colors = None

    try:
        faces = np.stack(plydata['face']['vertex_indices'], axis=0)
    except:
        faces = None

    try:
        edge_uv = np.stack(plydata['face']['texcoord'], axis=0).reshape(faces.shape[0], 3, 2)
    except:
        edge_uv = None

    # TODO add support for normals, colors, edge uv
    mesh = MeshData(verts, faces, None, None, None, None, v_colors, None, edge_uv)
    return mesh

def write_ply(
        fn: str,
        v: np.ndarray,
        f: np.ndarray = None,
        v_color: np.ndarray = None,
        f_color: np.ndarray = None,
        v_normal: np.ndarray = None,
        v_quality: np.ndarray = None,
        f_quality: np.ndarray = None,
        edge_uv: np.ndarray = None,
        texture_img: np.ndarray = None):

    assert fn.endswith('.ply')
    assert len(v.shape) == 2
    assert v.shape[1] == 3
    if f is not None:
        assert len(f.shape) == 2
        assert f.shape[1] == 3
    if v_color is not None:
        assert len(v_color.shape) == 2
        assert v_color.shape[1] in [3, 4]
        assert v_color.shape[0] == v.shape[0]
    if v_normal is not None:
        assert len(v_normal.shape) == 2
        assert v_normal.shape[1] == 3
        assert v_normal.shape[0] == v.shape[0]
    if v_quality is not None:
        assert len(v_quality.shape) == 1
        assert v_quality.shape[0] == v.shape[0]
    if f_color is not None:
        assert len(f_color.shape) == 2
        assert f_color.shape[1] == 3
        assert f_color.shape[0] == f.shape[0]
    if f_quality is not None:
        assert len(f_quality.shape) == 1
        assert f_quality.shape[0] == f.shape[0]
    if edge_uv is not None:
        assert len(edge_uv.shape) == 3
        assert edge_uv.shape[0] == f.shape[0]
        assert edge_uv.shape[1] == 3
        assert edge_uv.shape[2] == 2

    # v dtype
    v_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if v_color is not None:
        if v_color.shape[1] == 3:
            v_dtype = v_dtype + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        elif v_color.shape[1] == 4:
            v_dtype = v_dtype + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')]
        else:
            raise NotImplementedError

    if v_normal is not None:
        v_dtype = v_dtype + [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]

    if v_quality is not None:
        v_dtype = v_dtype + [('quality', 'f4')]

    
    # f dtype
    if f is not None:
        f_dtype = [('vertex_indices', 'i4', (3,))]
        if f_color is not None:
            f_dtype = f_dtype + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        if f_quality is not None:
            f_dtype = f_dtype + [('quality', 'f4')]
        if edge_uv is not None:
            f_dtype = f_dtype + [('texcoord', 'f4', (6,))]

    
    # v data
    v_list_of_tuples = []
    for vid in range(v.shape[0]):
        v_tuple = tuple(v[vid].tolist())
        if v_color is not None:
            v_tuple = v_tuple + tuple(v_color[vid].tolist())
        if v_normal is not None:
            v_tuple = v_tuple + tuple(v_normal[vid].tolist())
        if v_quality is not None:
            v_tuple = v_tuple + (v_quality[vid],)
        v_list_of_tuples.append(v_tuple)
    v_data = np.array(v_list_of_tuples, dtype=v_dtype)

    if texture_img is not None:
        tex_fn = os.path.basename(fn) + ".png"
        comments = [f'TextureFile {tex_fn}']
        write_rgb(os.path.join(os.path.dirname(fn), tex_fn), texture_img)
    else:
        comments = []
    v_el = plyfile.PlyElement.describe(v_data, 'vertex', comments=comments)
    
    # f data
    if f is not None:
        f_list_of_tuples = []
        for fid in range(f.shape[0]):
            f_tuple = (f[fid].tolist(),)
            if f_color is not None:
                f_tuple = f_tuple + tuple(f_color[fid].tolist())
            if f_quality is not None:
                f_tuple = f_tuple + (f_quality[fid],)
            if edge_uv is not None:
                f_tuple = f_tuple + (edge_uv[fid].reshape(-1).tolist(),)
            f_list_of_tuples.append(f_tuple)
        f_data = np.array(f_list_of_tuples, dtype=f_dtype)
        f_el = plyfile.PlyElement.describe(f_data, 'face')

        plyfile.PlyData([v_el, f_el]).write(fn)
    else:
        plyfile.PlyData([v_el]).write(fn)

