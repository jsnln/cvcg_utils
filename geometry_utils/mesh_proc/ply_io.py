import numpy as np
import plyfile

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

    # TODO add support for normals and colors

    return verts, vert_uv, v_colors, faces, edge_uv, plydata

def write_ply(
        fn: str,
        v: np.ndarray,
        f: np.ndarray = None,
        v_color: np.ndarray = None,
        f_color: np.ndarray = None,
        v_quality: np.ndarray = None,
        f_quality: np.ndarray = None):

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

    # v dtype
    v_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if v_color is not None:
        if v_color.shape[1] == 3:
            v_dtype = v_dtype + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        elif v_color.shape[1] == 4:
            v_dtype = v_dtype + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')]
        else:
            raise NotImplementedError
    if v_quality is not None:
        v_dtype = v_dtype + [('quality', 'f4')]

    
    # f dtype
    if f is not None:
        f_dtype = [('vertex_indices', 'i4', (3,))]
        if f_color is not None:
            f_dtype = f_dtype + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        if f_quality is not None:
            f_dtype = f_dtype + [('quality', 'f4')]
    
    # v data
    v_list_of_tuples = []
    for vid in range(v.shape[0]):
        v_tuple = tuple(v[vid].tolist())
        if v_color is not None:
            v_tuple = v_tuple + tuple(v_color[vid].tolist())
        if v_quality is not None:
            v_tuple = v_tuple + (v_quality[vid],)
        v_list_of_tuples.append(v_tuple)
    v_data = np.array(v_list_of_tuples, dtype=v_dtype)
    v_el = plyfile.PlyElement.describe(v_data, 'vertex')
    
    # f data
    if f is not None:
        f_list_of_tuples = []
        for fid in range(f.shape[0]):
            f_tuple = (f[fid].tolist(),)
            if f_color is not None:
                f_tuple = f_tuple + tuple(f_color[fid].tolist())
            if f_quality is not None:
                f_tuple = f_tuple + (f_quality[fid],)
            f_list_of_tuples.append(f_tuple)
        f_data = np.array(f_list_of_tuples, dtype=f_dtype)
        f_el = plyfile.PlyElement.describe(f_data, 'face')

        plyfile.PlyData([v_el, f_el]).write(fn)
    else:
        plyfile.PlyData([v_el]).write(fn)

