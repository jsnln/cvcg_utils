from typing import Union
import torch
import torch.nn.functional as F
import igl
import numpy as np
import scipy as sp

def get_face_normals(verts: torch.Tensor, faces: torch.Tensor, normalize: bool):
    """
    verts: [..., Nv, 3]
    faces: [Nf, 3]

    out: face_normals (batched): [..., Nf, 3]
    """
    assert len(faces.shape) == 2
    
    verts_by_faces = verts[..., faces, :]   # [..., Nf, 3 (verts), 3 (coords)]
    e01 = verts_by_faces[..., 1, :] - verts_by_faces[..., 0, :] # [..., Nf, 3 (coords)]
    e02 = verts_by_faces[..., 2, :] - verts_by_faces[..., 0, :] # [..., Nf, 3 (coords)]
    face_normals = torch.cross(e01, e02, dim=-1)                # [..., Nf, 3 (coords)]
    if normalize:
        face_normals = F.normalize(face_normals, dim=-1)
    return face_normals

def get_vert_normals(verts: torch.Tensor, faces: torch.Tensor, normalize: bool):
    """
    verts: [..., Nv, 3]
    faces: [Nf, 3]

    out: vert_normals (batched): [..., Nv, 3]
    """
    assert len(verts.shape) >= 2
    assert len(faces.shape) == 2
    faces_newaxis = [1] * (len(verts.shape)-2) + list(faces.shape) + [1]   # [... (1's), Nf, 3, 1]
    faces_newshape = list(verts.shape[:-2]) + list(faces.shape) + [3]   # [..., Nf, 3, 3]
    faces_expanded = faces.reshape(faces_newaxis).expand(faces_newshape).long()

    face_normals = get_face_normals(verts, faces, normalize=False)
    vert_normals = torch.zeros_like(verts)
    vert_normals.scatter_add_(dim=-2, index=faces_expanded[..., 0, :], src=face_normals)
    vert_normals.scatter_add_(dim=-2, index=faces_expanded[..., 1, :], src=face_normals)
    vert_normals.scatter_add_(dim=-2, index=faces_expanded[..., 2, :], src=face_normals)

    if normalize:
        vert_normals = F.normalize(vert_normals, dim=-1)

    return vert_normals

def get_unique_edges(faces):
    edges = torch.stack([
        torch.stack([faces[:, 0], faces[:, 1]], dim=-1),
        torch.stack([faces[:, 1], faces[:, 2]], dim=-1),
        torch.stack([faces[:, 2], faces[:, 0]], dim=-1),
    ], dim=-2).reshape(-1, 2)   # [Nf * 3, 2] <= [Nf, 3, 2]
    edges = torch.sort(edges, dim=1).values

    # uedges: unique edges
    uedges = torch.unique(edges, dim=0)
    return uedges

def get_boundary_edges(faces):
    # all edges
    edges = torch.stack([
        torch.stack([faces[:, 0], faces[:, 1]], dim=-1),
        torch.stack([faces[:, 1], faces[:, 2]], dim=-1),
        torch.stack([faces[:, 2], faces[:, 0]], dim=-1),
    ], dim=-2).reshape(-1, 2)   # [Nf * 3, 2] <= [Nf, 3, 2]
    edges = torch.sort(edges, dim=1).values

    edge_to_face = torch.arange(faces.shape[0], device=faces.device)    # [Nf]
    edge_to_face = edge_to_face[:, None].expand(-1, 3).reshape(-1)         # [Nf*3], in correspondence with edges

    # get opposite edge index
    edge_indices = torch.arange(edges.shape[0], dtype=faces.dtype, device=faces.device)

    # uedges: unique edges
    uedges, inverse_indices, counts = torch.unique(edges, return_inverse=True, return_counts=True, dim=0)
    assert counts.max() <= 2    # otherwise nonmanifold

    index_uedge_to_edge = -1 * torch.ones(uedges.shape[0], dtype=faces.dtype, device=faces.device)
    index_uedge_to_edge.scatter_(dim=0, index=inverse_indices, src=edge_indices)

    possibly_opposite_edges = index_uedge_to_edge[inverse_indices]
    has_opposite_edge_mask = (possibly_opposite_edges != edge_indices)

    selected_edges = edge_indices[has_opposite_edge_mask]
    opposite_edges = possibly_opposite_edges[selected_edges]

    all_edges_with_opposites = torch.cat([selected_edges, opposite_edges], dim=0)
    boundary_edge_mask = torch.ones_like(edge_indices, dtype=bool)
    boundary_edge_mask[all_edges_with_opposites] = False    # a mask to be applied to edges

    boundary_edges = edges[boundary_edge_mask]  # [N_be, 2]

    return boundary_edges
    
def get_boundary_verts(n_verts: int, faces: torch.Tensor):
    boundary_vert_mask = torch.zeros(n_verts, dtype=bool, device=faces.device)
    boundary_edges = get_boundary_edges(faces)
    boundary_vert_mask[boundary_edges] = True
    return boundary_vert_mask

def get_laplacian(n_verts: int, faces: torch.Tensor, dtype=torch.float) -> torch.Tensor:
    """
    **Modified from pytorch3d.ops.laplacian**
    
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        n_verts
        faces: tensor of shape (Nf, 2)
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    uedges = get_unique_edges(faces)

    e0, e1 = uedges.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    ones = torch.ones(idx.shape[1], dtype=dtype, device=faces.device)
    A = torch.sparse.FloatTensor(idx, ones, (n_verts, n_verts))

    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (n_verts, n_verts))

    idx = torch.arange(n_verts, device=faces.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=faces.device)
    L -= torch.sparse.FloatTensor(idx, ones, (n_verts, n_verts))

    return L

def get_mesh_eigenfunctions(verts, faces, k):
    L = -igl.cotmatrix(verts, faces)
    M = igl.massmatrix(verts, faces, igl.MASSMATRIX_TYPE_VORONOI)

    try:
        eigenvalues, eigenfunctions = sp.sparse.linalg.eigsh(L, k, M, sigma=0, which="LM")
    except RuntimeError as e:
        # singular coefficient matrix
        c = 1e-10
        L = L + c * sp.sparse.eye(L.shape[0])
        eigenvalues, eigenfunctions = sp.sparse.linalg.eigsh(L, k, M, sigma=0, which="LM")

    assert np.all(np.max(eigenfunctions, axis=0) != np.min(eigenfunctions, axis=0))

    return eigenfunctions, eigenvalues

def remove_unreferenced(vert_attrs: Union[np.ndarray, torch.Tensor],
                        faces: Union[np.ndarray, torch.Tensor],
                        vert_list_mask: Union[np.ndarray, torch.Tensor]=None,
                        face_list_mask: Union[np.ndarray, torch.Tensor]=None):
    """
    vert_attrs: [Nv, C]
    faces: [Nf, 3]

    vert_list_mask: [Nv], bool
    face_list_mask: [Nf], bool

    out:
    - vert_attrs_cleaned
    - faces_cleaned

    explanation:
    - a face will be removed if:
      (1) it's False in face_list_mask
      (2) any of its vertices is False in vert_list_mask
    - a vert will be removed if:
      (1) it's not referenced by any face after face masking
    """
    if isinstance(vert_attrs, np.ndarray):
        return remove_unreferenced_np(vert_attrs, faces, vert_list_mask, face_list_mask)
    elif isinstance(vert_attrs, torch.Tensor):
        return remove_unreferenced_th(vert_attrs, faces, vert_list_mask, face_list_mask)
    else:
        raise NotImplementedError(f'function not implemented for {type(vert_attrs)}')

def remove_unreferenced_th(vert_attrs: torch.Tensor, faces: torch.Tensor, vert_list_mask: torch.Tensor=None, face_list_mask: torch.Tensor=None):
    if vert_list_mask is None and face_list_mask is None:
        return vert_attrs.clone(), faces.clone()    # because the function below will return copies, we also return copies here just to be consistent

    face_mask_final = torch.ones_like(faces[:, 0], dtype=torch.bool)
    if face_list_mask is not None:
        face_mask_final = face_mask_final & face_list_mask
    if vert_list_mask is not None:
        face_mask_by_verts = vert_list_mask[faces]  # [Nf, 3], bool
        face_mask_final = face_mask_final & face_mask_by_verts.all(dim=-1)
    
    referenced_faces_ori_vidx = faces[face_mask_final]  # remaining faces, but holding old vertex indices
    referenced_vert_mask = torch.zeros_like(vert_attrs[:, 0], dtype=torch.bool)  # [Nv], bool, a mask denoting referenced verts after masking
    referenced_vert_mask[referenced_faces_ori_vidx] = True

    # final vert attrs to be returned
    vert_attrs_final = vert_attrs[referenced_vert_mask]  # [Nv_new, 3]
    num_verts_final = vert_attrs_final.shape[0] # Nv_new

    vert_index_mapping = -1 * torch.ones_like(vert_attrs[:, 0], dtype=torch.int64)  # [Nv], contains new vert idx. If unreferenced, it is -1
    vert_index_mapping[referenced_vert_mask] = torch.arange(num_verts_final, device=vert_index_mapping.device, dtype=vert_index_mapping.dtype)

    faces_final = vert_index_mapping[referenced_faces_ori_vidx]


    assert (faces_final >= 0).all()

    return vert_attrs_final, faces_final.to(faces.dtype)



def remove_unreferenced_np(vert_attrs: np.ndarray, faces: np.ndarray, vert_list_mask: np.ndarray=None, face_list_mask: np.ndarray=None):
    if vert_list_mask is None and face_list_mask is None:
        return vert_attrs.copy(), faces.copy()    # because the function below will return copies, we also return copies here just to be consistent

    face_mask_final = np.ones_like(faces[:, 0], dtype=bool)
    if face_list_mask is not None:
        face_mask_final = face_mask_final & face_list_mask
    if vert_list_mask is not None:
        face_mask_by_verts = vert_list_mask[faces]  # [Nf, 3], bool
        face_mask_final = face_mask_final & face_mask_by_verts.all(axis=-1)
    
    referenced_faces_ori_vidx = faces[face_mask_final]  # remaining faces, but holding old vertex indices
    referenced_vert_mask = np.zeros_like(vert_attrs[:, 0], dtype=bool)  # [Nv], bool, a mask denoting referenced verts after masking
    referenced_vert_mask[referenced_faces_ori_vidx] = True

    # final vert attrs to be returned
    vert_attrs_final = vert_attrs[referenced_vert_mask]  # [Nv_new, 3]
    num_verts_final = vert_attrs_final.shape[0] # Nv_new

    vert_index_mapping = -1 * np.ones_like(vert_attrs[:, 0], dtype=int)  # [Nv], contains new vert idx. If unreferenced, it is -1
    vert_index_mapping[referenced_vert_mask] = np.arange(num_verts_final, dtype=vert_index_mapping.dtype)

    faces_final = vert_index_mapping[referenced_faces_ori_vidx]

    assert (faces_final >= 0).all()

    return vert_attrs_final, faces_final.astype(faces.dtype)




if __name__ == '__main__':
    verts = torch.tensor([0., 1., 2., 3., 4.]).reshape(5, 1).numpy()
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4]]).numpy()
    vmask = None
    # vmask = torch.tensor([1, 0, 1, 1, 1]).bool()
    # fmask = None
    fmask = torch.tensor([1, 0, 1]).bool().numpy()


    verts_new, faces_new = remove_unreferenced(verts, faces, vmask, fmask)

    print(verts_new, faces_new)

