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

