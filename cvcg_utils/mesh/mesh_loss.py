import torch


def _cotangent(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # cot(theta) = dot(a,b) / ||a x b||
    # where theta is the angle between a and b
    dot = (a * b).sum(dim=-1)
    cross = torch.cross(a, b, dim=-1)
    denom = cross.norm(dim=-1).clamp_min(eps)
    return dot / denom


def cotangent_weights(verts_rest: torch.Tensor, faces: torch.Tensor, eps: float = 1e-12):
    """
    Returns symmetric undirected edge list and weights:
      edge_ij: [2, E] with i<j
      w_ij:    [E]
    Using w_ij = 0.5 * (cot(alpha) + cot(beta))  (sum over incident faces)
    """
    v = verts_rest
    f = faces.long()

    i, j, k = f[:, 0], f[:, 1], f[:, 2]
    vi, vj, vk = v[i], v[j], v[k]

    # For each face (i,j,k), contribute:
    #  cot(angle at k) to edge (i,j)
    #  cot(angle at i) to edge (j,k)
    #  cot(angle at j) to edge (k,i)
    cot_k = _cotangent(vi - vk, vj - vk, eps=eps)  # angle at k
    cot_i = _cotangent(vj - vi, vk - vi, eps=eps)  # angle at i
    cot_j = _cotangent(vk - vj, vi - vj, eps=eps)  # angle at j

    # Build undirected edges with per-face cot contributions
    e01 = torch.stack([i, j], dim=1)
    e12 = torch.stack([j, k], dim=1)
    e20 = torch.stack([k, i], dim=1)
    edges = torch.cat([e01, e12, e20], dim=0)              # [3Nf,2]
    cots  = torch.cat([cot_k, cot_i, cot_j], dim=0)        # [3Nf]

    # Canonicalize (min,max) so we can sum duplicates
    a = torch.minimum(edges[:, 0], edges[:, 1])
    b = torch.maximum(edges[:, 0], edges[:, 1])
    edges_c = torch.stack([a, b], dim=1)                   # [3Nf,2]

    # Sum cot contributions per unique undirected edge
    # Note: torch.unique with return_inverse is differentiable w.r.t. weights/verts,
    # but edges indices are discrete anyway.
    uniq, inv = torch.unique(edges_c, dim=0, return_inverse=True)
    w = torch.zeros(uniq.shape[0], device=v.device, dtype=v.dtype)
    w.index_add_(0, inv, cots)

    # Standard cotan weight uses 0.5 factor
    w = 0.5 * w
    return uniq.t().contiguous(), w  # [2,E], [E]


def arap_loss(verts: torch.Tensor,
              verts_rest: torch.Tensor,
              faces: torch.Tensor,
              eps: float = 1e-9,
              return_rotations: bool = False):
    """
    ARAP loss for a single mesh.

    Args:
      verts:      [Nv,3] deformed
      verts_rest: [Nv,3] rest/reference
      faces:      [Nf,3] long
    Returns:
      loss: scalar tensor
      (optional) R: [Nv,3,3] per-vertex rotations
    """
    device, dtype = verts.device, verts.dtype
    Nv = verts.shape[0]

    edge_ij, w_ij = cotangent_weights(verts_rest, faces, eps=eps)  # undirected i<j
    ii, jj = edge_ij[0], edge_ij[1]  # [E]

    # Build directed edges so we can sum over neighbors per vertex i
    src = torch.cat([ii, jj], dim=0)          # [2E]
    dst = torch.cat([jj, ii], dim=0)          # [2E]
    w   = torch.cat([w_ij, w_ij], dim=0)      # [2E]

    # Edge vectors in rest and deformed
    e0 = verts_rest[src] - verts_rest[dst]    # [2E,3]
    e  = verts[src]      - verts[dst]         # [2E,3]

    # Per-vertex covariance A_i = sum_j w_ij * (e_ij) (e0_ij)^T
    outer = (e[:, :, None] * e0[:, None, :]) * w[:, None, None]  # [2E,3,3]
    A = torch.zeros((Nv, 3, 3), device=device, dtype=dtype)
    A_flat = A.view(Nv, 9)
    outer_flat = outer.view(-1, 9)
    A_flat.index_add_(0, src, outer_flat)
    A = A_flat.view(Nv, 3, 3)

    # SVD for each vertex: A = U S Vh, R = U Vh (with det correction)
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)  # U:[Nv,3,3], Vh:[Nv,3,3]
        R = U @ Vh

        # Fix improper rotations (reflection) by flipping last column of U when det<0
        detR = torch.linalg.det(R)
        mask = detR < 0
        if mask.any():
            U_fix = U.clone()
            U_fix[mask, :, 2] *= -1.0
            R = U_fix @ Vh

    # Energy: 0.5 * sum_{directed} w_ij || R_i e0_ij - e_ij ||^2
    Re0 = (R[src] @ e0[:, :, None]).squeeze(-1)           # [2E,3]
    diff = Re0 - e
    loss = 0.5 * (w * (diff * diff).sum(dim=-1)).sum()

    return (loss, R) if return_rotations else loss
