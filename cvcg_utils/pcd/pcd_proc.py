import torch
from typing import Union, List, Tuple, Optional

def sample_farthest_points_given_idx(
    points: torch.Tensor,
    given_idx: torch.Tensor,
    K: Union[int, List, torch.Tensor] = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modifed from pytorch3d.ops.sample_farthest_points to support FPS from a given sampled set
    """
    assert len(points.shape) == 2
    assert len(given_idx.shape) == 1

    Ng, = given_idx.shape
    Np, D = points.shape
    device = points.device
    
    assert K < Np

    # initialize sample_idx and dists with given_idx
    sample_idx = torch.full(
        (K,),
        fill_value=-1,
        dtype=torch.int64,
        device=device,
    )
    closest_dists = points.new_full(
        (Np,),
        float("inf"),
        dtype=torch.float32,
    )
    sample_idx[:Ng] = given_idx

    for i in range(1, Ng):
        selected_idx = given_idx[i-1]
        dist = points[selected_idx, :] - points[:, :]
        dist_to_last_selected = (dist**2).sum(-1)  # (P - i)
        closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

    # continue from the given sample set
    for i in range(Ng, K):
        dist = points[selected_idx, :] - points[:, :]
        dist_to_last_selected = (dist**2).sum(-1)  # (P - i)

        closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

        selected_idx = torch.argmax(closest_dists)
        sample_idx[i] = selected_idx


    # Gather the points
    sampled_points = points[sample_idx]

    return sampled_points, sample_idx
