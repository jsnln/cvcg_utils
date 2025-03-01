import torch
import numpy as np

def np2cpu(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array)

def np2cuda(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).cuda()

def th2np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()