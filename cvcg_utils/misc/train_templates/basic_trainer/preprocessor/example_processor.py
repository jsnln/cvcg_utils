import torch

class ExamplePreprocessor:
    """
    should be a callable object
    should be used for preprocessing that are best used after loading, e.g.,
    cpu-to-cuda, random split, or transforms that run on gpus  
    """
    def __init__(self):
        pass

    def __call__(self, data_batch: dict) -> dict:
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor):
                data_batch[k] = v.cuda()

        return data_batch
