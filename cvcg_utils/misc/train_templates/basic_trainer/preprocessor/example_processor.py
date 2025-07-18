import torch

class ExamplePreprocessor(torch.nn.Module):
    """
    should be a callable object
    should be used for preprocessing that are best used after loading, e.g.,
    random split, or transforms that run on gpus  
    """
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, data_batch: dict) -> dict:
        return data_batch
        # for k, v in data_batch.items():
        #     if isinstance(v, torch.Tensor):
        #         data_batch[k] = v.cuda()
        # return data_batch
