import numpy as np
import torch

from config import TrainerConfig

class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, config: TrainerConfig):
        super().__init__()

        self.config = config
        self.num_items = 96

        self.data_x = [np.random.randn(32) for i in range(self.num_items)]
        self.data_y = [np.random.randn(1) for i in range(self.num_items)]

    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, index):
        data_x = torch.from_numpy(self.data_x[index]).float()
        data_y = torch.from_numpy(self.data_y[index]).float()

        ret_dict = {
            'data_x': data_x,
            'data_y': data_y,
        }

        return ret_dict