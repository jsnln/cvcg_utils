import torch

from config import TrainerConfig

class ExampleModel(torch.nn.Module):
    def __init__(self, config: TrainerConfig):
        super().__init__()

        self.config = config

        self.in_dim = config.in_dim
        self.out_dim = config.out_dim
        self.h_dim = config.h_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, self.h_dim),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(self.h_dim, self.h_dim),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(self.h_dim, self.out_dim),
        )

        self.init_parameters()
        self.freeze_parameters()
        self.activate_trainale_parameters()

    def init_parameters(self):
        """
        init if needed
        """
        pass

    def freeze_parameters(self):
        # freeze parameters
        self.requires_grad_(False)
        self.eval()
        
    def activate_trainale_parameters(self):
        # activate trainable ones (e.g., for finetuning)
        self.requires_grad_(True)
        self.train()

    def forward(self, data_batch):
        data_x = data_batch['data_x']
        model_pred = self.mlp(data_x)

        output_dict = {
            'model_pred': model_pred,
        }

        if 'data_y' in data_batch:
            output_dict.update(data_y=data_batch['data_y'])

        return output_dict
