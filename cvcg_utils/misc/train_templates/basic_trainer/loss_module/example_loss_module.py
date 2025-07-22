import torch

from config import TrainerConfig

class ExampleLossModule(torch.nn.Module):
    # this inherits torch.nn.Module just in case LPIPS is used (easier for moving to device)
    # but should not contain any trainable parameters

    def __init__(self, config: TrainerConfig):
        super().__init__()

        self.losses_and_weights = config.losses_and_weights

        # if 'lpips' in self.losses_and_weights:
        #     self.lpips_module = LPIPS()
        self.requires_grad_(False)
        self.eval()

        # loss repository
        def mse_loss(model_output):
            return torch.nn.functional.mse_loss(model_output['model_pred'], model_output['data_y'])

        self.loss_func_repo = {
            'mse': mse_loss,
        }

        # used losses
        self.loss_func_dict = {k: v for k, v in self.loss_func_repo.items() if (k in self.losses_and_weights and self.losses_and_weights[k] > 0)}


    def forward(self, model_output):
        total_loss = 0.
        loss_dict = {}
        loss_str = ''

        for loss_name, loss_func in self.loss_func_dict.items():
            loss_weight = self.losses_and_weights[loss_name]
            loss_val = loss_func(model_output)
            
            loss_dict[loss_name] = loss_val.item()
            loss_str = loss_str + f'{loss_name}: {loss_dict[loss_name]:.4e}, '

            total_loss = total_loss + loss_val * loss_weight

        loss_str = loss_str[:-2]

        return total_loss, loss_dict, loss_str

        