import torch

class TrainerConfig:
    model_name = 'example_model'
    dataset_name = 'example_dataset'

    in_dim = 32
    h_dim = 64
    out_dim = 1

    num_epochs =  100
    max_steps = 5000
    total_train_steps = 5000
    grad_accum_steps = 1
    batch_size = 8
    grad_clip_norm = 1.0
    lr_warmup_steps = 500
    lr = 1e-4

    reset_state = False
    resume_from_state: str = None
    resume_from_model: str = None

    losses_and_weights = {
        'mse': 1.0
    }
