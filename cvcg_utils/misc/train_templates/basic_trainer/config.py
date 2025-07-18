from typing import List
import torch

class TrainerConfig:
    model_name = 'example_model'
    dataset_name = 'example_dataset'

    in_dim = 32
    h_dim = 64
    out_dim = 1

    skip_grad_threshold = 8.0
    num_epochs =  100
    max_steps = 5000
    total_train_steps = 5000
    grad_accum_steps = 1
    batch_size = 8
    grad_clip_norm = 1.0
    lr_warmup_steps = 500
    lr = 1e-4

    reset_state = True
    resume_from_latest: str = True  # highest priority
    resume_from_state: str = None
    resume_from_model: str = None

    losses_and_weights = {
        'mse': 1.0
    }
    
    save_latest_every: int = 100
    save_ckpt_every: int = 1000
    extra_save_ckpt_at: List[int] = [1, 100, 500]
    out_dir: str = 'results/example_experiment'
