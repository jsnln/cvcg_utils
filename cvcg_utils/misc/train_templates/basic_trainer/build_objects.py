
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
from rich import print

import torch
from torchvision import transforms
from diffusers.optimization import get_cosine_schedule_with_warmup
from typing import List

from config import TrainerConfig

from model.example_model import ExampleModel
model_classes = {
    'example_model': ExampleModel,
}

from dataset.example_dataset import ExampleDataset
dataset_classes = {
    'example_dataset': ExampleDataset,
}

from loss_module.example_loss import ExampleLoss

def collate_fn(batch: List[dict]):
    all_keys = list(batch[0].keys())

    is_tensor = {}
    for key in all_keys:
        is_tensor[key] = isinstance(batch[0][key], torch.Tensor)

    ret_dict = {}
    for key in all_keys:
        item_list = [item[key] for item in batch]
        ret_dict[key] = torch.stack(item_list, dim=0) if is_tensor[key] else item_list
            
    return ret_dict


def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


def build_objects(config: TrainerConfig):
    model_cls = model_classes[config.model_name]
    model = model_cls(config)
    print(f'[model info] using model {model_cls}')

    dataset_cls = dataset_classes[config.dataset_name]
    dataset = dataset_cls(config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    print(f'\\[dataset info] using dataset {dataset_cls}')
    print(f'\\[dataloader info] dataloader batch size {config.batch_size}')

    # optimizer
    optimized_params = [p for p in model.parameters() if p.requires_grad]
    num_total_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in optimized_params)
    optimizer = torch.optim.AdamW(optimized_params, lr=config.lr)
    print(f'\\[optimizer info] trainable: {format_number(num_trainable_params)}, total: {format_number(num_total_params)}')

    # lr scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.max_steps,
    )
    print(f'\\[scheduler info] {config.lr_warmup_steps} warmup steps, {config.max_steps} total steps')


    # loss module
    loss_module = ExampleLoss(config)
    print(f'\\[loss info] losses and weights: {config.losses_and_weights}')

    
    return model, dataset, dataloader, optimized_params, optimizer, lr_scheduler, loss_module
