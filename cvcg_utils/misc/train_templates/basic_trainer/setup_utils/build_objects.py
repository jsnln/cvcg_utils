from dataclasses import dataclass
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

# base object classes, only for type hints
from loggers.base_logger import BaseLogger
from loss_module.base_loss_module import BaseLossModule
from preprocessor.base_preprocessor import BasePreprocessor


# actual classes to be used
from model.example_model import ExampleModel
model_classes = {
    'example_model': ExampleModel,
}

from dataset.example_dataset import ExampleDataset
dataset_classes = {
    'example_dataset': ExampleDataset,
}

from loss_module.example_loss_module import ExampleLossModule
loss_module_classes = {
    'example_loss_module': ExampleLossModule,
}

from preprocessor.example_preprocessor import ExamplePreprocessor
preprocessor_classes = {
    'example_preprocessor': ExamplePreprocessor,
}

from loggers.example_logger import ExampleLogger
logger_classes = {
    'example_logger': ExampleLogger,
}

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

amp_dtype_mapping = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
    'tf32': torch.float32,
}

@dataclass
class TrainObjectsCollection:
    model: torch.nn.Module = None
    dataset: torch.utils.data.Dataset = None
    dataloader: torch.utils.data.DataLoader = None
    optimized_params: List[torch.Tensor] = None
    optimizer: torch.optim.Optimizer = None
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
    loss_module: BaseLossModule = None
    preprocessor: BasePreprocessor = None
    grad_scaler: torch.amp.GradScaler = None
    logger: BaseLogger = None


def build_objects(config: TrainerConfig) -> TrainObjectsCollection:
    model_cls = model_classes[config.model_name]
    model = model_cls(config)
    print(f'[model info] using model {model_cls}')

    dataset_cls = dataset_classes[config.dataset_name]
    dataset = dataset_cls(config)

    if config.dist_gpus > 1:
        distsampler = torch.utils.data.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, sampler=distsampler)
    else:
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
    loss_module_cls = loss_module_classes[config.loss_module_name]
    loss_module = loss_module_cls(config)
    print(f'\\[loss info] loss module: {loss_module_cls}, losses and weights: {config.losses_and_weights}')

    preprocessor_cls = preprocessor_classes[config.preprocessor_name]
    preprocessor = preprocessor_cls(config)
    print(f'\\[preprocessor info] preprocessor: {preprocessor_cls}')

    # loss grad scaler (for fp16)
    if config.amp_dtype == 'bf16':
        assert torch.cuda.get_device_capability()[0] >= 8, f'bf16 can only be used on gpus with compute capability >= 8.0'

    if config.use_amp:
        used_dtype = amp_dtype_mapping[config.amp_dtype]

    use_grad_scaler = (config.use_amp and (used_dtype == torch.float16))
    grad_scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)
    print(f'\\[grad scaler info] use grad scaler: {use_grad_scaler}')

    # logger
    logger_cls = logger_classes[config.logger_name]
    logger = logger_cls(config)

    return TrainObjectsCollection(model, dataset, dataloader, optimized_params, optimizer, lr_scheduler, loss_module, preprocessor, grad_scaler, logger)
