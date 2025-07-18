# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
from dataclasses import dataclass
from omegaconf import OmegaConf
import argparse
from easydict import EasyDict as edict
import re
import os
import datetime
import torch
import torch.distributed
import numpy as np
import random
import yaml
import wandb
import shutil
import copy
from pathlib import Path
import time

#################Init Config  Begins#################

# def process_overrides(overrides):
#     """
#     Handle space around "="
#     """
#     # First, join all items with spaces to create a single string
#     combined = ' '.join(overrides)
    
#     # Use regex to identify and fix patterns like 'param = value' to 'param=value'
#     # This handles various spacing around the equals sign
#     fixed_string = re.sub(r'(\S+)\s*=\s*(\S+)', r'\1=\2', combined)
    
#     # Split the fixed string back into a list, preserving properly formatted args
#     # We split on spaces that are not within a parameter=value pair
#     processed = re.findall(r'[^\s=]+=\S+|\S+', fixed_string)
    
#     return processed

# def init_config():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--config", "-c", required=True)
#     parser.add_argument("overrides", nargs="*")  # Capture all "key=value" args
#     args = parser.parse_args()

#     # Load base config
#     config = OmegaConf.load(args.config)

#     # Parse CLI overrides using OmegaConf's native CLI parser
#     processed_overrides = process_overrides(args.overrides)
#     cli_overrides = OmegaConf.from_cli(processed_overrides)

#     # Merge configs (with type-safe automatic conversion)
#     config = OmegaConf.merge(config, cli_overrides)

#     # Convert to EasyDict if needed
#     config = OmegaConf.to_container(config, resolve=True)
#     config = edict(config)
#     return config

#################Init Config End#################

@dataclass
class DDPInfo:
    local_rank: int
    global_rank: int
    world_size: int
    device: torch.device
    is_main_process: bool
    process_seed: int



def init_distributed(seed=42):
    """
    Initialize distributed training environment and set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed for PyTorch, NumPy, and Python's random module.
                   Default is 42.
    
    Returns:
        edict: Dictionary with attribute access containing:
            - local_rank: GPU rank within the current node
            - global_rank: Global rank of the process
            - world_size: Total number of processes
            - device: The CUDA device assigned to this process
            - is_main_process: Flag to identify the main process
            - seed: The random seed used for this process
    """
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=3600)
    )
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Set random seeds
    # Each process gets a different seed derived from the base seed
    process_seed = seed + global_rank
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed) 
    np.random.seed(process_seed)
    random.seed(process_seed)
    
    # Optional: For better performance
    torch.backends.cudnn.benchmark = True
    
    return DDPInfo(local_rank, global_rank, world_size, device, global_rank==0, process_seed)
    # return edict({
    #     'local_rank': local_rank,
    #     'global_rank': global_rank,
    #     'world_size': world_size,
    #     'device': device,
    #     'is_main_process': global_rank == 0, 
    #     'seed': process_seed
    # })


