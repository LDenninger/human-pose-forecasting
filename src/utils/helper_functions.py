import numpy as np
import torch
import torch.nn as nn
import os
from typing import Optional

from data_utils import H36MDataset
from .schedulers import SchedulerBase, SPLScheduler
from .losses import LossBase, PerJointMSELoss

#####===== Random Seed =====#####
def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all

#####===== Setup Functions =====#####
# These functions are intended to minimize redundant code for initialization.

def getScheduler(config: dict, optimizer: nn.Module, **kwargs) -> SchedulerBase:
    if config["TYPE"] == 'baseline':
        assert "emb_size" in kwargs.keys(), 'Please provide the embedding size for the baseline scheduler.'
        return SPLScheduler(
            optimizer=optimizer,
            emb_size = kwargs.pop("emb_size"),
            warmup = config["WARMUP"]
        )
    else:
        raise NotImplementedError(f'Scheduler {config["TYPE"]} is not implemented.')
    
def getLoss(config: dict, **kwargs) -> LossBase:
    if config["TYPE"] =='mse':
        return PerJointMSELoss()
    else:
        raise NotImplementedError(f'Loss {config["TYPE"]} is not implemented.')

def getOptimizer(config: dict, model: nn.Module, **kwargs) -> nn.Module:
    if config["TYPE"] == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config["LEARNING_RATE"],
            betas=config["BETAS"],
            eps=config['EPSILON']
        )
    else:
        raise NotImplementedError(f'Optimizer {config["TYPE"]} is not implemented.')

def getDataset(config: dict, is_train: Optional[bool] =True, **kwargs) -> torch.utils.data.Dataset:
    if config["NAME"] == 'h36m':
        return H36MDataset(
            seed_length=config["SEED_LENGTH"],
            target_length=config["TARGET_LENGTH"],
            down_sampling_factor=config["DOWNSAMPLING_FACTOR"],
            sequence_spacing=config["SPACING"],
            is_train=is_train
        )
    else:
        raise NotImplementedError(f'Dataset {config["NAME"]} is not implemented.')
    