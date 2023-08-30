"""
    Helper functions to reduce redundant code between different functions and modules.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""
import numpy as np
import torch
import torch.nn as nn
import os
from typing import Optional

from .schedulers import SchedulerBase, SPLScheduler
from .losses import LossBase, PerJointMSELoss

#####===== Random Seed =====#####
def set_random_seed(seed):
    """ Set the random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all

#####===== Setup Functions =====#####
# These functions are intended to minimize redundant code for initialization.

def getScheduler(config: dict, optimizer: nn.Module, **kwargs) -> SchedulerBase:
    if config["type"] == 'baseline':
        assert "emb_size" in kwargs.keys(), 'Please provide the embedding size for the baseline scheduler.'
        return SPLScheduler(
            optimizer=optimizer,
            emb_size = kwargs.pop("emb_size"),
            warmup = config["warmup"]
        )
    else:
        raise NotImplementedError(f'Scheduler {config["type"]} is not implemented.')
    
def getLoss(config: dict, **kwargs) -> LossBase:
    if config["type"] =='mse':
        return PerJointMSELoss()
    else:
        raise NotImplementedError(f'Loss {config["type"]} is not implemented.')

def getOptimizer(config: dict, model: nn.Module, **kwargs) -> nn.Module:
    if config["type"] == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            betas=config["betas"],
            eps=config['epsilon']
        )
    else:
        raise NotImplementedError(f'Optimizer {config["type"]} is not implemented.')


    