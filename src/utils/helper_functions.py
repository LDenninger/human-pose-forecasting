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
from .losses import *
from .logging import print_

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
    
def getLoss(config: str, rot_representation: Optional[Literal['axis', 'mat', 'quat', '6d']] = 'mat') -> nn.Module:
    """
        Returns the loss module based on the given configuration.
    """
    if config['type'] == "mse":
        return PerJointMSELoss(org_representation=rot_representation)
    elif config['type'] == 'geodesic':
        return GeodesicLoss(org_representation=rot_representation, reduction=config['reduction'])
    elif config['type'] == 'euler':
        if rot_representation!= 'euler':
            print_('Euler loss only works with euler rotation representation.', 'warn')
        return EulerLoss(config['reduction'])
    elif config['type'] == 'quaternion':
        if rot_representation!= 'quaternion':
            print_('Quaternion loss only works with quaternion rotation representation.', 'warn')
        return QuaternionLoss(config['reduction'])
    elif config['type'] == 'rotation6d':
        if rot_representation!= 'rotation6d':
            print_('Rotation6D loss only works with rotation6d rotation representation.', 'warn')
        return Rotation6DLoss(config['reduction'])
    elif config['type'] == 'position_mse':
        return PositionMSE()
    else:
        raise ValueError(f"Loss {config['type']} is not supported.")
    
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


#####===== Data Parsing =====#####

