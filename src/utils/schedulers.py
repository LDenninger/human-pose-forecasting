"""
    Modules to implement a learning rate scheduler for training.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
import torch.nn as nn
import math
import numpy as np
from abc import abstractmethod

from ..utils import print_
#####===== Scheduler Modules =====#####

class SchedulerBase(nn.Module):
    """
        Base class for the learning rate schedulers.
    """
    def __init__(self, optimizer: nn.Module) -> None:
        super(SchedulerBase, self).__init__()
        self.optimizer = optimizer
        self.register_buffer('learning_rate', torch.zeros(1))

    def forward(self, step: int) -> None:
        """
            Forward function to update the learning rate for the next step.
        """
        self.compute_learning_rate(step)
        self.update_learning_rate()
        return

    @abstractmethod
    def compute_learning_rate(self, step: int) -> None:
        """
            Abstract method to be implemented by the schedulers.
            Computes the learning rate for the next step.
        """
        pass

    def update_learning_rate(self) -> None:
        """
            Updates the learning rate for all parameters groups at once.
            If the parameter groups have separate learning rates, this functions need to be reimplemented.
        """
        if self.learning_rate is None:
            print_('Learning rate has not been set yet.', 'warn')
            return
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate.item()


class SPLScheduler(SchedulerBase):

    def __init__(self,
                 optimizer: nn.Module,
                  emb_size: int,
                   warmup: int) -> None:
        super(SPLScheduler, self).__init__(optimizer)
        self.emb_size = emb_size
        self.warmup = warmup

    def compute_learning_rate(self, step: int) -> None:
        self.learning_rate = torch.FloatTensor([(self.emb_size ** -0.5) * np.min([step ** -0.5, step * self.warmup ** -1.5])])



