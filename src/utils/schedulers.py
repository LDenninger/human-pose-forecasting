"""
    Modules to implement a learning rate scheduler for training.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
import torch.nn as nn
import math
import numpy as np
from abc import abstractmethod
from typing import Optional

from ..utils import print_
#####===== Scheduler Modules =====#####

class SchedulerBase(nn.Module):
    """
        Base class for the learning rate schedulers.
    """
    def __init__(self, optimizer: nn.Module) -> None:
        super(SchedulerBase, self).__init__()
        self.optimizer = optimizer
        self.register_buffer('learning_rate', torch.ones(1))

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

class ExponentialScheduler(SchedulerBase):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self,
                    optimizer,
                    gamma: int,
                    base_lr: int,
                    update_frequency: Optional[int] = 1,
                    warmup_steps: Optional[int] = 0):
        super().__init__(optimizer)
        self.gamma = gamma
        self.learning_rate *= base_lr
        self.base_lr = base_lr
        self.update_frequency = update_frequency
        self.warmup_steps = warmup_steps
        if self.warmup_steps == 0:
            self.warmup_lr = base_lr * 1e-2

    def compute_learning_rate(self, step: int) -> None:
        if step == 0:
            return
        if step <= self.warmup_steps:
            self.learning_rate = self.warmup_lr + (self.base_lr - self.warmup_lr) * (step / self.warmup_steps)
        if step%self.update_frequency == 0:
            self.learning_rate *= self.gamma


