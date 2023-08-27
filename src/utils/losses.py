"""
    Modules that implement different losses on the joint representations.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

######===== Base Module =====#####

class LossBase(nn.Module):

    def __init__(self):
        super(LossBase, self).__init__()
    
    @abstractmethod
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

#####===== Loss Modules =====#####

class PerJointMSELoss(LossBase):

    def __init__(self):
        super(PerJointMSELoss, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(output, target, reduction='none') # mse loss between joints
        loss = torch.sum(loss, dim=-1) # Sum over rotation dimensions
        loss = torch.sqrt(loss) # mse over all rotation dimensions
        loss = torch.sum(loss, dim=-1) # Sum over all joints
        loss = torch.sum(loss, dim=-1) # sum over all timesteps
        return torch.mean(loss) # average over the batch
