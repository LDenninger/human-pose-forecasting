"""
    Modules that implement different losses on the joint representations.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Literal
from abc import abstractmethod

from .transformations import get_conv_to_rotation_matrix, matrix_to_axis_angle


######===== Base Module =====#####

class LossBase(nn.Module):
    """
        Base class for loss modules.
        This provides some basic functionalities to prevent redundant code.

    """

    def __init__(self):
        super(LossBase, self).__init__()
    
    @abstractmethod
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass
    
    ###=== Helper Functions ===###
    ##== Input Reductions ==##
    def _flatten_leading(self, input: torch.Tensor) -> torch.Tensor:
        return torch.reshape(input, (-1, input.shape[-1]))

    ##== Output Reductions ==##
    def _reduce_sum(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sum(input)
    
    def _reduce_mean(self, input: torch.Tensor) -> torch.Tensor:
        return torch.mean(input)
    
    def _reduce_sum_except_first(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sum(input, dim=torch.arange(1, len(input.shape)).tolist())
    
    def _reduce_sum_and_mean(self, input: torch.Tensor) -> torch.Tensor:
        result = self._reduce_sum_except_first(input)
        return torch.mean(result)

#####===== Loss Modules =====#####

class PerJointMSELoss(LossBase):
    """
        Module to compute the per joint mean squared error loss on the rotation matrix.
        This is equal to the loss function used in the original paper.
    """

    def __init__(self, org_representation: Optional[Literal['axis','mat', 'quat', '6d']] = 'mat'):
        super(PerJointMSELoss, self).__init__()
        self.conversion_func = get_conv_to_rotation_matrix(org_representation)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output, target = self.conversion_func(output), self.conversion_func(target)
        loss = F.mse_loss(output, target, reduction='none') # mse loss between joints
        loss = torch.sum(loss, dim=-1) # Sum over rotation dimensions
        loss = torch.sqrt(loss) # mse over all rotation dimensions
        return self._reduce_sum_and_mean(loss)
    
class GeodesicLoss(LossBase):
    """
        Module to compute the geodesic loss on an arbitrary rotation representation.
    """

    def __init__(self, 
                  org_representation: Optional[Literal['axis', 'mat', 'quat', '6d']] = 'mat',
                   reduction: Optional[Literal['mean','sum']] = 'mean'):
        super(GeodesicLoss, self).__init__()
        self.org_representation = org_representation
        self.reduction_func = self._reduce_sum_and_mean if reduction == 'sum' else self._reduce_mean
        self.conversion_func = get_conv_to_rotation_matrix(org_representation)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
            Forward function to compute the geodesic loss.
            The output and target is assumed to be provided in the rotation representation defined by the org_representation parameter.
            If rotation matrices are directly inputted, we assume they are flattened.
        """
        output, target = self._flatten_leading(output), self._flatten_leading(target)
    
        output = self.conversion_func(output)
        target = self.conversion_func(target)
        r = torch.matmul(output, target.transpose(-2,-1))
        angles = matrix_to_axis_angle(r)
        angles = torch.linalg.vector_norm(angles, dim=-1)
        return self.reduction_func(angles)
    
class EulerLoss(LossBase):
    """
        Module to compute a loss on euler angles using an absolute difference between two rotation.

        Implementation according to: https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
    """
    def __init__(self, reduction: Optional[Literal['mean','sum']] ='mean'):
        super(EulerLoss, self).__init__()
        self.reduction_func = self._reduce_sum_and_mean if reduction =='sum' else self._reduce_mean
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        output, target = self._flatten_leading(output), self._flatten_leading(target)
        # Compute absolute differences between euler angles and compensate for redundant angles
        dist1 = torch.sum(torch.abs(output-target), dim=-1)
        dist2 = torch.sum(torch.abs(2*torch.pi + output+target), dim=-1)
        dist3 = torch.sum(torch.abs(-2*torch.pi + output-target), dim=-1)
        # Differentiable implementation of the min
        loss = torch.where(dist1<dist2, dist1, dist2)
        loss = torch.where(loss<dist3, loss, dist3)
        return self.reduction_func(loss)
    
class QuaternionLoss(LossBase):
    """
        Module to compute a loss on quaternions.

        Implementation according to: https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
    """

    def __init__(self, reduction: Optional[Literal['mean','sum']] = 'mean'):
        super(QuaternionLoss, self).__init__()
        self.reduction_func = self._reduce_sum_and_mean if reduction =='sum' else self._reduce_mean
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output, target = self._flatten_leading(output), self._flatten_leading(target)
        # Compute absolute differences between quaternions and compensate for redundant angles
        dist1 = torch.sum(torch.abs(output-target/torch.linalg.norm(target, dim=-1).unsqueeze(-1)), dim=-1)
        dist2 = torch.sum(torch.abs(output+target/torch.linalg.norm(target, dim=-1).unsqueeze(-1)), dim=-1)
        loss = torch.where(dist1<dist2, dist1, dist2)
        return self.reduction_func(loss)
    
class Rotation6DLoss(LossBase):
    """
        Module to compute the loss on the 6D rotation representation.

        Implementation according to: https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
    """

    def __init__(self, reduction: Optional[Literal['mean','sum']] = 'mean'):
        super(Rotation6DLoss, self).__init__()
        self.reduction_func = self._reduce_sum_and_mean if reduction =='sum' else self._reduce_mean
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        loss = torch.sum(torch.abs(output - target), dim=-1)
        return self.reduction_func(loss)

