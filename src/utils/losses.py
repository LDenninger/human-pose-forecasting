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
from .logging import print_


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
    
class PositionMSE(LossBase):
    """
        Module to compute the mean squared error between joint positions
    """
    def __init__(self, reduction: Optional[Literal['mean','sum']] = 'mean'):
        super(PositionMSE, self).__init__()
        self.reduction_func = self._reduce_sum_and_mean if reduction == 'sum' else self._reduce_mean

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(output, target, reduction='none') # mse loss between joints
        loss = torch.sum(loss, dim=-1) # Sum over rotation dimensions
        loss = torch.sqrt(loss) # mse over all rotation dimensions
        return self.reduction_func(loss)
    
class STDWeightedPositionMSE(LossBase):
    """
        Module to compute a weighted mean squared error between joint positions.
        The weights are determined by the standard deviation within a single sequence,
        such that joints with a large movement get a higher weight.
    """
    def __init__(self, reduction: Optional[Literal['mean','sum']] = 'mean', scale: Optional[float] = 1.0):
        super(STDWeightedPositionMSE, self).__init__()
        self.scale = scale
        self.reduction_func = self._reduce_sum_and_mean if reduction =='sum' else self._reduce_mean
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the MSE loss per-joint
        loss = F.mse_loss(output, target, reduction='none')
        loss = torch.sum(loss, dim=-1) 
        loss = torch.sqrt(loss) 
        # Compute the weights using the std within the target sequence
        std = torch.std(target, dim=1, unbiased=False)
        std = torch.mean(std, dim=-1)
        std_corr = self.scale*std + torch.max(std)
        weights = std_corr/torch.sum(std_corr, dim=1).unsqueeze(-1)
        # Compute a weighted mean across the joints
        loss = torch.sum(loss*weights.unsqueeze(1), dim=-1)
        return self.reduction_func(loss)

class HandWeightedPositionMSE(LossBase):
    """
        Module to compute a weighted mean squared error between joint positions.
        The weights are hardcoded by hand depending on the difficulty of a joint.
        Possible weights are 0.1, 0.3, 0.7, 1.0. 
        With increasing distance to the torso the weights are also increased.
    """
    def __init__(self, reduction: Optional[Literal['mean','sum']] = 'mean', weights: Optional[list] = None):
        super(HandWeightedPositionMSE, self).__init__()
        if weights is None:
            self.weights = torch.FloatTensor([0.1, 0.3, 0.7, 1.0, 0.3, 0.7, 1.0, 0.3, 0.3, 0.7, 0.3, 0.7, 1.0, 0.3, 0.7, 1.0]).unsqueeze(0).unsqueeze(0)
        else:
            self.weights = weights
        self.device = None
        self.reduction_func = self._reduce_sum_and_mean if reduction =='sum' else self._reduce_mean

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
            Compute the MSE loss on the joints. Each joint is weighted by hardcoded weights.
        """
        if self.device is None:
            self.device = output.device
            self.weights = self.weights.to(self.device)
        loss = F.mse_loss(output, target, reduction='none') # mse loss between joints
        loss = torch.sum(loss, dim=-1)
        loss = torch.sqrt(loss)
        loss = torch.sum(loss*self.weights, dim=-1)
        return self.reduction_func(loss)

class LearningPositionMSE(LossBase):
    """
        Module to compute a weighted mean squared error between joint positions.
        The weights are first fixed for a number of warmup steps to produce the same results as the position mse.
        After the model is trained for a number of steps, the weights can also be learned to improve the loss.
    """
    def __init__(self, 
                 reduction: Optional[Literal['mean','sum']] ='mean', 
                 num_joints: Optional[int] = 16, 
                 warmup_steps: Optional[int] = 1000):
        super(LearningPositionMSE, self).__init__()
        self.reduction_func = self._reduce_sum_and_mean if reduction =='sum' else self._reduce_mean
        self.warmup_steps = warmup_steps
        self.register_parameter('weights', nn.Parameter(torch.zeros(num_joints), requires_grad=True if warmup_steps == 0 else False))
        self.steps = 0
        self.learning_active = False if warmup_steps == 0 else True
        self.device = None

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Do not learn during warm-up steps
        if self.device is None:
            self.device = output.device
            self.weights = nn.Parameter(self.weights.to(self.device))
        if not self.learning_active:
            if self.steps == self.warmup_steps:
                self.weights.requires_grad = True
                self.learning_active = True
            else:
                self.steps += 1
        # Compute weights using the softmax
        weights_sm = F.softmax(self.weights, dim=-1).unsqueeze(0).unsqueeze(0)
        loss = F.mse_loss(output, target, reduction='none') # mse loss between joints
        loss = torch.sum(loss, dim=-1)
        loss = torch.sqrt(loss)
        loss = torch.sum(loss*weights_sm, dim=-1)
        return self.reduction_func(loss)

    
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
        R_diffs = input @ target.permute(0, 2, 1)
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        return self.reduction_func(dists)
    
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

class AbsolutePositionLoss(LossBase):

    def __init__(self,
                  weight_factor: float,
                  reduction: Optional[Literal['mean','sum']] = 'mean',
                   rotation_loss: Optional[Literal['mse','geodesic','euler','quaternion','rotation6d']] = 'mse',
                    rot_representation: Optional[Literal['axis', 'mat', 'quat', '6d']] = 'mat'):
        self.weight_factor = weight_factor
        self.reduction_func = self._reduce_sum_and_mean if reduction =='sum' else self._reduce_mean
        if rotation_loss == "mse":
            self.rotation_loss = PerJointMSELoss(org_representation=rot_representation)
        elif rotation_loss == 'geodesic':
            self.rotation_loss = GeodesicLoss(org_representation=rot_representation, reduction=reduction)
        elif rotation_loss == 'euler':
            if rot_representation!= 'euler':
                print_('Euler loss only works with euler rotation representation.', 'warn')
            self.rotation_loss = EulerLoss(reduction)
        elif rotation_loss == 'quaternion':
            if rot_representation!= 'quaternion':
                print_('Quaternion loss only works with quaternion rotation representation.', 'warn')
            self.rotation_loss = QuaternionLoss(reduction)
        elif rotation_loss == 'rotation6d':
            if rot_representation!= 'rotation6d':
                print_('Rotation6D loss only works with rotation6d rotation representation.', 'warn')
            self.rotation_loss = Rotation6DLoss(reduction)
        else:
            raise ValueError(f"Loss {rotation_loss} is not supported.")
        
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        output_pos = output[:,:,0,:3]
        output_rot = output[:,:,1:]
        target_pos = target[:,:,0,:3]
        target_rot = target[:,:,1:]
        loss_rot = self.rotation_loss(output_rot, target_rot)
        loss_pos = F.mse_loss(output_pos, target_pos, reduction='none')
        loss_pos = torch.sum(loss_pos, dim=-1) # Sum over position dimensions
        loss_pos = torch.sqrt(loss_pos) # mse over all position dimensions
        loss_pos = self._reduce_sum_and_mean(loss_pos)
        loss = loss_rot + self.weight_factor * loss_pos
        return loss
