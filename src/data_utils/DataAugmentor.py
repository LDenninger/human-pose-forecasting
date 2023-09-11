"""
    This files containes a module that is used to augment the data according to the configuration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from typing import Optional

from ..utils import print_

class DataAugmentor(nn.Module):

    def __init__(self, 
                  normalize: Optional[bool] = False,
                   reverse_prob: Optional[bool] = False,
                    snp_noise_prob: Optional[int] = 0.0,
                     joint_cutout_prob: Optional[int] = 0.0,
                      timestep_cutout_prob: Optional[int] = 0.0):
        super().__init__()
        self.normalize = normalize
        self.norm_mean = None
        self.norm_var = None
        self.reverse_prob = reverse_prob
        self.snp_noise_prob = snp_noise_prob
        self.joint_cutout_prob = joint_cutout_prob
        self.timestep_cutout_prob = timestep_cutout_prob
        self.train_pipeline, self.eval_pipeline = self.__init_pipeline()

    def forward(self, x: torch.Tensor, is_train: Optional[bool] = True) -> torch.Tensor:
        """
            Apply pre-defined data augmentation steps to the input tensor.

            Input shape: [batch_size, seq_len, num_joints, joint_dim]
        """
        return self.train_pipeline(x) if is_train else self.eval_pipeline(x)
    
    def reverse_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
            Unnormalizes an output tensor from the model.        
        """
        device = x.device
        if self.norm_mean is None or self.norm_var is None:
            print_("Mean and variance are not set. Normalization is not performed.", "warn")
            return x
        unnorm_x = x*self.norm_var.to(device) + self.norm_mean.to(device)
        if torch.isnan(unnorm_x).any():
            nan_ind = torch.isnan(unnorm_x)
            unnorm_x.masked_fill_(nan_ind, 0.0)
        return unnorm_x
    
    def set_mean_var(self, mean: float, var: float) -> None:
        self.norm_mean = mean
        self.norm_var = var
    
    def processing_pipeline(self, *funcs):
        """
            Returns a function that applies a sequence of data augmentation steps in a pipeline.
        """
        return lambda x: reduce(lambda acc, f: f(acc), funcs, x)
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_mean is None or self.norm_var is None:
            print_("Mean and variance are not set. Normalization is not performed.", "warn")
            return x
        return (x-self.norm_mean) / self.norm_var
    
    def _snp_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise_mask = torch.rand(x.shape[:-1], device=x.device) < self.snp_noise_prob
        noise_mask = noise_mask.unsqueeze(-1)
        return x * noise_mask
    
    def _joint_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise_mask = torch.rand(x.shape[:-2], device=x.device) < self.joint_cutout_prob
        noise_mask = noise_mask.unsqueeze(-1).unsqueeze(-1)
        return x * noise_mask
    
    def _timestep_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise_mask = torch.rand(x.shape[[0,2,3]], device=x.device) < self.timestep_cutout_prob
        noise_mask = noise_mask.unsqueeze(1).unsqueeze(-1)
        return x * noise_mask
    
    def _reverse(self, x: torch.Tensor) -> torch.Tensor:
        """
            This operation is preferred to be applied in the data loader.
        """
        if torch.rand(1) < self.reverse_prob:
            return torch.flip(x, dims=[-1])
        return x
    def _blank_processing(self, x: torch.Tensor) -> torch.Tensor:
        return x
    def __init_pipeline(self):
        """
            Returns a sequence of functions to apply the data augmentation
        """
        train_pipeline = []
        eval_pipeline = []
        if self.reverse_prob > 0:
            train_pipeline.append(self._reverse)
        if self.normalize:
            train_pipeline.append(self._normalize)
            eval_pipeline.append(self._normalize)
        if self.snp_noise_prob > 0:
            train_pipeline.append(self._snp_noise)
        if self.joint_cutout_prob > 0:
            train_pipeline.append(self._joint_noise)
        if self.timestep_cutout_prob > 0:
            train_pipeline.append(self._timestep_noise)
        if len(train_pipeline) == 0:
            train_pipeline.append(self._blank_processing)
        if len(eval_pipeline) == 0:
            eval_pipeline.append(self._blank_processing)
        
        return self.processing_pipeline(*train_pipeline), self.processing_pipeline(*eval_pipeline)