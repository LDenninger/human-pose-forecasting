"""
    This files containes a module that is used to augment the data according to the configuration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from typing import Optional

class DataAugmentor(nn.Module):

    def __init__(self, 
                  normalize: Optional[bool] = False,
                   reverse_prob: Optional[bool] = False,
                    snp_noise_prob: Optional[int] = 0.0,
                     joint_cutout_prob: Optional[int] = 0.0,
                      timestep_cutout_prob: Optional[int] = 0.0):
        super().__init__()
        self.normalize = normalize
        self.reverse_prob = reverse_prob
        self.snp_noise_prob = snp_noise_prob
        self.joint_cutout_prob = joint_cutout_prob
        self.timestep_cutout_prob = timestep_cutout_prob
        self.pipeline = self.__init_pipeline()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply pre-defined data augmentation steps to the input tensor.

            Input shape: [batch_size, seq_len, num_joints, joint_dim]
        """
        return self.pipeline(x)
    
    def processing_pipeline(self, *funcs):
        """
            Returns a function that applies a sequence of data augmentation steps in a pipeline.
        """
        return lambda x: reduce(lambda acc, f: f(acc), funcs, x)
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)
    
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
    
    def __init_pipeline(self):
        """
            Returns a sequence of functions to apply the data augmentation
        """
        pipeline = []
        if self.reverse:
            pipeline.append(self._reverse)
        if self.normalize:
            pipeline.append(self._normalize)
        if self.snp_noise_prob > 0:
            pipeline.append(self._snp_noise)
        if self.joint_cutout_prob > 0:
            pipeline.append(self._joint_noise)
        if self.timestep_cutout_prob > 0:
            pipeline.append(self._timestep_noise)
        return self.processing_pipeline(*pipeline)