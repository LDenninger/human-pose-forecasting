"""
    This files containes a module that is used to augment the data according to the configuration.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from typing import Optional, Tuple

from ..utils import print_

def get_data_augmentor(config: dict) -> nn.Module:

    return DataAugmentor(
            normalize=config['normalize'],
            reverse_prob=config['reverse_prob'],
            snp_noise_prob=config['snp_noise_prob'],
            snp_portion=config['snp_noise_portion'],
            joint_cutout_prob=config['joint_cutout_prob'],
            num_joint_cutout=config['joint_cutout_portion'],
            timestep_cutout_prob=config['timestep_cutout_prob'],
            num_timestep_cutout=config['timestep_cutout_portion'],
            gaussian_noise_prob=config['gaussian_noise_prob'],
            gaussian_noise_std=config['gaussian_noise_std']
        )
    


class DataAugmentor(nn.Module):
    """
        Data augmentation module that is used to augment and process the data.
    """

    def __init__(self, 
                  normalize: Optional[bool] = False,
                   reverse_prob: Optional[bool] = False,
                    snp_noise_prob: Optional[float] = 0.0,
                    snp_portion: Optional[Tuple[float, float]] = (0.0,0.0),
                     joint_cutout_prob: Optional[float] = 0.0,
                     num_joint_cutout: Optional[Tuple[int, int]] = (0,0),
                      timestep_cutout_prob: Optional[int] = 0.0,
                      num_timestep_cutout: Optional[Tuple[int, int]] = (0,0),
                       gaussian_noise_prob: Optional[float] = 0.0,
                       gaussian_noise_std: Optional[float] = 0.0,):
        """
            Initialize the data augmentation module.
            Arguments:
                normalize (bool, optional): Whether to normalize the data. Defaul: False.
                    For this option the data augmentor needs to be passed the mean and variance for the training dataset.
                reverse_prob (bool, optional): Whether to reverse the provided batch. Default: False.
                snp_noise_prob (float, optional): Probability of adding SNP noise to the input. Default: 0.0.
                snp_portion (Tuple[float, float], optional): Proportion of the input to add SNP noise in percentage. Default: (0.0,0.0).
                joint_cutout_prob (float, optional): Probability of adding cuting out joints of a sequence. Default: 0.0.
                num_joint_cutout (Tuple[int, int], optional): Minimum and maximum of joints to cut out. Default: (0,0).
                timestep_cutout_prob (int, optional): Probability of cutting out single timesteps. Default: 0.0.
                num_timestep_cutout (Tuple[int, int], optional): Minimum and maximum of timesteps to cut out. Default: (0,0).
                gaussian_noise_prob (float, optional): Probability of adding gaussian noise to the input. Default: 0.0.
                gaussian_noise_std (float, optional): Standard deviation of the gaussian noise. Default: 0.0.

        """
        super().__init__()
        self.normalize = normalize
        self.norm_mean = None
        self.norm_var = None
        self.reverse_prob = reverse_prob
        self.snp_noise_prob = snp_noise_prob
        self.snp_portion = snp_portion
        self.joint_cutout_prob = joint_cutout_prob
        self.num_joint_cutout = num_joint_cutout
        self.timestep_cutout_prob = timestep_cutout_prob
        self.num_timestep_cutout = num_timestep_cutout
        self.gaussian_noise_prob = gaussian_noise_prob
        self.gaussian_noise_std = gaussian_noise_std
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
        unnorm_x = x*torch.sqrt((self.norm_var.to(device)+torch.finfo(torch.float32).eps))+ self.norm_mean.to(device)
        if torch.isnan(unnorm_x).any():
            nan_ind = torch.isnan(unnorm_x)
            unnorm_x.masked_fill_(nan_ind, 0.0)
        return unnorm_x
    
    def set_mean_var(self, mean: float, var: float) -> None:
        """
            Set the mean and variance for the data normalization.
        """
        self.norm_mean = mean
        self.norm_var = var
    
    def processing_pipeline(self, *funcs):
        """
            Returns a function that applies a sequence of data augmentation steps in a pipeline.
        """
        return lambda x: reduce(lambda acc, f: f(acc), funcs, x)
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """ Normalize the data """
        if self.norm_mean is None or self.norm_var is None:
            print_("Mean and variance are not set. Normalization is not performed.", "warn")
            return x
        return (x-self.norm_mean) / (torch.sqrt(self.norm_var)+torch.finfo(torch.float32).eps)
    
    def _snp_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
            Salt'n'Pepper noise. Single joints across all time steps are cut out and set to zero.
        """
        # Get the shape of the input
        full_length = x.shape[1]*x.shape[2]
        bs = x.shape[0]
        # Indices of the batches that are added with snp noise
        batch_mask = torch.rand(bs, device=x.device) < self.snp_noise_prob
        # For each batch compute the portion of the input to be cutted
        portion_to_cut = self.snp_portion[0] + (self.snp_portion[1] - self.snp_portion[0])*torch.rand(bs, device=x.device)
        portion_to_cut = torch.where(batch_mask, portion_to_cut, torch.tensor(0, device=x.device))
        # Compute the noise mask for each batch
        noise_mask = torch.ones((bs,full_length), device=x.device)
        # Since we have different length of the index tensors it is hard to do it batch-wise
        for i, portion in enumerate(portion_to_cut):
            random_ind = torch.randperm(full_length, device=x.device)
            ind = random_ind[:(torch.floor(portion*full_length).int())] 
            noise_mask[i, ind] = 0.0
        noise_mask = noise_mask.view(-1, x.shape[1], x.shape[2], 1)
        return x * noise_mask
    
    def _joint_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
            Single joints across all time steps are cut out and set to zero.

        """
        bs = x.shape[0]
        # batches to apply joint cutout to
        batch_mask = torch.rand(bs, device=x.device) < self.joint_cutout_prob
        joints_to_cut = torch.randint(self.num_joint_cutout[0], self.num_joint_cutout[1], size=(bs,), device=x.device)
        for i, num in enumerate(joints_to_cut):
            if not batch_mask[i]:
                continue
            joint_ids = torch.randint(x.shape[2], size=(num,), device=x.device)
            x[i, :, joint_ids, :] = torch.zeros((1,1,num,1), device=x.device)
        return x 
    
    def _timestep_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
            Single time steps are completely cut out and set to zero.
        """
        bs = x.shape[0]
        batch_mask = torch.rand(bs, device=x.device) < self.timestep_cutout_prob
        timesteps_to_cut = torch.randint(self.num_timestep_cutout[0], self.num_timestep_cutout[1], size=(bs,), device=x.device)
        for i, num in enumerate(timesteps_to_cut):
            if not batch_mask[i]:
                continue
            timestep_ids = torch.randint(x.shape[1], size=(num,), device=x.device)
            x[i, timestep_ids, :, :] = torch.zeros((1,num,1,1), device=x.device)

        return x
    
    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
            Gaussian noise on each joint.
        """
        bs = x.shape[0]
        # Compute which batch to add noise to
        batch_mask = torch.rand(bs, device=x.device) < self.gaussian_noise_prob
        # Compute the additive gaussian noise
        noise = self.gaussian_noise_std * torch.randn(size=x.shape, device=x.device)
        noise = torch.where(batch_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), noise, torch.zeros(x.shape, device=x.device))
        return x + noise
    
    def _reverse(self, x: torch.Tensor) -> torch.Tensor:
        """
            Reverse the provided batch.
            This operation is preferred to be applied in the data loader since we want it working on single sequences.
        """
        if torch.rand(1) < self.reverse_prob:
            return torch.flip(x, dims=[-1])
        return x
    def _blank_processing(self, x: torch.Tensor) -> torch.Tensor:
        return x
    def __init_pipeline(self):
        """
            Initialize the data augmentation pipeline according to the parameters provided at initialization separately for training and test inputs.

        """
        train_pipeline = []
        eval_pipeline = []
        # Reverse sequences
        if self.reverse_prob > 0:
            train_pipeline.append(self._reverse)
        # Gaussian noise
        if self.gaussian_noise_prob > 0:
            train_pipeline.append(self._gaussian_noise)
        # Normalization
        if self.normalize:
            train_pipeline.append(self._normalize)
            eval_pipeline.append(self._normalize)
        # Salt'n'Pepper noise
        if self.snp_noise_prob > 0:
            train_pipeline.append(self._snp_noise)
        # Joint cutout
        if self.joint_cutout_prob > 0:
            train_pipeline.append(self._joint_noise)
        # Timestep cutout
        if self.timestep_cutout_prob > 0:
            train_pipeline.append(self._timestep_noise)
        # If not augmentation is defined, add blank processing function
        if len(train_pipeline) == 0:
            train_pipeline.append(self._blank_processing)
        if len(eval_pipeline) == 0:
            eval_pipeline.append(self._blank_processing)
        
        return self.processing_pipeline(*train_pipeline), self.processing_pipeline(*eval_pipeline)
