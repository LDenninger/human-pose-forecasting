"""
    Dataset used for training and evaluating our method.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Literal, List
from abc import abstractmethod

from ..utils import get_conv_from_axis_angle
from .data_loading import load_data_h36m, load_data_visionlab3DPoses
from .skeleton_utils import parse_h36m_to_s26, h36m_forward_kinematics, convert_s26_to_s21, convert_s21_to_s16
from .meta_info import H36M_TRAIN_SUBJECTS, H36M_TEST_SUBJECTS, H36M_DEBUG_SPLIT, H36M_DATASET_ACTIONS

#####===== Helper Functions =====#####

def getDataset(config: dict, joint_representation: str, skeleton_model: str, is_train: Optional[bool] =True, debug: Optional[bool] = False, **kwargs) -> torch.utils.data.Dataset:
    """
        Load a dataset using a run config.

        Arguments:
            config (dict): The configuration dictionary of the dataset.
            joint_representation (str): The representation of the joints.

    """
    if config["name"] == 'h36m':
        return H36MDataset(
            seed_length=config["seed_length"],
            rot_representation=joint_representation,
            stacked_hourglass= True if skeleton_model=='s16' else False,
            reverse_prob=config['reverse_prob'],
            target_length=config["target_length"],
            down_sampling_factor=config["downsampling_factor"],
            sequence_spacing=config["spacing"],
            is_train=is_train,
            debug=debug
        )
    else:
        raise NotImplementedError(f'Dataset {config["name"]} is not implemented.')
    
#####===== H36M Dataset =====#####
class H36MDatasetBase(Dataset):
    """
        H36M dataset base class.
        This module implements the base data loading and can be used by other datasets for more advanced data loading techniques.
    """

    def __init__(self, 
                    seed_length: int,
                    target_length: int,
                    actions: Optional[List[str]] = H36M_DATASET_ACTIONS,
                    raw_data: Optional[bool] = False,
                    is_train: Optional[bool]=True,
                    debug: Optional[bool]=False)-> None:
        """
            Initialize the data loader.
            Arguments:
                seed_length (int): The length of the seed sequence.
                target_length (int): The length of the target sequence.
                actions (List[str]): The list of actions to use. Default: Use all actions.
                raw_data (bool): Whether to load the raw data or load the processed data in a skeleton format. Default: False.
                is_train (bool): Whether to load the training data or the test data. Default: True.
                debug (bool): Whether to load the debug split only consisting of a single person. Default: False.
        """
        ##== Meta Information ==##
        self.seed_length = seed_length
        self.target_length = target_length

        # Load the data from disk
        self.data, self.meta_info = load_data_h36m(person_id = (H36M_TRAIN_SUBJECTS if is_train else H36M_TEST_SUBJECTS) if not debug else H36M_DEBUG_SPLIT,
                                                    action_str=actions,
                                                     raw_data=raw_data,
                                                             return_tensor=True)
    
        
    def get_mean_variance(self):
        """
            Computes the mean and variance over the data for normalization.
        """
        mean = torch.mean(self.data, dim=0) + torch.finfo(torch.float32).eps
        var = torch.var(self.data, dim=0) + torch.finfo(torch.float32).eps
        return mean, var
    
    def _compute_valid_indices(self):
        """
            Compute the valid start indices for sequences.
            The valid indices are computed w.r.t. the sequence spacing, seed and target length, such that a sampled sequence is not overlapping different actions/persons etc.
            This is later used for easy indexing of the data in a flattened tensor.


        """

        highest_ind= 0
        valid_indices = []
        labels = []
        # Iterate over each sequence and add indices for the sequence to the valid indices
        for i, sequence in enumerate(self.data):
            # produce sequence indices
            seq_ind = torch.arange(len(sequence))
            seq_len = sequence.shape[0]
            # Cutout last frames such that sequences dont overlap
            seq_ind = seq_ind[:-(self.seed_length + self.target_length)]
            # Add previously highest index to get absolut index
            seq_ind = seq_ind + highest_ind
            highest_ind += seq_len
            if self.sequence_spacing!= 0:
                seq_ind = seq_ind[::self.sequence_spacing]
            valid_indices.append(seq_ind)
            if self.return_label:
                labels.extend([self.meta_info[i] for jjk in range(len(seq_ind))])
            #seq_ends.append(highest_ind)
        
        valid_indices = torch.cat(valid_indices)
        if self.return_label:
            return valid_indices, labels
        return valid_indices
    
    def _flatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """
            Flatten the data into a single dimension for all frames to correspond to the computed indices.

            Arguments:
                data (torch.Tensor): The data to flatten.

        """
        for i, sequence in enumerate(data):
            if i==0:
                flatten_data = sequence
                continue
            flatten_data = torch.cat((flatten_data, sequence), dim=0)
        return flatten_data
    
    def __len__(self):
        return len(self.valid_indices)
    
class H36MDataset(H36MDatasetBase):
    """
        Main dataset used within the project. This loads the data from the H36M dataset to torch tensors and applies the given
        transformations for the joint and skeleton representation.
    """

    def __init__(self, 
                    seed_length: int,
                    target_length: int,
                    actions: Optional[List[str]] = H36M_DATASET_ACTIONS,
                    down_sampling_factor: int=1,
                    sequence_spacing: int=0,
                    reverse_prob: Optional[float] = 0.0,
                    rot_representation: Optional[Literal['axis', 'mat', 'quat', '6d', 'pos']] = 'axis',
                    stacked_hourglass: Optional[bool] = False,
                    absolute_position: Optional[bool] = False,
                    return_label: Optional[bool] = False,
                    is_train: Optional[bool]=True,
                    raw_data: Optional[bool] = False,
                    debug: Optional[bool]=False):
        """
            Initialize the H36M dataset. This loads the data from the H36M dataset to torch tensors.

            Arguments:
                seed_length (int): The length of the seed sequence.
                target_length (int): The length of the target sequence.
                actions (List[str], optional): The list of actions to use. Default: Use all actions.
                down_sampling_factor (int, optional): The downsampling factor for the sequences to minimize redundant frames. Default: 1 -> 25 FPS.
                sequence_spacing (int, optional): The number of frames between two independent sequences. Default: 0.
                reverse_prob (float, optional): The probability of reversing a sequence. Default: 0.0.
                rot_representation (['axis','mat', 'quat', '6d', 'pos'], optional): The representation to use.
                    Rotation representations are automatically parsed to the 26 joint skeleton.
                    Position representations are automatically parsed to the 21 joint skeleton to remove positionally redundant joints.
                stacked_hourglass (bool, optional): Whether to use the stacked hourglass representation. This is only possible for position representations. Default: False.
                absolute_position (bool, optional): Whether to use the absolute positions. This is only possible for position representations. Default: False.
                return_label (bool, optional): Whether to return the action or not. Default: False.
                is_train (bool, optional): Whether to load the training or test data. Default: True.
        """
        super().__init__(seed_length, target_length, actions, raw_data=True, is_train=is_train, debug=debug)
        ##== Meta Information ==##
        self.sequence_spacing = sequence_spacing
        self.down_sampling_factor = down_sampling_factor
        self.reverse_prob = reverse_prob if is_train else 0.0
        self.return_label = return_label

        ##== Data Handling ==##
        if down_sampling_factor != 1:
            for sequence in self.data:
                sequence = sequence[::down_sampling_factor]

        self.valid_indices = self._compute_valid_indices()
        if self.return_label:
            self.valid_indices, self.labels = self.valid_indices

        # Write the data into a flattened tensor for easier indexing
        self.data = self._flatten_data(self.data)
        if not raw_data:
            if rot_representation != 'pos':
                self.data = parse_h36m_to_s26(self.data, get_conv_from_axis_angle(rot_representation))
            else:
                self.data, _ = h36m_forward_kinematics(self.data, 'axis', hip_as_root=not absolute_position)
                self.data = convert_s26_to_s21(self.data, interpolate=False)
                if stacked_hourglass:
                    self.data = convert_s21_to_s16(self.data, interpolate=False)
                self.data /= 1000
            
        self.full_length = len(self.data)

    def __getitem__(self, x):
        """
            Return a single sampled sequence from the dataset.
        """
        seq_start = self.valid_indices[x]
        sequence = self.data[seq_start:(seq_start + self.seed_length + self.target_length)]
        # Reverse the sequence with given probability
        if self.reverse_prob != 0.0 and torch.randn(1).item() < self.reverse_prob:
            sequence = torch.flip(sequence, dims=(0,))
        if self.return_label:
            return sequence, self.labels[x]
        return sequence

#####===== AIS Dataset =====#####

class AISDataset(Dataset):
    """
        Dataset class for the AIS dataset recorded within the lab.
        This dataset is only used for evaluation.
    """

    def __init__(self, 
                 seed_length: int,
                 target_length: int,
                 sequence_spacing: int,
                 absolute_position: Optional[bool] = False):
        """
            Initialize the dataset.

            Arguments:
                seed_length (int): The length of the seed sequence.
                target_length (int): The length of the target sequence.
                sequence_spacing (int): The number of frames between two independent sequences.
                absolute_position (bool, optional): Whether to use the absolute positions. This is only possible for position representations. Default: False.
        """
        self.seed_length = seed_length
        self.target_length = target_length
        self.sequence_spacing = sequence_spacing
        self.absolute_position = absolute_position
        self.data = self._load_data()
        self.len = len(self.data)

    def __getitem__(self, x):
        """ Retrieve a single sampled sequence from the dataset. """
        return self.data[x]

    def _load_data(self):
        """
            Load the raw data and sample it into sequences according to the sequence spacing, seed and target length.
        """
        full_data = []
        # Load the raw data
        raw_data = load_data_visionlab3DPoses(self.absolute_position)
        # Iterate through each sequence
        for fname, data in raw_data.items():
            max_ind = len(data)
            # Compute possible starting indices
            start_inds = np.arange(len(data))
            start_inds = start_inds[::self.sequence_spacing]
            for sid in start_inds:
                # Append data for the sampled sequence if it does not exceed the last frame of the complete sequence
                if sid+self.target_length+self.seed_length <= max_ind:
                    full_data.append(data[sid:sid+self.target_length+self.seed_length])
        return full_data

    def __len__(self):
        return self.len