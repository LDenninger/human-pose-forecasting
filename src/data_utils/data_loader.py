import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Literal, List

from .data_loading import load_data_h36m
from .meta_info import H36M_TRAIN_SUBJECTS, H36M_TEST_SUBJECTS, H36M_DEBUG_SPLIT, H36M_DATASET_ACTIONS

#####===== Helper Functions =====#####

def getDataset(config: dict, joint_representation: str, skeleton_model: str, is_train: Optional[bool] =True,  **kwargs) -> torch.utils.data.Dataset:
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
            skeleton_model=skeleton_model,
            target_length=config["target_length"],
            down_sampling_factor=config["downsampling_factor"],
            sequence_spacing=config["spacing"],
            is_train=is_train
        )
    else:
        raise NotImplementedError(f'Dataset {config["name"]} is not implemented.')

class H36MDataset(Dataset):
    def __init__(self, seed_length: int,
                        target_length: int,
                         actions: Optional[List[str]] = H36M_DATASET_ACTIONS,
                          down_sampling_factor: int=1,
                           sequence_spacing: int=0,
                            skeleton_model: Optional[Literal['s26', None]] = None,
                             rot_representation: Optional[Literal['axis', 'mat', 'quat', '6d', None]] = None,
                              return_label: Optional[bool] = False,
                               is_train=True):
        """
            Initialize the H36M dataset. This loads the data from the H36M dataset to torch tensors.

            Arguments:
                seed_length (int): The length of the seed sequence.
                target_length (int): The length of the target sequence.
                down_sampling_factor (int, optional): The downsampling factor for the sequences to minimize redundant frames. Defaults to 1.
                sequence_spacing (int, optional): The number of frames two independent sequences. Defaults to 0.
                skeleton_model (['s26'], optional): The skeleton model to use. Defaults to None.
                rot_representation (['axis','mat', 'quat', '6d'], optional): The rotation representation to use. Defaults to axis angles as provided by the datase.
                return_label (bool, optional): Whether to return the label or not. Defaults to False.
                is_train (bool, optional): Whether to load the training or test data. Defaults to True.

        """
        ##== Meta Information ==##
        self.seed_length = seed_length
        self.target_length = target_length
        self.sequence_spacing = sequence_spacing
        self.down_sampling_factor = down_sampling_factor
        self.return_label = return_label

        # Load the data from disk
        self.data, self.meta_info = load_data_h36m(person_id = H36M_TRAIN_SUBJECTS if is_train else H36M_TEST_SUBJECTS,
                                                    action_str=actions,
                                                     skeleton=skeleton_model,
                                                         representation=rot_representation,
                                                             return_tensor=True)
        # Downsample the data
        if down_sampling_factor != 1:
            for sequence in self.data:
                sequence = sequence[::down_sampling_factor]

        self.valid_indices = self._compute_valid_indices()
        if self.return_label:
            self.valid_indices, self.labels = self.valid_indices

        # Write the data into a flattened tensor for easier indexing
        for i, sequence in enumerate(self.data):
            #import ipdb; ipdb.set_trace()
            if i==0:
                flatten_data = sequence
                continue
            flatten_data = torch.cat((flatten_data, sequence), dim=0)
        self.data = flatten_data
        self.full_length = len(self.data)


    def __getitem__(self, x):
        seq_start = self.valid_indices[x]
        sequence = self.data[seq_start:(seq_start + self.seed_length + self.target_length)]
        if self.return_label:
            return sequence, self.labels[x]
        return sequence
        

    def __len__(self):
        return len(self.valid_indices)
    
    def _compute_valid_indices(self):
        """
            Compute the valid start indices for sequences.
            This is later used for easy indexing of the data.


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
    
    
