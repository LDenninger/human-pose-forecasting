import torch
from torch.utils.data import Dataset
import numpy as np

from .data_loading import load_data
from .meta_info import TRAIN_SUBJECTS, TEST_SUBJECTS, DEBUG_SPLIT


class H36MDataset(Dataset):
    def __init__(self, sequence_length: int,
                        down_sampling_factor: int=1,
                            sequence_spacing: int=0,
                                is_train=True):

        self.sequence_length = sequence_length
        self.sequence_spacing = sequence_spacing
        self.down_sampling_factor = down_sampling_factor

        if is_train:
            self.data, self.meta_info = load_data(person_id = DEBUG_SPLIT, return_tensor=True)
        else:
            self.data, self.meta_info = load_data(person_id = TEST_SUBJECTS, return_tensor=True)
        if down_sampling_factor != 1:
            for sequence in self.data:
                sequence = sequence[::down_sampling_factor]
         
        
        self.valid_indices, self.labels, self.seq_ends = self._compute_valid_indices()

        flatten_data = []
        for sequence in self.data:
            flatten_data.extend(sequence)
        self.data = flatten_data
        self.full_length = len(self.data)


    def __getitem__(self, x):
        seq_start = self.valid_indices[x]
        sequence = torch.stack(self.data[seq_start:(seq_start + self.sequence_length)], dim=0)

        return sequence, self.labels[x]
        

    def __len__(self):
        return len(self.valid_indices)
    
    def _compute_valid_indices(self):

        highest_ind= 0
        valid_indices = []
        seq_ends = []
        labels = []
        for i, sequence in enumerate(self.data):
            seq_ind = torch.arange(len(sequence))
            if i!= 0:            
                seq_ind += highest_ind
            highest_ind += len(sequence)
            if self.sequence_spacing!= 0:
                seq_ind = seq_ind[::self.sequence_spacing]
            valid_indices.append(seq_ind)
            labels.extend([self.meta_info[i] for jjk in range(len(seq_ind))])
            seq_ends.append(highest_ind)

        
        valid_indices = torch.cat(valid_indices)
        return valid_indices, labels, seq_ends

