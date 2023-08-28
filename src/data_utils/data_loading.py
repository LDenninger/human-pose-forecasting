"""
    Functions to load the data of the H36M dataset directly from the directory.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import numpy as np
import torch
import os
from typing import Optional, Literal, Tuple
from pathlib import Path as P
from .meta_info import *
from .transformations import *
from tqdm import tqdm

def read_file(person_id: int, action_str: str, sub_action_id: int = None, return_tensor=False):
    """
        Read a single file from the dataset
    """

    if sub_action_id is None:
        sub_action_id = [1,2]
    else:
        sub_action_id = [sub_action_id]
    
    data = []

    for sub_action in sub_action_id:
        file_path = P(os.getcwd()) / DATASET_PATH / f'S{person_id}' / f"{action_str}_{sub_action}.txt"
        try:
            lines = open(file_path, 'r').readlines()
        except Exception as e:
            print(f'Error reading file {file_path}:\n{e}')
            continue
        for i, line in enumerate(lines):
            try:
                line = line.strip().split(',')
            except Exception as e:
                print(f'Error parsing line {i} in {file_path}:\n{e}')
                continue
            if len(line) > 0:
                if return_tensor:
                    data.append(torch.FloatTensor([float(x) for x in line]))
                else:
                    data.append(np.array([float(x) for x in line]))
    return data

def load_data(  person_id: list = DATASET_PERSONS,
                action_str: list = DATASET_ACTIONS,
                sub_action_id: list = [1,2],
                 skeleton: Optional[Literal['s26']] = None,
                  representation: Optional[Literal['axis', 'mat', 'quat', '6d']] = 'axis',
                   return_tensor: bool=False,
                   return_seq_tensor: bool=True,
                    show_progress=True,
                     return_reverse_meta=False):
    """
        Load the data from the Human3.6M dataset.

        The data is stored in the following format:
            If return_seq_tensor is False:
                [                                                       # List of all action sequence
                    [                                                   # List of joint angles at each timestep
                        np.array(joint_dim) / torch.Tensor(joint_dim)   # Joint angles either as numpy arrays or torch tensors
                    ],...
                ]
            Else:
                [                                           # List of all action sequence
                    torch.Tensor(seq_len, joint_dim)        # joint angles over complete sequence
                ]

        The meta_dir is in the following format:
        [                                           # List of meta information about each action sequence              
            {
                'person_id': int,
                'action_str': str,
                'sub_action_id': int,
                'length': int
            }
        ]

        Arguments:
            person_id (list): List of all person ids
            action_str (list): List of all action strings
            sub_action_id (list): List of all sub action ids
            skeleton (['s26]): Skeleton model to be used. This parameter determines if the data is loaded raw or processed.
            representation (['axis','mat', 'quat', '6d']): Representation to be used. For this a skeleton must be provided
            return_tensor (bool): Whether to return the data as torch tensors or numpy arrays
            return_seq_tensor (bool): Whether to return the sequence as torch tensors or a list
                Currently sequences have to be returned as a torch tensor for efficient processing of the skeleton
        Returns:
            data (list): Data in above list format
            meta_dir (list): List of meta information about each action sequence
    """

    meta_dir = []
    reverse_meta_dir = {}
    data = []
    print('Loading Human3.6M data...')
    if show_progress:
        progress_bar = tqdm(total=len(person_id) * len(action_str) * len(sub_action_id))
    seq_index = 0
    # Iterate through all files for the defined parameters and load the data
    for p_id in person_id:
        for a_str in action_str:
            for s_id in sub_action_id:
                # Read data
                new_data = read_file(p_id, a_str, s_id, return_tensor)
                len_data = len(new_data)
                # Create entry to meta dir for sequence
                meta_info = {
                    'person_id': p_id,
                    'action_str': a_str,
                    'sub_action_id': s_id,
                    'length': len_data
                }
                # Parse into a specific skeleton structure
                # If a skeleton is provided we are also able to transform to another representation
                if skeleton is not None:
                    if skeleton=='s26':
                        conversion_func = get_conversion_func(representation)
                        new_data = _parse_sequence_efficient_to_s26(torch.stack(new_data), conversion_func)
                    else:
                        raise ValueError(f'Unknown skeleton {skeleton}')
                elif return_seq_tensor:
                    new_data = torch.stack(new_data)

                data.append(new_data)
                meta_dir.append(meta_info)
                if p_id not in reverse_meta_dir.keys():
                    reverse_meta_dir[p_id] = {}
                if a_str not in reverse_meta_dir[p_id].keys():
                    reverse_meta_dir[p_id][a_str] = {}
                reverse_meta_dir[p_id][a_str][s_id] = seq_index
                seq_index += 1
                if show_progress:
                    progress_bar.update(1)
    if show_progress:
        progress_bar.close()
    if return_reverse_meta:
        return data, meta_dir, reverse_meta_dir
    
    return data, meta_dir


def get_conversion_func(representation=Literal['axis', 'mat', 'quat', '6d']):
    """
        Returns a a function to transform from axis angle representation to the given representation.
    """
    if representation == 'axis':
        return blank_processing
    elif representation =='mat':
        return axis_angle_to_matrix_direct
    elif representation == 'quat':
        return axis_angle_to_quaternion
    elif representation == '6d':
        return axis_angle_to_rotation_6d
    else:
        raise ValueError(f'Unknown representation {representation}')

def parse_to_s26(data, representation=Literal['axis', 'mat', 'quat', '6d']):
    """
        Parses the data from the Human3.6M dataset to the given representation
        using the skeleton model consisting of 26 joints. Inactive joints are simply removed.

        The data is assumed to be in the format as produced by the function load_data()

        Return:
            Data structure holding the reduced and converted joint representation
            Format:
                        [                                           # List of all action sequence
                            [                                       # List of joint angles at each timestep
                                torch.Tensor(26, repr_dim)           # Joint angles either as numpy arrays or torch tensors
                            ],...
                        ]
    """
    conversion_func = get_conversion_func(representation)
    data_edited = []
    
    for seq_id, seq in enumerate(data):
        seq_edited = []
        for frame_id, frame in enumerate(seq):
            frame_repr = _parse_datum(frame, H36M_SKELETON_STRUCTURE, conversion_func)
            seq_edited.append(frame_repr)
        data_edited.append(seq_edited)

    return data_edited

def _parse_datum(frame: torch.Tensor, skeleton_structure: dict, indices: dict, conversion_func: callable) -> torch.Tensor:
    """
        Parse a single datum of joint angles in the H36M dataset format to a skeleton representaion.

        Arguments:
            frame (torch.Tensor): Joint angles in the H36M dataset format
            skeleton_structure (dict): Skeleton structure to parse to
            indices (dict): Indices of joints in the provided datum
            conversion_func (callable): Function to convert from axis angle representation to the given representation

    """
    frame_repr = []
    for id, (joint, parent) in skeleton_structure.items():
        angle_ind = indices[id]
        angle = frame[angle_ind]
        angle = conversion_func(angle)
        frame_repr.append(angle)
    frame_repr = torch.stack(frame_repr)
    return frame_repr

def _parse_sequence_efficient_to_s26(seq: torch.Tensor, conversion_func: callable) -> torch.Tensor:
    """
        Parse a single datum of joint angles in the H36M dataset format to a skeleton representaion.

        Arguments:
            frame (torch.Tensor): Joint angles in the H36M dataset format
            skeleton_structure (dict): Skeleton structure to parse to
            indices (dict): Indices of joints in the provided datum
            conversion_func (callable): Function to convert from axis angle representation to the given representation

    """
    seq_repr = torch.FloatTensor(seq.shape[0],26,3)
    seq_repr[:,0] =  seq[:,[3, 4, 5]]
    seq_repr[:,1] =  seq[:,[6, 7, 8]]
    seq_repr[:,2] =  seq[:,[9, 10, 11]]
    seq_repr[:,3] =  seq[:,[12, 13, 14]]
    seq_repr[:,4] =  seq[:,[15, 16, 17]]
    seq_repr[:,5] =  seq[:,[21, 22, 23]]
    seq_repr[:,6] =  seq[:,[24, 25, 26]]
    seq_repr[:,7] =  seq[:,[27, 28, 29]]
    seq_repr[:,8] =  seq[:,[30, 31, 32]]
    seq_repr[:,9] = seq[:,[36, 37, 38]]
    seq_repr[:,10] = seq[:,[39, 40, 41]]
    seq_repr[:,11] = seq[:,[42, 43, 44]]
    seq_repr[:,12] = seq[:,[45, 46, 47]]
    seq_repr[:,13] = seq[:,[48, 49, 50]]
    seq_repr[:,14] = seq[:,[51, 52, 53]]
    seq_repr[:,15] = seq[:,[54, 55, 56]]
    seq_repr[:,16] = seq[:,[57, 58, 59]]
    seq_repr[:,17] = seq[:,[60, 61, 62]]
    seq_repr[:,18] = seq[:,[63, 64, 65]]
    seq_repr[:,19] = seq[:,[69, 70, 71]]
    seq_repr[:,20] = seq[:,[75, 76, 77]]
    seq_repr[:,21] = seq[:,[78, 79, 80]]
    seq_repr[:,22] = seq[:,[81, 82, 83]]
    seq_repr[:,23] = seq[:,[84, 85, 86]]
    seq_repr[:,24] = seq[:,[87, 88, 89]]
    seq_repr[:,25] = seq[:,[93, 94, 95]]
    seq_repr = torch.flatten(seq_repr, 0, 1)
    seq_repr = conversion_func(seq_repr)
    seq_repr = torch.reshape(seq_repr,(seq.shape[0], 26, -1))
    return seq_repr