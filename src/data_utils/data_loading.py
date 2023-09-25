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
from ..utils import get_conv_from_rotation_matrix, get_conv_from_vectors
from .skeleton_utils import *
from tqdm import tqdm
import json


######===== H3.6M Data Loading =====#####

def read_file_h36m(person_id: int, action_str: str, sub_action_id: int = None, return_tensor=False):
    """
        Read a single file from the dataset

        Arguments:
            person_id (int): the person to load the data for.
            action_str (str): the action to load the data for.
            sub_action_id (int): the sub-action to load the data for.
            return_tensor (bool): whether to return the data as a torch.FloatTensor or numpy array.
    """

    if sub_action_id is None:
        sub_action_id = [1,2]
    else:
        sub_action_id = [sub_action_id]
    
    data = []

    for sub_action in sub_action_id:
        file_path = P(os.getcwd()) / H36M_DATASET_PATH / f'S{person_id}' / f"{action_str}_{sub_action}.txt"
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

def load_data_h36m(  person_id: list = H36M_DATASET_PERSONS,
                action_str: list = H36M_DATASET_ACTIONS,
                sub_action_id: list = [1,2],
                representation: Optional[Literal['axis', 'mat', 'quat', '6d', 'pos']] = 'axis',
                return_tensor: bool=False,
                return_seq_tensor: bool=True,
                show_progress=True,
                raw_data: Optional[bool] = False,
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
                new_data = read_file_h36m(p_id, a_str, s_id, return_tensor)
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
                if not raw_data:
                    conversion_func = get_conv_from_rotation_matrix(representation)
                    new_data = parse_h36m_to_s26(torch.stack(new_data), conversion_func)

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

#####===== VisionLab3DPoses Data Loading =====#####

def read_file_visionlab3DPoses(path: str, return_tensor: Optional[bool] = False):
    """
        Load the data from a single file of the AIS dataset.

        Arguments:
            path (str): Path to the file
            return_tensor (bool): Whether to return the data as a torch.FloatTensor or a list.
    """
    with open(path, 'r') as file:
        pose_data = json.load(file)

    joint_positions = []

    for frame in pose_data:
        frame_data = []
        for joint in frame['person']['keypoints']:
            frame_data.append(joint['pos'])
        joint_positions.append(frame_data)

    if return_tensor:
        return torch.FloatTensor(joint_positions)
    return joint_positions
    
def load_data_visionlab3DPoses(absolute: Optional[bool] = False):
    """
        Load the data from the files of the AIS dataset.
        The data is automatically parsed to the stacked hourglass skeleton format.

        Arguments:
            absolute (bool): Whether to return the data in absolute coordinates or relative coordinates.
    """
    file_paths = (P(os.getcwd()) / VLP_DATASET_PATH).rglob('**/*.json')
    data = {}
    for path in file_paths:
        file_name = path.stem
        raw_data = read_file_visionlab3DPoses(path, return_tensor=True)
        proc_data = parse_ais3dposes_to_s16(raw_data, absolute)
        
        data[file_name] = proc_data

    return data

