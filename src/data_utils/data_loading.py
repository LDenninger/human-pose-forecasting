import numpy as np
import torch
import os
from pathlib import Path as P
from .meta_info import *
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

def load_data(person_id: list = None, action_str: list = None, sub_action_id: list = None, return_tensor=False, show_progress=True, return_reverse_meta=False):
    """
        Load the data from the Human3.6M dataset.

        The data is stored in the following format:
        [                                           # List of all action sequence
            [                                       # List of joint angles at each timestep
                np.array(99) / torch.Tensor(99)     # Joint angles either as numpy arrays or torch tensors
            ],...
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
            return_tensor (bool): Whether to return the data as torch tensors or numpy arrays
        Returns:
            data (list): Data in above list format
            meta_dir (list): List of meta information about each action sequence
    """

    if person_id is None:
        person_id = DATASET_PERSONS
    if action_str is None:
        action_str = DATASET_ACTIONS
    if sub_action_id is None:
        sub_action_id = [1,2]

    meta_dir = []
    reverse_meta_dir = {}
    data = []
    print('Loading Human3.6M data...')
    if show_progress:
        progress_bar = tqdm(total=len(person_id) * len(action_str) * len(sub_action_id))
    seq_index = 0
    for p_id in person_id:
        for a_str in action_str:
            for s_id in sub_action_id:
                new_data = read_file(p_id, a_str, s_id, return_tensor)
                len_data = len(new_data)
                meta_info = {
                    'person_id': p_id,
                    'action_str': a_str,
                    'sub_action_id': s_id,
                    'length': len_data
                }
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

