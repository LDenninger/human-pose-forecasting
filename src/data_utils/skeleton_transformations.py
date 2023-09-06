import torch
from typing import Literal, Optional

from .meta_info import *
from ..utils.transformations import get_conv_from_rotation_matrix, get_conv_from_vectors, get_conv_to_rotation_matrix

#####===== H36M Skeleton Transformations =====#####
# This functions are used to parse the raw data to the skeleton representation.
# Each joint is additionally transformed to the dictated rotation representation

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
    conversion_func = get_conv_from_rotation_matrix(representation)
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

def parse_sequence_efficient_to_s26(seq: torch.Tensor, conversion_func: callable, absolute: Optional[bool] = False) -> torch.Tensor:
    """
        Parse a single datum of joint angles in the H36M dataset format to a skeleton representaion.

        Arguments:
            frame (torch.Tensor): Joint angles in the H36M dataset format
            skeleton_structure (dict): Skeleton structure to parse to
            indices (dict): Indices of joints in the provided datum
            conversion_func (callable): Function to convert from axis angle representation to the given representation

    """
    inds = torch.arange(27)
    seq_repr = torch.FloatTensor(seq.shape[0],(27 if absolute else 26),3)

    if absolute:
        joints = 27

    seq_repr[:, 1] = seq[:, [3, 4, 5]]
    seq_repr[:, 2] = seq[:, [6, 7, 8]]
    seq_repr[:, 3] = seq[:, [9, 10, 11]]
    seq_repr[:, 4] = seq[:, [12, 13, 14]]
    seq_repr[:, 5] = seq[:, [15, 16, 17]]
    seq_repr[:, 6] = seq[:, [21, 22, 23]]
    seq_repr[:, 7] = seq[:, [24, 25, 26]]
    seq_repr[:, 8] = seq[:, [27, 28, 29]]
    seq_repr[:, 9] = seq[:, [30, 31, 32]]
    seq_repr[:, 10] = seq[:, [36, 37, 38]]
    seq_repr[:, 11] = seq[:, [39, 40, 41]]
    seq_repr[:, 12] = seq[:, [42, 43, 44]]
    seq_repr[:, 13] = seq[:, [45, 46, 47]]
    seq_repr[:, 14] = seq[:, [48, 49, 50]]
    seq_repr[:, 15] = seq[:, [51, 52, 53]]
    seq_repr[:, 16] = seq[:, [54, 55, 56]]
    seq_repr[:, 17] = seq[:, [57, 58, 59]]
    seq_repr[:, 18] = seq[:, [60, 61, 62]]
    seq_repr[:, 19] = seq[:, [63, 64, 65]]
    seq_repr[:, 20] = seq[:, [69, 70, 71]]
    seq_repr[:, 21] = seq[:, [75, 76, 77]]
    seq_repr[:, 22] = seq[:, [78, 79, 80]]
    seq_repr[:, 23] = seq[:, [81, 82, 83]]
    seq_repr[:, 24] = seq[:, [84, 85, 86]]
    seq_repr[:, 25] = seq[:, [87, 88, 89]]
    seq_repr = torch.flatten(seq_repr, 0, 1)
    seq_repr = conversion_func(seq_repr)
    seq_repr = torch.reshape(seq_repr,(seq.shape[0], 26, -1))
    return seq_repr

def h36m_forward_kinematics(data: torch.Tensor, representation: Literal['axis', 'mat', 'quat', '6d'], absolute: Optional[bool] = False) -> torch.Tensor:
    """
        Compute the forward kinematics of the h36m skeleton model from a given representation.
        The functions returns the joint positions
    """
    # conversion function to convert to rotation matrices
    conversion_func = get_conv_to_rotation_matrix(representation)
    if data.shape[-1] == 99:
        # Data is in original dataset format and needs to be transformed first
        shape = list(data.shape[:-1]) + [26]
        if len(data.shape) > 2:
            data = torch.flatten(data, start_dim=0, end_dim=-2)
        data = parse_sequence_efficient_to_s26(data, conversion_func)
        data = torch.reshape(data, (*data.shape[:-1], 3, 3))
    else:
        # Data is already represented thorugh single joints
        data = conversion_func(data)
        shape = data.shape[:-2]
        if len(data.shape) > 4:
            data = torch.flatten(data, start_dim=0, end_dim=-3) 
    # Set variables need for the forward kinematics
    name_to_ind = H36M_REVERSED_REDUCED_ANGLE_INDICES
    offset = torch.reshape(torch.FloatTensor(H36M_BONE_LENGTH), (-1,3))
    parent_position = torch.zeros(data.shape[0], 3)
    # output tensors
    joint_positions = torch.zeros(data.shape[0], 26, 3)
    joint_rotations = torch.zeros(data.shape[0], 26, 3, 3)
    # Iterate through the kinematic chain
    for joint_id, (i, (cur_frame, par_frame)) in enumerate(H36M_SKELETON_STRUCTURE.items()):
        if joint_id == 0:
            # Set the position and rotation of the base frame, the hip
            joint_positions[:,joint_id] = offset[i].unsqueeze(0)
            if absolute:
                joint_positions[:,joint_id] += data[:, [0,1,2]]
            joint_rotations[:,joint_id] = data[:, joint_id]
            continue
        # Retrieve data from the previously processed parent frame
        parent_rotation = joint_rotations[:,name_to_ind[par_frame]]
        parent_position = joint_positions[:,name_to_ind[par_frame]]
        # Compute the position and rotation of the joint
        joint_positions[:,joint_id] = (offset[i].unsqueeze(0).unsqueeze(0) @ parent_rotation).squeeze() + parent_position
        joint_rotations[:,joint_id] = data[:, joint_id] @ parent_rotation
    # Reshpae the output to the original shape
    joint_positions = torch.reshape(joint_positions, (*shape, 3))
    joint_rotations = torch.reshape(joint_rotations, (*shape, 3, 3))
    return joint_positions, joint_rotations

def get_annotated_dictionary(data: torch.Tensor, skeleton: Literal['s26']) -> torch.Tensor:
    """
        Return an annotated dictionary for the given skeleton.
        This function asserts that the joint representation is squeezed to the last dimension.
    """
    result = []
    if skeleton =='s26':
        if data.shape[-2]!=26:
            raise ValueError(f'Provided data does not have the right shape: {data.shape}')
        if len(data.shape)>3:
            data = torch.flatten(data, start_dim=0, end_dim=-3)
        if len(data.shape)==2:
            data = data.unsqueeze(0)
        for fid in range(data.shape[0]):
            frame = data[fid]
            frame_dir = {name: value  for name, value in zip(H36M_REDUCED_ANGLE_INDICES.values(), frame)}
            result.append(frame_dir)
        if len(result)==1:
            result = result[0]
        return result
    


#####===== VisionLab3DPose Skeleton Transformations =====#####

def parse_sequence_to_s19(seq: torch.Tensor, conversion_func: callable) -> torch.Tensor:
    """
        Parse the raw data from the VisionLab3DPose dataset to the given representation
    """

    seq = seq[:,:19]
    parent_ids = VLP_PARENTS
    parent_ids[0] = 0
    parent_seq = seq[:, parent_ids]
    seq = conversion_func(seq, parent_seq)
    return seq

def 