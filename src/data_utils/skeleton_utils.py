"""
    This file bundles all functions to easily convert between different skeleton representations.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""
import torch
from typing import Optional, Literal, List

from ..utils import get_conv_to_rotation_matrix, get_conv_from_rotation_matrix, print_
from .meta_info import (
    H36M_REDUCED_IND_TO_CHILD,
    H36M_REVERSED_REDUCED_ANGLE_INDICES,
    H36M_BONE_LENGTH,
    H36M_SKELETON_STRUCTURE,
    H36M_NON_REDUNDANT_INDICES,
    H36M_REVERSED_NON_REDUNDANT_ANGLE_INDICES,
    H36M_NON_REDUNDANT_SKELETON_STRUCTURE,
    SH_SKELETON_STRUCTURE,
    H36M_REDUCED_ANGLE_INDICES,
    SH_NAMES,
    VLP_NAMES,
    H36M_NON_REDUNDANT_PARENT_IDS

)
#####===== Helper Functions =====#####
def get_annotated_dictionary(data: torch.Tensor, skeleton: Literal['s26', 's21', 's16', 'vlp']) -> torch.Tensor:
    """
        Return an annotated dictionary for the given skeleton.
        This function asserts that the joint representation is squeezed to the last dimension.
    """
    result = []
    if skeleton =='s26':
        joint_names = H36M_REDUCED_ANGLE_INDICES.values()
        num_joints = 26
    elif skeleton =='s21':
        joint_names = H36M_NON_REDUNDANT_INDICES.values()
        num_joints = 21
    elif skeleton =='s16':
        joint_names = SH_NAMES
        num_joints = 16
    elif skeleton == 'vlp':
        joint_names = VLP_NAMES[:18]
        num_joints = 18
    else:
        raise ValueError(f'Unknown skeleton: {skeleton}')

    if data.shape[-2]!=num_joints:
        raise ValueError(f'Provided data does not have the right shape: {data.shape}')
    if len(data.shape)>3:
        data = torch.flatten(data, start_dim=0, end_dim=-3)
    if len(data.shape)==2:
        data = data.unsqueeze(0)
    for fid in range(data.shape[0]):
        frame = data[fid]
        frame_dir = {name: value  for name, value in zip(joint_names, frame)}
        result.append(frame_dir)
    if len(result)==1:
        result = result[0]
    return result

#####===== Parser Functions =====#####

def parse_h36m_to_s26(seq: torch.Tensor, conversion_func: Optional[callable] = None):
    """
        Parses the raw h36m sequence to the basic skeleton used within the dataset.
        Only the site joints that are not part of the skeleton are removed.
        The resulting skeleton has 26 joints, where some are still redundant.
    """
    seq_repr = torch.FloatTensor(seq.shape[0],26,3)
    seq_repr[...,0,:] =  seq[...,[3, 4, 5]]
    seq_repr[...,1,:] =  seq[...,[6, 7, 8]]
    seq_repr[...,2,:] =  seq[...,[9, 10, 11]]
    seq_repr[...,3,:] =  seq[...,[12, 13, 14]]
    seq_repr[...,4,:] =  seq[...,[15, 16, 17]]
    seq_repr[...,5,:] =  seq[...,[21, 22, 23]]
    seq_repr[...,6,:] =  seq[...,[24, 25, 26]]
    seq_repr[...,7,:] =  seq[...,[27, 28, 29]]
    seq_repr[...,8,:] =  seq[...,[30, 31, 32]]
    seq_repr[...,9,:] =  seq[...,[36, 37, 38]]
    seq_repr[...,10,:] = seq[...,[39, 40, 41]]
    seq_repr[...,11,:] = seq[...,[42, 43, 44]]
    seq_repr[...,12,:] = seq[...,[45, 46, 47]]
    seq_repr[...,13,:] = seq[...,[48, 49, 50]]
    seq_repr[...,14,:] = seq[...,[51, 52, 53]]
    seq_repr[...,15,:] = seq[...,[54, 55, 56]]
    seq_repr[...,16,:] = seq[...,[57, 58, 59]]
    seq_repr[...,17,:] = seq[...,[60, 61, 62]]
    seq_repr[...,18,:] = seq[...,[63, 64, 65]]
    seq_repr[...,19,:] = seq[...,[69, 70, 71]]
    seq_repr[...,20,:] = seq[...,[75, 76, 77]]
    seq_repr[...,21,:] = seq[...,[78, 79, 80]]
    seq_repr[...,22,:] = seq[...,[81, 82, 83]]
    seq_repr[...,23,:] = seq[...,[84, 85, 86]]
    seq_repr[...,24,:] = seq[...,[87, 88, 89]]
    seq_repr[...,25,:] = seq[...,[93, 94, 95]]
    if conversion_func is not None:
        seq_repr = torch.flatten(seq_repr, 0, 1)
        seq_repr = conversion_func(seq_repr)
        seq_repr = torch.reshape(seq_repr,(seq.shape[0], 26, -1))
    return seq_repr

def parse_ais3dposes_to_s16(seq: torch.Tensor, absolute: Optional[bool] = False) -> torch.Tensor:

    seq_repr = torch.FloatTensor(seq.shape[0],16,3)
    seq_repr[...,0,:] = seq[...,8,:]
    seq_repr[...,1,:] = seq[...,9,:]
    seq_repr[...,2,:] = seq[...,10,:]
    seq_repr[...,3,:] = seq[...,11,:]
    seq_repr[...,4,:] = seq[...,12,:]
    seq_repr[...,5,:] = seq[...,13,:]
    seq_repr[...,6,:] = seq[...,14,:]
    seq_repr[...,7,:] = seq[...,8,:] + (seq[...,8,:] - seq[...,1,:])/2
    seq_repr[...,8,:] = seq[...,1,:]
    seq_repr[...,9,:] = seq[...,18,:] + (seq[...,18,:] - seq[...,17,:])/2 + (seq[...,0,:] - seq[...,1,:])/2
    seq_repr[...,10,:] = seq[...,5,:]
    seq_repr[...,11,:] = seq[...,6,:]
    seq_repr[...,12,:] = seq[...,7,:]
    seq_repr[...,13,:] = seq[...,2,:]
    seq_repr[...,14,:] = seq[...,3,:]
    seq_repr[...,15,:] = seq[...,4,:]

    if not absolute:
        seq_repr -= seq_repr[...,0,:]
    
    return seq_repr


#####===== Conversion Functions =====#####

def convert_s26_to_s21(seq: torch.Tensor, 
                        conversion_func: Optional[callable] = None,
                          interpolate: Optional[bool] = False,
                           rot_representation: Optional[Literal['axis', 'mat', 'quat', '6d', 'pos', None]] = None) -> torch.Tensor:
    """
        This functions takes the full skeleton of the h36m and removes the following redundant joints:
            - lShoulderAnchor / rShoulderAnchor (thorax)
            - lThumb / rThumb (lWrist / rWrist)
            - spine (hip)
        This function is specifically designed for the H36M dataset.
    """
    ind_to_cut = [9,14,18,20,24]
    if not interpolate: 
        # If we have absolute angles wrt a fixed anchor frame, we can simply remove joints as we like
        seq = _remove_joints(seq, ind_to_cut)
    else:
        # If we use relative angles wrt to the parent joint, we have to interpolate across removed joints.
        if rot_representation is None:
            print_("Please provide a valid rotation/position representation for interpolation", "warn")
            return seq
        # If we work with angles get conversion functions to rotation matrix
        if rot_representation != 'pos':
            to_rotmat = get_conv_to_rotation_matrix(rot_representation)
            to_org_rep = get_conv_from_rotation_matrix(rot_representation)
            pos_rep = False
        else:
            pos_rep = True
        for i, ind in enumerate(ind_to_cut):
            # Adjust relative rotations
            if not pos_rep:
                rot_mat_cut = to_rotmat(seq[...,(ind-i),:])
                
                # Iterate through all children joints of the joint to be removed
                for child_ind in H36M_REDUCED_IND_TO_CHILD[ind]:
                    # Query the rotation to get the rotation of the children joint wrt to its new parent
                    rot_mat_child = to_rotmat(seq[..., (child_ind-i),:])
                    # We have to compute the shoulder anchors differently since we want the parent frame to be thorax instead of spine1
                    if ind in [14,20]:
                        rot_mat_thorax = to_rotmat(seq[...,(11-i),:])
                        seq[...,(child_ind-i),:] = to_org_rep(torch.bmm(rot_mat_child,rot_mat_thorax))
                        continue
                    seq[...,(child_ind-i),:] = to_org_rep(torch.bmm(rot_mat_child,rot_mat_cut))
            # Adjust relative positions
            else:
                pos_cut = seq[...,(ind-i),:]
                # Iterate through all children joints of the joint to be removed
                for child_ind in H36M_REDUCED_IND_TO_CHILD[ind]:
                    # Query the position to get the position of the children joint wrt to its new parent
                    pos_child = seq[..., (child_ind-i),:]
                    seq[...,(child_ind-i),:] = pos_cut + pos_child
            seq = torch.cat([seq[...,0:(ind-i),:], seq[...,(ind-i)+1:,:]], dim=1)
    return _transform_representation(seq, conversion_func)


def convert_s21_to_s26(seq: torch.Tensor, conversion_func: Optional[callable] = None) -> torch.Tensor:
    """
        This function converts the reduced s21 skeleton back to the full skeleton of the h36m dataset.
    """
    seq_repr = torch.FloatTensor(seq.shape[0],26,3)
    seq_repr[...,0] = seq[...,0]
    seq_repr[...,1] = seq[...,1]
    seq_repr[...,2] = seq[...,2]
    seq_repr[...,3] = seq[...,3]
    seq_repr[...,4] = seq[...,4]
    seq_repr[...,5] = seq[...,5]
    seq_repr[...,6] = seq[...,6]
    seq_repr[...,7] = seq[...,7]
    seq_repr[...,8] = seq[...,8]
    seq_repr[...,9] = seq[...,0]
    seq_repr[...,10] = seq[...,9]
    seq_repr[...,11] = seq[...,10]
    seq_repr[...,12] = seq[...,11]
    seq_repr[...,13] = seq[...,12]
    seq_repr[...,14] = seq[...,10]
    seq_repr[...,15] = seq[...,13]
    seq_repr[...,16] = seq[...,14]
    seq_repr[...,17] = seq[...,15]
    seq_repr[...,18] = seq[...,15]
    seq_repr[...,19] = seq[...,16]
    seq_repr[...,20] = seq[...,10]
    seq_repr[...,21] = seq[...,17]
    seq_repr[...,22] = seq[...,18]
    seq_repr[...,23] = seq[...,19]
    seq_repr[...,24] = seq[...,19]
    seq_repr[...,25] = seq[...,20]
    return _transform_representation(seq_repr, conversion_func)

def convert_s21_to_s16(seq: torch.Tensor, 
                        conversion_func: Optional[callable] = None,
                          interpolate: Optional[bool] = False,
                           rot_representation: Optional[Literal['axis', 'mat', 'quat', '6d', 'pos', None]] = None):
    """
        Converts the reduced s21 skeleton to the stacked hourglass skeleton representation.
        This simply removes the last joints of each limb and the neck joint.
    """
    ind_to_cut = [4,8,11,16,20]
    if not interpolate:
        seq = _remove_joints(seq, ind_to_cut)
    else:
        # If we use relative angles wrt to the parent joint, we have to interpolate across removed joints.
        if rot_representation is None:
            print_("Please provide a valid rotation/position representation for interpolation", "warn")
            return seq
        # If we work with angles get conversion functions to rotation matrix
        if rot_representation != 'pos':
            to_rotmat = get_conv_to_rotation_matrix(rot_representation)
            to_org_rep = get_conv_from_rotation_matrix(rot_representation)
            pos_rep = False
        else:
            pos_rep = True
        for i, ind in enumerate(ind_to_cut):
            # Only the neck is not at the end of a kinematic chain and needs to be considered separate
            if ind == 11:
                if not pos_rep:
                    rot_mat_cut = to_rotmat(seq[...,(ind-i),:])
                    rot_mat_child = to_rotmat(seq[...,(12-i),:])
                    seq[...,(12-i),:] = to_org_rep(torch.matmul(rot_mat_child,rot_mat_cut))
                else:
                    pos_cut = seq[...,(ind-i),:]
                    pos_child = seq[...,(12-i),:]
                    seq[...,(12-i),:] = pos_cut + pos_child

            seq = torch.cat([seq[...,0:(ind-i),:], seq[...,(ind-i)+1:,:]], dim=1)
    return _transform_representation(seq, conversion_func)

def convert_s26_to_s16(seq: torch.Tensor, 
                        conversion_func: Optional[callable] = None,
                          relative: Optional[bool] = False,
                           rot_representation: Optional[Literal['axis', 'mat', 'quat', '6d', 'pos', None]] = None) -> torch.Tensor:
    """
        Converts the full H36M dataset to the stacked hourglass skeleton representation.
        This removes viable information from the skeleton and cannot be reversed.
        It is advised to only use this conversion with absolute angles and not relative angles between joints
        since joints are removed in the kinematic chain and relative angles do not make sense anymore.
        This effectively removes the following joints (equally for left and right):
            neck, ShoulderAnchor, Thumb, WristEnd, spine, Toe
    """
    s21_repr = convert_s26_to_s21(seq, None, relative, rot_representation)
    s16_repr = convert_s21_to_s16(s21_repr, conversion_func, relative, rot_representation)
    return s16_repr



#####===== Kinematic Functions =====#####

def h36m_forward_kinematics(data: torch.Tensor, 
                             representation: Literal['axis', 'mat', 'quat', '6d'], 
                              hip_as_root: Optional[bool] = True,
                               hip_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
        Compute the forward kinematics of the h36m skeleton model from a given representation.
        The functions returns the joint positions and rotations
    """
    # conversion function to convert to rotation matrices
    conversion_func = get_conv_to_rotation_matrix(representation)
    if data.shape[-1] == 99:
        # Data is in original dataset format and needs to be transformed first
        shape = list(data.shape[:-1]) + [26]
        if len(data.shape) > 2:
            data = torch.flatten(data, start_dim=0, end_dim=-2)
        hip_pos = data[:,[0,1,2]]
        data = parse_h36m_to_s26(data, conversion_func)
        data = torch.reshape(data, (*data.shape[:-1], 3, 3))
    else:
        # Data is already represented thorugh single joints
        data = conversion_func(data)
        shape = data.shape[:-2]
        if len(data.shape) > 4:
            data = torch.flatten(data, start_dim=0, end_dim=-4)
    if not hip_as_root and hip_pos is None:
        print_("Please provide a hip position if it is not the root frame.", 'error')
        return
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
            if not hip_as_root:
                joint_positions[:,joint_id] = offset[i].unsqueeze(0)
                joint_positions[:,joint_id] += hip_pos
                joint_rotations[:,joint_id] = data[:, joint_id]
            else:
                joint_rotations[:,joint_id] = torch.eye(3).unsqueeze(0)
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

#####===== Utility Functions =====#####
def _remove_joints(seq: torch.Tensor, inds: List[int]):
    """
        Removes the by the indices given joints from the sequence.
    """
    for i, ind in enumerate(inds):
        seq = torch.cat([seq[...,0:(ind-i),:], seq[...,(ind-i)+1:,:]], dim=-2)
    return seq

def _transform_representation(seq: torch.Tensor, conversion_func: Optional[callable] = None) -> torch.Tensor:
    """
        converts the joint representation
    """
    if conversion_func is not None:
        seq = torch.flatten(seq, 0, 1)
        seq = conversion_func(seq)
        seq = torch.reshape(seq,(seq.shape[0], 21, -1))
    return seq

def _compute_s21_bone_length() -> torch.Tensor:
    """
        Computes the bone length of the reduced skeleton based on the original bone lengths from the H36M dataset.
    """
    bone_length = torch.reshape(torch.FloatTensor(H36M_BONE_LENGTH), (-1,3))
    bone_length = _cutout_sites(bone_length)
    ind_to_cut = [9,14,18,20,24]
    for i, ind in enumerate(ind_to_cut):
        for child_ind in H36M_REDUCED_IND_TO_CHILD[ind]:
            child_bone_length = bone_length[child_ind]
            bone_length[(child_ind-i)] = child_bone_length + bone_length[(ind-i)]
        bone_length = torch.cat([bone_length[:(ind-i)], bone_length[(ind-i)+1:]])
    return bone_length

def _compute_s16_bone_length() -> torch.Tensor:
    """
        Computes the bone length of the reduced skeleton based on the original bone lengths from the H36M dataset.
    """
    bone_length = torch.reshape(torch.FloatTensor(H36M_BONE_LENGTH), (-1,3))
    bone_length = _cutout_sites(bone_length)
    ind_to_cut = [4,8,9,12,14,18,19,20,24,25]
    for i, ind in enumerate(ind_to_cut):
        for child_ind in H36M_REDUCED_IND_TO_CHILD[ind]:
            child_bone_length = bone_length[child_ind]
            bone_length[(child_ind-i)] = child_bone_length + bone_length[(ind-i)]
        bone_length = torch.cat([bone_length[:(ind-i)], bone_length[(ind-i)+1:]])
    return bone_length

def _cutout_sites(input: torch.Tensor) -> torch.Tensor:
    """
        Cuts out the site joints from the h36m dataset in some given data.
    """
    ind_to_cut = [5,10,21,23,29,31]
    for i, ind in enumerate(ind_to_cut):
        input = torch.cat([input[...,0:(ind-i),:], input[...,(ind-i)+1:,:]], dim=-2)
    return input
