"""
    This file bundles all functions to easily convert between different skeleton representations.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""
import torch
from typing import Optional, Literal, List
import numpy as np
from ..utils import get_conv_to_rotation_matrix, get_conv_from_rotation_matrix, print_, vectors_to_rotation_matrix, correct_rotation_matrix
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
    H36M_NAMES,
    BASELINE_FKL_IND,
    H36M_BASELINE_PARENTS,
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
    """
        Parse the AIS dataset to the 16 joint skeleton model.
        The belly joint is set to the middle between the thorax and the hip.
        The head is the middle between the left and right ear.
    """
    seq_repr = torch.FloatTensor(seq.shape[0],16,3)
    seq_repr[...,0,:] = seq[...,8,:]
    seq_repr[...,1,:] = seq[...,9,:]
    seq_repr[...,2,:] = seq[...,10,:]
    seq_repr[...,3,:] = seq[...,11,:]
    seq_repr[...,4,:] = seq[...,12,:]
    seq_repr[...,5,:] = seq[...,13,:]
    seq_repr[...,6,:] = seq[...,14,:]
    seq_repr[...,7,:] = seq[...,8,:] + (seq[...,1,:]-seq[...,8,:])/2
    seq_repr[...,8,:] = seq[...,1,:]
    seq_repr[...,9,:] = seq[...,18,:] + (seq[...,18,:] - seq[...,17,:])/2 
    seq_repr[...,10,:] = seq[...,5,:]
    seq_repr[...,11,:] = seq[...,6,:]
    seq_repr[...,12,:] = seq[...,7,:]
    seq_repr[...,13,:] = seq[...,2,:]
    seq_repr[...,14,:] = seq[...,3,:]
    seq_repr[...,15,:] = seq[...,4,:]
    if not absolute:
        seq_repr -= torch.repeat_interleave(seq_repr[...,0,:].unsqueeze(1), 16, dim=1)
    #seq_repr = seq_repr[...,[0,2,1]]
    return seq_repr


#####===== Conversion Functions =====#####
# Please note that a proper conversion is only possible for positions or absolute rotations
# Simply cutting out relative joint rotations produces an artifact of a skeleton that is not interpretable on its own.

def convert_s26_to_s21(seq: torch.Tensor,
                       conversion_func: Optional[callable] = None,) -> torch.Tensor:
    """
        This functions takes the full skeleton of the h36m and removes the following redundant joints:
            - lShoulderAnchor / rShoulderAnchor (thorax)
            - lThumb / rThumb (lWrist / rWrist)
            - spine (hip)
        This function is specifically designed for the H36M dataset.

        Arguments:
            seq (torch.Tensor): A sequence of joint positions
            conversion_func (callable, optional): A function that transforms the representation of the sequence. Default: Do not perform transformation.
    """
    ind_to_cut = [9,14,18,20,24]
    seq = _remove_joints(seq, ind_to_cut)
    return _transform_representation(seq, conversion_func)



def convert_s21_to_s26(seq: torch.Tensor, conversion_func: Optional[callable] = None) -> torch.Tensor:
    """
        This function converts the reduced s21 skeleton back to the full skeleton of the h36m dataset.

        Arguments:
            seq (torch.Tensor): A sequence of joint configurations
            conversion_func (callable, optional): A function that transforms the representation of the sequence. Default: Do not perform transformation.
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
                        conversion_func: Optional[callable] = None) -> torch.Tensor:
    """
        Converts the reduced s21 skeleton to the stacked hourglass skeleton representation.
        This simply removes the last joints of each limb and the neck joint.

        Arguments:
            seq (torch.Tensor): A sequence of joint configurations
            conversion_func (callable, optional): A function that transforms the representation of the sequence. Default: Do not perform transformation.
    """
    ind_to_cut = [4,8,11,16,20]
    seq = _remove_joints(seq, ind_to_cut)
    return _transform_representation(seq, conversion_func)

def convert_s26_to_s16(seq: torch.Tensor, 
                        conversion_func: Optional[callable] = None) -> torch.Tensor:
    """
        Converts the full H36M dataset to the stacked hourglass skeleton representation.
        This removes viable information from the skeleton and cannot be reversed.
        It is advised to only use this conversion with absolute angles and not relative angles between joints
        since joints are removed in the kinematic chain and relative angles do not make sense anymore.
        This effectively removes the following joints (equally for left and right):
            neck, ShoulderAnchor, Thumb, WristEnd, spine, Toe

        Arguments:
            seq (torch.Tensor): A sequence of joint configurations
            conversion_func (callable, optional): A function that transforms the representation of the sequence. Default: Do not perform transformation.
    """
    s21_repr = convert_s26_to_s21(seq)
    s16_repr = convert_s21_to_s16(s21_repr, conversion_func)
    return s16_repr



#####===== Kinematic Functions =====#####

def h36m_forward_kinematics(data: torch.Tensor, 
                             representation: Literal['axis', 'mat', 'quat', '6d'], 
                              hip_as_root: Optional[bool] = True,
                               hip_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
        Compute the forward kinematics of the h36m skeleton model from a given representation.
        The functions returns the joint positions and rotations.

        Arguments:
            data (torch.Tensor): A sequence of joint configurations
            representation (Literal['axis','mat', 'quat', '6d']): The representation of the joints.
            hip_as_root (bool, optional): If True, the hip joint is at the origin. Default: True.
            hip_pos (torch.Tensor, optional): The position of the hip joint required if the hip joint is not the root frame.
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

###--- Kinematic Module from Motion Mixer ---###
# These modules were taken from: https://github.com/MotionMLP/MotionMixer/tree/main
# They are mainly used to test our own kinematic module

def convert_baseline_representation(xyz_struct):
    positions = {}
    for i, joint in enumerate(xyz_struct):
        joint_name = H36M_NAMES[i]
        positions[joint_name] = torch.from_numpy(joint)
    return positions


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x);
    return R

def baseline_forward_kinematics(angles, parent = H36M_BASELINE_PARENTS, angle_indices = BASELINE_FKL_IND, offset = H36M_BONE_LENGTH):
    """
        Convert joint angles and bone lenghts into the 3d points of a person.

        adapted from
        https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L14

        which originaly based on expmap2xyz.m, available at
        https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m
        Args
        angles: 99-long vector with 3d position and 3d joint angles in expmap format
        parent: 32-long vector with parent-child relationships in the kinematic tree
        offset: 96-long vector with bone lenghts
        rotInd: 32-long list with indices into angles
        expmapInd: 32-long list with indices into expmap angles
        Returns
        xyz: 32x3 3d points that represent a person in 3d space
    """

    assert len(angles) == 99
    # Structure that indicates parents for each joint
    njoints = 32
    xyzStruct = [dict() for x in range(njoints)]
    offset = np.reshape(offset, (-1, 3))

    for i in np.arange(njoints):

        if i == 0:
            xangle = angles[0]
            yangle = angles[1]
            zangle = angles[2]
            thisPosition = np.array([xangle, yangle, zangle])
        else:
            thisPosition = np.array([0, 0, 0])

        r = angles[angle_indices[i]]

        thisRotation = expmap2rotmat(r)

        if parent[i] == -1:  # Root node
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz'] = np.reshape(offset[i, :], (1, 3)) + thisPosition
        else:
            xyzStruct[i]['xyz'] = (offset[i, :] + thisPosition).dot(xyzStruct[parent[i]]['rotation']) + \
                                  xyzStruct[parent[i]]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot(xyzStruct[parent[i]]['rotation'])
    xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()

    return xyz, xyzStruct   

#####===== Utility Functions =====#####
def _remove_joints(seq: torch.Tensor, inds: List[int]):
    """
        Removes the by the indices given joints from the sequence.

        Arguments:
            seq (torch.Tensor): the sequence to remove the joints from
            inds (List[int]): the indices of the joints to remove
    """
    for i, ind in enumerate(inds):
        seq = torch.cat([seq[...,0:(ind-i),:], seq[...,(ind-i)+1:,:]], dim=-2)
    return seq

def _transform_representation(seq: torch.Tensor, conversion_func: Optional[callable] = None) -> torch.Tensor:
    """
        converts the joint representation

        Arguments:
            seq (torch.Tensor): the sequence to transform
            conversion_func (callable): a function that takes in a sequence and returns a transformed sequence
    """
    if conversion_func is not None:
        seq = torch.flatten(seq, 0, 1)
        seq = conversion_func(seq)
        seq = torch.reshape(seq,(seq.shape[0], 21, -1))
    return seq

def smooth_sequence(seq: torch.Tensor, sigma: float = 1.0, num: int = 5) -> torch.Tensor:
    """
        Smooth a given sequence using an exponential moving average.
    """
    k = sigma / (num+1)
    for i in range(seq.shape[0]):
        if i == 0:
            continue
        seq[i] = seq[i]*k + (1-k)*seq[i-1]

    return seq

def normalize_sequence_orientation(seq: torch.Tensor) -> torch.Tensor:
    """
        Normalize the orientation of a given sequence in stacked hourglass position format.
    """
    orig_shape = seq.shape
    if len(seq.shape) > 3:
        seq = torch.flatten(seq, start_dim=0, end_dim=-3)
    # Compute the hip vector -> new y axis
    hip_vector = seq[:,4] - seq[:,1] # right hip to left hip
    hip_vector = hip_vector / (torch.norm(hip_vector, dim=-1).unsqueeze(-1)+torch.finfo(torch.float32).eps) # normalize hip vector
    # Compute the spine vector -> new z axis
    spine_vector = seq[:,8] - seq[:,0] # hip to thorax
    spine_vector = spine_vector / torch.norm(spine_vector, dim=-1).unsqueeze(-1) # normalize spine vector
    # Orthogonalize the spine vector with respect to the hip vector
    dot_spine_hip = torch.einsum('bi,bi->b', spine_vector, hip_vector) 
    spine_vector_proj = spine_vector -  dot_spine_hip.unsqueeze(-1) * hip_vector# normalize spine
    spine_vector_proj = spine_vector_proj / (torch.norm(spine_vector_proj, dim=-1).unsqueeze(-1)+torch.finfo(torch.float32).eps)
    # Compute the orthogonal vector facing forward -> new x axis
    ortho_vector = torch.cross(hip_vector, spine_vector, dim=-1)
    ortho_vector = ortho_vector / (torch.norm(ortho_vector, dim=-1).unsqueeze(-1)+torch.finfo(torch.float32).eps)
    # Define the rotation matrix to convert to new basis vectors
    rotation_matrix = torch.zeros(seq.shape[0], 3, 3)
    rotation_matrix[:,:,0] = ortho_vector
    rotation_matrix[:,:,1] = hip_vector
    rotation_matrix[:,:,2] = spine_vector_proj
    rotation_matrix = correct_rotation_matrix(rotation_matrix)
    # Convert the complete sequence according to the rotation matrix
    seq = torch.einsum('bij,bnj->bni', torch.transpose(rotation_matrix, -2, -1), seq)
    seq = torch.reshape(seq, orig_shape)

    return seq

