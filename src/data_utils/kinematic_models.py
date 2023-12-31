"""
    This file contains a fully-differentiable abstract skeleton model.
    This can be used to implement different skeleton representations.
    Unfortunately it does not work batch-wsie. Thus, we chose to use our forward kinematics function instead.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
import torch.nn as nn
from typing import Literal, Optional
from abc import abstractmethod

import numpy as np

from .meta_info import *
from ..utils.transformations import *

#####===== Skeleton Models =====#####

class SkeletonBase(nn.Module):
    """
        This is the base module for a skeleton structure.

        The usage of this base required the implementation of the following functions:
            _init_skeleton(): Initializes the skeleton structure
            _load_data_from_h36m(): Loads the data from the H36M dataset and saves it to the appropriate variables
            _get_angle_vector(): Parses the angles back to a vector in form of the H36M dataset.
        
        This includes the definition of the following required variables:
            kinematic_chain: A dictionary holding the kinematic chain.
                Format: {
                            joint_number: (frame_name, parent_frame_name)
                        }
            joint buffers: Buffers to hold joint positions and angles.
                Naming convention:  Position: "{name}_pos"
                                    Angle: "{name}_angle" -> rotation matrix format
        
        Using this variables and functions this module implements the following:
         - forward kinematics
         - rotation representation conversions
         - Getter functions for angles and positions
    """

    def __init__(self, device = 'cpu', hip_as_root: Optional[bool]=True) -> None:
        super(SkeletonBase, self).__init__()

        self.kinematic_chain = None
        self.offset = None
        self.device = device
        self.hip_as_root = hip_as_root
        self._init_skeleton()
    
    ###=== Forward Kinematics ===###

    def forward(self, x: torch.Tensor, format: Optional[Literal['ds_angle']] = 'ds_angle'):
        """
            Forward function that takes the joint angles and applies forward kinematics.
        """
        if format == 'ds_angle':
            self._load_data_from_h36m(x)
            self.forward_kinematics(x)
        
    
    def forward_kinematics(self, x: torch.Tensor):
        """
            Uses the pre-defined kinematic chain and computes the transformation 
            from the hip joint to each other joint of the skeleton model.
            Positions and angles of each joint are overwritten.
        """
        
        for id, (cur_frame, par_frame) in self.kinematic_chain.items():

            if id==0 and self.hip_as_root:
                # Skip processing and setting position of hip frame
                continue
            
            frame_angle = getattr(self, f'{cur_frame}_angle')
            par_pos = getattr(self, f'{par_frame}_pos')
            par_angle = getattr(self, f'{par_frame}_angle')

            setattr(self,  f'{cur_frame}_pos',  self.offset[id] @ par_angle + par_pos)
            setattr(self, f'{cur_frame}_angle', frame_angle @ par_angle)
    
    ###=== Getter Functions ===###
        
    def getJointAngles(self, format: Optional[Literal['vector', 'dict']] = 'vector',
                                representation: Optional[Literal['axis', 'mat', 'quat']] = 'axis'):
        """
            Get the joint angles from the skeleton in different structure and formats.
            This function requires several abstract functions that require to be implemented.

            Arguments:
                format (['vector', 'dict']): format of the data structure containing the joint angles
                    vector: A 99-dimensional vector as in the dataset
                    dict: A dictionary containing the joint angles where the keys describe the joint names.
                representation (['axis', 'mat', 'quat']): Rotation representation
        """
        if format == 'vector':
            return self._
            

    def getJointPositions(self, incl_names = False):
        """
            Get the joint positions previously computed by the forward kinematics.

            Arguments:
                incl_names (bool, optional): whether to include the names of the joints in the output
            
            Return:
                positions:  If incl_names is True: dictionary of form {joint_name: joint_position}
                            Else: torch.Tensor of shape (num_joints, 3)
        """
        if incl_names:
            positions = {}
            positions['hip'] = self.hip_pos
        else:
            positions = []
            positions.append(self.hip_pos)
        for id, (cur_frame, par_frame) in self.kinematic_chain.items():
            poss = getattr(self, f'{cur_frame}_pos')
            if incl_names:
                positions[cur_frame] = poss
            else:
                positions.append(poss)
        if not incl_names:
            positions = torch.cat(positions, dim=0)
        return positions

    ###=== Helper Functions ===###

    def _get_angle_dictionary(self, representation: Optional[Literal['axis', 'mat', 'quat']] = 'axis'):
        """
            Function to build a dictionary that contains the joint angles in a given format.
        """
        angle_dict = {}
        conversion_func = self._get_from_mat_conversion(representation)
        for id, (cur_frame, par_frame) in self.kinematic_chain.items():
            angle = getattr(self, f'{cur_frame}_angle')
            angle = conversion_func(angle)
            angle_dict['cur_frame'] = angle
        return angle_dict

    def _get_from_mat_conversion(representation: Optional[Literal['axis', 'mat', 'quat']] = 'axis'):
        """
            Defines the appropriate conversion function from rotation matrix representation.
        """
        if representation == 'axis':
            return matrix_to_axis_angle
        elif representation == 'mat':
            return blank_processing
        elif representation == 'quat':
            return matrix_to_quaternion
        else:
            raise NotImplementedError()
    
    def _get_to_mat_conversion(representation: Optional[Literal['axis', 'mat', 'quat']] = 'axis'):
        """
            Defines the appropriate conversion to rotation matrix representation.
        """
        if representation == 'axis':
            return axis_angle_to_matrix_direct
        elif representation == 'mat':
            return blank_processing
        elif representation == 'quat':
            return quaternion_to_matrix
        else:
            raise NotImplementedError()

    ###=== Abstract Interface ===###    
    @abstractmethod
    def _init_skeleton(self):
        """
            Abstract function that initializes the skeleton structure.
            That includes initializing buffers for each joint position and angle
            and initializing the object variable kinematic_chain that defines the 
            order of computation for the forward kinematics.

            Please read the documentation of this class for further information.
        """
        raise NotImplementedError(self._init_skeleton)
    
    @abstractmethod
    def _load_data_from_h36m(self, data: torch.Tensor):
        """
            Abstract function to load data from the H36M dataset and write it into the appropriate buffers
        
        """
        raise NotImplementedError(self._load_data_from_h36m)
    
    @abstractmethod
    def _get_angle_vector(self, representation: Optional[Literal['axis', 'mat', 'quat']] = 'axis'):
        """
            Abstract function to parse the joint angles into a vector.
            The shape of the vector should be determined by the skeleton model.
        """
        raise NotImplementedError(self._get_angle_vector)

#####===== H36M Skeleton Model =====#####
# This is the original skeleton model used in the H36M dataset.
    
class SkeletonModel32(SkeletonBase):
    """
        Skeleton model used in the H36M dataset.
    """

    def __init__(self, device='cpu'):
        super(SkeletonModel32, self).__init__(device)
    

    def _init_skeleton(self):
        """
            Initialize a skeleton structure with 32 joints
            Buffers registered:
                For all active joints of the H36M dataset:
                    {name}_pos: position of the joint w.r.t either hip or root
                    {name}_angle: angle of the joint
                    {name}_rot: rotation matrix of the joint w.r.t either hip or root
        
        """
        for id, name in H36M_REDUCED_ANGLE_INDICES.items():
            self.register_buffer(f'{name}_pos', torch.zeros(3).to(self.device))
            self.register_buffer(f'{name}_angle', torch.zeros(3).to(self.device))
            self.register_buffer(f'{name}_rot', torch.zeros(3).to(self.device))


        self.kinematic_chain = H36M_SKELETON_STRUCTURE
        self.offset = torch.reshape(torch.FloatTensor(H36M_BONE_LENGTH), (-1,3))
    
    def _load_data_from_h36m(self, data: torch.Tensor):
        """
            Load the data from the 99-dimension input vector
        """
        self.hip_pos = self.offset[0] + data[[0, 1, 2]]# root
        self.hip_angle = axis_angle_to_matrix(data[               [3, 4, 5]])
        self.rHip_angle = axis_angle_to_matrix(data[              [6, 7, 8]])# right leg
        self.rKnee_angle = axis_angle_to_matrix(data[             [9, 10, 11]])
        self.rAnkle_angle = axis_angle_to_matrix(data[            [12, 13, 14]])
        self.rToe_angle = axis_angle_to_matrix(data[              [15, 16, 17]])
        self.lHip_angle = axis_angle_to_matrix(data[              [21, 22, 23]])# left leg
        self.lKnee_angle = axis_angle_to_matrix(data[             [24, 25, 26]])
        self.lAnkle_angle = axis_angle_to_matrix(data[            [27, 28, 29]])
        self.lToe_angle = axis_angle_to_matrix(data[              [30, 31, 32]])
        self.spine_angle = axis_angle_to_matrix(data[             [36, 37, 38]])# torso
        self.spine1_angle = axis_angle_to_matrix(data[            [39, 40, 41]])
        self.thorax_angle = axis_angle_to_matrix(data[            [42, 43, 44]])
        self.neck_angle = axis_angle_to_matrix(data[              [45, 46, 47]])
        self.lShoulderAnchor_angle = axis_angle_to_matrix(data[   [51, 52, 53]])# left arm
        self.lShoulder_angle = axis_angle_to_matrix(data[         [54, 55, 56]])
        self.lElbow_angle = axis_angle_to_matrix(data[            [57, 58, 59]])
        self.lWrist_angle = axis_angle_to_matrix(data[            [60, 61, 62]])
        self.lThumb_angle = axis_angle_to_matrix(data[            [63, 64, 65]])
        self.lWristEnd_angle = axis_angle_to_matrix(data[         [69, 70, 71]])
        self.rShoulderAnchor_angle = axis_angle_to_matrix(data[   [75, 76, 77]])# right arm
        self.rShoulder_angle = axis_angle_to_matrix(data[         [78, 79, 80]])
        self.rElbow_angle = axis_angle_to_matrix(data[            [81, 82, 83]])
        self.rWrist_angle = axis_angle_to_matrix(data[            [84, 85, 86]])
        self.rThumb_angle = axis_angle_to_matrix(data[            [87, 88, 89]])
        self.rWristEnd_angle = axis_angle_to_matrix(data[         [93, 94, 95]])
    
    def _get_angle_vector(self,  representation: Optional[Literal['axis', 'mat', 'quat']] = 'axis'):
        joint_angles = torch.zeros(99)
        conv_func = self._get_from_mat_conversion(representation)

        joint_angles[[0, 1, 2]] = self.hip_pos - self.offset[0]
        joint_angles[[3,4,5]] = conv_func(self.hip_angle)
        # right leg
        joint_angles[[6, 7, 8]] = conv_func(self.rHip_angle)
        joint_angles[[9, 10, 11]] = conv_func(self.rKnee_angle)
        joint_angles[[12, 13, 14]] = conv_func(self.rAnkle_angle)
        joint_angles[[15, 16, 17]] = conv_func(self.rToe_angle)

        # left leg
        joint_angles[[21, 22, 23]] = conv_func(self.lHip_angle)
        joint_angles[[24, 25, 26]] = conv_func(self.lKnee_angle)
        joint_angles[[27, 28, 29]] = conv_func(self.lAnkle_angle)
        joint_angles[[30, 31, 32]] = conv_func(self.lToe_angle)

        # torso
        joint_angles[[36, 37, 38]] = conv_func(self.spine_angle)
        joint_angles[[39, 40, 41]] = conv_func(self.spine1_angle)
        joint_angles[[42, 43, 44]] = conv_func(self.thorax_angle)
        joint_angles[[45, 46, 47]] = conv_func(self.neck_angle)

        # left arm
        joint_angles[[51, 52, 53]] = conv_func(self.lShoulderAnchor_angle)
        joint_angles[[54, 55, 56]] = conv_func(self.lShoulder_angle)
        joint_angles[[57, 58, 59]] = conv_func(self.lElbow_angle)
        joint_angles[[60, 61, 62]] = conv_func(self.lWrist_angle)
        joint_angles[[63, 64, 65]] = conv_func(self.lThumb_angle)
        joint_angles[[69, 70, 71]] = conv_func(self.lWristEnd_angle)

        # right arm
        joint_angles[[75, 76, 77]] = conv_func(self.rShoulderAnchor_angle)
        joint_angles[[78, 79, 80]] = conv_func(self.rShoulder_angle)
        joint_angles[[81, 82, 83]] = conv_func(self.rElbow_angle)
        joint_angles[[84, 85, 86]] = conv_func(self.rWrist_angle)
        joint_angles[[87, 88, 89]] = conv_func(self.rThumb_angle)
        joint_angles[[93, 94, 95]] = conv_func(self.rWristEnd_angle)

        return joint_angles