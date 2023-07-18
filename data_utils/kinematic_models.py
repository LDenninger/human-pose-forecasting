import torch
import torch.nn as nn
import pytorch3d as p3d
from pytorch3d import transforms as p3dTransforms

import numpy as np

from .meta_info import SKELETON_H36M_MODEL, BASELINE_FKL_IND, SKELETON_H36M_BONE_LENGTH, SKELETON_H36M_PARENT_IDS, H36M_NAMES

class SkeletonModel32(nn.Module):

    def __init__(self, device='cpu'):
        super(SkeletonModel32, self).__init__()
        self.device = torch.device(device)


        self.register_buffer('root_pos', torch.zeros(3).to(self.device))
        self.register_buffer('hip_pos', torch.zeros(3).to(self.device))
        self.register_buffer('hip_angle', torch.eye(3).to(self.device))
        self.register_buffer('rHip_pos', torch.zeros(3).to(self.device))
        self.register_buffer('rHip_angle', torch.eye(3).to(self.device))
        self.register_buffer('lHip_pos', torch.zeros(3).to(self.device))
        self.register_buffer('lHip_angle', torch.eye(3).to(self.device))
        self.register_buffer('rKnee_pos', torch.zeros(3).to(self.device))
        self.register_buffer('rKnee_angle', torch.eye(3).to(self.device))
        self.register_buffer('lKnee_pos', torch.zeros(3).to(self.device))
        self.register_buffer('lKnee_angle', torch.eye(3).to(self.device))
        self.register_buffer('rAnkle_pos', torch.zeros(3).to(self.device))
        self.register_buffer('rAnkle_angle', torch.eye(3).to(self.device))
        self.register_buffer('lAnkle_pos', torch.zeros(3).to(self.device))
        self.register_buffer('lAnkle_angle', torch.eye(3).to(self.device))
        self.register_buffer('rToe_pos', torch.zeros(3).to(self.device))
        self.register_buffer('rToe_angle', torch.eye(3).to(self.device))
        self.register_buffer('lToe_pos', torch.zeros(3).to(self.device))
        self.register_buffer('lToe_angle', torch.eye(3).to(self.device))
        self.register_buffer('rShoulderAnchor_pos', torch.zeros(3).to(self.device))
        self.register_buffer('rShoulderAnchor_angle', torch.eye(3).to(self.device))
        self.register_buffer('lShoulderAnchor_pos', torch.zeros(3).to(self.device))
        self.register_buffer('lShoulderAnchor_angle', torch.eye(3).to(self.device))
        self.register_buffer('rShoulder_pos', torch.zeros(3).to(self.device))
        self.register_buffer('rShoulder_angle', torch.eye(3).to(self.device))
        self.register_buffer('lShoulder_pos', torch.zeros(3).to(self.device))
        self.register_buffer('lShoulder_angle', torch.eye(3).to(self.device))
        self.register_buffer('rElbow_pos', torch.zeros(3).to(self.device))
        self.register_buffer('rElbow_angle', torch.eye(3).to(self.device))
        self.register_buffer('lElbow_pos', torch.zeros(3).to(self.device))
        self.register_buffer('lElbow_angle', torch.eye(3).to(self.device))
        self.register_buffer('rWrist_pos', torch.zeros(3).to(self.device))
        self.register_buffer('rWrist_angle', torch.eye(3).to(self.device))
        self.register_buffer('lWrist_pos', torch.zeros(3).to(self.device))
        self.register_buffer('lWrist_angle', torch.eye(3).to(self.device))
        self.register_buffer('rThumb_pos', torch.zeros(3).to(self.device))
        self.register_buffer('rThumb_angle', torch.eye(3).to(self.device))
        self.register_buffer('lThumb_pos', torch.zeros(3).to(self.device))
        self.register_buffer('lThumb_angle', torch.eye(3).to(self.device))
        self.register_buffer('rWristEnd_pos', torch.zeros(3).to(self.device))
        self.register_buffer('rWristEnd_angle', torch.eye(3).to(self.device))
        self.register_buffer('lWristEnd_pos', torch.zeros(3).to(self.device))
        self.register_buffer('lWristEnd_angle', torch.eye(3).to(self.device))
        self.register_buffer('neck_pos', torch.zeros(3).to(self.device))
        self.register_buffer('neck_angle', torch.eye(3).to(self.device))
        self.register_buffer('spine_pos', torch.zeros(3).to(self.device))
        self.register_buffer('spine_angle', torch.eye(3).to(self.device))
        self.register_buffer('spine1_pos', torch.zeros(3).to(self.device))
        self.register_buffer('spine1_angle', torch.eye(3).to(self.device))
        self.register_buffer('thorax_pos', torch.zeros(3).to(self.device))
        self.register_buffer('thorax_angle', torch.eye(3).to(self.device))
        self.register_buffer('head_pos', torch.zeros(3).to(self.device))
        self.register_buffer('head_angle', torch.eye(3).to(self.device))

        self.name_to_ind = SKELETON_H36M_MODEL['name_to_ind']
        self.ind_to_angle = SKELETON_H36M_MODEL['angle_ind']
        self.offset = torch.FloatTensor(SKELETON_H36M_MODEL['bone_length'])
        self.offset = torch.reshape(self.offset, (-1, 3))

        # Directory defining the order and forward kinematic chain 
        self.fkm_computation_order = {
            1: ('rHip', 'hip'),
            2: ('rKnee', 'rHip'),
            3: ('rAnkle', 'rKnee'),
            4: ('rToe', 'rAnkle'),
            6: ('lHip', 'hip'),
            7: ('lKnee', 'lHip'),
            8: ('lAnkle', 'lKnee'),
            9: ('lToe', 'lAnkle'),
            11: ('spine', 'hip'),
            12: ('spine1', 'spine'),
            13: ('thorax', 'spine1'),
            14: ('neck', 'thorax'),
            15: ('head', 'neck'),
            16: ('lShoulderAnchor', 'spine1'),
            17: ('lShoulder', 'lShoulderAnchor'),
            18: ('lElbow', 'lShoulder'),
            19: ('lWrist', 'lElbow'),
            20: ('lThumb', 'lWrist'),
            22: ('lWristEnd', 'lWrist'),
            24: ('rShoulderAnchor', 'spine1'),
            25: ('rShoulder', 'rShoulderAnchor'),
            26: ('rElbow', 'rShoulder'),
            27: ('rWrist', 'rElbow'),
            28: ('rThumb', 'rWrist'),
            30: ('rWristEnd', 'rWrist')
        }

        

    def forward_kinematics(self, x: torch.Tensor):
        
        for id, (cur_frame, par_frame) in self.fkm_computation_order.items():
            
            frame_angle = getattr(self, f'{cur_frame}_angle')
            par_pos = getattr(self, f'{par_frame}_pos')
            par_angle = getattr(self, f'{par_frame}_angle')

            setattr(self,  f'{cur_frame}_pos',  par_angle.T @ self.offset[id] + par_pos)
            setattr(self, f'{cur_frame}_angle', frame_angle @ par_angle)


    def forward(self, x: torch.Tensor, format: str = 'ds_angle') -> torch.Tensor:
        if format == 'ds_angle':
            self._set_joint_angles_from_dataset(x)
            self.forward_kinematics(x)

    def get_joint_positions(self, incl_names = False):
        if incl_names:
            positions = {}
            positions['hip'] = self.hip_pos
        else:
            positions = []
            positions.append(self.hip_pos)
        for id, (cur_frame, par_frame) in self.fkm_computation_order.items():
            poss = getattr(self, f'{cur_frame}_pos')
            if incl_names:
                positions[cur_frame] = poss
            else:
                positions.append(poss)
        if not incl_names:
            positions = torch.cat(positions, dim=0)
        return positions
    
    def get_joint_angles(self):
        return self._get_joint_angles_dsFormat()

    def _set_joint_angles_from_dataset(self, joint_angles: torch.Tensor):
        
        self.hip_pos = self.offset[0] + joint_angles[[0, 1, 2]]# root
        self.hip_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[           [4, 5, 6]])
        self.rHip_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[          [7, 8, 9]])
        self.rKnee_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[         [10, 11, 12]])
        self.rAnkle_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[        [13, 14, 15]])
        self.rToe_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[          [16, 17, 18]])
        self.lHip_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[          [19, 20, 21]])# left leg
        self.lKnee_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[         [22, 23, 24]])
        self.lAnkle_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[        [25, 26, 27]])
        self.lToe_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[          [28, 29, 30]])
        self.spine_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[         [31, 32, 33]])# torso
        self.spine1_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[        [34, 35, 36]])
        self.thorax_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[        [37, 38, 39]])
        self.neck_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[          [40, 41, 42]])
        self.lShoulderAnchor_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[   [43, 44, 45]])# left arm
        self.lShoulder_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[         [46, 47, 48]])
        self.lElbow_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[            [49, 50, 51]])
        self.lWrist_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[            [52, 53, 54]])
        self.rShoulderAnchor_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[   [61, 62, 63]])# right arm
        self.rShoulder_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[         [64, 65, 66]])
        self.rElbow_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[            [67, 68, 69]])
        self.rWrist_angle = p3dTransforms.axis_angle_to_matrix(joint_angles[            [70, 71, 72]])





    def _get_joint_angles_dsFormat(self):
        joint_angles = torch.zeros(99)
        joint_angles[[0, 1, 2]] = self.hip_pos - self.offset[0]
        joint_angles[[4, 5, 6]] = p3dTransforms.matrix_to_axis_angle(self.hip_angle)

        joint_angles[[7, 8, 9]] = p3dTransforms.matrix_to_axis_angle(self.rHip_angle)
        joint_angles[[10, 11, 12]] = p3dTransforms.matrix_to_axis_angle(self.rKnee_angle)
        joint_angles[[13, 14, 15]] = p3dTransforms.matrix_to_axis_angle(self.rAnkle_angle)
        joint_angles[[16, 17, 18]] = p3dTransforms.matrix_to_axis_angle(self.rToe_angle)
        # left leg
        joint_angles[[19, 20, 21]] = p3dTransforms.matrix_to_axis_angle(self.lHip_angle)
        joint_angles[[22, 23, 24]] = p3dTransforms.matrix_to_axis_angle(self.lKnee_angle)
        joint_angles[[25, 26, 27]] = p3dTransforms.matrix_to_axis_angle(self.lAnkle_angle)
        joint_angles[[28, 29, 30]] = p3dTransforms.matrix_to_axis_angle(self.lToe_angle)
        # torso
        joint_angles[[31, 32, 33]] = p3dTransforms.matrix_to_axis_angle(self.spine_angle)
        joint_angles[[34, 35, 36]] = p3dTransforms.matrix_to_axis_angle(self.spine1_angle)
        joint_angles[[37, 38, 39]] = p3dTransforms.matrix_to_axis_angle(self.thorax_angle)
        joint_angles[[40, 41, 42]] = p3dTransforms.matrix_to_axis_angle(self.neck_angle)
        # left arm
        joint_angles[[43, 44, 45]] = p3dTransforms.matrix_to_axis_angle(self.lShoulderAnchor_angle)
        joint_angles[[46, 47, 48]] = p3dTransforms.matrix_to_axis_angle(self.lShoulder_angle)
        joint_angles[[49, 50, 51]] = p3dTransforms.matrix_to_axis_angle(self.lElbow_angle)
        joint_angles[[52, 53, 54]] = p3dTransforms.matrix_to_axis_angle(self.lWrist_angle)
        joint_angles[[55, 56, 57]] = p3dTransforms.matrix_to_axis_angle(self.lThumb_angle)
        joint_angles[[58, 59, 60]] = p3dTransforms.matrix_to_axis_angle(self.lWristEnd_angle)

        # right arm
        joint_angles[[61, 62, 63]] = p3dTransforms.matrix_to_axis_angle(self.rShoulderAnchor_angle)
        joint_angles[[64, 65, 66]] = p3dTransforms.matrix_to_axis_angle(self.rShoulder_angle)
        joint_angles[[67, 68, 69]] = p3dTransforms.matrix_to_axis_angle(self.rElbow_angle)
        joint_angles[[70, 71, 72]] = p3dTransforms.matrix_to_axis_angle(self.rWrist_angle)
        joint_angles[[73, 74, 75]] = p3dTransforms.matrix_to_axis_angle(self.lThumb_angle)
        joint_angles[[76, 77, 78]] = p3dTransforms.matrix_to_axis_angle(self.lWristEnd_angle)


        return joint_angles
    
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


def baseline_forward_kinematics(angles, parent = SKELETON_H36M_PARENT_IDS, angle_indices = BASELINE_FKL_IND, offset = SKELETON_H36M_BONE_LENGTH):
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

        # if not rotInd[i]:  # If the list is empty
        #     xangle, yangle, zangle = 0, 0, 0
        # else:
        #     xangle = angles[rotInd[i][0] - 1]
        #     yangle = angles[rotInd[i][1] - 1]
        #     zangle = angles[rotInd[i][2] - 1]
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
    # xyz = xyz[:, [0, 2, 1]]
    # xyz = xyz[:,[2,0,1]]

    return xyz    