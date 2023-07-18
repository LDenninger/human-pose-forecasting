import torch
import torch.nn as nn

from .meta_info import SKELETON_H36M_MODEL

class SkeletonModel32(nn.Module):

    def __init__(self):
        super(SkeletonModel32, self).__init__()
        self.register_buffer('root_pos', torch.zeros(3))
        self.register_buffer('hip_pos', torch.zeros(3))
        self.register_buffer('hip_angle', torch.eye(3))
        self.register_buffer('rHip_pos', torch.zeros(3))
        self.register_buffer('rHip_angle', torch.eye(3))
        self.register_buffer('lHip_pos', torch.zeros(3))
        self.register_buffer('lHip_angle', torch.eye(3))
        self.register_buffer('rKnee_pos', torch.zeros(3))
        self.register_buffer('rKnee_angle', torch.eye(3))
        self.register_buffer('lKnee_pos', torch.zeros(3))
        self.register_buffer('lKnee_angle', torch.eye(3))
        self.register_buffer('rAnkle_pos', torch.zeros(3))
        self.register_buffer('rAnkle_angle', torch.eye(3))
        self.register_buffer('lAnkle_pos', torch.zeros(3))
        self.register_buffer('lAnkle_angle', torch.eye(3))
        self.register_buffer('rToe_pos', torch.zeros(3))
        self.register_buffer('rToe_angle', torch.eye(3))
        self.register_buffer('lToe_pos', torch.zeros(3))
        self.register_buffer('lToe_angle', torch.eye(3))
        self.register_buffer('rShoulderAnchor_pos', torch.zeros(3))
        self.register_buffer('rShoulderAnchor_angle', torch.eye(3))
        self.register_buffer('lShoulderAnchor_pos', torch.zeros(3))
        self.register_buffer('lShoulderAnchor_angle', torch.eye(3))
        self.register_buffer('rShoulder_pos', torch.zeros(3))
        self.register_buffer('rShoulder_angle', torch.eye(3))
        self.register_buffer('lShoulder_pos', torch.zeros(3))
        self.register_buffer('lShoulder_angle', torch.eye(3))
        self.register_buffer('rElbow_pos', torch.zeros(3))
        self.register_buffer('rElbow_angle', torch.eye(3))
        self.register_buffer('lElbow_pos', torch.zeros(3))
        self.register_buffer('lElbow_angle', torch.eye(3))
        self.register_buffer('rWrist_pos', torch.zeros(3))
        self.register_buffer('rWrist_angle', torch.eye(3))
        self.register_buffer('lWrist_pos', torch.zeros(3))
        self.register_buffer('lWrist_angle', torch.eye(3))
        self.register_buffer('rThumb_pos', torch.zeros(3))
        self.register_buffer('rThumb_angle', torch.eye(3))
        self.register_buffer('lThumb_pos', torch.zeros(3))
        self.register_buffer('lThumb_angle', torch.eye(3))
        self.register_buffer('rWristEnd_pos', torch.zeros(3))
        self.register_buffer('rWristEnd_angle', torch.eye(3))
        self.register_buffer('lWristEnd_pos', torch.zeros(3))
        self.register_buffer('lWristEnd_angle', torch.eye(3))
        self.register_buffer('neck_pos', torch.zeros(3))
        self.register_buffer('neck_angle', torch.eye(3))
        self.register_buffer('spine_pos', torch.zeros(3))
        self.register_buffer('spine_angle', torch.eye(3))
        self.register_buffer('spine1_pos', torch.zeros(3))
        self.register_buffer('spine1_angle', torch.eye(3))
        self.register_buffer('thorax', torch.zeros(3))
        self.register_buffer('thorax_angle', torch.eye(3))
        self.register_buffer('head_pos', torch.zeros(3))
        self.register_buffer('head_angle', torch.eye(3))

        self.name_to_ind = SKELETON_H36M_MODEL.name_to_ind
        self.ind_to_angle = SKELETON_H36M_MODEL.angle_ind
        self.offset = torch.FloatTensor(SKELETON_H36M_MODEL.bone_length)
        self.offset = torch.reshape(self.offset, (-1, 3))

        # Directory defining the order and forward kinematic chain 
        self.fkm_computation_order = {
            1: ('rHip', 'hip'),
            2: ('rKnee', 'rHip'),
            3: ('rAnkle', 'rKnee'),
            4: ('rToe', 'rAnkle'),
            5: ('lHip', 'hip'),
            6: ('lKnee', 'lHip'),
            7: ('lAnkle', 'lKnee'),
            8: ('lToe', 'lAnkle'),
            9: ('spine', 'hip'),
            10: ('spine1', 'spine'),
            11: ('thorax', 'spine1'),
            12: ('neck', 'thorax'),
            13: ('head', 'neck'),
            14: ('lShoulderAnchor', 'spine1'),
            15: ('lShoulder', 'lShoulderAnchor'),
            16: ('lElbow', 'lShoulder'),
            17: ('lWrist', 'lElbow'),
            18: ('lThumb', 'lWrist'),
            20: ('lWristEnd', 'lWrist'),
            21: ('rShoulderAnchor', 'spine1'),
            22: ('rShoulder', 'rShoulderAnchor'),
            23: ('rElbow', 'rShoulder'),
            24: ('rWrist', 'rElbow'),
            25: ('rThumb', 'rWrist'),
            26: ('rWristEnd', 'rWrist')
        }

        

    def forward_kinematics(self, x: torch.Tensor):
        def compute_forward_kinematics():
            pass
        
        self.hip_pos = self.offset[0]

        for id, (cur_frame, par_frame) in self.fkm_computation_order.items():
            
            frame_pos = getattr(self, f'{cur_frame}_pos')
            frame_angle = getattr(self, f'{cur_frame}_angle')
            par_pos = getattr(self, f'{par_frame}_pos')
            par_angle = getattr(self, f'{par_frame}_angle')



    def forward(self, x: torch.Tensor, format: str = 'ds_angle') -> torch.Tensor:
        if format == 'ds_angle':
            self._set_joint_angles_from_dataset(x)
            self.forward_kinematics(x)

    
    def _set_joint_angles_from_dataset(self, joint_angles: torch.Tensor):
        # root
        self.hip_angle = joint_angles[[3, 4, 5]]
        # right leg
        self.rHip_angle = joint_angles[[6, 7, 8]]        
        self.rKnee_angle = joint_angles[[9, 10, 11]]     
        self.rAnkle_angle = joint_angles[[12, 13, 14]]   
        self.rToe_angle = joint_angles[[15, 16, 17]]   
        # left leg
        self.lHip_angle = joint_angles[[21, 22, 23]]
        self.lKnee_angle = joint_angles[[24, 25, 26]]
        self.lAnkle_angle = joint_angles[[27, 28, 29]]
        self.lToe_angle = joint_angles[[30, 31, 32]]
        # torso
        self.spine_angle = joint_angles[[36, 37, 38]]
        self.spine1_angle = joint_angles[[39, 40, 41]]
        self.thorax_angle = joint_angles[[42, 43, 44]]
        self.neck_angle = joint_angles[[45, 46, 47]]
        # left arm
        self.lShoulderAnchor_angle = joint_angles[[51, 52, 53]]
        self.lShoulder_angle = joint_angles[[54, 55, 56]]
        self.lElbow_angle = joint_angles[[57, 58, 59]]
        self.lWrist_angle = joint_angles[[60, 61, 62]]
        # right arm
        self.rShoulderAnchor_angle = joint_angles[[75, 76, 77]]
        self.rShoulder_angle = joint_angles[[78, 79, 80]]
        self.rElbow_angle = joint_angles[[81, 82, 83]]
        self.rWrist_angle = joint_angles[[84, 85, 86]]
