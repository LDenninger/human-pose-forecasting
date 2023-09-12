import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 
import numpy as np
from typing import List, Union, Optional, Tuple, Literal

from ..data_utils import h36m_forward_kinematics, H36M_REDUCED_SKELETON_STRUCTURE, H36M_REVERSED_REDUCED_ANGLE_INDICES

JOINT_COLOR_MAP = {
    "": "#228B22",
    "hip": "b",
    "rHip": "g",
    "rKnee": "r",
    "rAnkle": "c",
    "rToe": "m",
    "site": "y",
    "lHip": "k",
    "lKnee": "#FF5733",
    "lAnkle": "#800080",
    "lToe": "#00FF00",
    "spine": "#008080",
    "spine1": "#FFD700",
    "thorax": "#FF69B4",
    "neck": "#9ACD32",
    "head": "#00FFFF",
    "lShoulderAnchor": "#8A2BE2",
    "lShoulder": "#ADFF2F",
    "lElbow": "#DC143C",
    "lWrist": "#FF6347",
    "lThumb": "#4B0082",
    "lWristEnd": "#1E90FF",
    "rShoulderAnchor": "#FF4500",
    "rShoulder": "#32CD32",
    "rElbow": "#20B2AA",
    "rWrist": "#FF8C00",
    "rThumb": "#6A5ACD",
    "rWristEnd": "#FA8072",
}

def draw_sequence_comp_plot(sequences: Union[List[torch.Tensor], torch.Tensor],
                             timesteps: List[int],
                              skeleton_model: Optional[Literal['s26']] = 's26',
                               rot_representation: Literal['axis', 'mat', 'quat', '6d'] = 'mat',
                                visualize_joints: Optional[bool] = False):

    def create_pose_visualization(ax: plt.axis, frame: torch.Tensor):
        ax.cla()
        ax.set_xlim3d([-10, 10])  # Set the x-axis limits
        ax.set_ylim3d([-10, 10])  # Set the y-axis limits
        ax.set_zlim3d([-10, 10])  # Set the z-axis limits
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        for id, (cur_frame, par_frame) in skeleton_structure.items():
            if visualize_joints:
                ax.scatter(
                    frame[id][0] / 100.0,
                    frame[id][2] / 100.0,
                    frame[id][1] / 100.0,
                    label=cur_frame,
                    c=JOINT_COLOR_MAP[cur_frame],
                )
            if id==0:
                continue
            start_pos = frame[id].numpy() / 100.0
            end_pos = frame[name_to_ind[par_frame]].numpy() / 100.0
            ax.plot(
                [start_pos[0], end_pos[0]],
                [start_pos[2], end_pos[2]],
                [start_pos[1], end_pos[1]],
                c="g",
            )
        return ax

    num_sequences = len(sequences)
    num_timesteps = len(timesteps)

    if skeleton_model =='s26':
        forward_kinematics = h36m_forward_kinematics
        skeleton_structure = H36M_REDUCED_SKELETON_STRUCTURE
        name_to_ind = H36M_REVERSED_REDUCED_ANGLE_INDICES
    else:
        raise ValueError(f'Skeleton model {skeleton_model} is not supported')

    fig = plt.figure(figsize=(30,15))
    main_grid = GridSpec(num_sequences*2, num_timesteps, figure=fig)

    for row in range(num_sequences):
        sequence = sequences[row]
        positions, abs_rotations = forward_kinematics(sequence, rot_representation)
        for col in range(num_timesteps):
            frame = positions[col]
            ax = fig.add_subplot(main_grid[row*2:row*2+1, col], projection='3d')
            ax = create_pose_visualization(ax, frame)
            if row == 0:
                ax.set_title(f'Timestep: {timesteps[col]}')
    return fig
