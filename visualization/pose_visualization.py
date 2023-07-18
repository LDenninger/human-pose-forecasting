import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def visualize_skeleton(position_data, skeleton_structure, title_text = ''):

    def update(frame):
        ax.cla()
        ax.set_xlim3d([-10, 10])  # Set the x-axis limits
        ax.set_ylim3d([-10, 10])  # Set the y-axis limits
        ax.set_zlim3d([-10, 10])  # Set the z-axis limits
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        joint_positions = position_data[frame]
        for joint_name, joint_position in joint_positions.items():
            ax.scatter(joint_position[0]/100., joint_position[2]/100., joint_position[1]/100., label=joint_name)
        
        for id, (cur_frame, par_frame) in skeleton_structure.items():
            if cur_frame == 'hip':
                continue
            start_pos = joint_positions[cur_frame].numpy()/100.
            end_pos = joint_positions[par_frame].numpy()/100.
            ax.plot([start_pos[0], end_pos[0]], [start_pos[2], end_pos[2]], [start_pos[1], end_pos[1]],  c='g')
        #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5)
        ax.legend(loc='upper left')

        ax.set_title((f'Frame {frame}/{len(position_data)} ' + title_text))
        plt.pause(0.01)

    fig = plt.figure(figsize=(18,18))
    ax = fig.add_subplot(111, projection='3d')
    animation = FuncAnimation(fig, update, frames=len(position_data), interval=100)
    plt.show()