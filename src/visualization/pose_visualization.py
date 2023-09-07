import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import plotly.graph_objects as go

#####===== Visualization Parameters =====#####
JOINT_COLORS = [
    "b",  # blue
    "g",  # green
    "r",  # red
    "c",  # cyan
    "m",  # magenta
    "y",  # yellow
    "k",  # black
    "w",  # white
    "#FF5733",  # orange
    "#800080",  # purple
    "#00FF00",  # lime
    "#008080",  # teal
    "#FFD700",  # gold
    "#FF69B4",  # hot pink
    "#9ACD32",  # yellow green
    "#00FFFF",  # aqua
    "#8A2BE2",  # blue violet
    "#ADFF2F",  # green yellow
    "#DC143C",  # crimson
    "#FF6347",  # tomato
    "#4B0082",  # indigo
    "#1E90FF",  # dodger blue
    "#FF4500",  # orange red
    "#32CD32",  # lime green
    "#20B2AA",  # light sea green
    "#FF8C00",  # dark orange
    "#6A5ACD",  # slate blue
    "#FA8072",  # salmon
    "#00CED1",  # dark turquoise
    "#BA55D3",  # medium orchid
    "#228B22",  # forest green
]

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


def visualize_skeleton(position_data, skeleton_structure, title_text=""):
    def update(frame):
        ax.cla()
        ax.set_xlim3d([-10, 10])  # Set the x-axis limits
        ax.set_ylim3d([-10, 10])  # Set the y-axis limits
        ax.set_zlim3d([-10, 10])  # Set the z-axis limits
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        joint_positions = position_data[frame]
        for joint_name, joint_position in joint_positions.items():
            ax.scatter(
                joint_position[0] / 100.0,
                joint_position[2] / 100.0,
                joint_position[1] / 100.0,
                label=joint_name,
                c=JOINT_COLOR_MAP[joint_name],
            )

        for id, (cur_frame, par_frame) in skeleton_structure.items():
            if cur_frame == "hip":
                continue
            start_pos = joint_positions[cur_frame].numpy() / 100.0
            end_pos = joint_positions[par_frame].numpy() / 100.0
            ax.plot(
                [start_pos[0], end_pos[0]],
                [start_pos[2], end_pos[2]],
                [start_pos[1], end_pos[1]],
                c="g",
            )
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5)
        ax.legend(loc="upper left")

        ax.set_title((f"Frame {frame}/{len(position_data)} " + title_text))
        plt.pause(0.01)

    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(111, projection="3d")
    animation = FuncAnimation(fig, update, frames=len(position_data), interval=100)
    plt.show()


def compare_skeleton(position_data_1, position_data_2, skeleton_structure_1, skeleton_structure_2, title_text=""):
    """
        Compare two skeleton models. The first model is painted green and the second model is painted red.
    """

    def update(frame):
        ax.cla()
        ax.set_xlim3d([-10, 10])  # Set the x-axis limits
        ax.set_ylim3d([-10, 10])  # Set the y-axis limits
        ax.set_zlim3d([-10, 10])  # Set the z-axis limits
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        joint_positions_1 = position_data_1[frame]
        joint_positions_2 = position_data_2[frame]
        for i, (joint_name, joint_position) in enumerate(joint_positions_1.items()):
            ax.scatter(
                joint_position[0] / 100.0,
                joint_position[2] / 100.0,
                joint_position[1] / 100.0,
                label=joint_name,
                c=JOINT_COLOR_MAP[joint_name],
            )
        for i, (joint_name, joint_position) in enumerate(joint_positions_2.items()):
            ax.scatter(
                joint_position[0] / 100.0,
                joint_position[2] / 100.0,
                joint_position[1] / 100.0,
                c=JOINT_COLOR_MAP[joint_name],
            )

        for id, (cur_frame, par_frame) in skeleton_structure_1.items():
            if cur_frame == "hip":
                continue
            start_pos = joint_positions_1[cur_frame].numpy() / 100.0
            end_pos = joint_positions_1[par_frame].numpy() / 100.0
            ax.plot(
                [start_pos[0], end_pos[0]],
                [start_pos[2], end_pos[2]],
                [start_pos[1], end_pos[1]],
                c="g",
            )
            
        for id, (cur_frame, par_frame) in skeleton_structure_2.items():
            if cur_frame == "hip":
                continue
            start_pos = joint_positions_2[cur_frame].numpy() / 100.0
            end_pos = joint_positions_2[par_frame].numpy() / 100.0
            ax.plot(
                [start_pos[0], end_pos[0]],
                [start_pos[2], end_pos[2]],
                [start_pos[1], end_pos[1]],
                c="r",
            )
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5)
        ax.legend(loc="upper left")

        ax.set_title((f"Frame {frame}/{len(position_data_1)} " + title_text))
        plt.pause(0.01)

    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(111, projection="3d")
    animation = FuncAnimation(fig, update, frames=len(position_data_1), interval=100)
    plt.show()


def create_skeleton_subplot(
    position_data,
    skeleton_structure,
    ax,
    title_text="",
):
    ax.set_xlim3d([-4.5, 4.5])
    ax.set_ylim3d([-4.5, 4.5])
    ax.set_zlim3d([-4.5, 4.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    ax.set_axis_off()

    joint_positions = position_data
    for joint_name, joint_position in joint_positions.items():
        ax.scatter(
            joint_position[0] / 100.0,
            joint_position[2] / 100.0,
            joint_position[1] / 100.0,
            label=joint_name,
            alpha=0,
        )

    for id, (cur_frame, par_frame) in skeleton_structure.items():
        if cur_frame == "hip":
            continue
        start_pos = joint_positions[cur_frame].numpy() / 100.0
        end_pos = joint_positions[par_frame].numpy() / 100.0
        ax.plot(
            [start_pos[0], end_pos[0]],
            [start_pos[2], end_pos[2]],
            [start_pos[1], end_pos[1]],
            c="g",
        )

    # ax.legend(loc="upper left")
    ax.set_title(title_text)
    ax.view_init(elev=0, azim=-90)


def create_skeleton_subplot_plotly(
    subplot, position_data, skeleton_structure, show_joints=False
):
    """
    Add data creating a skeleton to a subplot.

    @param subplot: The subplot to add the data to.
    @param position_data: The data to add to the subplot.
    @param skeleton_structure: The skeleton structure to use for the data.
    @param show_joints: Whether to show the joints in the plot.

    """
    # Extract joint positions
    joint_positions = position_data

    # Convert PyTorch tensors to NumPy arrays
    for joint_name, joint_position in joint_positions.items():
        joint_positions[joint_name] = joint_position.numpy()

    if show_joints:
        # Create scatter plot for each joint within the subplot
        for joint_name, joint_position in joint_positions.items():
            subplot.x += joint_position[0]
            subplot.y += joint_position[2]
            subplot.z += joint_position[1]

    # Define lines connecting joints within the subplot
    for id, (cur_frame, par_frame) in skeleton_structure.items():
        if cur_frame == "hip":
            continue
        start_pos = joint_positions[cur_frame]
        end_pos = joint_positions[par_frame]
        subplot.x += (start_pos[0], end_pos[0], None)
        subplot.y += (start_pos[2], end_pos[2], None)
        subplot.z += (start_pos[1], end_pos[1], None)


    # Return the subplot
    return subplot
