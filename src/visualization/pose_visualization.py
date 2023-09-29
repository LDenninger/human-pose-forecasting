import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import numpy as np
import plotly.graph_objects as go
import os
import shutil
import subprocess

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

def visualize_single_pose(position_data, skeleton_parents, ax=None, return_img=False):
    if ax is None:
        fig = plt.figure(figsize=(18, 18))
        ax = fig.add_subplot(111, projection="3d")
        return_fig = True
    else:
        return_fig = False

    joint_positions = position_data

    for id, par_id in enumerate(skeleton_parents):
        if id == 0:
            continue
        start_pos = joint_positions[id].numpy() 
        end_pos = joint_positions[par_id].numpy()
        ax.plot(
            [start_pos[0], end_pos[0]],
            [start_pos[2], end_pos[2]],
            [start_pos[1], end_pos[1]],
            c="g",
        )
    ax.set_aspect('equal', adjustable='box')
    if return_img:
        ax.axis("off")
        canvas = fig.canvas
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image_array = image_array.reshape(int(height), int(width), 3)
        return image_array
    if return_fig:
        return fig
    else:
        return ax

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


def get_line_data(
    position_data, parent_ids
):
    """
    Generate data for creating lines for the a skeleton plot.

    @param position_data: The data to add to the subplot.
    @param parent_ids: The parent ids of the skeleton structure.

    """
    import ipdb; ipdb.set_trace()
    
    x, y, z, text = [], [], [], []
    # Extract joint positions
    joint_positions = position_data.numpy()

    # Define lines connecting joints within the subplot
    for idx, joint_position in enumerate(joint_positions):
        if idx == 0:
            continue
        # Draw line from current position to parent position
        parent_position = joint_positions[parent_ids[idx]]
        y.extend([joint_position[2], parent_position[2], None])
        z.extend([joint_position[1], parent_position[1], None])
        x.extend([joint_position[0], parent_position[0], None])
        # if show_joint_labels:
        #     subplot.text += ("", "", "")
    # Return the data
    return x, y, z, text

def get_joint_data(position_data, skeleton_structure):
    x, y, z, text = [], [], [], []
    # Extract joint positions
    joint_positions = position_data.numpy()
 
    # Create scatter plot for each joint within the subplot
    for i, joint_position in enumerate(joint_positions):
        x.append(joint_position[0])
        y.append(joint_position[2])
        z.append(joint_position[1])
        text.append(skeleton_structure[i][0])
    return x, y, z, text


def animate_pose_matplotlib(positions, colors, titles, fig_title, parents, change_color_after_frame=None,
                       color_after_change=None, overlay=False, fps=60, step_size=40, out_dir=None, to_video=True, fname=None,
                       keep_frames=True, show_axis=False, constant_limits=False, notebook=False):
    """
    Visualize motion given 3D positions. Can visualize several motions side by side. If the sequence lengths don't
    match, all animations are displayed until the shortest sequence length.
    Args:
        positions: a list of np arrays in shape (seq_length, n_joints, 3) giving the 3D positions per joint and frame
        colors: list of color for each entry in `positions`
        titles: list of titles for each entry in `positions`
        fig_title: title for the entire figure
        parents: skeleton structure
        fps: frames per second
        change_color_after_frame: after this frame id, the color of the plot is changed (for each entry in `positions`)
        color_after_change: what color to apply after `change_color_after_frame`
        overlay: if true, all entries in `positions` are plotted into the same subplot
        out_dir: output directory where the frames and video is stored. Don't pass for interactive visualization.
        to_video: whether to convert frames into video clip or not.
        fname: video file name.
        keep_frames: Whether to keep video frames or not.
    """
    seq_length = np.amin([pos.shape[0] for pos in positions])
    n_joints = positions[0].shape[1]
    pos = positions

    # create figure with as many subplots as we have skeletons
    fig = plt.figure(figsize=(16, 9))
    plt.clf()
    n_axes = 1 if overlay else len(pos)
    axes = [fig.add_subplot(1, n_axes, i + 1, projection='3d') for i in range(n_axes)]
    fig.suptitle(fig_title)

    # create point object for every bone in every skeleton
    all_lines = []
    # available_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    for i, joints in enumerate(pos):
        idx = 0 if overlay else i
        ax = axes[idx]

        lines_j = [
            ax.plot(joints[0:1, n, 0], joints[0:1, n, 1], joints[0:1, n, 2], '-o',
                    markersize=2.0, color=colors[i])[0] for n in range(1, n_joints)]
        all_lines.append(lines_j)
        ax.set_title(titles[i])

    # dirty hack to get equal axes behaviour
    min_val = np.amin(pos[0], axis=(0, 1))
    max_val = np.amax(pos[0], axis=(0, 1))
    max_range = (max_val - min_val).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max_val[0] + min_val[0])
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max_val[1] + min_val[1])
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max_val[2] + min_val[2])

    for ax in axes:
        ax.set_aspect('equal')
        if not show_axis:
            ax.axis('off')
        else:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')


        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        ax.view_init(elev=20, azim=-56)
        if constant_limits:
            ax.set_box_aspect([1.0, 1.0, 1.0])


    def on_move(event):
        # find which axis triggered the event
        source_ax = None
        for i in range(len(axes)):
            if event.inaxes == axes[i]:
                source_ax = i
                break

        # transfer rotation and zoom to all other axes
        if source_ax is None:
            return

        for i in range(len(axes)):
            if i != source_ax:
                axes[i].view_init(elev=axes[source_ax].elev, azim=axes[source_ax].azim)
                axes[i].set_xlim3d(axes[source_ax].get_xlim3d())
                axes[i].set_ylim3d(axes[source_ax].get_ylim3d())
                axes[i].set_zlim3d(axes[source_ax].get_zlim3d())
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig_text = fig.text(0.05, 0.05, '')

    def update_frame(num, positions, lines):
        for l in range(len(positions)):
            k = 0
            pos = positions[l]
            points_j = lines[l]
            for i in range(1, len(parents)):
                a = pos[num, i]
                b = pos[num, parents[i]]
                p = np.vstack([b, a])
                points_j[k].set_data(p[:, :2].T)
                points_j[k].set_3d_properties(p[:, 2].T)
                if change_color_after_frame and change_color_after_frame[l] and num >= change_color_after_frame[l]:
                    points_j[k].set_color(color_after_change)
                else:
                    points_j[k].set_color(colors[l])
                k += 1

        time_passed = '{:>.2f} seconds passed'.format(step_size/1000 * num)
        fig_text.set_text(time_passed)

    # create the animation object, for animation to work reference to this object must be kept
    fargs = (pos, all_lines)
    line_ani = FuncAnimation(fig, update_frame, seq_length, fargs=fargs, interval=1000 / fps)
    
    if notebook:
        plt.close()
        return line_ani
    elif out_dir is None:
        plt.show()  # interactive
    else:
        video_dir = os.path.join(out_dir, "videos")
        save_to = os.path.join(out_dir, "frames", fname + "_skeleton")

        if not os.path.exists(save_to):
            os.makedirs(save_to)
            
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        # Save frames to disk.
        for j in range(0, seq_length):
            update_frame(j, *fargs)
            fig.savefig(os.path.join(save_to, 'frame_{:0>4}.{}'.format(j, "png")), dpi=1000)

        # Create a video clip.
        if to_video:
            save_to_movie(os.path.join(video_dir, fname + "_skeleton.mp4"), os.path.join(save_to, 'frame_%04d.png'))
        
        # Delete frames if they are not required to store.
        if not keep_frames:
            shutil.rmtree(save_to)
        
    plt.close()


def save_to_movie(out_path, frame_path_format, fps=60, start_frame=0):
    """Creates an mp4 video clip by using already stored frames in png format.

    Args:
        out_path: <output-file-path>.mp4
        frame_path_format: <path-to-frames>frame_%04d.png
        fps:
        start_frame:
    Returns:
    """
    # create movie and save it to destination
    command = ['ffmpeg',
               '-start_number', str(start_frame),
               '-framerate', str(fps),  # must be this early, otherwise it is not respected
               '-r', '30',  # output is 30 fps
               '-loglevel', 'panic',
               '-i', frame_path_format,
               '-c:v', 'libx264',
               '-preset', 'slow',
               '-profile:v', 'high',
               '-level:v', '4.0',
               '-pix_fmt', 'yuv420p',
               '-y',
               out_path]
    fnull = open(os.devnull, 'w')
    subprocess.Popen(command, stdout=fnull).wait()
    fnull.close()
    

