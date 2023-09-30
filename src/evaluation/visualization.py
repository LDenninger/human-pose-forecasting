"""
    Visualization functions for the evluation.

    Author: Leon Herbrik
"""

import time
from typing import List, Tuple
import math

import torch
from matplotlib import pyplot as plt
from matplotlib import gridspec
import plotly.graph_objects as go
import plotly.io as pio
import io
from PIL import Image
from plotly.subplots import make_subplots
import numpy as np

from ..data_utils import (
    SkeletonModel32,
    H36MDataset,
    H36M_SKELETON_STRUCTURE,
    baseline_forward_kinematics,
    convert_baseline_representation,
    expmap2rotmat,
    axis_angle_to_matrix,
)
from ..visualization import (
    JOINT_COLORS,
    JOINT_COLOR_MAP,
    create_skeleton_subplot,
    get_line_data,
    get_joint_data,
)

def compare_sequences_plotly(
    sequence_names: List[str],
    sequences: List[torch.Tensor],
    time_steps_ms: List[List[int]],
    skeleton_structure: dict,
    parent_ids: List[int],
    prediction_positions: List[int] = None,
    title_text: str = "Time (ms)",
    save_path: str = None,
    line_width: int = 4,
    font_size: int = 24,
    colors: Tuple[str, str] = ("green", "red"),
    show_joints: bool = False,
    show_joint_labels: bool = False,
    size: int = 500,
    
):
    """
    Visualize comparison of predicted sequence and ground truth sequence in 3D plotly.

    @param sequence_names: List of sequence names - ['gt', 'pred']
    @param sequences: List of sequences - [gt_seq, pred_seq]
    @param prediction_positions: List of positions of where prediction starts in the sequence
    @param skeleton_structure: Dictionary containing the skeleton structure
    @param parent_ids: List of parent ids for each joint
    @param time_steps_ms: List of time steps in milliseconds for each sequence
    @param title_text: Title of the plot
    @param figsize: Size of the figure (width, height) in pixels
    @param save_path: Path to save the figure to
    @param line_width: Width of the lines in the plot (scaled with size of plot)
    @param font_size: Font size of the text in the plot (scaled with size of plot)
    @param colors: Tuple of two colors of the lines in the plot. First one is the color the ground truth should have, second one is the color the prediction should have.
    @param show_joints: Whether to show the joints in the plot
    @param size: Size of the plot (roughly in pixels, gets multiplied by aspect (so 500 with two rows equates a 4000x1000 plot))

    @return: Numpy array of the image

    """
    # Make sure the number of sequence names, sequences and prediction positions are the same
    assert len(sequence_names) == len(sequences)
    if prediction_positions is not None:
        assert len(sequence_names) == len(prediction_positions)
    nrows = len(sequences)
    ncols = sequences[0].shape[0]

    max_sequence_length = max([sequence.shape[0] for sequence in sequences])

    # Calculate aspect ratio of the plot (0.215 came from trial and error)
    aspect = (0.28 * max_sequence_length,  len(sequences))

    # Round size up to the nearest multiple of 100
    figsize = (
        math.ceil(aspect[0] * size / 100) * 100,
        math.ceil(aspect[1] * size / 100) * 100,
    )

    # Make fontsize and linewidth scale with size
    font_size = int(50 / 500 * size)
    line_width = int(4 / 500 * size)
    marker_size = int(5 / 500 * size)

    # Flatten time_steps_ms if it is not None
    if time_steps_ms is not None:
        time_steps_ms = time_steps_ms
    # Turn time_steps_ms into a list of strings
    time_steps_ms = list(map(lambda x: str(x), time_steps_ms))

    # Create a figure with no gaps between subplots
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "scatter3d"} for i in range(ncols)] for j in range(nrows)],
        vertical_spacing=0,
        horizontal_spacing=0,
        subplot_titles=time_steps_ms
    )

    # Create y positions as mean points between each row
    y_positions = calculate_mean_points(nrows)

    # Loop through the sequences and frames
    for i, (name, sequence) in enumerate(zip(sequence_names, sequences)):
        prediction_position = math.inf if prediction_positions is None or len(prediction_positions) == 0 or prediction_positions[i] is None else prediction_positions[i]
        
        for j in range(ncols):
            # Get joint positions from skeleton
            joint_positions = sequence[j]
            # Get color based on prediction_position
            color = colors[0] if j < prediction_position else colors[1]
            if show_joints:
                x, y, z, text = get_joint_data(joint_positions, skeleton_structure)
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        text=text,
                        mode="markers+text" if show_joint_labels else "markers",
                        marker=dict(size=marker_size, color=color),
                    ),
                    row=i + 1,
                    col=j + 1,
                )
            x, y, z, text = get_line_data(
                joint_positions,
                parent_ids
            )
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    text=text,
                    mode="lines",
                    line=dict(width=line_width, color=color),
                    marker=dict()
                ),
                row=i + 1,
                col=j + 1,
            )

        # Calculate the y-coordinate for the annotation to position it correctly
        y_coord = y_positions[len(y_positions) - i - 1] + 100 / figsize[1]

        # Add row name to the left of each row
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=-(70 / figsize[0]),
            y=y_coord,
            text=name,
            showarrow=False,
            font=dict(size=font_size),
            textangle=-90,  # Rotate text 90 degrees counter-clockwise
        )

    # Set title, disable legend and adjust size
    fig.update_layout(
        title_text=title_text,
        showlegend=False,
        width=figsize[0],
        height=figsize[1],
        margin=dict(l=100, r=150, b=0, t=100, pad=0),
    )
    # Remove axes and background and ticks
    fig.update_scenes(
        dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
            aspectmode="manual",
            aspectratio=dict(x=0.375, y=0.25, z=1.875),
        )
    )   

    # Set figure title to title_text
    fig.update_layout(
        font_family="Arial Black",
        title_text=title_text,
        title_x=0.5,
        title_font_size=font_size,
        title_font_family="Arial Black"
    )

    # Update font size
    fig.update_annotations(font_size=font_size)

    if save_path is not None:
        fig.write_image(save_path)

    # Use Plotly's to_image method to convert the figure to an image
    fig_data = pio.to_image(fig, format="png")

    # Convert the binary image data to a Pillow Image object
    image = Image.open(io.BytesIO(fig_data))

    # Convert the Pillow Image to a numpy array
    image_array = np.array(image)

    return image_array


def calculate_mean_points(n):
    interval_width = 1 / n  # Calculate the width of each interval
    mean_points = []  # Initialize a list to store the mean points

    for i in range(n):
        start = i * interval_width  # Calculate the starting point of the interval
        end = (i + 1) * interval_width  # Calculate the ending point of the interval
        mean = (start + end) / 2  # Calculate the mean point for the interval
        mean_points.append(mean)  # Append the mean point to the list

    return mean_points





# def compare_sequences(
#     sequence_names: List[str],
#     sequences: List[torch.Tensor],
#     title_text="",
#     figsize=(16, 9),
# ):
#     """
#     Visualize comparison of predicted sequence and ground truth sequence in 2D image.

#     @param sequence_names: List of sequence names - ['gt', 'pred']
#     @param sequences: List of sequences - [gt_seq, pred_seq]
#     @param title_text: Title of the plot

#     """

#     assert len(sequence_names) == len(sequences)

#     # Calculate the subplot size based on the number of sequences and frames
#     # subplot_size = (
#     #     width := figsize[0] / sequences[0].shape[0],
#     #     height := figsize[1] / len(sequences),
#     # )

#     nrows = len(sequences)
#     ncols = sequences[0].shape[0]
#     x_aspect, y_aspect, z_aspect = 4, 4, 6

#     # Create a figure with no gaps between subplots
#     fig, axs = plt.subplots(figsize=figsize)
#     gs = gridspec.GridSpec(
#         nrows,
#         ncols,
#         figure=fig,
#         height_ratios=[5] * nrows,
#         width_ratios=[1] * ncols,
#     )

#     # Loop through the sequences and frames
#     for i, sequence in enumerate(sequences):
#         for j in range(sequence.shape[0]):
#             # Set joints of skeleton to those in the sequence
#             self.skeleton(sequence[j])
#             joint_positions = self.skeleton.getJointPositions(incl_names=True)
#             ax = fig.add_subplot(gs[i, j], projection="3d")
#             # Call the create_skeleton_subplot function
#             create_skeleton_subplot(joint_positions, H36M_SKELETON_STRUCTURE, ax)
#             # Draw red rectangle around subplot so its size is obvious
#             # ax.patch.set_edgecolor("red")
#             # ax.patch.set_linewidth(2)
#             ax.set_box_aspect([1, 1, 5])

#             # Adjust the aspect ratio of the subplot itself
#             ax.set_aspect("auto")  # 'auto' allows the subplot to adjust

#     # Add title to figure
#     fig.suptitle(title_text)

#     # Remove all paddings around subplots
#     fig.tight_layout(pad=0.0)

#     # # Store image in same folder for testing
#     # plt.savefig(
#     #     f"compare_sequences.png",
#     #     bbox_inches="tight",
#     #     pad_inches=0,
#     # )

#     # Adjust layout and show figure
#     plt.show()

