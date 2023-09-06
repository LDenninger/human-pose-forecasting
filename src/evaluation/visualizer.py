import time
from typing import List, Tuple
import math

import torch
from matplotlib import pyplot as plt
from matplotlib import gridspec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from IPython.display import Image

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
    create_skeleton_subplot_plotly,
)


class Visualizer:
    def __init__(self, experiment_name: str = "test", run_name: str = "test"):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.skeleton: SkeletonModel32 = SkeletonModel32()

    def compare_sequences(
        self,
        sequence_names: List[str],
        sequences: List[torch.Tensor],
        title_text="",
        figsize=(16, 9),
    ):
        """
        Visualize comparison of predicted sequence and ground truth sequence in 2D image.

        @param sequence_names: List of sequence names - ['gt', 'pred']
        @param sequences: List of sequences - [gt_seq, pred_seq]
        @param title_text: Title of the plot

        """

        assert len(sequence_names) == len(sequences)

        # Calculate the subplot size based on the number of sequences and frames
        # subplot_size = (
        #     width := figsize[0] / sequences[0].shape[0],
        #     height := figsize[1] / len(sequences),
        # )

        nrows = len(sequences)
        ncols = sequences[0].shape[0]
        x_aspect, y_aspect, z_aspect = 4, 4, 6

        # Create a figure with no gaps between subplots
        fig, axs = plt.subplots(figsize=figsize)
        gs = gridspec.GridSpec(
            nrows,
            ncols,
            figure=fig,
            height_ratios=[5] * nrows,
            width_ratios=[1] * ncols,
        )

        # Loop through the sequences and frames
        for i, sequence in enumerate(sequences):
            for j in range(sequence.shape[0]):
                # Set joints of skeleton to those in the sequence
                self.skeleton(sequence[j])
                joint_positions = self.skeleton.getJointPositions(incl_names=True)
                ax = fig.add_subplot(gs[i, j], projection="3d")
                # Call the create_skeleton_subplot function
                create_skeleton_subplot(joint_positions, H36M_SKELETON_STRUCTURE, ax)
                # Draw red rectangle around subplot so its size is obvious
                # ax.patch.set_edgecolor("red")
                # ax.patch.set_linewidth(2)
                ax.set_box_aspect([1, 1, 5])

                # Adjust the aspect ratio of the subplot itself
                ax.set_aspect("auto")  # 'auto' allows the subplot to adjust

        # Add title to figure
        fig.suptitle(title_text)

        # Remove all paddings around subplots
        fig.tight_layout(pad=0.0)

        # # Store image in same folder for testing
        # plt.savefig(
        #     f"compare_sequences.png",
        #     bbox_inches="tight",
        #     pad_inches=0,
        # )

        # Adjust layout and show figure
        plt.show()

    def compare_sequences_plotly(
        self,
        sequence_names: List[str],
        sequences: List[torch.Tensor],
        time_steps_ms: List[List[int]],
        prediction_positions: List[int] = None,
        title_text: str = "",
        save_path: str = None,
        line_width: int = 4,
        colors: Tuple[str, str] = ("green", "blue"),
        show_joints: bool = False,
        size: int = 500,
    ):
        """
        Visualize comparison of predicted sequence and ground truth sequence in 3D plotly.

        @param sequence_names: List of sequence names - ['gt', 'pred']
        @param sequences: List of sequences - [gt_seq, pred_seq]
        @param prediction_positions: List of positions of where prediction starts in the sequence
        @param time_steps_ms: List of time steps in milliseconds for each sequence
        @param title_text: Title of the plot
        @param figsize: Size of the figure (width, height) in pixels
        @param save_path: Path to save the figure to
        @param line_width: Width of the lines in the plot
        @param colors: Tuple of two colors of the lines in the plot. First one is the color the ground truth should have, second one is the color the prediction should have.
        @param show_joints: Whether to show the joints in the plot
        @param size: Size of the plot (roughly in pixels, gets multiplied by aspect (so 500 with two rows equates a 4000x1000 plot))



        """

        # Make sure the number of sequence names, sequences and prediction positions are the same
        assert len(sequence_names) == len(sequences)
        if prediction_positions is not None:
            assert len(sequence_names) == len(prediction_positions)
        nrows = len(sequences)
        ncols = sequences[0].shape[0]

        max_sequence_length = max([sequence.shape[0] for sequence in sequences])

        # Calculate aspect ratio of the plot (0.215 came from trial and error)
        aspect = (0.215 * max_sequence_length, len(sequences))

        # Round size up to the nearest multiple of 100
        figsize = (
            math.ceil(aspect[0] * size / 100) * 100,
            math.ceil(aspect[1] * size / 100) * 100,
        )

        # Flatten time_steps_ms if it is not None
        if time_steps_ms is not None:
            time_steps_ms = time_steps_ms

        # Create a figure with no gaps between subplots
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            specs=[[{"type": "scatter3d"} for i in range(ncols)] for j in range(nrows)],
            vertical_spacing=0.1,
            horizontal_spacing=0,
            subplot_titles=time_steps_ms,
        )

        # Create y positions as mean points between each row
        y_positions = calculate_mean_points(nrows)

        # Loop through the sequences and frames
        for i, (name, sequence) in enumerate(zip(sequence_names, sequences)):
            prediction_position = (
                prediction_positions[i] if prediction_positions else math.inf
            )
            for j in range(ncols):
                # Set joints of skeleton to those in the sequence
                self.skeleton(sequence[j])
                # Get joint positions from skeleton
                joint_positions = self.skeleton.getJointPositions(incl_names=True)
                # Get color based on prediction_position
                color = colors[0] if j < prediction_position else colors[1]
                # Fill subplot with skeleton and additional information
                subplot = create_skeleton_subplot_plotly(
                    go.Scatter3d(
                        x=[],
                        y=[],
                        z=[],
                        mode="lines",
                        line=dict(width=line_width, color=color),
                    ),
                    joint_positions,
                    H36M_SKELETON_STRUCTURE,
                    show_joints=show_joints,
                )

                # Add newly created subplot to figure
                fig.add_trace(subplot, row=i + 1, col=j + 1)

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
                font=dict(size=24),
                textangle=-90,  # Rotate text 90 degrees counter-clockwise
            )

        # Set title, disable legend and adjust size
        fig.update_layout(
            title_text=title_text,
            showlegend=False,
            width=figsize[0],
            height=figsize[1],
            margin=dict(l=100, r=0, b=0, t=100, pad=0),
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

        if save_path is not None:
            fig.write_image(save_path)
        else:
            fig.write_image(
                f"compare_sequences_plotly_{ncols}_{figsize[0]}_{figsize[1]}.png"
            )


def calculate_mean_points(n):
    interval_width = 1 / n  # Calculate the width of each interval
    mean_points = []  # Initialize a list to store the mean points

    for i in range(n):
        start = i * interval_width  # Calculate the starting point of the interval
        end = (i + 1) * interval_width  # Calculate the ending point of the interval
        mean = (start + end) / 2  # Calculate the mean point for the interval
        mean_points.append(mean)  # Append the mean point to the list

    return mean_points
