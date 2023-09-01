import time
from typing import List

import torch
from matplotlib import pyplot as plt
from matplotlib import gridspec

from ..data_utils import (
    SkeletonModel32,
    H36MDataset,
    H36M_SKELETON_STRUCTURE,
    baseline_forward_kinematics,
    convert_baseline_representation,
    expmap2rotmat,
    axis_angle_to_matrix,
)
from ..visualization import JOINT_COLORS, JOINT_COLOR_MAP, create_skeleton_subplot


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
        figsize=(16, 10),
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
        figsize = (24, 24)

        # Create a figure with no gaps between subplots
        fig, axs = plt.subplots(figsize=figsize)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig, height_ratios=[1.5] * nrows)

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
                ax.patch.set_edgecolor("red")
                ax.patch.set_linewidth(2)

        # Add title to figure
        fig.suptitle(title_text)

        # Remove all paddings around subplots
        fig.tight_layout(pad=0.0)

        # Adjust layout and show figure
        plt.show()
