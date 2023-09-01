import time

import torch

from ..data_utils import (
    SkeletonModel32,
    H36MDataset,
    H36M_SKELETON_STRUCTURE,
    baseline_forward_kinematics,
    convert_baseline_representation,
    expmap2rotmat,
    axis_angle_to_matrix,
)
from ..visualization import visualize_skeleton, compare_skeleton
from ..evaluation import Visualizer


def test_visualizer():
    skeleton = SkeletonModel32()

    dataset = H36MDataset(
        seed_length=40,
        target_length=1,
        down_sampling_factor=2,
        sequence_spacing=0,
        return_label=True,
        is_train=True,
    )

    # Grab three sequences from the dataset
    seq1, label1 = dataset[0]
    seq2, label2 = dataset[1]
    seq3, label3 = dataset[2]

    visualizer = Visualizer()

    # Visualize sequences
    visualizer.compare_sequences(
        ["seq1", "seq2"], [seq1[:5], seq2[:5]], title_text="Test Sequence"
    )

    pass
