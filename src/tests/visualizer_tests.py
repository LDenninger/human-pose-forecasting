import time
import random

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
        return_label=False,
        is_train=True,
    )

    # Adjust amount of sequences to be displayed
    seq_amount = 4

    # Adjust amount of frames to be displayed
    frame_amount = 50

    # Grab random sequences from the dataset
    sequences = [
        dataset[random.randint(0, len(dataset) - 1)][:frame_amount]
        for _ in range(seq_amount)
    ]
    # Create sequence names
    sequence_names = [f"seq{i}" for i in range(seq_amount)]

    visualizer = Visualizer()

    # Visualize sequences
    visualizer.compare_sequences_plotly(
        sequence_names=sequence_names,
        sequences=sequences,
        title_text="Test Sequence",
        time_steps_ms=[f"{i * 100}" for i in range(sequences[0].shape[0])],
        prediction_positions=[
            (i * 4 + 10) % sequences[0].shape[0] for i in range(len(sequences))
        ],
    )

    pass
