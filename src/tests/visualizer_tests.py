import time
import random

import torch
import numpy as np
from PIL import Image

from ..data_utils import (
    SkeletonModel32,
    H36MDataset,
    H36M_SKELETON_STRUCTURE,
    H36M_SKELETON_PARENTS,
    H36M_NON_REDUNDANT_PARENT_IDS,
    SH_SKELETON_PARENTS,
    baseline_forward_kinematics,
    convert_baseline_representation,
    expmap2rotmat,
    axis_angle_to_matrix,
    SH_SKELETON_STRUCTURE,
    SH_SKELETON_PARENTS,
    h36m_forward_kinematics,
    normalize_sequence_orientation
)
from ..visualization import visualize_skeleton, compare_skeleton, animate_pose_matplotlib, visualize_attention, visualize_single_pose
from ..evaluation import (
    compare_sequences_plotly
)


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

    # Visualize sequences
    img = compare_sequences_plotly(
        sequence_names=sequence_names,
        sequences=sequences,
        skeleton_structure=H36M_SKELETON_STRUCTURE,
        parent_ids=H36M_SKELETON_PARENTS,
        title_text="Test Sequence",
        time_steps_ms=[f"{i * 100}" for i in range(sequences[0].shape[0])],
        prediction_positions=[
            (i * 4 + 10) % sequences[0].shape[0] for i in range(len(sequences))
        ],
    )

    # Show image which is in numpy format
    Image.fromarray(img).show()

    pass

def test_baseline_visualization():

    seed_length=40
    target_length=20
    dataset = H36MDataset(
        seed_length=seed_length,
        target_length=target_length,
        down_sampling_factor=2,
        rot_representation = 'pos',
        stacked_hourglass=True,
        sequence_spacing=100,
        return_label=False,
        raw_data=False,
        is_train=True,
        debug=True)

    for seq in dataset:

        #seq = seq[...,[2,0,1]]
        seq_orth = normalize_sequence_orientation(seq)

        animate_pose_matplotlib(
                positions = (seq_orth.numpy(), seq.numpy()),
                colors = ('g', 'g'),
                titles = ("norm", "gt"),
                fig_title = "Visualization Test",
                parents = SH_SKELETON_PARENTS,
                change_color_after_frame=(seed_length, None),
                show_axis=True,
                color_after_change='r',
                overlay=False,
                fps=25,
                
            )

def test_attention_visualization():
    dataset = H36MDataset(
        seed_length=10,
        target_length=10,
        down_sampling_factor=2,
        rot_representation = 'pos',
        stacked_hourglass=True,
        sequence_spacing=5,
        return_label=False,
        raw_data=False,
        normalize_orientation=False,
        is_train=True,
        debug=True)
    
    seq = dataset[0][:4]
    np.random.seed(24)
    spatial_attention = np.random.rand(4,10,10)
    temporal_attention = np.random.rand(4,10,10)
    skeleton_parents = SH_SKELETON_PARENTS
    fig = visualize_single_pose(seq[0], skeleton_parents )
    fig.savefig('debug/test_single_pose.png')
    fig = visualize_attention(temporal_attention=temporal_attention, 
                              poses=seq, 
                              spatial_attention=spatial_attention, 
                              skeleton_parents=skeleton_parents,
                              timesteps=[80,120,160,200])
    import ipdb; ipdb.set_trace()
    fig.savefig('debug/test_attn_2.png')


def test_plotly_visualization():
    dataset = H36MDataset(
        seed_length=10,
        target_length=10,
        down_sampling_factor=2,
        rot_representation = 'pos',
        stacked_hourglass=True,
        sequence_spacing=5,
        return_label=False,
        raw_data=False,
        normalize_orientation=False,
        is_train=True,
        debug=True)
    
    seq = dataset[0]

    