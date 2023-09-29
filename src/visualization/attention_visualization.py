import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 

from .pose_visualization import visualize_single_pose


def visualize_attention(temporal_attention, poses, timesteps, skeleton_parents, spatial_attention = None):
    """
        Visualize the attention weights
    """
    num_columns = len(timesteps)
    num_rows = 2

    fig = plt.figure(figsize=(40,15))
    figure_grid = GridSpec(num_rows, num_columns, height_ratios=[3,2], wspace=0.2, hspace=None)

    for col in range(num_columns):
        if spatial_attention is not None:
            sub_grid = figure_grid[0,col].subgridspec(1,2, wspace=0.2)
            ax = fig.add_subplot(sub_grid[0,0])
            ax.imshow(temporal_attention[col])
            ax = fig.add_subplot(sub_grid[0,1])
            ax.imshow(spatial_attention[col])
        else:
            sub_grid = figure_grid[0,col*2:col*2+1].subgridspec(1,1)
            ax = fig.add_subplot(sub_grid[0,0])
            ax.imshow(temporal_attention[col])
        
    for col in range(num_columns):
        sub_grid = figure_grid[1,col].subgridspec(1,1)
        ax = fig.add_subplot(sub_grid[0,0], projection='3d')
        ax = visualize_single_pose(poses[col], skeleton_parents, ax)
        ax.axis('off')
        ax.set_title(timesteps[col], y=-0.1, fontsize=32)

    return fig

