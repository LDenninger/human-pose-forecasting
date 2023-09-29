import os
import argparse
import json
import typing
from typing import List, Union
import re

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.utils import logging, print_

def main():
    """
    Visualize the distribution metrics of the data.
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiments', type=str, nargs='+', default=None, help='Experiment names to resume the visualization from')
    parser.add_argument('-c','--checkpoints', type=str, nargs='+', default=None, help='Checkpoint names to resume the visualization from')
    parser.add_argument('-d', '--dataset', type=str, default='h36m', help='Dataset to use in visualization')

    args = parser.parse_args()

    # Set up default values
    if args.experiments is None:
        if 'CURRENT_EXP' not in os.environ.keys():
            print_('No experiment name passed. Visualization aborted!')
            return
        print_('Take experiment name from environment.')
        exp_name = os.environ['CURRENT_EXP']
        if 'CURRENT_RUN' not in os.environ.keys():
            print_('No run name passed. Visualization aborted!')
            return
        print_('Take run name from environment.')
        run_name = os.environ['CURRENT_RUN']
        exp_ids = [f'{exp_name}.{run_name}']
    else:
        exp_ids = args.experiments
    dataset = args.dataset
    if args.checkpoints is None:
        checkpoint_names = ['final' for i in range(len(exp_ids))]
    else:
        checkpoint_names = args.checkpoints
        # Check if the number of checkpoint names matches the number of experiment names or is 1
        if len(checkpoint_names) == 1:
            checkpoint_names = checkpoint_names * len(exp_ids)
        elif len(checkpoint_names) != len(exp_ids):
            raise ValueError(f'Number of checkpoint names ({len(checkpoint_names)}) does not match number of experiment names ({len(exp_ids)}).')
    data_overall = []
    for i, exp_id in enumerate(exp_ids):    
        exp_name, run_name = exp_id.split('.')
        # Set up logging (only for getting the path to the log folder and visualization folder)
        logger = logging.Logger(exp_name=exp_name, run_name=run_name)
        log_path = logger.get_path('log')
        vis_path = logger.get_path('visualization')
        try:
            data = load_data(log_path, checkpoint=checkpoint_names[i], dataset=dataset)
        except (IOError, ValueError) as e:
            print_(e.message)
            return
        data_overall.append(data)
    
    # Load baselines
    baselines = load_baselines(dataset=dataset)

    # Create plot of distribution metrics
    fig = create_plots(data_overall, exp_ids, baselines=baselines)

    # Create list of experiment names concatenated with respective checkpoint names
    exp_runs = [f'{exp_id}.{checkpoint}' for exp_id, checkpoint in zip(exp_ids, checkpoint_names)]

    # Save the plot
    fig.savefig(os.path.join(vis_path, f'distribution_metrics_{exp_runs}_{dataset}.png'))

    print_('Visualization of distribution metrics finished.')
    
def load_baselines(dataset: str='h36m'):
    """
    Load the baseline distribution metrics from file.
    """
    baseline_names = [f'entropy_s16_{dataset}_norm_False_abs_True', f'kld_s16_{dataset}_norm_False_abs_True', f'entropy_s16_{dataset}_norm_True_abs_False', f'kld_s16_{dataset}_norm_True_abs_False']
    baselines = []
    for baseline_name in baseline_names:
        try:
            # Load the data
            path = os.path.join('configurations', 'distribution_values', f'{baseline_name}.pt')
            baselines.append(torch.load(path).numpy())
        except IOError as e:
            e.message = f'Could not load the baseline distribution metrics from file: {path}.'
            raise e
    return baselines
    

def load_data(path: str, action: str='overall', checkpoint: str='final', dataset: str='h36m'):
    """
    Load the distribution metrics from file.
    """
    try:
        # Load the data
        with open(os.path.join(path, f'eval_results_distribution_{checkpoint}_{dataset}.json'), 'r') as f:
            data = json.load(f)
    except IOError as e:
        e.message = f'Could not load the distribution metrics from file: {path}.'
        raise e
    if action not in data.keys():
        raise ValueError(f'No {action} distribution metrics found in the data. Not yet implemented for this action type.')
    data = data[action]
    return data

def create_plots(data: Union[dict, List[dict]], exp_ids: List[str], baselines:List=None, plot_size: tuple=(35, 10.5)):
    """
    Create one plot of predictions' entropy vs. baseline entropy and one of predictions' kld vs baseline kld.
    """
    data = data if isinstance(data, list) else [data]
    fonttype = 'Computer Modern Roman'
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": fonttype,
    })
    fig, axs = plt.subplots(1, 2, figsize=(plot_size[0], plot_size[1]))

    # Extract the baseline values
    entropy_norm_false = baselines[0]
    kld_norm_false = baselines[1]
    entropy_norm_true = baselines[2]
    kld_norm_true = baselines[3]

    fontsize = 56
    line_width = 8

    

    # Create linspace for each baseline value 
    length_kld = len(data[0]['kld'])
    length_entropy = len(data[0]['entropy'])
    entropy_baseline = np.linspace(entropy_norm_false, entropy_norm_false, length_entropy)
    kld_baseline = np.linspace(kld_norm_false, kld_norm_false, length_kld)
    entropy_norm_true = np.linspace(entropy_norm_true, entropy_norm_true, length_entropy)
    kld_norm_true = np.linspace(kld_norm_true, kld_norm_true, length_kld)

    # Get crest color palette for plotting
    palette = [(0.34509803921568627, 0.4666666666666667, 0.5725490196078431), (0.9254901960784314, 0.3058823529411765, 0.12549019607843137)]

    # Nicer less technical legend names
    # (remove all the numbers and underscores and the word model)
    exp_ids = [re.sub(r'[0-9]+', '', exp_id) for exp_id in exp_ids]
    exp_ids = [re.sub(r'_', ' ', exp_id) for exp_id in exp_ids]
    exp_ids = [re.sub(r'model', '', exp_id) for exp_id in exp_ids]

    # Create a plot for each metric pair
    for i, (result, exp_id) in enumerate(zip(data, exp_ids)):
        exp_name, run_name = exp_id.split('.')
        run_name = run_name.strip()
        # Extract the data
        entropy = result['entropy']
        kld = result['kld']
        # Create the plots
        axs[1].plot(range(len(kld)),kld, label=f'{run_name}', color=palette[i], linewidth=line_width)
        axs[0].plot(entropy, label=f'{run_name}', color=palette[i], linewidth=line_width)
    
    axs[0].plot(entropy_baseline, '--', label='baseline', color=palette[0], linewidth=line_width)
    axs[1].plot(kld_baseline, '--', label='baseline', color=palette[0], linewidth=line_width)
    axs[0].plot(entropy_norm_true, '--', label='baseline (norm)', color=palette[-1], linewidth=line_width)
    axs[1].plot(kld_norm_true, '--', label='baseline (norm)', color=palette[-1], linewidth=line_width)

    # Convert xaxis of entropy plot from frame number to seconds (with a framerate of 25fps)
    axs[0].set_xlabel('Time (s)', fontsize=fontsize)
    axs[1].set_xlabel('Time (s)', fontsize=fontsize)
    # Set yaxis labels
    axs[0].set_ylabel('PS Entropy', fontsize=fontsize)
    axs[1].set_ylabel('PS KLD', fontsize=fontsize)
    # Set xticks so that every other second is shown
    axs[0].set_xticks(range(0, len(entropy), 50))
    axs[1].set_xticks(range(0, len(kld), 2))
    # Set xticklabels
    axs[0].set_xticklabels(range(0, len(entropy)//25, 2))
    axs[1].set_xticklabels(range(0, len(kld), 2))

    # Increase font size of all text
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)
        ax.legend(fontsize=fontsize).set_visible(False)
        # Also for titles
        ax.title.set_size(fontsize)
        # Add grid to the plots
        ax.grid(True, linewidth=line_width-5)

    # Increase horizontal space between subplots
    fig.subplots_adjust(wspace=0.2)

    # Add whitespace below the plots
    fig.subplots_adjust(bottom=0.15, left = 0.07, right = 0.99)

    # Return the figure
    return fig

        

            


    
if __name__ == '__main__':
    main()