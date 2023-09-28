import os
import argparse
import json
import typing
from typing import List, Union

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import logging, print_

def main():
    """
    Visualize the distribution metrics of the data.
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiments', type=str, nargs='+', default=None, help='Experiment names to resume the visualization from')
    parser.add_argument('-c','--checkpoints', type=str, nargs='+', default=None, help='Checkpoint names to resume the visualization from')

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
            data = load_data(log_path, checkpoint=checkpoint_names[i])
        except (IOError, ValueError) as e:
            print_(e.message)
            return
        data_overall.append(data)
    
    # Create plot of distribution metrics
    fig = create_plots(data_overall, exp_ids)

    # Create list of experiment names concatenated with respective checkpoint names
    exp_runs = [f'{exp_id}.{checkpoint}' for exp_id, checkpoint in zip(exp_ids, checkpoint_names)]

    # Save the plot
    fig.savefig(os.path.join(vis_path, f'distribution_metrics_{exp_runs}.png'))

    print_('Visualization of distribution metrics finished.')
    

def load_data(path: str, action: str='overall', checkpoint: str='final'):
    """
    Load the distribution metrics from file.
    """
    try:
        # Load the data
        with open(os.path.join(path, f'eval_results_distribution_{checkpoint}.json'), 'r') as f:
            data = json.load(f)
    except IOError as e:
        e.message = f'Could not load the distribution metrics from file: {path}.'
        raise e
    if action not in data.keys():
        raise ValueError(f'No {action} distribution metrics found in the data. Not yet implemented for this action type.')
    data = data[action]
    return data

def create_plots(data: Union[dict, List[dict]], exp_ids: List[str], plot_size: tuple=(21, 7.5)):
    """
    Create one plot of predictions' entropy vs. baseline entropy and one of predictions' kld vs baseline kld.
    """
    data = data if isinstance(data, list) else [data]
    fig, axs = plt.subplots(1, 2, figsize=(plot_size[0], plot_size[1]))

    # Get crest color palette for plotting
    palette = sns.color_palette('crest', 15 * len(data))

    # Create a plot for each metric pair
    for i, (result, exp_id) in enumerate(zip(data, exp_ids)):
        exp_name, run_name = exp_id.split('.')
        # Extract the data
        entropy = result['entropy']
        entropy_baseline = result['entropy_baseline']
        kld = result['kld']
        kld_baseline = result['kld_baseline']
        # Add kld_baseline value again at the end of the list to make the plot look nicer
        kld_baseline.append(kld_baseline[-1])
        # Create the plots
        axs[0].plot(entropy, label=f'{run_name}', color=palette[6 + i * 15])
        axs[1].scatter(range(len(kld)),kld, label=f'{run_name}', color=palette[6 + i * 15])
    
    axs[0].plot(entropy_baseline, '--', label='baseline', color=palette[-1])
    axs[1].plot(kld_baseline, '--', label='baseline', color=palette[-1])
    # Set the legend
    axs[0].legend()
    axs[1].legend()
    # Convert xaxis of entropy plot from frame number to seconds (with a framerate of 25fps)
    axs[0].set_xlabel('Time (s)', fontsize=16)
    axs[1].set_xlabel('Time (s)', fontsize=16)
    # Set yaxis labels
    axs[0].set_ylabel('PS Entropy', fontsize=16)
    axs[1].set_ylabel('PS KLD', fontsize=16)
    # Set xticks so that every other second is shown
    axs[0].set_xticks(range(0, len(entropy_baseline), 50))
    axs[1].set_xticks(range(0, len(kld_baseline), 2))
    # Set xticklabels
    axs[0].set_xticklabels(range(0, len(entropy_baseline)//25, 2))
    axs[1].set_xticklabels(range(0, len(kld_baseline), 2))

    # Increase font size of all text
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)
        ax.legend(fontsize=16)
        # Also for titles
        ax.title.set_size(16)
        # Add grid to the plots
        ax.grid(True)

    # Increase horizontal space between subplots
    fig.subplots_adjust(wspace=0.3)

    # Return the figure
    return fig

        

            


    
if __name__ == '__main__':
    main()