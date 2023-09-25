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
    parser.add_argument('-e','--experiment', type=str, default=None, help='Experiment name')
    parser.add_argument('-r','--run', type=str, default=None, help='Run name')
    parser.add_argument('-c','--checkpoint', type=str, default=None, help='Checkpoint name to resume the visualization from')

    args = parser.parse_args()

    # Set up default values
    if args.experiment is None:
        if 'CURRENT_EXP' not in os.environ.keys():
            print_('No experiment name passed. Visualization aborted!')
            return
        print_('Take experiment name from environment.')
        exp_name = os.environ['CURRENT_EXP']
    else:
        exp_name = args.experiment
    if args.run is None:
        if 'CURRENT_RUN' not in os.environ.keys():
            print_('No run name passed. Visualization aborted!')
            return
        print_('Take run name from environment.')
        run_name = os.environ['CURRENT_RUN']
    if args.checkpoint is None:
        checkpoint_name = 'final'
    else:
        checkpoint_name = args.checkpoint

    # Set up logging (only for getting the path to the log folder and visualization folder)
    logger = logging.Logger(exp_name=exp_name, run_name=run_name)
    log_path = logger.get_path('log')
    vis_path = logger.get_path('visualization')

    try:
        data = load_data(log_path)
    except (IOError, ValueError) as e:
        print_(e.message)
        return
    
    # Create plot of distribution metrics
    fig = create_plots(data)

    # Save the plot
    fig.savefig(os.path.join(vis_path, 'distribution_metrics.png'))

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

def create_plots(data: Union[dict, List[dict]], plot_size: tuple=(16, 9)):
    """
    Create one plot of predictions' entropy vs. baseline entropy and one of predictions' kld vs baseline kld.
    """
    data = data if isinstance(data, list) else [data]
    
    fig, axs = plt.subplots(1, 2, figsize=plot_size)

    # Get crest color palette for plotting
    palette = sns.color_palette('crest', 15)

    # Create a plot for each metric pair
    for result in data:
        # Extract the data
        entropy = result['entropy']
        entropy_baseline = result['entropy_baseline']
        kld = result['kld']
        kld_baseline = result['kld_baseline']
        # Add kld_baseline value again at the end of the list to make the plot look nicer
        kld_baseline.append(kld_baseline[-1])
        # Set title of each subplot
        axs[0].set_title('Entropy')
        axs[1].set_title('KLD')
        # Create the plots
        axs[0].plot(entropy_baseline, '--', label='baseline', color=palette[13])
        axs[0].plot(entropy, label='prediction', color=palette[6])
        axs[1].plot(kld_baseline, '--', label='baseline', color=palette[13])
        axs[1].scatter(range(len(kld)),kld, label='prediction', color=palette[6])
        # Set the legend
        axs[0].legend()
        axs[1].legend()
        # Convert xaxis of entropy plot from frame number to seconds (with a framerate of 25fps)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_xticks([i for i in range(0, len(entropy) + 1, 25)])
        axs[0].set_xticklabels([int(i/25) for i in range(0, len(entropy) + 1, 25)])
        # Convert xaxis of kld so that it shows every second
        axs[1].set_xticks([i for i in range(0, len(kld) + 1)])
        axs[1].set_xticklabels([int(i) for i in range(0, len(kld) + 1)])

        axs[1].set_xlabel('Time (s)')

    
    # Return the figure
    return fig

        

            


    
if __name__ == '__main__':
    main()