"""
    File to run the evaluation for different models.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
import argparse
import os

from src import Session
from src.utils import print_ 
from typing import List, Optional

from src.utils import set_random_seed

#####===== Visualization Functions =====#####



#####===== Run/Visualization Information =====#####
# These list of runs can be used to run multiple trainings sequentially.
QUEUED = False # Activate the usage of the training queue
EXPERIMENT_NAMES = ['repr_loss_study']*7
RUN_NAMES = ['6d_abs_1', '6d_geo_1', '6d_matMSE_1', 'mat_geo_1', 'quat_geo_1', 'quat_matMSE_1', 'quat_quatLoss_1']
PREDICTION_TIMESTEPS = None

#####===== Meta Information =====#####

#####===== Helper Functions =====#####
def log_script_setup():
    """
        Log the script setup.
    """
    if QUEUED:
        pstr = 'Evaluation Queue:'
        pstr += '\n '.join([f'{exp_name}/{run_name}' for exp_name, run_name in zip(EXPERIMENT_NAMES, RUN_NAMES)])
    else:
        pstr = 'No training queue defined.'
    print_(pstr)


#####===== Main Function =====#####
def run_visualization(experiment_name: str,
                      run_name: str, 
                      checkpoint_name: str,
                      dataset: str,
                      visualization_type: List[str],
                      log: bool, 
                      num_visualizations: Optional[int] = 1,
                      interactive: Optional[bool] = False,
                      overlay: Optional[bool] = False,
                      pred_length: Optional[int] = None,
                      debug: bool = False,
                      split_actions: Optional[bool] = False,
                      random_seed: Optional[int] = None):
    """
        Run a training using the baseline trainer.

        TODO: Implement checkpoint loading
    """

    num_threads = 0 if debug else 4
    # Initialize the trainer
    session = Session(experiment_name, run_name, log_process_external=log, num_threads=num_threads, debug=debug)
    # Manually set the random seed
    if random_seed is not None:
        set_random_seed(random_seed)
    # Log some information
    log_script_setup()
    # Initialize the model
    session.initialize_model()
    session.load_checkpoint(checkpoint_name)
    # Initialize the evaluation
    if pred_length is None:
        pred_length = PREDICTION_TIMESTEPS
    else:
        pred_length = [pred_length]
    session.initialize_visualization(
        visualization_type=visualization_type,
        prediction_timesteps=pred_length,
        dataset=dataset,
        interactive=interactive,
        overlay_visualization=overlay,
        split_actions=split_actions

    )
    if debug:
        session.num_iterations = 10
        session.num_eval_iterations = 10
        session.num_epochs = 5
    # Evaluate the model
    session.visualize(num_visualizations)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiment', type=str, default=None, help='Experiment name')
    parser.add_argument('-r','--run', type=str, default=None, help='Run name')
    parser.add_argument('-c','--checkpoint', type=str, default=None, help='Checkpoint name to resume the training from')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed')
    parser.add_argument('-d', '--dataset', type=str, default='h36m', help='Dataset to visualize')
    parser.add_argument('--vis2d', action='store_true', default=False, help='2D visualization')
    parser.add_argument('--num', type=int, default=None, help='Number of sequences to visualize')
    parser.add_argument("--vis3d", action='store_true', default=False, help='3D visualization')
    parser.add_argument('--length', type=int, default=None, help='Length of the sequence to be visualized')
    parser.add_argument('--interactive', action='store_true', default=False, help='Interactive visualization')
    parser.add_argument('--overlay', action='store_true', default=False, help='Overlay visualization of different skeletons')
    parser.add_argument('--split_actions', action='store_true', default=False, help='Split actions in the dataset')
    parser.add_argument('--log', action='store_true', default=False, help='Log the training process to WandB')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    args = parser.parse_args()
    # Run a training using the queued runs from above
    if QUEUED:
        if len(EXPERIMENT_NAMES)==0:
            print_(f'No experiment names defined.')
            return
        elif len(EXPERIMENT_NAMES)!=len(RUN_NAMES):
            print_(f'Number of experiment names ({len(EXPERIMENT_NAMES)}) does not match number of run names ({len(RUN_NAMES)})')
            return
        else:
            for i in range(len(EXPERIMENT_NAMES)):
                exp_name = EXPERIMENT_NAMES[i]
                run_name = RUN_NAMES[i]
                visualization_type = []
                if args.vis3d:
                    visualization_type.append('3d')
                if args.vis2d:
                    visualization_type.append('2d')
                run_visualization(
                    experiment_name=exp_name,
                    run_name=run_name,
                    checkpoint_name=args.checkpoint,
                    dataset=args.dataset,
                    visualization_type=visualization_type,
                    log=args.log,
                    pred_length=args.length,
                    num_visualizations=args.num,
                    interactive=args.interactive,
                    overlay=args.overlay,
                    debug=args.debug,
                    split_actions=args.split_actions,
                    random_seed=args.seed
                )
    # Run a single training for the run defined by the environment or passed as arguments
    else:
        if args.experiment is None:
            if 'CURRENT_EXP' not in os.environ.keys():
                print_(f'No experiment name passed. Training aborted!')
                return
            print_('Take experiment name from environment.')
            exp_name = os.environ['CURRENT_EXP']
        else:
            exp_name = args.experiment
        
        if args.run is None:
            if 'CURRENT_RUN' not in os.environ.keys():
                print_(f'No run name passed. Training aborted!')
                return
            else:
                print_('Take run name from environment.')
                run_name = os.environ['CURRENT_RUN']
        else:
            run_name = args.run
        
        visualization_type = []
        if args.vis3d:
            visualization_type.append('3d')
        if args.vis2d:
            visualization_type.append('2d')
        run_visualization(
            experiment_name=exp_name,
            run_name=run_name,
            checkpoint_name=args.checkpoint,
            dataset=args.dataset,
            visualization_type=visualization_type,
            num_visualizations=args.num,
            interactive=args.interactive,
            pred_length=args.length,
            overlay=args.overlay,
            log=args.log,
            debug=args.debug,
            split_actions=args.split_actions,
            random_seed=args.seed
        )

if __name__ == '__main__':
    main()


