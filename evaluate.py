"""
    File to run the evaluation for different models.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
import argparse
import os
from typing import List

from src import Session
from src.utils import print_ 

#####===== Evaluation Functions =====#####
# Functions to initialize a session and run an evaluation.

def run_evaluation_00(experiment_name: str, run_name: str, checkpoint_name: str, log: bool, debug: bool = False, iterations: int = None, metric_names: List[str] = None, split_actions: bool = False, distribution_metric_names: List[str] = None, untrained: bool = False):
    """
        Run an evaluation.

    """

    num_threads = 0 if debug else 4
    # Initialize the evaluator
    session = Session(experiment_name, run_name, log_process_external=log, num_threads=num_threads, debug=debug)
    # Log some information
    log_script_setup()
    # Initialize the model
    session.initialize_model()
    if not untrained:
        session.load_checkpoint(checkpoint_name)
    # Initialize the evaluation
    session.initialize_evaluation(num_iterations=iterations,
                                  distance_metrics=metric_names,
                                  split_actions=split_actions,
                                  distribution_metrics=distribution_metric_names
                                  )
    if debug:
        session.num_iterations = 10
        session.num_eval_iterations = 10 if iterations is None else iterations
        session.num_epochs = 5
    # Evaluate the model
    session.evaluate()


#####===== Run Information =====#####
# These list of runs can be used to run multiple evaluations sequentially.
QUEUED = False # Activate the usage of the evaluation queue
EXPERIMENT_NAMES = ['repr_loss_study']*7
RUN_NAMES = ['6d_abs_1', '6d_geo_1', '6d_matMSE_1', 'mat_geo_1', 'quat_geo_1', 'quat_matMSE_1', 'quat_quatLoss_1']
PREDICTION_TIMESTEPS = None


#####===== Meta Information =====#####
EVALUATION_FUNCTIONS = {
    0: run_evaluation_00
}

#####===== Helper Functions =====#####
def log_script_setup():
    """
        Log the script setup.
    """
    if QUEUED:
        pstr = 'Evaluation Queue:'
        pstr += '\n '.join([f'{exp_name}/{run_name}' for exp_name, run_name in zip(EXPERIMENT_NAMES, RUN_NAMES)])
    else:
        pstr = 'No evaluation queue defined.'
    print_(pstr)


#####===== Main Functions =====#####
def run_evaluation(experiment_name: str, run_name: str, checkpoint_name: str, evaluation_id: int, log: bool = False, debug: bool = False, iterations: int = None, metric_names: List[str] = None, split_actions: bool = False, distribution_metric_names: List[str] = None, untrained: bool = False):
    eval_func = EVALUATION_FUNCTIONS[evaluation_id]
    eval_func(experiment_name, run_name, checkpoint_name, log, debug, iterations, metric_names, split_actions, distribution_metric_names, untrained)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiment', type=str, default=None, help='Experiment name')
    parser.add_argument('-r','--run', type=str, default=None, help='Run name')
    parser.add_argument('-c','--checkpoint', type=str, default=None, help='Checkpoint name to resume the evaluation from')
    parser.add_argument('-i','--evaluation_id', type=int, default=0, help='Evaluation ID: 0: Per-joint metrics')
    parser.add_argument('--log', action='store_true', default=False, help='Log the evaluation process to WandB')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--iterations', type=int, default=None, help='Number of iterations to run the evaluation for')
    parser.add_argument('--metric_names', type=str, nargs='+', default=None, help='List of metric names to evaluate')
    parser.add_argument('--distribution_metric_names', type=str, nargs='+', default=None, help='List of distribution metric names to evaluate')
    parser.add_argument('--split_actions', action='store_true', default=False, help='Split the actions')
    # Use untrained model for debugging purposes
    parser.add_argument('--untrained', action='store_true', default=False, help='Use untrained model')


    args = parser.parse_args()
    # Run an evaluation using the queued runs from above
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
                run_evaluation(exp_name, run_name, args.checkpoint, args.evaluation_id, args.log, args.debug, args.iterations, args.metric_names, args.split_actions, args.distribution_metric_names, args.untrained)
    # Run a single evaluation for the run defined by the environment or passed as arguments
    else:
        if args.experiment is None:
            if 'CURRENT_EXP' not in os.environ.keys():
                print_(f'No experiment name passed. Evaluation aborted!')
                return
            print_('Take experiment name from environment.')
            exp_name = os.environ['CURRENT_EXP']
        else:
            exp_name = args.experiment
        
        if args.run is None:
            if 'CURRENT_RUN' not in os.environ.keys():
                print_(f'No run name passed. Evaluation aborted!')
                return
            else:
                print_('Take run name from environment.')
                run_name = os.environ['CURRENT_RUN']
        else:
            run_name = args.run
        
        run_evaluation(exp_name, run_name, args.checkpoint, args.evaluation_id, args.log, args.debug, args.iterations, args.metric_names, args.split_actions, args.distribution_metric_names, args.untrained)

if __name__ == '__main__':
    main()


