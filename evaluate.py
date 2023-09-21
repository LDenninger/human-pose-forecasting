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

def run_distance_evaluation(experiment_name: str,
                             run_name: str,
                              checkpoint_name: str = None,
                               iterations: int = None,
                                debug: bool = False,
                                 dataset: str = 'h36m',
                                  split_actions: bool = False):
    """
    """
    num_threads = 0 if debug else 4
    # Initialize the evaluator
    session = Session(experiment_name, run_name, log_process_external=False, num_threads=num_threads, debug=debug)
    # Log some information
    log_script_setup()
    # Initialize the model
    session.initialize_model()
    if checkpoint_name is not None:
        session.load_checkpoint(checkpoint_name)
    # Initialize the evaluation
    session.initialize_evaluation(
                                    evaluation_type = ['distance'],
                                    num_iterations=iterations,
                                    distance_metrics=DISTANCE_METRICS,
                                    split_actions=split_actions,
                                    prediction_timesteps=DISTANCE_PREDICTION_TIMESTEPS,
                                    dataset = dataset
                                  )
    if debug:
        session.num_iterations = 10
        session.num_eval_iterations = 10 if iterations is None else iterations
        session.num_epochs = 5
    # Evaluate the model
    session.evaluate()
    # Save the evaluation results
    file_name = f'eval_results_distance_{checkpoint_name}.txt'
    session.evaluation_engine.print(file_name)

def run_distribution_evaluation(experiment_name: str,
                             run_name: str,
                              checkpoint_name: str = None,
                               iterations: int = None,
                                debug: bool = False,
                                 dataset: str = 'h36m',
                                  split_actions: bool = False,
                                    distr_pred_sec: int = 15):
    num_threads = 0 if debug else 4
    # Initialize the evaluator
    session = Session(experiment_name, run_name, log_process_external=False, num_threads=num_threads, debug=debug)
    # Log some information
    log_script_setup()
    # Initialize the model
    session.initialize_model()
    if checkpoint_name is not None:
        session.load_checkpoint(checkpoint_name)
    # Initialize the evaluation
    session.initialize_evaluation(
                                    evaluation_type = ['distribution'],
                                    num_iterations=iterations,
                                    distribution_metrics=DISTRIBUTION_METRICS,
                                    split_actions=split_actions,
                                    prediction_timesteps=DISTRIBUTION_PREDICTION_TIMESTEPS,
                                    dataset = dataset,
                                    distr_pred_sec = distr_pred_sec
                                  )
    if debug:
        session.num_iterations = 10
        session.num_eval_iterations = 10 if iterations is None else iterations
        session.num_epochs = 5
    # Evaluate the model
    session.evaluate()
    # Save the evaluation results
    file_name = f'eval_results_distribution_{checkpoint_name}.txt'
    session.evaluation_engine.print(file_name)


#####===== Meta Information =====#####
# Define a queue of runs to be evaluated.
# Definition of the evaluation to be run.
QUEUED = False # Activate the usage of the evaluation queue
EXPERIMENT_NAMES = ['repr_loss_study']*7
RUN_NAMES = ['6d_abs_1', '6d_geo_1', '6d_matMSE_1', 'mat_geo_1', 'quat_geo_1', 'quat_matMSE_1', 'quat_quatLoss_1']
DISTANCE_METRICS = ['positional_mse', 'auc']
DISTRIBUTION_METRICS = ['ps_entropy', 'ps_kld', 'npss']
DISTANCE_PREDICTION_TIMESTEPS = [80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560]
#DISTRIBUTION_PREDICTION_TIMESTEPS = [200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
DISTRIBUTION_PREDICTION_TIMESTEPS = [80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560]




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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiment', type=str, default=None, help='Experiment name')
    parser.add_argument('-r','--run', type=str, default=None, help='Run name')
    parser.add_argument('-c','--checkpoint', type=str, default=None, help='Checkpoint name to resume the evaluation from')
    parser.add_argument('--log', action='store_true', default=False, help='Log the evaluation process to WandB')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--distance', action='store_true', default=False, help='Evaluation of the distance metrics')
    parser.add_argument('--distribution', action='store_true', default=False, help='Evaluation of the distance metrics')
    parser.add_argument('--ais', action='store_true', default=False, help='Use AIS dataset for evaluation')
    parser.add_argument('--iterations', type=int, default=500, help='Number of iterations to run the evaluation for')
    parser.add_argument('--split_actions', action='store_true', default=False, help='Split the actions')
    # Use untrained model for debugging purposes
    parser.add_argument('--untrained', action='store_true', default=False, help='Use untrained model')
    parser.add_argument('--distr_pred_sec', type=int, default=15, help='Number of seconds to predict for the distribution evaluation')
    parser.add_argument('--distr_iterations', type=int, default=3, help='Number of iterations to run the distribution evaluation for')

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
                if args.distance:
                    run_distance_evaluation(
                        experiment_name=exp_name,
                        run_name=run_name,
                        checkpoint_name=args.checkpoint,
                        iterations=args.iterations,
                        debug=args.debug,
                        dataset="h36m" if not args.ais else 'ais',
                        split_actions=args.split_actions
                    )
                if args.distribution:
                    run_distribution_evaluation(
                        experiment_name=exp_name,
                        run_name=run_name,
                        checkpoint_name=args.checkpoint,
                        iterations=args.distr_iterations,
                        debug=args.debug,
                        dataset="h36m" if not args.ais else 'ais',
                        split_actions=args.split_actions,
                        distr_pred_sec=args.distr_pred_sec
                    )
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
        
        if args.distance:
                    run_distance_evaluation(
                        experiment_name=exp_name,
                        run_name=run_name,
                        checkpoint_name=args.checkpoint,
                        iterations=args.iterations,
                        debug=args.debug,
                        dataset="h36m" if not args.ais else 'ais',
                        split_actions=args.split_actions
                    )
        if args.distribution:
            run_distribution_evaluation(
                experiment_name=exp_name,
                run_name=run_name,
                checkpoint_name=args.checkpoint,
                iterations=args.distr_iterations,
                debug=args.debug,
                dataset="h36m" if not args.ais else 'ais',
                split_actions=args.split_actions,
                distr_pred_sec=args.distr_pred_sec
            )

if __name__ == '__main__':
    main()


