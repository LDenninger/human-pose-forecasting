"""
    File to run the training for different models.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
import argparse
import os

from src import Session
from src.utils import print_

#####===== Training Functions =====#####
# Functions to initialize a session and run a training.

def run_training_00(experiment_name: str, run_name: str, checkpoint_name: str, log: bool, debug: bool = False):
    """
        Run a training using the baseline trainer.

        TODO: Implement checkpoint loading
    """

    num_threads = 0 if debug else 4

    # Initialize the trainer
    session = Session(experiment_name, run_name, log_process_external=log, num_threads=num_threads, debug=debug)
    # Log some information
    log_script_setup()
    # Initialize the model
    session.initialize_model()
    # Initialize the components of the optimization process (optimizer, scheduler, loss function)
    session.initialize_optimization()
    # Load the data
    session.load_train_data()
    # Initialize the evaluation
    session.initialize_evaluation()
    if debug:
        session.num_iterations = 10
        session.num_eval_iterations = 10
        session.num_epochs = 5
        session.evaluation_engine.iterations = 10
    # Train the model
    session.train()

#####===== Run Information =====#####
# These list of runs can be used to run multiple trainings sequentially.
QUEUED = False # Activate the usage of the training queue
EXPERIMENT_NAMES = ['repr_loss_study']*7
RUN_NAMES = ['6d_abs_1', '6d_geo_1', '6d_matMSE_1', 'mat_geo_1', 'quat_geo_1', 'quat_matMSE_1', 'quat_quatLoss_1']

#####===== Meta Information =====#####
TRAINING_FUNCTIONS = {
    0: run_training_00
}

#####===== Helper Functions =====#####
def log_script_setup():
    """
        Log the script setup.
    """
    if QUEUED:
        pstr = 'Training Queue:'
        pstr += '\n '.join([f'{exp_name}/{run_name}' for exp_name, run_name in zip(EXPERIMENT_NAMES, RUN_NAMES)])
    else:
        pstr = 'No training queue defined.'
    print_(pstr)


#####===== Main Functions =====#####
def run_training(experiment_name: str, run_name: str, checkpoint_name: str, training_id: int, log: bool = False, debug: bool = False):
    train_func = TRAINING_FUNCTIONS[training_id]
    train_func(experiment_name, run_name, checkpoint_name, log, debug)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiment', type=str, default=None, help='Experiment name')
    parser.add_argument('-r','--run', type=str, default=None, help='Run name')
    parser.add_argument('-c','--checkpoint', type=str, default=None, help='Checkpoint name to resume the training from')
    parser.add_argument('-t','--training_id', type=int, default=0, help='Training ID: 0: baseline')
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
                run_training(exp_name, run_name, args.checkpoint, args.training_id, args.log, args.debug)
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
        
        run_training(exp_name, run_name, args.checkpoint, args.training_id, args.log, args.debug)

if __name__ == '__main__':
    main()


