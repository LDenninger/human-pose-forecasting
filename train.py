import torch
import argparse
import os

from src import TrainerBaseline
from src.utils import print_

#####===== Training Functions =====#####
# Functions to initialize a trainer and run a training.

def run_training_00(experiment_name: str, run_name: str, checkpoint_name: str, log: bool):
    """
        Run a training using the baseline trainer.
    """
    # Initialize the trainer
    trainer = TrainerBaseline(experiment_name, run_name, log_process_external=log, num_threads=0)
    # Log some information
    log_script_setup()
    # Initialize the model
    trainer.initialize_model()
    # Initialize the components of the optimization process (optimizer, scheduler, loss function)
    trainer.initialize_optimization()
    # Load the data
    trainer.load_data()
    # Train the model
    import ipdb; ipdb.set_trace()
    trainer.train()

#####===== Run Information =====#####
# These list of runs can be used to run multiple trainings sequentially.
QUEUED = False
EXPERIMENT_NAMES = []
RUN_NAMES = []


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
def run_training(experiment_name: str, run_name: str, checkpoint_name: str, training_id: int, log: bool = False):
    train_func = TRAINING_FUNCTIONS[training_id]
    train_func(experiment_name, run_name, checkpoint_name, log)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiment', type=str, default=None, help='Experiment name')
    parser.add_argument('-r','--run', type=str, default=None, help='Run name')
    parser.add_argument('-c','--checkpoint', type=str, default=None, help='Checkpoint name to resume the training from')
    parser.add_argument('-t','--training_id', type=int, default=0, help='Training ID: 0: baseline')
    parser.add_argument('--log', action='store_true', default=False, help='Log the training process to WandB')
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
            for i in range(EXPERIMENT_NAMES):
                exp_name = EXPERIMENT_NAMES[i]
                run_name = RUN_NAMES[i]
                run_training(exp_name, run_name, args.checkpoint, args.training_id, args.log)
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
        
        run_training(exp_name, run_name, args.checkpoint, args.training_id, args.log)

if __name__ == '__main__':
    main()

