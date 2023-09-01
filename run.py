import argparse
import os

import src.utils as utils

from src.tests import *


###--- Run Information ---###
# These list of runs can be used to run multiple trainings sequentially.

EXPERIMENT_NAMES = []
RUN_NAMES = []
EVALUATION_METRICS = []
EPOCHS = []



###--- Training ---###

###--- Evaluation ---###


###--- Hyperparameter Tuning ---###

###--- Testing ---###





if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # Flags to signal which function to run
    argparser.add_argument('--train', action='store_true', default=False, help='Train the model')
    argparser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate the model')
    argparser.add_argument('--tuning', action='store_true', default=False, help='Tune the hyperparameters')

    # Testing flags
    argparser.add_argument('--test_data', action='store_true', default=False, help='Test the model')
    argparser.add_argument('--test_sk32', action='store_true', default=False, help='Test the Skeleton32 model')
    argparser.add_argument('--test_model', action='store_true', default=False, help='Test the Skeleton32 model')
    argparser.add_argument('--test_comp', action='store_true', default=False, help='Test the high-level computation functions of the model')

    argparser.add_argument('--augm_study', action='store_true', default=False, help='Run the augmentation study')
    argparser.add_argument('--opt_study', action='store_true', default=False, help='Run the augmentation study')

    argparser.add_argument('--init_exp', action='store_true', default=False, help='Initialize a new experiment')
    argparser.add_argument('--init_run', action='store_true', default=False, help='Initialize a new run')

    argparser.add_argument('--copy_conf', action='store_true', default=False, help='Load a configuration file to run')

    argparser.add_argument('--clear_logs', action='store_true', default=False, help='Clear the logs of a given run')

    # Hyperparameters 
    argparser.add_argument('-exp', type=str, default=None, help='Experiment name')
    argparser.add_argument('-run', type=str, default=None, help='Run name')
    argparser.add_argument('-conf', type=str, default=None, help='Config name')

    # Additional parameter
    argparser.add_argument('-n', type=int, default=None)
    
    ##-- Function Calls --##
    # Here we simply determien which function to call and how to set the experiment and run name

    args = argparser.parse_args()
    # If no experiment or run name is provided, the environment variables defining these have to be set
    if args.init_exp:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ), 'Please provide an experiment name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        utils.create_experiment(exp_name)
    
    if args.init_run:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ), 'Please provide an experiment and run name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        utils.create_run(exp_name, run_name)
    
    if args.copy_conf:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ) and args.conf is not None, 'Please provide an experiment and run name and the name of the config file'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        config_name = args.conf if args.conf is not None else os.environ.get('CURRENT_CONFIG')
        utils.load_config(exp_name, run_name, config_name)
    if args.clear_logs:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ) and args.conf is not None, 'Please provide an experiment and run name and the name of the config file'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        utils.clear_logs(exp_name, run_name)

    if args.test_data:
        test_data_loading()
    if args.test_sk32:
        test_skeleton32_model()
    if args.test_model:
        test_transformer()
    if args.test_comp:
        test_processing_functions()

    if args.train:
        assert ((args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ)) or (len(EXPERIMENT_NAMES)!=0 and len(RUN_NAMES)!=0), 'Please provide an experiment and run name'
        if len(EXPERIMENT_NAMES) != 0:
            assert len(EXPERIMENT_NAMES) == len(RUN_NAMES), 'Length of experiment and run list must be the same'
            exp_name = EXPERIMENT_NAMES
            run_name = RUN_NAMES
        else:       
            exp_name = [args.exp] if args.exp is not None else [os.environ.get('CURRENT_EXP')]
            run_name = [args.run] if args.run is not None else [os.environ.get('CURRENT_RUN')]
        training(exp_name, run_name)

    if args.evaluate:
        assert ((args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ)) or (len(EXPERIMENT_NAMES)!=0 and len(RUN_NAMES)!=0), 'Please provide an experiment and run name'
        if len(EXPERIMENT_NAMES) != 0:
            assert len(EXPERIMENT_NAMES) == len(RUN_NAMES), 'Length of experiment and run list must be the same'
            exp_name = EXPERIMENT_NAMES
            run_name = RUN_NAMES
        else:       
            exp_name = [args.exp] if args.exp is not None else [os.environ.get('CURRENT_EXP')]
            run_name = [args.run] if args.run is not None else [os.environ.get('CURRENT_RUN')]
        evaluation(exp_name, run_name)
    