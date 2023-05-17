import argparse
import os

import torch
import copy
import optuna

import utils

import torchgadgets as tg

from prettytable import PrettyTable

import matplotlib.pyplot as plt

import seaborn as sn


###--- Run Information ---###
# These list of runs can be used to run multiple trainings sequentially.

#EXPERIMENT_NAMES = ['convnext_large']*3
#RUN_NAMES = ['norm_class', 'large_class', 'large_bn_class']

EXPERIMENT_NAMES = ['convnext_large']
RUN_NAMES = ['final_2']
EVALUATION_METRICS = ['accuracy', 'accuracy_top3', 'accuracy_top5', 'confusion_matrix', 'f1', 'recall', 'precision']
EPOCHS = [20]



###--- Training ---###
# This is the function used for training all our experiments.
# The experiments are structured into the "/experiments" directory, where all TensorBoard and PyTorch files can be found

def training(exp_names, run_names):
    assert len(exp_names) == len(run_names)
    for exp_name, run_name in zip(exp_names, run_names):
        ##-- Load Config --##
        # Load the config from the run directory
        # All interactions with the experiments directory should be performed via the utils package

        # Config for augmentation whe nthe dataset is initially loaded, in our case only random cropping
        load_augm_config_train = utils.load_config('augm_train_preLoad') 
        load_augm_config_test = utils.load_config('augm_test_preLoad')

        # Load config for the model
        config = utils.load_config_from_run(exp_name, run_name)
        config['num_iterations'] = config['dataset']['train_size'] // config['batch_size']

        tg.tools.set_random_seed(config['random_seed'])
        ##-- Load Dataset --##
        # Simply load the dataset using TorchGadgets and define our dataset to apply the initial augmentations
        data = tg.data.load_dataset('oxfordpet')
        train_dataset = data['train_dataset']
        test_dataset = data['test_dataset']
        train_dataset = tg.data.ImageDataset(dataset=train_dataset, transforms=load_augm_config_train)
        test_dataset = tg.data.ImageDataset(dataset=test_dataset, transforms=load_augm_config_test, train_set=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
        ##-- Logging --##
        # Directory of the run that we write our logs to
        log_dir = os.path.join(os.getcwd(),'experiments', exp_name, run_name, 'logs')
        checkpoint_dir = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'checkpoints')

        # Explicitely define logger to enable TensorBoard logging and setting the log directory
        logger = tg.logging.Logger(log_dir=log_dir, checkpoint_dir=checkpoint_dir, model_config=config, save_internal=True)
        
        tg.training.trainNN(config=config, logger=logger, train_loader=train_loader, test_loader=test_loader, return_all=False)



###--- Evaluation ---###
# This is the function used for evaluation.
# The different implementations of the evaluation metrics can be found in the TorchGadgets package
# The structure of this function is close to the training function.

def evaluation(exp_names, run_names, verbose=True):
    assert len(exp_names)==len(run_names)
    
    # Verbose output
    if verbose:
        print(f'\n======---- Evaluation ----======')
        print(f' Experiment Names: {exp_names}')
        print(f' Run Names: {run_names}')
        print(f' Evluation Metrics: {EVALUATION_METRICS}\n')
        table = PrettyTable()
        field_names = ['Exp. Name', 'Run Name']
        for m in EVALUATION_METRICS:
            if m!='confusion_matrix':
                field_names.append(m)
        table.field_names = field_names

    for i, (exp_name, run_name) in enumerate(zip(exp_names, run_names)):
        ##-- Load Config --##
        load_augm_config_train = utils.load_config('augm_train_preLoad')
        load_augm_config_test = utils.load_config('augm_test_preLoad')
        config = utils.load_config_from_run(exp_name, run_name)
        config['num_iterations'] = config['dataset']['train_size'] // config['batch_size']

        tg.tools.set_random_seed(config['random_seed'])
        ##-- Load Dataset --##
        data = tg.data.load_dataset('oxfordpet')
        train_dataset = data['train_dataset']
        test_dataset = data['test_dataset']
        train_dataset = tg.data.ImageDataset(dataset=train_dataset, transforms=load_augm_config_train)
        test_dataset = tg.data.ImageDataset(dataset=test_dataset, transforms=load_augm_config_test, train_set=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True, num_workers=8)
        ##-- Logging --##
        log_dir = os.path.join(os.getcwd(),'experiments', exp_name, run_name, 'logs')
        checkpoint_dir = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'checkpoints')
        vis_dir = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'visualizations')
        logger = tg.logging.Logger(log_dir=log_dir, checkpoint_dir=checkpoint_dir, model_config=config, save_internal=True)
        data_augmentor = tg.data.ImageDataAugmentor(config['pre_processing'])

        ##-- Model Loading --##
        # Load the weights from the checkpoint and initialize the model
        model = tg.models.NeuralNetwork(config['layers'])
        utils.load_model_from_checkpoint(exp_name, run_name, model, EPOCHS[i])

        ##-- Run Evaluation --##
        evaluation_result = tg.evaluation.run_evaluation(model, data_augmentor, test_loader, config, evaluation_metrics=EVALUATION_METRICS)

        # Extract data and set a prefix to not confuse the logged data with data logged during the training
        data = {}
        for k, v in evaluation_result.items():
            if k!='confusion_matrix':
                data[('evaluation/'+k)] = v
        # Log data
        logger.log_data(0, data)

        conf_dir = os.path.join(vis_dir, 'confusion_matrix.png')
        fig, ax = plt.subplots(figsize=(18,16))
        sn.heatmap(evaluation_result['confusion_matrix'][0], annot=True, linewidths=.5, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        fig.savefig(conf_dir)

        if verbose:
            data = [m[0] for k, m in data.items()]
            data = [exp_name, run_name] + data
            table.add_row(data)
    if verbose:
        print(table)
        
    




###--- Hyperparameter Tuning ---###
# These are the functions used for conducting Optuna studies.

LEARNING_RATE_RANGE = (7e-06, 1e-03)
DECAY_FACTOR_RANGE = (0.6, 1.0)
BATCH_SIZES = [16,32,64]

##-- Training Parameter Study --##
# Optimize the learning rate, decay factor and batch size

def optimization_study(exp_name, run_name, study_name, n_trials):

    def objective(trial):

        # Sample training parameter between the boundaries
        learning_rate = trial.suggest_float('learning_rate', LEARNING_RATE_RANGE[0], LEARNING_RATE_RANGE[1])
        decay_factor = trial.suggest_float('decay_factor', DECAY_FACTOR_RANGE[0], DECAY_FACTOR_RANGE[1])
        batch_size = trial.suggest_categorical('batch_size',  BATCH_SIZES)
        # Copy config
        config_run = copy.deepcopy(config)

        # Apply sampled training parameters to the config
        config_run['learning_rate'] = learning_rate
        config_run['batch_size'] = batch_size
        config_run['scheduler']['epoch_scheduler']['gamma'] = decay_factor
        config['num_iterations'] = config['dataset']['train_size'] // config['batch_size']

        # Initialize data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True, num_workers=8)

        # Run a full training with evaluation while reporting the score to optuna
        score = tg.training.optimizeNN(config_run, trial, train_loader=train_loader, test_loader=test_loader)

        return score


    ##-- Load Config --##
    load_augm_config_train = utils.load_config('augm_train_preLoad')
    load_augm_config_test = utils.load_config('augm_test_preLoad')
    config = utils.load_config_from_run(exp_name, run_name)
    config['num_iterations'] = config['dataset']['train_size'] // config['batch_size']
    ##-- Load Dataset --##
    data = tg.data.load_dataset('oxfordpet')
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']
    train_dataset = tg.data.ImageDataset(dataset=train_dataset, transforms=load_augm_config_train)
    test_dataset = tg.data.ImageDataset(dataset=test_dataset, transforms=load_augm_config_test, train_set=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True, num_workers=8)


    if study_name is None:
        study_name = 'opt_study'
    
    log_dir = 'sqlite:///' + os.path.join(os.getcwd(),'experiments', exp_name, run_name, 'logs', study_name+'.db')

    study = optuna.create_study(study_name=study_name,
                                    storage=log_dir, load_if_exists=True,
                                        direction='maximize',
                                            sampler = optuna.samplers.TPESampler(), 
                                                pruner = optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials)

##-- Data Augmentation Study --##
# This is a further study we did not use at the end.
# This part can be skipped.

COLOR_JITTER_CONFIG = { "brightness": 0.4, "contrast": 0.4, "saturation": 0.3, "hue": 0.0, "train": True, "eval": False}
RANDOM_ROTATION_CONFIG =    {"type": "random_rotation", "degrees": 45, "train": True, "eval": False}
RANDOM_HORIZONTAL_FLIP = {"type": "random_horizontal_flip", "prob": 0.5, "train": True, "eval": False}
GAUSSIAN_BLUR_CONFIG = {"type": "gaussian_blur", "kernel_size": [5, 5], "sigma": [0.1, 2.0], "train": True, "eval": False}
NORMALIZE_CONFIG =    {"type": "normalize", "train": True, "eval": True}


def data_augmentation_study(exp_name, run_name, study_name=None, n_trials=20):

    def objective(trial):
        data_augmentation = []

        if trial.suggest_categorical('random_horizontal_flip'):
            data_augmentation.append(RANDOM_HORIZONTAL_FLIP)

        if trial.suggest_categorical('random_rotation', [True, False]):
            data_augmentation.append(RANDOM_ROTATION_CONFIG)

        if trial.suggest_categorical('color_jitter', [True, False]):
            data_augmentation.append(COLOR_JITTER_CONFIG)
        
        if trial.suggest_categorical('gaussian_blur', [True, False]):
            data_augmentation.append(GAUSSIAN_BLUR_CONFIG)

        data_augmentation.append(NORMALIZE_CONFIG)

        config['pre_processing'] = data_augmentation

        score = tg.training.optimizeNN(config, trial, train_loader=train_loader, test_loader=test_loader)

        return score


    

    ##-- Load Config --##
    load_augm_config_train = utils.load_config('augm_train_preLoad')
    load_augm_config_test = utils.load_config('augm_test_preLoad')
    config = utils.load_config_from_run(exp_name, run_name)
    config['num_iterations'] = config['dataset']['train_size'] // config['batch_size']
    ##-- Load Dataset --##
    data = tg.data.load_dataset('oxfordpet')
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']
    train_dataset = tg.data.ImageDataset(dataset=train_dataset, transforms=load_augm_config_train)
    test_dataset = tg.data.ImageDataset(dataset=test_dataset, transforms=load_augm_config_test, train_set=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True, num_workers=8)


    if study_name is None:
        study_name = 'data_augm'
    
    log_dir = 'sqlite:///' + os.path.join(os.getcwd(),'experiments', exp_name, run_name, 'logs', study_name+'.db')

    study = optuna.create_study(study_name=study_name,
                                    storage=log_dir, load_if_exists=True,
                                        direction='maximize',
                                            sampler = optuna.samplers.TPESampler(), 
                                                pruner = optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials)
    



def hyperparameter_tuning(exp_name, run_name):
    print("Hyperparameter tuning not implemented...")
    return




if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # Flags to signal which function to run
    argparser.add_argument('--train', action='store_true', default=False, help='Train the model')
    argparser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate the model')
    argparser.add_argument('--tuning', action='store_true', default=False, help='Tune the hyperparameters')

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
    
    if args.opt_study:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ), 'Please provide an experiment and run name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        optimization_study(exp_name, run_name, study_name='opt_study', n_trials=args.n)
    
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

    if args.tuning:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ), 'Please provide an experiment and run name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        hyperparameter_tuning(exp_name, run_name)

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
    

