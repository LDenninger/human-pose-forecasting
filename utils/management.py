import os
import shutil

import torch

import json

###--- Experiment Management Scripts ---###
# These functions handle all interaction from outside with the experiments
# We can simply initialize experiments, load and save configs, and load model checkpoints

###--- Experiments Directory ---###

def create_experiment(experiment_name):
    path = os.path.join(os.getcwd(), 'experiments', experiment_name)
    if os.path.exists(path):
        print('Experiment directory already exists')
        return -1
    os.makedirs(path)
    print(f'Experiment directory created at: {path}')
    return 1

def create_run(experiment_name, run_name):
    path = os.path.join(os.getcwd(), 'experiments', experiment_name, run_name)
    if os.path.exists(path):
        print('Run directory already exists')
        return -1
    os.makedirs(path)
    os.makedirs(os.path.join(path, 'logs'))
    os.makedirs(os.path.join(path, 'checkpoints'))
    os.makedirs(os.path.join(path, 'visualizations'))
    print(f'Run directory created at: {path}')
    return 1

def clear_run(experiment_name, run_name):
    path = os.path.join(os.getcwd(), 'experiments', experiment_name, run_name)
    if not os.path.exists(path):
        print('Run directory does not exist')
        return -1
    shutil.rmtree(path)
    os.makedirs(path)
    print(f'Run directory removed at: {path}')
    return 1

def clear_logs(experiment_name, run_name):
    path = os.path.join(os.getcwd(), 'experiments', experiment_name, run_name, 'logs')
    if not os.path.exists(path):
        print('Logs directory does not exist')
        return -1
    shutil.rmtree(path)
    os.makedirs(path)
    print(f'Logs directory removed at: {path}')
    return 1


###--- Configurations ---###

def load_config_to_run(config_name, exp_name, run_name):
    config = load_config(config_name)
    save_config(config, exp_name, run_name)

def load_config_from_run(exp_name, run_name):
    path = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'config.json')
    if not os.path.exists(path):
        print('Config file does not exist')
        return None
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def load_config(config_name):
    with open(os.path.join(os.getcwd(), 'config', config_name+".json"), 'r') as f:
        config = json.load(f)
    return config

def save_config(config, exp_name, run_name):
    with open(os.path.join(os.getcwd(), 'experiments', exp_name, run_name), 'w') as f:
        json.dump(config, f, indent=4)

def load_config_from_run(exp_name, run_name):
    path = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'config.json')
    if not os.path.exists(path):
        print('Config file does not exist')
        return None
    with open(path, 'r') as f:
        config = json.load(f)
        
    return config

###--- Checkpoint Loading ---###

def load_model_from_checkpoint(exp_name, run_name, model, epoch, optimizer=None):
    cp_dir = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'checkpoints', f'checkpoint_{epoch}.pth')
    try:
        cp_data = torch.load(cp_dir)
        model.load_state_dict(cp_data['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(cp_data['optimizer_state_dict'])
    except:
        print(f'Checkpoint could not been load from: {cp_dir}')
        return
    print(f'Model checkpoint was load from: {cp_dir}')

    
    

###--- Data ---###

def reset_data():
    if os.path.exists(os.path.join(os.getcwd(), 'data')):
        shutil.rmtree(os.path.join(os.getcwd(), 'data'))
    os.makedirs(os.path.join(os.getcwd(), 'data'))
    print('Data directory resetted')
    return 1