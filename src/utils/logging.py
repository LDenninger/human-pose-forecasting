"""
    This module capsules the logging and complete communication with the WandB API.
    It can be used for either logging or load previously logged data.

    Some parts of the logging module were adapted from: https://github.com/angelvillar96/TemplaTorch

    Author: Luis Denninger <l_denninger@uni-bonn.de>

"""
import wandb
wandb.login()
import numpy as np
import json
import torch
import torch.nn as nn
import torchvision
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union, Literal
import traceback
from datetime import datetime
import os
import git
from pathlib import Path as P


#####===== Logging Decorators =====#####

def log_function(func):
    """
        Decorator to catch a function in case of an exception and writing the output to a log file.
    """
    def try_call_log(*args, **kwargs):
        """
            Calling the function but calling the logger in case an exception is raised.
        """
        try:
            if(LOGGER is not None):
                message = f"Calling: {func.__name__}..."
                LOGGER.log_info(message=message, message_type="info")
            return func(*args, **kwargs)
        except Exception as e:
            if(LOGGER is None):
                raise e
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()
    return try_call_log

def for_all_methods(decorator):
    """
        Decorator that applies a decorator to all methods inside a class.
    """
    def decorate(cls):
        for attr in cls.__dict__:  # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

def emergency_save(f):
    """
    Decorator for saving a model in case of exception, either from code or triggered.
    Use for decorating the training loop:
        @setup_model.emergency_save
        def train_loop(self):
    """

    def try_call_except(*args, **kwargs):
        """ Wrapping function and saving checkpoint in case of exception """
        try:
            return f(*args, **kwargs)
        except (Exception, KeyboardInterrupt):
            print_("There has been an exception. Saving emergency checkpoint...")
            self_ = args[0]
            if hasattr(self_, "model") and hasattr(self_, "optimizer"):
                fname = f"emergency_checkpoint_epoch_{self_.epoch}.pth"
                save_checkpoint(
                    model=self_.model,
                    optimizer=self_.optimizer,
                    scheduler=self_.scheduler,
                    epoch=self_.epoch,
                    exp_path=self_.exp_path,
                    savedir="models",
                    savename=fname
                )
                print_(f"  --> Saved emergency checkpoint {fname}")
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()

    return try_call_except

def print_(message, message_type="info", file_name: str=None, path_type: Literal['log', 'plot', 'checkpoint', 'visualization'] = None):
    """
    Overloads the print method so that the message is written both in logs file and console
    """
    print(message)
    if(LOGGER is not None):
        if file_name is None:
            LOGGER.log_info(message, message_type)
        elif file_name is not None and path_type is not None:
            LOGGER.log_to_file(message, file_name, path_type)
    return


def log_info(message, message_type="info"):
    if(LOGGER is not None):
        LOGGER.log_info(message, message_type)
    return

def get_current_git_hash():
    """ Obtaining the hexadecimal last commited git hash """
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except:
        print("Current codebase does not take part of a Git project...")
        sha = None
    return sha

#####===== Logging Functions =====#####

@log_function
def log_architecture(model: nn.Module, save_path: str):
    """
    Printing architecture modules into a txt file
    """
    assert save_path[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"

    # getting all_params
    with open(save_path, "w") as f:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Params: {num_params}")

    for i, layer in enumerate(model.children()):
        if(isinstance(layer, torch.nn.Module)):
            log_module(module=layer, save_path=save_path)
    return


def log_module(module, save_path, append=True):
    """
    Printing architecture modules into a txt file
    """
    assert save_path[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"

    # writing from scratch or appending to existing file
    if (append is False):
        with open(save_path, "w") as f:
            f.write("")
    else:
        with open(save_path, "a") as f:
            f.write("\n\n")

    # writing info
    with open(save_path, "a") as f:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        f.write(f"Params: {num_params}")
        f.write("\n")
        f.write(str(module))
    return

@log_function
def save_checkpoint(model, optimizer, scheduler, epoch, save_path, finished=False, savename=None):
    """
    Saving a checkpoint in the models directory of the experiment. This checkpoint
    contains state_dicts for the mode, optimizer and lr_scheduler
    Args:
    -----
    model: torch Module
        model to be saved to a .pth file
    optimizer, scheduler: torch Optim
        modules corresponding to the parameter optimizer and lr-scheduler
    epoch: integer
        current epoch number
    exp_path: string
        path to the root directory of the experiment
    finished: boolean
        if True, current checkpoint corresponds to the finally trained model
    """

    if(savename is not None):
        checkpoint_name = savename
    elif(savename is None and finished is True):
        checkpoint_name = "checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"

    savepath = os.path.join(save_path, checkpoint_name)

    scheduler_data = "" if scheduler is None else scheduler.state_dict()
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scheduler_state_dict": scheduler_data
            }, savepath)
    print_(f'Checkpoint was saved to: {savepath}')

    return


#####===== Logger Module =====#####

class Logger(object):



    def __init__(self, 
                  exp_name: Optional[str] = None,
                   run_name: Optional[str] = None,
                    log_to_file: Optional[bool]=True,
                     log_file_name: Optional[str] = 'log.txt',
                      log_internal: Optional[bool] = False,
                       plot_path: Optional[str] = None,
                        checkpoint_path: Optional[str] = None,
                         log_path: Optional[str] = None,
                          visualization_path: Optional[str] = None) -> None:
        """
            Initialize the logger.

            Arguments:
                exp_name [str]: Name of the experiment within the local project.
                run_name [str]: Name of the run within the experiment.
                log_file_name Optional[str]: Name of the log file.
        """
        if exp_name is not None and run_name is not None:
            self.initialize(exp_name, run_name,log_to_file, log_file_name, log_internal, plot_path, checkpoint_path,log_path,visualization_path)
            self.run_initialized = True
        else:
            self.run_initialized = False            


    def initialize(self,
                    exp_name: str,
                     run_name: str,
                      log_to_file: Optional[bool]=True,
                       log_file_name: Optional[str] = 'log.txt',
                        log_internal: Optional[bool] = False,
                         plot_path: Optional[str] = None,
                          checkpoint_path: Optional[str] = None,
                           log_path: Optional[str] = None,
                            visualization_path: Optional[str] = None) -> None:
        ##-- Experiment Meta Data --##
        self.exp_name = exp_name
        self.run_name = run_name
        self.run_path = P('experiments') / self.exp_name / self.run_name
        ##-- Logging Parameters --##
        self.log_to_file_activated = log_to_file
        self.log_internal_activated = log_internal
        self.log_external = False
        self._internal_log_dir = {}
        ##-- Logging Paths --##
        self.plot_path = self.run_path / "plots" if plot_path is None else plot_path
        self.log_path = self.run_path / "logs" if log_path is None else log_path
        self.log_file_path = self.log_path / log_file_name
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
        self.vis_path = self.run_path / "visualizations" if visualization_path is None else visualization_path
        self.checkpoint_path = self.run_path / "checkpoints" if checkpoint_path is None else checkpoint_path
        self.run_initialized = True

    def initialize_logging(self,
                              project_name: str,
                               entity: Optional[str]=None,
                                config: Optional[Union[Dict, str, None]]=None,
                                 group: Optional[str]=None,
                                  job_type: Optional[str]=None,
                                   resume: Optional[Literal['allow','must','never','auto',None]]= None,
                                    mode: Optional[Literal['offline','online','disabled']]='online'
                            ) -> None:
        """
            Initialize logging using WandB.

            Arguments:
                project_name [str]: Name of the project within WandB.
                entity Optional[str]: Name of the group or user to record the data to.
                config Optional[Union[Dict, str, None]]: Configuration of the run.
                group Optional[str]: Name of the group of runs to group runs within a project.
                job_type Optional[str]: Type of the job to give extra information about the run.
                resume Optional[Literal['allow','must','never','auto',None]]: Option to resume a previously logged run.
                mode Optional[Literal['offline','online','disabled']]: Flag to determine if and where data is logged to.
        
        """
        self.run = wandb.init(
                    project=project_name,
                    name=(self.exp_name+'/'+self.run_name),
                    entity=entity,
                    config=config,
                    group=group,
                    job_type=job_type,
                    resume=resume,
                    mode=mode,
        )
        self.log_external = True
        log_str = 'WandB initialized\n'
        log_str += f'project_name: {project_name}, entity: {entity}, group: {group}, job_type: {job_type}, resume: {resume}, mode: {mode}'
        self.log_info(log_str, message_type="info")
        self.log_config(config)

    def finish_logging(self) -> None:
        """
            Finish logging using WandB.
        """
        if not self.run_initialized:
            return
        wandb.finish()
        self.log_external = False
        log_str = 'WandB logging finished\n'
        print_(log_str, message_type="info")
    
    def watch_model(self, model: torch.nn.Module, log: Optional[Literal['gradients', 'parameters', 'all']] = "gradients"):
        if not self.log_external:
            print_('Cannot watch model without WandB', 'warn')
            return
        self.run.watch(
            model,
            log=log
        )

    ###--- Data Retrieval Functions ---###

    def get_path(self, name: Optional[Literal['log', 'plot', 'checkpoint', 'visualization']] = None) -> str:
        """
            Get the path to the specified directory.

            Arguments:
                name Optional[Literal['log', 'plot', 'checkpoint', 'visualization']]: Type of the directory to retrieve.
                    If None is provided, the run path is returned.


        """
        if not self.run_initialized:
            return
        assert name in ['log', 'plot', 'checkpoint', 'visualization'], "Please provide a valid directory type"
        if name is None:
            return self.run_path
        if name == 'log':
            return self.log_path
        elif name == 'plot':
            return self.plot_path
        elif name == 'checkpoint':
            return self.checkpoint_path
        elif name == 'visualization':
            return self.vis_path

    def get_internal_log(self, name: Optional[str]=None) -> Union[Dict, list, None]:
        """
            Get logs from the internally saved dictionary.

            Arguments:
                name Optional[str]: Name of the metric to retrieve. If not provided, all logs are retrieved.
            
        """
        if not self.run_initialized:
            return
        if name is not None:
            if name in self._internal_log_dir.keys():
                return self._internal_log_dir[name]
            else:
                self.log_info(f"Internal log {name} not found", message_type='warning')
                return None
        return self._internal_log_dir
        
    ###--- External Logging Functions ---###
    # These functions log data to the WandB server.

    def log(self, data: Dict[str, Any], step: Optional[int]=None) -> bool:
        """
            Log data to WandB.

            Arguments:
                data [Dict[str, Any]]: Data to log of form: {metric_name: value, metric_name2: value2,...}

        """
        if not self.run_initialized:
            return False
        
        if self.log_external:
            try:
                wandb.log(data, step)
            except Exception as e:
                print('Logging failed: ', e)
                return False
        return True
    
    def log_image(self, name: str, image: Union[torch.Tensor, np.array], step: Optional[int]=None) -> None:
        """
            Log images to WandB.
        """
        # import ipdb; ipdb.set_trace()
        if not self.run_initialized or not self.log_external:
            return
        assert len(image.shape) in [3, 4], "Please provide images of shape [H, W, C], [B, H, W, C], [C, H, W] or [B, C, H, W]"
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        if image.shape[-1] not in [1, 3]:
            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            elif len(image.shape) == 4:
                image = np.transpose(image, (0, 2, 3, 1))
        try:
            torchvision.utils.save_image(image, self.vis_path / f"{name}.png")
        except: 
            print_(f"Failed to save image {name} to {self.vis_path}.")
        wandbImage = wandb.Image(image)
        wandb.log({name: wandbImage}, step=step)
    
    def log_segmentation_image(self, name: str,
                  image: Union[torch.Tensor, np.array],
                   segmentation: Optional[Union[torch.Tensor, np.array]],
                    ground_truth_segmentation: Optional[Union[torch.Tensor, np.array]]=None,
                     class_labels: Optional[list] = None,
                      step: Optional[int]=None) -> None:
        """
            Log a segmentation image to WandB.

            Arguments:
                image [Union[torch.Tensor, np.array]]: Image to log.

        """
        if not self.run_initialized or not self.log_external:
            return
        assert len(image.shape) in [3, 4], "Please provide images of shape [H, W, C], [B, H, W, C], [C, H, W] or [B, C, H, W]"
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        if image.shape[-1] not in [1, 3]:
            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            elif len(image.shape) == 4:
                image = np.transpose(image, (0, 2, 3, 1))
        if torch.is_tensor(segmentation):
            segmentation = segmentation.detach().cpu().numpy()
        if ground_truth_segmentation is not None:
            if class_labels is not None:
                wandbImage = wandb.Image(image, masks={
                    "predictions": {
                        "mask_data": segmentation,
                        "class_labels": class_labels
                    },
                    "ground_truth": {
                        "mask_data": ground_truth_segmentation,
                        "class_labels": class_labels
                    }
                    })
            else:
                wandbImage = wandb.Image(image, masks={
                    "predictions": {
                        "mask_data": segmentation,
                    },
                    "ground_truth": {
                        "mask_data": ground_truth_segmentation,
                    }
                    })
        else:
            if class_labels is not None:
                wandbImage = wandb.Image(image, masks={
                        "predictions": {
                            "mask_data": segmentation,
                            "class_labels": class_labels
                        }})
            else:
                wandbImage = wandb.Image(image, masks={
                        "predictions": {
                            "mask_data": segmentation,
                        }})              
        wandb.log({name: wandbImage}, step=step)


    ###--- Internal Logging Functions ---###
    # These functions save data to the internal storage and to the local disk.

    def log_internal(self, data: Dict[str, Any]):
        if not self.run_initialized:
            return
        if self.log_internal_activated:
            for key, value in data.items():
                if key not in self._internal_log_dir.keys():
                    self._internal_log_dir[key] = []
                self._internal_log_dir[key].append(value)

    def log_info(self, message: str, message_type: str='info') -> None:
        if not self.run_initialized:
            return
        cur_time = self._get_datetime()
        msg_str = f'{cur_time}   [{message_type}]: {message}\n'
        with open(self.log_file_path, 'a') as f:
            f.write(msg_str)            

    def log_to_file(self, message: str, file_name: str, type: Literal['log', 'plot', 'checkpoint', 'visualization']) -> None:
        """
            Log a message to a specific file within the run directory.
        """
        path = self.get_path(type)
        with open(path / f"{file_name}.txt", 'a') as f:
            f.write(f'{message}\n')

    def log_config(self, config: Dict[str, Any]) -> None:
        """
            Log configuration to the log file.
        """
        if not self.run_initialized:
            return
        cur_time = self._get_datetime()
        msg_str = f'{cur_time}   [config]:\n'
        msg_str += '\n'.join([f'  {k}: {v}' for k,v in config.items()])

        with open(self.log_file_path, 'a') as f:
            f.write(msg_str)
    
    def log_git_hash(self) -> None:
        if not self.run_initialized:
            return
        hash = get_current_git_hash()
        self.log_info(f'git hash: {hash}')

    def log_architecture(self, model: nn.Module) -> None:
        if not self.run_initialized:
            return
        save_path = P(self.log_file_path) / "architecture.txt"
        log_architecture(model=model, save_path=save_path)

    def save_checkpoint(self, model: nn.Module, optimizer, scheduler, epoch: int, finished: bool = False) -> None:
        if not self.run_initialized:
            return
        save_path = P(self.run_path) / "checkpoints"
        save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, save_path=save_path, finished=finished)

    def dump_dictionary(self, data: Dict[str, Any], file_name: str):
        if not self.run_initialized:
            return
        save_path = P(self.log_path) / file_name
        with open(save_path, 'w') as f:
            json.dump(data, f)

    def _get_datetime(self) -> str:
        return datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    
#####===== Metric Tracker =====#####

class MetricTracker:
    def __init__(self):
        self.metrics = {}
        self.epoch = 1
        self.iteration = 1

    def step_iteration(self):
        self.iteration += 1

    def step_epoch(self):
        self.epoch += 1

    def log(self, metric_name: str, metric_value: float):
        """ Log a metric value"""
        if metric_name not in self.metrics.keys():
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(metric_value)

    def get_metric(self, metric_name: Optional[str] = None):
        if metric_name is None:
            return self.metrics
        elif metric_name in self.metrics.keys():
            return self.metrics[metric_name]
        else:
            print_('Tried to fetch non-existing from MetricTracker', 'warn')
            return None

    def get_mean(self, metric_name: str = None) -> float:
        """ Get the mean value of a metric """
        if metric_name is None:
            return {key: np.mean(values) for key, values in self.metrics.items()}
        if metric_name not in self.metrics.keys():
            print_(f'MetricTracker received an invalid metric name for retrieval {metric_name}')
            return 0
        return np.mean(self.metrics[metric_name])

    def get_variance(self, metric_name: str = None) -> float:
        """ Get the variance value of a metric """
        if metric_name is None:
            return {key: np.var(values) for key, values in self.metrics.items()}
        if metric_name not in self.metrics.keys():
            print_(f'MetricTracker received an invalid metric name for retrieval {metric_name}')
            return 0
        return np.var(self.metrics[metric_name])
    
    def get_median(self, metric_name: str = None) -> float:
        """ Get the median value of a metric """
        if metric_name is None:
            return {key: np.median(values) for key, values in self.metrics.items()}
        if metric_name not in self.metrics.keys():
            print_(f'MetricTracker received an invalid metric name for retrieval {metric_name}')
            return 0
        return np.median(self.metrics[metric_name])

    def reset(self, metric_name: str = None):
        """
            Reset a metric. If no metric name is provided, all metrics are resetted.
        """
        if metric_name is not None:
            self.metrics[metric_name] = []
        else:
            self.metrics = {}
    

#####===== Global Logger =====#####
# This logger should be used throughout the project

LOGGER = Logger()