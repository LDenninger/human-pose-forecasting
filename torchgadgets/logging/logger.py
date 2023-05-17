import torch
from torch.utils.tensorboard import SummaryWriter

import os

from pathlib import Path as P


class Logger():
    def __init__(self,
                log_dir = None,
                checkpoint_dir = None,
                    log_name: str = None,
                        model_config: dict = None,
                            writer: SummaryWriter = None,
                            save_external: bool = True,                                    
                            save_internal: bool = False, 
                                ):
        """
            The module logs the data internally to be accessed later or externally to the disk using TensorBoard.
            It can be simply implemented in the training pipeline to handle all logging tasks.

            Arguments:
                save_dir (str): The path to save tensorboard logs.
                log_name (str): The suffix of the name of the log.
                model_config (dict): The configuration of the model.
                writer (SummaryWriter): The tensorboard writer. If None and save_external, a new SummaryWriter is created
                save_external (bool): Whether to save the logs externally to TensorBoard
                save_internal (bool): Whether to save the logs internally in a dictionary.
        """ 
        # Save directory
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        ### Run Information ###
        self.log_name = log_name
        self.writer = writer
        # Enable logging to TensorBoard
        if save_external:
            assert log_dir is not None or self.writer is not None, "Path to save tensorboard logs has to be specified..."
            if self.writer is None:
                self.writer = SummaryWriter(log_dir=self.log_dir) if log_name is None else SummaryWriter(log_dir=self.log_dir, filename_suffix=log_name)

        self.model_config = model_config

        self.save_internal = save_internal
        self.save_external = save_external
        self._internal_log = {}

        self.log_gradients = False
        
    def log_data(self, epoch: int, data: dict, model = None, optimizer=None, iteration: int=None):
        """
            Log the data.
            Arguments:
                epoch (int): The current epoch. 
                data (dict): The data to log.
                    Format:
                            {
                                [name]: value,
                                ...
                            }
        
        """
        # Save on disk using TensorBoard
        if self.save_external:
            for key, value in data.items():
                # If we get a nested list but the the nested list contains only a single element we simply unpack this element.
                if type(value) == list and len(value) == 1:
                    value = value[0]
                # If we have a list of values as a value in the data dictionary, we assume this comes from a metric logged iteration-wise
                # We distringuish between metrics logged epoch-wise and iteration-wise when saved with TensorBoard
                if type(value) == list:
                    prefix_name = f'iteration_metrics/'
                    log_iter = self.model_config['num_iterations']*(epoch-1)
                    for i, item in enumerate(value):
                        self.writer.add_scalar(prefix_name + key, item, log_iter + i +1)
                    continue
                # Set prefix to log the metric either epoch-wise or iteration-wise
                prefix_name = f'epoch_metrics/' if iteration is None else f'iteration_metrics/'
                log_iter = epoch if iteration is None else (self.model_config['num_iterations']*(epoch-1) + iteration)
                # Write to TensorBoard
                self.writer.add_scalar(prefix_name + key, value, log_iter)
            
        if self.save_internal:
            self._save_internal(data)

        if epoch >= self.model_config['save_start'] and epoch % self.model_config['save_frequency'] == 0 and model is not None and (iteration is None or iteration==self.model_config['num_iterations']):
            self._save_checkpoint(epoch, model, optimizer=optimizer)
        elif epoch == self.model_config['num_epochs'] and iteration==self.model_config['num_iterations']:
            self._save_checkpoint(epoch, model, optimizer=optimizer)
    
    def get_log(self):
        """
            Get all logs that were saved internally.
        """
        return self._internal_log

    def get_last_log(self):
        """
            Get the last entry for all internally logger metrics.
        """
        last_log = {}
        for key in self._internal_log.keys():
            last_log[key] = self._internal_log[key][-1]
        return last_log

    def enable_internal_log(self):
        self.save_internal = True
    
    def disable_internal_log(self):
        self.save_internal = False

    
    def _save_internal(self, data):
        """
            Save the data internally in the internal log dictionary. The values can be retrieved later when needed
        """
        for key, value in data.items():
            if not key in self._internal_log.keys():
                self._internal_log[key] = []
            if type(value) == list:
                self._internal_log[key] += value
                continue
            self._internal_log[key].append(value)
    
    def _save_checkpoint(self, epoch, model, optimizer=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        }, os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}.pth"))
