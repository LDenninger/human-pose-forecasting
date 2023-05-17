import numpy as np
import torch

###--- Custom Scheduler Class ---###
# Right now it only includes a custom scheduler
# It should eventually be a Interface for all other PyTorch schedulers 
# such that custom and standard schedulers can be mixed arbitrarly with the training functions

class SchedulerManager:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.scheduler_config = config['scheduler']

        self.epoch_scheduler = None
        self.iteration_scheduler = None
        self.warmup_scheduler = None

        self._num_iterations = config['num_iterations']
        self._num_epoch = config['num_epochs']
        self._base_lr = config['learning_rate']
        self._epoch = 0
        self._iteration = 0
        self._init_scheduler()
    
    def step(self, step: int):
        if step == 1:
            if self.epoch_scheduler is not None:
                self.epoch_scheduler.step()
            self._epoch += 1
        self._iteration = step
        if self.iteration_scheduler is not None:
            self.iteration_scheduler.step()

    def get_last_lr(self):
        if self.iteration_scheduler is not None:
            return self.iteration_scheduler.get_last_lr()
        if self.epoch_scheduler is not None:
            return self.epoch_scheduler.get_last_lr()
        return self._base_lr

    def _init_scheduler(self):
        if self.scheduler_config['epoch_scheduler'] is not None:
            if self.scheduler_config['epoch_scheduler']['type']=='warmup_cosine_decay':
                self.epoch_scheduler = WarmUpCosineDecayLR(self.optimizer, self.scheduler_config['epoch_scheduler']['warmup_steps'], self._num_epoch, self._base_lr)
            elif self.scheduler_config['epoch_scheduler']['type']=='lambda':
                self.epoch_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,self.scheduler_config['epoch_scheduler']['lambda'])
            elif self.scheduler_config['epoch_scheduler']['type']=='step':
                self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.scheduler_config['epoch_scheduler']['step_size'], self.scheduler_config['epoch_scheduler']['gamma'])
            elif self.scheduler_config['epoch_scheduler']['type']=='exponential':
                self.epoch_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.scheduler_config['epoch_scheduler']['gamma'])
            
        if self.scheduler_config['iteration_scheduler'] is not None:
            if self.scheduler_config['iteration_scheduler']['type']=='warmup_cosine_decay':
                self.iteration_scheduler = WarmUpCosineDecayLR(self.optimizer, self.scheduler_config['iteration_scheduler']['warmup_steps'], self._num_iterations, self._base_lr)
            elif self.scheduler_config['epoch_scheduler']['type']=='lambda':
                self.iteration_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,self.scheduler_config['epoch_scheduler']['lambda'])
            elif self.scheduler_config['epoch_scheduler']['type']=='step':
                self.iteration_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.scheduler_config['epoch_scheduler']['step_size'], self.scheduler_config['epoch_scheduler']['gamma'])
            elif self.scheduler_config['epoch_scheduler']['type']=='exponential':
                self.iteration_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.scheduler_config['epoch_scheduler']['gamma'])
                

class  WarmUpCosineDecayLR():
    def __init__(self, optimizer, warmup_steps, num_steps, base_lr) -> None:
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.base_lr = base_lr
        self.last_lr = base_lr
        self._step = 0
        self.optimizer = optimizer
        self.set_learning_rate(self._warmup_cosine_decay(1))

    def step(self):
        self._step += 1
        learning_rate = self._warmup_cosine_decay(self._step)
        self.set_learning_rate(learning_rate)

    def set_learning_rate(self, learning_rate):
        self.last_lr = learning_rate
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate

    def set_base_lr(self, base_lr):
        self.base_lr = base_lr

    def get_last_lr(self):
        return self.last_lr

    def _warmup_cosine_decay(self, step):
        """
            Custom learning rate scheduler. In the first warmup_steps steps the learning rate is linearly increased.
            After this points is reached the learning rate cosine decays.
        
        """
        warmup_factor = min(step, self.warmup_steps) / self.warmup_steps
        decay_step = max(step - self.warmup_steps, 0) / (self.num_steps- self.warmup_steps)
        return self.base_lr * warmup_factor * (1 + np.cos(decay_step * np.pi)) / 2