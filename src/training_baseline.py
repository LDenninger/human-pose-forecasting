import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from .data_utils import getDataset
from .utils import *
from .models import PosePredictor, getModel
from .evaluation import EvaluationEnginePassive

class TrainerBaseline:

    @log_function
    def __init__(self,
                  experiment_name: str,
                  run_name: str,
                  log_process_internal: Optional[bool] = False,
                  log_process_external: Optional[bool] = True,
                  num_threads: int = 2) -> None:
        # Run meta information
        self.exp_name = experiment_name
        self.run_name = run_name
        # Setup logging
        self.logger = LOGGER
        self.logger.initialize(
            exp_name=self.exp_name,
            run_name=self.run_name,
            log_internal=log_process_internal,
        )

        self.metric_tracker = MetricTracker()
        # Load the config for the run
        self.config = load_config_from_run(experiment_name, run_name)
        # Set random seed
        set_random_seed(self.config['random_seed'])
        # Setup logging to WandB if desired
        if log_process_external:
            self.logger.initialize_logging(
                project_name='HumanPoseForecasting',
                config=self.config
            )
        # Modules
        self.model = None
        self.evaluation_engine = EvaluationEnginePassive(metric_names=self.config['evaluation']['metrics'], representation=self.config['joint_representation']['type'], keep_log=True)
        self.scheduler = None
        self.optimizer = None
        self.loss = None
        self.train_loader = None
        self.test_loader = None
        self.len_train = 0
        self.len_test = 0
        self.epoch = 1
        self.iteration = 1
        # Backend configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_threads = num_threads
        print_(f"Initialized trainer for run: {experiment_name}/{run_name}")
        print_(f"Using device: {self.device}")

    ###=== Calls ===###
    @log_function
    def train(self) -> None:
        self.training_loop()

    ###=== Initialization Functions ===###
    @log_function
    def initialize_model(self):
        """
            Initialize the PosePredictor model using the loaded config.
        """
        self.model = getModel(self.config, self.device)
        self.logger.watch_model(self.model)
        print_(f"Initialized model")

    @log_function
    def initialize_optimization(self) -> bool:
        """
            Initializes the optimizer and the learning rate scheduler for training.
        """
        if self.model is None:
            print_("Cannot initialize optimization without a model.", 'error')
            return False
        self.optimizer = getOptimizer(self.config['optimizer'], self.model)
        self.scheduler = getScheduler(self.config['lr_scheduler'], self.optimizer, emb_size=self.config['model']['embedding_dim'])
        self.loss = getLoss(self.config['loss'])
        print_(f"Initialized optimizer")
        return True

    @log_function
    def load_data(self, train: bool = True, test: bool = True) -> None:
        """
            Loads the training and test split for the appropriate dataset.
        """
        if train:
            dataset = getDataset(
                self.config['dataset'],
                joint_representation = self.config['joint_representation']['type'],
                skeleton_model = self.config['skeleton']['type'],
                is_train=True
            )
            self.train_loader = DataLoader(
                dataset=dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                drop_last=True,
                num_workers=self.num_threads,
            )
            self.num_iterations = self.config['num_train_iterations'] if self.config['num_train_iterations']!=-1 else len(self.train_loader)
            p_str = f'Loaded training data: Length: {len(dataset)}, Batched length: {len(self.train_loader)}, Iterations per epoch: {self.num_iterations}'
            print_(p_str)

        if test:
            dataset = getDataset(
                self.config['dataset'],
                joint_representation = self.config['joint_representation']['type'],
                skeleton_model = self.config['skeleton']['type'],
                is_train=False
            )
            self.test_loader = DataLoader(
                dataset=dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                drop_last=True,
                num_workers=self.num_threads
            )
            self.len_test = len(self.test_loader)
            self.num_eval_iterations = self.config['num_eval_iterations'] if self.config['num_eval_iterations']!=-1 else len(self.test_loader)
            p_str = f'Loaded test data: Length: {len(dataset)}, Batched length: {len(self.test_loader)}, Iterations per epoch: {self.num_eval_iterations}'
            print_(p_str)

    @log_function
    def load_checkpoint(self, checkpoint: str):
        load_model_from_checkpoint(
            exp_name=self.exp_name,
            run_name=self.run_name,
            model=self.model,
            epoch=checkpoint,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
    
    ###=== Training Functions ===###
    @log_function
    def training_loop(self) -> bool:
        """
            Training loop for the model.
            Model, optimizer and datasets have to be initialized manually prior to training.
        """
        # Check if the trainer was properly initialized
        if self.model is None:
            print_("Cannot train without a model.", 'error')
            return False
        if self.optimizer is None:
            print_("Cannot train without an optimizer.", 'error')
            return False
        if self.scheduler is None:
            print_("Cannot train without a scheduler.", 'error')
            return False
        if self.loss is None:
            print_("Cannot train without a loss function.", 'error')
            return False
        if self.train_loader is None:
            print_("Cannot train without a training loader.", 'error')
            return False
        if self.test_loader is None:
            print_("Cannot train without a test loader.", 'error')
            return False
        
        print_(f'Start training for run {self.exp_name}/{self.run_name}', 'info')
        print_(f"Initial Evaluation:")
        self.metric_tracker.reset()
        self.evaluation_epoch()
        self._print_epoch_results()

        for self.epoch in range(1, self.config['num_epochs'] + 1):
            print_(f"Epoch {self.epoch}/{self.config['num_epochs']}", 'info')
            # Reset the tracked metrics for the new epoch
            self.metric_tracker.reset()
            # Run a single epoch of training
            self.train_epoch()
            # Evaluation in pre-defined intervals
            if self.epoch % self.config['evaluation']['frequency'] == 0:
                self.evaluation_epoch()
            # Print out the epoch results
            self._print_epoch_results()
            # Save checkpoint 
            if self.epoch % self.config['checkpoint_frequency'] == 0:
                self.logger.save_checkpoint(self.model, self.optimizer, self.scheduler, self.epoch)
        # Save the final model
        self.logger.save_checkpoint(self.model, self.optimizer, self.scheduler, self.epoch, True)
        print_('Training finished!')

    @log_function
    def train_epoch(self) -> None:
        """
            A single epoch of training.
            The training results are saved through the MetricTracker and later retrieved.
        """
        self.model.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=self.num_iterations)

        for batch_idx, data in progress_bar:
            if batch_idx==self.num_iterations:
                break
            # Load data to GPU and split into seed and target data
            seed_data, target_data = self._prepare_data(data)
            # Update the learning rate according to the schedule
            self.optimizer.zero_grad()
            self.scheduler(self.iteration)
            # Forward pass through the network
            output = self.model(seed_data)
            # Compute loss using the target data
            loss = self.loss(output, target_data)
            # Backward pass through the network
            loss.backward()
            # Update model weights
            self.optimizer.step()
            # Update all the meta information
            self.iteration += 1
            # Update the metric tracker to track the metrics of a single epoch
            self.metric_tracker.log('train_loss', loss.item())
            self.metric_tracker.step_iteration()
            # Update the logger
            self.logger.log({
                'train_loss': loss.item()
            }, step=self.iteration)
            # Update the progress bar description
            if batch_idx == 0:
                running_loss = loss.item()
            else:
                running_loss = 0.8*running_loss + 0.2*loss.item()
            progress_bar.set_description(f"Train loss: {running_loss:.4f}")

    @log_function
    def evaluation_epoch(self) -> None:
        """
            A single epoch of evaluation
        """
        self.model.eval()
        progress_bar = tqdm(enumerate(self.test_loader), total=self.num_eval_iterations)
        self.evaluation_engine.clear_log()

        for batch_idx, data in progress_bar:
            if batch_idx==self.num_eval_iterations:
                break

            # Load data to GPU and split into seed and target data
            seed_data, target_data = self._prepare_data(data)
            # Forward pass through the network
            output = self.model(seed_data)
            # Compute loss using the target data
            loss = self.loss(output, target_data)
            # Compute evaluation metrics
            # TODO: Implement evaluation object

            # Update all the meta information
            self.metric_tracker.log('test_loss', loss.item())
            self.metric_tracker.step_iteration()
            # Save output and target to the evaluation engine
            self.evaluation_engine.log(output=output[:,-1], target=target_data[:,-1])

            # Update the progress bar description
            if batch_idx == 0:
                running_loss = loss.item()
            else:
                running_loss = 0.8*running_loss + 0.2*loss.item()
            progress_bar.set_description(f"Eval loss: {running_loss:.4f}")
        # Update the logger
        eval_loss = self.metric_tracker.get_mean('test_loss')
        eval_result = self.evaluation_engine.compute()
        for name, val in eval_result.items():
            self.metric_tracker.log(name, val)
        eval_result['eval_loss'] = eval_loss
        self.logger.log(eval_result, step=self.iteration)
    
    ###=== Utility Functions ===###

    def _prepare_data(self, data: torch.Tensor) -> torch.Tensor:
        """
            Function loads the data to the GPU and splits it into the seed and target sequence.
        """
        seed_data = data[:, :self.config['dataset']['seed_length']]
        target_data = data[:, self.config['dataset']['target_length']:]
        return seed_data.to(self.device), target_data.to(self.device)
    
    def _print_epoch_results(self) -> None:
        """
            Print the results in the MetricTracker for a given epoch.
        """
        metric_means = self.metric_tracker.get_mean()
        metric_vars = self.metric_tracker.get_variance()
        metric_names = [key.replace('_',' ') for key in metric_means.keys()]
        pstr = f'Iterations completed: {self.iteration-1} Results:\n '
        pstr += '\n '.join([f'{metric_names[i]}: mean: {mean:.4f}, var: {var:.4f}' for i, (mean, var) in enumerate(zip(list(metric_means.values()), list(metric_vars.values())))])
        print_(pstr, 'info')