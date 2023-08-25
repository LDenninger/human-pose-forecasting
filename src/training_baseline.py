import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from .utils import *
from .models import PosePredictor

@for_all_methods(log_function)
class TrainerBaseline:
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
        self.logger = Logger(
            exp_name=self.exp_name,
            run_name=self.run_name,
            log_internal=log_process_internal,
        )
        logger = self.logger
        self.metric_tracker = MetricTracker()
        # Load the config for the run
        self.config = load_config_from_run(experiment_name, run_name)
        # Setup logging to WandB if desired
        if log_process_external:
            self.logger.initialize_logging(
                project_name='HumanPoseForecasting',
                config=self.config
            )
        # Modules
        self.model = None

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
    def train(self) -> None:
        self.training_loop()

    ###=== Initialization Functions ===###
    def initialize_model(self):
        self.model = PosePredictor(
            positional_encoding_config=self.config['model']['positional_encoding'],
            transformer_config=self.config['model']['transformer'],
            num_joints=self.config['skeleton']['num_joints'],
            seq_len=self.config['seq_length'],
            num_blocks=self.config['model']['num_blocks'],
            emb_dim=self.config['model']['embedding_dim'],
            joint_dim=self.config['joint_representation']['joint_dim'],
            input_dropout=self.config['model']['input_dropout']
        ).to(self.device)
        print_(f"Initialized model")

    def initialize_optimization(self) -> bool:
        if self.model is None:
            print_("Cannot initialize optimization without a model.", 'error')
            return False
        self.optimizer = getOptimizer(self.config['optimizer'], self.model)
        self.scheduler = getScheduler(self.config['lr_scheduler'], self.optimizer, emb_size=self.config['model']['embedding_dim'])
        self.loss = getLoss(self.config['loss'])
        print_(f"Initialized optimizer")
        return True

    def load_data(self, train: bool = True, test: bool = True) -> None:
        if train:
            dataset = getDataset(self.config['dataset'], is_train=True)
            self.train_loader = DataLoader(
                dataset=dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                drop_last=True,
                num_workers=self.num_threads
            )
            self.len_train = len(self.train_loader)
            print_(f"Loaded {self.len_train} training samples")

        if test:
            dataset = getDataset(self.config['dataset'], is_train=False)
            self.test_loader = DataLoader(
                dataset=dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                drop_last=False,
                num_workers=self.num_threads
            )
            self.len_test = len(self.test_loader)
            print_(f"Loaded {self.len_test} test samples")

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
        import ipdb; ipdb.set_trace()
        
        print_(f'Start training for run {self.exp_name}/{self.run_name}', 'info')
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

    def train_epoch(self) -> None:
        """
            A single epoch of training.
            The training results are saved through the MetricTracker and later retrieved.
        """
        import ipdb; ipdb.set_trace()
        self.model.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=self.len_train)
        running_loss = 1.0

        for batch_idx, (data, _) in progress_bar:
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
            running_loss = 0.8*running_loss + 0.2*loss.item()
            # Update the metric tracker to track the metrics of a single epoch
            self.metric_tracker.log('train_loss', loss.item())
            self.metric_tracker.step_iteration()
            progress_bar.set_description(f"Eval loss: {running_loss:.4f}")

    def evaluation_epoch(self) -> None:
        """
            A single epoch of evaluation
        """
        self.model.eval()
        progress_bar = tqdm(enumerate(self.test_loader), total=self.len_test)
        running_loss = 1.0

        for batch_idx, data in progress_bar:
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
            progress_bar.set_description(f"Iter.: {self.iteration}, Loss: {running_loss:.4f}")
    
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
        pstr = f' Epoch {self.epoch}/{self.config["num_epochs"]}, Iteration: {self.iteration} Results:\n'
        pstr += '\n '.join([f'{metric_names}: mean: {mean:.4f}, var: {var:.4f}' for mean, var in enumerate(zip(metric_means.values(), metric_vars.value()))])
        print_(pstr, 'info')