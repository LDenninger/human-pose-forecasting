import torch
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from .data_utils import getDataset, H36M_DATASET_ACTIONS, DataAugmentor
from .utils import *
from .models import PosePredictor, getModel
from .evaluation import EvaluationEngine

class Session:
    """
        The session module capsules the complete interaction with the model.
        Within a session we load the run data, configs and anything else needed for the model.
        THe session can be used to either train or evaluate the model.
    """

    @log_function
    def __init__(self,
                  experiment_name: str,
                  run_name: str,
                  log_process_internal: Optional[bool] = False,
                  log_process_external: Optional[bool] = True,
                  num_threads: int = 2,
                  debug: Optional[bool] = False) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Run meta information
        self.exp_name = experiment_name
        self.run_name = run_name
        self.debug = debug
        self.evaluate_model = False
        self.visualize_model = False
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
        self.evaluation_engine = EvaluationEngine(device=self.device)
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
        self.num_threads = num_threads
        print_(f"Initialized trainer for run: {experiment_name}/{run_name}")
        print_(f"Using device: {self.device}")

    ###=== Calls ===###
    @log_function
    def train(self) -> None:
        """
            Train the model.
        """
        self.training_loop()

    @log_function
    def evaluate(self) -> None:
        """ Evaluate the model. """
        if not self.evaluate_model:
            print_('Evaluation was not properly initialized!', 'error')
            return
        self.evaluation_engine.evaluate(self.model)
        self.evaluation_engine.log_results(self.iteration)
        self._print_epoch_results()
    
    def visualize(self, num_visualization: Optional[int] = 1) -> None:
        """ Visualize the model. """
        if not self.visualize_model:
            print_('Visualization was not properly initialized!', 'error')
            return
        self.evaluation_engine.visualize(self.model)

    ###=== Initialization Functions ===###

    @log_function
    def initialize_evaluation(self, 
                              evaluation_type: List[str] = ['distance'],
                              num_iterations: Optional[int] = None,
                              distance_metrics: List[str] = None,
                              distribution_metrics: List[str] = None,
                              split_actions: Optional[bool]=False,
                              prediction_timesteps: Optional[List[int]] = None,
                              dataset: Literal['h36m','ais'] = 'h36m') -> bool:
        """
            Initialize the evaluation procedure and load the corresponding data
        """
        self.evaluate_model = True
        # Perform an exhaustive evaluation that includes the evaluation of separate actions for different prediction lengths
        if not self.evaluation_engine.data_loaded:
            self._load_evaluation_data(dataset, split_actions, max(self.config['evaluation']['timesteps'] if prediction_timesteps is None else prediction_timesteps))
        if 'distance' in evaluation_type:
            self.evaluation_engine.initialize_distance_evaluation(
                iterations = self.config['num_eval_iterations'] if num_iterations is None else num_iterations,
                prediction_timesteps = self.config['evaluation']['timesteps'],
                metric_names=self.config['evaluation']['metrics'] if distance_metrics is None else distance_metrics,
            )
            self.num_eval_iterations = self.config['num_eval_iterations']
            print_(f"Initialized an evaluation for joint distances with {self.evaluation_engine.num_iterations['distance_metric']}")
        if 'distribution' in evaluation_type:
            self.evaluation_engine.initialize_long_prediction_evaluation(
                iterations = self.config['num_eval_iterations'] if num_iterations is None else num_iterations,
                prediction_timesteps = self.config['evaluation']['timesteps'],
                metric_names=self.config['evaluation']['metrics'] if distribution_metrics is None else distribution_metrics,
            )
            self.num_eval_iterations = self.config['num_eval_iterations']
            print_(f"Initialized an evaluation for long predictions with {self.evaluation_engine.num_iterations['long_predictions']}")

    @log_function
    def initialize_visualization(self,
                                 visualization_type: List[str] = ['2d'],
                                 prediction_timesteps: List[int] = None,
                                 dataset: Literal['h36m','ais'] = 'h36m',
                                 split_actions: Optional[bool]=False,
                                 ):
        if prediction_timesteps is None:
            prediction_timesteps = self.config['evaluation']['timesteps']

        if not self.evaluation_engine.data_loaded:
            self._load_evaluation_data(dataset, split_actions, max(prediction_timesteps))
        
        if '2d' in visualization_type:
            self.evaluation_engine.initialize_visualization_2d(
                prediction_timesteps=prediction_timesteps
            )
            self.visualize_model = True
        if '3d' in visualization_type:
            self.evaluation_engine.initialize_visualization_3d(
                max_length=prediction_timesteps[-1]
            )
            self.visualize_model = True
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
        self.loss = getLoss(self.config['loss'], self.config['joint_representation']['type'])
        print_(f"Initialized optimizer")
        return True

    @log_function
    def load_train_data(self) -> None:
        """
            Load the training data.
        """
        dataset = getDataset(
            self.config['dataset'],
            joint_representation = self.config['joint_representation']['type'],
            skeleton_model = self.config['skeleton']['type'],
            is_train=True,
            debug=self.debug
        )
        self.train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=self.num_threads,
        )
        mean, var = dataset.get_mean_variance()
        self.num_iterations = self.config['num_train_iterations'] if self.config['num_train_iterations']!=-1 else len(self.train_loader)
        p_str = f'Loaded training data: Length: {len(dataset)}, Batched length: {len(self.train_loader)}, Iterations per epoch: {self.num_iterations}'
        print_(p_str)
        self.data_augmentor = DataAugmentor(
            normalize=self.config['data_augmentation']['normalize'],
            reverse_prob=self.config['data_augmentation']['reverse_prob'],
            snp_noise_prob=self.config['data_augmentation']['snp_noise_prob'],
            joint_cutout_prob=self.config['data_augmentation']['joint_cutout_prob'],
            timestep_cutout_prob=self.config['data_augmentation']['timestep_cutout_prob']
        )
        self.data_augmentor.set_mean_var(mean.to(self.device), var.to(self.device))

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

        print_(f'Start training for run {self.exp_name}/{self.run_name}', 'info')
        # Initial evaluation for untrained model
        if self.evaluate_model:
            self.evaluation_engine.evaluate(self.model)
            self.evaluation_engine.log_results(self.iteration)
            self._print_epoch_results()
        for self.epoch in range(1, self.config['num_epochs'] + 1):
            print_(f"Epoch {self.epoch}/{self.config['num_epochs']}", 'info')
            # Reset the tracked metrics for the new epoch
            self.metric_tracker.reset()
            self.evaluation_engine.reset()
            # Run a single epoch of training
            if self.config['training_scheme'] == 'single_step':
                self.train_epoch_single_step()
            elif self.config['training_scheme'] == 'auto_regressive':
                self.train_epoch_auto_regressive()
            # Evaluation in pre-defined intervals
            if self.evaluate_model and self.epoch % self.config['evaluation']['frequency'] == 0:
                self.evaluation_engine.evaluate(self.model)
                self.evaluation_engine.log_results(self.iteration)
         
            # Print out the epoch results
            self._print_epoch_results()
            # Save checkpoint 
            if self.epoch % self.config['checkpoint_frequency'] == 0:
                self.logger.save_checkpoint(self.model, self.optimizer, self.scheduler, self.epoch)
        # Save the final model
        self.logger.save_checkpoint(self.model, self.optimizer, self.scheduler, self.epoch, True)
        self.logger.finish_logging()
        print_('Training finished!')

    @log_function
    def train_epoch_single_step(self) -> None:
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
            seed_data = self.data_augmentor(seed_data)
            target_data = self.data_augmentor(target_data)
            # Update the learning rate according to the schedule
            self.optimizer.zero_grad()
            self.scheduler(self.iteration)
            # Forward pass through the network
            output = self.model(seed_data)
            #if self.config['data_augmentation']['normalize']:
            #    output = self.data_augmentor.reverse_normalization(output)
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
                self.config['loss']['type']: loss.item()
            }, step=self.iteration)
            # Update the progress bar description
            if batch_idx == 0:
                running_loss = loss.item()
            else:
                running_loss = 0.8*running_loss + 0.2*loss.item()
            progress_bar.set_description(f"Train loss: {running_loss:.4f}")
    
    def train_epoch_auto_regressive(self) -> None:
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
            seed_data = self.data_augmentor(seed_data)
            target_data = self.data_augmentor(target_data)
            # Update the learning rate according to the schedule
            self.optimizer.zero_grad()
            self.scheduler(self.iteration)
            predictions = []
            # Forward pass through the network
            cur_input = seed_data
            for i in range(self.config['dataset']['target_length']):
                output = self.model(cur_input)
                predictions.append(output[...,-1,:,:])
                cur_input = output
            
            predictions = torch.stack(predictions)
            predictions = torch.transpose(predictions, 0,1)

            #if self.config['data_augmentation']['normalize']:
            #    output = self.data_augmentor.reverse_normalization(output)
            # Compute loss using the target data
            loss = self.loss(predictions, target_data)
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
                self.config['loss']['type']: loss.item()
            }, step=self.iteration)
            # Update the progress bar description
            if batch_idx == 0:
                running_loss = loss.item()
            else:
                running_loss = 0.8*running_loss + 0.2*loss.item()
            progress_bar.set_description(f"Train loss: {running_loss:.4f}")
    
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
        print_(f'Evaluation Results:\n',)
        self.evaluation_engine.print()
    
    def _load_evaluation_data(self, 
                              dataset: Literal['h36m','ais'], 
                              split_actions: Optional[bool]=False, 
                              prediction_length: Optional[int]=None):
        self.evaluation_engine.load_data(
            dataset=dataset,
            seed_length = self.config['dataset']['seed_length'],
            prediction_length = prediction_length,
            down_sampling_factor=self.config['dataset']['downsampling_factor'],
            split_actions=split_actions,
            skeleton_representation = self.config['skeleton']['type'],
            normalize=self.config['data_augmentation']['normalize'],
            representation= self.config['joint_representation']['type']

        )