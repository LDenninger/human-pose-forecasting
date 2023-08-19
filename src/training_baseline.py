import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from utils import *
from models import PosePredictor

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
        LOGGER = self.logger
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
        # Backend configuration
        self.device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
        self.num_threads = num_threads

    ###=== Initialization Functions ===###
    def initialize_model(self):
        self.model = PosePredictor(
                            positional_encoding_config=self.config['POSITIONAL_ENCODING'],
                            transformer_config=self.config['TRANSFORMER'],
                            num_joints=self.config['NUM_JOINTS'],
                            seq_len=self.config['SEQ_LENGTH'],
                            num_blocks=self.config['NUM_BLOCKS'],
                            emb_dim=self.config['EMBEDDING_DIM'],
                            joint_dim=self.config['JOINT_DIM'],
                            input_dropout=self.config['INPUT_DROPOUT']
        ).to(self.device)

    def initialize_optimization(self) -> bool:
        if self.model is None:
            print_("Cannot initialize optimization without a model.", 'error')
            return False
        self.optimizer = getOptimizer(self.config['OPTIMIZER'], self.model)
        self.scheduler = getScheduler(self.config['SCHEDULER'], self.optimizer)
        self.loss = getLoss(self.config['LOSS'])
        return True
    
    def load_data(self):
        dataset = getDataset(self.config['DATASET'], is_train=True)
        self.train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=True,
            drop_last=True,
            num_workers=self.num_threads
        )
        dataset = getDataset(self.config['DATASET'], is_train=False)
        self.test_loader = DataLoader(
            dataset=dataset,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=False,
            drop_last=False,
            num_workers=self.num_threads
        )
    def load_checkpoint(self, checkpoint: str):
        load_model_from_checkpoint(
            exp_name=self.exp_name,
            run_name=self.run_name,
            model=self.model,
            epoch=checkpoint,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    
    def training_loop(self) -> bool:
        """
            Training loop for the model.
        """
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
        
       