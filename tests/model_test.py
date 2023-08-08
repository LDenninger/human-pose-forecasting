import torch
import torch.nn as nn
import ipdb
import numpy as np
import os


from models import *
from config import *
######===== Test Data Structures and Parameters =====######

config_data = {
    # Meta Parameters
    'RANDOM_SEED': 42,
    'DEBUG': True,
    'LOG': False,
    
    # Model Parameters
    'NUM_BLOCKS': 2,
    'INPUT_DROPOUT': 0.1,
    'EMBEDDING_DIM': 32,
    'TRANSFORMER_TYPE': 'spl',
    
    # Transformer Parameters
    'SPATIAL_HEADS': 4,
    'TEMPORAL_HEADS': 4,
    'SPATIAL_DROPOUT': 0.2,
    'TEMPORAL_DROPOUT': 0.2,
    'FF_DROPOUT': 0.2,
    
    # Training Parameters
    'BATCH_SIZE': 2,
    
    # Optimizer Parameters
    'OPTIMIZER_TYPE': 'Adam',
    'LEARNING_RATE': 0.01,
    'OPTIMIZER_ADD_PARAMS': {},
    
    # Learning Rate Scheduler Parameters
    'LR_SCHEDULER_TYPE': 'StepLR',
    'LR_SCHEDULER_DECAY_RATE': 0.1,
    'LR_SCHEDULER_ADD_PARAMS': {},
    
    # Data Parameters
    'SEQ_LENGTH': 10,
    'NUM_JOINTS': 27,
    'JOINT_DIM': 9,
    'TRAINING_SIZE': 1000,
    'VALIDATION_SIZE': 1000,
    'TEST_SIZE': 1000,
}

def test_transformer():
    ipdb.set_trace()
    config = getConfigFromDict(config_data)
    
    print('Attention Block')
    attn_block = SpatioTemporalAttentionBlock(
                        emb_dim = config.EMBEDDING_DIM,
                        num_emb = config.NUM_JOINTS,
                        temporal_heads = config.TEMPORAL_HEADS,
                        spatial_heads = config.SPATIAL_HEADS,
                        temporal_dropout=config.TEMPORAL_DROPOUT,
                        spatial_dropout=config.SPATIAL_DROPOUT,
                        ff_dropout=config.FF_DROPOUT,
            )