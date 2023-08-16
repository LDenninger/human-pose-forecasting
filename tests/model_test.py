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
    'TRANSFORMER_TYPE': 'spl',
    'NUM_BLOCKS': 2,
    'INPUT_DROPOUT': 0.1,
    'EMBEDDING_DIM': 32,
    'POSITIONAL_ENCODING_TYPE': 'sin',
    
    # Transformer Parameters
    'SPATIAL_HEADS': 4,
    'TEMPORAL_HEADS': 4,
    'SPATIAL_DROPOUT': 0.2,
    'TEMPORAL_DROPOUT': 0.2,
    'FF_DROPOUT': 0.2,
    'INPUT_DROPOUT': 0.2,
    
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
    configuration = getConfigFromDict(config_data)

    test_tensor = torch.randn((configuration.BATCH_SIZE, configuration.SEQ_LENGTH, configuration.NUM_JOINTS, configuration.JOINT_DIM))
    
    print('Attention Block')
    attn_block = SpatioTemporalAttentionBlock(
                        emb_dim = configuration.EMBEDDING_DIM,
                        num_emb = configuration.NUM_JOINTS,
                        temporal_heads = configuration.TEMPORAL_HEADS,
                        spatial_heads = configuration.SPATIAL_HEADS,
                        temporal_dropout=configuration.TEMPORAL_DROPOUT,
                        spatial_dropout=configuration.SPATIAL_DROPOUT,
                        ff_dropout=configuration.FF_DROPOUT,
            )
    ipdb.set_trace()
    print('Motion Predictor')
    motion_predictor = PosePredictor(
                                        positionalEncodingType=configuration.TRANSFORMER_TYPE,
                                        transformerType = configuration.POSITIONAL_ENCODING_TYPE,
                                        transformerConfig= configuration.TRANSFORMER_CONFIG,
                                        num_joints = configuration.NUM_JOINTS,
                                        seq_len = configuration.SEQ_LENGTH,
                                        num_blocks= configuration.NUM_BLOCKS,
                                        emb_dim = configuration.EMBEDDING_DIM,
                                        joint_dim = configuration.JOINT_DIM,
                                        input_dropout = configuration.INPUT_DROPOUT
    )
    ipdb.set_trace()
    output = motion_predictor(test_tensor)

#####===== High-Level Processing Function Tests =====#####

def test_multiHeadTemporalMMM():

    batch_size = 8
    num_heads = 4
    num_emb = 32
    seq_len = 10
    emb_dim = 16
    proj_dim = 4

    X = torch.randn((batch_size,num_emb,seq_len,emb_dim)) 
    W = torch.randn((num_heads,num_emb,emb_dim,proj_dim)) 
    import ipdb; ipdb.set_trace()
    # Compute goal output using for loops
    goal_output = torch.zeros((batch_size,num_heads,num_emb,seq_len,proj_dim))
    for bs in range(batch_size):
        for head in range(num_heads):
            for emb in range(num_emb):
                goal_output[bs, head, emb] = X[bs, emb] @ W[head, emb]

    # Compute the output using our function
    test_output = multiHeadTemporalMMM(X, W)
    all_close = torch.allclose(test_output, goal_output, atol=1e-03)
    import ipdb; ipdb.set_trace()
    print(f'Goal and test output equal: {all_close}')
    
    



