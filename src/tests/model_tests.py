"""
    Unit tests for functions used within the model.

    Author: Luis Denninger <l_denninger@uni-bonn.de>

"""

import torch
import torch.nn as nn
import ipdb
import numpy as np
import os

from ..models import *
from ..utils import load_config
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
    'TRANSFORMER_CONFIG': {
        'SPATIAL_HEADS': 4,
        'TEMPORAL_HEADS': 4,
        'SPATIAL_DROPOUT': 0.2,
        'TEMPORAL_DROPOUT': 0.2,
        'FF_DROPOUT': 0.2,
        'INPUT_DROPOUT': 0.2
    },
    
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
    configuration = config_data

    test_tensor = torch.randn((configuration['BATCH_SIZE'], configuration['SEQ_LENGTH'], configuration['NUM_JOINTS'], configuration['JOINT_DIM']))

    print('Attention Block')
    attn_block = SpatioTemporalTransformer(
                    emb_dim=configuration['EMBEDDING_DIM'],
                    num_emb=configuration['NUM_JOINTS'],
                    seq_len=configuration['SEQ_LENGTH'],
                    temporal_heads=configuration['TRANSFORMER_CONFIG']['TEMPORAL_HEADS'],
                    spatial_heads=configuration['TRANSFORMER_CONFIG']['SPATIAL_HEADS'],
                    temporal_dropout=configuration['TRANSFORMER_CONFIG']['TEMPORAL_DROPOUT'],
                    spatial_dropout=configuration['TRANSFORMER_CONFIG']['SPATIAL_DROPOUT'],
                    ff_dropout=configuration['TRANSFORMER_CONFIG']['FF_DROPOUT'],
                )

    print('Motion Predictor')
    motion_predictor = PosePredictor(
                            positionalEncodingType=configuration['POSITIONAL_ENCODING_TYPE'],
                            transformerType=configuration['TRANSFORMER_TYPE'],
                            transformerConfig=configuration['TRANSFORMER_CONFIG'],
                            num_joints=configuration['NUM_JOINTS'],
                            seq_len=configuration['SEQ_LENGTH'],
                            num_blocks=configuration['NUM_BLOCKS'],
                            emb_dim=configuration['EMBEDDING_DIM'],
                            joint_dim=configuration['JOINT_DIM'],
                            input_dropout=configuration['INPUT_DROPOUT']
    )
    ipdb.set_trace()
    output = motion_predictor(test_tensor)

def test_pose_encoding_decoding():
    """
        Test function for the joint encoding and decoding of the pose predictor.
    """
    print('Test joint encoding and decoding of the pose predictor.')
    configuration = load_config('baseline_config')
    # Define the test tensors
    encoding_test_tensor = torch.randn((configuration['batch_size'], configuration['seq_length'], configuration['skeleton']['num_joints'], configuration['joint_representation']['joint_dim']))
    decoding_test_tensor = torch.randn((configuration['batch_size'], configuration['seq_length'], configuration['skeleton']['num_joints'], configuration['model']['embedding_dim']))
    # Define explicitly the output tensors
    encoding_target_output = torch.zeros((configuration['batch_size'], configuration['seq_length'], configuration['skeleton']['num_joints'], configuration['model']['embedding_dim']))
    decoding_target_output = torch.zeros((configuration['batch_size'], configuration['seq_length'], configuration['skeleton']['num_joints'], configuration['joint_representation']['joint_dim']))
    # Initialize the pose predictor
    motion_predictor = PosePredictor(
            positional_encoding_config=configuration['model']['positional_encoding'],
            transformer_config=configuration['model']['transformer'],
            num_joints=configuration['skeleton']['num_joints'],
            seq_len=configuration['seq_length'],
            num_blocks=configuration['model']['num_blocks'],
            emb_dim=configuration['model']['embedding_dim'],
            joint_dim=configuration['joint_representation']['joint_dim'],
            input_dropout=configuration['model']['input_dropout']
        )
    # Retrieve weight and bias tensors
    W_enc = motion_predictor.W_enc
    b_enc = motion_predictor.b_enc
    W_dec = motion_predictor.W_dec
    b_dec = motion_predictor.b_dec
    # Compute the target output for the encoding
    for bs in range(configuration['batch_size']):
        for seq in range(configuration['seq_length']):
            for joint in range(configuration['skeleton']['num_joints']):
                encoding_target_output[bs, seq, joint] = W_enc[joint] @ encoding_test_tensor[bs,seq,joint] # + b_enc[joint]
    # Compute the target output for the decoding
    for bs in range(configuration['batch_size']):
        for seq in range(configuration['seq_length']):
            for joint in range(configuration['skeleton']['num_joints']):
                decoding_target_output[bs, seq, joint] = W_dec[joint] @ decoding_test_tensor[bs,seq,joint] # + b_dec[joint]
    # Compute the output of the pose predictor
    encoding_model_output = motion_predictor.encode_joints(encoding_test_tensor)
    decoding_model_output = motion_predictor.decode_joints(decoding_test_tensor)
    # Check if the model output matches the target outputs

    enc_all_close = torch.allclose(encoding_model_output, encoding_target_output, atol=1e-03)
    dec_all_close = torch.allclose(decoding_model_output, decoding_target_output, atol=1e-03)
    # Print out the results
    if enc_all_close:
        print('Success! Encoding model output matches target output.')
    else:
        print('Failure! Encoding model output does not match target output.')
    if dec_all_close:
        print('Success! Decoding model output matches target output.')
    else:
        print('Failure! Decoding model output does not match target output.')
    import ipdb; ipdb.set_trace()
    print('\n')







#####===== High-Level Processing Function Tests =====#####

def test_processing_functions():
    print("Test 1: multiHeadTemporalMMM():")
    test_multiHeadTemporalMMM()
    print("Test 2: multiHeadSpatialMMVM():")
    test_multiHeadSpatialMMVM()
    print("Test 3: multiHeadSpatialMMM():")
    test_multiHeadSpatialMMM()


def test_multiHeadTemporalMMM():
    """
        Test functions for the function multiHeadTemporalMMM().
    
    """

    batch_size = 8
    num_heads = 4
    num_emb = 32
    seq_len = 10
    emb_dim = 16
    proj_dim = 4

    X = torch.randn((batch_size,num_emb,seq_len,emb_dim)) 
    W = torch.randn((num_heads,num_emb,emb_dim,proj_dim)) 
    # Compute goal output using for loops
    goal_output = torch.zeros((batch_size,num_heads,num_emb,seq_len,proj_dim))
    for bs in range(batch_size):
        for head in range(num_heads):
            for emb in range(num_emb):
                goal_output[bs, head, emb] = X[bs, emb] @ W[head, emb]
           

    # Compute the output using our function
    test_output = multiHeadTemporalMMM(X, W)
    print(f' Input shape: X: {X.shape} W: {W.shape}\n -> Goal Output shape: {goal_output.shape}\n -> Test Output shape: {test_output.shape}')
    all_close = torch.allclose(test_output, goal_output, atol=1e-03)
    if all_close:
        print(f' Output of function is equal to the goal output.\n Test successful!')
    else:
        print(f' Output of function is not equal to the goal output.\n Test failed!')

def test_multiHeadSpatialMMVM():
    """
        Test functions for the function multiHeadSpatialMMVM().
    
    """

    batch_size = 8
    num_heads = 4
    num_emb = 32
    seq_len = 10
    emb_dim = 16
    proj_dim = 4

    X = torch.randn((batch_size,num_emb,seq_len,emb_dim)) 
    W = torch.randn((num_heads, num_emb, emb_dim,proj_dim)) 
    # Compute goal output using for loops
    goal_output = torch.zeros((batch_size,seq_len,num_heads,num_emb,proj_dim))
    for bs in range(batch_size):
        for seq in range(seq_len):
            for head in range(num_heads):
                for emb in range(num_emb):
                    goal_output[bs, seq, head, emb] = W[head,emb].T @ X[bs, emb, seq]
           
    # Compute the output using our function
    test_output = multiHeadSpatialMMVM(torch.transpose(W, -2, -1), X)
    print(f' Input shape: X: {X.shape} W: {W.shape}\n -> Goal Output shape: {goal_output.shape}\n -> Test Output shape: {test_output.shape}')
    all_close = torch.allclose(test_output, goal_output, atol=1e-03)
    if all_close:
        print(f' Output of function is equal to the goal output.\n Test successful!')
    else:
        print(f' Output of function is not equal to the goal output.\n Test failed!')
    
def test_multiHeadSpatialMMM():
    """
        Test functions for the function multiHeadSpatialMMM().
    
    """

    batch_size = 8
    num_heads = 4
    num_emb = 32
    seq_len = 10
    emb_dim = 16
    proj_dim = 4

    X = torch.randn((batch_size,num_emb,seq_len,emb_dim)) 
    W = torch.randn((num_heads, emb_dim,proj_dim)) 
    # Compute goal output using for loops
    import ipdb; ipdb.set_trace()
    goal_output = torch.zeros((batch_size,seq_len,num_heads,num_emb,proj_dim))
    for bs in range(batch_size):
        for seq in range(seq_len):
                for head in range(num_heads):
                    goal_output[bs, seq, head] = X[bs,:,seq] @ W[head]
           
    import ipdb; ipdb.set_trace()
    # Compute the output using our function
    test_output = multiHeadSpatialMMM(X, W)
    print(f' Input shape: X: {X.shape} W: {W.shape}\n -> Goal Output shape: {goal_output.shape}\n -> Test Output shape: {test_output.shape}')
    all_close = torch.allclose(test_output, goal_output, atol=1e-03)
    if all_close:
        print(f' Output of function is equal to the goal output.\n Test successful!')
    else:
        print(f' Output of function is not equal to the goal output.\n Test failed!')
    
def test_multiWeightMMM():
    """
        Test functions for the function multiWeightMMM().
    
    """

    batch_size = 8
    num_heads = 4
    num_emb = 32
    seq_len = 10
    emb_dim = 16
    proj_dim = 4

    X = torch.randn((batch_size,num_emb,seq_len,emb_dim)) 
    W = torch.randn((num_heads, emb_dim,proj_dim)) 
    # Compute goal output using for loops
    goal_output = torch.zeros((batch_size,seq_len,num_emb,proj_dim))
    for bs in range(batch_size):
        for seq in range(seq_len):
                for head in range(num_heads):
                    goal_output[bs, seq, head] = X[bs,:,seq] @ W[head]
           
    # Compute the output using our function
    test_output = multiWeightMMM(X, W)
    print(f' Input shape: X: {X.shape} W: {W.shape}\n -> Goal Output shape: {goal_output.shape}\n -> Test Output shape: {test_output.shape}')
    all_close = torch.allclose(test_output, goal_output, atol=1e-03)
    if all_close:
        print(f' Output of function is equal to the goal output.\n Test successful!')
    else:
        print(f' Output of function is not equal to the goal output.\n Test failed!')



