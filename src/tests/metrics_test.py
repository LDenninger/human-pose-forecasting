# Third party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np

# Local application imports
from ..data_utils import *
from ..evaluation import evaluate_distribution_metrics, euler_angle_error, geodesic_distance
from ..utils import (
    print_,
    matrix_to_euler_angles,
    matrix_to_axis_angle,
    get_conv_to_rotation_matrix,
    correct_rotation_matrix,
    get_conv_from_axis_angle
)

def test_metrics():
    # Create 26 x 3 x 3 tensor
    frame = torch.zeros((26, 3, 3))

    # Generate random rotation matrix for each joint using R.random().as_matrix()
    for i in range(26):
        frame[i] = torch.from_numpy(R.random().as_matrix())

    # TODO: Test euler diff
    # Test 1 - Test if the euler diff is zero for the same input
    euler_same = euler_angle_error(frame, frame)
    assert torch.all(euler_same == 0)
    # Test 2 - Test if the euler diff is slightly larger than zero for a slightly different input
    slightly_different_frame = frame.clone()
    # Add small random rotation to each joint
    for i in range(26):
        slightly_different_frame[i] = (
            slightly_different_frame[i]
            + torch.from_numpy(R.random().as_matrix()) * 0.01
        )
    euler_slightly_different = euler_angle_error(frame, slightly_different_frame)
    # Test 3 - Test if the euler diff is larger for a 90 degree rotation
    frame_90_degrees = frame.clone()
    # Add 90 degree rotation to each joint
    for i in range(26):
        frame_90_degrees[i] = frame_90_degrees[i] + torch.from_numpy(
            R.from_euler("x", 90, degrees=True).as_matrix()
        )
    euler_90_degrees = euler_angle_error(frame, frame_90_degrees)

    # TODO: Test angle diff
    # Test 1 - Test if the angle diff is zero for the same input
    angle_same = geodesic_distance(frame, frame)
    assert torch.all(angle_same == 0)

    # Test 2 - Test if the angle diff is slightly larger than zero for a slightly different input
    angle_slightly_different = geodesic_distance(frame, slightly_different_frame)

    # Test 3 - Test the angle diff is larger for a 90 degree rotation
    angle_90_degrees = geodesic_distance(frame, frame_90_degrees)

    pass


def test_distribution_metrics():
    # Create 64x41x99 tensor with random values
    batch_pred = torch.rand((41, 64, 33, 3))
    batch_gt = torch.rand((41, 64, 33, 3))

    results = evaluate_distribution_metrics(batch_pred, batch_gt)


    # Create same shape np array with same values as tensors
    batch_pred_np = batch_pred.numpy()
    batch_gt_np = batch_gt.numpy()

    results_np = {}

    # Permute to (batch_size, num_joints, seq_len, joint_dim)
    batch_pred_np = np.transpose(batch_pred_np, (1, 2, 0, 3))
    power_spec_pred = power_spectrum(batch_pred_np)

    # Same for ground truth
    batch_gt_np = np.transpose(batch_gt_np, (1, 2, 0, 3))
    power_spec_gt = power_spectrum(batch_gt_np)

    # Calculate ps_entropy and remove singleton dimensions
    results_np["ps_entropy"] = ps_entropy(power_spec_pred).squeeze()

    # Calculate ps_kld and remove singleton dimensions
    results_np["ps_kld"] = ps_kld(power_spec_gt, power_spec_pred).squeeze()

    # Calculate npss
    # Reshape to (batch_size, seq_len, num_joints * joint_dim)
    npss_preds = get_conv_from_axis_angle("euler")(batch_pred, "ZYX").numpy()
    npss_targs = get_conv_from_axis_angle("euler")(batch_gt, "ZYX").numpy()
    batch_pred_np = np.transpose(npss_preds, (1, 0, 2, 3))
    batch_pred_np = np.reshape(batch_pred_np, (batch_pred_np.shape[0], batch_pred_np.shape[1], -1))
    batch_gt_np = np.transpose(npss_targs, (1, 0, 2, 3))
    batch_gt_np = np.reshape(batch_gt_np, (batch_gt_np.shape[0], batch_gt_np.shape[1], -1))
    results_np["npss"] = compute_npss(batch_gt_np, batch_pred_np)



    
    # Round np results to same decimal amount as torch results
    for key in results_np.keys():
        # Loop over all numbers in the tensor and check if they are equal to the corresponding number in the np array
        if not key == "npss":
            for i in range(results[key].shape[0]):
                # Check if rounded values are equal
                if not np.isclose(
                    np.round(results[key][i], 6), np.round(results_np[key][i], 6)
                ):
                    print(
                        f"Tensor value: {results[key][i]}, np value: {results_np[key][i]}"
                    )
        else:
            if not np.isclose(
                np.round(results[key], 6), np.round(results_np[key], 6)
            ):
                print(f"Tensor value: {results[key]}, np value: {results_np[key]}")

    # Calculate mean of npss
    results_np["npss"] = np.mean(results_np["npss"])
        
    

    print(results)



def power_spectrum(seq):
    """
    # seq = seq[:, :, 0:-1:12, :]  # 5 fps for amass (in 60 fps)
    
    Args:
      seq: (batch_size, n_joints, seq_len, feature_size)
  
    Returns:
        (n_joints, seq_len, feature_size)
    """
    feature_size = seq.shape[-1]
    n_joints = seq.shape[1]

    seq_t = np.transpose(seq, [0, 2, 1, 3])
    dims_to_use = np.where((np.reshape(seq_t, [-1, n_joints, feature_size]).std(0) >= 1e-4).all(axis=-1))[0]
    seq_t = seq_t[:, :, dims_to_use]

    seq_t = np.reshape(seq_t, [seq_t.shape[0], seq_t.shape[1], 1, -1])
    seq = np.transpose(seq_t, [0, 2, 1, 3])
    
    seq_fft = np.fft.fft(seq, axis=2)
    seq_ps = np.abs(seq_fft)**2
    
    seq_ps_global = seq_ps.sum(axis=0) + 1e-8
    seq_ps_global /= seq_ps_global.sum(axis=1, keepdims=True)
    return seq_ps_global


def ps_entropy(seq_ps):
    """
    
    Args:
        seq_ps: (n_joints, seq_len, feature_size)

    Returns:
    """
    return -np.sum(seq_ps * np.log(seq_ps), axis=1)


def ps_kld(seq_ps_from, seq_ps_to):
    """ Calculates KL(seq_ps_from, seq_ps_to).
    Args:
        seq_ps_from:
        seq_ps_to:

    Returns:
    """
    return np.sum(seq_ps_from * np.log(seq_ps_from / seq_ps_to), axis=1)


def compute_npss(euler_gt_sequences, euler_pred_sequences):
    """
    Computing normalized Normalized Power Spectrum Similarity (NPSS)
    Taken from @github.com neural_temporal_models/blob/master/metrics.py#L51
    
    1) fourier coeffs
    2) power of fft
    3) normalizing power of fft dim-wise
    4) cumsum over freq.
    5) EMD
    
    Args:
        euler_gt_sequences: (batch_size, seq_len, num_joints * joint_dim)
        euler_pred_sequences: (batch_size, seq_len, num_joints * joint_dim)
    Returns:
    """
    gt_fourier_coeffs = np.zeros(euler_gt_sequences.shape)
    pred_fourier_coeffs = np.zeros(euler_pred_sequences.shape)
    
    # power vars
    gt_power = np.zeros((gt_fourier_coeffs.shape))
    pred_power = np.zeros((gt_fourier_coeffs.shape))
    
    # normalizing power vars
    gt_norm_power = np.zeros(gt_fourier_coeffs.shape)
    pred_norm_power = np.zeros(gt_fourier_coeffs.shape)
    
    cdf_gt_power = np.zeros(gt_norm_power.shape)
    cdf_pred_power = np.zeros(pred_norm_power.shape)
    
    emd = np.zeros(cdf_pred_power.shape[0:3:2])
    
    # used to store powers of feature_dims and sequences used for avg later
    seq_feature_power = np.zeros(euler_gt_sequences.shape[0:3:2])
    power_weighted_emd = 0
    
    for s in range(euler_gt_sequences.shape[0]):
        
        for d in range(euler_gt_sequences.shape[2]):
            gt_fourier_coeffs[s, :, d] = np.fft.fft(
                euler_gt_sequences[s, :, d])  # slice is 1D array
            pred_fourier_coeffs[s, :, d] = np.fft.fft(
                euler_pred_sequences[s, :, d])
            
            # computing power of fft per sequence per dim
            gt_power[s, :, d] = np.square(
                np.absolute(gt_fourier_coeffs[s, :, d]))
            pred_power[s, :, d] = np.square(
                np.absolute(pred_fourier_coeffs[s, :, d]))
            
            # matching power of gt and pred sequences
            gt_total_power = np.sum(gt_power[s, :, d])
            pred_total_power = np.sum(pred_power[s, :, d])
            # power_diff = gt_total_power - pred_total_power
            
            # adding power diff to zero freq of pred seq
            # pred_power[s,0,d] = pred_power[s,0,d] + power_diff
            
            # computing seq_power and feature_dims power
            seq_feature_power[s, d] = gt_total_power
            
            # normalizing power per sequence per dim
            if gt_total_power != 0:
                gt_norm_power[s, :, d] = gt_power[s, :, d]/gt_total_power
            
            if pred_total_power != 0:
                pred_norm_power[s, :, d] = pred_power[s, :, d]/pred_total_power
            
            # computing cumsum over freq
            cdf_gt_power[s, :, d] = np.cumsum(gt_norm_power[s, :, d])  # slice is 1D
            cdf_pred_power[s, :, d] = np.cumsum(pred_norm_power[s, :, d])
            
            # computing EMD
            emd[s, d] = np.linalg.norm((cdf_pred_power[s, :, d] - cdf_gt_power[s, :, d]), ord=1)
    
    # computing weighted emd (by sequence and feature powers)
    power_weighted_emd = np.average(emd, weights=seq_feature_power)
    
    return power_weighted_emd

if __name__ == "__main__":
    test_distribution_metrics()
