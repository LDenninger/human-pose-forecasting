# Third party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np






def test_distribution_metrics():

    # Create 32x1x1x78 tensor with values from one to 78
    batch_pred = torch.randint(1, 79, (32, 26, 50, 3)).float()
    batch_targ = torch.randint(79, 100, (32, 26, 50, 3)).float()

    # Compoute power spectrum
    ps = power_spectrum_torch(batch_pred)
    ps_targ = power_spectrum_torch(batch_targ)

    # Compute entropy
    entropy = ps_entropy_torch(ps)

    # Compute KL divergence
    kld_to_targ = ps_kld_torch(ps, ps_targ)
    kld_from_targ = ps_kld_torch(ps_targ, ps)

    # Print entropy, kl divergences to console together with shape
    print(f"Entropy: {entropy}, shape: {entropy.shape}")
    print(f"KL divergence from target: {kld_from_targ}, shape: {kld_from_targ.shape}")
    print(f"KL divergence to target: {kld_to_targ}, shape: {kld_to_targ.shape}")
    
    # Create same shape np array with same values as tensors
    batch_pred_np = batch_pred.numpy()

    # Compute power spectrum
    ps_np = power_spectrum(batch_pred_np)
    print(ps_np.shape)
    ps_targ_np = power_spectrum(batch_targ.numpy())

    # Compute entropy
    entropy_np = ps_entropy(ps_np)

    # Compute KL divergence
    kld_to_targ_np = ps_kld(ps_np, ps_targ_np)
    kld_from_targ_np = ps_kld(ps_targ_np, ps_np)

    # Print entropy, kl divergences to console together with shape
    print(f"Entropy: {entropy_np}, shape: {entropy_np.shape}")
    print(f"KL divergence from target: {kld_from_targ_np}, shape: {kld_from_targ_np.shape}")
    print(f"KL divergence to target: {kld_to_targ_np}, shape: {kld_to_targ_np.shape}")

    # Test closeness of results
    assert np.isclose(entropy, entropy_np).all()
    assert np.isclose(kld_to_targ, kld_to_targ_np).all()
    assert np.isclose(kld_from_targ, kld_from_targ_np).all()
    

#     # # Create 64x41x99 tensor with random values
#     # batch_pred = torch.rand((41, 64, 33, 3))
#     # batch_gt = torch.rand((41, 64, 33, 3))

#     # results = evaluate_distribution_metrics(batch_pred, batch_gt)


#     # # Create same shape np array with same values as tensors
#     # batch_pred_np = batch_pred.numpy()
#     # batch_gt_np = batch_gt.numpy()

#     # results_np = {}

#     # # Permute to (batch_size, num_joints, seq_len, joint_dim)
#     # batch_pred_np = np.transpose(batch_pred_np, (1, 2, 0, 3))
#     # power_spec_pred = power_spectrum(batch_pred_np)

#     # # Same for ground truth
#     # batch_gt_np = np.transpose(batch_gt_np, (1, 2, 0, 3))
#     # power_spec_gt = power_spectrum(batch_gt_np)

#     # # Calculate ps_entropy and remove singleton dimensions
#     # results_np["ps_entropy"] = ps_entropy(power_spec_pred).squeeze()

#     # # Calculate ps_kld and remove singleton dimensions
#     # results_np["ps_kld"] = ps_kld(power_spec_gt, power_spec_pred).squeeze()

#     # # Calculate npss
#     # # Reshape to (batch_size, seq_len, num_joints * joint_dim)
#     # npss_preds = get_conv_from_axis_angle("euler")(batch_pred, "ZYX").numpy()
#     # npss_targs = get_conv_from_axis_angle("euler")(batch_gt, "ZYX").numpy()
#     # batch_pred_np = np.transpose(npss_preds, (1, 0, 2, 3))
#     # batch_pred_np = np.reshape(batch_pred_np, (batch_pred_np.shape[0], batch_pred_np.shape[1], -1))
#     # batch_gt_np = np.transpose(npss_targs, (1, 0, 2, 3))
#     # batch_gt_np = np.reshape(batch_gt_np, (batch_gt_np.shape[0], batch_gt_np.shape[1], -1))
#     # results_np["npss"] = compute_npss(batch_gt_np, batch_pred_np)



    
#     # Round np results to same decimal amount as torch results
#     for key in results_np.keys():
#         # Loop over all numbers in the tensor and check if they are equal to the corresponding number in the np array
#         if not key == "npss":
#             for i in range(results[key].shape[0]):
#                 # Check if rounded values are equal
#                 if not np.isclose(
#                     np.round(results[key][i], 6), np.round(results_np[key][i], 6)
#                 ):
#                     print(
#                         f"Tensor value: {results[key][i]}, np value: {results_np[key][i]}"
#                     )
#         else:
#             if not np.isclose(
#                 np.round(results[key], 6), np.round(results_np[key], 6)
#             ):
#                 print(f"Tensor value: {results[key]}, np value: {results_np[key]}")

#     # Calculate mean of npss
#     results_np["npss"] = np.mean(results_np["npss"])
        
    

#     print(results)

#####===== Distribution Metrics =====#####
def power_spectrum_torch(seq: torch.Tensor) -> torch.Tensor:
    """
    # seq = seq[:, :, 0:-1:12, :]  # 5 fps for amass (in 60 fps)

    Args:
      seq: (batch_size, n_joints, seq_len, feature_size)

    Returns:
        (n_joints, seq_len, feature_size)
    """
    seq = torch.permute(seq, [0, 2, 1, 3]) # (batch_size, seq_len, n_joints, feature_size)
    seq = torch.reshape(seq, [*seq.shape[:2], 1, -1]) # (batch_size, seq_len, 1, n_joints * feature_size)
    seq = torch.permute(seq, [0, 2, 1, 3]) # (batch_size, 1, seq_len, n_joints * feature_size)
    seq_fft = torch.fft.fft(seq, dim=2) # fast fourier over seq_len dimension
    seq_ps = torch.abs(seq_fft) ** 2 # power spectrum definition

    seq_ps_global = seq_ps.sum(dim=0) + torch.finfo(torch.float32).eps # sum over batch dimension
    seq_ps_global /= seq_ps_global.sum(dim=1, keepdims=True) # normalize over seq_len dimension (which is then dimension 1)
    return seq_ps_global.squeeze() # (seq_len, n_joints * feature_size)


def ps_entropy_torch(seq_ps):
    """

    Args:
        seq_ps: (n_joints, seq_len, feature_size)

    Returns:
    """
    res = -torch.sum(seq_ps * torch.log(seq_ps), axis = 0)
    return res


def ps_kld_torch(seq_ps_from, seq_ps_to):
    """Calculates KL(seq_ps_from, seq_ps_to).
    Args:
        seq_ps_from:
        seq_ps_to:

    Returns:
    """
    res = torch.sum(seq_ps_from * torch.log(seq_ps_from / seq_ps_to), axis = 0)
    return res




def power_spectrum(seq):
    """
    # seq = seq[:, :, 0:-1:12, :]  # 5 fps for amass (in 60 fps)
    
    Args:
      seq: (batch_size, n_joints, seq_len, feature_size)
  
    Returns:
        (n_joints, seq_len, feature_size)
    """
    seq_t = np.transpose(seq, [0, 2, 1, 3])
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
    res = -np.sum(seq_ps * np.log(seq_ps), axis=1)
    return res


def ps_kld(seq_ps_from, seq_ps_to):
    """ Calculates KL(seq_ps_from, seq_ps_to).
    Args:
        seq_ps_from:
        seq_ps_to:

    Returns:
    """
    res = np.sum(seq_ps_from * np.log(seq_ps_from / seq_ps_to), axis=1)
    return res


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
