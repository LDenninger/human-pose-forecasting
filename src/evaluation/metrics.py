"""
Metrics for quantitative evaluation of the results ported to PyTorch and adapted.
Following https://github.com/eth-ait/motion-transformer/blob/master/metrics/motion_metrics.py.
@Author: Leon Herbrik, Luis Denninger
"""

import numpy as np
import torch
from typing import Optional, Literal, Union, List
import warnings
warnings.filterwarnings('ignore')

from ..utils import (
    print_,
    matrix_to_euler_angles,
    matrix_to_axis_angle,
    get_conv_to_rotation_matrix,
    correct_rotation_matrix,
    get_conv_from_axis_angle
)
from ..data_utils import h36m_forward_kinematics, SH_MASK_FROM_H36M

#####===== General Evaluation Constants =====#####
ACC_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]


#####===== Evaluation Functions =====#####


def evaluate_distribution_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metrics: List[str] = None,
    reduction: Optional[Literal["mean", "sum", "mse", None]] = None,
):
    """
    Evaluate the distribution metrics for long predictions.

    Args:
        predictions: torch tensor of predicted joints in shape (seq_len, batch_size, num_joints, joint_dim)
        targets: torch tensor of target joints in shape (seq_len, batch_size, num_joints, joint_dim)
        metrics: List of metrics to evaluate. If None, all implemented metrics are evaluated.
        reduction: Reduction to apply to the metrics. If None, no reduction is applied.

    """
    METRICS_IMPLEMENTED = {
        "ps_entropy": ps_entropy,
        "ps_kld": ps_kld,
        "npss": npss,
    }

    if metrics is None:
        metrics = METRICS_IMPLEMENTED.keys()

    results = {}

    # Compute the power spectrum of the predictions
    # For that we need to reshape the predictions to (batch_size, num_joints, seq_len, joint_dim)
    power_spec_pred = torch.permute(predictions, (1, 2, 0, 3))
    power_spec_pred = power_spectrum(power_spec_pred)

    # Compute the power spectrum of the targets
    power_spec_targ = torch.permute(targets, (1, 2, 0, 3))
    power_spec_targ = power_spectrum(power_spec_targ)

    # Compute npss
    # For this we need to reshape the predictions and targets to (batch_size, seq_len, num_joints * joint_dim)
    # Convert to euler angle representation from axis angle
    npss_preds = get_conv_from_axis_angle("euler")(predictions, "ZYX")
    npss_targs = get_conv_from_axis_angle("euler")(targets, "ZYX")
    
    npss_preds = torch.permute(npss_preds, (1, 0, 2, 3))
    npss_preds = torch.reshape(npss_preds, (npss_preds.shape[0], npss_preds.shape[1], -1))
    npss_targs = torch.permute(npss_targs, (1, 0, 2, 3))
    npss_targs = torch.reshape(npss_targs, (npss_targs.shape[0], npss_targs.shape[1], -1))

    # Convert both to numpy arrays
    npss_preds = npss_preds.numpy()
    npss_targs = npss_targs.numpy()

    for metric in metrics:
        if metric not in METRICS_IMPLEMENTED.keys():
            print_(f"Metric {metric} not implemented.")
        elif metric == "ps_entropy":
            results[metric] = METRICS_IMPLEMENTED[metric](power_spec_pred)
        elif metric == "ps_kld":
            results[metric] = METRICS_IMPLEMENTED[metric](
                power_spec_targ, power_spec_pred
            )
        elif metric == "npss":
            npss_res = torch.from_numpy(np.array([npss(npss_preds, npss_targs)]))
            # Change to torch float32
            npss_res = npss_res.to(torch.float32)
            results[metric] = npss_res
        if reduction is not None:
            results[metric] = _reduce(results[metric], reduction)
    return results


def evaluate_distance_metrics(
    predictions: torch.tensor,
    targets: torch.tensor,
    metrics: List[str] = None,
    reduction: Optional[Literal["mean", "sum", "mse", None]] = None,
    representation: Optional[Literal["axis", "mat", "quat", "6d", "pos"]] = "mat",
    s16_mask: Optional[bool] = False
):
    """
    Compute the pair-wise distance metrics between single joints.

    """
    METRICS_IMPLEMENTED = {
        "geodesic_distance": geodesic_distance,
        "positional_mse": positional_mse,
        "euler_error": euler_angle_error,
        "auc": accuracy_under_curve,
    }
    if metrics is None:
        metrics = METRICS_IMPLEMENTED.keys()
    results = {}

    if representation != "pos":
        conversion_func = get_conv_to_rotation_matrix(representation)
        predictions = conversion_func(predictions)
        targets = conversion_func(targets)

    if representation == "mat":
        # If we directly predict rotation matrices we have to make sure they are actually a rotation matrix
        predictions = correct_rotation_matrix(predictions)
    if representation != "pos":
        predictions = torch.reshape(predictions, (*predictions.shape[:-2], 9))
        targets = torch.reshape(targets, (*targets.shape[:-2], 9))
    for metric in metrics:
        if metric not in METRICS_IMPLEMENTED.keys():
            print_(f"Metric {metric} not implemented.")
        if metric in ["auc","positional_mse"]:
            # Compute the joint positions using forward kinematics
            if representation != "pos":
                prediction_positions, _ = h36m_forward_kinematics(predictions, 'mat')
                target_positions, _ = h36m_forward_kinematics(targets, 'mat')
                prediction_positions /= 1000
                target_positions /= 1000
                if s16_mask:
                    prediction_positions = prediction_positions[...,SH_MASK_FROM_H36M,:]
                    target_positions = target_positions[...,SH_MASK_FROM_H36M,:]
            else:
                prediction_positions = predictions
                target_positions = targets
            # Scale to meters for evaluation

            results[metric] = METRICS_IMPLEMENTED[metric](
                prediction_positions, target_positions, reduction=reduction
            )
            if torch.is_tensor(results[metric]):
                results[metric] = results[metric].item()
        else:
            results[metric] = METRICS_IMPLEMENTED[metric](
                predictions, targets, reduction=reduction
            ).item()

    return results


#####===== Distribution Metrics =====#####
def power_spectrum(seq: torch.Tensor) -> torch.Tensor:
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


def ps_entropy(seq_ps: torch.Tensor):
    """

    Args:
        seq_ps: (n_joints, seq_len, feature_size)

    Returns:
        entropy: (feature_size, )
    """
    res = -torch.sum(seq_ps * torch.log(seq_ps), axis = 0)
    return res


def ps_kld(seq_ps_from: torch.Tensor, seq_ps_to: torch.Tensor):
    """Calculates KL(seq_ps_from, seq_ps_to).
    Args:
        seq_ps_from:
        seq_ps_to:

    Returns:
        kld: (feature_size, )
    """
    res = torch.sum(seq_ps_from * torch.log(seq_ps_from / seq_ps_to), axis = 0)
    return res


def npss(euler_gt_sequences: np.ndarray, euler_pred_sequences: np.ndarray):
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


#####===== Pair-Wise Distance Metrics =====#####


def accuracy_under_curve(
    predictions: torch.tensor,
    targets: torch.tensor,
    thresholds: List[float] = ACC_THRESHOLDS,
    reduction: Optional[str] = None,
) -> torch.tensor:
    """
    Area und the Curve metric to measure the accuracy at different thresholds.
    """

    accs = []
    for threshold in thresholds:
        accs.append(
            accuracy_at_threshold(predictions, targets, threshold, "mean").item()
        )
    auc = np.mean(accs) * 100
    return auc


def accuracy_at_threshold(
    predictions: torch.tensor,
    targets: torch.tensor,
    thresh: float,
    reduction: Optional[Literal["mean", "sum", "mse", None]] = None,
) -> Union[torch.tensor, float]:
    """
    Percentage of correct keypoints.
    Args:
        predictions: torch tensor of predicted 3D joint positions in format (..., n_joints, 3)
        targets: torch tensor of same shape as `predictions`
        thresh: radius within which a predicted joint has to lie.

    Returns:
        Percentage of correct keypoints at the given threshold level, stored in a torch tensor of shape (..., len(threshs))
    """
    dist = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1))
    pck = torch.mean((dist <= thresh).to(dtype=torch.float32), dim=-1)
    return _reduce(pck, reduction)


def geodesic_distance(
    predictions: torch.tensor,
    targets: torch.tensor,
    reduction: Optional[Literal["mean", "sum", "mse", None]] = None,
) -> Union[torch.tensor, float]:
    """
    Computes the angular distance between the target and predicted rotations. We define this as the angle that is
    required to rotate one rotation into the other. This essentially computes || log(R_diff) || where R_diff is the
    difference rotation between prediction and target.

    Args:
        predictions: torch tensor of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3) or (..., n_joints, 9)
        targets: torch tensor of same shape as `predictions`

    Returns:
        The geodesic distance for each joint as a torch tensor of shape (..., n_joints)
    """
    

    preds, _ = _fix_dimensions(predictions)
    targs, orig_shape = _fix_dimensions(targets)
    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = torch.matmul(preds, targs.transpose(-2, -1))
    angles = matrix_to_axis_angle(r)
    angles = torch.linalg.vector_norm(angles, dim=-1)

    return _reduce(angles.view(*orig_shape), reduction)

    # batch=predictions.shape[0]
    # # Transpose preds and targets, so that they are of (n_joints, batch_size, 9)
    # predictions = torch.transpose(predictions, 0, 1)
    # targets = torch.transpose(targets, 0, 1)
    # # Reshape them so the last dimension is 3x3
    # predictions = torch.reshape(predictions, (predictions.shape[0], predictions.shape[1], 3, 3))
    # targets = torch.reshape(targets, (targets.shape[0], targets.shape[1], 3, 3))
    # # Loop over all joints
    # thetas = []
    # for i in range(predictions.shape[0]):
    #     m = torch.bmm(predictions[i], targets[i].transpose(1,2)) #batch*3*3
    #     cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    #     cos = torch.min(cos, torch.ones(batch))
    #     cos = torch.max(cos, torch.ones(batch)*-1 )
    #     theta = torch.acos(cos)
    #     thetas.append(theta)
    # # Stack the list of tensors to one tensor
    # thetas = torch.stack(thetas)

    # # Transpose to (batch_size, n_joints)
    # thetas = torch.transpose(thetas, 0, 1)

    # return _reduce(theta, reduction)
    


def positional_mse(
    predictions: torch.tensor,
    targets: torch.tensor,
    reduction: Optional[Literal["mean", "sum", "mse", None]] = None,
) -> Union[torch.tensor, float]:
    """
    Computes the Euclidean distance between joints in 3D space.
    Args:
        predictions: torch tensor of predicted 3D joint positions in format (..., n_joints, 3)
        targets: torch tensor of same shape as `predictions`

    Returns:
        The Euclidean distance for each joint as a torch tensor of shape (..., n_joints)
    """
    return _reduce(
        torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1)), reduction
    )


def euler_angle_error(
    predictions: torch.tensor,
    targets: torch.tensor,
    reduction: Optional[Literal["mean", "sum", "mse", None]] = None,
) -> torch.tensor:
    """
    Computes the Euler angle error using pytorch3d.
    Args:
        predictions: torch tensor of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 9)
        targets: torch tensor of same shape as `predictions`

    Returns:
        The Euler angle error as a torch tensor of shape (..., )
    """
    preds, _ = _fix_dimensions(predictions)
    targs, orig_shape = _fix_dimensions(targets)
    shape = predictions.shape

    # Convert rotation matrices to Euler angles using pytorch3d
    euler_preds = matrix_to_euler_angles(preds, "ZYX")  # (N, 3)
    euler_targs = matrix_to_euler_angles(targs, "ZYX")  # (N, 3)

    euler_preds = torch.reshape(euler_preds, (*shape[:-1], 3))
    euler_targs = torch.reshape(euler_targs, (*shape[:-1], 3))

    # l2 error on euler angles
    id_to_use = torch.where(torch.mean(torch.std(euler_targs, dim=0),dim=-1) > 1e-04)
    euc_error = torch.pow(euler_targs[id_to_use] - euler_preds[id_to_use], 2)
    euc_error = torch.sqrt(torch.sum(euc_error, dim=-1))  # (-1, ...)

    # reshape to original
    return _reduce(euc_error, reduction)


#####===== Helper Functions =====#####


def _reduce(input, reduction: Literal["mean", "sum", "mse", None] = None):
    """
    Reduce the output of the quantitative metrics to a scalar.
    """
    if reduction is None:
        return input
    if reduction == "mean":
        return torch.mean(input)
    elif reduction == "sum":
        return torch.sum(input)
    elif "mse":
        return torch.mean(torch.sqrt(torch.sum((input) ** 2, dim=-1)))
    else:
        raise NotImplementedError


def _fix_dimensions(
    input: torch.Tensor,
):
    """
    Fix input dimensions by flattening the leading dimensions and eventually stacking a flattened rotation matrix.
    """
    shape = input.shape
    if shape[-1] == 9:
        return torch.reshape(input, (-1, 3, 3)), input.shape[:-1]
    elif shape[-1] == 3 and shape[-2] == 3:
        return input.view(-1, 3, 3), input.shape[:-2]
    else:
        print_(f"Evaluation functions received invalid input shape: {shape}")
