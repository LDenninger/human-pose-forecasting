"""
Metrics for quantitative evaluation of the results ported to PyTorch and adapted.
Following https://github.com/eth-ait/motion-transformer/blob/master/metrics/motion_metrics.py.
@Author: Leon Herbrik
"""

import numpy as np
import torch
from typing import Optional, Literal, Union, List

from ..utils import (
    print_,
    matrix_to_euler_angles,
    matrix_to_axis_angle,
    get_conv_to_rotation_matrix,
    correct_rotation_matrix,
)
from ..data_utils import h36m_forward_kinematics

#####===== General Evaluation Constants =====#####
ACC_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]


#####===== Evaluation Functions =====#####


def evaluate_distribution_metrics(
    predictions: torch.Tensor,
    metrics: List[str] = None,
    reduction: Optional[Literal["mean", "sum", "mse", None]] = None,
    representation: Optional[Literal["axis", "mat", "quat", "6d"]] = "mat",
):
    """
    Evaluate the distribution metrics for long predictions.
    """
    METRICS_IMPLEMENTED = {
        "power_spectrum": power_spectrum,
        "ps_entropy": ps_entropy,
        "ps_kld": ps_kld,
    }

    if metrics is None:
        metrics = METRICS_IMPLEMENTED.keys()

    results = {}
    conversion_func = get_conv_to_rotation_matrix(representation)
    predictions = conversion_func(predictions)
    targets = conversion_func(targets)

    if representation == "mat":
        # If we directly predict rotation matrices we have to make sure they are actually a rotation matrix
        predictions = correct_rotation_matrix(predictions)

    for metric in metrics:
        if metric not in METRICS_IMPLEMENTED.keys():
            print_(f"Metric {metric} not implemented.")
        results[metric] = METRICS_IMPLEMENTED[metric](predictions).item()

    pass


def evaluate_distance_metrics(
    predictions: torch.tensor,
    targets: torch.tensor,
    metrics: List[str] = None,
    reduction: Optional[Literal["mean", "sum", "mse", None]] = None,
    representation: Optional[Literal["axis", "mat", "quat", "6d", "pos"]] = "mat",
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
    if representation != 'pos':
        conversion_func = get_conv_to_rotation_matrix(representation)
        predictions = conversion_func(predictions)
        targets = conversion_func(targets)

    if representation == "mat":
        # If we directly predict rotation matrices we have to make sure they are actually a rotation matrix
        predictions = correct_rotation_matrix(predictions)
    if representation != 'pos':
        predictions = torch.reshape(predictions, (*predictions.shape[:-2], 9))
        targets = torch.reshape(targets, (*targets.shape[:-2], 9))

    for metric in metrics:
        if metric not in METRICS_IMPLEMENTED.keys():
            print_(f"Metric {metric} not implemented.")
        if metric == "auc":
            # Compute the joint positions using forward kinematics
            if representation != "pos":
                prediction_positions, _ = h36m_forward_kinematics(predictions, representation)
                target_positions, _ = h36m_forward_kinematics(targets, representation)
            else:
                prediction_positions = predictions
                target_positions = targets
            # Scale to meters for evaluation
            prediction_positions /= 1000
            target_positions /= 1000
            results[metric] = accuracy_under_curve(
                prediction_positions, target_positions
            )
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
    feature_size = seq.shape[-1]
    n_joints = seq.shape[1]

    seq_t = torch.transpose(seq, [0, 2, 1, 3])
    dims_to_use = torch.where(
        (torch.reshape(seq_t, [-1, n_joints, feature_size]).std(0) >= 1e-4).all(dim=-1)
    )[0]
    seq_t = seq_t[:, :, dims_to_use]

    seq_t = torch.reshape(seq_t, [seq_t.shape[0], seq_t.shape[1], 1, -1])
    seq = torch.transpose(seq_t, [0, 2, 1, 3])

    seq_fft = torch.fft.fft(seq, dim=2)
    seq_ps = torch.abs(seq_fft) ** 2

    seq_ps_global = seq_ps.sum(dim=0) + 1e-8
    seq_ps_global /= seq_ps_global.sum(dim=1, keepdims=True)
    return seq_ps_global


def ps_entropy(seq_ps):
    """

    Args:
        seq_ps: (n_joints, seq_len, feature_size)

    Returns:
    """
    return -torch.sum(seq_ps * torch.log(seq_ps), dim=1)


def ps_kld(seq_ps_from, seq_ps_to):
    """Calculates KL(seq_ps_from, seq_ps_to).
    Args:
        seq_ps_from:
        seq_ps_to:

    Returns:
    """
    return torch.sum(seq_ps_from * torch.log(seq_ps_from / seq_ps_to), dim=1)


#####===== Pair-Wise Distance Metrics =====#####


def accuracy_under_curve(
    predictions: torch.tensor,
    targets: torch.tensor,
    thresholds: List[float] = ACC_THRESHOLDS,
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

    # reshape to (-1, n_joints*3) to be consistent with previous work
    euler_preds = euler_preds.view(-1, shape[-3] * 3)
    euler_targs = euler_targs.view(-1, shape[-3] * 3)

    # l2 error on euler angles
    idx_to_use = torch.where(torch.std(euler_targs, dim=0) > 1e-4)[0]
    euc_error = torch.pow(euler_targs[:, idx_to_use] - euler_preds[:, idx_to_use], 2)
    euc_error = torch.sqrt(torch.sum(euc_error, dim=1))  # (-1, ...)

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
